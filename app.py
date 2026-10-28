import os
import re
import shutil
import uuid

from dotenv import load_dotenv
load_dotenv()

import nltk
import pymupdf as fitz
import pytesseract

from PIL import Image
from pdf2image import convert_from_path
from legal_ai_service import contextualize_existing_findings, detect_additional_risks_with_ai
from flask import Flask, request, render_template, abort, redirect
from werkzeug.utils import secure_filename
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from clauses import clauses
from history import save_review, list_reviews, get_review, delete_review


# ---------------- APP ----------------
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024  # 10 MB upload limit


# ---------------- FRIENDLY UPLOAD ERRORS ----------------
@app.errorhandler(400)
def bad_request(error):
    return render_template(
        "index.html",
        results=[],
        summary=None,
        overall_analysis=None,
        error_message=getattr(error, "description", "Invalid request."),
    ), 400


@app.errorhandler(413)
def request_too_large(error):
    return render_template(
        "index.html",
        results=[],
        summary=None,
        overall_analysis=None,
        error_message="The uploaded file is too large. Please upload a file smaller than 10 MB.",
    ), 413

UPLOAD_FOLDER = "uploads"
ALLOWED_UPLOAD_EXTENSIONS = {"pdf", "png", "jpg", "jpeg"}

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)


def allowed_upload(filename):
    """Allow only document/image formats that cleAIra can actually process."""
    safe_name = secure_filename(filename or "")
    if not safe_name or "." not in safe_name:
        return False
    extension = safe_name.rsplit(".", 1)[1].lower()
    return extension in ALLOWED_UPLOAD_EXTENSIONS


# ---------------- TESSERACT ----------------
def get_local_binary(binary_name, extra_paths=None):
    resolved = shutil.which(binary_name)
    if resolved:
        return resolved

    for candidate in extra_paths or []:
        if os.path.exists(candidate):
            return candidate

    return None


tesseract_path = get_local_binary(
    "tesseract",
    [r"C:\Program Files\Tesseract-OCR\tesseract.exe"],
)

if tesseract_path:
    pytesseract.pytesseract.tesseract_cmd = tesseract_path


# ---------------- NLTK ----------------
for package in ("punkt", "punkt_tab"):
    try:
        nltk.data.find(f"tokenizers/{package}")
    except LookupError:
        try:
            nltk.download(package, quiet=True, raise_on_error=False)
        except Exception:
            pass


# ---------------- TFIDF ----------------
texts = [c["text"] for c in clauses]

vectorizer = TfidfVectorizer(
    stop_words="english"
)

X = vectorizer.fit_transform(texts)


# ---------------- CLEAN TEXT ----------------
def clean_text(text):
    # Keep line breaks so standalone document headings can be identified.
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[^\S\n]+", " ", text)
    text = re.sub(r'[^a-zA-Z0-9.,:;?!()\-\'"%\n ]', "", text)
    return text.strip()


# ---------------- HEADING DETECTION ----------------
def is_heading(line):
    """Identify standalone document and section headings without relying only on ALL-CAPS."""
    line = line.strip()
    if not line or len(line) > 100:
        return False
    if re.search(r"[.!?]$", line):
        return False

    # Numbered contractual sections: 1. TERM, 18. GOVERNING LAW AND JURISDICTION.
    if re.match(r"^\d{1,3}\s*[.)-]\s*[A-Za-z]", line):
        return True

    # Common standalone headings written in capitals.
    if line.isupper() and any(ch.isalpha() for ch in line):
        return True

    # Title-case headings are accepted only when they are short and do not look
    # like a normal sentence. This makes the parser useful for varied PDFs.
    words = line.split()
    if 1 <= len(words) <= 8 and not re.search(r"[,;:]", line):
        heading_terms = {
            "term", "services", "fees", "payment", "expenses", "taxes",
            "intellectual property", "confidentiality", "data protection",
            "security", "subcontracting", "warranties", "indemnification",
            "limitation of liability", "termination", "non-solicitation",
            "dispute resolution", "governing law", "assignment", "force majeure",
            "severability", "entire agreement", "amendments", "electronic signatures",
            "survival", "signatures", "compliance with law", "compliance with laws",
        }
        normalized = normalize_text(line)
        if any(phrase_present(normalized, term) for term in heading_terms):
            return True

    return False


def split_into_sentences(text):
    """Remove standalone headings, then split the remaining clause text."""
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    content_lines = []

    for line in lines:
        if is_heading(line):
            continue
        content_lines.append(line)

    # Join actual content lines while keeping their original wording.
    body_text = " ".join(content_lines)
    return re.split(r"(?<=[.!?])\s+", body_text)


# ---------------- PDF EXTRACTION ----------------
def extract_text_from_pdf(pdf_path):
    text = ""

    pdf = fitz.open(pdf_path)

    for page in pdf:
        text += page.get_text()

    return text


# ---------------- OCR ----------------
def extract_text_using_ocr(file_path):
    text = ""

    if (
        not shutil.which("tesseract")
        and not os.path.exists(r"C:\Program Files\Tesseract-OCR\tesseract.exe")
    ):
        return text

    # IMAGE
    if file_path.lower().endswith((".png", ".jpg", ".jpeg")):
        image = Image.open(file_path)

        text = pytesseract.image_to_string(
            image,
            config="--psm 6"
        )

    # PDF
    elif file_path.lower().endswith(".pdf"):
        poppler_path = get_local_binary(
            "pdftoppm",
            [
                r"C:\Program Files\poppler\Library\bin",
                r"C:\Users\itsme\Downloads\Release-26.02.0-0\poppler-26.02.0\Library\bin",
            ],
        )

        try:
            pages = (
                convert_from_path(file_path, poppler_path=poppler_path)
                if poppler_path
                else convert_from_path(file_path)
            )
        except Exception:
            pages = convert_from_path(file_path)

        for page in pages:
            text += pytesseract.image_to_string(
                page,
                config="--psm 6"
            )

    return text


# ---------------- RISK PRIORITY ----------------
def calculate_priority(sentence, risk):
    """Convert the contextual risk assessment into a review urgency label.

    Generic legal vocabulary is not enough to make a clause Critical or High.
    The clause's contextual risk assessment is the primary signal.
    """
    text = re.sub(r"\s+", " ", str(sentence or "").lower()).strip()
    risk_value = str(risk or "").strip().lower()

    critical_patterns = (
        "unlimited liability",
        "waive all liability",
        "100% of the invoice amount",
        "100 percent of the invoice amount",
        "penalty equal to 100%",
        "penalty equal to 100 percent",
    )
    if any(pattern in text for pattern in critical_patterns):
        return "Critical"

    if risk_value in {"critical", "high", "high risk"}:
        return "High Priority"
    if risk_value in {"medium", "medium risk", "moderate"}:
        return "Review"
    if risk_value in {"low", "low risk"}:
        return "Lower Priority"
    return "Review"


def add_priority(result):
    """Apply final context-aware safeguards before priority and summary are computed.

    This is intentionally a narrow post-processing guard: it corrects the known
    false positive for immediate termination tied to serious cause, even if a
    clause-library default or a stale detection path assigned a higher risk.
    """
    sentence = re.sub(r"\s+", " ", str(result.get("sentence", "") or "")).strip()
    text = sentence.lower()
    title = str(result.get("title", "") or "").lower()
    clause_id = str(result.get("clause_id", "") or "").lower()

    termination_match = (
        "termination_without_notice" in clause_id
        or "termination without notice" in title
        or ("terminate immediately" in text and "fraud" in text)
    )
    serious_cause = (
        ("fraud" in text or "willful misconduct" in text or "wilful misconduct" in text)
        and ("material breach" in text or "substantial risk of harm" in text)
        and ("terminate immediately" in text or "immediately without" in text or "immediate termination" in text)
    )
    if termination_match and serious_cause:
        result["risk"] = "Medium"
        result["explanation"] = (
            "The immediate termination right is limited to specified serious events, "
            "including fraud, wilful/willful misconduct, or a material data-security "
            "breach that creates a substantial risk of harm. This is narrower than "
            "an unrestricted right to terminate without notice."
        )
        result["simple"] = (
            "The client can end the agreement immediately only if one of the serious "
            "problems described in the clause occurs."
        )
        result["why_it_matters"] = (
            "The serious-cause triggers should be clear and proportionate. The parties "
            "should also understand any payment, transition, and continuing obligations "
            "after termination."
        )
        result["suggested_action"] = (
            "Confirm that the serious-cause triggers are clearly defined and review "
            "related cure rights, payment obligations, and transition assistance."
        )

    # A late-fee clause does not itself establish the invoice due date. Rename
    # it so the UI does not misstate what the detected text actually covers.
    if clause_id == "payment_due_date" and (
        "late-payment charge" in text or "late payment charge" in text or "late-payment fee" in text
    ) and "payment shall be made within" not in text and "pay within" not in text:
        result["title"] = "Late Payment Charge"

    # The report should describe an employee non-solicitation restriction, not a
    # client/customer restriction, when that is what the detected wording says.
    if "non-solicitation" in title or "non_solicitation" in clause_id:
        if "employee" in text and "solicit" in text:
            result["explanation"] = (
                "The clause restricts knowingly soliciting employees who were materially "
                "involved in the services during the stated period. It includes an "
                "exception for general advertisements not targeted at those employees."
            )
            result["simple"] = (
                "Neither party should specifically recruit the other party's involved "
                "employees during the restricted period, subject to the stated exception."
            )
            result["why_it_matters"] = (
                "The duration, employees covered, and scope of the restriction may affect "
                "ordinary recruitment and should be clear and proportionate."
            )
            result["suggested_action"] = (
                "Confirm the 12-month duration, the definition of employees materially "
                "involved in the services, and the general-advertising exception."
            )

    result["priority"] = calculate_priority(result.get("sentence", ""), result.get("risk", ""))
    return result


# ---------------- CONTRACT RISK SUMMARY ----------------
RISK_ORDER = {"High": 0, "Medium": 1, "Low": 2}

def build_contract_summary(results):
    high = sum(1 for result in results if result.get("risk") == "High")
    medium = sum(1 for result in results if result.get("risk") == "Medium")
    low = sum(1 for result in results if result.get("risk") == "Low")
    total = len(results)
    if high > 0:
        overall = "High"
        message = "The contract contains high-risk provisions that may require careful review."
    elif medium > 0:
        overall = "Medium"
        message = "The contract contains provisions that may require additional review."
    elif low > 0:
        overall = "Low"
        message = "No high-risk provisions were detected among the clauses identified."
    else:
        overall = "Low"
        message = "No review findings were detected."
    return {"total": total, "high": high, "medium": medium, "low": low, "overall": overall, "message": message}



# ---------------- CLAUSE MATCHING V7 ----------------
# Detection V7 builds on V6 with domain/context-aware high-precision overrides.
# It now distinguishes financial/payment language from service-level language,
# rewards clause-specific context, penalizes generic overlap, and uses a
# conservative semantic fallback only after contextual gating.

GENERIC_WORDS = {
    "agreement", "contract", "party", "parties", "company", "client",
    "employee", "employer", "provider", "services", "service", "shall",
    "may", "must", "will", "including", "terms", "work", "payment",
    "amount", "date", "information", "notice", "days", "period",
    "provide", "provided", "receive", "received", "arising", "under",
}

CLAUSE_CONTEXT = {
    "service_fee_contract_price": [
        "service fee", "service fees", "contract price", "agreed fee",
        "fee for the services", "fees for the services",
        "pay the service provider", "payable to the service provider",
        "contract value", "total fee", "total fees",
    ],
    "service_level_commitment": [
        "service level", "service levels", "uptime", "availability",
        "performance standard", "response time", "resolution time",
        "sla", "service credit", "service credits", "support level",
    ],
    "payment_due_date": [
        "invoice", "due date", "days from the date of",
        "payment shall be made", "pay within", "payment deadline",
    ],
    "salary_compensation": [
        "salary", "wages", "compensation", "monthly salary", "annual salary",
        "remuneration", "payable each month", "payable monthly",
    ],
    "confidentiality_obligation": [
        "confidential", "confidential information", "strictly confidential",
        "keep confidential", "maintain confidentiality", "not disclose",
        "disclosure of confidential", "disclose confidential",
    ],
    "confidentiality_duration": [
        "survive termination", "survives termination", "after termination",
        "after the agreement ends", "confidentiality period",
    ],
    "indirect_damages_exclusion": [
        "indirect damages", "incidental damages", "consequential damages",
        "shall not be liable", "not be liable", "limitation of liability",
    ],
    "termination_without_notice": [
        "without prior notice", "without notice", "terminate immediately",
        "terminated immediately", "immediate termination",
    ],
    "late_payment_penalty": [
        "late payment", "late payment penalty", "penalty", "interest on overdue",
        "overdue payment", "delayed payment",
    ],
    "notice_period": [
        "notice period", "days written notice", "written notice before",
        "notice prior to", "advance notice", "resignation notice",
    ],
    "indemnification": [
        "indemnify", "indemnification", "hold harmless", "defend and indemnify",
    ],
    "force_majeure": [
        "force majeure", "natural disasters", "government restrictions",
        "events beyond its reasonable control", "acts of god",
    ],
    "jurisdiction": [
        "exclusive jurisdiction", "courts of", "subject to the jurisdiction",
        "jurisdiction of the courts",
    ],
    "non_compete": [
        "non-compete", "non compete", "competing organization",
        "competing organizations", "competing organisation", "competing organisations",
        "join a competing", "provide services to a competing",
        "shall not join", "shall not work for", "restricted from working",
        "work for competitors", "provide services to any competing",
    ],
    "ip_assignment": [
        "intellectual property", "intellectual property rights",
        "work created by the employee", "work created during employment",
        "property of the company", "exclusive property of the company",
        "inventions", "designs", "work product",
    ],
    "general_dispute_resolution": [
        "any dispute", "disputes shall be resolved", "resolved through mutual discussion",
        "mutual discussion", "amicable resolution", "resolve any dispute",
        "dispute resolution", "disputes arising under",
    ],
}

# Some clauses are especially easy to confuse with generic service/payment text.
# These rules explicitly favor their defining concepts.
# V6: specific confidentiality-related concepts should beat generic confidentiality.
CLAUSE_CONTEXT.update({
    "confidentiality_permitted_use": [
        "solely for the purpose of", "only for the purpose of",
        "use the confidential information solely", "use confidential information only",
    ],
    "non_disclosure": [
        "shall not disclose", "not disclose any confidential information",
        "without prior written consent", "disclose any confidential information",
    ],
    "confidentiality_exception": [
        "already publicly available", "becomes publicly available",
        "not include information", "publicly available through no breach",
    ],
    "confidentiality_duration": [
        "survive termination", "following termination", "after termination",
        "confidentiality period", "remain in effect for a period",
    ],
    "unauthorized_disclosure_notification": [
        "unauthorized disclosure", "unauthorized access", "promptly notify",
        "upon discovering", "notify the disclosing party",
    ],
    "confidential_information_ownership": [
        "exclusive property of the disclosing party", "remain the property of the disclosing party",
        "confidential information shall remain",
    ],
    "return_destroy_confidential_information": [
        "return or destroy", "return and destroy", "destroy all confidential information",
        "return all confidential information", "upon termination or written request",
    ],
    # V9: disclosure liability must contain actual disclosure-related
    # evidence. Generic phrases such as "losses arising from" or
    # "responsible for any losses" are too broad and can misclassify
    # ordinary limitation-of-liability provisions.
    "disclosure_liability": [
        "losses arising from unauthorized disclosure",
        "responsible for losses caused by disclosure",
        "responsible for losses resulting from disclosure",
        "liable for losses caused by disclosure",
        "liable for losses resulting from disclosure",
        "unauthorized disclosure",
        "liability for disclosure",
        "disclosure liability",
    ],
})

CLAUSE_CONFLICTS = {
    "service_level_commitment": {
        "service_fee_contract_price": 55,
        "payment_due_date": 20,
    },
    "service_fee_contract_price": {
        "service_level_commitment": 55,
        "payment_due_date": 15,
    },
    "salary_compensation": {
        "service_fee_contract_price": 25,
    },
    "payment_due_date": {
        "late_payment_penalty": 20,
    },
    "confidentiality_duration": {
        "non_compete": 120,
    },
    "non_compete": {
        "confidentiality_duration": 35,
    },
    "service_level_commitment": {
        "service_fee_contract_price": 75,
        "payment_due_date": 30,
    },
}

# V6 conflict priorities: generic confidentiality is a fallback.
V6_SPECIFICITY_BONUS = {
    "confidentiality_permitted_use": 75,
    "non_disclosure": 70,
    "confidentiality_exception": 70,
    "confidentiality_duration": 65,
    "unauthorized_disclosure_notification": 80,
    "confidential_information_ownership": 65,
    "return_destroy_confidential_information": 80,
    "disclosure_liability": 80,
}

V6_GENERIC_FALLBACK_PENALTY = {
    "confidentiality_obligation": {
        "confidentiality_permitted_use": 90,
        "non_disclosure": 85,
        "confidentiality_exception": 85,
        "confidentiality_duration": 75,
        "unauthorized_disclosure_notification": 100,
        "confidential_information_ownership": 80,
        "return_destroy_confidential_information": 100,
        "disclosure_liability": 100,
    },
}

def extract_primary_provision(sentence, clause):
    """Remove common PDF extraction contamination and keep the legal provision.

    This is concept-based: it looks for the start of the actual contractual
    statement rather than matching a particular contract's full sentence.
    """
    text = re.sub(r"\s+", " ", sentence).strip()
    clause_id = clause.get("id", "")
    title = clause.get("title", "").lower()
    category = clause.get("category", "").lower()

    # Governing-law provisions often get joined to the party-identification
    # paragraph by PDF extraction. Keep the actual governing-law sentence and
    # any jurisdiction sentence that follows it.
    if (
        clause_id == "governing_law"
        or "governing law" in title
        or "governing law" in category
        or "jurisdiction" in title
        or "jurisdiction" in category
    ):
        # PDF extraction can merge the party-identification paragraph with the
        # governing-law provision. Work from the legal anchor itself instead
        # of trusting the extracted sentence boundary.
        governed_match = re.search(
            r"(?:This Agreement|The Agreement|Agreement)\s+(?:shall|will)\s+be\s+governed\s+by\b",
            text,
            re.IGNORECASE,
        )

        if governed_match:
            # Start exactly at the contractual governing-law statement.
            primary_start = governed_match.start()
            remaining = text[primary_start:].strip()

            # Take the governing-law sentence up to its first real terminator.
            # Keep the legal statement beginning at the governing-law anchor;
            # this avoids carrying over party-identification text from PDF extraction.
            sentence_match = re.match(
                r"((?:This Agreement|The Agreement|Agreement)\s+(?:shall|will)\s+be\s+governed\s+by\s+[^.!?]+[.!?])",
                remaining,
                re.IGNORECASE,
            )
            if sentence_match:
                primary = sentence_match.group(1).strip()
                remainder = remaining[sentence_match.end():].strip()
            else:
                primary = remaining
                remainder = ""

            # A jurisdiction sentence immediately following the governing-law
            # sentence is part of the same legal provision.
            jurisdiction = re.match(
                r"(Subject to\b.*?\b(?:exclusive\s+)?jurisdiction\b.*?[.!?])(?:\s+|$)",
                remainder,
                re.IGNORECASE,
            )
            if jurisdiction:
                primary += " " + jurisdiction.group(1).strip()

            return primary

        # Defensive fallback when PDF extraction changes the opening words.
        governed = re.search(
            r"(?:shall|will)\s+be\s+governed\s+by\b",
            text,
            re.IGNORECASE,
        )
        if governed:
            # Prefer the nearest sentence boundary before the governing-law
            # anchor, then discard everything before the anchor itself if the
            # sentence is clearly a merged party-introduction block.
            candidate = text[governed.start():].strip()
            prefix = re.match(
                r"((?:This Agreement|The Agreement|Agreement)?\s*(?:shall|will)\s+be\s+governed\s+by\b.*?[.!?])",
                candidate,
                re.IGNORECASE,
            )
            if prefix:
                return prefix.group(1).strip()

    # Remove a common party-introduction prefix when the actual provision is
    # clearly present later in the same extracted unit. This is intentionally
    # generic and only activates for recognizable contractual anchors.
    anchors = [
        r"This Agreement\s+(?:begins|commences|shall|will|is|continues)",
        r"Either Party\s+(?:may|shall|will)",
        r"Neither Party\s+(?:may|shall|will)",
        r"Each Party\s+(?:shall|will|must)",
        r"Consultant\s+(?:shall|will|must|is)",
        r"Client\s+(?:shall|will|may|must|is)",
    ]
    if len(text) > 300:
        for anchor in anchors:
            match = re.search(anchor, text, re.IGNORECASE)
            if match and match.start() > 0:
                text = text[match.start():].strip()
                break

    return text

def clean_detected_sentence(sentence, clause):
    """Final normalization boundary for detected clause text.

    Detection may use a larger PDF text unit, but the UI and all downstream
    analysis should receive only the actual contractual provision.
    """
    cleaned = extract_primary_provision(sentence, clause)
    return re.sub(r"\\s+", " ", cleaned).strip()


def contextual_risk_assessment(sentence, clause_id, default_risk):
    """Adjust broad clause-library defaults using the actual provision wording.

    This is a conservative heuristic, not legal advice. It avoids treating a
    clause as high risk merely because it belongs to a traditionally sensitive
    category when the specific wording materially narrows the obligation.
    """
    text = re.sub(r"\s+", " ", str(sentence or "").lower()).strip()
    risk = default_risk
    assessment = None

    if clause_id == "ip_assignment":
        limited_work = (
            ("specifically created" in text or "created for client" in text or
             "created for the client" in text or "work product" in text)
            and ("full payment" in text or "payment of all amounts due" in text)
        )
        if limited_work:
            risk = "Medium"
            assessment = {
                "explanation": "The clause transfers ownership of work product specifically created for the client after the relevant amounts are paid. This is common in consulting agreements, but the scope and exclusions still matter.",
                "simple": "The client owns the work made specifically for them once the agreed payment is made.",
                "why_it_matters": "The key issue is whether the assignment is limited to paid, client-specific work and preserves the consultant's pre-existing tools, templates, and know-how.",
                "suggested_action": "Confirm that pre-existing materials, reusable tools, third-party components, and unpaid or unrelated work are expressly excluded where appropriate.",
            }

    elif clause_id in {"license_grant", "broad_license_grant"}:
        scoped_license = (
            ("pre-existing materials" in text or "preexisting materials" in text or "background materials" in text)
            and ("as part of the deliverable" in text or "as part of a deliverable" in text)
            and ("non-exclusive" in text or "non exclusive" in text)
        )
        if scoped_license:
            risk = "Medium"
            assessment = {
                "explanation": "The clause gives the client a continuing, non-exclusive right to use the consultant's pre-existing material only as part of the deliverable. Duration and permitted use remain important, but this is narrower than a transfer of ownership.",
                "simple": "The client can use the consultant's existing material as part of the delivered work, but does not receive ownership of that material under this wording.",
                "why_it_matters": "A perpetual licence may still affect future reuse, distribution, sublicensing, and support obligations, depending on the rest of the contract.",
                "suggested_action": "Check that the licence is limited to what the client needs to use the deliverable and clarify sublicensing, modification, and redistribution rights.",
            }

    elif clause_id == "data_security":
        reasonable_safeguards = (
            "reasonable" in text and ("safeguard" in text or "security measure" in text)
        )
        purpose_limited = ("only as necessary" in text or "solely to perform" in text)
        if reasonable_safeguards and purpose_limited:
            risk = "Medium"
            assessment = {
                "explanation": "The clause limits personal-data processing to service delivery and requires reasonable technical and organisational safeguards. It is primarily a protective obligation; implementation details should be checked against the data and services involved.",
                "simple": "The consultant should use the data only to do the agreed work and take reasonable steps to protect it.",
                "why_it_matters": "The parties should know which safeguards, access controls, and incident procedures are expected and who is responsible for them.",
                "suggested_action": "Confirm that the safeguards are achievable and align with the sensitivity of the data, applicable law, and any security schedule.",
            }

    elif clause_id == "data_breach_notification":
        if "24 hours" in text or "twenty-four (24) hours" in text or "twenty four hours" in text:
            risk = "Medium"
            assessment = {
                "explanation": "The clause requires prompt notice of a confirmed security incident, with a 24-hour target where practicable. The obligation is protective, although the timeline may require a prepared incident-response process.",
                "simple": "The consultant must tell the client quickly if a confirmed incident affects the client's personal data.",
                "why_it_matters": "A short reporting window can be difficult to meet unless the parties define the incident trigger, contact route, and initial information required.",
                "suggested_action": "Clarify when the reporting clock starts, what qualifies as a reportable incident, and whether an initial notice can be followed by details.",
            }

    elif clause_id == "indirect_damages_exclusion":
        mutual = ("neither party" in text or "each party" in text or "both parties" in text)
        if mutual:
            risk = "Medium"
            assessment = {
                "explanation": "The clause excludes specified categories of indirect or consequential damages for both parties. Its effect depends on the liability cap and any exceptions elsewhere in the agreement.",
                "simple": "Neither side can generally claim the listed types of indirect losses under this clause.",
                "why_it_matters": "A mutual exclusion can be commercially common, but it may still exclude important loss types unless appropriate carve-outs apply.",
                "suggested_action": "Check exceptions for confidentiality, data protection, intellectual property, indemnity obligations, fraud, and wilful misconduct, and read this with the liability cap.",
            }

    elif clause_id == "termination_without_notice":
        serious_cause = (
            ("fraud" in text or "willful misconduct" in text or "wilful misconduct" in text)
            and ("material breach" in text or "substantial risk of harm" in text)
        )
        if serious_cause:
            risk = "Medium"
            assessment = {
                "explanation": "The immediate termination right is tied to serious specified events, such as fraud, wilful misconduct, or a material security breach creating substantial risk of harm. This is narrower than an unrestricted right to terminate without notice.",
                "simple": "The client can end the agreement immediately only for certain serious problems described in the clause.",
                "why_it_matters": "The trigger should be clear and proportionate, and the agreement should explain payment, transition, and continuing obligations after termination.",
                "suggested_action": "Confirm the serious-cause triggers are clearly defined and review any related cure rights, payment obligations, and transition assistance.",
            }

    elif clause_id == "payment_due_date":
        late_fee_only = (
            ("late-payment charge" in text or "late payment charge" in text or "late-payment fee" in text)
            and ("per month" in text or "monthly" in text)
            and "payment shall be made within" not in text
            and "pay within" not in text
        )
        if late_fee_only:
            risk = "Medium"
            assessment = {
                "explanation": "This wording sets a charge for overdue amounts; it does not itself state the original invoice due date.",
                "simple": "It explains the consequence of paying late, but not when the invoice is first due.",
                "why_it_matters": "The contract should state the payment deadline separately so the date when a late charge begins is clear.",
                "suggested_action": "Find or add the invoice payment period and clarify how the monthly charge is calculated, subject to applicable law.",
            }

    return risk, assessment


def build_result(sentence, clause):
    # Keep detected clauses focused on the actual provision. OCR/text
    # extraction can sometimes attach the previous party-description line to
    # a governing-law sentence.
    clause_id = clause.get("id", "")
    title = clause.get("title", clause["category"])
    display_sentence = clean_detected_sentence(sentence, clause)

    # If a security-safeguards provision is accidentally classified as a
    # notification clause, correct the presentation without changing the
    # underlying document text. A true notification clause normally contains
    # explicit notification language.
    if (
        "unauthorized disclosure notification" in title.lower()
        and "notify" not in display_sentence.lower()
        and "notification" not in display_sentence.lower()
        and "personal data" in display_sentence.lower()
        and "safeguard" in display_sentence.lower()
    ):
        title = "Data Security Safeguards"
        clause_category = "Data Security"
        explanation = "The provision requires reasonable safeguards to protect personal data from unauthorized access, disclosure, alteration, or destruction."
        simple = "The consultant must use security measures to protect the client's data."
        why_it_matters = "Weak or unclear security requirements can increase the risk of unauthorized access or data incidents."
        what_to_check = [
            "Security measures required",
            "Types of data covered",
            "Incident response requirements",
            "Responsibility for security failures",
        ]
        suggested_action = "Review whether the required security safeguards are specific enough for the data and services involved."
    else:
        clause_category = clause["category"]
        explanation = clause.get("explanation", "")
        simple = clause.get("simple", "")
        why_it_matters = clause.get("why_it_matters", "")
        what_to_check = clause.get("what_to_check", [])
        suggested_action = clause.get("suggested_action", "")

    risk, contextual_assessment = contextual_risk_assessment(
        display_sentence, clause_id, clause.get("risk", "Medium")
    )
    if contextual_assessment:
        explanation = contextual_assessment.get("explanation", explanation)
        simple = contextual_assessment.get("simple", simple)
        why_it_matters = contextual_assessment.get("why_it_matters", why_it_matters)
        suggested_action = contextual_assessment.get("suggested_action", suggested_action)

    # Final presentation guard: a narrowly triggered immediate-termination
    # right for fraud, wilful/willful misconduct, or a material security breach
    # causing substantial harm is termination for serious cause, not an
    # unrestricted termination-without-notice right. Apply this after the
    # general assessment so a library default cannot overwrite the result.
    termination_text = re.sub(r"\s+", " ", str(display_sentence or "").lower()).strip()
    is_termination_clause = (
        clause_id == "termination_without_notice"
        or "termination without notice" in str(title).lower()
    )
    serious_termination_cause = (
        ("fraud" in termination_text
         or "willful misconduct" in termination_text
         or "wilful misconduct" in termination_text)
        and ("material breach" in termination_text
             or "substantial risk of harm" in termination_text)
        and ("terminate immediately" in termination_text
             or "immediately without" in termination_text
             or "immediate termination" in termination_text)
    )
    if is_termination_clause and serious_termination_cause:
        risk = "Medium"
        explanation = (
            "The immediate termination right is limited to specified serious events, "
            "including fraud, wilful/willful misconduct, or a material data-security "
            "breach that creates a substantial risk of harm. This is narrower than "
            "an unrestricted right to terminate without notice."
        )
        simple = (
            "The client can end the agreement immediately only if one of the serious "
            "problems described in the clause occurs."
        )
        why_it_matters = (
            "The serious-cause triggers should be clear and proportionate. The parties "
            "should also understand any payment, transition, and continuing obligations "
            "after termination."
        )
        suggested_action = (
            "Confirm that the serious-cause triggers are clearly defined and review "
            "related cure rights, payment obligations, and transition assistance."
        )

    result = {
        "sentence": display_sentence,
        "clause_id": clause_id,
        "title": title,
        "category": clause_category,
        "risk": risk,
        "explanation": explanation,
        "simple": simple,
        "why_it_matters": why_it_matters,
        "what_to_check": what_to_check,
        "suggested_action": suggested_action,
    }
    return result

def normalize_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()

def phrase_present(text, phrase):
    normalized_phrase = normalize_text(phrase)
    if not normalized_phrase:
        return False
    return re.search(
        r"(?<![a-z0-9])" + re.escape(normalized_phrase) + r"(?![a-z0-9])",
        text,
    ) is not None

def phrase_score(keyword):
    words = normalize_text(keyword).split()
    if not words:
        return 0
    if len(words) >= 5:
        return 120 + len(words) * 5
    if len(words) == 4:
        return 105 + len(words) * 4
    if len(words) == 3:
        return 88 + len(words) * 3
    if len(words) == 2:
        return 65 + len(words) * 2
    word = words[0]
    if word in GENERIC_WORDS:
        return 0
    return 10 + min(len(word), 12) * 0.5

def clause_keyword_score(text, clause):
    score = 0
    evidence = []
    for keyword in clause.get("keywords", []):
        if phrase_present(text, keyword):
            value = phrase_score(keyword)
            if value > 0:
                score += value
                evidence.append(keyword)
    return score, evidence

def clause_context_score(text, clause):
    clause_id = clause.get("id", "")
    score = 0
    evidence = []
    for phrase in CLAUSE_CONTEXT.get(clause_id, []):
        if phrase_present(text, phrase):
            score += phrase_score(phrase) * 0.72
            evidence.append(phrase)
    return score, evidence

def clause_penalty(clause_id, candidate_ids):
    penalty = 0
    for other_id in candidate_ids:
        penalty += CLAUSE_CONFLICTS.get(clause_id, {}).get(other_id, 0)
    return penalty


# ---------------- V7 DOMAIN / CONTEXT RULES ----------------
# High-precision rules for lease/property language. These run before the
# general scorer so a specific domain concept cannot be hijacked by a
# generic pattern such as "after termination" or "period".
V8_DOMAIN_RULES = [
    # V8: lease-term rules must run before permitted-use rules. A sentence
    # describing the duration of a lease may also contain "residential
    # premises", but its primary concept is the lease term.
    ("contract_term", ["lease"], ["period", "term", "months", "month", "years", "year"]),
    ("contract_term", ["lease", "period"], ["months", "month", "years", "year"]),
    ("contract_term", ["lease"], ["for a period", "for a term", "duration of"]),
    ("lease_deposit_refund", ["security deposit"], ["refund", "refunded", "returned", "return"]),
    ("lease_return_possession", ["return possession"], []),
    ("lease_return_possession", ["return", "keys"], ["premises", "property"]),
    ("lease_subletting", ["sublet"], []),
    ("lease_subletting", ["assign", "transfer"], ["premises", "landlord"]),
    ("lease_renewal", ["lease", "renew"], []),
    ("lease_renewal", ["renewed", "additional"], ["lease", "term"]),
    ("lease_permitted_use", ["premises"], ["illegal", "commercial", "residential"]),
    ("lease_inspection_access", ["inspection"], ["premises", "landlord"]),
    ("lease_maintenance_repairs", ["repairs"], ["premises", "tenant"]),
    ("lease_maintenance_repairs", ["maintain", "premises"], []),
    ("lease_utilities", ["utility"], []),
    ("lease_utilities", ["electricity", "water"], []),
    ("lease_rent", ["monthly rent"], []),
    ("lease_rent", ["rent"], ["tenant", "pay", "paid"]),
    ("exclusive_jurisdiction", ["jurisdiction"], ["courts", "court"]),
]

def _rule_matches(text, required, alternatives):
    if not all(phrase_present(text, phrase) for phrase in required):
        return False
    if alternatives and not any(phrase_present(text, phrase) for phrase in alternatives):
        return False
    return True

def find_v8_domain_override(sentence):
    text = normalize_text(sentence)
    for clause_id, required, alternatives in V8_DOMAIN_RULES:
        if _rule_matches(text, required, alternatives):
            for clause in clauses:
                if clause.get("id") == clause_id:
                    return clause
    return None


# ---------------- V9 CLAUSE FAMILIES / CONTEXT ----------------
CLAUSE_FAMILY = {
    "salary_compensation": "payment",
    "service_fee_contract_price": "payment",
    "payment_due_date": "payment",
    "late_payment_penalty": "payment",
    "interest_on_delayed_payment": "payment",
    "additional_fees": "payment",
    "expenses_reimbursement": "payment",
    "expense_responsibility": "payment",
    "taxes": "payment",
    "service_level_commitment": "performance",
    "representations_warranties": "performance",
    "intellectual_property": "ip",
    "ip_assignment": "ip",
    "confidentiality_obligation": "confidentiality",
    "confidentiality_permitted_use": "confidentiality",
    "non_disclosure": "confidentiality",
    "confidentiality_exception": "confidentiality",
    "confidentiality_duration": "confidentiality",
    "unauthorized_disclosure_notification": "confidentiality",
    "confidential_information_ownership": "confidentiality",
    "return_destroy_confidential_information": "confidentiality",
    "disclosure_liability": "risk",
    "indemnification": "risk",
    "broad_indemnification": "risk",
    "indirect_damages_exclusion": "risk",
    "termination_without_notice": "termination",
    "termination_for_breach": "termination",
    "termination_for_convenience": "termination",
    "notice_period": "termination",
    "contract_term": "term",
    "automatic_renewal": "renewal",
    "general_dispute_resolution": "dispute",
    "exclusive_jurisdiction": "dispute",
    "independent_contractor": "relationship",
    "subcontracting_restriction": "relationship",
    "deliverables": "performance",
    "lease_term": "term",
    "lease_rent": "payment",
    "lease_deposit": "payment",
    "lease_deposit_refund": "payment",
    "lease_utilities": "payment",
    "lease_maintenance_repairs": "performance",
    "lease_subletting": "relationship",
    "lease_renewal": "renewal",
    "lease_permitted_use": "performance",
    "lease_inspection_access": "performance",
    "lease_return_possession": "termination",
}

V9_CONTEXT = {
    "consulting_term": ["consultant", "consulting", "services", "period", "months"],
    "service_fee_contract_price": ["professional fee", "consulting fee", "monthly fee", "fee per month", "pay the consultant"],
    "payment_due_date": ["monthly invoice", "submit an invoice", "receiving the invoice", "payment within fifteen days"],
    "deliverables": ["deliverables", "specifications and deliverables", "work product", "deliver the services"],
    "taxes": ["responsible for paying all taxes", "taxes arising from the fees", "tax liability"],
    "expenses_reimbursement": ["reimburse expenses", "reimbursement of expenses", "expenses shall be reimbursed", "business expenses", "travel expenses", "reimbursed"],
    "expense_responsibility": ["responsible for all expenses", "responsible for expenses", "solely responsible for all expenses", "expenses incurred in performing the services", "bear all expenses"],
    "independent_contractor": ["independent contractor", "not be considered an employee", "not an employee", "employee, agent or partner"],
    "indemnification": ["indemnify the client", "indemnify", "hold harmless", "claims arising from"],
    "service_level_commitment": ["deliverables", "specifications", "uptime", "availability", "response time", "resolution time", "service level"],
    "representations_warranties": ["warrants that", "represents and warrants", "in accordance with industry standards"],
}

V9_FAMILY_HINTS = {
    "payment": ["pay", "payment", "fee", "fees", "invoice", "rent", "salary", "compensation", "tax", "expense"],
    "performance": ["deliverable", "specification", "performance", "standard", "maintain", "repair", "inspection", "use"],
    "relationship": ["employee", "contractor", "subcontract", "delegate", "assign", "partner"],
    "risk": ["liable", "liability", "indemnify", "losses", "damages"],
    "confidentiality": ["confidential", "disclose", "disclosure", "secret"],
    "termination": ["terminate", "termination", "notice", "resign", "end"],
    "term": ["term", "period", "duration", "months", "years"],
    "renewal": ["renew", "renewal", "extend", "extension"],
    "dispute": ["dispute", "arbitration", "jurisdiction", "court", "negotiation"],
}

def v9_family_score(text, clause):
    clause_id = clause.get("id", "")
    family = CLAUSE_FAMILY.get(clause_id)
    if not family:
        return 0
    return sum(3 for hint in V9_FAMILY_HINTS.get(family, []) if phrase_present(text, hint))

def find_v9_consulting_override(sentence):
    text = normalize_text(sentence)
    rules = [
        ("consulting_term", ["consultant"], ["period", "months", "years"]),
        ("service_fee_contract_price", ["professional fee"], []),
        ("service_fee_contract_price", ["pay", "consultant"], ["fee", "fees"]),
        ("payment_due_date", ["invoice"], ["payment", "days", "within"]),
        ("deliverables", ["deliverables"], []),
        ("taxes", ["taxes"], ["responsible", "paying", "fees"]),
        ("expense_responsibility", ["expenses"], ["responsible", "incurred", "services", "bear"]),
        ("expenses_reimbursement", ["expenses"], ["reimburse", "reimbursement", "reimbursed"]),
        ("independent_contractor", ["independent contractor"], []),
        ("independent_contractor", ["not be considered an employee"], []),
        ("indemnification", ["indemnify"], []),
    ]
    for clause_id, required, alternatives in rules:
        if _rule_matches(text, required, alternatives):
            for clause in clauses:
                if clause.get("id") == clause_id:
                    return clause
    return None

def find_best_keyword_clause(sentence):
    text = normalize_text(sentence)
    candidates = []

    for clause_index, clause in enumerate(clauses):
        keyword_score, keyword_evidence = clause_keyword_score(text, clause)
        context_score, context_evidence = clause_context_score(text, clause)
        total = keyword_score + context_score
        total += V6_SPECIFICITY_BONUS.get(clause.get("id", ""), 0) if context_evidence else 0
        total += v9_family_score(text, clause)
        for phrase in V9_CONTEXT.get(clause.get("id", ""), []):
            if phrase_present(text, phrase):
                total += phrase_score(phrase) * 0.45

        if total <= 0:
            continue

        if keyword_score == 0 and context_score < 40:
            continue

        candidates.append({
            "score": total,
            "clause_index": clause_index,
            "clause": clause,
            "keyword_evidence": keyword_evidence,
            "context_evidence": context_evidence,
        })

    if not candidates:
        return None

    candidate_ids = [item["clause"].get("id", "") for item in candidates]
    for item in candidates:
        item["adjusted_score"] = item["score"] - clause_penalty(
            item["clause"].get("id", ""), candidate_ids
        )

        # Generic confidentiality should lose when a specific confidentiality concept is present.
        item["adjusted_score"] -= sum(
            V6_GENERIC_FALLBACK_PENALTY.get(item["clause"].get("id", ""), {}).get(other_id, 0)
            for other_id in candidate_ids
        )

    candidates.sort(
        key=lambda item: (
            item["adjusted_score"],
            item["score"],
            len(item["keyword_evidence"]),
            len(item["context_evidence"]),
        ),
        reverse=True,
    )

    best = candidates[0]
    second_score = candidates[1]["adjusted_score"] if len(candidates) > 1 else 0
    margin = best["adjusted_score"] - second_score

    strong_phrase = any(
        len(normalize_text(k).split()) >= 2
        for k in best["keyword_evidence"]
    )

    # A strong clause-specific phrase can stand on its own.
    if best["adjusted_score"] >= 85 and strong_phrase:
        return best["clause"]

    # Otherwise require both enough evidence and a meaningful lead.
    if best["adjusted_score"] >= 55 and margin >= 16:
        return best["clause"]

    return None

def semantic_context_available(sentence, clause):
    text = normalize_text(sentence)
    clause_id = clause.get("id", "")

    if clause_id in CLAUSE_CONTEXT:
        return any(
            phrase_present(text, phrase)
            for phrase in CLAUSE_CONTEXT[clause_id]
        )

    meaningful_keywords = []
    for keyword in clause.get("keywords", []):
        words = normalize_text(keyword).split()
        if not words:
            continue
        if len(words) >= 2 or words[0] not in GENERIC_WORDS:
            meaningful_keywords.append(keyword)

    return any(phrase_present(text, keyword) for keyword in meaningful_keywords)

def find_semantic_clause(sentence):
    text = normalize_text(sentence)
    allowed = []

    for index, clause in enumerate(clauses):
        if semantic_context_available(sentence, clause):
            allowed.append((index, clause))

    if not allowed:
        return None

    user_vector = vectorizer.transform([text])
    similarities = cosine_similarity(user_vector, X)[0]

    ranked = sorted(
        ((index, similarities[index], clause) for index, clause in allowed),
        key=lambda item: item[1],
        reverse=True,
    )

    best_index, best_score, best_clause = ranked[0]
    second_score = ranked[1][1] if len(ranked) > 1 else 0

    if best_score >= 0.60 and (best_score - second_score) >= 0.12:
        return best_clause

    return None

# ---------------- V11 PRECISION OVERRIDES ----------------
def find_v11_precision_override(sentence):
    text = normalize_text(sentence)

    # Tax withholding language must never be mistaken for governing law.
    if (phrase_present(text, "deduct taxes at source") or
        phrase_present(text, "withholding documentation") or
        phrase_present(text, "withhold tax") or
        phrase_present(text, "taxes at source")):
        for clause in clauses:
            if clause.get("id") == "taxes":
                return clause

    # The opening party-identification sentence is not a contract-term clause.
    if phrase_present(text, "entered into as of") and phrase_present(text, "by and between"):
        return {"_skip": True}

    # Explicit immediate-termination language must never be classified as the
    # agreement term. It belongs to the termination concept.
    if (phrase_present(text, "terminate immediately") or
        phrase_present(text, "immediately without prior notice") or
        phrase_present(text, "immediately without a cure period") or
        phrase_present(text, "immediate termination")):
        for clause in clauses:
            if clause.get("id") == "termination_without_notice":
                return clause

    # Explicit term language should beat generic contract/consulting signals.
    if (phrase_present(text, "continues for twelve") or
        phrase_present(text, "continues for") or
        phrase_present(text, "remains in effect for") or
        phrase_present(text, "initial term")):
        for clause in clauses:
            if clause.get("id") in {"consulting_term", "contract_term"}:
                return clause

    # Force majeure is highly distinctive and should be detected directly.
    if (phrase_present(text, "force majeure") or
        (phrase_present(text, "beyond its reasonable control") and
         (phrase_present(text, "natural disasters") or phrase_present(text, "government actions")))):
        for clause in clauses:
            if clause.get("id") == "force_majeure":
                return clause

    return None


# ---------------- GENERALIZED CONTEXT ENGINE V20 ----------------
# Instead of adding a new special-case for every bad sentence, V15 builds a
# small semantic context around each sentence.  Clause decisions can then use
# section headings, positive signals, negative signals, and concept families.
# This keeps the engine maintainable as new contract types are added.

CONCEPT_CONTEXT_RULES = {
    "governing_law": {
        "positive": [
            "governed by", "governed under", "construed under",
            "applicable law", "governing law",
        ],
        "heading": ["governing law", "governing law and jurisdiction", "applicable law"],
        "negative": ["tax", "withholding", "taxes at source"],
    },

    "contract_term": {
        "positive": [
            "continues for", "continue for", "remains in effect for",
            "initial term", "term of this agreement", "term shall be",
            "effective for a period", "for a period of", "for a term of",
            "duration of this agreement",
        ],
        "heading": ["term", "contract term", "agreement term", "duration"],
        "negative": [
            "terminate immediately", "immediate termination",
            "cure period", "termination notice", "notice period",
        ],
    },

    "termination_without_notice": {
        "positive": [
            "terminate immediately", "immediate termination",
            "immediately without notice", "without prior notice",
            "without a cure period",
        ],
        "heading": ["termination", "termination for cause", "termination rights"],
        "negative": ["contract term", "agreement term"],
    },

    "termination_for_convenience": {
        "strong_positive": [
            "terminate this agreement for convenience",
            "termination for convenience",
            "terminate for convenience",
        ],
        "positive": [
            "terminate this agreement for convenience",
            "termination for convenience",
            "terminate for convenience",
            "without cause",
            "without reason",
        ],
        "heading": ["termination", "termination rights", "termination for convenience"],
        "negative": ["terminate immediately", "without prior notice"],
    },

    "notice_period": {
        "positive": [
            "notice period", "days written notice",
            "written notice before", "notice prior to",
            "advance notice", "resignation notice",
        ],
        "heading": ["notice", "notice period"],
        "negative": [
            "terminate this agreement for convenience",
            "termination for convenience",
        ],
    },

    "data_security": {
        "positive": [
            "reasonable safeguards", "security safeguards", "security measures",
            "protect personal data", "protect personal information",
            "unauthorized access", "data security", "information security",
            "technical and organizational measures",
        ],
        "heading": ["data protection", "data security", "security", "privacy and security"],
        "negative": ["notify", "notification", "notice of disclosure"],
    },

    "data_breach_notification": {
        "strong_positive": [
            "data breach notification", "notify of a data breach",
            "security incident notification", "breach notification",
        ],
        "all_of": [["security incident"], ["notify"]],
        "positive": [
            "data breach notification", "notify of a data breach",
            "security incident notification", "breach notification",
            "personal data breach", "notify the client of a security incident",
            "notify the other party of a data breach",
        ],
        "heading": ["data breach", "security incident", "incident notification"],
        "negative": [],
    },

    "unauthorized_disclosure_notification": {
        "positive": [
            "unauthorized disclosure", "unauthorized access",
            "promptly notify", "upon discovering", "notify the disclosing party",
        ],
        "heading": ["unauthorized disclosure", "incident notification"],
        "negative": ["data breach", "security incident"],
    },

    "payment_due_date": {
        "positive": [
            "payment within", "pay within", "payment shall be made within",
            "due within", "invoice", "days after receipt", "days of receipt",
        ],
        "heading": ["payment", "fees and payment", "payment terms", "invoicing"],
        "negative": [],
    },

    "service_fee_contract_price": {
        "positive": [
            "service fee", "professional fee", "consulting fee", "monthly fee",
            "fee per month", "contract price", "fees payable",
        ],
        "heading": ["fees", "service fee", "compensation"],
        "negative": ["payment within", "due within", "invoice date"],
    },

    "taxes": {
        "positive": [
            "tax", "taxes", "withholding", "withhold tax", "tax liability",
            "deduct taxes at source",
        ],
        "heading": ["tax", "taxes", "withholding"],
        "negative": [],
    },

    "force_majeure": {
        "positive": [
            "force majeure", "beyond its reasonable control",
            "natural disaster", "government action", "act of god",
        ],
        "heading": ["force majeure"],
        "negative": [],
    },

    "independent_contractor": {
        "positive": [
            "independent contractor", "independent consultant",
            "not an employee", "not be considered an employee",
            "employee, agent or partner",
        ],
        "heading": ["independent contractor", "relationship of the parties"],
        "negative": [],
    },

    "confidentiality_permitted_use": {
        "strong_positive": [
            "solely for the purpose of", "only for the purpose of",
            "use the confidential information solely",
            "use confidential information only",
            "may be used solely for", "may be used only for",
            "may be used solely", "may be used only",
            "may be used solely for", "may be used only for",
            "may be used solely", "may be used only",
        ],
        "positive": [
            "solely for the purpose of", "only for the purpose of",
            "use the confidential information solely",
            "use confidential information only",
        ],
        "heading": ["permitted use", "use of confidential information"],
        "negative": [],
    },

    "non_disclosure": {
        "strong_positive": [
            "shall not disclose confidential information",
            "shall not disclose any confidential information",
            "not disclose any confidential information",
            "shall not disclose confidential information to any unauthorized person",
        ],
        "positive": [
            "shall not disclose confidential information",
            "shall not disclose any confidential information",
            "not disclose any confidential information",
            "disclose confidential information to third parties",
        ],
        "heading": ["non-disclosure", "confidentiality", "non disclosure"],
        "negative": ["solely for the purpose of", "only for the purpose of"],
    },

    "confidentiality_duration": {
        "strong_positive": [
            "confidentiality obligations shall survive termination",
            "confidentiality obligations survive termination",
            "confidentiality shall survive termination",
            "confidentiality survives termination",
        ],
        "positive": [
            "confidentiality obligations shall survive termination",
            "confidentiality obligations survive termination",
            "confidentiality shall survive termination",
            "confidentiality survives termination",
            "confidentiality period",
        ],
        "heading": ["confidentiality duration", "confidentiality period"],
        "negative": [],
    },

    "non_compete": {
        "strong_positive": [
            "non-compete", "non compete",
            "competing services", "direct competitor",
            "shall not provide competing services",
        ],
        "positive": [
            "non-compete", "non compete", "competing organization",
            "competing organizations", "competing organisation",
            "competing organisations", "join a competing",
            "provide services to a competing", "shall not join",
            "shall not work for", "restricted from working",
            "work for competitors", "provide services to any competing",
        ],
        "heading": ["non-compete", "non compete", "restrictive covenants"],
        "negative": [],
    },

    "non_solicitation": {
        "positive": [
            "non-solicitation", "non solicitation",
            "shall not solicit", "not solicit the client's employees",
            "not solicit employees", "solicit the employees",
            "solicit customers", "solicit clients",
        ],
        "heading": ["non-solicitation", "non solicitation"],
        "negative": [],
    },

    "mandatory_arbitration": {
        "strong_positive": [
            "binding arbitration", "finally resolved by binding arbitration",
            "resolved by arbitration", "submitted to arbitration",
        ],
        "positive": [
            "binding arbitration", "finally resolved by arbitration",
            "resolved by arbitration", "submitted to arbitration",
            "arbitration shall be final", "arbitration proceedings",
        ],
        "heading": ["arbitration", "dispute resolution and arbitration"],
        "negative": [],
    },

    "deliverables": {
        "positive": [
            "deliverables", "deliverable", "shall deliver",
            "deliver the software", "deliver the services",
            "performance requirements", "scope of services",
        ],
        "heading": ["deliverables", "deliverables and performance",
                     "scope of services", "services"],
        "negative": [],
    },
}

CLAUSE_ID_ALIASES = {}


def _clause_by_id(clause_id):
    for clause in clauses:
        if clause.get("id") == clause_id:
            return clause
    return None


def _context_blob(sentence, heading=""):
    # Section context is structural evidence, but sentence text remains primary.
    return normalize_text(f"{heading} {sentence}")


def normalize_heading(heading):
    """Normalize numbered headings while preserving their legal meaning."""
    heading = re.sub(r"^\d{1,3}\s*[.)-]\s*", "", heading.strip())
    return normalize_text(heading)


def _rule_signal_count(text, phrases):
    return sum(1 for phrase in phrases if phrase_present(text, phrase))


def _heading_signal(heading, phrases):
    normalized_heading = normalize_heading(heading)
    if not normalized_heading:
        return 0
    return sum(1 for phrase in phrases if phrase_present(normalized_heading, phrase))


def generalized_context_match(sentence, heading=""):
    """Return the most specific clause supported by structural/context signals.

    Specific clause phrases outrank generic concepts. This keeps the engine
    generalized: precedence is based on the evidence in the clause concept,
    not on individual test sentences.
    """
    sentence_text = normalize_text(sentence)

    ranked = []
    for concept_id, rule in CONCEPT_CONTEXT_RULES.items():
        strong = _rule_signal_count(sentence_text, rule.get("strong_positive", []))
        positive = _rule_signal_count(sentence_text, rule.get("positive", []))
        heading_hits = _heading_signal(heading, rule.get("heading", []))
        negative = _rule_signal_count(sentence_text, rule.get("negative", []))

        # Some concepts need two independent signals (e.g. a security
        # incident + a notification action) without requiring one exact phrase.
        all_of_hits = 0
        for group in rule.get("all_of", []):
            if all(phrase_present(sentence_text, p) for p in group):
                all_of_hits += 1

        # Strong phrases represent a highly specific legal concept. They get
        # a large bonus so a specific clause beats a generic fallback such as
        # Notice Period, Survival, or Dispute Resolution.
        score = positive * 18 + strong * 48 + all_of_hits * 44 + heading_hits * 24 - negative * 22

        if positive >= 2:
            score += 18
        if positive == 1 and heading_hits:
            score += 14

        if score <= 0:
            continue

        target_id = CLAUSE_ID_ALIASES.get(concept_id, concept_id)
        clause = _clause_by_id(target_id)
        if clause is not None:
            ranked.append((score, strong, all_of_hits, clause))

    if not ranked:
        return None

    ranked.sort(key=lambda item: (item[0], item[1], item[2]), reverse=True)
    best_score, best_strong, best_all_of, best_clause = ranked[0]
    second_score = ranked[1][0] if len(ranked) > 1 else 0

    # Strong or paired concept evidence is sufficient by itself. Otherwise
    # retain the original meaningful-evidence threshold for structural rules.
    if best_strong > 0 or best_all_of > 0:
        return best_clause

    if best_score >= 42 and (best_score - second_score >= 10 or best_score >= 70):
        return best_clause

    return None


def build_document_units(text):
    """Segment the document while retaining the nearest structural heading."""
    raw_lines = [line.strip() for line in text.split("\n") if line.strip()]
    units = []
    current_heading = ""
    buffer = []

    def flush_buffer():
        nonlocal buffer
        if not buffer:
            return
        body = " ".join(buffer).strip()
        if body:
            parts = re.split(r"(?<=[.!?])\s+", body)
            for part in parts:
                part = part.strip()
                if part:
                    units.append({"sentence": part, "heading": current_heading})
        buffer = []

    for line in raw_lines:
        if is_heading(line):
            flush_buffer()
            current_heading = line
        else:
            buffer.append(line)

    flush_buffer()
    return units

def detect_clause(sentence, heading=""):
    # V15: generalized structural/context rules run first.
    # These rules describe concepts, not individual document sentences.
    clause = generalized_context_match(sentence, heading)
    if clause is not None:
        return clause

    # Existing scorers remain as fallbacks so we do not throw away the tested
    # 82-clause knowledge base while the generalized layer matures.
    precision = find_v11_precision_override(sentence)
    if precision is not None:
        if precision.get("_skip"):
            return None
        return precision

    clause = find_v8_domain_override(sentence)
    if clause is not None:
        return clause

    clause = find_v9_consulting_override(sentence)
    if clause is not None:
        return clause

    clause = find_best_keyword_clause(sentence)
    if clause is not None:
        return clause
    return find_semantic_clause(sentence)



# ---------------- OVERALL CONTRACT ANALYSIS ----------------
CONTRACT_PROFILES = {
    "Employment Agreement": {
        "signals": ["employee", "employer", "employment", "salary", "resignation", "employee shall"],
        "expected": ["Salary / Compensation", "Notice Period", "Termination", "Confidentiality", "Intellectual Property", "Non-Compete", "Leave / Benefits", "Dispute Resolution"],
    },
    "Service Agreement": {
        "signals": ["service provider", "service fee", "invoice", "service agreement"],
        "expected": ["Service Fee / Contract Price", "Payment Due Date", "Service Level", "Termination", "Confidentiality", "Limitation of Liability", "Indemnification", "Force Majeure", "Dispute Resolution"],
    },
    "Non-Disclosure Agreement": {
        "signals": ["confidential information", "disclosing party", "receiving party", "non-disclosure", "nda", "unauthorized disclosure"],
        "expected": ["Permitted Use", "Non-Disclosure", "Confidentiality Exceptions", "Confidentiality Duration", "Unauthorized Disclosure Notification", "Confidential Information Ownership", "Return / Destruction", "Liability for Unauthorized Disclosure", "Dispute Resolution"],
    },
    "Residential Lease Agreement": {
        "signals": ["landlord", "tenant", "premises", "monthly rent", "lease", "security deposit"],
        "expected": ["Lease Term", "Monthly Rent", "Security Deposit", "Security Deposit Refund", "Utilities", "Maintenance / Repairs", "Subletting", "Termination", "Lease Renewal", "Permitted Use", "Inspection / Access", "Return of Possession", "Dispute Resolution"],
    },
    "Consulting Services Agreement": {
        "signals": ["software development and consulting services agreement", "consulting services agreement", "consultant", "consulting", "professional fee", "deliverables", "independent contractor", "technical consulting"],
        "expected": ["Consulting Agreement Term", "Service Fee / Contract Price", "Payment Due Date", "Deliverables & Performance Requirements", "Taxes", "Expense Responsibility", "Intellectual Property", "Confidentiality", "Subcontracting", "Notice Period", "Independent Contractor", "Warranties", "Indemnification", "Limitation of Liability", "Dispute Resolution", "Force Majeure"],
    },
}


def detect_contract_type(text):
    normalized = normalize_text(text)
    scores = {}
    for name, profile in CONTRACT_PROFILES.items():
        score = 0
        for signal in profile["signals"]:
            if phrase_present(normalized, signal):
                score += 2 if len(signal.split()) > 1 else 1
        scores[name] = score
    best = max(scores, key=scores.get) if scores else "General Contract"
    if scores.get(best, 0) < 2:
        return "General Contract"
    return best


def _has_concept(results, concept):
    concept = concept.lower().strip()

    # Normalize display labels so equivalent wording is treated as the same
    # provision (for example, "Deliverables & Performance Requirements" and
    # "Deliverables and Performance Requirements"). This keeps the
    # missing-provision check tied to detected findings instead of relying on
    # a growing list of one-off clause mappings.
    def normalize_label(value):
        value = value.lower().replace("&", " and ")
        value = re.sub(r"[^a-z0-9]+", " ", value)
        return re.sub(r"\s+", " ", value).strip()

    concept_label = normalize_label(concept)

    # Small set of intentional aliases where the profile name and the
    # detector's display title use different but equivalent wording.
    concept_aliases = {
        "non disclosure": ["non disclosure restriction", "non disclosure obligation"],
        "confidentiality duration": ["confidentiality duration", "confidentiality period"],
        "unauthorized disclosure notification": ["unauthorized disclosure notification"],
        "confidential information ownership": ["confidential information ownership", "confidential information owner"],
        "return destruction": ["return or destruction of confidential information", "return destruction"],
        "liability for unauthorized disclosure": ["liability for unauthorized disclosure"],
        "deliverables and performance requirements": ["deliverables and performance requirements", "deliverables and performance"],
        "service fee contract price": ["service fee contract price", "service fee", "contract price"],
        "consulting agreement term": ["consulting agreement term"],
        "non compete": ["non compete restriction", "non compete"],
        "leave benefits": ["employee benefits", "leave benefits"],
        "limitation of liability": ["limitation of liability", "liability cap", "exclusion of indirect damages"],
    }

    acceptable_labels = {concept_label}
    acceptable_labels.update(normalize_label(x) for x in concept_aliases.get(concept_label, []))

    for result in results:
        title = result.get("title", "").lower()
        category = result.get("category", "").lower()
        clause_id = result.get("clause_id", "").lower()
        normalized_title = normalize_label(title)
        normalized_category = normalize_label(category)

        # First, match the profile concept directly against the detected
        # finding's title/category/clause id. This is the primary consistency
        # check and prevents a detected provision from being reported as
        # missing simply because its clause_id has a different name.
        if any(label and (label == normalized_title or label in normalized_title)
               for label in acceptable_labels):
            return True
        if any(label and label in normalized_category for label in acceptable_labels):
            return True
        if concept_label.replace(" ", "_") in clause_id or clause_id in {x.replace(" ", "_") for x in acceptable_labels}:
            return True

        title = result.get("title", "").lower()
        category = result.get("category", "").lower()
        clause_id = result.get("clause_id", "").lower()
        if concept == "consulting agreement term" and (
            "consulting agreement term" in title or "contract term" in title or
            clause_id in {"consulting_term", "contract_term"}
        ):
            return True
        if concept == "service fee / contract price" and (
            "service fee" in title or "contract price" in title or
            "payment / fees" in category or "fee" in clause_id
        ):
            return True
        if concept == "payment due date" and (
            "payment" in title and ("due" in title or "deadline" in title)
            or "payment_due_date" in clause_id
        ):
            return True
        if concept == "deliverables & performance requirements" and (
            "deliverable" in title or "performance" in title
        ):
            return True
        if concept == "notice period" and (
            "notice" in title or "notice" in category or "notice_period" in clause_id
        ):
            return True
        if concept == "independent contractor" and (
            "independent contractor" in title or "independent" in category or "independent_contractor" in clause_id
        ):
            return True
        if concept == "salary / compensation" and (
            "salary" in title or "compensation" in title
        ):
            return True
        if concept == "leave / benefits" and (
            "leave" in title or "benefit" in title
        ):
            return True
        if concept == "termination" and ("termination" in title or "termination" in category or clause_id.startswith("termination_")):
            return True
        if concept == "confidentiality" and ("confidential" in title or "confidential" in category):
            return True
        if concept == "limitation of liability" and ("liability" in title or "liability" in category):
            return True
        if concept == "indemnification" and "indemnif" in (title + " " + category + " " + clause_id):
            return True
        if concept == "force majeure" and "force majeure" in (title + " " + clause_id):
            return True
        if concept == "service level" and ("service level" in title or "service level" in clause_id):
            return True
        if concept == "dispute resolution" and ("dispute" in title or "arbitration" in title or "dispute" in category):
            return True
        if concept == "intellectual property" and ("intellectual property" in title or title == "ip assignment" or "intellectual property" in category):
            return True
        if concept == "taxes" and ("tax" in title or "tax" in category):
            return True
        if concept == "expense responsibility" and "expense" in title:
            return True

        # Lease-specific concepts. These are intentionally explicit so that
        # a lease clause detected under a slightly different display title
        # is still recognized as present in the overall-analysis check.
        if concept == "lease term" and (
            "lease term" in title or "contract / lease term" in title or
            clause_id in {"lease_term", "contract_term"}
        ):
            return True
        if concept == "monthly rent" and (
            "monthly rent" in title or clause_id == "lease_rent"
        ):
            return True
        if concept == "security deposit" and (
            "security deposit" in title and "refund" not in title
            or clause_id == "security_deposit"
        ):
            return True
        if concept == "security deposit refund" and (
            "security deposit refund" in title or
            ("deposit" in title and "refund" in title) or
            clause_id == "lease_deposit_refund"
        ):
            return True
        if concept == "utilities" and (
            "utilities" in title or clause_id == "lease_utilities"
        ):
            return True
        if concept == "maintenance / repairs" and (
            "maintenance" in title or "repair" in title or
            clause_id == "lease_maintenance_repairs"
        ):
            return True
        if concept == "subletting" and (
            "subletting" in title or "assignment / subletting" in title or
            "sublet" in title or clause_id in {"lease_subletting", "subletting"}
        ):
            return True
        if concept == "lease renewal" and (
            "renewal" in title or clause_id == "lease_renewal"
        ):
            return True
        if concept == "permitted use" and (
            "permitted use" in title or clause_id == "lease_permitted_use"
        ):
            return True
        if concept == "inspection / access" and (
            "inspection" in title or "access" in title or
            clause_id == "lease_inspection_access"
        ):
            return True
        if concept == "return of possession" and (
            "return of possession" in title or clause_id == "lease_return_possession"
        ):
            return True

        if concept == "warranties" and "warrant" in (title + " " + category):
            return True
        if concept == "subcontracting" and ("subcontract" in title or "subcontract" in category):
            return True
    return False

def _deduplicate_results(results):
    """Collapse repeated detections without merging unrelated provisions.

    A clause concept can appear in several sentences. Older versions joined
    every repeated sentence together, which could make a service-fee finding
    swallow a liability-cap sentence. For review output we keep the strongest,
    most focused sentence for concepts that are normally single-value clauses.
    """
    grouped = {}
    order = []

    for result in results:
        key = result.get("clause_id") or result.get("title")
        if key not in grouped:
            grouped[key] = result.copy()
            grouped[key]["_sentences"] = [result.get("sentence", "").strip()]
            order.append(key)
        else:
            sentence = result.get("sentence", "").strip()
            if sentence and sentence not in grouped[key]["_sentences"]:
                grouped[key]["_sentences"].append(sentence)

    # Concepts that should normally be represented by one focused provision.
    single_provision_ids = {
        "service_fee_contract_price",
        "payment_due_date",
        "salary_compensation",
        "taxes",
        "consulting_term",
        "contract_term",
        "termination_for_convenience",
        "termination_for_breach",
        "termination_without_notice",
        "breach_cure_period",
        "force_majeure",
        "governing_law",
        "entire_agreement",
        "independent_contractor",
    }

    merged = []
    for key in order:
        item = grouped[key]
        sentences = [x for x in item.pop("_sentences", []) if x]

        if len(sentences) > 1:
            if key in {"consulting_term", "contract_term"}:
                # Prefer the sentence that actually describes duration.
                duration_candidates = [
                    x for x in sentences
                    if re.search(
                        r"\\b(?:continues|remains in effect|runs)\\b.*\\b(?:until|for)\\b",
                        x,
                        re.IGNORECASE,
                    )
                ]
                item["sentence"] = (
                    max(duration_candidates, key=len)
                    if duration_candidates
                    else max(sentences, key=len)
                )
            elif key in single_provision_ids:
                # Keep the sentence with the strongest clause-specific signal.
                # This prevents unrelated adjacent provisions from being
                # concatenated into one result.
                clause_title = item.get("title", "").lower()
                scored = []
                for sentence in sentences:
                    text = sentence.lower()
                    score = 0
                    if key == "service_fee_contract_price":
                        score += 50 * sum(
                            phrase in text
                            for phrase in (
                                "service fee", "professional fee", "consulting fee",
                                "contract price", "fee for the services",
                            )
                        )
                        score -= 40 * sum(
                            phrase in text
                            for phrase in (
                                "aggregate liability", "liability", "liable",
                                "indirect damages", "consequential damages",
                            )
                        )
                    elif key == "payment_due_date":
                        score += 30 * sum(
                            phrase in text
                            for phrase in ("invoice", "due date", "payable within", "payment")
                        )
                    elif key == "taxes":
                        score += 30 * ("tax" in text or "withholding" in text)
                    elif key == "governing_law":
                        score += 40 * ("governed by" in text)
                        score += 20 * ("jurisdiction" in text)
                    elif key == "force_majeure":
                        score += 50 * ("force majeure" in text)
                    elif key == "independent_contractor":
                        score += 50 * ("independent contractor" in text)
                    elif key == "entire_agreement":
                        score += 50 * ("entire agreement" in text or "supersede" in text)
                    elif key.startswith("termination"):
                        score += 30 * ("terminate" in text or "termination" in text)
                    elif key == "breach_cure_period":
                        score += 40 * ("cure" in text)
                    else:
                        score += 1

                    # Prefer shorter focused provisions when scores tie.
                    scored.append((score, -len(sentence), sentence))

                scored.sort(reverse=True)
                item["sentence"] = scored[0][2]
            else:
                # For concepts where multiple occurrences are meaningful, keep
                # the previous behavior but only join distinct provisions.
                item["sentence"] = " ".join(sentences)

        merged.append(item)

    # consulting_term and contract_term describe the same duration concept in
    # consulting agreements. Keep the domain-specific consulting label and
    # remove the generic duplicate when both were detected.
    has_consulting_term = any(
        item.get("clause_id") == "consulting_term" for item in merged
    )
    if has_consulting_term:
        merged = [
            item for item in merged
            if item.get("clause_id") != "contract_term"
        ]

    return merged

def _unique_term(items, value, context):
    """Add a structured analysis item once per value/context pair."""
    value = value.strip()
    context = context.strip()
    if not value or not context:
        return
    key = (value.lower(), context.lower())
    if not any((x.get("value", "").lower(), x.get("context", "").lower()) == key for x in items):
        items.append({"value": value, "context": context})


def _classify_time_context(sentence):
    """Give a detected duration/deadline a human-readable meaning."""
    text = sentence.lower()

    if "after acceptance" in text or "warrant" in text:
        return "Warranty period"
    if "security" in text and ("hour" in text or "notify" in text or "notification" in text):
        return "Security incident notification"
    if "expense" in text and ("submit" in text or "reimburse" in text or "receipt" in text):
        return "Expense reimbursement deadline"
    if "continues for" in text or "remains in effect for" in text or "initial term" in text:
        return "Agreement term"
    if "cure" in text:
        return "Breach cure period"
    if "payment" in text or "invoice" in text or "receipt" in text:
        return "Payment deadline"
    if "arbitration" in text or re.search(r"\bdispute\b", text):
        return "Dispute resolution period"
    if "aggregate liability" in text or ("liability" in text and "preceding the event" in text):
        return "Liability look-back period"
    if "terminate" in text or "termination" in text:
        if "notice" in text or "prior" in text:
            return "Termination notice"
        return "Termination period"
    if "non-solicit" in text or "non solicit" in text or "non-compete" in text or "non compete" in text or "solicit for employment" in text:
        return "Restriction period"
    if "confidential" in text and ("year" in text or "month" in text or "period" in text):
        return "Confidentiality period"
    if "notice" in text:
        return "Notice period"
    return "Time period"


def _extract_financial_and_time_terms(document_text):
    """Extract meaningful monetary, percentage, duration and date terms.

    This intentionally works from the original document text rather than from
    detected-clause snippets. That prevents sentence fragments such as
    'days prior written notice' from appearing in the UI.
    """
    financial_terms = []
    time_terms = []
    key_dates = []

    sentences = split_into_sentences(document_text)

    currency_pattern = re.compile(
        r"(?:INR|USD|EUR|GBP|Rs\.?|₹)\s?[0-9][0-9,]*(?:\.[0-9]+)?",
        re.IGNORECASE,
    )
    percent_pattern = re.compile(r"\b\d+(?:\.\d+)?\s*%")
    duration_pattern = re.compile(
        r"\b(\d+(?:\.\d+)?)\s*\)?\s*(hours?|days?|weeks?|months?|years?)\b",
        re.IGNORECASE,
    )
    date_patterns = [
        re.compile(r"\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},\s+\d{4}\b", re.IGNORECASE),
        re.compile(r"\b\d{1,2}\s+(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4}\b", re.IGNORECASE),
        re.compile(r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b"),
    ]

    for index, sentence in enumerate(sentences):
        text = sentence.strip()
        if not text:
            continue
        lower = text.lower()
        previous_lower = sentences[index - 1].lower() if index > 0 else ""
        context_window = previous_lower + " " + lower

        # ---- Money ----
        for match in currency_pattern.findall(text):
            if any(word in lower for word in ("service fee", "professional fee", "monthly fee", "contract price", "contract value")):
                context = "Monthly service fee" if "monthly" in lower else "Service fee / contract price"
            elif any(word in lower for word in ("late payment", "late-payment", "penalty", "overdue", "delayed payment")):
                context = "Late payment charge"
            elif "rent" in lower:
                context = "Monthly rent"
            elif "salary" in lower or "compensation" in lower or "wages" in lower:
                context = "Salary / compensation"
            elif "deposit" in lower:
                context = "Security deposit"
            else:
                context = "Monetary term"
            _unique_term(financial_terms, match, context)

        # Percentages are financial terms even when no currency is present.
        for match in percent_pattern.findall(text):
            late_payment_context = any(
                word in context_window
                for word in (
                    "late payment", "late-payment", "penalty", "overdue",
                    "delayed payment", "unpaid", "past due", "after the due date",
                    "interest", "interest at",
                )
            )
            if late_payment_context:
                context = "Late payment charge"
            elif "tax" in lower:
                context = "Tax rate"
            else:
                context = "Percentage term"

            display = re.sub(r"\s+", "", match)
            if "per month" in lower or "monthly" in lower:
                display += " per month"
            _unique_term(financial_terms, display, context)

        # ---- Durations / deadlines ----
        for match in duration_pattern.finditer(text):
            value = f"{match.group(1)} {match.group(2)}"
            context = _classify_time_context(text)
            _unique_term(time_terms, value, context)

        # ---- Calendar dates ----
        # Classify each date from its LOCAL sentence/window. Using the entire
        # document here is unsafe: a start date near the beginning can be
        # incorrectly labelled "End date" merely because the agreement later
        # contains the phrase "ending on".
        for pattern in date_patterns:
            for match in pattern.finditer(text):
                start = match.start()
                end = match.end()
                left = text.rfind(".", 0, start)
                right = text.find(".", end)
                sentence_start = 0 if left == -1 else left + 1
                sentence_end = len(text) if right == -1 else right + 1
                local = text[sentence_start:sentence_end].lower().strip()
                nearby = text[max(0, start - 90):min(len(text), end + 90)].lower()

                # V6: choose the date label from the NEAREST explicit marker.
                # This handles sentences containing both a start and an end date.
                markers = [
                    (r"entered into on|entered into as of|dated as of|dated on", "Contract date"),
                    (r"effective date|effective on", "Effective Date"),
                    # In term clauses phrased as "from [date] until [date]",
                    # label the date immediately following "from" as the start.
                    # Restrict this marker to date-like text so phrases such as
                    # "from the date of signature" do not become false start dates.
                    (r"\bfrom\s+(?=(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}|\d{1,2}\s+(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4}|\d{1,2}[/-]\d{1,2}[/-]\d{2,4})", "Start date"),
                    (r"start date|starting on|starts on|begins on|commence on|commences on|commencing on", "Start date"),
                    (r"ending on|ends on|end date|until|expires on|terminates on", "End date"),
                    (r"signature date|date of signature|signed on", "Signature date"),
                ]

                candidates = []
                for marker_pattern, label in markers:
                    for marker_match in re.finditer(marker_pattern, local, re.IGNORECASE):
                        marker_pos = sentence_start + marker_match.start()
                        marker_end = sentence_start + marker_match.end()
                        if marker_end <= start:
                            distance = start - marker_end
                        elif marker_pos >= end:
                            distance = marker_pos - end
                        else:
                            distance = 0
                        candidates.append((distance, label))

                if candidates:
                    candidates.sort(key=lambda item: item[0])
                    context = candidates[0][1]
                else:
                    context = "Contract date"

                # Avoid duplicate labels for the same calendar date. If the
                # same date is explicitly a Start/Effective/End date, keep the
                # explicit label and do not add a generic Contract date.
                if any(
                    item.get("value", "").lower() == match.group(0).lower()
                    and item.get("context", "") == context
                    for item in key_dates
                ):
                    continue
                if context == "Contract date" and any(
                    item.get("value", "").lower() == match.group(0).lower()
                    and item.get("context", "") in {"Effective Date", "Start date", "End date", "Signature date"}
                    for item in key_dates
                ):
                    continue
                _unique_term(key_dates, match.group(0), context)

    # If the same date is both a contract date and a start/effective date,
    # keep the contract-date label rather than showing duplicates.
    for item in list(key_dates):
        if item.get("context") == "Contract date":
            key_dates[:] = [
                other for other in key_dates
                if not (
                    other is not item
                    and other.get("value", "").lower() == item.get("value", "").lower()
                    and other.get("context") in {"Start date", "Effective Date"}
                )
            ]

    # Do not infer a signature date merely because a signature block contains
    # a generic "Date:" field. It is labelled as a signature date only when
    # the document explicitly uses wording such as "Signature Date" or
    # "Date of Signature".

    # Put the most useful items first while preserving document order.
    def time_rank(item):
        order = {
            "Agreement term": 0,
            "Payment deadline": 1,
            "Termination notice": 2,
            "Breach cure period": 3,
            "Warranty period": 4,
            "Security incident notification": 5,
            "Expense reimbursement deadline": 6,
            "Dispute resolution period": 7,
            "Liability look-back period": 8,
            "Confidentiality period": 9,
            "Restriction period": 10,
            "Notice period": 11,
            "Termination period": 12,
            "Time period": 13,
        }
        return order.get(item["context"], 99)

    time_terms.sort(key=time_rank)
    return financial_terms[:10], time_terms[:12], key_dates[:8]


def build_overall_analysis(document_text, results):
    contract_type = detect_contract_type(document_text)
    profile = CONTRACT_PROFILES.get(contract_type)

    high_priority = [
        result for result in results
        if result.get("priority") in {"Critical", "High Priority"}
    ]

    if profile:
        missing = []
        normalized_document = normalize_text(document_text)

        # A provision should not be reported as potentially missing merely
        # because the clause detector failed to classify it. We therefore use
        # a conservative document-level fallback for clear, explicit phrases.
        raw_concept_patterns = {
            "independent contractor": [
                "independent contractor",
                "independent consultant",
            ],
            "warranties": [
                "warrant", "warranty", "warranties",
            ],
            "indemnification": [
                "indemnify", "indemnification", "indemnity",
            ],
            "force majeure": [
                "force majeure",
            ],
            "taxes": [
                "tax", "taxes", "withholding",
            ],
            "subcontracting": [
                "subcontract", "subcontracting",
            ],
            "confidentiality": [
                "confidential information", "confidentiality",
            ],
            # A duration provision may be present in the document without
            # being emitted as a separate finding (for example, when it is
            # grouped into a broader confidentiality finding). Check explicit
            # duration language in the source text before listing it as missing.
            "confidentiality duration": [
                "confidentiality duration",
                "confidentiality period",
                "confidentiality obligations shall continue",
                "confidentiality obligations will continue",
                "confidentiality obligations remain in effect",
                "confidentiality obligations shall survive",
                "for three years from the date of disclosure",
                "for two years from the date of disclosure",
                "for one year from the date of disclosure",
                "for five years from the date of disclosure",
                "for a period of three years",
                "for a period of two years",
                "for a period of one year",
                "for a period of five years",
            ],
            "intellectual property": [
                "intellectual property", "work product", "source code",
            ],
            "notice period": [
                "written notice", "notice period", "prior written notice",
            ],
            "dispute resolution": [
                "dispute resolution", "arbitration",
            ],
            "limitation of liability": [
                "limitation of liability", "aggregate liability",
                "indirect damages", "consequential damages",
            ],
            "expense responsibility": [
                "expense", "reimburse",
            ],
            # Lease-specific document-level fallbacks. These prevent a
            # detected lease provision from being incorrectly listed as
            # potentially missing when its display title differs slightly.
            "lease term": ["lease term", "lease shall commence", "lease shall begin"],
            "monthly rent": ["monthly rent", "rent of", "rent shall"],
            "security deposit": ["security deposit"],
            "security deposit refund": [
                "deposit shall be refunded", "deposit refund", "refund the deposit",
                "return the remaining deposit", "return the deposit",
                "remaining deposit within", "refundable security deposit",
            ],
            "utilities": ["utilities", "electricity", "water", "internet"],
            "maintenance / repairs": ["maintenance", "repairs", "structural repairs"],
            "subletting": ["sublet", "subletting", "assign the lease", "transfer the lease"],
            "lease renewal": [
                "lease may be renewed", "lease renewal", "renewed for",
                "tenant may request renewal", "request renewal",
                "renewal must be agreed to in writing",
                "renewal in writing", "unless renewed in writing",
            ],
            "permitted use": ["permitted use", "residential purposes only", "use the property"],
            "inspection / access": ["inspection", "enter the property", "access"],
            "return of possession": ["return possession", "deliver possession", "vacate the property"],
        }

        for item in profile["expected"]:
            if _has_concept(results, item):
                continue

            concept_key = item.lower()
            patterns = raw_concept_patterns.get(concept_key, [])
            if patterns and any(phrase_present(normalized_document, p) for p in patterns):
                continue

            missing.append(item)
    else:
        missing = []

    # These are review prompts, not legal conclusions. Absence here only means
    # the clause detector did not find a matching provision.
    obligations = []
    seen_obligations = set()
    for result in results:
        sentence = result.get("sentence", "")
        title = result.get("title", "")
        if any(word in sentence.lower() for word in ("shall", "must", "agrees to", "responsible for", "may terminate")):
            key = (title.lower(), sentence.lower())
            if key not in seen_obligations:
                obligations.append({"title": title, "text": sentence})
                seen_obligations.add(key)

    financial_terms, time_terms, key_dates = _extract_financial_and_time_terms(document_text)

    if not results:
        headline = "No review findings were detected."
    elif high_priority:
        headline = f"{contract_type} identified with {len(results)} review findings, including {len(high_priority)} high-priority item(s)."
    else:
        headline = f"{contract_type} identified with {len(results)} review findings. No Critical or High Priority findings were detected."

    return {
        "contract_type": contract_type,
        "headline": headline,
        "obligations": obligations[:10],
        "financial_terms": financial_terms,
        "time_terms": time_terms,
        "key_dates": key_dates,
        # Kept for backward compatibility with older templates.
        "amounts": [f"{x['value']} — {x['context']}" for x in financial_terms + time_terms + key_dates],
        "missing": missing,
    }

# ---------------- HOME ----------------
@app.route("/", methods=["GET", "POST"])
def home():
    results = []

    if request.method == "POST":
        user_input = request.form.get("clause", "")
        uploaded_file = request.files.get("file")
        source = "Pasted text"

        # ---------------- FILE ----------------
        if uploaded_file and uploaded_file.filename != "":
            original_filename = uploaded_file.filename

            # Never use the user's filename directly as a filesystem path.
            # Validate the extension and save the upload under a random name.
            if not allowed_upload(original_filename):
                abort(400, description="Unsupported file type. Please upload a PDF, PNG, JPG, or JPEG file.")

            safe_name = secure_filename(original_filename)
            extension = safe_name.rsplit(".", 1)[1].lower()
            source = safe_name

            # A unique server-side filename prevents collisions and avoids
            # path traversal through crafted upload names.
            stored_name = f"{uuid.uuid4().hex}.{extension}"
            file_path = os.path.join(UPLOAD_FOLDER, stored_name)

            uploaded_file.save(file_path)

            try:
                # PDF
                if extension == "pdf":
                    try:
                        user_input = extract_text_from_pdf(file_path)
                    except fitz.FileDataError:
                        abort(400, description="Invalid or corrupted PDF file. Please upload a valid PDF document.")

                    # OCR fallback
                    if len(user_input.strip()) < 20:
                        try:
                            user_input = extract_text_using_ocr(file_path)
                        except fitz.FileDataError:
                            abort(400, description="Invalid or corrupted PDF file. Please upload a valid PDF document.")

                # IMAGE
                elif extension in {"png", "jpg", "jpeg"}:
                    user_input = extract_text_using_ocr(file_path)
            finally:
                # Uploaded files are temporary and must not remain on disk.
                if os.path.exists(file_path):
                    os.remove(file_path)

        # ---------------- CLEAN ----------------
        user_input = clean_text(user_input)

        # ---------------- SPLIT + STRUCTURAL CONTEXT ----------------
        # Keep the nearest section heading attached to each sentence. This
        # gives the detector document structure without hard-coding individual
        # sentences from a particular contract.
        document_units = build_document_units(user_input)

        clean_units = []

        for unit in document_units:
            original_sentence = unit["sentence"].strip()

            if len(original_sentence) < 40:
                continue

            if len(original_sentence) > 300:
                continue

            if original_sentence.isupper():
                continue

            clean_units.append(unit)

        # ---------------- ANALYSIS V15 ----------------
        for unit in clean_units:
            sentence = unit["sentence"]
            heading = unit.get("heading", "")
            best_clause = detect_clause(sentence, heading)

            if best_clause is not None:
                # V20: sanitize the sentence BEFORE building the result so that
                # every downstream consumer (including Overall Contract Analysis)
                # receives the cleaned contractual provision, not the raw PDF unit.
                cleaned_sentence = extract_primary_provision(sentence, best_clause)

                # V20: final normalization boundary.  Some PDF extractors can
                # reintroduce a merged party-introduction block after the
                # clause has been classified.  Normalize the actual result
                # immediately before it enters the shared results list.
                clause_id = best_clause.get("id", "")
                if clause_id == "governing_law":
                    governed_match = re.search(
                        r"((?:This Agreement|The Agreement|Agreement)\s+(?:shall|will)\s+be\s+governed\s+by\b.*?)(?=\s+(?:Subject to\b|The courts?\b|Courts?\b)|$)",
                        cleaned_sentence,
                        re.IGNORECASE,
                    )
                    if governed_match:
                        cleaned_sentence = governed_match.group(1).strip()
                        jurisdiction_match = re.search(
                            r"((?:Subject to\b.*?\b(?:exclusive\s+)?jurisdiction\b.*?)(?=$|[.!?](?:\s|$)))",
                            sentence,
                            re.IGNORECASE,
                        )
                        if jurisdiction_match:
                            cleaned_sentence = cleaned_sentence.rstrip(". ") + ". " + jurisdiction_match.group(1).strip()

                results.append(build_result(cleaned_sentence, best_clause))

        # Keep a payment deadline with the payment/invoice provision it qualifies.
        # Contracts often split invoice issuance and the actual payment deadline
        # into adjacent sentences, so the invoice sentence alone is incomplete.
        payment_deadline_sentences = []
        for unit in document_units:
            candidate = re.sub(r"\s+", " ", unit.get("sentence", "")).strip()
            has_payment_context = re.search(
                r"\b(?:pay|pays|paid|payment|payments|payable|invoice|invoices)\b",
                candidate,
                re.IGNORECASE,
            )
            has_deadline = re.search(
                r"\b(?:within\s+\d+\s+(?:calendar\s+|business\s+)?days?|\d+\s+days?\s+of\s+(?:receipt|submission|delivery)|no later than|due\s+(?:within|on|by))\b",
                candidate,
                re.IGNORECASE,
            )
            if has_payment_context and has_deadline:
                payment_deadline_sentences.append(candidate)

        if payment_deadline_sentences:
            for result in results:
                title_text = result.get("title", "").lower()
                clause_id = result.get("clause_id", "").lower()
                if "payment due date" not in title_text and "payment_due_date" not in clause_id:
                    continue
                combined = result.get("sentence", "").strip()
                for deadline_sentence in payment_deadline_sentences:
                    if deadline_sentence.lower() not in combined.lower():
                        combined = (combined.rstrip(" .") + ". " + deadline_sentence).strip()
                result["sentence"] = combined
                break

        # V11: group repeated detections of the same clause concept.
        results = _deduplicate_results(results)

        # Performance mode: skip contextualizing every existing finding.
        # Multiple sequential Gemini calls can make a review unnecessarily slow.
        # Keep the rule-based findings intact and make only one AI pass to look
        # for additional risks. This avoids the previous multi-call bottleneck.
        ai_status = "limited"

        # AI pass: look for additional risks not caught by the rule engine.
        # Validate and deduplicate each candidate against existing findings.
        try:
            additional_risks = detect_additional_risks_with_ai(user_input)
            existing_texts = [
                normalize_text(str(item.get("sentence", "")))
                for item in results
            ]
            for item in additional_risks:
                clause_text = re.sub(r"\s+", " ", item.get("clause_text", "")).strip()
                normalized_clause = normalize_text(clause_text)
                if not clause_text or len(clause_text) < 25:
                    continue
                # Avoid adding the same provision/risk twice. Exact or near
                # containment matches are intentionally conservative.
                if any(
                    normalized_clause in prior or prior in normalized_clause
                    for prior in existing_texts if prior and normalized_clause
                ):
                    continue
                priority_to_risk = {
                    "Critical": "High", "High": "High",
                    "Medium": "Medium", "Low": "Low",
                }
                results.append({
                    "sentence": clause_text,
                    "clause_id": "ai_additional_risk",
                    "title": item["risk"],
                    "category": "AI-identified risk",
                    "risk": priority_to_risk[item["priority"]],
                    "explanation": item["explanation"],
                    "simple": item["risk"],
                    "why_it_matters": item["explanation"],
                    "what_to_check": [item["recommendation"]],
                    "suggested_action": item["recommendation"],
                    "ai_generated": True,
                })
                existing_texts.append(normalized_clause)
        except (RuntimeError, ValueError, TypeError, KeyError) as exc:
            app.logger.warning("AI additional-risk scan unavailable: %s", exc)
            ai_status = "partial"

        # Deduplicate again after AI candidates are added, then apply priorities.
        results = _deduplicate_results(results)
        

        # Add review priority after detection, then sort by urgency.
        for result in results:
            add_priority(result)

        PRIORITY_ORDER = {
            "Critical": 0,
            "High Priority": 1,
            "Review": 2,
            "Lower Priority": 3,
        }

        results.sort(
            key=lambda result: (
                PRIORITY_ORDER.get(result.get("priority"), 99),
                RISK_ORDER.get(result.get("risk"), 99),
            )
        )
        summary = build_contract_summary(results)
        overall_analysis = build_overall_analysis(user_input, results)

        save_review(
            0,
            source,
            user_input,
            results,
            summary=summary,
            overall_analysis=overall_analysis,
        )

    else:
        summary = None
        overall_analysis = None

    return render_template(
        "index.html",
        results=results,
        summary=summary,
        overall_analysis=overall_analysis,
        ai_status=ai_status if request.method == "POST" else None,
    )


@app.route("/history")
def history():
    reviews = list_reviews(0)
    return render_template("history.html", reviews=reviews)


@app.route("/history/<int:review_id>")
def history_detail(review_id):
    review = get_review(0, review_id)
    if not review:
        abort(404)
    return render_template("history_detail.html", review=review)


@app.route("/history/<int:review_id>/delete", methods=["POST"])
def history_delete(review_id):
    delete_review(0, review_id)
    return redirect("/history")


# ---------------- RUN ----------------
if __name__ == "__main__":
    app.run(debug=True)
