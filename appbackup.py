import os
import re
import shutil

import nltk
import pymupdf as fitz
import pytesseract

from PIL import Image
from pdf2image import convert_from_path

from functools import wraps

from flask import Flask, request, render_template, redirect, url_for, session, abort
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from auth import (
    init_db,
    create_user,
    get_user_by_id,
    get_user_by_login,
    verify_password,
    validate_register,
    update_password,
)
from clauses import clauses
from history import save_review, list_reviews, get_review, delete_review


# ---------------- APP ----------------
app = Flask(__name__)
app.secret_key = os.environ.get("CLEAIRA_SECRET_KEY", "cleaira-dev-secret-key")
app.config["SESSION_COOKIE_HTTPONLY"] = True
app.config["SESSION_COOKIE_SAMESITE"] = "Lax"

init_db()


def login_required(view):
    @wraps(view)
    def wrapped(*args, **kwargs):
        user = current_user()
        if not user:
            session.clear()
            return redirect(url_for("login"))
        return view(*args, **kwargs)

    return wrapped


def current_user():
    user_id = session.get("user_id")
    if not user_id:
        return None
    return get_user_by_id(user_id)


@app.context_processor
def inject_user():
    return {"user": current_user()}


def start_session(user):
    session.clear()
    session["user_id"] = user["id"]
    session["username"] = user["username"]


UPLOAD_FOLDER = "uploads"

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)


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
    text = re.sub(r'[^a-zA-Z0-9.,:;?!()\-\'"\n ]', "", text)
    return text.strip()


# ---------------- HEADING DETECTION ----------------
def is_heading(line):
    """Return True for standalone document/section headings."""
    line = line.strip()

    if not line:
        return False

    # Long lines are much more likely to be actual clause text.
    if len(line) > 80:
        return False

    # A line ending in sentence punctuation is normally clause text.
    if re.search(r"[.!?]$", line):
        return False

    # Standalone ALL-CAPS headings, e.g. EMPLOYMENT AGREEMENT.
    if line.isupper() and any(ch.isalpha() for ch in line):
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


# ---------------- AUTH ----------------
@app.route("/register", methods=["GET", "POST"])
def register():
    if session.get("user_id"):
        return redirect(url_for("home"))

    errors = []
    form = {"username": "", "email": ""}

    if request.method == "POST":
        form["username"] = request.form.get("username", "").strip()
        form["email"] = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")
        confirm = request.form.get("confirm", "")

        errors = validate_register(
            form["username"],
            form["email"],
            password,
            confirm,
        )

        if not errors:
            user_id = create_user(
                form["username"],
                form["email"],
                password,
            )

            user = get_user_by_id(user_id)
            start_session(user)

            return redirect(url_for("home"))

    return render_template(
        "register.html",
        errors=errors,
        form=form,
    )


@app.route("/login", methods=["GET", "POST"])
def login():
    if session.get("user_id"):
        return redirect(url_for("home"))

    error = None
    login_value = ""

    if request.method == "POST":
        login_value = request.form.get("username", "").strip()
        password = request.form.get("password", "")

        user = get_user_by_login(login_value)

        if user and verify_password(user, password):
            start_session(user)
            return redirect(url_for("home"))

        error = "Invalid username/email or password."

    return render_template(
        "login.html",
        error=error,
        login_value=login_value,
    )


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))


@app.route("/account", methods=["GET", "POST"])
@login_required
def account():
    errors = []
    success = None

    if request.method == "POST":
        current_password = request.form.get("current_password", "")
        new_password = request.form.get("new_password", "")
        confirm = request.form.get("confirm", "")

        user = current_user()

        if not user or not verify_password(user, current_password):
            errors.append("Current password is incorrect.")
        elif len(new_password) < 8:
            errors.append("New password must be at least 8 characters.")
        elif new_password != confirm:
            errors.append("New passwords do not match.")
        else:
            update_password(user["id"], new_password)
            success = "Password updated."

    return render_template(
        "account.html",
        errors=errors,
        success=success,
    )


@app.route("/history")
@login_required
def history():
    reviews = list_reviews(current_user()["id"])
    return render_template(
        "history.html",
        reviews=reviews,
    )


@app.route("/history/<int:review_id>")
@login_required
def history_detail(review_id):
    review = get_review(
        current_user()["id"],
        review_id,
    )

    if not review:
        abort(404)

    return render_template(
        "history_detail.html",
        review=review,
    )


@app.route("/history/<int:review_id>/delete", methods=["POST"])
@login_required
def history_delete(review_id):
    delete_review(
        current_user()["id"],
        review_id,
    )

    return redirect(url_for("history"))



# ---------------- RISK PRIORITY ----------------
def calculate_priority(sentence, risk):
    """
    Determine how urgently a detected clause should be reviewed.

    Risk = severity of the clause.
    Priority = how urgently the user should review it.
    """

    text = sentence.lower()

    critical_patterns = [
        "immediately without prior notice",
        "immediate termination",
        "terminate immediately",
        "without prior notice",
        "financial penalties",
        "penalty",
        "shall not be liable",
        "not be liable",
        "waive all liability",
        "unlimited liability",
        "indemnify",
        "indemnification",
    ]

    high_priority_patterns = [
        "non-compete",
        "non compete",
        "competing organization",
        "competing organisations",
        "after resignation",
        "after termination",
        "intellectual property",
        "confidential information",
        "confidentiality",
        "personal data",
        "data protection",
        "refund",
        "termination",
        "assignment",
    ]

    if any(pattern in text for pattern in critical_patterns):
        return "Critical"

    if any(pattern in text for pattern in high_priority_patterns):
        return "High Priority"

    # A High-risk clause should never be shown as merely "Review".
    if risk == "High":
        return "High Priority"

    if risk == "Medium":
        return "Review"

    return "Lower Priority"


def add_priority(result):
    result["priority"] = calculate_priority(
        result["sentence"],
        result["risk"],
    )
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
    "disclosure_liability": [
        "losses arising from", "responsible for any losses",
        "unauthorized disclosure", "liable for losses",
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

def build_result(sentence, clause):
    return {
        "sentence": sentence,
        "title": clause.get("title", clause["category"]),
        "category": clause["category"],
        "risk": clause["risk"],
        "explanation": clause.get("explanation", ""),
        "simple": clause.get("simple", ""),
        "why_it_matters": clause.get("why_it_matters", ""),
        "what_to_check": clause.get("what_to_check", []),
        "suggested_action": clause.get("suggested_action", ""),
    }

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

def detect_clause(sentence):
    # V8 domain rules first for high-precision lease/property handling.
    clause = find_v8_domain_override(sentence)
    if clause is not None:
        return clause

    # V9 consulting/business rules prioritize explicit domain concepts.
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
        "signals": ["service provider", "client", "services", "service fee", "invoice", "service agreement"],
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
        "signals": ["consultant", "consulting", "professional fee", "deliverables", "independent contractor", "technical consulting"],
        "expected": ["Consulting Agreement Term", "Service Fee / Contract Price", "Payment Due Date", "Deliverables & Performance Requirements", "Taxes", "Expense Responsibility", "Intellectual Property", "Confidentiality", "Subcontracting", "Notice Period", "Independent Contractor", "Warranties", "Indemnification", "Limitation of Liability", "Dispute Resolution"],
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


def build_overall_analysis(document_text, results):
    contract_type = detect_contract_type(document_text)
    profile = CONTRACT_PROFILES.get(contract_type)

    detected_titles = {result.get("title", "") for result in results}
    high_priority = [
        result for result in results
        if result.get("priority") in {"Critical", "High Priority"}
    ]

    if profile:
        missing = [item for item in profile["expected"] if item not in detected_titles]
    else:
        missing = []

    # These are review prompts, not legal conclusions. Absence here only means
    # the clause detector did not find a matching provision.
    obligations = []
    for result in results:
        sentence = result.get("sentence", "")
        title = result.get("title", "")
        if any(word in sentence.lower() for word in ("shall", "must", "agrees to", "responsible for", "may terminate")):
            obligations.append({"title": title, "text": sentence})

    amounts = []
    amount_pattern = re.compile(r"(?:INR|USD|EUR|GBP|Rs\.?|₹)\s?[0-9][0-9,]*(?:\.[0-9]+)?", re.IGNORECASE)
    date_pattern = re.compile(r"\b(?:within|for a period of|period of|notice|days|months|years)\b[^.]{0,80}", re.IGNORECASE)
    for sentence in [result.get("sentence", "") for result in results]:
        for match in amount_pattern.findall(sentence):
            amounts.append(match)
        for match in date_pattern.findall(sentence):
            value = match.strip()
            if value and value not in amounts:
                amounts.append(value)

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
        "amounts": amounts[:10],
        "missing": missing,
    }

# ---------------- HOME ----------------
@app.route("/", methods=["GET", "POST"])
@login_required
def home():
    results = []

    if request.method == "POST":
        user_input = request.form.get("clause", "")
        uploaded_file = request.files.get("file")
        source = "Pasted text"

        # ---------------- FILE ----------------
        if uploaded_file and uploaded_file.filename != "":
            source = uploaded_file.filename

            file_path = os.path.join(
                UPLOAD_FOLDER,
                uploaded_file.filename,
            )

            uploaded_file.save(file_path)

            # PDF
            if file_path.lower().endswith(".pdf"):
                user_input = extract_text_from_pdf(file_path)

                # OCR fallback
                if len(user_input.strip()) < 20:
                    user_input = extract_text_using_ocr(file_path)

            # IMAGE
            elif file_path.lower().endswith(
                (".png", ".jpg", ".jpeg")
            ):
                user_input = extract_text_using_ocr(file_path)

            # DELETE FILE
            os.remove(file_path)

        # ---------------- CLEAN ----------------
        user_input = clean_text(user_input)

        # ---------------- SPLIT ----------------
        # Remove standalone headings such as "EMPLOYMENT AGREEMENT"
        # before splitting the actual clause text.
        sentences = split_into_sentences(user_input)

        clean_sentences = []

        for s in sentences:
            # Keep the original wording/capitalization for display.
            original_sentence = s.strip()

            if len(original_sentence) < 40:
                continue

            if len(original_sentence) > 300:
                continue

            if original_sentence.isupper():
                continue

            clean_sentences.append(original_sentence)

        # ---------------- ANALYSIS V3 ----------------
        for sentence in clean_sentences:
            best_clause = detect_clause(sentence)

            if best_clause is not None:
                results.append(build_result(sentence, best_clause))

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
            current_user()["id"],
            source,
            user_input,
            results,
        )

    else:
        summary = None
        overall_analysis = None

    return render_template(
        "index.html",
        results=results,
        summary=summary,
        overall_analysis=overall_analysis,
    )


# ---------------- RUN ----------------
if __name__ == "__main__":
    app.run(debug=True)
