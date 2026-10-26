
"""
cleAIra clause-engine regression tests.

This file does NOT modify cleAIra. It imports the current app.py and checks
representative positive and negative sentences against detect_clause().
Run with the same Python environment used for cleAIra:
    python test_clause_engine.py
"""

from app import detect_clause


POSITIVE_TESTS = [
    # Termination / notice
    ("Immediate termination", 
     "Either party may terminate this Agreement immediately without prior notice in the event of a serious violation.",
     "termination_without_notice"),
    ("Termination for breach",
     "Either party may terminate this Agreement upon a material breach of its obligations.",
     "termination_for_breach"),
    ("Termination for convenience",
     "Either party may terminate this Agreement for convenience by providing thirty days prior written notice.",
     "termination_for_convenience"),
    ("Notice period",
     "The employee shall provide thirty days advance written notice before resignation.",
     "notice_period"),

    # Payment / fees
    ("Late payment penalty",
     "A late payment penalty of 1.5 percent per month shall apply to overdue amounts.",
     "late_payment_penalty"),
    ("Delayed payment interest",
     "Interest charges shall apply to delayed payments at the rate specified in this Agreement.",
     "interest_on_delayed_payment"),
    ("Payment due date",
     "Payment shall be made within thirty days from the date of receipt of the invoice.",
     "payment_due_date"),
    ("Service fee",
     "The Client shall pay the Service Provider an agreed service fee of INR 85,000 per month.",
     "service_fee_contract_price"),

    # Liability / indemnity
    ("Liability cap",
     "The Consultant's total liability shall be limited to the amount paid under this Agreement.",
     "liability_cap"),
    ("Indirect damages",
     "The Consultant shall not be liable for any indirect or consequential damages.",
     "indirect_damages_exclusion"),
    ("Indemnification",
     "The Consultant shall indemnify and hold harmless the Client from third party claims arising from negligence.",
     "broad_indemnification"),

    # Confidentiality / data
    ("Confidentiality obligation",
     "The Consultant shall keep all confidential information strictly confidential and shall not disclose it.",
     "confidentiality_obligation"),
    ("Confidentiality duration",
     "The confidentiality obligations shall survive termination of this Agreement for three years.",
     "confidentiality_duration"),
    ("Permitted use",
     "Confidential Information may be used solely for performing the Services under this Agreement.",
     "confidentiality_permitted_use"),
    ("Non-disclosure",
     "The Receiving Party shall not disclose Confidential Information to any unauthorized person.",
     "non_disclosure"),
    ("Data security",
     "The Consultant shall implement appropriate technical and organizational measures to protect personal data.",
     "data_security"),
    ("Breach notification",
     "The Consultant shall notify the Client within twenty four hours of discovering a security incident.",
     "data_breach_notification"),

    # IP / restrictions
    ("IP assignment",
     "All intellectual property created specifically for the Client under this Agreement shall be assigned to the Client.",
     "ip_assignment"),
    ("Non-compete",
     "For twelve months after termination, the Consultant shall not provide competing services to a direct competitor.",
     "non_compete"),
    ("Non-solicitation",
     "For one year after termination, the Consultant shall not solicit the Client's employees.",
     "non_solicitation"),
    ("Assignment",
     "Neither party may assign this Agreement without the prior written consent of the other party.",
     "assignment"),

    # Dispute / law
    ("Governing law",
     "This Agreement shall be governed by the laws of India.",
     "governing_law"),
    ("Exclusive jurisdiction",
     "The courts located in Bengaluru, Karnataka shall have exclusive jurisdiction over matters arising from this Agreement.",
     "exclusive_jurisdiction"),
    ("Arbitration",
     "Any dispute arising out of this Agreement shall be finally resolved by binding arbitration.",
     "mandatory_arbitration"),
    ("Force majeure",
     "Neither party shall be liable for delay caused by events beyond its reasonable control, including natural disasters.",
     "force_majeure"),

    # Consulting / employment
    ("Consulting term",
     "This Agreement shall remain in force for a period of twelve months from the Effective Date.",
     "consulting_term"),
    ("Deliverables",
     "The Consultant shall deliver the software modules and reports described in Schedule A in accordance with the agreed requirements.",
     "deliverables"),
    ("Independent contractor",
     "The Consultant is an independent contractor and is not an employee, agent, or partner of the Client.",
     "independent_contractor"),
    ("Taxes",
     "The Consultant shall be responsible for all taxes arising from payments received under this Agreement.",
     "taxes"),

    # Lease
    ("Monthly rent",
     "The Tenant shall pay monthly rent of INR 25,000 on or before the fifth day of each month.",
     "lease_rent"),
    ("Security deposit",
     "The Tenant shall pay a refundable security deposit of INR 50,000 before occupancy.",
     "security_deposit"),
    ("Maintenance",
     "The Landlord shall be responsible for structural maintenance and major repairs to the premises.",
     "lease_maintenance_repairs"),
]


NEGATIVE_TESTS = [
    ("General agreement statement",
     "The parties agree to cooperate in good faith to achieve the objectives of this Agreement."),
    ("Ordinary service description",
     "The Consultant will provide technical support and professional services to the Client."),
    ("Generic payment mention",
     "The parties will discuss payment arrangements during the project kickoff meeting."),
    ("Generic information mention",
     "The parties may exchange information necessary to coordinate the project."),
    ("Generic date mention",
     "The project team met on Monday to review the current implementation."),
    ("Generic notice mention",
     "The parties discussed the notice received from the building management office."),
    ("Generic termination mention",
     "The project manager discussed the possibility of termination during the meeting."),
    ("Generic data mention",
     "The application processes data submitted by users during normal operation."),
]


def run():
    positive_pass = 0
    negative_pass = 0

    print("=" * 72)
    print("cleAIra CLAUSE ENGINE REGRESSION TEST")
    print("=" * 72)

    print("\nPOSITIVE TESTS")
    print("-" * 72)

    for name, sentence, expected in POSITIVE_TESTS:
        result = detect_clause(sentence)
        actual = result.get("id") if result else None
        ok = actual == expected
        positive_pass += ok

        status = "PASS" if ok else "FAIL"
        print(f"[{status}] {name}")
        if not ok:
            print(f"       Expected: {expected}")
            print(f"       Actual:   {actual}")
            if result:
                print(f"       Title:    {result.get('title')}")

    print("\nNEGATIVE / FALSE-POSITIVE TESTS")
    print("-" * 72)

    for name, sentence in NEGATIVE_TESTS:
        result = detect_clause(sentence)
        actual = result.get("id") if result else None
        ok = actual is None
        negative_pass += ok

        status = "PASS" if ok else "FAIL"
        print(f"[{status}] {name}")
        if not ok:
            print(f"       Unexpected detection: {actual} ({result.get('title')})")

    total_positive = len(POSITIVE_TESTS)
    total_negative = len(NEGATIVE_TESTS)
    total = total_positive + total_negative
    passed = positive_pass + negative_pass

    print("\n" + "=" * 72)
    print("SUMMARY")
    print("=" * 72)
    print(f"Positive tests : {positive_pass}/{total_positive}")
    print(f"Negative tests : {negative_pass}/{total_negative}")
    print(f"Overall        : {passed}/{total} ({passed / total * 100:.1f}%)")

    if positive_pass == total_positive and negative_pass == total_negative:
        print("STATUS         : ALL TESTS PASSED")
        return 0

    print("STATUS         : SOME TESTS FAILED")
    return 1


if __name__ == "__main__":
    raise SystemExit(run())
