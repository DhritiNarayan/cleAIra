from pprint import pprint

from legal_ai_service import detect_additional_risks_with_ai


sample_contract = """
1. PAYMENT
The client must pay invoices within 30 days of receipt.
A penalty equal to 100 percent of the invoice amount applies
if payment is one day late.

2. TERMINATION
The service provider may terminate the agreement at any time
without prior notice or compensation to the client.

3. LIABILITY
The service provider shall not be liable for any direct,
indirect, incidental, or consequential damages, regardless
of the circumstances.

4. GENERAL
Either party may terminate this agreement by providing
30 days written notice.
"""

try:
    risks = detect_additional_risks_with_ai(sample_contract)

    print(f"\nPotential risks identified: {len(risks)}\n")
    pprint(risks, sort_dicts=False)

except (RuntimeError, ValueError) as error:
    print(f"AI review failed: {error}")