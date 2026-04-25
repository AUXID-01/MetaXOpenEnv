import sys
import os

# Add root so client imports work correctly
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, root_dir)

from client.utils import action_from_text

def test_action_from_text():
    print("Running Gate 2 Tests (Parse Roundtrip)...")

    # 1. Happy path (Valid XML)
    fake_llm_output = """
<action_type>offer_emi</action_type>
<text>I understand you're going through a difficult time. Could we look at a smaller EMI of 1800 per month?</text>
<metadata>{"emi_amount": 1800}</metadata>
"""
    action = action_from_text(fake_llm_output)
    assert action["action_type"] == "offer_emi", f"Failed action_type: {action}"
    assert action["text"].startswith("I understand you're going through a difficult time."), "Failed to parse text"
    assert action["metadata"].get("emi_amount") == 1800, f"Failed metadata: {action}"
    print("[x] Happy path passed")

    # 2. Fallback path (No tags at all)
    bad_output = "Sorry I cannot help with this."
    action = action_from_text(bad_output)
    assert action["action_type"] == "send_message", "Failed to fallback action_type"
    assert action["text"] == bad_output, "Failed to capture raw text on fallback"
    print("[x] Fallback path (No tags) passed")

    # 3. Edge Case: Invalid action_type 
    bad_action = """
<action_type>threaten_borrower</action_type>
<text>Pay now.</text>
<metadata>{}</metadata>
"""
    action = action_from_text(bad_action)
    assert action["action_type"] == "send_message", "Should fallback to send_message for invalid types"
    print("[x] Invalid action_type fallback passed")

    # 4. Edge Case: Malformed metadata JSON inside the tag
    malformed = """
<action_type>offer_emi</action_type>
<text>Here is an offer.</text>
<metadata>{emi: 1800,}</metadata>
"""
    action = action_from_text(malformed)
    assert action["metadata"] == {}, "Malformed JSON should default to empty dict"
    print("[x] Malformed JSON fallback passed")

    print("\nGate 2 Testing Complete! All assertions passed.")

if __name__ == "__main__":
    test_action_from_text()
