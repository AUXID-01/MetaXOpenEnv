import sys
import os

# Add root so client imports work correctly
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, root_dir)

from client.utils import action_from_text


def test_action_from_text():
    print("Running Gate 2 Tests (JSON Parse Roundtrip)...")

    # 1. Happy path — valid JSON object emitted verbatim by the model.
    fake_llm_output = """
{"thought_process": "Borrower sounds stressed; offer a softer EMI to lower escalation.",
 "action_type": "offer_emi",
 "text": "I understand you're going through a difficult time. Could we look at a smaller EMI of 1800 per month?",
 "metadata": {"emi_amount": 1800}}
"""
    action = action_from_text(fake_llm_output)
    assert action["action_type"] == "offer_emi", f"Failed action_type: {action}"
    assert action["text"].startswith("I understand you're going through a difficult time."), "Failed to parse text"
    assert action["metadata"].get("emi_amount") == 1800, f"Failed metadata: {action}"
    # The private monologue must be funnelled into metadata, not into `text`.
    assert "thought_process" in action["metadata"], f"Missing thought_process in metadata: {action}"
    print("[x] Happy path passed")

    # 2. Fallback path — pure prose, no JSON object at all.
    bad_output = "Sorry I cannot help with this."
    action = action_from_text(bad_output)
    assert action["action_type"] == "send_message", "Failed to fallback action_type"
    # CRITICAL: the parser must NOT echo the raw model output into `text`,
    # otherwise the agent's chain-of-thought leaks to the borrower and the
    # TF-IDF anti-exploit penalty becomes trivial to game.
    assert action["text"] == "", f"Raw prose leaked into text: {action.get('text')!r}"
    print("[x] Fallback path (no JSON, no leak) passed")

    # 3. Invalid action_type value — JSON parses, but the action is rejected.
    bad_action = '{"action_type": "threaten_borrower", "text": "Pay now.", "metadata": {}}'
    action = action_from_text(bad_action)
    assert action["action_type"] == "send_message", "Should fallback to send_message for invalid types"
    # Text should still survive because the JSON itself is well-formed.
    assert action["text"] == "Pay now.", f"Text mangled: {action.get('text')!r}"
    print("[x] Invalid action_type fallback passed")

    # 4. Malformed JSON — extractor regex finds no closing brace, fallback fires.
    malformed = '{"action_type": "offer_emi", "text": "Here is an offer.", "metadata": {emi: 1800,}}'
    action = action_from_text(malformed)
    assert action["action_type"] == "send_message", "Malformed JSON should fallback safely"
    assert action["text"] == "", f"Malformed JSON must NOT leak text: {action.get('text')!r}"
    print("[x] Malformed JSON fallback passed")

    print("\nGate 2 Testing Complete! All assertions passed.")


if __name__ == "__main__":
    test_action_from_text()
