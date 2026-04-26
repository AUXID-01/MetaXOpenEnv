import sys
import os

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, root_dir)

from client.env_client import DummyEnvClient
from training.rollout import run_episode

def mock_unsloth_generate(prompt: str) -> str:
    """Mocks the messy output of a real base model for the smoke test.

    We deliberately wrap the JSON object in chatty preamble and a postscript
    so the parser's greedy ``{.*}`` regex has to skip both. The contract is
    JSON: action_type / text / metadata at the top level, plus the optional
    private ``thought_process`` field.
    """
    return """
Sure, here is my plan:
{"thought_process": "Borrower sounds stressed; offer a softer EMI to drop anger.",
 "action_type": "offer_emi",
 "text": "I hear you completely. How about a 1800 EMI for 6 months?",
 "metadata": {"emi_amount": 1800}}
Hope this helps!
"""

def test_gate4():
    print("Running Gate 4 -> Colab Smoke Test (Cell 6 Mock)...\n")
    client = DummyEnvClient()
    
    trajectory = run_episode(client, mock_unsloth_generate, stage=1)
    
    print(f"Turns taken: {trajectory['turns_taken']}")
    print(f"Total score: {trajectory['total_score']:.2f}")
    print(f"Format Fallbacks (Parse Failures): {trajectory['parse_failures']}")

    print("\n--- SAMPLE GENERATION (Turn 1 Baseline) ---")
    print("PROMPT IN:\n----------------")
    print(trajectory["prompts"][0])
    print("\nMODEL OUT:\n----------------")
    print(trajectory["completions"][0])
    print("-----------------------------------------")
    
    # We verify the parser successfully extracted everything despite the messy output.
    assert trajectory["parse_failures"] == 0, "Parser failed to locate the JSON object in messy output!"
    
    print("\n[x] Gate 4 Complete! Cell 6 Smoke test logic executed flawlessly.")

if __name__ == "__main__":
    test_gate4()
