import sys
import os

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, root_dir)

from client.env_client import DummyEnvClient
from training.rollout import run_episode

def dummy_model(prompt: str) -> str:
    """Mock LLM that always outputs a valid format."""
    return "<action_type>send_message</action_type>\n<text>I hear you.</text>\n<metadata>{}</metadata>"

def test_gate3():
    print("Running Gate 3 Tests (Mock Rollout)...\n")
    client = DummyEnvClient()
    
    trajectory = run_episode(client, dummy_model, stage=1)
    
    assert "prompts" in trajectory, "Missing prompts"
    assert "completions" in trajectory, "Missing completions"
    assert "rewards" in trajectory, "Missing rewards"
    
    print(f"Turns taken     : {trajectory['turns_taken']}")
    print(f"Total score     : {trajectory['total_score']:.2f}")
    print(f"Parse failures  : {trajectory['parse_failures']}")
    print(f"Final Anger     : {trajectory['final_anger']}")
    print(f"Final Trust     : {trajectory['final_trust']}")
    
    assert len(trajectory["prompts"]) == trajectory["turns_taken"], "Prompt length mismatch"
    assert len(trajectory["completions"]) == trajectory["turns_taken"], "Completions length mismatch"
    assert len(trajectory["rewards"]) == trajectory["turns_taken"], "Rewards length mismatch"
    
    # 4 dummy steps are expected with the current DUMMY_EPISODE_ARC.
    assert trajectory["turns_taken"] == 4, f"Expected 4 dummy steps, got {trajectory['turns_taken']}"
    
    # Ensure tracking works correctly
    assert trajectory["final_anger"] is not None
    assert trajectory["final_trust"] is not None
    
    print("\n[x] Gate 3 Testing Complete! Rollout trajectory shape is rock solid.")

if __name__ == "__main__":
    test_gate3()
