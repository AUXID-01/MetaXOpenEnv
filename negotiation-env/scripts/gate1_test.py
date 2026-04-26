import sys
import os

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, root_dir)

from training.prompt_builder import build_system_prompt, build_turn_prompt
from client.env_client import DUMMY_EPISODE_ARC

def test_gate1():
    print("Running Gate 1 Tests (Prompt Sanity Check)...\n")
    
    obs = DUMMY_EPISODE_ARC[0]
    sys_prompt = build_system_prompt()
    turn_prompt = build_turn_prompt(obs)
    
    print("=== SYSTEM PROMPT ===")
    print(sys_prompt)
    print("\n=== TURN PROMPT ===")
    print(turn_prompt)
    
    # Simple assertions to ensure correct formatting (JSON contract).
    assert "You are a professional" in sys_prompt
    assert '"action_type"' in sys_prompt and '"text"' in sys_prompt and '"thought_process"' in sys_prompt
    assert "Turn: 1" in turn_prompt
    assert "I can't pay anything right now" in turn_prompt
    assert "need 3 month moratorium" in turn_prompt
    
    print("\n[x] Gate 1 Testing Complete! Assertions passed.")

if __name__ == "__main__":
    test_gate1()
