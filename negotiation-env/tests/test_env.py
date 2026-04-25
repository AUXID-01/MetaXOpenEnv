# tests/test_env.py
import pytest
from environment.env import NegotiationEnv

def test_imports_work():
    from environment.env import NegotiationEnv
    from environment.adversary import BorrowerAdversary
    from environment.classifier import classify_action
    from environment.response_generator import pick_template
    assert True

def test_basic_turn_flow():
    env = NegotiationEnv()
    obs = env.reset()
    assert obs.turn == 0
    
    action = {"action_type": "send_message", "text": "I understand you are stressed about the job loss."}
    obs, reward, done, info = env.step(action)
    
    assert info["turn"] == 1
    assert done is False

def test_reproducibility():
    """With a fixed seed, reset should pick the same profile."""
    env1 = NegotiationEnv(seed=42)
    obs1 = env1.reset()
    p1 = env1._profile["id"]
    
    env2 = NegotiationEnv(seed=42)
    obs2 = env2.reset()
    p2 = env2._profile["id"]
    
    assert p1 == p2
    assert obs1.borrower_msg == obs2.borrower_msg

def test_no_shared_mutable_state():
    env1 = NegotiationEnv()
    env1.reset()
    env1.step({"text": "Hello"})
    
    env2 = NegotiationEnv()
    env2.reset()
    assert env2._turn == 0
    assert len(env2._episode_history) == 1 # only opening msg
