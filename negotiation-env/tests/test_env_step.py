# tests/test_env_step.py
import pytest
from environment.env import NegotiationEnv
from environment import config

def test_step_contract():
    env = NegotiationEnv()
    env.reset()
    
    action = {"action_type": "send_message", "text": "Hello, how can I help you today?"}
    obs_dict, reward, done, info = env.step(action)
    
    assert isinstance(obs_dict, dict)
    assert "borrower_msg" in obs_dict
    assert isinstance(reward, (int, float))
    assert isinstance(done, bool)
    assert isinstance(info, dict)
    
    # Check info keys
    assert "reward_breakdown" in info
    assert "anger" in info
    assert "signals" in info
    assert info["turn"] == 1

def test_step_turn_increment():
    env = NegotiationEnv()
    env.reset()
    assert env._turn == 0
    
    env.step({"text": "turn 1"})
    assert env._turn == 1
    
    env.step({"text": "turn 2"})
    assert env._turn == 2

def test_step_after_termination():
    env = NegotiationEnv()
    env.reset()
    
    # Force termination
    env._terminated = True
    
    with pytest.raises(RuntimeError, match="already terminated"):
        env.step({"text": "invalid"})

def test_timeout_termination():
    env = NegotiationEnv()
    env.reset()
    
    # Fast forward to last turn
    env._turn = config.MAX_TURNS - 1
    
    obs, reward, done, info = env.step({"text": "last turn"})
    assert done is True
    assert info["termination_reason"] == "timeout"
