# tests/test_env_reward_integration.py
import pytest
from environment.env import NegotiationEnv
from rewards.rubric import Rubric

def test_step_with_mocked_reward(monkeypatch):
    """Verify env.step works with a fake reward from monkeypatched Rubric."""
    
    def mock_compose(self, state_before, state_after, action, episode_done):
        return 0.888, {"deescalation": 0.5, "compliance": 0.388}
        
    monkeypatch.setattr(Rubric, "compose", mock_compose)
    
    env = NegotiationEnv()
    env.reset()
    
    action = {"action_type": "send_message", "text": "Hello, how can I help?"}
    obs, reward, done, info = env.step(action)
    
    assert reward == 0.888
    assert info["reward_breakdown"]["deescalation"] == 0.5
    assert info["reward_breakdown"]["compliance"] == 0.388

def test_integration_lightweight_smoke():
    """Verify env.step works with the actual Rubric class (even if a stub)."""
    env = NegotiationEnv()
    env.reset()
    
    action = {"action_type": "send_message", "text": "Test"}
    obs, reward, done, info = env.step(action)
    
    assert isinstance(reward, (int, float))
    assert isinstance(info["reward_breakdown"], dict)
