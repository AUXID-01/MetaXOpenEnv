# tests/test_env_privacy.py
import pytest
from environment.env import NegotiationEnv

def test_state_raises_before_reset():
    env = NegotiationEnv()
    with pytest.raises(RuntimeError, match="Call reset"):
        env.state()

def test_observation_state_separation():
    env = NegotiationEnv()
    obs = env.reset()
    obs_dict = obs.model_dump()
    state_dict = env.state()
    
    # Observation (what LLM sees)
    assert "borrower_msg" in obs_dict
    assert "escalation_level" in obs_dict
    
    # State (hidden)
    assert "real_emi_capacity" in state_dict
    assert "trust" in state_dict
    assert "anger" in state_dict
    
    # Critical Privacy Check: Observation MUST NOT contain sensitive fields
    assert "real_emi_capacity" not in obs_dict
    assert "trust" not in obs_dict
    assert "states" not in obs_dict
