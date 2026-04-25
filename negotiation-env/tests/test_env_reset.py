# tests/test_env_reset.py
import pytest
from environment.env import NegotiationEnv
from environment import config

def test_reset_returns_valid_observation():
    env = NegotiationEnv()
    obs = env.reset()
    
    # Check types and keys (matching models/observation.py)
    assert isinstance(obs.turn, int)
    assert obs.turn == 0
    assert isinstance(obs.borrower_msg, str)
    assert len(obs.borrower_msg) > 0
    assert obs.turns_remaining == config.MAX_TURNS
    assert obs.episode_id is not None

def test_reset_curriculum_fallback():
    env = NegotiationEnv()
    # Stage 99 doesn't exist, should fallback to PROFILES
    obs = env.reset(stage=99)
    assert obs is not None
    assert obs.turn == 0

def test_multiple_resets_are_isolated():
    env = NegotiationEnv()
    obs1 = env.reset()
    id1 = obs1.episode_id
    
    obs2 = env.reset()
    id2 = obs2.episode_id
    
    assert id1 != id2
    # Ensure turn reset
    assert obs2.turn == 0
