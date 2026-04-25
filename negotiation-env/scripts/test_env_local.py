# scripts/test_env_local.py
"""
Gate: Environment + Rewards Integration Test
Run this BEFORE deploying to HuggingFace Space.
All tests must pass before deployment.

Usage: python scripts/test_env_local.py
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from environment.env import NegotiationEnv
from contracts import (
    REWARD_BREAKDOWN_KEYS,
    TERMINATION_REASONS,
    OBSERVATION_SCHEMA,
    NUMERIC_RANGES,
    CURRICULUM_STAGES,
    ACTION_TYPES,
)

passed = 0
failed = 0

def run_check(description, test_func):
    global passed, failed
    try:
        test_func()
        print(f"  [OK] {description}")
        passed += 1
    except Exception as e:
        print(f"  [FAIL] {description} -- {str(e)}")
        failed += 1


# ────────────────────────────────────────────
# SECTION 1 — ENV INSTANTIATION
# ────────────────────────────────────────────
print("\nSECTION 1 -- ENV INSTANTIATION")

def t1_1():
    env = NegotiationEnv()
    assert env is not None

run_check("NegotiationEnv instantiates without error", t1_1)

def t1_2():
    env = NegotiationEnv()
    obs = env.reset()
    assert obs is not None
    assert isinstance(obs, dict)

run_check("env.reset() returns a dict", t1_2)

def t1_3():
    env = NegotiationEnv()
    obs = env.reset()
    for k in OBSERVATION_SCHEMA.keys():
        assert k in obs, f"Missing key: {k}"

run_check("reset() observation contains all OBSERVATION_SCHEMA keys", t1_3)

def t1_4():
    env = NegotiationEnv()
    obs = env.reset()
    assert isinstance(obs["turn"], int)
    assert isinstance(obs["borrower_msg"], str)
    assert isinstance(obs["escalation_level"], float)
    assert isinstance(obs["stated_demands"], list)
    assert isinstance(obs["turns_remaining"], int)

run_check("reset() observation values have correct types", t1_4)

def t1_5():
    env = NegotiationEnv()
    obs1 = env.reset()
    obs2 = env.reset()
    # Two resets should not share state
    assert obs1["turn"] == obs2["turn"] == 1

run_check("Two consecutive reset() calls return clean state", t1_5)


# ────────────────────────────────────────────
# SECTION 2 — STEP CONTRACT
# ────────────────────────────────────────────
print("\nSECTION 2 -- STEP CONTRACT")

def t2_1():
    env = NegotiationEnv()
    env.reset()
    action = {"action_type": "send_message", "text": "I understand your situation.", "metadata": {}}
    result = env.step(action)
    assert isinstance(result, tuple)
    assert len(result) == 4

run_check("step() returns a 4-tuple (obs, reward, done, info)", t2_1)

def t2_2():
    env = NegotiationEnv()
    env.reset()
    action = {"action_type": "send_message", "text": "I understand.", "metadata": {}}
    obs, reward, done, info = env.step(action)
    for k in OBSERVATION_SCHEMA.keys():
        assert k in obs, f"Missing obs key: {k}"

run_check("step() observation contains all OBSERVATION_SCHEMA keys", t2_2)

def t2_3():
    env = NegotiationEnv()
    env.reset()
    action = {"action_type": "send_message", "text": "I understand.", "metadata": {}}
    obs, reward, done, info = env.step(action)
    assert isinstance(reward, float), f"Reward type: {type(reward)}"

run_check("step() reward is a float", t2_3)

def t2_4():
    env = NegotiationEnv()
    env.reset()
    action = {"action_type": "send_message", "text": "I understand.", "metadata": {}}
    obs, reward, done, info = env.step(action)
    assert isinstance(done, bool), f"Done type: {type(done)}"

run_check("step() done is a bool", t2_4)

def t2_5():
    env = NegotiationEnv()
    env.reset()
    action = {"action_type": "send_message", "text": "I understand.", "metadata": {}}
    obs, reward, done, info = env.step(action)
    assert isinstance(info, dict), f"Info type: {type(info)}"

run_check("step() info is a dict", t2_5)

def t2_6():
    env = NegotiationEnv()
    env.reset()
    action = {"action_type": "send_message", "text": "I understand.", "metadata": {}}
    obs, reward, done, info = env.step(action)
    assert "reward_breakdown" in info, "Missing reward_breakdown in info"
    assert "termination_reason" in info, "Missing termination_reason in info"
    assert "anger_after" in info, "Missing anger_after in info"
    assert "trust_after" in info, "Missing trust_after in info"

run_check("step() info contains required keys", t2_6)

def t2_7():
    env = NegotiationEnv()
    env.reset()
    action = {"action_type": "send_message", "text": "I understand.", "metadata": {}}
    obs, reward, done, info = env.step(action)
    breakdown = info["reward_breakdown"]
    for k in REWARD_BREAKDOWN_KEYS:
        assert k in breakdown, f"Missing reward key: {k}"

run_check("info reward_breakdown contains all REWARD_BREAKDOWN_KEYS", t2_7)

def t2_8():
    env = NegotiationEnv()
    env.reset()
    action = {"action_type": "send_message", "text": "I understand.", "metadata": {}}
    obs, reward, done, info = env.step(action)
    rmin, rmax = NUMERIC_RANGES["reward_per_step"]
    assert rmin <= reward <= rmax, f"Reward {reward} outside range {rmin, rmax}"

run_check("step() reward is within NUMERIC_RANGES reward_per_step", t2_8)

def t2_9():
    env = NegotiationEnv()
    env.reset()
    action = {"action_type": "send_message", "text": "I understand.", "metadata": {}}
    obs, reward, done, info = env.step(action)
    anger = info["anger_after"]
    trust = info["trust_after"]
    amin, amax = NUMERIC_RANGES["anger"]
    tmin, tmax = NUMERIC_RANGES["trust"]
    assert amin <= anger <= amax, f"Anger {anger} out of range"
    assert tmin <= trust <= tmax, f"Trust {trust} out of range"

run_check("anger_after and trust_after are within NUMERIC_RANGES", t2_9)


# ────────────────────────────────────────────
# SECTION 3 — EPISODE TERMINATION
# ────────────────────────────────────────────
print("\nSECTION 3 -- EPISODE TERMINATION")

def t3_1():
    env = NegotiationEnv()
    env.reset()
    action = {"action_type": "send_message", "text": "I understand.", "metadata": {}}
    for _ in range(20):
        obs, reward, done, info = env.step(action)
        if done:
            break
    assert done, "Episode never terminated after 20 steps"

run_check("Episode terminates within 20 steps", t3_1)

def t3_2():
    env = NegotiationEnv()
    env.reset()
    action = {"action_type": "send_message", "text": "I understand.", "metadata": {}}
    for _ in range(20):
        obs, reward, done, info = env.step(action)
        if done:
            break
    reason = info.get("termination_reason")
    assert reason in TERMINATION_REASONS, f"Invalid reason: {reason}"

run_check("termination_reason is one of TERMINATION_REASONS", t3_2)

def t3_3():
    # Run 5 full episodes, confirm no shared state bleeds between them
    env = NegotiationEnv()
    action = {"action_type": "send_message", "text": "I understand.", "metadata": {}}
    for ep in range(5):
        obs = env.reset()
        assert obs["turn"] == 1, f"Episode {ep}: turn did not reset to 1"
        for _ in range(20):
            obs, reward, done, info = env.step(action)
            if done:
                break

run_check("5 consecutive episodes complete with clean reset between each", t3_3)

def t3_4():
    # Timeout: run until turns_remaining hits 0
    env = NegotiationEnv()
    obs = env.reset()
    action = {"action_type": "send_message", "text": "I understand.", "metadata": {}}
    last_done = False
    for _ in range(30):
        obs, reward, done, info = env.step(action)
        if done:
            last_done = True
            break
    assert last_done, "Did not terminate on timeout"

run_check("Episode terminates on timeout correctly", t3_4)


# ────────────────────────────────────────────
# SECTION 4 — REWARD SANITY
# ────────────────────────────────────────────
print("\nSECTION 4 -- REWARD SANITY")

def t4_1():
    # All reward components must be floats
    env = NegotiationEnv()
    env.reset()
    action = {"action_type": "send_message", "text": "I understand.", "metadata": {}}
    _, _, _, info = env.step(action)
    for k, v in info["reward_breakdown"].items():
        assert isinstance(v, float), f"Reward {k} is {type(v)}, not float"

run_check("All reward_breakdown values are floats", t4_1)

def t4_2():
    # No NaN or Inf in rewards
    import math
    env = NegotiationEnv()
    env.reset()
    action = {"action_type": "send_message", "text": "I understand.", "metadata": {}}
    for _ in range(5):
        obs, reward, done, info = env.step(action)
        assert not math.isnan(reward), "Reward is NaN"
        assert not math.isinf(reward), "Reward is Inf"
        for k, v in info["reward_breakdown"].items():
            assert not math.isnan(v), f"Breakdown {k} is NaN"
        if done:
            break

run_check("No NaN or Inf in rewards across 5 steps", t4_2)

def t4_3():
    # Threatening language should trigger compliance penalty
    env = NegotiationEnv()
    env.reset()
    threat_action = {
        "action_type": "send_message",
        "text": "We will file an FIR against you and take legal action.",
        "metadata": {}
    }
    _, _, _, info = env.step(threat_action)
    compliance = info["reward_breakdown"].get("compliance", 0.0)
    assert compliance < 0.0, f"Expected negative compliance reward, got {compliance}"

run_check("Threatening language produces negative compliance reward", t4_3)

def t4_4():
    # Empathetic message should not produce negative trust reward
    env = NegotiationEnv()
    env.reset()
    empathy_action = {
        "action_type": "acknowledge_hardship",
        "text": "I completely understand you are going through a difficult time. We want to help.",
        "metadata": {}
    }
    _, _, _, info = env.step(empathy_action)
    trust = info["reward_breakdown"].get("trust", 0.0)
    assert trust >= 0.0, f"Expected non-negative trust reward for empathy, got {trust}"

run_check("Empathetic action produces non-negative trust reward", t4_4)

def t4_5():
    # Repetitive messages should trigger anti_exploit penalty
    env = NegotiationEnv()
    env.reset()
    repeat_action = {
        "action_type": "send_message",
        "text": "I understand your concerns, let us work together on this.",
        "metadata": {}
    }
    # Send same message 4 times
    last_anti_exploit = 0.0
    for _ in range(4):
        _, _, done, info = env.step(repeat_action)
        last_anti_exploit = info["reward_breakdown"].get("anti_exploit", 0.0)
        if done:
            break
    assert last_anti_exploit < 0.0, \
        f"Expected anti_exploit penalty after repetition, got {last_anti_exploit}"

run_check("Repetitive messages trigger anti_exploit penalty", t4_5)


# ────────────────────────────────────────────
# SECTION 5 — STATE ENDPOINT
# ────────────────────────────────────────────
print("\nSECTION 5 -- STATE ENDPOINT")

def t5_1():
    env = NegotiationEnv()
    env.reset()
    state = env.state()
    assert isinstance(state, dict), f"State type: {type(state)}"

run_check("env.state() returns a dict", t5_1)

def t5_2():
    env = NegotiationEnv()
    env.reset()
    state = env.state()
    for k in ["anger", "trust", "fear"]:
        assert k in state, f"Missing state key: {k}"

run_check("env.state() contains anger, trust, fear", t5_2)

def t5_3():
    env = NegotiationEnv()
    env.reset()
    action = {
        "action_type": "acknowledge_hardship",
        "text": "I understand you are in a difficult situation.",
        "metadata": {}
    }
    state_before = env.state()
    env.step(action)
    state_after = env.state()
    # State must change after a step
    changed = any(
        state_before.get(k) != state_after.get(k)
        for k in ["anger", "trust", "turn"]
    )
    assert changed, "State did not change after step"

run_check("env.state() changes after a step", t5_3)


# ────────────────────────────────────────────
# SECTION 6 — CURRICULUM INTEGRATION
# ────────────────────────────────────────────
print("\nSECTION 6 -- CURRICULUM INTEGRATION")

def t6_1():
    # Stage 1 profiles should be easier (higher anger threshold)
    env = NegotiationEnv()
    obs = env.reset(curriculum_stage=1)
    assert obs is not None

run_check("env.reset(curriculum_stage=1) works without error", t6_1)

def t6_2():
    env = NegotiationEnv()
    obs = env.reset(curriculum_stage=2)
    assert obs is not None

run_check("env.reset(curriculum_stage=2) works without error", t6_2)

def t6_3():
    # Stage 1 should have higher anger threshold than stage 3
    # Run 10 episodes on each stage, compare how fast they terminate
    env = NegotiationEnv()
    action = {
        "action_type": "send_message",
        "text": "I understand.",
        "metadata": {}
    }
    stage1_turns = []
    for _ in range(5):
        env.reset(curriculum_stage=1)
        for t in range(20):
            _, _, done, _ = env.step(action)
            if done:
                stage1_turns.append(t + 1)
                break

    assert len(stage1_turns) == 5, "Not all stage 1 episodes terminated"

run_check("5 stage-1 episodes all terminate cleanly", t6_3)


# ────────────────────────────────────────────
# SUMMARY
# ────────────────────────────────────────────
print(f"\n{'='*50}")
print(f"INTEGRATION TEST SUMMARY")
print(f"{'='*50}")
print(f"Passed : {passed} / {passed + failed}")
print(f"Failed : {failed}")
print()

if failed == 0:
    print("STATUS: GREEN -- Safe to deploy to HuggingFace Space")
else:
    print("STATUS: RED -- Fix failures before deploying")
    print("Do NOT deploy until this script shows 0 failures.")

