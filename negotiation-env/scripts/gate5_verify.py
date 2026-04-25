# scripts/gate5_verify.py
import requests
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from contracts import OBSERVATION_SCHEMA, REWARD_BREAKDOWN_KEYS, TERMINATION_REASONS

BASE_URL = "https://auxid01-metaxopenenv.hf.space"

passed = 0
failed = 0

def check(desc, fn):
    global passed, failed
    try:
        fn()
        print(f"  [OK] {desc}")
        passed += 1
    except Exception as e:
        print(f"  [FAIL] {desc} -- {e}")
        failed += 1

print("\n=== GATE 5 VERIFICATION ===\n")

# Health
def t1():
    r = requests.get(f"{BASE_URL}/health", timeout=10)
    assert r.status_code == 200
    assert r.json().get("status") == "ok"
check("Health endpoint returns ok", t1)

# Reset shape
def t2():
    r = requests.post(f"{BASE_URL}/reset", json={"curriculum_stage": 1}, timeout=10)
    assert r.status_code == 200
    data = r.json()
    obs = data.get("observation", data)
    for k in OBSERVATION_SCHEMA.keys():
        assert k in obs, f"Missing key: {k}"
check("POST /reset returns all OBSERVATION_SCHEMA keys", t2)

# Step shape
def t3():
    r = requests.post(f"{BASE_URL}/reset", json={"curriculum_stage": 1}, timeout=10)
    data = r.json()
    episode_id = data.get("episode_id")
    payload = {
        "episode_id": episode_id,
        "action_type": "send_message",
        "text": "I understand your situation.",
        "metadata": {}
    }
    r2 = requests.post(f"{BASE_URL}/step", json=payload, timeout=10)
    assert r2.status_code == 200
    step_data = r2.json()
    assert "observation" in step_data
    assert "reward" in step_data
    assert "done" in step_data
    assert "info" in step_data
check("POST /step returns observation, reward, done, info", t3)

# Reward breakdown keys
def t4():
    r = requests.post(f"{BASE_URL}/reset", json={"curriculum_stage": 1}, timeout=10)
    data = r.json()
    episode_id = data.get("episode_id")
    payload = {
        "episode_id": episode_id,
        "action_type": "send_message",
        "text": "I understand your situation.",
        "metadata": {}
    }
    r2 = requests.post(f"{BASE_URL}/step", json=payload, timeout=10)
    info = r2.json()["info"]
    breakdown = info.get("reward_breakdown", {})
    for k in REWARD_BREAKDOWN_KEYS:
        assert k in breakdown, f"Missing reward key: {k}"
check("reward_breakdown contains all REWARD_BREAKDOWN_KEYS", t4)

# Reward is non-zero somewhere
def t5():
    r = requests.post(f"{BASE_URL}/reset", json={"curriculum_stage": 1}, timeout=10)
    data = r.json()
    episode_id = data.get("episode_id")
    payload = {
        "episode_id": episode_id,
        "action_type": "send_message",
        "text": "We will file an FIR and take legal action against you.",
        "metadata": {}
    }
    r2 = requests.post(f"{BASE_URL}/step", json=payload, timeout=10)
    info = r2.json()["info"]
    compliance = info["reward_breakdown"].get("compliance", 0.0)
    assert compliance < 0.0, f"Expected negative compliance, got {compliance}"
check("Threatening message produces negative compliance reward", t5)

# Termination reason is valid
def t6():
    r = requests.post(f"{BASE_URL}/reset", json={"curriculum_stage": 1}, timeout=10)
    data = r.json()
    episode_id = data.get("episode_id")
    action = {"episode_id": episode_id, "action_type": "send_message", 
               "text": "I understand.", "metadata": {}}
    for _ in range(20):
        r2 = requests.post(f"{BASE_URL}/step", json=action, timeout=10)
        step = r2.json()
        if step["done"]:
            reason = step["info"].get("termination_reason")
            assert reason in TERMINATION_REASONS, f"Invalid reason: {reason}"
            break
check("Episode terminates with valid termination_reason", t6)

print(f"\n=== GATE 5 RESULT ===")
print(f"Passed: {passed} / {passed + failed}")
print(f"Failed: {failed}")
if failed == 0:
    print("\nGATE 5: GREEN -- Swap client and proceed to real model")
else:
    print("\nGATE 5: RED -- Fix failures before loading real model")