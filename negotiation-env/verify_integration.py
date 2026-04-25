# verify_integration.py
import sys
import os

# Add current dir to path
sys.path.append(os.getcwd())

from environment.env import NegotiationEnv
from reward import compose

try:
    print("Testing initialization and reset...")
    env = NegotiationEnv()
    obs = env.reset(stage=1)
    print(f"Reset successful. Episode ID: {obs.episode_id}")
    print(f"Profile context: {obs.profile_context}")

    print("\nTesting first step...")
    action = {
        "action_type": "send_message",
        "text": "Hello Ramesh, I understand you are going through a difficult time with your medical bills."
    }
    obs_dict, reward, done, info = env.step(action)
    
    print(f"Step successful. Reward: {reward}")
    print("Breakdown keys:")
    breakdown = info.get("reward_breakdown", {})
    for key in breakdown:
        print(f"  - {key}: {breakdown[key]}")

    expected_keys = [
        "outcome", "deescalation", "trust_building", "demand_coverage",
        "efficiency", "compliance", "anti_exploit"
    ]
    
    missing = [k for k in expected_keys if k not in breakdown]
    if not missing:
        print("\nSUCCESS: All 7 reward breakdown keys are present.")
    else:
        print(f"\nFAILURE: Missing keys: {missing}")

    if abs(reward - breakdown.get("total", 0)) < 1e-6:
        print("SUCCESS: Total reward matches breakdown total.")
    else:
        print(f"FAILURE: Total reward mismatch. Env Reward: {reward}, Breakdown Total: {breakdown.get('total')}")

except Exception as e:
    print(f"\nERROR during verification: {e}")
    import traceback
    traceback.print_exc()
