# scripts/gate5_rollout.py
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from client.env_client import NegotiationEnvClient
from training.rollout import run_episode

BASE_URL = "https://auxid01-metaxopenenv.hf.space"
client = NegotiationEnvClient(BASE_URL)

# Dummy model — no LLM yet
dummy = lambda p: "<action_type>send_message</action_type>\n<text>I understand your situation and want to help you find a solution.</text>\n<metadata>{}</metadata>"

print("=== 5 EPISODE DUMMY ROLLOUT ===\n")

results = []
for i in range(5):
    try:
        t = run_episode(client, dummy, stage=1)
        results.append(t)
        print(f"Episode {i+1}:")
        print(f"  Turns        : {t['turns_taken']}")
        print(f"  Total score  : {t['total_score']:.4f}")
        print(f"  Success      : {t['success']}")
        print(f"  Reason       : {t['last_info'].get('termination_reason')}")
        print(f"  Final anger  : {t['final_anger']:.2f}")
        print(f"  Final trust  : {t['final_trust']:.2f}")
        print(f"  Parse fails  : {t['parse_failures']}")
        print(f"  Breakdown    : {t['reward_breakdown']}")
        print()
    except Exception as e:
        print(f"Episode {i+1} CRASHED: {e}")
        import traceback
        traceback.print_exc()

print("=== SUMMARY ===")
if len(results) == 5:
    avg_score = sum(r['total_score'] for r in results) / 5
    avg_turns = sum(r['turns_taken'] for r in results) / 5
    success_rate = sum(1 for r in results if r['success']) / 5
    print(f"All 5 episodes completed without error")
    print(f"Avg score  : {avg_score:.4f}")
    print(f"Avg turns  : {avg_turns:.1f}")
    print(f"Success rate: {success_rate*100:.0f}%")
    print()
    print("READY FOR COLAB: YES")
    print("Next step: Load Qwen 1.5B in Colab and run first real episode")
else:
    print(f"Only {len(results)}/5 episodes completed")
    print("READY FOR COLAB: NO -- fix crashes first")