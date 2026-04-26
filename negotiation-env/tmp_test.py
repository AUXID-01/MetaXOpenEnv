import sys
import os
sys.path.insert(0, os.path.abspath('.'))

from environment.env import NegotiationEnv
from training.rollout import run_episode
from environment.models.action import Action
from reward import compute_reward
from environment.models.state import State

print("1. Testing compose weight defaults...")
dummy_state = State(turn=1, max_turns=15)
dummy_action = Action(action_type="send_message", text="dummy", metadata={"raw_text": ""})
_, bd = compute_reward(dummy_state, dummy_state, dummy_action)
print(f"Format Compliance Weight applied: {bd.get('weight_format_compliance')}")

print("\n2. Testing rollout execution speed and print muting...")
env = NegotiationEnv()
def dummy_gen(prompt):
    return "<action_type>send_message</action_type>\n<text>This is a fast test.</text>"

trajectory = run_episode(env, dummy_gen, stage=1, verbose=False)
print("Rollout completed successfully without stdout spam.")
print(f"Turns taken: {trajectory['turns_taken']}")
print(f"Keys in breakdown: {list(trajectory['reward_breakdown'].keys())}")
if not any(k.startswith('weight_') for k in trajectory['reward_breakdown']):
    print("SUCCESS: reward_breakdown successfully filtered out weight_ elements.")
else:
    print("FAILED: weight_ elements are still accumulating.")

# Test Curriculum
import training.train_grpo as tg
print("\n3. Current train_grpo TRAIN_STEPS:", tg.TRAIN_STEPS)
print("Current train_grpo ROLLOUT_EVERY:", tg.ROLLOUT_EVERY)
