import unittest
import numpy as np
import sys
import os

# Ensure we can import from the environment
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from environment.env import NegotiationEnv
from environment import config

class TestNegotiationEnv(unittest.TestCase):

    def setUp(self):
        self.env = NegotiationEnv()

    def test_reset(self):
        obs = self.env.reset()
        # In current env, reset() returns an Observation object, not a dict.
        self.assertTrue(hasattr(obs, "borrower_msg") or "borrower_msg" in obs)

    def test_step_output_format(self):
        self.env.reset()
        action = {
            "action_type": "send_message",
            "text": "I understand your concern and I am here to help you resolve this.",
            "metadata": {}
        }
        obs, reward, done, info = self.env.step(action)

        self.assertIsInstance(obs, dict)
        self.assertIsInstance(reward, float)
        self.assertIsInstance(done, bool)
        self.assertIsInstance(info, dict)

    def test_episode_terminates(self):
        self.env.reset()
        done = False
        steps = 0
        max_turns = config.MAX_TURNS

        while not done and steps < max_turns + 5:
            action = {
                "action_type": "send_message",
                "text": "Repeating message to reach timeout or termination condition.",
                "metadata": {}
            }
            obs, reward, done, _ = self.env.step(action)
            steps += 1

        self.assertTrue(done)
        self.assertLessEqual(steps, max_turns)

    def test_reward_non_zero(self):
        # Use stage 3 to ensure shaping rewards are active
        self.env.reset(curriculum_stage=3)
        rewards = []

        # Use an empathetic message to trigger positive trust/de-escalation reward
        for _ in range(5):
            action = {
                "action_type": "acknowledge_hardship",
                "text": "I truly understand your situation and I'm willing to work with you on a solution.",
                "metadata": {}
            }
            obs, reward, done, _ = self.env.step(action)
            rewards.append(reward)
            if done:
                break

        self.assertTrue(any(r != 0 for r in rewards), f"All rewards are zero: {rewards} -> RL will fail")

    def test_state_changes(self):
        self.env.reset()

        state_before = self.env.state() # state() is a method
        anger_before = state_before["anger"]
        
        # Threatening message to ensure state change (anger increase)
        action = {
            "action_type": "send_message",
            "text": "If you don't pay now, we will take legal action and visit your home.",
            "metadata": {}
        }
        self.env.step(action)
        
        state_after = self.env.state()
        anger_after = state_after["anger"]

        self.assertNotEqual(anger_before, anger_after, "State is not updating")

    def test_no_state_leakage(self):
        self.env.reset()
        action = {
            "action_type": "send_message",
            "text": "test message",
            "metadata": {}
        }
        self.env.step(action)
        self.assertNotEqual(self.env._turn, 0)

        self.env.reset()
        self.assertEqual(self.env._turn, 0)

    def test_repetition_penalty(self):
        # Must use stage 3 for anti_exploit weight > 0
        self.env.reset(curriculum_stage=3)
        msg = "I understand your concern and want to help you solve this debt problem today."
        action = {
            "action_type": "send_message",
            "text": msg,
            "metadata": {}
        }

        # First time
        _, r1, _, _ = self.env.step(action)
        # Second time (same message)
        _, r2, _, _ = self.env.step(action)

        self.assertTrue(r2 < r1, f"No repetition penalty detected (r1={r1}, r2={r2})")

    def test_invalid_action_handling(self):
        self.env.reset()

        try:
            # Passing a string instead of dict to see if it crashes or handles it.
            # Based on env.py, it expects a dict. Let's see if it handles malformed dict.
            obs, reward, done, _ = self.env.step({"action_type": "invalid", "text": "..."})
            self.assertTrue(True)
        except Exception as e:
            self.fail(f"Env crashed on invalid action: {e}")

    def test_random_policy_runs(self):
        self.env.reset()

        actions = [
            {"action_type": "send_message", "text": "Hello, how can I help you today?"},
            {"action_type": "offer_emi", "text": "We can offer you a partial payment plan.", "metadata": {"emi_amount": 1000}},
            {"action_type": "ask_open_question", "text": "Could you tell me more about your current income situation?"},
            {"action_type": "stall", "text": "Let me check our records for a moment."}
        ]

        total_reward = 0
        done = False

        while not done:
            action = actions[np.random.choice(len(actions))]
            _, reward, done, _ = self.env.step(action)
            total_reward += reward

        self.assertIsInstance(total_reward, float)


if __name__ == "__main__":
    unittest.main()
