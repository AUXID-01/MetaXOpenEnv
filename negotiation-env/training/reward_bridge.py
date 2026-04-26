"""
training/reward_bridge.py

Why this bridge exists:
    GRPOTrainer is built for stateless question/answer tasks where each
    prompt/completion pair can be scored independently without affecting an
    environment. Our environment, however, is a stateful interactive system
    (the FastAPI backend or DummyEnvClient). By embedding the conversation
    history completely inside the prompt, we isolate each turn into a
    stateless input. This bridge lets GRPOTrainer evaluate those independent
    turns by parsing the actions and stepping the environment.

How to wire it into GRPOTrainer:
    >>> from training.reward_bridge import make_env_reward_fn
    >>> env_reward_fn = make_env_reward_fn(client)
    >>> trainer = GRPOTrainer(..., reward_funcs=[env_reward_fn])

History note (Bug-C fix):
    A prior version called ``client.reset()`` *inside* the per-completion
    loop. That destructively rewound the active episode every time GRPO
    asked for a reward, so every completion was scored against the canned
    turn-1 state and multi-turn temporal credit signal collapsed. The reset
    has been removed: episode lifecycle is now the caller's responsibility,
    managed at batch boundaries via :func:`reset_env_for_batch`.

Known limitation:
    Each completion is stepped and scored independently in a single-turn
    context. Because GRPOTrainer evaluates advantages over independent
    generations and does not compute generalised delayed returns across
    multiple turns, temporal credit assignment is approximate. The agent
    will strongly prefer immediate per-turn rewards over deep delayed
    negotiation outcomes.
"""

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from client.utils import action_from_text
from contracts import NUMERIC_RANGES


def make_env_reward_fn(client):
    """Build a GRPO-compatible reward function bound to ``client``.

    The returned closure scores each completion by parsing it into an
    action dict and stepping the environment **without resetting**. The
    caller manages episode boundaries (e.g. via :func:`reset_env_for_batch`
    at the start of each rollout collection).
    """
    def reward_fn(prompts: list[str], completions: list, **kwargs) -> list[float]:
        rewards = []
        min_reward, max_reward = NUMERIC_RANGES["reward_per_step"]

        for completion in completions:
            try:
                # Handle string, dict, and list-of-dict completion formats.
                if isinstance(completion, list):
                    text = completion[0].get("content", "") if completion else ""
                elif isinstance(completion, dict):
                    text = completion.get("content", str(completion))
                else:
                    text = str(completion)

                # NOTE — do NOT call client.reset() here. Resetting per-
                # completion was the Bug-C state-nuking pathology: every
                # alternate generation got scored against the canned turn-1
                # state, destroying multi-turn credit assignment and quietly
                # collapsing rewards to identical values. Reset belongs at
                # episode boundaries (see reset_env_for_batch).
                action_dict = action_from_text(text)
                _, raw_reward, _, _ = client.step(action_dict)
                clipped = max(min_reward, min(max_reward, float(raw_reward)))
                rewards.append(clipped)
            except Exception:
                rewards.append(0.0)

        return rewards

    return reward_fn


def reset_env_for_batch(client, batch_size: int) -> list:
    """Calls ``client.reset()`` once per item in batch and returns the
    initial observation dicts. Use this at the *start* of a rollout pass —
    never inside :func:`make_env_reward_fn`'s inner loop.
    """
    initial_states = []
    for _ in range(batch_size):
        initial_states.append(client.reset())
    return initial_states


if __name__ == "__main__":
    from client.env_client import DummyEnvClient

    print("=== Testing Reward Bridge ===")
    mock_client = DummyEnvClient()

    initial_obs = reset_env_for_batch(mock_client, batch_size=1)

    reward_fn = make_env_reward_fn(mock_client)

    mock_prompts = ["prompt1", "prompt2", "prompt3"]
    # JSON contract — see training/prompt_builder.build_system_prompt.
    # The third fixture is intentionally malformed to exercise the parser's
    # safe fallback (action_from_text returns text="" so it cannot leak the
    # model's chain-of-thought to the adversary).
    mock_completions = [
        '{"action_type": "send_message", "text": "Hello, can we talk about your repayment plan today?", "metadata": {}}',
        '{"action_type": "offer_emi", "text": "How about Rs 2000 per month for the next six months?", "metadata": {"emi_amount": 2000}}',
        "Malformed completion — no JSON, broken contract.",
    ]

    print("Evaluating 3-step completions...")
    rewards = reward_fn(mock_prompts, mock_completions)

    for i, (comp, rew) in enumerate(zip(mock_completions, rewards)):
        print(f"Step {i + 1} Reward: {rew}")

    print("\nBridge smoke test complete!")
