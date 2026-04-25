"""
training/reward_bridge.py

Why this bridge exists:
GRPOTrainer is built for stateless question/answer tasks where each prompt/completion pair can be scored independently without affecting an environment. Our environment, however, is a stateful interactive system (like the FastAPI backend or DummyEnvClient). By embedding the conversation history completely inside the prompt, we isolate each turn into a stateless input. This bridge lets GRPOTrainer evaluate those independent turns by parsing the actions and stepping the environment.

How to wire it into GRPOTrainer:
> from training.reward_bridge import make_env_reward_fn
> env_reward_fn = make_env_reward_fn(client)
> trainer = GRPOTrainer(..., reward_funcs=[env_reward_fn])

Known limitation:
Each completion is stepped and scored independently in a single turn context. Because GRPOTrainer evaluates advantages over independent generations and does not compute generalized delayed returns across multiple turns, temporal credit assignment is merely approximate. The agent will strongly prefer immediate turn-by-turn rewards over deep delayed negotiation outcomes.
"""

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from client.utils import action_from_text
from contracts import NUMERIC_RANGES

def make_env_reward_fn(client):
    """
    Creates a reward function closure that binds the environment client.
    """
    def reward_fn(prompts: list[str], completions: list, **kwargs) -> list[float]:
        rewards = []
        min_reward, max_reward = NUMERIC_RANGES["reward_per_step"]
        
        for completion in completions:
            try:
                # Handle both string and dict format
                if isinstance(completion, list):
                    text = completion[0].get("content", "") if completion else ""
                elif isinstance(completion, dict):
                    text = completion.get("content", str(completion))
                else:
                    text = str(completion)
                
                client.reset()
                action_dict = action_from_text(text)
                _, raw_reward, _, _ = client.step(action_dict)
                clipped = max(min_reward, min(max_reward, float(raw_reward)))
                rewards.append(clipped)
            except Exception:
                rewards.append(0.0)
        
        return rewards
        
    return reward_fn

def reset_env_for_batch(client, batch_size: int) -> list:
    """
    Calls client.reset() once per item in batch to yield list of initial obs dicts.
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
    mock_completions = [
        "<action_type>send_message</action_type><text>Hello</text>",
        "<action_type>offer_emi</action_type><text>How about 2000?</text><metadata>{\"emi_amount\": 2000}</metadata>",
        "Malformed completion xml broken"
    ]
    
    print("Evaluating 3-step completions...")
    rewards = reward_fn(mock_prompts, mock_completions)
    
    for i, (comp, rew) in enumerate(zip(mock_completions, rewards)):
        print(f"Step {i+1} Reward: {rew}")
        
    print("\nBridge smoke test complete!")
