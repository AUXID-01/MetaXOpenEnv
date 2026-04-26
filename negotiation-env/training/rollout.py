import sys
import os
import copy

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, root_dir)

from training.prompt_builder import build_system_prompt, build_turn_prompt
from client.utils import action_from_text

def run_episode(client, model_generate_fn, stage: int = 1, 
                max_turns: int = 15, curriculum_stage=None, tokenizer=None) -> dict:
    """
    Runs one full episode against the provided client. 
    Returns a trajectory dictionary shaped for GRPOTrainer or custom reward logic.
    
    `model_generate_fn` is a callable that takes a string prompt and returns a string completion.
    """
    # Try to grab tokenizer from function closure if not passed
    if tokenizer is None and hasattr(model_generate_fn, "__globals__"):
        tokenizer = model_generate_fn.__globals__.get("tokenizer")

    effective_stage = curriculum_stage if curriculum_stage is not None else stage
    obs = client.reset(curriculum_stage=effective_stage)
    if hasattr(obs, 'model_dump'):
        obs = obs.model_dump()
    
    trajectory = {
        "prompts": [],
        "completions": [],
        "rewards": [],
        "total_score": 0.0,
        "success": False,
        "turns_taken": 0,
        "reward_breakdown": {},
        "parse_failures": 0,
        "final_anger": 0.0,
        "final_trust": 0.0,
        "last_info": {},  
    }
    
    history = []
    per_turn_reward_breakdowns = [] # Addresses: Problem 3
    per_turn_adversary_states = []  # Addresses: Problem 3
    import logging
    
    for turn in range(max_turns):
        trajectory["turns_taken"] += 1
        
        # 1. Build prompt
        prompt = build_turn_prompt(obs, history)
        
        # Emergency token length check
        if tokenizer is not None:
            try:
                # tokenizer(prompt) returns a dict with 'input_ids' as a list
                token_count = len(tokenizer(prompt).input_ids)
                if token_count > 1900:
                    logging.warning(f"Turn {turn}: Prompt tokens ({token_count}) > 1900. Emergency trimming history.")
                    # Trim to last 4 turns
                    trimmed_history = history[-4:] if history else []
                    prompt = build_turn_prompt(obs, trimmed_history)
            except Exception as e:
                logging.warning(f"Token count check failed: {e}")

        trajectory["prompts"].append(prompt)
        
        # 2. Connect to model
        generated_text = model_generate_fn(prompt)
        trajectory["completions"].append(generated_text)
        
        # 3. Parse action and track failures
        action_dict = action_from_text(generated_text)
        if "<action_type>" not in generated_text:
            # We track when the model completely misses semantic structure
            trajectory["parse_failures"] += 1 
            
        # 4. Step environment
        next_obs, reward, done, info = client.step(action_dict)
        if hasattr(next_obs, 'model_dump'):
            next_obs = next_obs.model_dump()
        trajectory["rewards"].append(reward)
        trajectory["total_score"] += reward
        
        # Addresses: Problem 3
        per_turn_reward_breakdowns.append(info.get("reward_breakdown", {}))
        per_turn_adversary_states.append({
            "anger": info.get("anger"),
            "trust": info.get("trust")
        })
        
        history.append({
            "agent": action_dict.get("text", generated_text), 
            "borrower": obs["borrower_msg"]
        })
        
        # 5. Accumulate final state and metrics from info dict
        trajectory["last_info"] = info
        if "anger_after" in info:
            trajectory["final_anger"] = info["anger_after"]
        if "trust_after" in info:
            trajectory["final_trust"] = info["trust_after"]
            
        if "reward_breakdown" in info:
            for k, v in info["reward_breakdown"].items():
                trajectory["reward_breakdown"][k] = \
                    trajectory["reward_breakdown"].get(k, 0.0) + v
            
        if done:
            trajectory["success"] = (info.get("termination_reason") == "commitment_reached")
            break
            
        # 6. Hard guard on turns remaining
        turns_remaining = next_obs.get("turns_remaining", max_turns - turn - 1)
        if turns_remaining <= 0:
            break
            
        obs = next_obs
        
    # Addresses: Problem 3
    trajectory["per_turn_reward_breakdowns"] = per_turn_reward_breakdowns
    trajectory["per_turn_adversary_states"] = per_turn_adversary_states
    print_episode_transcript(trajectory)
    
    return trajectory


# Addresses: Problem 3
def print_episode_transcript(trajectory: dict):
    print("\n" + "=" * 50)
    print("EPISODE TRANSCRIPT")
    print("=" * 50)
    
    turns = trajectory.get("turns_taken", 0)
    prompts = trajectory.get("prompts", [])
    completions = trajectory.get("completions", [])
    rewards = trajectory.get("rewards", [])
    breakdowns = trajectory.get("per_turn_reward_breakdowns", [])
    states = trajectory.get("per_turn_adversary_states", [])
    
    for i in range(turns):
        print(f"\n--- Turn {i+1} / 15 ---")
        prompt_trim_len = 400
        trim_prompt = prompts[i][-prompt_trim_len:] if len(prompts[i]) > prompt_trim_len else prompts[i]
        
        print("[OBSERVATION SENT TO MODEL]:")
        print(f"...{trim_prompt}")
        print("-" * 30)
        
        print("[MODEL OUTPUT]:")
        print(completions[i])
        print("-" * 30)
        
        reward_val = rewards[i]
        sign = "+" if reward_val >= 0 else ""
        print(f"[TURN REWARD]: {sign}{reward_val:.4f}")
        
        if i < len(breakdowns):
            print("[REWARD BREAKDOWN]:")
            for k, v in breakdowns[i].items():
                if "total" not in k and "weight" not in k and not k.startswith("raw_"):
                    v_sign = "+" if v >= 0 else ""
                    print(f"  {k}: {v_sign}{v:.4f}")
                    
        if i < len(states):
            st = states[i]
            print(f"[ADVERSARY STATE]: Anger={st.get('anger')}, Trust={st.get('trust')}")

    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    
    outcome = "SUCCESS" if trajectory.get("success") else "FAIL"
    print(f"OUTCOME: {outcome}")
    total_score = trajectory.get("total_score", 0.0)
    score_sign = "+" if total_score >= 0 else ""
    print(f"Total episode score: {score_sign}{total_score:.4f}")
    print(f"Total turns taken: {turns}")
    print(f"Format failures: {trajectory.get('parse_failures', 0)}")
    print(f"Final anger: {trajectory.get('final_anger', 0.0)}")
    print(f"Final trust: {trajectory.get('final_trust', 0.0)}")
    
    print("\nEpisode-level reward breakdown:")
    for k, v in trajectory.get("reward_breakdown", {}).items():
        v_sign = "+" if v >= 0 else ""
        print(f"  {k}: {v_sign}{v:.4f}")
        
    print("=" * 50 + "\n")
