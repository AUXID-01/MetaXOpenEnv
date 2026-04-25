import sys
import os
import copy

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, root_dir)

from training.prompt_builder import build_system_prompt, build_turn_prompt
from client.utils import action_from_text

def run_episode(client, model_generate_fn, stage: int = 1, 
                max_turns: int = 15, curriculum_stage=None) -> dict:
    """
    Runs one full episode against the provided client. 
    Returns a trajectory dictionary shaped for GRPOTrainer or custom reward logic.
    
    `model_generate_fn` is a callable that takes a string prompt and returns a string completion.
    """
    effective_stage = curriculum_stage if curriculum_stage is not None else stage
    obs = client.reset(curriculum_stage=effective_stage)
    
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
    
    
    for turn in range(max_turns):
        trajectory["turns_taken"] += 1
        
        # 1. Build prompt
        sys_prompt = build_system_prompt()
        turn_prompt = build_turn_prompt(obs, history)
        prompt = sys_prompt + turn_prompt
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
        trajectory["rewards"].append(reward)
        trajectory["total_score"] += reward
        
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
        
    return trajectory
