# %% [markdown]
# # Training Scaffolding — GRPO for Debt Negotiation
# This script connects the TRL GRPOTrainer to our Custom Env via the DummyEnvClient.

# %% [markdown]
# ### Cell 1 — Install Dependencies
# %%
# !pip install trl unsloth bitsandbytes wandb sentence-transformers requests

# %% [markdown]
# ### Cell 2 — Imports + Config
# %%
import os
import sys
import copy
from dotenv import load_dotenv
load_dotenv() # Load WANDB_API_KEY from .env
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from contracts import HF_SPACE_URL

# Adjust path for local absolute imports
sys.path.insert(0, os.path.abspath('..'))

CURRICULUM_STAGE = 1
MODEL_NAME = "unsloth/Qwen2.5-1.5B-Instruct"
MAX_TURNS = 15

# %% [markdown]
# ### Cell 3 — WandB Login
# %%
import wandb
wandb.login()
wandb.init(project="debt-negotiation-rl", config={
    "model": MODEL_NAME,
    "max_turns": MAX_TURNS,
    "stage": CURRICULUM_STAGE
})

# %% [markdown]
# ### Cell 4 — Model Load (Unsloth 4-bit QLoRA)
# %%
from unsloth import FastLanguageModel
import torch
max_seq_length = 2048
load_in_4bit = True 
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_NAME,
    max_seq_length=max_seq_length,
    load_in_4bit=load_in_4bit,
)
tokenizer.truncation_side = "left"

# %% [markdown]
# ### Cell 5 — Client Setup
# %%
from environment.env import NegotiationEnv
client = NegotiationEnv() # Direct instance for zero-latency training

# %% [markdown]
# ### Pre-Cell 6: Generation Wrapper
# %%
def generate_action(prompt: str) -> str:
    """Wrapper to generate text using the loaded unsloth model."""
    inputs = tokenizer([prompt], return_tensors="pt").to("cuda")
    outputs = model.generate(**inputs, max_new_tokens=200, temperature=0.7)
    return tokenizer.batch_decode(outputs, skip_special_tokens=True)[0][len(prompt):]

# %% [markdown]
# ### Cell 6 — Smoke Test (Gate 4)
# Run before any training to inspect parsing tracking and trajectory shape.
# %%
from training.rollout import run_episode

# Run one episode and print the output exactly as we need it for debugging formatting.
trajectory = run_episode(client, generate_action, stage=CURRICULUM_STAGE)

print(f"Turns taken: {trajectory['turns_taken']}")
print(f"Total score: {trajectory['total_score']}")
print(f"Format Fallbacks: {trajectory['parse_failures']}")

# Addresses: Problem 4
parse_failures = trajectory.get('parse_failures', 0)
turns_taken = trajectory.get('turns_taken', 1)
print(f"[SMOKE TEST] Format failure rate: {(parse_failures / max(1, turns_taken)) * 100:.1f}%")
bd = trajectory.get('reward_breakdown', {})
is_format_present = 'format_compliance' in bd
print(f"format_compliance present in breakdown: {is_format_present}")
if not is_format_present:
    print("WARNING: format_compliance is missing from reward_breakdown! Task 1 did not wire correctly.")

print("\n--- SAMPLE GENERATION (Turn 1 Baseline) ---")
print("PROMPT IN:\n", trajectory["prompts"][0][:200], "...\n")
print("MODEL OUT:\n", trajectory["completions"][0])
print("---------------------------------")

# %% [markdown]
# ### Cell 7 & 8 — Rollout Buffer, GRPOTrainer Integration, and WandB Logging
# %%
from datasets import Dataset
from trl import GRPOConfig, GRPOTrainer
from collections import deque
# import wandb

reward_window = deque(maxlen=100)

def collect_rollout_buffer(client, generate_fn, num_episodes: int = 8) -> Dataset:
    """
    Runs `num_episodes` full episodes and unpacks each turn
    into an independent (prompt, completion, reward) training example.
    
    GRPOTrainer sees flat independent examples — not episode structure.
    The multi-turn dependency is captured implicitly: each prompt already
    contains the conversation history up to that turn.
    """
    all_prompts = []
    all_completions = []
    all_rewards = []
    
    for ep in range(num_episodes):
        trajectory = run_episode(client, generate_fn, stage=CURRICULUM_STAGE)
        
        # Unpack episode into per-turn training examples
        for i in range(trajectory["turns_taken"]):
            all_prompts.append(trajectory["prompts"][i])
            all_completions.append(trajectory["completions"][i])
            all_rewards.append(trajectory["rewards"][i])
            
        # Log episode-level metrics to wandb
        reward_window.append(trajectory["total_score"])
        try:
            wandb.log({
                "episode/success": int(trajectory["success"]),
                "episode/turns": trajectory["turns_taken"],
                "reward/total": trajectory["total_score"],
                "episode/reason": trajectory.get("last_info", {}).get("termination_reason") if "last_info" in trajectory else None,
                "meta/parse_failure_rate": trajectory["parse_failures"] / max(1, trajectory["turns_taken"]),
                "state/anger": trajectory["final_anger"],
                "state/trust": trajectory["final_trust"],
                **{f"reward/{k}": v for k, v in trajectory.get("reward_breakdown", {}).items()},
                "train/stage": CURRICULUM_STAGE,
                "train/mean_reward_100": sum(reward_window) / len(reward_window),
            })
        except Exception:
            pass # handle missing wandb in local test
            
    return Dataset.from_dict({
        "prompt": all_prompts,
        "completion": all_completions,
        "reward": all_rewards,
    })

def reward_fn_from_buffer(completions: list[str], **kwargs) -> list[float]:
    """
    GRPOTrainer calls this to get rewards for its own generated completions.
    We return pre-computed rewards from the rollout buffer via kwargs.
    """
    return kwargs["reward"]  # pre-computed from rollout buffer

# --- GRPO Config (tuned for 1.5B model on Colab T4) ---
grpo_config = GRPOConfig(
    output_dir="./grpo-debt-negotiator",
    num_train_epochs=1,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=2,
    learning_rate=1e-5,
    max_prompt_length=2048,
    max_completion_length=200,
    num_generations=8,       # rollouts per prompt
    beta=0.01,               # KL penalty — keep low initially
    logging_steps=10,
    save_steps=100,
#   report_to="wandb",
)

# --- Training Loop ---
TRAIN_STEPS = 500
ROLLOUT_EVERY = 50   # collect fresh episodes every N steps

from training.curriculum_scheduler import CurriculumScheduler
scheduler = CurriculumScheduler()

for step in range(0, TRAIN_STEPS, ROLLOUT_EVERY):
    
    # 1. Collect fresh rollouts with current model
    print(f"[Step {step}] Collecting rollouts...")
    buffer = collect_rollout_buffer(client, generate_action, num_episodes=8)
    
    # 2. Train on buffer
    trainer = GRPOTrainer(
        model=model,
        args=grpo_config,
        train_dataset=buffer,
        reward_funcs=reward_fn_from_buffer,
    )
    
    # Addresses: Problem 4
    if len(buffer) > 0:
        print(f"Current step number: {step}")
        print(f"Number of examples in the training buffer: {len(buffer)}")
        print(f"Reward of the first example: {buffer[0]['reward']}")
        all_rewards = buffer["reward"]
        print(f"Min reward in buffer: {min(all_rewards)}")
        print(f"Max reward in buffer: {max(all_rewards)}")
    print("Starting GRPOTrainer.train()...")
    
    trainer.train()
    
    # Addresses: Problem 4
    print("GRPOTrainer.train() completed.")
    mean_reward_computed = sum(reward_window) / max(1, len(reward_window))
    print(f"Mean reward over last {len(reward_window)} episodes: {mean_reward_computed}")

    
    # 3. Check curriculum advance
    mean_reward_last_100 = sum(reward_window) / max(1, len(reward_window))
    if scheduler.advance_if_ready(mean_reward_last_100):
        print("Advancing curriculum stage!")
        # CURRICULUM_STAGE += 1
