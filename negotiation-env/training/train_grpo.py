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
# import wandb
# wandb.login()
# wandb.init(project="debt-negotiation-rl", config={
#     "model": MODEL_NAME,
#     "max_turns": MAX_TURNS,
#     "stage": CURRICULUM_STAGE
# })

# %% [markdown]
# ### Cell 4 — Model Load (Unsloth 4-bit QLoRA)
# %%
# from unsloth import FastLanguageModel
# import torch
# max_seq_length = 2048
# load_in_4bit = True 
# model, tokenizer = FastLanguageModel.from_pretrained(
#     model_name=MODEL_NAME,
#     max_seq_length=max_seq_length,
#     load_in_4bit=load_in_4bit,
# )

# %% [markdown]
# ### Cell 5 — Client Setup
# %%
from client.env_client import DummyEnvClient, NegotiationEnvClient
client = DummyEnvClient() # Swap to NegotiationEnvClient(HF_SPACE_URL) when Gate 5 passes

# %% [markdown]
# ### Pre-Cell 6: Generation Wrapper
# %%
def generate_action(prompt: str) -> str:
    """Wrapper to generate text using the loaded unsloth model."""
    # inputs = tokenizer([prompt], return_tensors="pt").to("cuda")
    # outputs = model.generate(**inputs, max_new_tokens=150, temperature=0.7)
    # return tokenizer.batch_decode(outputs, skip_special_tokens=True)[0][len(prompt):]
    
    # Using local mock to mimic messy model output for now
    return "Let me check.\n<action_type>send_message</action_type>\n<text>I can help.</text>\n<metadata>{}</metadata>\nThank you."

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
    max_new_tokens=150,
    num_generations=8,       # rollouts per prompt
    temperature=0.7,
    beta=0.01,               # KL penalty — keep low initially
    logging_steps=10,
    save_steps=100,
#   report_to="wandb",
)

# --- Training Loop ---
"""
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
    trainer.train()
    
    # 3. Check curriculum advance
    mean_reward_last_100 = sum(reward_window) / max(1, len(reward_window))
    if scheduler.advance_if_ready(mean_reward_last_100):
        print("Advancing curriculum stage!")
        # CURRICULUM_STAGE += 1 (mock logic depending on your scheduler)
"""
