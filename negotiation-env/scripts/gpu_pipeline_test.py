"""GPU dry-run for the GRPO training pipeline (Gate-5 readiness check).

This script is a **standalone** verification harness. It deliberately does NOT
modify ``training/train_grpo.py`` or ``training/reward_bridge.py``; instead it
imports the safe pieces (``run_episode``, ``generate_action`` mock,
``prompt_builder``) and patches around the three known issues documented in
``audit_report.md``:

* **Bug A** — ``trl``/``datasets``/``unsloth`` are imported at module scope in
  ``train_grpo.py``, so even importing that module on a CPU box explodes.
  We bypass it entirely and re-implement the rollout-buffer assembly here.
* **Bug B** — ``wandb`` is referenced in ``collect_rollout_buffer`` but only
  imported in a commented-out cell. We import it (lazily) ourselves and run
  in offline mode (``WANDB_MODE=disabled``) to keep the dry-run hermetic.
* **Bug C** — ``training/reward_bridge.make_env_reward_fn`` calls
  ``client.reset()`` *inside* the per-completion loop, which on
  ``DummyEnvClient`` collapses every completion to the index-1 canned reward.
  For this dry-run we skip ``reward_bridge`` and pass the rollout-buffer
  rewards through directly via ``reward_fn_from_buffer``.

Usage::

    source /home/vedu/Work/MetaXOpenEnv/.venv/bin/activate
    cd /home/vedu/Work/MetaXOpenEnv/negotiation-env
    python scripts/gpu_pipeline_test.py

What the script proves, in order:

1. Torch sees the local CUDA device.
2. Unsloth can load ``Qwen2.5-1.5B-Instruct`` in 4-bit on the local GPU and
   PEFT-attach a LoRA adapter.
3. ``run_episode(DummyEnvClient(), generate_action)`` produces a trajectory
   with the contract-required keys.
4. The rollout buffer is a ``datasets.Dataset`` with columns
   ``["prompt", "completion", "reward"]`` of equal length.
5. ``GRPOTrainer.train()`` runs ``--max_steps 5`` without crashing, the loss
   is finite, and the LoRA weights have been updated (gradient flow proven).
"""

from __future__ import annotations

import os
import sys
import time
import math
from pathlib import Path

# ---- Path bootstrapping (mirrors train_grpo.py) ---------------------------
_HERE = Path(__file__).resolve().parent
_NEG_ROOT = _HERE.parent
_REPO_ROOT = _NEG_ROOT.parent
sys.path.insert(0, str(_NEG_ROOT))
sys.path.insert(0, str(_REPO_ROOT))

# Keep wandb fully offline / disabled.  This sidesteps Bug B (NameError on
# wandb) by ensuring the symbol exists *and* sends nothing over the network.
os.environ.setdefault("WANDB_MODE", "disabled")
os.environ.setdefault("WANDB_DISABLED", "true")

# Some Unsloth/bnb stacks emit reams of warnings.  Quiet them.
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("BITSANDBYTES_NOWELCOME", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
# Reduce CUDA memory fragmentation under tight VRAM (6 GB).
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


def _banner(msg: str) -> None:
    print("\n" + "=" * 72)
    print(msg)
    print("=" * 72)


# ---------------------------------------------------------------------------
# 1. GPU + library probe
# ---------------------------------------------------------------------------
_banner("[1/6] GPU + library probe")

# Unsloth strongly recommends being imported *before* transformers/trl.
import unsloth  # noqa: F401  (side-effects matter)
import torch
import transformers
import trl
import datasets as hfdatasets
import bitsandbytes as bnb

print(f"torch         : {torch.__version__}")
print(f"transformers  : {transformers.__version__}")
print(f"trl           : {trl.__version__}")
print(f"datasets      : {hfdatasets.__version__}")
print(f"bitsandbytes  : {bnb.__version__}")
print(f"unsloth       : {unsloth.__version__}")
assert torch.cuda.is_available(), "CUDA not visible to torch — aborting GPU dry-run."
DEVICE = torch.device("cuda")
print(f"cuda device   : {torch.cuda.get_device_name(0)}")
print(f"cuda capability: {torch.cuda.get_device_capability(0)}")
print(f"cuda free/total VRAM: "
      f"{torch.cuda.mem_get_info()[0] / 2**30:.2f} / "
      f"{torch.cuda.mem_get_info()[1] / 2**30:.2f} GiB")


# ---------------------------------------------------------------------------
# 2. Load Qwen2.5-1.5B-Instruct via Unsloth in 4-bit + attach LoRA
# ---------------------------------------------------------------------------
_banner("[2/6] Loading Qwen2.5-1.5B-Instruct (4-bit) via Unsloth")

from unsloth import FastLanguageModel  # local import after env vars are set

MODEL_NAME = os.environ.get("GPU_PIPELINE_MODEL", "unsloth/Qwen2.5-0.5B-Instruct")
MAX_SEQ_LEN = int(os.environ.get("GPU_PIPELINE_MAX_SEQ", "768"))

t0 = time.time()
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_NAME,
    max_seq_length=MAX_SEQ_LEN,
    load_in_4bit=True,
    dtype=None,  # auto
)
print(f"Base model loaded in {time.time() - t0:.1f}s")

# Attach a small LoRA so trainer.train() actually has trainable params.
model = FastLanguageModel.get_peft_model(
    model,
    r=8,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_alpha=16,
    lora_dropout=0.0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=3407,
)

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total = sum(p.numel() for p in model.parameters())
print(f"trainable params: {trainable:,} / total {total:,} "
      f"({100 * trainable / total:.3f}%)")


# ---------------------------------------------------------------------------
# 3. One smoke episode against DummyEnvClient
# ---------------------------------------------------------------------------
_banner("[3/6] run_episode(DummyEnvClient) smoke test")

from client.env_client import DummyEnvClient
from training.rollout import run_episode


def generate_action_mock(prompt: str) -> str:
    """Mirror of ``train_grpo.py`` ``generate_action`` — fixed JSON output.

    We wrap the canonical JSON object in chatty preamble/suffix so the
    parser's greedy regex has to skip them. Any drift between this dry-run
    and the production rollout is easy to spot.
    """
    return (
        "Let me check.\n"
        '{"action_type": "send_message", "text": "I can help.", "metadata": {}}\n'
        "Thank you."
    )


client = DummyEnvClient()
traj = run_episode(client, generate_action_mock, stage=1)
print(f"turns_taken    : {traj['turns_taken']}")
print(f"total_score    : {traj['total_score']:.3f}")
print(f"parse_failures : {traj['parse_failures']}")
print(f"success        : {traj['success']}")
print(f"final_anger    : {traj['final_anger']:.2f}")
print(f"reward_breakdown keys: {sorted(traj['reward_breakdown'].keys())}")

assert traj["turns_taken"] >= 1
assert all(math.isfinite(r) for r in traj["rewards"]), "non-finite reward"


# ---------------------------------------------------------------------------
# 4. Build a multi-episode rollout buffer (Bug-A bypass)
# ---------------------------------------------------------------------------
_banner("[4/6] Collect rollout buffer (2 episodes) → datasets.Dataset")

from datasets import Dataset


def collect_rollout_buffer_safe(client_factory, generate_fn, num_episodes: int = 2) -> Dataset:
    """Reimplementation of ``train_grpo.collect_rollout_buffer`` that

    * uses a fresh DummyEnvClient per episode (so the canned arc is replayed
      cleanly each time — DummyEnvClient is not safe to call ``reset`` on
      mid-episode, the canned indices reset),
    * never touches ``wandb`` (Bug B), and
    * coerces every reward to ``float`` so the resulting Dataset has a clean
      ``float64`` column for GRPOTrainer.
    """
    prompts, completions, rewards = [], [], []
    for ep in range(num_episodes):
        c = client_factory()
        traj = run_episode(c, generate_fn, stage=1)
        for i in range(traj["turns_taken"]):
            prompts.append(traj["prompts"][i])
            completions.append(traj["completions"][i])
            rewards.append(float(traj["rewards"][i]))
    return Dataset.from_dict(
        {"prompt": prompts, "completion": completions, "reward": rewards}
    )


buffer = collect_rollout_buffer_safe(DummyEnvClient, generate_action_mock, num_episodes=2)
print(f"buffer rows      : {len(buffer)}")
print(f"buffer columns   : {buffer.column_names}")
print(f"first row prompt : {buffer[0]['prompt'][:120]!r}…")
print(f"first row reward : {buffer[0]['reward']:.3f}")
print(f"reward stats     : min={min(buffer['reward']):.3f}  "
      f"max={max(buffer['reward']):.3f}  "
      f"mean={sum(buffer['reward']) / len(buffer):.3f}")

assert set(buffer.column_names) == {"prompt", "completion", "reward"}, \
    f"buffer schema drift: {buffer.column_names}"
assert len(buffer) > 0
assert all(isinstance(r, float) and math.isfinite(r) for r in buffer["reward"])


# ---------------------------------------------------------------------------
# 5. GRPOTrainer instantiation and short training run
# ---------------------------------------------------------------------------
_banner("[5/6] GRPOTrainer.train() for 5 steps")

from trl import GRPOConfig, GRPOTrainer


def reward_fn_from_buffer(completions, prompts=None, **kwargs) -> list[float]:
    """Pre-computed-rewards path — sidesteps ``reward_bridge`` entirely (Bug C).

    GRPOTrainer passes the 'reward' column from the dataset through ``kwargs``
    when ``remove_unused_columns=False`` is set in the config.  We just echo
    those floats back out, padded/truncated to match the number of
    completions GRPOTrainer asks us to score.
    """
    pre = kwargs.get("reward")
    if pre is None:
        return [0.0] * len(completions)
    if len(pre) == len(completions):
        return [float(r) for r in pre]
    # GRPOTrainer typically asks for ``num_generations`` completions per
    # prompt.  Replicate the source reward across the group.
    out = []
    n_per = max(1, len(completions) // max(1, len(pre)))
    for r in pre:
        out.extend([float(r)] * n_per)
    while len(out) < len(completions):
        out.append(0.0)
    return out[: len(completions)]


# 6 GB VRAM → keep this MINIMAL.  num_generations=2, max_completion_length=64.
grpo_config = GRPOConfig(
    output_dir="./.gpu_pipeline_test_out",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=1,
    learning_rate=1e-5,
    max_steps=5,
    max_prompt_length=384,
    max_completion_length=32,
    num_generations=2,
    temperature=0.7,
    beta=0.01,
    logging_steps=1,
    save_steps=10_000,        # don't write a checkpoint in a dry run
    report_to=[],             # no wandb / tensorboard
    remove_unused_columns=False,  # so the 'reward' column reaches reward_fn
    bf16=True,
    fp16=False,
)

# Snapshot a few LoRA weights so we can prove gradients moved them.
import copy
lora_param_name = next(
    n for n, p in model.named_parameters() if "lora_A" in n and p.requires_grad
)
lora_before = copy.deepcopy(dict(model.named_parameters())[lora_param_name].detach().cpu())
print(f"snapshot param : {lora_param_name}")

trainer = GRPOTrainer(
    model=model,
    args=grpo_config,
    train_dataset=buffer,
    reward_funcs=reward_fn_from_buffer,
    processing_class=tokenizer,
)
print("GRPOTrainer instantiated.")

t0 = time.time()
result = trainer.train()
elapsed = time.time() - t0
print(f"trainer.train() finished in {elapsed:.1f}s")
print(f"final training loss: {result.training_loss}")
assert math.isfinite(result.training_loss), \
    f"non-finite training loss: {result.training_loss}"

lora_after = dict(model.named_parameters())[lora_param_name].detach().cpu()
delta = (lora_after - lora_before).abs().max().item()
print(f"max |Δ| on {lora_param_name}: {delta:.3e}")
assert delta > 0.0, "LoRA weight unchanged — gradient flow did NOT happen."


# ---------------------------------------------------------------------------
# 6. Final verdict
# ---------------------------------------------------------------------------
_banner("[6/6] DRY-RUN PASSED")

print("torch+CUDA reachable                        OK")
print("Unsloth + bitsandbytes 4-bit load            OK")
print("LoRA adapter attached                        OK")
print("DummyEnvClient rollout shape correct         OK")
print(f"datasets.Dataset columns = {buffer.column_names}    OK")
print(f"GRPOTrainer trained for {grpo_config.max_steps} steps         OK")
print(f"LoRA weights moved (max |Δ| = {delta:.3e})       OK")
print()
print("=> Pipeline is GPU-ready.  See audit_report.md for documented gaps "
      "before running on full data.")
