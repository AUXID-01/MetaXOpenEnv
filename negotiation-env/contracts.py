# ============================================================
# contracts.py
# Location: negotiation-env/contracts.py  (project root)
#
# PURPOSE:
#   Single source of truth for all cross-person data contracts.
#   Every locked key, every boundary rule, every handoff shape
#   lives here and ONLY here.
#
# WHO READS THIS FILE:
#   Person A → imports REWARD_BREAKDOWN_KEYS, TERMINATION_REASONS
#   Person B → imports REWARD_BREAKDOWN_KEYS
#   Person C → imports REWARD_BREAKDOWN_KEYS, ACTION_TYPES
#   api/      → imports TERMINATION_REASONS
#   tests/    → imports everything for contract validation
#
# CHANGE POLICY:
#   Any change to this file requires all three persons to acknowledge.
#   Do not change this file mid-hackathon without a team sync.
#   Version this file — add a comment with date + initials on any edit.
# ============================================================


# ────────────────────────────────────────────────────────────
# SECTION 1 — HANDOFF CONTRACT SUMMARY (human-readable)
# ────────────────────────────────────────────────────────────
#
#  PERSON A  →  PERSON B
#  ─────────────────────
#  WHERE:    Inside env.step(), after adversary.react() returns.
#  WHAT:     Person A builds a RubricInput object and calls
#            rubric.compose(rubric_input).
#  RETURNS:  (total_reward: float, breakdown: Dict[str, float])
#  CONTRACT: RubricInput dataclass — defined in environment/models/rubric_input.py
#  RULE:     Person B NEVER imports env.py, adversary.py, classifier.py,
#            or response_generator.py. If B needs a new field, A adds it
#            to RubricInput and populates it. B reads it from there.
#
#  PERSON A  →  PERSON C
#  ─────────────────────
#  WHERE:    HTTP. Person C calls POST /step from Colab.
#  WHAT:     Person A's api/routes.py returns StepResponse JSON.
#            Person C's client/env_client.py parses it into ClientStepResult.
#  CONTRACT: StepResponse JSON shape — keys locked in STEP_RESPONSE_KEYS below.
#  RULE:     Person C NEVER imports environment/, rewards/, or api/ directly.
#            All communication is HTTP only. If C needs new data from the env,
#            A adds it to StepResponse.info and C reads it from there.
#
#  PERSON B  →  PERSON C
#  ─────────────────────
#  WHERE:    Indirect. B's output travels through A's StepResult.info.
#  WHAT:     rubric.compose() returns breakdown dict →
#            A puts it in StepResult.info["reward_breakdown"] →
#            HTTP → C's ClientStepResult.reward_breakdown →
#            C logs each key as a separate wandb column.
#  CONTRACT: reward_breakdown key names — locked in REWARD_BREAKDOWN_KEYS below.
#  RULE:     B and C never communicate directly. B's key names must exactly
#            match C's wandb column names. Both sides read from
#            REWARD_BREAKDOWN_KEYS — never hardcode key strings in two places.


# ────────────────────────────────────────────────────────────
# SECTION 2 — REWARD BREAKDOWN KEYS
# Locked across Person B (writes them) and Person C (reads them)
# ────────────────────────────────────────────────────────────

# Exact shape returned in info dict from env.step()
REWARD_BREAKDOWN_SCHEMA = {
    "outcome": 0.0,           # +1.0 on commitment_reached, else 0.0
    "deescalation": 0.0,      # delta anger per turn
    "trust": 0.0,             # delta trust per turn
    "demand_coverage": 0.0,   # fraction of demands addressed
    "efficiency": 0.0,        # bonus for resolving fast
    "compliance": 0.0,        # penalty for RBI violations
    "anti_exploit": 0.0,      # penalty for repetitive messages
}
REWARD_BREAKDOWN_KEYS: list[str] = list(REWARD_BREAKDOWN_SCHEMA.keys())
# HOW TO ADD A NEW REWARD FUNCTION:
#   1. Person B writes the new .py file in rewards/
#   2. Person B adds the key name here (one line)
#   3. Person B imports and calls it in rubric.compose()
#   4. Person C's wandb logging picks it up automatically — no change needed
#   5. Person A adds the key to STEP_RESPONSE_KEYS.info_keys if needed


# ────────────────────────────────────────────────────────────
# SECTION 3 — ACTION TYPES
# Locked across Person A (validates them) and Person C (sends them)
# ────────────────────────────────────────────────────────────

ACTION_TYPES: list[str] = [
    "send_message",
    "offer_emi",
    "acknowledge_hardship",
    "request_clarification",
    "escalate_authority",
    "stall",
]
# HOW TO ADD A NEW ACTION TYPE:
#   1. Person A adds it here
#   2. Person A adds the classifier rule in classifier.py
#   3. Person A adds the response template in response_generator.py
#   4. Person C updates the system prompt in prompt_builder.py
#   5. Person C updates action_from_text() parser in client/utils.py


# ────────────────────────────────────────────────────────────
# SECTION 4 — TERMINATION REASONS
# Locked across Person A (sets them) and Person B + C (read them)
# ────────────────────────────────────────────────────────────

TERMINATION_REASONS: list[str] = [
    "commitment_reached",
    "threshold_breach",
    "timeout",
    "forbidden_move",
]

# Exact Observation keys
OBSERVATION_SCHEMA = {
    "turn": 0,
    "borrower_msg": "",
    "escalation_level": 0.0,
    "stated_demands": [],
    "turns_remaining": 15,
}
# Person B: outcome.py checks reason == "commitment_reached" for +1.0
# Person C: wandb logs reason as episode/reason column
# Person A: adversary.py and env.py set the reason string


# ────────────────────────────────────────────────────────────
# SECTION 5 — STEP RESPONSE INFO KEYS
# Locked across Person A (writes info dict) and Person C (reads it)
# ────────────────────────────────────────────────────────────

STEP_RESPONSE_INFO_KEYS: list[str] = [
    "reward_breakdown",      # Dict[str, float] — one key per REWARD_BREAKDOWN_KEYS
    "termination_reason",    # str | None — from TERMINATION_REASONS
    "anger_after",           # float — quick access without calling /state
    "trust_after",           # float — quick access without calling /state
    "episode_id",            # str  — UUID, same as in Observation
]
# Person A: api/routes.py populates all of these in StepResponse.info
# Person C: ClientStepResult.from_http() reads all of these by name
# RULE: If A adds a new info key, A adds it here. C reads from this list.


# ────────────────────────────────────────────────────────────
# SECTION 6 — CURRICULUM STAGE THRESHOLDS
# Locked across Person A (filters profiles) and Person C (triggers advance)
# ────────────────────────────────────────────────────────────

CURRICULUM_STAGES: dict[int, dict] = {
    1: {
        "description"         : "calm, single demand, no hidden, threshold >= 8.5",
        "advance_on_reward"   : 0.30,   # Person C's scheduler advances when mean > this
        "profile_ids"         : ["P01", "P02", "P03", "P04", "P05"],
    },
    2: {
        "description"         : "mildly hostile, 2 demands, 1 hidden, threshold 7.2–8.4",
        "advance_on_reward"   : 0.50,
        "profile_ids"         : ["P06", "P07", "P08", "P09", "P10"],
    },
    3: {
        "description"         : "hostile, multiple hidden, capacity mismatch, threshold 5.5–6.9",
        "advance_on_reward"   : 0.65,
        "profile_ids"         : ["P11", "P12", "P13", "P14", "P15"],
    },
    4: {
        "description"         : "full complexity, adversarial, threshold <= 5.4",
        "advance_on_reward"   : None,   # final stage — no advance
        "profile_ids"         : ["P16", "P17", "P18", "P19", "P20"],
    },
}
# Person A: curriculum.py reads profile_ids to filter PROFILES list
# Person C: curriculum_scheduler.py reads advance_on_reward thresholds
# RULE: Profile IDs here must exactly match IDs in profiles.py


# ────────────────────────────────────────────────────────────
# SECTION 7 — NUMERIC RANGES (guard rails for all three persons)
# ────────────────────────────────────────────────────────────

NUMERIC_RANGES: dict[str, tuple] = {
    "anger"            : (0.0, 10.0),
    "trust"            : (0.0, 10.0),
    "fear"             : (0.0, 10.0),
    "escalation_level" : (0.0, 10.0),   # shown to LLM — mirrors anger
    "reward_per_step"  : (-1.0, 1.5),   # any single step reward outside this = bug
    "episode_total"    : (-5.0, 10.0),  # full episode score range
}
# Person A: adversary.py clamps anger/trust/fear to these ranges after every update
# Person B: rubric.py asserts no reward function returns outside reward_per_step
# Person C: wandb alerts if episode_total goes outside episode_total range


# ────────────────────────────────────────────────────────────
# SECTION 8 — WANDB COLUMN NAMING CONVENTION
# Locked for Person C — consistent across all training runs
# ────────────────────────────────────────────────────────────

WANDB_COLUMNS: dict[str, str] = {
    # Reward components — one per reward function
    **{f"reward/{k}": f"Per-step {k} reward" for k in REWARD_BREAKDOWN_KEYS},
    # Episode-level
    "reward/total"      : "Total scalar reward this step",
    "episode/success"   : "1 if commitment_reached else 0",
    "episode/reason"    : "Termination reason string",
    "episode/turns"     : "Number of turns taken",
    # State tracking
    "state/anger"       : "Anger level at end of step",
    "state/trust"       : "Trust level at end of step",
    # Training health
    "train/kl_div"      : "KL divergence from reference policy",
    "train/stage"       : "Current curriculum stage",
    "train/mean_reward_100" : "Rolling mean reward over last 100 steps",
}
# Person C: train_grpo.py calls wandb.log() with exactly these key names
# This ensures all wandb runs are comparable across experiments


# ────────────────────────────────────────────────────────────
# SECTION 9 — VALIDATION UTILITY (run this file directly to verify)
# python contracts.py
# ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=== Contract Validation ===\n")

    # Check reward keys are unique
    assert len(REWARD_BREAKDOWN_KEYS) == len(set(REWARD_BREAKDOWN_KEYS)), \
        "REWARD_BREAKDOWN_KEYS contains duplicates"

    # Check action types are unique
    assert len(ACTION_TYPES) == len(set(ACTION_TYPES)), \
        "ACTION_TYPES contains duplicates"

    # Check termination reasons cover success + all failure modes
    assert "commitment_reached" in TERMINATION_REASONS, \
        "Missing success termination reason"
    assert "threshold_breach" in TERMINATION_REASONS
    assert "timeout" in TERMINATION_REASONS
    assert "forbidden_move" in TERMINATION_REASONS

    # Check curriculum profile IDs don't overlap
    all_ids = []
    for stage, data in CURRICULUM_STAGES.items():
        all_ids.extend(data["profile_ids"])
    assert len(all_ids) == len(set(all_ids)), \
        "Duplicate profile IDs across curriculum stages"

    # Check numeric ranges are (min, max) tuples
    for k, v in NUMERIC_RANGES.items():
        assert v[0] < v[1], f"Invalid range for {k}: {v}"

    # Check wandb columns cover all reward breakdown keys
    for k in REWARD_BREAKDOWN_KEYS:
        assert f"reward/{k}" in WANDB_COLUMNS, \
            f"Missing wandb column for reward key: {k}"

    print(f"  Reward breakdown keys : {len(REWARD_BREAKDOWN_KEYS)} OK")
    print(f"  Action types          : {len(ACTION_TYPES)} OK")
    print(f"  Termination reasons   : {len(TERMINATION_REASONS)} OK")
    print(f"  Curriculum stages     : {len(CURRICULUM_STAGES)} OK")
    print(f"  Total profiles        : {sum(len(v['profile_ids']) for v in CURRICULUM_STAGES.values())} OK")
    print(f"  WandB columns         : {len(WANDB_COLUMNS)} OK")
    print(f"  Numeric range guards  : {len(NUMERIC_RANGES)} OK")
    print("\nAll contracts valid OK")
    print("\nLocked keys (share with team before Day 1):")
    print(f"  REWARD_BREAKDOWN_KEYS = {REWARD_BREAKDOWN_KEYS}")
    print(f"  ACTION_TYPES          = {ACTION_TYPES}")
    print(f"  STAGE thresholds      = { {k: v['advance_on_reward'] for k,v in CURRICULUM_STAGES.items()} }")