"""Cross-team contracts used by environment, rewards, client, and training."""

# Reward component keys used in reward breakdown logging.
REWARD_BREAKDOWN_SCHEMA = {
    "outcome": 0.0,
    "deescalation": 0.0,
    "trust_building": 0.0,
    "demand_coverage": 0.0,
    "efficiency": 0.0,
    "compliance": 0.0,
    "anti_exploit": 0.0,
}
REWARD_BREAKDOWN_KEYS: list[str] = list(REWARD_BREAKDOWN_SCHEMA.keys())

# Canonical action registry across prompt parser, Action model, and API payloads.
ACTION_TYPES: list[str] = [
    "send_message",
    "offer_emi",
    "acknowledge_hardship",
    "ask_open_question",
    "confirm_in_writing",
    "stall",
    "escalate_authority",
]

# Canonical episode termination reasons.
TERMINATION_REASONS: list[str] = [
    "commitment_reached",
    "anger_threshold_crossed",
    "timeout",
    "forbidden_action",
]

OBSERVATION_SCHEMA = {
    "turn": 0,
    "borrower_msg": "",
    "escalation_level": 0.0,
    "stated_demands": [],
    "turns_remaining": 15,
    "profile_context": "",
    "episode_id": "",
}

STEP_RESPONSE_INFO_KEYS: list[str] = [
    "reward_breakdown",
    "termination_reason",
    "anger_after",
    "trust_after",
    "episode_id",
]

CURRICULUM_STAGES: dict[int, dict] = {
    1: {"description": "easy", "advance_on_reward": 0.30, "profile_ids": ["P01", "P02", "P03", "P04", "P05"]},
    2: {"description": "medium", "advance_on_reward": 0.50, "profile_ids": ["P06", "P07", "P08", "P09", "P10"]},
    3: {"description": "hard", "advance_on_reward": 0.65, "profile_ids": ["P11", "P12", "P13", "P14", "P15"]},
    4: {"description": "full", "advance_on_reward": None, "profile_ids": ["P16", "P17", "P18", "P19", "P20"]},
}

NUMERIC_RANGES: dict[str, tuple] = {
    "anger": (0.0, 10.0),
    "trust": (0.0, 10.0),
    "fear": (0.0, 10.0),
    "escalation_level": (0.0, 10.0),
    "reward_per_step": (-1.0, 1.5),
    "episode_total": (-5.0, 10.0),
}

WANDB_COLUMNS: dict[str, str] = {
    **{f"reward/{k}": f"Per-step {k}" for k in REWARD_BREAKDOWN_KEYS},
    "reward/total": "Total scalar reward",
    "episode/success": "1 if resolved else 0",
    "episode/reason": "Termination reason",
    "episode/turns": "Turns taken",
    "state/anger": "Anger at end of turn",
    "state/trust": "Trust at end of turn",
    "train/kl_div": "KL divergence",
    "train/stage": "Curriculum stage",
    "train/mean_reward_100": "Rolling mean reward",
}


if __name__ == "__main__":
    assert len(REWARD_BREAKDOWN_KEYS) == len(set(REWARD_BREAKDOWN_KEYS))
    assert len(ACTION_TYPES) == len(set(ACTION_TYPES))
    assert "commitment_reached" in TERMINATION_REASONS
    assert "anger_threshold_crossed" in TERMINATION_REASONS
    assert "timeout" in TERMINATION_REASONS
    assert "forbidden_action" in TERMINATION_REASONS
    for name, rng in NUMERIC_RANGES.items():
        assert rng[0] < rng[1], f"Invalid range for {name}"
    for key in REWARD_BREAKDOWN_KEYS:
        assert f"reward/{key}" in WANDB_COLUMNS
    print("All contracts valid OK")