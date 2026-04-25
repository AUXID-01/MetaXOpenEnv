# environment/scenarios/curriculum.py
from .profiles import PROFILES

"""
HOLDS: get_profiles_for_stage(stage) filter function.
RUNS: called by env.reset() and training/curriculum_scheduler.py.
CONNECTS TO: profiles.py, env.py, training/curriculum_scheduler.py.
"""

def get_profiles_for_stage(stage: int) -> list:
    """
    Returns all profiles that match the current curriculum stage.
    """
    return [p for p in PROFILES if p["curriculum_stage"] <= stage]

def advance_stage(current_stage: int, mean_reward: float) -> int:
    """
    Logic to decide if the curriculum should advance.
    """
    thresholds = {
        1: 0.30,
        2: 0.50,
        3: 0.65,
        4: None,
    }
    threshold = thresholds.get(current_stage)
    if threshold is None:
        return current_stage
    if mean_reward >= threshold and current_stage < 4:
        return current_stage + 1
    return current_stage
