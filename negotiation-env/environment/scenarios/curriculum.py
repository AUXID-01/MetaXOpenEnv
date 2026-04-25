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
    # Threshold logic mentioned in project_structure.md
    pass
