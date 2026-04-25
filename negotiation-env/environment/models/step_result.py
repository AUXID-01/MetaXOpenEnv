from pydantic import BaseModel
from typing import Dict, Any
 
class StepResult(BaseModel):
    """
    Everything returned by env.step(action).
    This is the handoff object between Person A and Person C.
    Person B's reward breakdown lives inside info["reward_breakdown"].
    """
    observation      : Observation  # next obs — fed back to LLM as next prompt
    reward           : float        # total scalar reward for this step
    done             : bool         # whether episode ended
    info             : Dict[str, Any]
    # info keys (all logged to wandb by Person C):
    #   "reward_breakdown": {
    #       "outcome"         : float,
    #       "deescalation"    : float,
    #       "trust_building"  : float,
    #       "demand_coverage" : float,
    #       "efficiency"      : float,
    #       "compliance"      : float,
    #       "anti_exploit"    : float,
    #   }
    #   "termination_reason"  : str | None
    #   "state_snapshot"      : dict   # serialised State for logging
    #   "anger_after"         : float  # for quick wandb column
    #   "trust_after"         : float  # for quick wandb column
    #   "episode_id"          : str
 