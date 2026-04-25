from pydantic import BaseModel
from typing import Dict, Any, Optional
 
class ResetRequest(BaseModel):
    """POST /reset — optional overrides from trainer"""
    curriculum_stage : int = 1          # trainer tells env which stage to sample
    seed             : Optional[int] = None  # for reproducible eval episodes
 
class ResetResponse(BaseModel):
    """Response from POST /reset"""
    observation      : dict             # serialised Observation
    episode_id       : str
    profile_id       : str              # for logging — which borrower was picked
    curriculum_stage : int
 
class StepRequest(BaseModel):
    """POST /step — what Person C's client sends"""
    episode_id       : str              # must match active episode
    action_type      : str              # ActionType value
    text             : str              # LLM's message
    metadata         : Dict[str, Any] = {}
 
class StepResponse(BaseModel):
    """Response from POST /step — everything Person C needs"""
    observation      : dict             # serialised Observation (next turn)
    reward           : float            # total scalar
    done             : bool
    info             : Dict[str, Any]   # reward_breakdown + state_snapshot + metadata
 
class StateResponse(BaseModel):
    """GET /state — full hidden state for logging (eval scripts only)"""
    state            : dict             # serialised State
    episode_id       : str
 
 