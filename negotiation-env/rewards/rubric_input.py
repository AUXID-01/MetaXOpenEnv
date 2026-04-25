from pydantic import BaseModel
from typing import List
 
class RubricInput(BaseModel):
    """
    The single object Person A hands to Person B's rubric.compose().
    Contains everything any reward function could possibly need.
    Person B's reward functions only read from this — they never
    touch environment/ or adversary.py directly.
    """
    # ── State snapshots ──
    state_before     : State        # full state before action was applied
    state_after      : State        # full state after adversary reacted
 
    # ── The action that was taken ──
    action           : Action       # what the LLM said this turn
 
    # ── Episode status ──
    episode_done     : bool
    termination_reason: str | None  # None if not done
 
    # ── Message history ──
    message_history  : List[str]    # all agent messages so far (for anti-exploit)
 
    # ── Convenience deltas (pre-computed by Person A, saves Person B repeating) ──
    anger_delta      : float        # state_after.anger - state_before.anger
    trust_delta      : float        # state_after.trust - state_before.trust
 