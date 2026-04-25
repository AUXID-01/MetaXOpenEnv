from pydantic import BaseModel
from typing import List, Optional
 
class State(BaseModel):
    """
    Full hidden state of the environment. NEVER sent to the LLM.
    Person A writes this. Person B reads it in every reward function.
    Passed as state_before and state_after to rubric.compose().
    """
    # ── Adversary emotional floats (0.0–10.0) ──
    anger            : float        # drives failure if > anger_threshold
    trust            : float        # drives success when high
    fear             : float        # affects response tone (not terminal)
 
    # ── Financial hidden state ──
    real_emi_capacity: float        # what borrower CAN actually pay (₹/month)
    stated_capacity  : float        # what borrower CLAIMS they can pay
    loan_amount      : float        # original loan amount
 
    # ── Demands tracking ──
    demands_total    : List[str]    # all demands (stated + hidden combined)
    demands_stated   : List[str]    # demands borrower has voiced so far
    demands_hidden   : List[str]    # demands not yet revealed
    demands_addressed: List[str]    # demands agent has responded to
 
    # ── Termination tracking ──
    anger_threshold  : float        # profile-specific ceiling
    turn             : int          # current turn
    max_turns        : int          # episode length limit
    terminated       : bool         # whether episode is over
    termination_reason: Optional[str] = None
                                    # "commitment_reached" | "anger_threshold_crossed"
                                    # "timeout" | "forbidden_action"
 
    # ── History for anti-exploit ──
    message_history  : List[str]    # last N agent messages (for cosine sim check)
 
    # ── Profile metadata ──
    profile_id       : str          # e.g. "P01"
    curriculum_stage : int          # 1–4
    episode_id       : str          # matches Observation.episode_id
 
 