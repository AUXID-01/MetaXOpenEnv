from pydantic import BaseModel
from typing import List
 
class Observation(BaseModel):
    """
    What the LLM sees each turn. Built by env.py, returned via HTTP,
    consumed by client/utils.py to build the LLM prompt string.
    THIS IS THE ONLY THING THE LLM EVER SEES.
    The hidden state (anger, trust, real_emi) is never in here.
    """
    turn             : int          # current turn number (0-indexed)
    borrower_msg     : str          # adversary's last natural-language reply
    escalation_level : float        # visible anger proxy — shown to LLM (0–10)
    stated_demands   : List[str]    # demands borrower has explicitly voiced
    turns_remaining  : int          # countdown — creates time pressure
    profile_context  : str          # one-line borrower context for LLM
                                    # e.g. "Ramesh Kumar, job loss, 60 days overdue"
                                    # does NOT include hidden state
    episode_id       : str          # UUID — used for logging and wandb tracking
 
 