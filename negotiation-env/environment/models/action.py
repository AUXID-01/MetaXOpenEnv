from pydantic import BaseModel, validator
from typing import Optional, Dict, Any
from enum import Enum
 
class ActionType(str, Enum):
    SEND_MESSAGE        = "send_message"         # core conversational turn
    OFFER_EMI           = "offer_emi"            # structured EMI proposal
    ACKNOWLEDGE_HARDSHIP= "acknowledge_hardship" # explicit empathy signal
    ASK_OPEN_QUESTION   = "ask_open_question"    # needs assessment
    CONFIRM_IN_WRITING  = "confirm_in_writing"   # promise of written record
    STALL               = "stall"                # buy time explicitly
    ESCALATE_AUTHORITY  = "escalate_authority"   # invoke senior (risky)
 
class Action(BaseModel):
    """
    What the LLM sends to the environment each turn.
    Person C builds this. Person A consumes it in env.step().
    Person B reads action.text in compliance.py and anti_exploit.py.
    """
    action_type : ActionType
    text        : str                            # the actual message text
    metadata    : Optional[Dict[str, Any]] = {}  # e.g. {"emi_amount": 3000}
 
    @validator("text")
    def text_not_empty(cls, v):
        if not v or not v.strip():
            raise ValueError("Action text cannot be empty")
        return v.strip()
 
    @validator("metadata", pre=True, always=True)
    def metadata_default(cls, v):
        return v or {}
 
    class Config:
        use_enum_values = True   # serialises to string for JSON
 