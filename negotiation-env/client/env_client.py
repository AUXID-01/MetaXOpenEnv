import requests
import copy
import sys
import os

# Add project root to sys.path to allow absolute imports from contracts
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from contracts import REWARD_BREAKDOWN_SCHEMA, TERMINATION_REASONS, OBSERVATION_SCHEMA

DUMMY_EPISODE_ARC = [
    # index 0 — consumed by reset()
    {"turn": 1, "borrower_msg": "I already told your colleagues, I can't pay anything right now.", 
     "escalation_level": 3.0, "stated_demands": ["need 3 month moratorium"], "turns_remaining": 14, "reward": 0.1, "done": False},
    
    # index 1 — step 1, done=False
    {"turn": 2, "borrower_msg": "Why do you people keep calling? I lost my job in January.",
     "escalation_level": 5.5, "stated_demands": ["need 3 month moratorium", "reduce interest"], "turns_remaining": 13, "reward": -0.2, "done": False},
    
    # index 2 — step 2, done=False
    {"turn": 3, "borrower_msg": "My wife is unwell. If you can reduce the EMI I might manage something.",
     "escalation_level": 4.0, "stated_demands": ["need 3 month moratorium", "reduce interest", "lower emi"], "turns_remaining": 12, "reward": 0.3, "done": False},
    
    # index 3 — step 3, done=False
    {"turn": 4, "borrower_msg": "What options do you have for me?",
     "escalation_level": 3.0, "stated_demands": ["lower emi"], "turns_remaining": 11, "reward": 0.1, "done": False},
    
    # index 4 — step 4, commitment_reached
    {"turn": 5, "borrower_msg": "Okay. If you can make it 1800 per month for 6 months, I can try.",
     "escalation_level": 2.0, "stated_demands": ["lower emi to 1800", "6 month plan"], "turns_remaining": 10, "reward": 1.0, "done": True},
    
    # index 5 — timeout fallback
    {"turn": 6, "borrower_msg": "I need more time to think about this.",
     "escalation_level": 9.0, "stated_demands": [], "turns_remaining": 0, "reward": 0.0, "done": True},
]

class DummyEnvClient:
    """Mock client — cycles through pre-written episode arc, no server needed"""
    def __init__(self):
        self.step_idx = 0
        
    def reset(self, curriculum_stage: int = 1) -> dict:
        self.step_idx = 0
        arc_step = DUMMY_EPISODE_ARC[self.step_idx]
        obs = {k: v for k, v in arc_step.items() if k in OBSERVATION_SCHEMA}
        self.step_idx += 1
        return obs
        
    def step(self, action: dict) -> tuple:
        if self.step_idx >= len(DUMMY_EPISODE_ARC):
            # If they step past the end of the predefined dummy episode
            last = DUMMY_EPISODE_ARC[-1]
            obs = {k: v for k, v in last.items() if k in OBSERVATION_SCHEMA}
            info = {
                "reward_breakdown": copy.deepcopy(REWARD_BREAKDOWN_SCHEMA),
                "termination_reason": TERMINATION_REASONS[2],  # timeout
                "anger_after": last.get("escalation_level", 0.0),
                "trust_after": 5.0
            }
            return obs, 0.0, True, info
            
        arc_step = DUMMY_EPISODE_ARC[self.step_idx]
        obs = {k: v for k, v in arc_step.items() if k in OBSERVATION_SCHEMA}
        reward = arc_step["reward"]
        done = arc_step["done"]
        
        termination_reason = None
        if done:
            if self.step_idx == 4:
                termination_reason = TERMINATION_REASONS[2]  # timeout
            else:
                termination_reason = TERMINATION_REASONS[0]  # commitment_reached

        info = {
            "reward_breakdown": copy.deepcopy(REWARD_BREAKDOWN_SCHEMA),
            "termination_reason": termination_reason,
            "anger_after": arc_step.get("escalation_level", 0.0),
            "trust_after": 5.0
        }
        
        # Make the breakdown a bit dynamic for the dummy
        if done:
            info["reward_breakdown"]["outcome"] = 1.0
            
        self.step_idx += 1
        return obs, reward, done, info
        
    def state(self) -> dict:
        return {"status": "Dummy state — not backed by real adversary"}


class NegotiationEnvClient:
    """Real client — points at FastAPI server (local uvicorn or HF Space)"""
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")
        self.current_episode_id = None
        
    def reset(self, curriculum_stage: int = 1) -> dict:
        response = requests.post(f"{self.base_url}/reset", json={"curriculum_stage": curriculum_stage})
        response.raise_for_status()
        data = response.json()
        self.current_episode_id = data["episode_id"]
        return data["observation"]
        
    def step(self, action: dict) -> tuple:
        if not self.current_episode_id:
            raise ValueError("Must call reset() before step()")
            
        payload = {
            "episode_id": self.current_episode_id,
            "action_type": action.get("action_type", "send_message"),
            "text": action.get("text", ""),
            "metadata": action.get("metadata", {})
        }
        response = requests.post(f"{self.base_url}/step", json=payload)
        response.raise_for_status()
        data = response.json()
        return data["observation"], data["reward"], data["done"], data["info"]
        
    def state(self) -> dict:
        response = requests.get(f"{self.base_url}/state")
        response.raise_for_status()
        return response.json()["state"]
