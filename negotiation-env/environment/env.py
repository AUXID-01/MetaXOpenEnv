# environment/env.py
import random
import uuid
from typing import Dict, Any, Tuple, List
from environment.adversary import BorrowerAdversary
from environment.classifier import classify_action
from environment.scenarios.profiles import PROFILES
from environment.scenarios.curriculum import get_profiles_for_stage
from environment.models.action import Action
from environment.models.observation import Observation
from environment.models.state import State
from environment import config
from rewards.rubric import Rubric

"""
HOLDS: NegotiationEnv class.
RUNS: called by api/routes.py on every HTTP request.
CONNECTS TO: adversary.py, classifier.py, scenarios/, rewards/rubric.py.
"""

class NegotiationEnv:
    """
    OpenEnv-compliant negotiation environment.
    reset() → picks scenario, inits adversary, returns first observation.
    step(action) → passes to adversary, gets response, computes reward, checks termination.
    """
    
    def __init__(self, seed: int | None = None):
        """
        seed: optional random seed for reproducibility (used only in reset profile selection).
        """
        self.seed = seed
        if seed is not None:
            random.seed(seed)
        
        self.rubric = Rubric()
        
        # Hidden state (not sent to LLM)
        self._state: State | None = None
        self._profile: dict | None = None
        self._adversary: BorrowerAdversary | None = None
        self._turn: int = 0
        self._terminated: bool = False
        self._termination_reason: str = ""
        self._episode_history: List[str] = []  # Full dialogue history
        self._agent_history: List[str] = []     # For anti-exploit check
        self._episode_id: str = ""
    
    def reset(self, curriculum_stage: int | None = None, stage: int | None = None) -> Observation:
        """
        Start a new episode.
        stage: curriculum stage (filters profiles by difficulty).
        Returns first observation for the LLM.
        """
        # Choose profile
        if curriculum_stage is None and stage is not None:
            curriculum_stage = stage
        if curriculum_stage is None:
            curriculum_stage = config.CURRICULUM_STAGE
        
        eligible_profiles = get_profiles_for_stage(curriculum_stage)
        if not eligible_profiles:
            eligible_profiles = PROFILES  # fallback
        
        self._profile = random.choice(eligible_profiles)
        self._episode_id = str(uuid.uuid4())
        
        # Init adversary and reward rubric
        self._adversary = BorrowerAdversary(self._profile)
        self.rubric = Rubric(curriculum_stage=curriculum_stage)
        
        # Reset tracker values
        self._turn = 0
        self._terminated = False
        self._termination_reason = ""
        self._episode_history = []
        self._agent_history = []
        
        # Build hidden state (matching environment.models.state.State)
        self._state = State(
            anger=self._adversary.anger,
            trust=self._adversary.trust,
            fear=self._adversary.fear,
            real_emi_capacity=self._adversary.real_emi,
            stated_capacity=self._adversary.stated_capacity,
            loan_amount=float(self._profile.get("loan_amount", 0)),
            demands_total=list(self._profile.get("demands_total", self._profile.get("demands", []))),
            demands_stated=list(self._profile.get("demands_stated", [])),
            demands_hidden=list(self._profile.get("demands_hidden", self._profile.get("hidden_demands", []))),
            demands_addressed=[],
            anger_threshold=self._adversary.anger_threshold,
            turn=self._turn,
            max_turns=config.MAX_TURNS,
            terminated=False,
            termination_reason=None,
            message_history=[],
            profile_id=self._profile.get("id", "unknown"),
            curriculum_stage=curriculum_stage,
            episode_id=self._episode_id
        )
        
        # Get opening message from adversary
        opening_msg = self._adversary.opening_turn()
        self._episode_history.append(opening_msg)
        
        # Build observation (what LLM sees, matching environment.models.observation.Observation)
        obs = Observation(
            turn=self._turn,
            borrower_msg=opening_msg,
            escalation_level=self._adversary.anger,
            stated_demands=self._state.demands_stated,
            turns_remaining=config.MAX_TURNS - self._turn,
            profile_context=f"{self._profile.get('name')}, {self._profile.get('reason')}, {self._profile.get('overdue_days')} days overdue",
            episode_id=self._episode_id
        )
        return obs
    
    def step(self, action_dict: Dict[str, Any]) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        """
        Process one turn of negotiation.
        action_dict: {"action_type": str, "text": str, "metadata": dict}
        Returns: (observation_dict, reward, done, info)
        """
        if self._terminated:
            raise RuntimeError("Episode already terminated. Call reset().")
        
        self._turn += 1
        action_text = action_dict.get("text", "").strip()
        action_type = action_dict.get("action_type", "send_message")
        action_metadata = action_dict.get("metadata", {})
        if not action_text:
            action_text = "..."
        if action_metadata is None:
            action_metadata = {}
        action_dict = {
            "action_type": action_type,
            "text": action_text,
            "metadata": action_metadata,
        }
        
        # Step 1: Classify the agent's action
        signals = classify_action(action_text)
        
        # Step 2: Adversary reacts
        response_text, terminated, term_reason = self._adversary.react(action_text, signals)
        
        # Step 3: Update episode history (for anti-exploit)
        self._agent_history.append(action_text)
        self._episode_history.append(action_text) # add agent msg
        self._episode_history.append(response_text) # add borrower reply
        
        # Step 4: Check environment-level termination
        if terminated:
            self._terminated = True
            self._termination_reason = term_reason
        elif self._turn >= config.MAX_TURNS:
            self._terminated = True
            self._termination_reason = "timeout"
        
        # Step 5: Update state and compute reward
        state_before = self._state
        
        # Convert input dict to Action model for reward logic
        action_model = Action(**action_dict)
        
        state_after = State(
            anger=self._adversary.anger,
            trust=self._adversary.trust,
            fear=self._adversary.fear,
            real_emi_capacity=self._adversary.real_emi,
            stated_capacity=self._adversary.stated_capacity,
            loan_amount=state_before.loan_amount,
            demands_total=state_before.demands_total,
            demands_stated=self._adversary.revealed_demands, # Updated list from adversary
            demands_hidden=self._adversary.hidden_demands,
            demands_addressed=state_before.demands_addressed, # Would be updated by rubric/logic later
            anger_threshold=self._adversary.anger_threshold,
            turn=self._turn,
            max_turns=config.MAX_TURNS,
            terminated=self._terminated,
            termination_reason=self._termination_reason,
            message_history=self._agent_history[-6:], # only agent msgs for anti-exploit
            profile_id=state_before.profile_id,
            curriculum_stage=state_before.curriculum_stage,
            episode_id=self._episode_id
        )
        self._state = state_after
        
        total_reward, breakdown = self.rubric.compose(
            state_before=state_before,
            state_after=state_after,
            action=action_model,
            episode_done=self._terminated
        )
        
        # Step 6: Build next observation
        obs = Observation(
            turn=self._turn,
            borrower_msg=response_text,
            escalation_level=self._adversary.anger,
            stated_demands=state_after.demands_stated,
            turns_remaining=config.MAX_TURNS - self._turn,
            profile_context=f"{self._profile.get('name')}, {self._profile.get('reason')}, {self._profile.get('loan_type')}",
            episode_id=self._episode_id
        )
        
        # Step 7: Build info dict (for logging)
        info = {
            "reward_breakdown": breakdown,
            "anger_after": round(self._adversary.anger, 2),
            "trust_after": round(self._adversary.trust, 2),
            "fear_after": round(self._adversary.fear, 2),
            # Backward-compatible aliases used in legacy tests.
            "anger": round(self._adversary.anger, 2),
            "trust": round(self._adversary.trust, 2),
            "terminated": self._terminated,
            "termination_reason": self._termination_reason,
            "turn": self._turn,
            "profile_id": self._profile.get("id", "unknown"),
            "episode_id": self._episode_id,
            "action_type": action_type,
            "signals": signals
        }
        
        return obs.model_dump(), total_reward, self._terminated, info
    
    def state(self) -> Dict[str, Any]:
        """
        Return full hidden state (for logging/inspection only).
        Never sent to LLM.
        """
        if self._state is None:
            raise RuntimeError("Call reset() before state().")
        
        return self._state.model_dump()
