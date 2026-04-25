# environment/adversary.py
import math
from typing import Tuple, Dict, Any, List, Optional
from .classifier import classify_action
# pick_template is kept importable for legacy callers and tests; the live
# voice path goes through the ResponseGenerator singleton instead.
from .response_generator import (
    pick_template,                   # noqa: F401  (re-export / fallback)
    get_response_generator,
)


def _safe_float(value: Any, default: float = 0.0) -> float:
    """
    BOUNDARY GUARD primitive used inside _update_state().

    Coerce `value` to a finite float.  NaN, +/-Inf, None, and unparseable
    types all collapse to `default`.  A pathological multiplier (say a
    rogue `float('inf')` slipping into a personality table) cannot
    produce a non-finite anger / trust / fear value once this filter is
    applied to every read and write inside the state machine.

    NB: this DOES NOT clamp to [0, 10].  Range clamping happens after the
    delta is added so we don't lose the sign of the update.  Only finiteness
    is enforced here.
    """
    try:
        v = float(value)
    except (TypeError, ValueError):
        return float(default)
    if not math.isfinite(v):
        return float(default)
    return v


def _clamp_emotion(value: Any, prev: float = 0.0) -> float:
    """
    Force `value` to a finite float in [0.0, 10.0].  If `value` is non-finite
    we fall back to the previous (already-finite) value for that emotion so
    a single bad delta doesn't reset the whole episode's mood.
    """
    v = _safe_float(value, default=prev)
    if v < 0.0:
        return 0.0
    if v > 10.0:
        return 10.0
    return v

"""
HOLDS: BorrowerAdversary class.
RUNS: called every step inside env.py.
CONNECTS TO: classifier.py, response_generator.py (ResponseGenerator
             singleton — Hybrid-Dynamic voice over NVIDIA NIM /
             Llama-3.1-70B), env.py.
"""

PERSONALITY_MODIFIERS = {
    "default": {
        "empathy_anger_scale": 1.0,
        "empathy_trust_scale": 1.0,
        "threat_anger_scale": 1.0,
        "threat_trust_scale": 1.0,
        "home_visit_fear_scale": 1.0,
        "payment_offer_trust_scale": 1.0,
        "probe_trust_scale": 1.0,
    },
    "defensive_and_emotionally_raw": {
        "empathy_anger_scale": 0.5,       # hard to de-escalate
        "empathy_trust_scale": 1.5,       # but trust blooms if empathy is felt
        "threat_anger_scale": 2.0,        # very sensitive to threats
        "home_visit_fear_scale": 1.5,
    },
    "frightened_and_secretive": {
        "empathy_anger_scale": 1.2,
        "empathy_trust_scale": 1.8,
        "threat_anger_scale": 0.5,        # shuts down rather than getting angry
        "home_visit_fear_scale": 3.0,     # extremely panicked by visits
        "threat_trust_scale": 2.0,        # threats destroy trust completely
    },
    "cooperative_and_responsible": {
        "empathy_anger_scale": 1.5,
        "payment_offer_trust_scale": 2.0, # rewards offers highly
        "threat_anger_scale": 1.2,
    },
    "legally_aware_and_outraged": {
        "threat_anger_scale": 1.1,
        "threat_trust_scale": 2.5,        # legal threats trigger massive distrust
        "empathy_trust_scale": 0.8,       # suspicious of "fake" empathy
    },
    "nihilistic_and_calculating": {
        "empathy_trust_scale": 0.5,       # very cynical
        "empathy_anger_scale": 0.3,
        "threat_anger_scale": 0.8,        # doesn't care much about threats
        "payment_offer_trust_scale": 0.7,
    },
    "resigned_but_cooperative": {
        "empathy_anger_scale": 1.2,
        "payment_offer_trust_scale": 1.5,
    },
    "honest_and_anxious": {
        "empathy_trust_scale": 1.4,
        "home_visit_fear_scale": 1.8,
    },
    "traditional_respectful": {
        "empathy_trust_scale": 1.5,
        "threat_anger_scale": 1.3,        # feels insulted by threats
    },
    "pragmatic_and_direct": {
        "empathy_trust_scale": 0.9,
        "payment_offer_trust_scale": 1.4,
    },
    "guarded_and_proud": {
        "empathy_trust_scale": 0.7,
        "threat_anger_scale": 1.5,
    },
    "frustrated_but_articulate": {
        "empathy_anger_scale": 1.1,
        "threat_anger_scale": 1.4,
    },
    "dignified_and_ashamed": {
        "empathy_trust_scale": 1.6,
        "home_visit_fear_scale": 2.5,
    },
    "angry_and_victimised": {
        "threat_anger_scale": 1.8,
        "empathy_anger_scale": 0.8,
    },
    "street_smart_and_suspicious": {
        "empathy_trust_scale": 0.6,
        "payment_offer_trust_scale": 1.5,
    },
    "experienced_negotiator_in_control": {
        "empathy_trust_scale": 0.8,
        "threat_anger_scale": 0.9,
        "payment_offer_trust_scale": 1.2,
    },
    "young_and_explosive": {
        "threat_anger_scale": 2.5,
        "empathy_anger_scale": 1.2,
    },
    "calm_strategic_and_legally_informed": {
        "empathy_trust_scale": 1.0,
        "threat_trust_scale": 2.0,
    },
    "collectively_influenced_but_privately_reachable": {
        "empathy_trust_scale": 1.5,
        "threat_anger_scale": 1.2,
    }
}

class BorrowerAdversary:
    """
    Simulates a defaulting borrower in a debt-collection scenario.
    Manages emotional state machine and responds to bank agent actions.
    """

    def __init__(self, profile: Dict[str, Any]):
        self.profile = profile
        self.anger = float(profile["anger_init"])
        self.trust = float(profile["trust_init"])
        self.fear = float(profile["fear_init"])
        self.anger_threshold = float(profile["anger_threshold"])
        
        self.real_emi = float(profile["real_emi"])
        self.stated_capacity = float(profile["stated_capacity"])
        
        self.demands = list(profile["demands"])
        self.hidden_demands = list(profile["hidden_demands"])
        self.revealed_demands: List[str] = []
        
        # Determine internal personality modifiers
        p_name = profile["personality"]
        self.mods = PERSONALITY_MODIFIERS.get(p_name, PERSONALITY_MODIFIERS["default"])
        
        self.terminated = False
        self.termination_reason = ""

        # Cache of the last agent message — passed to ResponseGenerator so
        # the LLM can ground its reply in what the agent actually said.
        # Reward computation does NOT read this; it stays a pure adversary
        # internal so the deterministic state machine remains the only
        # source of truth for state transitions.
        self._last_agent_text: str = ""

    def opening_turn(self) -> str:
        """Returns the initial message from the borrower."""
        return self.profile["opening_msg"]

    def react(self, llm_action_text: str, signals: Optional[Dict[str, Any]] = None) -> Tuple[str, bool, str]:
        """
        Updates borrower state based on agent action and returns (reply, is_done, reason).
        """
        if self.terminated:
            return "Episode already ended.", True, self.termination_reason

        # Remember the agent's text so the dynamic voice can quote/ground in it.
        self._last_agent_text = llm_action_text or ""

        # 1. Get signals if not provided
        if signals is None:
            signals = classify_action(llm_action_text)

        # 2. Update emotional state
        self._update_state(signals)

        # 3. Handle hidden demand revelation
        reveal_msg = self._handle_demand_revelation()

        # 4. Check termination conditions
        self._check_termination()

        # 5. Generate response
        response_text = self._generate_response(signals)
        if reveal_msg:
            response_text = f"{reveal_msg}\n{response_text}"

        return response_text, self.terminated, self.termination_reason

    def _update_state(self, signals: Dict[str, Any]):
        """Applies signal deltas with personality modifiers.

        BOUNDARY GUARD: every delta and every running multiplier is run
        through `_safe_float()` so a NaN / Inf can't leak into the state.
        After all the arithmetic, `_clamp_emotion()` enforces the [0, 10]
        range and finiteness — so even if a personality entry contained a
        bad value, the episode would silently continue with a sane mood.
        """
        # Base deltas from classifier — sanitised at the boundary.
        a_delta = _safe_float(signals.get("anger_delta", 0.0))
        t_delta = _safe_float(signals.get("trust_delta", 0.0))
        f_delta = _safe_float(signals.get("fear_delta", 0.0))

        # Apply personality scaling (each modifier is also sanitised so a
        # rogue Inf in PERSONALITY_MODIFIERS can't propagate).
        if a_delta < 0: # De-escalation (empathy/ack)
            a_delta *= _safe_float(self.mods.get("empathy_anger_scale", 1.0), 1.0)
        else: # Escalation (threat/pressure)
            a_delta *= _safe_float(self.mods.get("threat_anger_scale", 1.0), 1.0)

        if t_delta > 0:
            if signals["meta"]["has_empathy"]:
                t_delta *= _safe_float(self.mods.get("empathy_trust_scale", 1.0), 1.0)
            if signals["extracted_offer"]["amount"] or signals["extracted_offer"]["emi"]:
                t_delta *= _safe_float(self.mods.get("payment_offer_trust_scale", 1.0), 1.0)
            if signals["meta"]["has_open_question"]:
                t_delta *= _safe_float(self.mods.get("probe_trust_scale", 1.0), 1.0)
        else:
            t_delta *= _safe_float(self.mods.get("threat_trust_scale", 1.0), 1.0)

        if f_delta > 0:
            if "physical_visit" in signals["risk_flags"]:
                f_delta *= _safe_float(self.mods.get("home_visit_fear_scale", 1.0), 1.0)

        # Re-sanitise after the multipliers in case a 0 × Inf produced NaN.
        a_delta = _safe_float(a_delta)
        t_delta = _safe_float(t_delta)
        f_delta = _safe_float(f_delta)

        # Context-aware adjustments
        if signals["primary_action_type"] == "warn_noncompliance" and signals["compliance_flags"]:
            a_delta += 1.5  # Extra penalty for actual policy-violating threats
        
        if signals["primary_action_type"] == "offer_plan" and (signals["extracted_offer"]["emi"] or signals["extracted_offer"]["amount"]):
            t_delta += 0.2  # Bonus for structured offers

        # Apply to state — `_clamp_emotion` enforces both finiteness and
        # the [0, 10] interval, falling back to the previous (good) value
        # if the new one is somehow non-finite.
        self.anger = _clamp_emotion(self.anger + a_delta, prev=self.anger)
        self.trust = _clamp_emotion(self.trust + t_delta, prev=self.trust)
        self.fear  = _clamp_emotion(self.fear  + f_delta, prev=self.fear)

    def _handle_demand_revelation(self) -> Optional[str]:
        """Checks if hidden demands should be revealed based on trust level."""
        if self.trust >= 5.0 and self.hidden_demands:
            demand = self.hidden_demands.pop(0)
            self.revealed_demands.append(demand)
            if demand not in self.demands:
                self.demands.append(demand)
            
            # Vulnerability signal
            p_name = self.profile.get("name", "The borrower")
            return f"(Trust gained: {p_name} is finally sharing a hidden concern: '{demand}')"
        return None

    def _check_termination(self):
        """Logic for escalation failure or commitment success."""
        # FAILURE: Escalation
        if self.anger >= self.anger_threshold:
            self.terminated = True
            self.termination_reason = "anger_threshold_crossed"
            return

        # SUCCESS: Commitment
        # Condition: trust high, anger low, and some level of agreement (placeholder for demand satisfaction)
        if self.trust >= 7.0 and self.anger <= 3.0:
            # We assume if the model is here, it resolved enough concerns
            self.terminated = True
            self.termination_reason = "commitment_reached"
            return
            
        # Time-out/Turns handled at the env.py level

    def _generate_response(self, signals: Dict[str, Any]) -> str:
        """
        Generate the borrower's natural-language reply for the current turn.

        Routes through the Hybrid-Dynamic voice (ResponseGenerator singleton):
          • If NEG_LLM_ENABLED=1 and NVIDIA_API_KEY is set, NIM /
            Llama-3.1-70B-Instruct produces a reply that reflects the
            current Archetype (a function of self.anger / self.trust /
            self.fear) and the agent's last message.  Replies are cached
            by (Archetype, action_type) so the same emotional state +
            agent intent re-uses an already-generated line, keeping
            training-loop latency bounded.
          • Otherwise (default for hermetic test runs and offline / CI), the
            singleton transparently falls back to the deterministic
            template bank via pick_template().

        IMPORTANT: this function does NOT mutate self.* — it only reads
        the state that _update_state has already committed.  The reward
        signal is therefore independent of which voice path was used.
        """
        action_type = signals.get("primary_action_type", "unknown")

        return get_response_generator().generate(
            anger=self.anger,
            trust=self.trust,
            fear=self.fear,
            action_type=action_type,
            agent_text=self._last_agent_text,
            profile=self.profile,
            terminated=self.terminated,
            termination_reason=self.termination_reason,
        )
