"""
negotiation-env/reward.py
=========================
Reward module for the MetaX negotiation RL environment.

Stack : OpenEnv (FastAPI) → TRL GRPOTrainer → Unsloth
Agent : LLM negotiator (Qwen 1.5B / 3B)
Adversary : Deterministic state machine

Field-name mapping (PS1 spec → actual State model)
────────────────────────────────────────────────────
PS1 spec field          State field              Treatment
──────────────────────────────────────────────────────────
anger                   anger                    direct
trust_accumulator       trust                    direct
fear                    fear                     direct
hope                    —                        default 5.0 (not in State)
distrust                —                        default 5.0 (not in State)
escalation_threshold    anger_threshold          direct
stated_demands          demands_stated           direct
hidden_demands          demands_hidden           direct
revealed_demands        —                        default [] (not in State;
                                                 demands_hidden already
                                                 tracks what is hidden, so
                                                 the complement of
                                                 demands_hidden relative to
                                                 demands_total is what was
                                                 revealed — computed inline)
turn_number             turn                     direct
max_turns               max_turns                direct
resolved                terminated +             True iff terminated and
                        termination_reason       reason == "commitment_reached"
escalated               terminated +             True iff terminated and
                        termination_reason       reason == "anger_threshold_crossed"
addressed_demands       demands_addressed        direct (field on State)
message_history         message_history          direct (field on State)

Action mapping (PS1 spec VALID_ACTIONS → contracts.ACTION_TYPES)
────────────────────────────────────────────────────────────────
The locked action registry in contracts.py is the source of truth.
PS1 spec / brief used generic names; the real env uses the names below.

contracts.py / compliance function key names
──────────────────────────────────────────────
"outcome"         → outcome.py       (PS1 row 1)
"deescalation"    → deescalation.py  (PS1 row 2)
"trust_building"  → trust_building.py(PS1 row 3 + 4 — trust + emotional combined)
"demand_coverage" → demand_coverage.py(PS1 row 5)
"efficiency"      → efficiency.py    (PS1 row 6)
"compliance"      → compliance.py    (PS1 row 7 — format / RBI language)
"anti_exploit"    → anti_exploit.py  (PS1 row 8)

All seven REWARD_BREAKDOWN_KEYS are populated by compose().
Person A calls compose(rubric_input) and puts the returned breakdown dict
into StepResult.info["reward_breakdown"].  Person C reads it from there.

Design rules
────────────
• Every function is independently callable — no shared mutable state.
• Every function has a docstring citing which State fields it reads
  and which PS1 reward-spec row it implements.
• compute_reward / compose() returns (scalar, breakdown) — the breakdown
  dict satisfies REWARD_BREAKDOWN_KEYS for per-column wandb monitoring.
• All checks are deterministic and programmatic — zero LLM-as-judge.
• anti_exploit weight is 1.0 (it is already a small number — not downscaled).
"""

from __future__ import annotations

import sys
import os
from typing import Any

import numpy as np
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    _HAS_SKLEARN = True
except Exception:
    TfidfVectorizer = None
    cosine_similarity = None
    _HAS_SKLEARN = False

# ---------------------------------------------------------------------------
# Project-relative imports — reward.py lives at negotiation-env/reward.py
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from environment.models.state import State
from environment.models.action import Action, ActionType
from contracts import (
    ACTION_TYPES,
    REWARD_BREAKDOWN_KEYS,
    TERMINATION_REASONS,
)

# ---------------------------------------------------------------------------
# Locked action registry — sourced from contracts.ACTION_TYPES
# (PS1 spec called these VALID_ACTIONS; the real env uses ActionType enum)
# ---------------------------------------------------------------------------

VALID_ACTIONS: set[str] = set(ACTION_TYPES)

# ---------------------------------------------------------------------------
# Curriculum weight presets
# Key names must match REWARD_BREAKDOWN_KEYS from contracts.py exactly:
#   "outcome", "deescalation", "trust_building", "demand_coverage",
#   "efficiency", "compliance", "anti_exploit"
# ---------------------------------------------------------------------------

CURRICULUM_WEIGHTS: dict[str, dict[str, float]] = {
    "stage_1": {
        # Outcome + de-escalation only — reward landscape stays simple.
        # compliance on so the model learns not to make malformed actions.
        "outcome": 1.0,
        "deescalation": 0.3,
        "trust_building": 0.0,
        "demand_coverage": 0.0,
        "efficiency": 0.0,
        "compliance": 0.1,
        "anti_exploit": 0.0,
    },
    "stage_2": {
        # Add trust-building + demand coverage.
        "outcome": 1.0,
        "deescalation": 0.3,
        "trust_building": 0.3,
        "demand_coverage": 0.2,
        "efficiency": 0.0,
        "compliance": 0.1,
        "anti_exploit": 0.0,
    },
    "stage_3": {
        # Full stack — anti-exploit on.
        "outcome": 1.0,
        "deescalation": 0.3,
        "trust_building": 0.3,
        "demand_coverage": 0.2,
        "efficiency": 0.1,
        "compliance": 0.1,
        "anti_exploit": 1.0,
    },
    "stage_4": {
        # Same as stage_3 — tune from here.
        "outcome": 1.0,
        "deescalation": 0.3,
        "trust_building": 0.3,
        "demand_coverage": 0.2,
        "efficiency": 0.1,
        "compliance": 0.1,
        "anti_exploit": 1.0,
    },
}

_DEFAULT_WEIGHTS: dict[str, float] = CURRICULUM_WEIGHTS["stage_3"]

# ---------------------------------------------------------------------------
# Required extra keys / validators for each action type
# Mirrors the Action model's @validators and Person C's parser contract.
# ---------------------------------------------------------------------------

#   action_type          required metadata keys (besides .text which is always needed)
_ACTION_METADATA_KEYS: dict[str, list[str]] = {
    "send_message": [],
    "offer_emi": ["emi_amount"],       # metadata.emi_amount must be present
    "acknowledge_hardship": [],
    "ask_open_question": [],
    "confirm_in_writing": [],
    "stall": [],
    "escalate_authority": [],
}

# ---------------------------------------------------------------------------
# De-escalation filler phrases — checked by reward_anti_exploit
#
# FIX [V-FILLER-OVERLAP]: Phrases are ordered longest-first and matched
# exclusively (each position in the text is credited to at most one phrase).
# This prevents "I understand your concerns" from also triggering the
# shorter "I understand" substring and double-counting one sentence as two hits.
# ---------------------------------------------------------------------------

# Ordered longest-first so that exclusive matching works correctly.
_FILLER_PHRASES: list[str] = [
    "I understand your concerns",   # must come before "I understand"
    "let's work together",
    "I hear you",
    "we can find a solution",
    "I understand",                 # shorter — only matched if the longer form absent
]


# ===========================================================================
# Helper — derive resolved / escalated booleans from the State model
# ===========================================================================

def _is_resolved(state: State) -> bool:
    """
    Maps State.terminated + State.termination_reason → resolved bool.
    resolved = True iff terminated AND reason == "commitment_reached".
    """
    return bool(
        state.terminated
        and state.termination_reason == "commitment_reached"
    )


def _is_escalated(state: State) -> bool:
    """
    Maps State.terminated + State.termination_reason → escalated bool.
    escalated = True iff terminated AND reason == "anger_threshold_crossed".
    """
    return bool(
        state.terminated
        and state.termination_reason == "anger_threshold_crossed"
    )


def _revealed_demands(state: State) -> list[str]:
    """
    Derives the subset of originally-hidden demands that have now been
    revealed (i.e., moved out of demands_hidden).

    Strategy: revealed = demands_total − demands_stated − demands_hidden
    (demands_total = stated + hidden at episode start; anything that was
    hidden but is no longer in demands_hidden has been revealed).

    Falls back to an empty list if demands_total is not populated.
    """
    if not state.demands_total:
        return []
    hidden_set = set(state.demands_hidden)
    stated_set = set(state.demands_stated)
    return [d for d in state.demands_total if d not in hidden_set and d not in stated_set]


# ===========================================================================
# Function 1 — reward_outcome
# PS1 row 1 | contracts key: "outcome"
# ===========================================================================

def reward_outcome(prev_state: State, curr_state: State, action: Action) -> float:
    """
    PS1 reward spec — Row 1: Terminal outcome signal.
    Contracts key: "outcome"

    State fields read:
        curr_state.terminated, curr_state.termination_reason,
        curr_state.turn, curr_state.max_turns

    Returns:
        +1.0  — termination_reason == "commitment_reached"  (resolved)
        -0.5  — termination_reason == "anger_threshold_crossed"  (escalated)
         0.0  — timeout (turn >= max_turns, not resolved, not escalated)
         0.0  — any other termination reason ("forbidden_action")
         0.0  — mid-episode (not yet terminated)

    This is the only non-differentiable signal in the module.
    It is the primary anchor for the reward landscape.
    """
    if _is_resolved(curr_state):
        return 1.0
    if _is_escalated(curr_state):
        return -0.5
    # timeout or forbidden_action or mid-episode — no terminal bonus/penalty
    return 0.0


# ===========================================================================
# Function 2 — reward_deescalation
# PS1 row 2 | contracts key: "deescalation"
# ===========================================================================

def reward_deescalation(prev_state: State, curr_state: State, action: Action) -> float:
    """
    PS1 reward spec — Row 2: Anger de-escalation shaping signal.
    Contracts key: "deescalation"

    State fields read:
        prev_state.anger, curr_state.anger

    Computes delta = prev_state.anger - curr_state.anger
    (positive when anger dropped, negative when anger rose).
    Normalises by dividing by 10.0 (field range 0–10).

    FIX [V-DEESC-ASYMMETRY]: Applies a 1.5× amplifier on anger *rises*
    (negative delta) to match the trust asymmetry and discourage the agent
    from treating anger-up and anger-down steps as symmetric.  Without this,
    an oscillation strategy (drop 2pt → rise 2pt → repeat) nets 0.0 per
    cycle and is reward-neutral rather than penalised.

    Returns float clipped to [-1.0, +1.0].
    """
    delta = prev_state.anger - curr_state.anger
    normalised = delta / 10.0
    # Asymmetric: anger rising is penalised harder than the reward for dropping.
    if normalised < 0:
        normalised *= 1.5
    return float(np.clip(normalised, -1.0, 1.0))


# ===========================================================================
# Function 3 — reward_trust_building
# PS1 rows 3 + 4 | contracts key: "trust_building"
# ===========================================================================

def reward_trust_building(prev_state: State, curr_state: State, action: Action) -> float:
    """
    PS1 reward spec — Rows 3 + 4: Trust accumulator + emotional state signal.
    Contracts key: "trust_building"

    State fields read:
        prev_state.trust, curr_state.trust   (PS1 row 3)
        prev_state.fear,  curr_state.fear    (PS1 row 4 — emotional)

    The State model does not carry `hope` or `distrust` fields.
    We implement a composite signal:
        trust_component  = Δtrust / 10.0  with asymmetric ×1.5 on drops
        emotional_component = -Δfear / 10.0  (fear rising = negative)
        raw = trust_component + 0.3 * emotional_component

    The 0.3 weight on the emotional sub-component keeps trust the dominant
    signal while still rewarding fear reduction (which maps to PS1 row 4's
    intent).  The caller's curriculum weight for "trust_building" provides
    the outer scaling.

    Asymmetric trust penalty: if Δtrust < 0 (trust dropped), multiply by 1.5
    before combining — drops hurt more than gains help (PS1 doc design intent).

    Returns float clipped to [-1.0, +1.0].
    """
    # --- Trust sub-component (PS1 row 3) ---
    trust_delta = curr_state.trust - prev_state.trust
    trust_norm = trust_delta / 10.0
    if trust_norm < 0:
        trust_norm *= 1.5          # asymmetric: drops penalised harder

    # --- Emotional sub-component (PS1 row 4, fear proxy) ---
    fear_delta = curr_state.fear - prev_state.fear
    emotional_norm = -fear_delta / 10.0  # fear rising → negative contribution

    raw = trust_norm + 0.3 * emotional_norm
    return float(np.clip(raw, -1.0, 1.0))


# ===========================================================================
# Function 4 — reward_demand_coverage
# PS1 row 5 | contracts key: "demand_coverage"
# ===========================================================================

def reward_demand_coverage(prev_state: State, curr_state: State, action: Action) -> float:
    """
    PS1 reward spec — Row 5: Demand-coverage terminal signal.
    Contracts key: "demand_coverage"

    State fields read:
        curr_state.terminated, curr_state.termination_reason
        curr_state.demands_stated     (= PS1 stated_demands)
        curr_state.demands_hidden     (used to derive revealed_demands)
        curr_state.demands_total      (used to derive revealed_demands)
        curr_state.demands_addressed  (= PS1 addressed_demands)
        curr_state.turn, curr_state.max_turns

    Only computed at episode end (resolved / escalated / timeout / forbidden).
    Returns 0.0 mid-episode.

    At episode end:
        revealed_demands = demands_total − demands_stated − demands_hidden
        all_demands      = demands_stated + revealed_demands
        coverage         = len(demands_addressed) / len(all_demands)
                           (0.0 if all_demands is empty)

    Returns float clipped to [0.0, 1.0].
    """
    episode_ended = (
        curr_state.terminated
        or curr_state.turn >= curr_state.max_turns
    )
    if not episode_ended:
        return 0.0

    revealed = _revealed_demands(curr_state)
    all_demands = curr_state.demands_stated + revealed
    if not all_demands:
        return 0.0

    coverage = len(curr_state.demands_addressed) / len(all_demands)
    return float(np.clip(coverage, 0.0, 1.0))


# ===========================================================================
# Function 5 — reward_efficiency
# PS1 row 6 | contracts key: "efficiency"
# ===========================================================================

def reward_efficiency(prev_state: State, curr_state: State, action: Action) -> float:
    """
    PS1 reward spec — Row 6: Turn-efficiency bonus signal.
    Contracts key: "efficiency"

    State fields read:
        curr_state.terminated, curr_state.termination_reason,
        curr_state.turn, curr_state.max_turns

    Only awarded when the episode is resolved successfully.
    Returns 0.0 for all other termination reasons and mid-episode.

    Formula (when resolved):
        bonus = (max_turns - turn) / max_turns

    Faster resolution → higher bonus.
    Resolving on the last possible turn → bonus approaches 0.

    Returns float clipped to [0.0, 1.0].
    """
    if not _is_resolved(curr_state):
        return 0.0

    if curr_state.max_turns <= 0:
        return 0.0

    bonus = (curr_state.max_turns - curr_state.turn) / curr_state.max_turns
    return float(np.clip(bonus, 0.0, 1.0))


# ===========================================================================
# Function 6 — reward_compliance
# PS1 row 7 | contracts key: "compliance"
# ===========================================================================

import re

# RBI Regulatory Compliance Checks (Fair Practices Code)
_RE_LEGAL_THREAT = re.compile(r"(legal action|court|case|judge|arbitration|notice|police|fir|arrest|jail)", re.IGNORECASE)
_RE_SOCIAL_THREAT = re.compile(r"(family|neighbours?|neighbors?|neighborhood|boss|relative|parents|office|colleagues)", re.IGNORECASE)
_RE_VISIT_THREAT = re.compile(r"(visit|home|house|place|coming|ghar|aadmi bhej)", re.IGNORECASE)
_RE_ABUSIVE = re.compile(r"(shame|useless|defaulter|fraud|cheater|chor|liar)", re.IGNORECASE)

# Minimum word counts per action type.
# Prevents single-word or near-empty texts from farming the +0.1 compliance reward.
# FIX [V-COMPLIANCE-FARM]: Without a minimum-length check, the agent can send
# "ok" or "noted" on every turn and collect +0.1 * 0.1 = +0.01 per step with
# zero risk, accumulating +0.10 over a 10-turn episode for free.
_MIN_WORD_COUNT: dict[str, int] = {
    "send_message": 8,           # must form a coherent sentence
    "offer_emi": 6,              # must describe the offer
    "acknowledge_hardship": 6,   # must say something substantive
    "ask_open_question": 6,      # question must have context
    "confirm_in_writing": 6,     # confirmation needs specifics
    "stall": 5,                  # minimal but must explain why
    "escalate_authority": 4,     # brief invocation is fine
}


def reward_compliance(prev_state: State, curr_state: State, action: Action) -> float:
    """
    PS1 reward spec — Row 7: Action-format / RBI language compliance signal.
    Contracts key: "compliance"

    Action fields read:
        action.action_type  — must be a valid ActionType
        action.text         — must be non-empty AND meet minimum word count
        action.metadata     — must contain required keys for action_type

    Valid action types (from contracts.ACTION_TYPES):
        send_message, offer_emi, acknowledge_hardship, ask_open_question,
        confirm_in_writing, stall, escalate_authority

    Per-type metadata requirements:
        offer_emi          → metadata["emi_amount"] must be present
        all others         → no additional metadata keys required

    Per-type minimum word counts (see _MIN_WORD_COUNT):
        send_message       → 8 words minimum
        offer_emi          → 6 words minimum
        (etc.)

    FIX [V-COMPLIANCE-FARM]: Added minimum word counts per action type.
    Without this, the agent farms +0.1 per turn by sending single-word
    valid actions ("ok", "noted") indefinitely.

    Returns:
        +0.1  if all checks pass
        -0.2  if any check fails

    Intentionally NOT clipped — the negative penalty is a deliberate design
    choice to discourage malformed outputs from reaching the adversary.
    """
    # Check 1: valid action type
    action_type_str = (
        action.action_type
        if isinstance(action.action_type, str)
        else action.action_type.value
    )
    if action_type_str not in VALID_ACTIONS:
        return -0.2

    # Check 2: text is non-empty
    if not action.text or not action.text.strip():
        return -0.2

    # Check 3: minimum word count — prevents single-word compliance farming
    word_count = len(action.text.strip().split())
    min_words = _MIN_WORD_COUNT.get(action_type_str, 5)
    if word_count < min_words:
        return -0.2

    # Check 4: required metadata keys for this action type
    required_meta = _ACTION_METADATA_KEYS.get(action_type_str, [])
    for key in required_meta:
        if key not in (action.metadata or {}):
            return -0.2

    # Check 5: RBI Fair Practices Code — No threats/harassment
    # This matches the requirement of t4_3 in the integration test
    if (
        _RE_LEGAL_THREAT.search(action.text) or
        _RE_SOCIAL_THREAT.search(action.text) or
        _RE_VISIT_THREAT.search(action.text) or
        _RE_ABUSIVE.search(action.text)
    ):
        return -0.2

    return 0.1


# ===========================================================================
# Function 7 — reward_anti_exploit
# PS1 row 8 | contracts key: "anti_exploit"
# ===========================================================================

def reward_anti_exploit(prev_state: State, curr_state: State, action: Action) -> float:
    """
    PS1 reward spec — Row 8: Anti-exploit / degenerate-strategy penalty.
    Contracts key: "anti_exploit"

    State fields read:
        curr_state.message_history  — last N agent send_message texts

    Action fields read:
        action.action_type, action.text

    FIX [V-ANTI-EXPLOIT-SCOPE]: Penalty now applies to ALL action types,
    not only send_message.  An agent that spams acknowledge_hardship or stall
    on every turn would previously receive 0.0 anti_exploit penalty while
    collecting +0.1 compliance each turn.  We now apply the repetition check
    to the full message_history regardless of current action type, and the
    filler check to any action text.

    Check 1 — Repetition (all action types):
        If len(message_history) >= 2, computes TF-IDF cosine similarity
        between the last two messages in message_history.
        Threshold lowered from 0.85 → 0.75 to catch synonym-substitution
        evasion (e.g. swapping 'check' for 'review' while keeping the rest
        identical, which scores ~0.72–0.82 under TF-IDF).
        FIX [V-TFIDF-THRESHOLD]: 0.85 was empirically too permissive;
        measured synonym pairs score 0.67–0.86 depending on token overlap.
        similarity > 0.75  →  -0.2

    Check 2 — Keyword / filler-phrase stuffing (all action types):
        Uses exclusive longest-match counting so that "I understand your
        concerns" does not also trigger "I understand" in the same sentence.
        FIX [V-FILLER-OVERLAP]: Previous substring counting double-counted
        overlapping phrases, making the threshold of 3 easier to reach than
        intended.
        count >= 3 (exclusive matches)  →  -0.1

    Penalties are additive:
        Both triggered  → -0.3
        Only Check 1    → -0.2
        Only Check 2    → -0.1
        Neither         →  0.0
    """
    penalty = 0.0
    history = curr_state.message_history or []

    action_type_str = (
        action.action_type
        if isinstance(action.action_type, str)
        else action.action_type.value
    )

    # --- Check 1: repetition via TF-IDF cosine similarity (all action types) ---
    # FIX [V-ANTI-EXPLOIT-SCOPE] + FIX [V-TFIDF-THRESHOLD]
    if len(history) >= 2:
        last_two = history[-2:]
        if _HAS_SKLEARN:
            try:
                vectorizer = TfidfVectorizer()
                tfidf_matrix = vectorizer.fit_transform(last_two)
                sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
                if sim > 0.75:   # lowered from 0.85 to catch synonym evasion
                    penalty -= 0.2
            except ValueError:
                # Empty vocabulary (pure stop-words or numeric-only) — no signal.
                pass
        else:
            # Fallback when sklearn is unavailable: lexical overlap proxy.
            tokens_a = set(last_two[0].lower().split())
            tokens_b = set(last_two[1].lower().split())
            union = tokens_a | tokens_b
            overlap = len(tokens_a & tokens_b) / len(union) if union else 0.0
            if overlap > 0.75:
                penalty -= 0.2

    # --- Check 2: filler-phrase stuffing with exclusive longest-match counting ---
    # FIX [V-FILLER-OVERLAP]: iterate through the text once, consuming each
    # matched phrase position so shorter substrings cannot double-count.
    text_lower = (action.text or "").lower()
    filler_count = 0
    remaining = text_lower
    for phrase in _FILLER_PHRASES:   # ordered longest-first
        phrase_lower = phrase.lower()
        if phrase_lower in remaining:
            filler_count += remaining.count(phrase_lower)
            # Consume matched occurrences to prevent shorter-phrase double-counting
            remaining = remaining.replace(phrase_lower, " " * len(phrase_lower))
    if filler_count >= 3:
        penalty -= 0.1

    return penalty


# ===========================================================================
# Function 8 — reward_format_compliance
# Added for Format Compliancy (Addresses: Problem 1)
# ===========================================================================

import json as _json

# Mirror the extractor in client/utils.action_from_text so the scorer grades
# the same JSON object the parser will accept.  Greedy on purpose: we want
# the outermost {...} even when the model wraps it in markdown fences.
_JSON_SNIFFER_RE = re.compile(r"\{.*\}", re.DOTALL)


def reward_format_compliance(action: Action) -> float:
    """JSON-format compliance reward.

    Grades the LLM's raw completion (preserved in
    ``action.metadata["raw_text"]`` by ``client.utils.action_from_text``)
    against the JSON contract specified in
    ``training.prompt_builder.build_system_prompt``::

        {"thought_process": ..., "action_type": ..., "text": ..., "metadata": {}}

    Scoring (additive)
    ------------------
    +0.15  valid JSON decoded **and** has both ``action_type`` and ``text``
    +0.05  ``action_type`` value is in :data:`VALID_ACTIONS`
    -0.25  JSON decoding fails  **OR**  decoded JSON lacks the ``text`` key

    A successful parse with ``text`` present but ``action_type`` missing
    yields 0.0 — neither bonus applies, but the ``-0.25`` penalty does
    not trigger because the user-spec only conditions it on JSON-decode
    failure or missing ``text``.

    The function is deterministic, has no I/O, and runs in O(n) on the
    raw_text length (a single regex scan + one ``json.loads`` call), so
    it stays well inside the per-step budget on the GRPO hot path.
    """
    raw_text = (action.metadata or {}).get("raw_text", "")
    if not isinstance(raw_text, str) or not raw_text.strip():
        return -0.25

    match = _JSON_SNIFFER_RE.search(raw_text)
    if not match:
        return -0.25

    try:
        parsed = _json.loads(match.group(0))
    except _json.JSONDecodeError:
        return -0.25

    if not isinstance(parsed, dict):
        return -0.25

    if "text" not in parsed:
        return -0.25

    reward = 0.0
    if "action_type" in parsed and "text" in parsed:
        reward += 0.15

    action_type_val = parsed.get("action_type")
    if isinstance(action_type_val, str) and action_type_val.strip() in VALID_ACTIONS:
        reward += 0.05

    return reward

# ===========================================================================
# Combiner — compose()   (also aliased as compute_reward for test compat)
# ===========================================================================

def compose(
    prev_state: State,
    curr_state: State,
    action: Action,
    weights: dict[str, float] | None = None,
) -> tuple[float, dict[str, Any]]:
    """Fast-Hybrid combiner — deterministic gates + LLM-judged soft skills.

    Pipeline:

    1. **Cheap deterministic raws first.** Outcome, demand-coverage,
       efficiency, anti-exploit, format-compliance, RBI/compliance — all
       computed from State + Action with zero network I/O. These are
       cheap (microseconds) and *always* run.

    2. **Short-circuit gate.** If either of the deterministic gates has
       already condemned the turn — broken JSON (``format_compliance < 0``)
       or RBI-forbidden words (``compliance < 0``) — we skip the LLM
       judge entirely. The LLM-dependent metrics (``deescalation`` and
       ``trust_building``) are forced to 0.0 so that a malformed turn
       cannot cherry-pick empathy bonuses while the deterministic
       penalties are also firing. This keeps GRPO honest and saves an
       entire NIM round-trip per failing completion.

    3. **Qualitative replacement.** When the deterministic gates pass,
       we ask :mod:`environment.llm_judge` to score the agent's reply
       on (empathy, strategy). The judge's clamped [0,1] outputs *replace*
       the legacy deterministic deescalation / trust_building values for
       this turn:

         * ``empathy_score``  →  ``deescalation``
         * ``strategy_score`` →  ``trust_building``

       The judge is engineered to never raise — on disabled / timeout /
       malformed output it returns 0.0 / 0.0 with ``fallback_used=True``.
       That fallback rate is surfaced in the breakdown for monitoring.

    Args:
        prev_state : State before the action was applied (state_before).
        curr_state : State after the adversary reacted (state_after).
        action     : The Action the LLM sent this turn. Optional context
                     for the judge can be passed via ``action.metadata``:
                       * ``borrower_msg``        — most recent borrower line
                       * ``conversation_history``— list of {agent, borrower}
        weights    : Optional weight dict keyed by REWARD_BREAKDOWN_KEYS.
                     Missing keys fall back to _DEFAULT_WEIGHTS (stage_3).
                     Pass CURRICULUM_WEIGHTS["stage_N"] directly here.

    Returns:
        ``(scalar_reward, breakdown)`` — signature is preserved so the
        GRPOTrainer integration is unchanged.

        scalar_reward — weighted sum (float) — this goes into GRPOTrainer.
        breakdown     — dict with keys:
            <component>          → raw component value (matches contract)
            raw_<component>      → unweighted component value
            weighted_<component> → weight * raw value
            weight_<component>   → weight applied
            total                → scalar_reward  (convenience copy)

            Plus diagnostic-only keys for the judge:
            judge_used           → bool: True if judge actually ran
            judge_reason         → str:  "ok" | "disabled" | "short_circuit" | …
            judge_latency_ms     → float
            judge_empathy        → float in [0,1]    (raw judge output)
            judge_strategy       → float in [0,1]    (raw judge output)

        These ``judge_*`` keys live alongside the existing breakdown but
        are intentionally NOT in REWARD_BREAKDOWN_KEYS so they don't
        accidentally feed back into the weighted sum.
    """
    resolved_weights: dict[str, float] = {**_DEFAULT_WEIGHTS, **(weights or {})}
    resolved_weights["format_compliance"] = 0.5  # reduced to prevent dominant strategy

    # ── Step 1: deterministic raws (always run, cheap) ──────────────────────
    raw: dict[str, float] = {
        "outcome":           reward_outcome(prev_state, curr_state, action),
        "demand_coverage":   reward_demand_coverage(prev_state, curr_state, action),
        "efficiency":        reward_efficiency(prev_state, curr_state, action),
        "compliance":        reward_compliance(prev_state, curr_state, action),
        "anti_exploit":      reward_anti_exploit(prev_state, curr_state, action),
        "format_compliance": reward_format_compliance(action),
        # Placeholders — populated below either by the short-circuit branch
        # (zeroed) or by the LLM judge branch (replaced with judge scores).
        "deescalation":   0.0,
        "trust_building": 0.0,
    }

    # ── Step 2: short-circuit gate ──────────────────────────────────────────
    #
    # The two cheap "format guards" act as kill-switches for the qualitative
    # signal. We deliberately use strict "< 0" so a zero score from a benign
    # turn (no penalty, no reward) still flows into the judge; only an
    # actively penalised turn forfeits its empathy/strategy budget.
    short_circuited = (raw["format_compliance"] < 0.0) or (raw["compliance"] < 0.0)

    judge_used = False
    judge_reason = "short_circuit" if short_circuited else "pending"
    judge_latency_ms = 0.0
    judge_empathy_raw = 0.0
    judge_strategy_raw = 0.0
    judge_fallback_used = True  # default; flipped below on success

    if short_circuited:
        # Deterministic gate failed — keep deescalation / trust_building at 0.0
        # and DO NOT call the LLM judge. This is the optimisation that makes
        # the Fast-Hybrid system viable on a hot training loop.
        pass
    else:
        # ── Step 3: qualitative replacement ─────────────────────────────────
        #
        # Pull optional context out of action.metadata. We import lazily so
        # the (already very chatty) reward.py module does not pay for the
        # judge module on import — and so the test suite can monkey-patch
        # `environment.llm_judge.score_response` cleanly.
        try:
            from environment import llm_judge as _llm_judge   # type: ignore
        except Exception as exc:                              # pragma: no cover
            _llm_judge = None  # type: ignore[assignment]
            judge_reason = f"import_error:{type(exc).__name__}"

        if _llm_judge is not None:
            meta = action.metadata or {}
            borrower_msg = ""
            history = None
            if isinstance(meta, dict):
                bm = meta.get("borrower_msg") or meta.get("borrower_message")
                if isinstance(bm, str):
                    borrower_msg = bm
                hist = meta.get("conversation_history")
                if isinstance(hist, list):
                    history = hist

            try:
                score = _llm_judge.score_response(
                    agent_text=action.text or "",
                    borrower_msg=borrower_msg,
                    history=history,
                )
            except Exception as exc:                          # pragma: no cover
                # llm_judge is hardened to never raise, but belt-and-braces:
                # if it ever does, treat the turn as a soft fallback rather
                # than blowing up the whole rollout.
                judge_reason = f"judge_exception:{type(exc).__name__}"
            else:
                judge_empathy_raw = float(score.empathy_score)
                judge_strategy_raw = float(score.strategy_score)
                judge_latency_ms = float(score.latency_ms)
                judge_fallback_used = bool(score.fallback_used)
                judge_reason = score.reason
                judge_used = not score.fallback_used

                # Map judge scores onto the deterministic slots. We keep the
                # raw values so wandb can compare deterministic-vs-judge.
                raw["deescalation"] = judge_empathy_raw
                raw["trust_building"] = judge_strategy_raw

    # ── Step 4: contract enforcement & weighted sum ─────────────────────────
    for key in REWARD_BREAKDOWN_KEYS:
        assert key in raw, (
            f"reward.py bug: missing key '{key}' — must match REWARD_BREAKDOWN_KEYS"
        )

    total = sum(resolved_weights.get(k, 0.0) * raw[k] for k in raw)

    breakdown: dict[str, Any] = {}

    # Raw values at top-level (matches StepResult.info["reward_breakdown"] contract).
    for component, raw_val in raw.items():
        breakdown[component] = raw_val

    # Detailed per-component monitoring columns (for wandb).
    for component, raw_val in raw.items():
        w = resolved_weights.get(component, 0.0)
        breakdown[f"raw_{component}"] = raw_val
        breakdown[f"weight_{component}"] = w
        breakdown[f"weighted_{component}"] = w * raw_val

    # Judge diagnostics — purely informational, never fed back into the sum.
    breakdown["judge_used"] = judge_used
    breakdown["judge_reason"] = judge_reason
    breakdown["judge_latency_ms"] = judge_latency_ms
    breakdown["judge_empathy"] = judge_empathy_raw
    breakdown["judge_strategy"] = judge_strategy_raw
    breakdown["judge_fallback_used"] = judge_fallback_used
    breakdown["judge_short_circuited"] = short_circuited

    breakdown["total"] = total

    return float(total), breakdown



# ===========================================================================
# Public Wrapper — Rubric
# ===========================================================================

class Rubric:
    """
    Standard interface for the reward system.
    This class wraps the compose() function to provide a modular handoff
    between the environment (Person A) and the reward logic (Person B).
    """

    def __init__(self, curriculum_stage: int = 3):
        """
        Initializes the rubric with weights for a specific curriculum stage.
        """
        stage_key = f"stage_{curriculum_stage}"
        self.weights = CURRICULUM_WEIGHTS.get(stage_key, _DEFAULT_WEIGHTS)
        self.stage = curriculum_stage

    def compose(
        self, 
        state_before: State, 
        state_after: State, 
        action: Action, 
        episode_done: bool
    ) -> tuple[float, dict]:
        """
        Computes the total weighted reward and breakdown.
        Matches the signature expected by NegotiationEnv.
        """
        return compose(
            prev_state=state_before,
            curr_state=state_after,
            action=action,
            weights=self.weights
        )

# Alias so external code and tests can call either name.
compute_reward = compose
