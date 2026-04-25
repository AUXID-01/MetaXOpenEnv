# environment/response_generator.py
"""
HOLDS:
    1. ResponseGenerator class — Hybrid-Dynamic borrower voice powered by
       NVIDIA NIM (Llama-3.1-70B-Instruct).  Maps the deterministic emotional
       state (anger / trust / fear) to an Intersection Archetype, builds a
       strict "translator" system prompt, queries NIM, and caches by
       (Archetype, action_type) to keep training latency tractable.
    2. TEMPLATES + pick_template() — the legacy deterministic template bank,
       kept as the offline fallback when the NIM API is unavailable
       (no key, network error, or timeout) and as a back-compat surface for
       existing tests (tests/test_response_generator.py imports pick_template).

DESIGN INVARIANT — INTERFACE LOCK
─────────────────────────────────
The reward signal is computed from the deterministic State (anger, trust,
fear, demands, terminated, ...) which is mutated by adversary.py PRIOR to
the borrower text being generated.  The LLM is *only* a translator of that
state into natural language; it never writes back to State.  This keeps the
reward function deterministic and reproducible while the dialogue feels
human.

RUNS:
    adversary.py -> _generate_response() -> get_response_generator().generate(...)

CONNECTS TO:
    adversary.py (caller), classifier.py (signal taxonomy aligns with
    action_type bins below), env.py (consumes the resulting borrower_msg
    on Observation).
"""

from __future__ import annotations

import hashlib
import logging
import math
import os
import re
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

# ─────────────────────────────────────────────────────────────────────────────
# Secrets management — load `.env` once at module import.
# ─────────────────────────────────────────────────────────────────────────────
#
# We use python-dotenv so that NVIDIA_API_KEY (and any future credentials)
# can live in a project-root `.env` file that is git-ignored.  Calling
# `load_dotenv()` here is idempotent — repeated calls do not override
# variables already set in the real environment, which means:
#
#   • Shell exports (`NVIDIA_API_KEY=... python ...`) still win, just like
#     before — required for CI / Docker / Kubernetes deployments where
#     secrets come from a mounted file or vault rather than .env.
#   • Local developers can drop a key into the repo's `.env` and run
#     `python -m pytest` or `python voice_proof_of_life.py` without any
#     extra `export` step.
#
# `find_dotenv()` walks upward from this file's location looking for a
# `.env`, so the lookup works whether the user runs the env from
# `negotiation-env/`, the repo root, or some training rig elsewhere.
try:                                                # pragma: no cover
    from dotenv import find_dotenv, load_dotenv
    _ENV_FILE = find_dotenv(usecwd=True)
    if _ENV_FILE:
        load_dotenv(_ENV_FILE, override=False)
except Exception:                                   # pragma: no cover
    # python-dotenv is in requirements.txt, but failing to import it must
    # never crash the env at training time — the operator can still export
    # NVIDIA_API_KEY in their shell, which is how CI does it anyway.
    _ENV_FILE = ""

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# NIM API CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

NIM_BASE_URL = "https://integrate.api.nvidia.com/v1"
NIM_MODEL    = "meta/llama-3.1-70b-instruct"
NIM_TIMEOUT_S = 30.0          # hardened: per-request hard ceiling
NIM_MAX_TOKENS = 90
NIM_TEMPERATURE = 0.7
NIM_TOP_P = 0.9

# ─────────────────────────────────────────────────────────────────────────────
# Sentinel placeholder values that ship in `.env` / `.env.example`.  We treat
# them as "not configured" so a casual `cp .env.example .env` doesn't end up
# pointing the OpenAI client at the literal string "your_actual_key_here".
# ─────────────────────────────────────────────────────────────────────────────
_PLACEHOLDER_KEYS: frozenset[str] = frozenset({
    "",
    "your_actual_key_here",
    "your-actual-key-here",
    "changeme",
    "<your-key>",
    "<your_key_here>",
    "REPLACE_ME",
})


def _read_nvidia_api_key() -> Optional[str]:
    """
    Resolve NVIDIA_API_KEY from the (already-loaded) environment.

    Returns the trimmed key string when one is configured, else None.
    Empty strings, whitespace-only values, and known placeholder strings
    from `.env.example` all collapse to None so they cannot accidentally
    be sent to NVIDIA's API as if they were real credentials.
    """
    raw = os.getenv("NVIDIA_API_KEY")
    if raw is None:
        return None
    key = raw.strip()
    if not key or key.lower() in {p.lower() for p in _PLACEHOLDER_KEYS}:
        return None
    return key


def require_nvidia_api_key() -> str:
    """
    Public helper for callers that *know* they need a real NIM key
    (e.g. ``voice_proof_of_life.py``).

    Raises a clear, descriptive ``ValueError`` if the key is missing,
    blank, or still set to the `.env.example` placeholder.  The exception
    message tells the operator exactly which variable to set and where —
    this is the contract requested in the secrets-management spec.
    """
    key = _read_nvidia_api_key()
    if key is None:
        raise ValueError(
            "NVIDIA_API_KEY is not configured.\n"
            "  • Copy `.env.example` to `.env` at the project root and fill "
            "in your real NVIDIA NIM key, or\n"
            "  • Export it in your shell:  export NVIDIA_API_KEY=nvapi-...\n"
            "Get a key at https://build.nvidia.com/."
        )
    return key

# ─────────────────────────────────────────────────────────────────────────────
# HARDENING CONFIGURATION  (see tests/test_edge_cases.py)
# ─────────────────────────────────────────────────────────────────────────────
#
# Three independent guards keep the env from crashing during long RL training:
#
#   1. NETWORK GUARD  — strict timeout + 3-attempt exponential back-off on
#                       transient HTTP errors (429 / 500 / 502 / 503 / 504,
#                       connection drops, read timeouts).  After all retries
#                       are exhausted we fall through to the legacy template
#                       bank instead of letting the exception escape.
#
#   2. DATA GUARD     — agent_text is truncated to MAX_AGENT_WORDS before it
#                       reaches the prompt builder; LLM replies that are
#                       empty, model-refusals ("I cannot ...", "I'm sorry,
#                       I can't help"), or that leak template placeholders
#                       ("{loan_amount}") are rejected and replaced with a
#                       short in-character "sullen" line from the template
#                       bank — which still sounds like the borrower.
#
#   3. BOUNDARY GUARD — every numeric field touched by this module is run
#                       through `_finite()` before bucketing so NaN / Inf
#                       can never index into _BUCKET_NAMES or _ARCHETYPES.
#
# All three guards are best-effort:  none of them mutate the deterministic
# State, so the reward signal is unchanged whether they fire or not.

NIM_MAX_RETRIES: int = 3              # total attempts (initial + retries)
NIM_BACKOFF_BASE_S: float = 0.5       # 0.5s, 1.0s, 2.0s … exponential
NIM_BACKOFF_CAP_S: float = 8.0        # don't sleep longer than this between retries

MAX_AGENT_WORDS: int = 500            # tokens-ish; words are a stable proxy
                                      # so the prompt never blows the context.

# HTTP statuses we consider transient (and therefore retry-worthy).
_RETRYABLE_HTTP_STATUSES: frozenset[int] = frozenset({429, 500, 502, 503, 504})

# Regexes used by _is_invalid_reply(); compiled once at import.
# Refusal patterns cover the common "I'm an AI / I cannot help / as a
# language model" outputs that occasionally slip through Llama's persona
# constraints.  Any match → reject and fall back to the sullen template.
_RE_REFUSAL = re.compile(
    r"("
    r"\bas an? (ai|language model|assistant)\b"
    r"|\bi(?:'| a)?m an? (ai|language model|assistant)\b"
    r"|\bi cannot (answer|help|assist|comply|do)\b"
    r"|\bi can'?t (answer|help|assist|comply|do)\b"
    r"|\bi(?:'| a)?m (sorry|unable),? (?:but )?i can(?:'?t|not)\b"
    r"|\bi(?:'?m| am)? not able to (answer|help|assist|comply)\b"
    r"|\bi(?:'?m| am)? unable to (?:answer|provide|help|assist|comply)\b"
    r"|\bi (?:do not|don'?t) feel comfortable\b"
    r")",
    re.IGNORECASE,
)
_RE_PLACEHOLDER_LEAK = re.compile(r"\{[a-zA-Z_][a-zA-Z0-9_]*\}")


def _nim_enabled() -> bool:
    """
    The voice is OFF by default unless explicitly enabled.  This keeps the
    existing pytest suite hermetic (no outbound HTTP, no API-key dependency)
    while letting end-to-end / training runs opt in via NEG_LLM_ENABLED=1.
    A valid NVIDIA_API_KEY is also required (placeholder values from
    `.env.example` are treated as "not configured" — see
    `_read_nvidia_api_key`).
    """
    if os.getenv("NEG_LLM_ENABLED", "0") not in ("1", "true", "True"):
        return False
    return _read_nvidia_api_key() is not None


# ─────────────────────────────────────────────────────────────────────────────
# SEMANTIC PERSONA MAPPER
# ─────────────────────────────────────────────────────────────────────────────
#
# Each emotion (Anger, Trust, Fear) is bucketed into 5 zones on [0, 10]:
#     bucket 0:  [0.0,  2.0)
#     bucket 1:  [2.0,  4.0)
#     bucket 2:  [4.0,  6.0)
#     bucket 3:  [6.0,  8.0)
#     bucket 4:  [8.0, 10.0]   (closed at the right edge)
#
# Boundary semantics: a value lying exactly on a boundary belongs to the
# UPPER bucket (>= rule).  This is consistent with the existing classifier
# and avoids the "is 6.0 angry or agitated?" ambiguity raised in the
# Hybrid-Voice spec.
#
# The Intersection Archetype is a coarse 3x3x2 lattice (LOW / MID / HIGH on
# each axis with fear collapsed to LOW/HIGH) yielding 18 distinct archetypes.
# We give each a short evocative label so the prompt reads naturally and the
# cache key is stable across runs.

_BUCKET_EDGES: List[float] = [2.0, 4.0, 6.0, 8.0]   # exclusive lower edges of buckets 1..4

_BUCKET_NAMES: List[str] = ["very_low", "low", "moderate", "elevated", "extreme"]


def _finite(value: Any, default: float = 0.0) -> float:
    """
    BOUNDARY GUARD primitive.

    Coerce *anything* that is supposed to be a [0, 10] emotional float into a
    real, finite number on that interval.  NaN, +/-Inf, None, strings, and
    other junk all collapse to `default` (clamped) so downstream bucketing
    can never index out of bounds.

    The guard is intentionally permissive — better to silently land on a safe
    default mid-training than to crash a 10k-step RL run because of one
    pathological multiplier in the personality table.
    """
    try:
        v = float(value)
    except (TypeError, ValueError):
        v = float(default)
    if not math.isfinite(v):
        v = float(default)
    if v < 0.0:
        return 0.0
    if v > 10.0:
        return 10.0
    return v


def _bucket(value: float) -> int:
    """Bucket a float on [0,10] into 0..4.  Boundary-on-edge → upper bucket.

    Hardened: non-finite or out-of-range values collapse to 0.0 via
    `_finite()` before bucketing, so NaN / Inf can never index out of
    bounds on `_BUCKET_NAMES`.
    """
    v = _finite(value)
    for i, edge in enumerate(_BUCKET_EDGES):
        if v < edge:
            return i
    return 4


def _band(value: float) -> str:
    """LOW / MID / HIGH coarsening used by the archetype lattice."""
    b = _bucket(value)
    if b <= 1:
        return "LOW"
    if b == 2:
        return "MID"
    return "HIGH"


# 3 (anger band) × 3 (trust band) × 2 (fear: low|high) = 18 archetypes.
# Fear is collapsed to LOW (bucket <=2) vs HIGH (bucket >=3) because in
# practice fear modulates *tone* (begging vs defiant) more than *content*.
_ARCHETYPES: Dict[Tuple[str, str, str], str] = {
    # Low anger ────────────────────────────────────────────────────────────
    ("LOW",  "LOW",  "LOW_F"):  "Detached",        # cold, evasive, minimal disclosure
    ("LOW",  "LOW",  "HIGH_F"): "Anxious_Wary",    # nervous but not aggressive
    ("LOW",  "MID",  "LOW_F"):  "Cooperative",     # neutral, willing to listen
    ("LOW",  "MID",  "HIGH_F"): "Worried_Open",    # fearful but ready to engage
    ("LOW",  "HIGH", "LOW_F"):  "Trusting_Calm",   # the textbook good outcome
    ("LOW",  "HIGH", "HIGH_F"): "Grateful_Fragile",# trusts, but easily destabilised
    # Mid anger ────────────────────────────────────────────────────────────
    ("MID",  "LOW",  "LOW_F"):  "Agitated_Skeptic",# argumentative, cynical
    ("MID",  "LOW",  "HIGH_F"): "Defensive",       # snappy + scared
    ("MID",  "MID",  "LOW_F"):  "Stressed_Pragmatic", # frustrated but transactional
    ("MID",  "MID",  "HIGH_F"): "Cornered_Negotiator", # fear pushes them to deal
    ("MID",  "HIGH", "LOW_F"):  "Frustrated_Ally", # trusts you, hates the situation
    ("MID",  "HIGH", "HIGH_F"): "Pleading_Cooperative",# desperate to make it work
    # High anger ───────────────────────────────────────────────────────────
    ("HIGH", "LOW",  "LOW_F"):  "Antagonistic",    # outright hostile, no trust
    ("HIGH", "LOW",  "HIGH_F"): "Panicked_Hostile",# screaming + terrified
    ("HIGH", "MID",  "LOW_F"):  "Outraged_Rational",# articulate anger, will sue
    ("HIGH", "MID",  "HIGH_F"): "Volatile",        # whiplashing — careful
    ("HIGH", "HIGH", "LOW_F"):  "Betrayed_Loyal",  # trusted you, feels burned
    ("HIGH", "HIGH", "HIGH_F"): "Breakdown",       # collapsing under pressure
}


def _classify_archetype(anger: float, trust: float, fear: float) -> str:
    a_band = _band(anger)
    t_band = _band(trust)
    f_band = "HIGH_F" if _bucket(fear) >= 3 else "LOW_F"
    return _ARCHETYPES.get(
        (a_band, t_band, f_band),
        "Cooperative",  # safe default if a band combo is somehow missing
    )


def persona_snapshot(
    anger: float,
    trust: float,
    fear: float,
    profile: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Builds the structured persona description that the LLM will be asked to
    voice.  Returned dict is the canonical input to both the prompt builder
    and the cache key.

    Returned keys (stable contract — used by tests):
        archetype          : str   — one of the 18 archetype labels
        anger_bucket       : str   — very_low | low | moderate | elevated | extreme
        trust_bucket       : str
        fear_bucket        : str
        anger              : float — clamped to [0,10]
        trust              : float
        fear               : float
        profile_id         : str   — pass-through for logging
        profile_short      : str   — one-line context the LLM can quote from
    """
    # BOUNDARY GUARD: NaN / Inf / type-junk all collapse to 0.0 here.
    a = _finite(anger)
    t = _finite(trust)
    f = _finite(fear)

    archetype = _classify_archetype(a, t, f)

    profile = profile or {}
    profile_short_parts: List[str] = []
    if profile.get("name"):
        profile_short_parts.append(str(profile["name"]))
    if profile.get("reason"):
        profile_short_parts.append(str(profile["reason"]).replace("_", " "))
    if profile.get("loan_type"):
        profile_short_parts.append(str(profile["loan_type"]).replace("_", " "))
    if profile.get("overdue_days") is not None:
        profile_short_parts.append(f"{profile['overdue_days']} days overdue")
    profile_short = ", ".join(profile_short_parts) if profile_short_parts else "borrower"

    return {
        "archetype":     archetype,
        "anger_bucket":  _BUCKET_NAMES[_bucket(a)],
        "trust_bucket":  _BUCKET_NAMES[_bucket(t)],
        "fear_bucket":   _BUCKET_NAMES[_bucket(f)],
        "anger":         a,
        "trust":         t,
        "fear":          f,
        "profile_id":    profile.get("id", "P00"),
        "profile_short": profile_short,
    }


# ─────────────────────────────────────────────────────────────────────────────
# STRICT VOICE GUARDRAILS
# ─────────────────────────────────────────────────────────────────────────────
#
# The system prompt forces the LLM into the role of a TRANSLATOR of the
# numerical state, not a co-author of the negotiation.  Two hard rules:
#
#   1. The borrower's tone is dictated by the Archetype.  The model MUST NOT
#      decide to be conciliatory if the state is hostile, even if the agent
#      message is empathetic — anger only drops via the deterministic state
#      machine.
#   2. The borrower MUST NOT agree to a deal unless `terminated=True` AND
#      `termination_reason="commitment_reached"`.  Agreement is a state
#      transition, not a free choice of the language model.

_VOICE_RULES = (
    "You are an Indian retail borrower being contacted by a debt-collection "
    "agent.  You are NOT an AI assistant.  You are a character whose entire "
    "emotional state has already been decided by an external state machine.  "
    "Your only job is to TRANSLATE that state into one short, natural reply.\n"
    "\n"
    "ABSOLUTE RULES — BREAKING ANY OF THESE INVALIDATES THE RESPONSE:\n"
    "  1. Reply in 1–3 sentences (max ~50 words).  No greetings, no sign-off.\n"
    "  2. Match the Archetype's tone EXACTLY.  If the archetype is hostile, "
    "you are hostile no matter what the agent said.  If trust is LOW, you "
    "do NOT believe the agent's empathy.\n"
    "  3. NEVER agree to a deal, sign anything, or say 'I commit' unless the "
    "Persona Snapshot explicitly says terminated=True and "
    "termination_reason=commitment_reached.  Otherwise you may *consider* an "
    "offer at most.\n"
    "  4. NEVER threaten, abuse, or insult the agent — you are the borrower, "
    "not the collector.  Anger is expressed as frustration, not slurs.\n"
    "  5. NEVER reveal hidden information about yourself unless the snapshot "
    "lists trust as HIGH.\n"
    "  6. Stay in first-person, present tense.  Mix English with one or two "
    "natural Hindi/Hinglish words only if it fits the persona (e.g. 'bhai', "
    "'pakka', 'arre').  Do NOT translate or explain Hindi terms.\n"
    "  7. Output ONLY the borrower's spoken line.  No stage directions, no "
    "quotation marks, no 'Borrower:' prefix, no markdown.\n"
)


_ARCHETYPE_TONE_HINTS: Dict[str, str] = {
    "Detached":             "flat, minimal, evasive — answers in fragments",
    "Anxious_Wary":         "cautious, short sentences, double-checks intent",
    "Cooperative":          "polite, neutral, willing to discuss numbers",
    "Worried_Open":         "softly anxious but engaged — leans toward help",
    "Trusting_Calm":        "warm, candid, treats the agent like an ally",
    "Grateful_Fragile":     "thanks the agent but voice may shake",
    "Agitated_Skeptic":     "cynical, pushes back, questions every claim",
    "Defensive":            "snappy, defensive, scared underneath",
    "Stressed_Pragmatic":   "tired, transactional, wants to be done with this",
    "Cornered_Negotiator":  "fear-driven bargaining — willing to reveal one demand",
    "Frustrated_Ally":      "trusts agent personally, frustrated at the system",
    "Pleading_Cooperative": "begging-tone but cooperative — 'please help me'",
    "Antagonistic":         "openly hostile, sarcastic, dismissive of empathy",
    "Panicked_Hostile":     "shouting and panicking simultaneously",
    "Outraged_Rational":    "articulate fury — invokes RBI / consumer rights",
    "Volatile":             "whiplashing between anger and fear within one reply",
    "Betrayed_Loyal":       "the wounded-friend tone — 'I trusted you, and now this?'",
    "Breakdown":            "near-collapse — short fractured sentences, may cry",
}


# Coarse map: action_type (from classifier) → what the borrower is reacting TO.
# Aligns with classifier.primary_action_type values used elsewhere in this repo.
_ACTION_REACTION_FRAME: Dict[str, str] = {
    "empathize":          "The agent has just expressed empathy or acknowledgement.",
    "clarify":            "The agent has asked you to clarify your situation.",
    "probe_capacity":     "The agent is probing how much you can actually pay.",
    "offer_plan":         "The agent has put a structured payment / EMI plan on the table.",
    "seek_commitment":    "The agent is asking you to commit to a payment.",
    "warn_noncompliance": "The agent has issued a warning, threat, or escalation.",
    "unknown":            "The agent's last message was unclear or off-topic.",
    # Action-registry names from contracts.ACTION_TYPES — sometimes adversary
    # is called with these instead of the classifier-bin names.
    "send_message":       "The agent has just sent you a message.",
    "offer_emi":          "The agent has offered you a specific EMI amount.",
    "acknowledge_hardship": "The agent has acknowledged your hardship.",
    "ask_open_question":  "The agent has asked you an open-ended question.",
    "confirm_in_writing": "The agent has offered to confirm an arrangement in writing.",
    "stall":              "The agent is buying time / not committing.",
    "escalate_authority": "The agent has invoked a senior / supervisor.",
}


def _build_user_prompt(
    snapshot: Dict[str, Any],
    action_type: str,
    agent_text: str,
    profile: Optional[Dict[str, Any]],
    terminated: bool,
    termination_reason: str,
) -> str:
    """Builds the user-turn prompt for NIM."""
    tone_hint = _ARCHETYPE_TONE_HINTS.get(snapshot["archetype"], "")
    frame = _ACTION_REACTION_FRAME.get(action_type, _ACTION_REACTION_FRAME["unknown"])
    profile = profile or {}

    demands = profile.get("demands_stated") or profile.get("demands") or []
    demands_str = ", ".join(demands) if demands else "(none stated yet)"

    return (
        f"PERSONA SNAPSHOT\n"
        f"  Archetype          : {snapshot['archetype']}  ({tone_hint})\n"
        f"  Anger              : {snapshot['anger']:.1f}/10  ({snapshot['anger_bucket']})\n"
        f"  Trust              : {snapshot['trust']:.1f}/10  ({snapshot['trust_bucket']})\n"
        f"  Fear               : {snapshot['fear']:.1f}/10  ({snapshot['fear_bucket']})\n"
        f"  Borrower           : {snapshot['profile_short']}\n"
        f"  Stated demands     : {demands_str}\n"
        f"  terminated         : {terminated}\n"
        f"  termination_reason : {termination_reason or 'None'}\n"
        f"\n"
        f"CONTEXT\n"
        f"  {frame}\n"
        f"  Agent's last message: \"{(agent_text or '').strip()[:280]}\"\n"
        f"\n"
        f"TASK\n"
        f"  Produce ONE short borrower reply (1–3 sentences) that voices the "
        f"Persona Snapshot above.  Follow every ABSOLUTE RULE from the system "
        f"message.  Do not narrate, do not break character, do not exceed 50 "
        f"words.  Reply now:"
    )


def _cache_key(archetype: str, action_type: str) -> str:
    """
    Stable hash of (Archetype, action_type) — independent of the agent's
    exact wording.  This is the "Semantic Cache" the spec requests: two
    identical state-action contexts yield the same borrower line and skip
    the API call entirely.
    """
    raw = f"{archetype}::{action_type}".encode("utf-8")
    return hashlib.sha1(raw).hexdigest()[:16]


# ─────────────────────────────────────────────────────────────────────────────
# DATA GUARD — input truncation + invalid-output detection
# ─────────────────────────────────────────────────────────────────────────────


def _truncate_agent_text(text: str, max_words: int = MAX_AGENT_WORDS) -> str:
    """
    Hard cap the agent's message at `max_words` words before it is shown to
    the LLM.  The borrower never benefits from a 5000-word agent rant — and a
    runaway prompt would (a) blow the model's context window, (b) inflate
    NIM latency above the per-step budget, and (c) bloat outbound traffic
    on every cache miss.

    Words are a stable, language-agnostic proxy for tokens (≈ 1 token per
    English word, more for Hindi).  We preserve the *head* of the message
    because the classifier already extracted the intent from the first few
    sentences, and append an ellipsis so the LLM can see the cut happened.
    """
    if not text:
        return ""
    words = text.split()
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words]) + " ..."


def _is_invalid_reply(text: str) -> bool:
    """
    Reject LLM outputs that the env should never expose to the agent:

      • Empty / whitespace-only.
      • Refusals ("I'm an AI", "I cannot answer", etc.) — Llama-3.1-70B
        almost never produces these once the persona prompt is set, but
        guardrail tripping happens occasionally and we don't want the agent
        to learn that meta-string.
      • Template-placeholder leakage ("{loan_amount}") — usually a sign that
        a malformed fallback string somehow reached this point unrendered.

    Returns True iff the reply should be replaced by the sullen fallback.
    """
    if text is None:
        return True
    s = str(text).strip()
    if not s:
        return True
    if _RE_REFUSAL.search(s):
        return True
    if _RE_PLACEHOLDER_LEAK.search(s):
        return True
    return False


def _is_retryable_exception(exc: BaseException) -> bool:
    """
    Decide whether `exc` (raised by `openai.OpenAI(...).chat.completions.create`
    or its underlying `httpx` client) describes a transient problem worth
    retrying.

    True for: `RateLimitError` (HTTP 429), `APITimeoutError` (read timeout),
    `APIConnectionError` (DNS / TCP / TLS / connection-reset), and
    `InternalServerError` / generic `APIStatusError` whose `.status_code`
    is in `_RETRYABLE_HTTP_STATUSES`.

    False for: `AuthenticationError`, `BadRequestError`, `PermissionDeniedError`,
    `NotFoundError`, and any other 4xx that won't change on retry.

    The function imports `openai` lazily and degrades to a structural check
    so the module is still importable in environments where `openai` is not
    installed (e.g. minimal CI containers running unit tests).
    """
    # Structural fallback: anything carrying a `.status_code` we recognise.
    status = getattr(exc, "status_code", None)
    if isinstance(status, int) and status in _RETRYABLE_HTTP_STATUSES:
        return True

    try:
        from openai import (
            APIConnectionError,
            APITimeoutError,
            InternalServerError,
            RateLimitError,
        )
    except Exception:
        return False

    if isinstance(exc, (RateLimitError, APITimeoutError, APIConnectionError)):
        return True
    if isinstance(exc, InternalServerError):
        return True
    return False


# ─────────────────────────────────────────────────────────────────────────────
# RESPONSE GENERATOR
# ─────────────────────────────────────────────────────────────────────────────


class ResponseGenerator:
    """
    Hybrid-Dynamic borrower voice.

    Public API (the only thing adversary.py is allowed to call):

        gen = ResponseGenerator()                # idempotent — safe to share
        text = gen.generate(
            anger=...,             # float in [0,10] — current adversary anger
            trust=...,             # float in [0,10]
            fear=...,              # float in [0,10]
            action_type="empathize",
            agent_text="I understand this is hard.",
            profile=profile_dict,  # optional — for color in the prompt
            terminated=False,
            termination_reason="",
        ) -> str

    Behaviour:
      • If the NIM API is enabled AND a cached response for
        (Archetype, action_type) exists, return the cached string instantly.
      • If the NIM API is enabled AND no cache hit, query Llama-3.1-70B on
        NIM, store the answer in the cache, and return it.
      • If the NIM API is disabled OR raises, fall back to pick_template()
        from the legacy bank.  This keeps every existing test deterministic
        and guarantees the environment never blocks a training step on a
        network outage.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = NIM_BASE_URL,
        model: str = NIM_MODEL,
        timeout_s: float = NIM_TIMEOUT_S,
        enabled: Optional[bool] = None,
    ):
        self.model = model
        self.timeout_s = timeout_s
        self.base_url = base_url

        # Resolve the API key from (1) the explicit constructor argument,
        # (2) the dotenv-loaded environment.  Both routes go through the
        # same placeholder-aware sanitiser so an unfilled `.env` template
        # never produces a half-initialised client.
        env_key = _read_nvidia_api_key()
        explicit_key = (api_key.strip() if isinstance(api_key, str) else None) or None
        resolved_key: Optional[str] = explicit_key or env_key

        # Auto-enable when the user didn't say either way: the LLM voice is
        # ON when both the gating flag and a real key are present.
        if enabled is None:
            enabled = _nim_enabled()

        # If the operator *explicitly* asked for the LLM but the key is
        # absent / blank / a placeholder, fail loudly — this is the
        # behaviour the secrets-management spec asks for ("raise a clear,
        # descriptive ValueError explaining exactly which key is missing").
        # When `enabled=False` (or `NEG_LLM_ENABLED` unset) we silently
        # stay in template mode — that path is the documented hermetic
        # fallback used by every offline pytest run and CI environment,
        # and we MUST NOT regress it.
        if enabled and resolved_key is None:
            raise ValueError(
                "NVIDIA_API_KEY is not configured but the LLM voice is "
                "enabled (NEG_LLM_ENABLED=1).\n"
                "  • Copy `.env.example` to `.env` at the project root and "
                "fill in your real NVIDIA NIM key, or\n"
                "  • Export it in your shell:  export NVIDIA_API_KEY=nvapi-...\n"
                "  • Or pass it programmatically: "
                "ResponseGenerator(api_key='nvapi-...', enabled=True).\n"
                "Get a key at https://build.nvidia.com/."
            )

        self.enabled = bool(enabled and resolved_key is not None)

        self._client = None
        if self.enabled:
            try:
                from openai import OpenAI
                self._client = OpenAI(
                    base_url=self.base_url,
                    api_key=resolved_key,
                    timeout=self.timeout_s,
                )
            except Exception as exc:                       # pragma: no cover
                logger.warning("NIM client init failed (%s) — disabling.", exc)
                self.enabled = False

        self._cache: Dict[str, str] = {}
        self._cache_lock = threading.Lock()
        # Counters used by tests/test_edge_cases.py and voice_proof_of_life.py.
        # `retries` counts the number of *additional* attempts beyond the
        # first; `truncated_inputs` counts how many times a >MAX_AGENT_WORDS
        # message was clipped; `invalid_outputs` counts refusal/empty/leak
        # rejections from _is_invalid_reply.
        self._stats = {
            "hits": 0,
            "misses": 0,
            "api_errors": 0,
            "fallbacks": 0,
            "retries": 0,
            "truncated_inputs": 0,
            "invalid_outputs": 0,
        }

    # ── public API ──────────────────────────────────────────────────────

    def generate(
        self,
        anger: float,
        trust: float,
        fear: float,
        action_type: str,
        agent_text: str = "",
        profile: Optional[Dict[str, Any]] = None,
        terminated: bool = False,
        termination_reason: str = "",
        force_fresh: bool = False,
    ) -> str:
        # ── BOUNDARY GUARD: persona_snapshot already runs every emotion
        #    through `_finite()`, so NaN / Inf inputs cannot blow up the
        #    archetype lookup downstream.
        snapshot = persona_snapshot(anger, trust, fear, profile)
        key = _cache_key(snapshot["archetype"], action_type)

        # ── DATA GUARD #1: clip the agent's message before it ever reaches
        #    either the prompt builder or the cache key (cache key doesn't
        #    depend on agent_text, but truncation also caps prompt length
        #    and outbound bytes for the API call below).
        original_agent_text = agent_text or ""
        agent_text = _truncate_agent_text(original_agent_text, MAX_AGENT_WORDS)
        if agent_text != original_agent_text:
            self._stats["truncated_inputs"] += 1

        if not force_fresh:
            with self._cache_lock:
                cached = self._cache.get(key)
            if cached is not None:
                self._stats["hits"] += 1
                return cached

        self._stats["misses"] += 1

        # ── NETWORK GUARD: call NIM with retry-and-back-off, then validate
        #    the response with _is_invalid_reply().  Any non-retryable
        #    exception bubbles out of _call_nim_with_retries() and we fall
        #    through to the deterministic template bank.
        if self.enabled and self._client is not None:
            try:
                raw = self._call_nim_with_retries(
                    snapshot, action_type, agent_text,
                    profile, terminated, termination_reason,
                )
                cleaned = self._clean(raw)
                if cleaned and not _is_invalid_reply(cleaned):
                    with self._cache_lock:
                        self._cache[key] = cleaned
                    return cleaned
                # DATA GUARD #2: empty / refusal / placeholder-leak.
                self._stats["invalid_outputs"] += 1
                logger.warning(
                    "NIM produced an invalid reply (%r) — using sullen fallback.",
                    (cleaned or "")[:120],
                )
                sullen = self._clean(
                    self._sullen_fallback(snapshot, action_type, profile)
                )
                with self._cache_lock:
                    self._cache[key] = sullen
                return sullen
            except Exception as exc:
                # All retries exhausted (or non-retryable error).  We are
                # deliberately silent at WARNING level so a failing NIM
                # never spams the trainer log.
                logger.warning("NIM call failed (%s) — falling back.", exc)
                self._stats["api_errors"] += 1
                # fall through to template fallback

        # ── Fallback path — deterministic template bank.  Always cached so
        #    repeated misses don't keep paying the cost.
        self._stats["fallbacks"] += 1
        text = self._fallback(snapshot, action_type, profile)
        text = self._clean(text)
        with self._cache_lock:
            self._cache[key] = text
        return text

    # ── internals ───────────────────────────────────────────────────────

    def _call_nim(
        self,
        snapshot: Dict[str, Any],
        action_type: str,
        agent_text: str,
        profile: Optional[Dict[str, Any]],
        terminated: bool,
        termination_reason: str,
    ) -> str:
        user_prompt = _build_user_prompt(
            snapshot, action_type, agent_text,
            profile, terminated, termination_reason,
        )
        # Per-request hard timeout — overrides the client-level default and
        # ensures even a hung connection releases the worker eventually.
        resp = self._client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": _VOICE_RULES},
                {"role": "user",   "content": user_prompt},
            ],
            temperature=NIM_TEMPERATURE,
            top_p=NIM_TOP_P,
            max_tokens=NIM_MAX_TOKENS,
            timeout=self.timeout_s,
        )
        choice = resp.choices[0].message.content if resp.choices else ""
        return choice or ""

    def _call_nim_with_retries(
        self,
        snapshot: Dict[str, Any],
        action_type: str,
        agent_text: str,
        profile: Optional[Dict[str, Any]],
        terminated: bool,
        termination_reason: str,
    ) -> str:
        """
        NETWORK GUARD.

        Wraps `_call_nim()` with up to `NIM_MAX_RETRIES` total attempts and
        exponential back-off (`NIM_BACKOFF_BASE_S * 2**attempt`, capped at
        `NIM_BACKOFF_CAP_S`).  Only `_is_retryable_exception(exc)` triggers
        another attempt — every other failure is re-raised immediately so
        the caller can fall through to the template bank without burning
        the retry budget on a permanent error (e.g. invalid API key).

        Empirically the NIM endpoint produces transient 429s during the
        first few RL warm-up steps when many envs come online together,
        and very occasional 5xx during long training runs.  Three attempts
        with 0.5s / 1.0s / 2.0s backoff has been enough to absorb both
        without blocking forward progress for more than ~3.5s in the
        worst case.
        """
        last_exc: Optional[BaseException] = None
        for attempt in range(NIM_MAX_RETRIES):
            try:
                return self._call_nim(
                    snapshot, action_type, agent_text,
                    profile, terminated, termination_reason,
                )
            except BaseException as exc:  # noqa: BLE001  (we re-raise non-retryable below)
                last_exc = exc
                if not _is_retryable_exception(exc):
                    raise
                if attempt >= NIM_MAX_RETRIES - 1:
                    break  # exhausted — let the caller's except handle it
                backoff = min(
                    NIM_BACKOFF_BASE_S * (2 ** attempt),
                    NIM_BACKOFF_CAP_S,
                )
                self._stats["retries"] += 1
                logger.info(
                    "NIM call attempt %d/%d failed (%s); retrying in %.2fs",
                    attempt + 1, NIM_MAX_RETRIES, exc, backoff,
                )
                self._sleep(backoff)
        # All retries used up on a retryable error — surface the last one.
        assert last_exc is not None  # for type-checkers
        raise last_exc

    @staticmethod
    def _sleep(seconds: float) -> None:
        """Indirection point so tests can monkey-patch backoff to 0s."""
        time.sleep(seconds)

    @staticmethod
    def _clean(text: str) -> str:
        """Strip stage directions, prefixes, and excess whitespace."""
        if not text:
            return ""
        cleaned = text.strip().strip('"').strip("'").strip()
        # Drop a leading 'Borrower:' / 'Reply:' prefix if the model adds one.
        for prefix in ("Borrower:", "borrower:", "Reply:", "reply:"):
            if cleaned.startswith(prefix):
                cleaned = cleaned[len(prefix):].strip()
        # Collapse internal newlines so the borrower_msg fits on one line.
        cleaned = " ".join(cleaned.split())
        return cleaned

    def _fallback(
        self,
        snapshot: Dict[str, Any],
        action_type: str,
        profile: Optional[Dict[str, Any]],
    ) -> str:
        """Map the archetype back to the legacy zone vocabulary and reuse the
        deterministic template bank — guarantees a coherent borrower line
        even if NIM is unreachable."""
        # Legacy zones: calm / agitated / angry / panicked.
        if snapshot["fear"] > 7.0 or snapshot["anger"] > 8.0:
            zone = "panicked"
        elif snapshot["anger"] > 6.0:
            zone = "angry"
        elif snapshot["anger"] > 3.0:
            zone = "agitated"
        else:
            zone = "calm"
        legacy_state = {
            "anger": snapshot["anger"],
            "trust": snapshot["trust"],
            "fear":  snapshot["fear"],
            "zone":  zone,
            "turn":  0,
        }
        return pick_template(action_type, legacy_state, profile or {})

    def _sullen_fallback(
        self,
        snapshot: Dict[str, Any],
        action_type: str,
        profile: Optional[Dict[str, Any]],
    ) -> str:
        """
        DATA GUARD #2 fallback line — used when the LLM returns empty /
        refusal / malformed output (see `_is_invalid_reply`).

        Per the Edge-Case-Hardening spec the borrower must respond with a
        "sullen" line in this case — i.e. minimally engaged, emotionally
        flat, *not* a chirpy template — so the agent isn't accidentally
        rewarded for triggering an LLM refusal.  We pick from the
        agitated-zone template bank (curt, frustrated, but in-character)
        and forward the rendered string.
        """
        legacy_state = {
            "anger": snapshot.get("anger", 5.0),
            "trust": snapshot.get("trust", 2.0),
            "fear":  snapshot.get("fear",  3.0),
            # Force the agitated bank regardless of true zone — sullen, not
            # panicked or warm.
            "zone":  "agitated",
            "turn":  0,
        }
        return pick_template(action_type, legacy_state, profile or {})

    # ── instrumentation ────────────────────────────────────────────────

    def cache_size(self) -> int:
        with self._cache_lock:
            return len(self._cache)

    def stats(self) -> Dict[str, int]:
        return dict(self._stats)

    def clear_cache(self) -> None:
        with self._cache_lock:
            self._cache.clear()


# ─────────────────────────────────────────────────────────────────────────────
# Module-level singleton — adversary.py uses this so the cache persists
# across episodes within a process (which is the whole point of caching).
# ─────────────────────────────────────────────────────────────────────────────

_singleton: Optional[ResponseGenerator] = None
_singleton_lock = threading.Lock()


def get_response_generator() -> ResponseGenerator:
    """Lazy, thread-safe accessor for the process-wide ResponseGenerator."""
    global _singleton
    if _singleton is None:
        with _singleton_lock:
            if _singleton is None:
                _singleton = ResponseGenerator()
    return _singleton


def reset_response_generator() -> None:
    """Test helper — drop the singleton so a fresh one is built on next call."""
    global _singleton
    with _singleton_lock:
        _singleton = None


# ─────────────────────────────────────────────────────────────────────────────
# LEGACY DETERMINISTIC TEMPLATE BANK (unchanged — kept for back-compat)
# ─────────────────────────────────────────────────────────────────────────────

from typing import Dict, List, Any  # noqa: E402  (re-import for clarity)

# ─────────────────────────────────────────────────────────────────────────────
# TEMPLATES BANK
# Organized by [Signal Type][Emotion Zone]
# ─────────────────────────────────────────────────────────────────────────────

TEMPLATES: Dict[str, Dict[str, List[str]]] = {
    "empathize": {
        "calm": [
            "I appreciate you listening to me, {name}. It's been hard since the {reason}.",
            "It's rare to find someone at the bank who actually listens. My {loan_type} has been a lot of stress.",
            "Thank you for saying that. I really am trying my best to manage everything.",
            "I hope you mean that. I've been very worried about my {backstory_short} situation.",
            "It helps to feel understood. My priority is to resolve this {loan_type} as soon as I can.",
            "I'm glad we can talk like this. Most people just demand money without hearing the reason."
        ],
        "agitated": [
            "You say you understand, but the {loan_type} is still overdue and the pressure is real.",
            "Words are fine, but my situation with {reason} isn't going away.",
            "I hear you, but I'm still getting three calls a day. It doesn't feel like you understand.",
            "If you understood, you'd know why I'm asking for {demands_str}.",
            "I appreciate the words, but I need a real solution for my {loan_amount} loan.",
            "I'm trying to stay calm, but it's hard when {backstory_short} is constantly on my mind."
        ],
        "angry": [
            "Don't give me your scripted empathy. If you understood, you'd stop the harassment!",
            "You 'understand'? Then why are your agents calling my family about {reason}?",
            "Save your pity. Just tell me what can be done about the {loan_type} interest.",
            "You say this is tough? Try being in my shoes with {backstory_short} and no help!",
            "Stop pretending to care. You just want the {loan_amount} paid, that's all.",
            "I don't need your 'understanding.' I need you to listen to my {demands_str}!"
        ],
        "panicked": [
            "Please... if you understand, then help me. I'm so scared about the {loan_type}.",
            "I don't know what to do... my {reason} has ruined everything. Please don't be harsh.",
            "I'm at my limit. If you really see my struggle, please give me some time.",
            "Everything is falling apart... {backstory_short}... I can't breathe with this debt.",
            "Please... just don't tell my family. I'll do anything to fix this.",
            "I'm lost. I just need one person to actually help me with this {loan_amount}."
        ]
    },
    "clarify": {
        "calm": [
            "Let me explain. Since the {reason}, my monthly income hasn't been enough for {loan_type}.",
            "To be clear, I'm not refusing to pay. My {backstory_short} makes it complicated.",
            "The main issue is my {demands_str}. If we solve that, I can pay the {loan_amount}.",
            "I want to ensure you have the full picture of why the {loan_type} is delayed.",
            "The situation started when {backstory_short}. That's why I'm in this position.",
            "I'm trying to be transparent. The {reason} was unexpected for my family."
        ],
        "agitated": [
            "I've told you already, the {reason} changed everything! I can't pay the full EMI.",
            "Listen, my {backstory_short} is the reason the {loan_type} is stuck. It's simple.",
            "I'm telling you the truth. My {demands_str} are what I need addressed first.",
            "Can't you see? My {loan_amount} loan is overdue because of the {reason}!",
            "I'm trying to explain, but it feels like you're not listening to the {backstory_short} part.",
            "My situation is {reason}. I don't know how else to clarify it for the bank."
        ],
        "angry": [
            "How many times do I have to repeat about the {reason}? Are you even recording this?",
            "My {backstory_short} is why I can't pay! Stop asking the same questions!",
            "I'm being clear: I need {demands_str} or there is no way forward with this {loan_amount}.",
            "You keep pushing for the {loan_type} payment, but you ignore the {reason} entirely!",
            "I've clarified everything! My {backstory_short} is a fact, not an excuse!",
            "Read your notes! I've explained the {reason} fifty times already!"
        ],
        "panicked": [
            "I'm trying to tell you... {backstory_short}... please, it's just so hard.",
            "The {reason} happened and I lost control of the {loan_type}. I'm so sorry.",
            "Please, listen carefully... my {demands_str} are all I care about right now.",
            "I don't know how else to say it... {backstory_short}... I'm desperate.",
            "The {loan_amount} is too much for me now because of {reason}. Please believe me.",
            "I'm so confused... {backstory_short}... please tell me you understand now."
        ]
    },
    "probe_capacity": {
        "calm": [
            "You're asking about my capacity. Truthfully, {stated_capacity} is all I can manage for now.",
            "I've looked at my finances. With {reason}, I can only commit to {stated_capacity} per month.",
            "I can't pay the full {loan_amount}, but I can start with small amounts if you help.",
            "My current capacity is low because of {backstory_short}. Let's discuss a realistic plan.",
            "I want to be honest. I can pay {stated_capacity} now and more later when things improve.",
            "My priority is {demands_str}, but I can put some money toward the {loan_type} too."
        ],
        "agitated": [
            "Why are you asking about my income again? I told you, {stated_capacity} is my limit!",
            "I can't magic money out of nowhere. {backstory_short} has left me with nothing.",
            "I'm trying to stay afloat. Even {stated_capacity} is a stretch with the {reason}.",
            "You keep probing, but the answer remains {stated_capacity}. I can't do more.",
            "My {loan_amount} is a burden. Asking about my capacity won't change the {reason}!",
            "I don't have hidden money. {backstory_short} is my current reality."
        ],
        "angry": [
            "Are you calling me a liar? I said {stated_capacity} and I meant it!",
            "Stop digging into my life! The {reason} happened, and that's all you need to know!",
            "You want to know my capacity? It's zero if you keep harassing me about the {loan_type}!",
            "My {backstory_short} is none of your business beyond the {loan_amount} I owe!",
            "I'm sick of these questions. {stated_capacity} is the final word.",
            "Why should I tell you anything? Your bank didn't help when the {reason} hit!"
        ],
        "panicked": [
            "I don't have anything... {backstory_short}... please don't pressure me.",
            "Even {stated_capacity} is so hard to find right now. My {reason} was a disaster.",
            "I'm scared... I don't know how I'll even pay {stated_capacity}. Please help me.",
            "The {loan_type} is 150 days overdue and I'm drowning. {backstory_short} is too much.",
            "Please... I'm doing my best. {stated_capacity} is all I can possibly offer.",
            "I'm so worried... if you ask for more than {stated_capacity}, I won't be able to eat."
        ]
    },
    "offer_plan": {
        "calm": [
            "A plan for the {loan_type}? I'm listening. If it addresses my {demands_str}, I'm interested.",
            "I appreciate the offer. Let me see if the EMI fits my current {stated_capacity}.",
            "If the restructuring helps with the {reason} situation, I'll take it.",
            "Finally, a realistic approach. I want to settle the {loan_amount} properly.",
            "Let's discuss. I need the plan to be manageable given my {backstory_short}.",
            "Small EMIs would really help. {stated_capacity} is what I can realistically commit to."
        ],
        "agitated": [
            "Is the EMI high? Because my {reason} has made me very tight for cash.",
            "I need this plan to be final. No more changes for the {loan_type} later.",
            "Will this stop the calls? Because my {backstory_short} is already enough stress.",
            "I'll listen, but it has to be better than a simple demand for the {loan_amount}.",
            "Show me the numbers. My {stated_capacity} is fixed, so the plan must fit.",
            "I hope this plan includes my request for {demands_str}."
        ],
        "angry": [
            "Another 'plan'? Is this just another trick to get the {loan_amount} out of me?",
            "Unless this plan solves the {reason} issue, I don't want to hear it!",
            "You're making an offer now? After all the harassment about the {loan_type}?",
            "It better be a deep discount. My {backstory_short} deserves some consideration.",
            "I'm not signing anything until I'm sure my {demands_str} are met!",
            "Why should I trust your plan? You've been pushing me since day one!"
        ],
        "panicked": [
            "Please make it a small amount... I'm so scared of the {loan_type} debt.",
            "Will this fix everything? Will the {reason} pressure finally stop?",
            "I'll try to pay... just make the EMI very low. {stated_capacity} is all I have.",
            "Oh god, please let this work. My {backstory_short} situation is so fragile.",
            "I'm desperate for a way out of the {loan_amount}. Please, tell me the plan.",
            "Please don't take my home... I'll agree to a plan if it's manageable."
        ]
    },
    "seek_commitment": {
        "calm": [
            "I can commit to {stated_capacity} by next week. I give you my word.",
            "Yes, I will pay. My {reason} is resolving and I want the {loan_type} settled.",
            "I am a man of my word. I will ensure the {loan_amount} is paid as agreed.",
            "You have my commitment. I just need you to respect my {demands_str} in return.",
            "I'll pay. My {backstory_short} won't stand in the way of my word.",
            "Pukka promise. I'll make the first payment on the {loan_type} by Monday."
        ],
        "agitated": [
            "I'll try my best. But you have to understand the {reason} is still fresh.",
            "I'm promising, but don't hold the {loan_type} over my head if there's a day's delay.",
            "I'll give you {stated_capacity}, but I need that written confirmation we discussed.",
            "It's a promise, okay? Just stop the pressure about the {loan_amount} for a while.",
            "I'm committing, but my {backstory_short} is still a daily struggle.",
            "Fine, I agree. But please ensure no one visits my home for the {loan_type}."
        ],
        "angry": [
            "I'll pay when I can! My word is better than your bank's threats anyway!",
            "You want a commitment? Fix the {reason} interest and then we'll talk!",
            "I'm not promising anything until I see my {demands_str} in writing!",
            "Stop pushing for a 'promise'. My {backstory_short} is real, and life is unpredictable!",
            "I'll pay the {loan_amount} when I have it! That's my commitment, take it or leave it!",
            "You have my word that I won't forget how your bank treated me during {reason}!"
        ],
        "panicked": [
            "I promise! Just please don't tell my neighbors about the {loan_type}!",
            "I'll pay whatever I can... {stated_capacity}... just please help me.",
            "I give you my word, just stop the calls! My {reason} is too much to bear.",
            "I'll sign anything, just don't take the {loan_amount} to court. Please.",
            "Pukka waada. I'll pay. Please just give me one more chance.",
            "I'm so sorry... I'll commit to the {loan_type}. Please, I'm trying."
        ]
    },
    "warn_noncompliance": {
        "calm": [
            "You are mentioning legal action. I know my rights under the RBI guidelines.",
            "Threats won't help resolve the {loan_amount}. Let's be professional.",
            "I'm aware of the consequences. But my {reason} is a real hardship.",
            "If you go legal, it will just take longer for everyone. {loan_type} is best settled here.",
            "I don't appreciate the tone. My {backstory_short} doesn't make me a criminal.",
            "We should avoid escalation. My {demands_str} are very reasonable."
        ],
        "agitated": [
            "Why are you threatening me? My {reason} wasn't my fault!",
            "This is harassment! I will report this to the consumer forum immediately.",
            "You're pushing me too far. My {backstory_short} is stress enough without your threats.",
            "Legal action? For a {loan_amount} loan during a family crisis?",
            "I'm not scared of your 'consequences'. My {loan_type} will be paid when I'm able.",
            "Stop the threats. It won't make the money appear any faster."
        ],
        "angry": [
            "Go to court then! See if the judge cares about your bank more than my {reason}!",
            "I'm going to file an FIR for this harassment! You can't threaten me like this!",
            "You think I'm a thief? My {backstory_short} is a tragedy, and you're making it worse!",
            "Tell your senior I'm not paying a paisa if this attitude continues!",
            "I'll call the police myself! Your agents are visiting my neighborhood for {loan_type}!",
            "Shut up with the threats! I'll see you in the consumer forum, you wait and watch!"
        ],
        "panicked": [
            "Please... no legal action. I'll pay the {loan_amount}. Please don't do this.",
            "Oh god, police? No, please, my family... {backstory_short}... I'll fix it.",
            "I'm begging you, don't escalate. I'll find the money for {loan_type} somehow.",
            "I'll be ruined... {reason}... please, just one more week. Don't go to court.",
            "Please... I'm so scared. I'll agree to anything, just no legal notices.",
            "Don't visit my house... my {backstory_short}... I can't let them know."
        ]
    },
    "unknown": {
        "calm": [
            "I'm not sure what you're asking. Can we focus on the {loan_type}?",
            "I hear you, but my priority is resolving the {loan_amount} debt.",
            "Let's get back to the point. My {reason} is the main issue here.",
            "I'm a bit confused by that statement. How does it help with my {demands_str}?",
            "Can we stay focused? I want to settle this {loan_type} and move on.",
            "I didn't quite catch that. Let's talk about the next steps for my EMI."
        ]
    }
}

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def deterministic_hash(string: str) -> int:
    """A cross-run deterministic integer hash."""
    h = 0
    for char in string:
        h = (h * 31 + ord(char)) & 0xFFFFFFFF
    return h

def _inject_profile(template: str, profile: dict, state_dict: dict) -> str:
    """Replaces placeholders in the template with actual profile and state values."""
    backstory = profile.get("backstory", "")
    backstory_short = backstory[:80] + ("..." if len(backstory) > 80 else "")
    
    demands = profile.get("demands_stated", profile.get("demands", []))
    demands_str = ", ".join(demands) if demands else "no specific demands"
    
    return template.format(
        name=profile.get("name", "Customer"),
        age=profile.get("age", "N/A"),
        reason=profile.get("reason", "financial hardship"),
        loan_type=profile.get("loan_type", "loan"),
        loan_amount=profile.get("loan_amount", "total amount"),
        backstory_short=backstory_short,
        demands_str=demands_str,
        anger=round(state_dict.get("anger", 0), 1),
        trust=round(state_dict.get("trust", 0), 1),
        fear=round(state_dict.get("fear", 0), 1),
        stated_capacity=profile.get("stated_capacity", "a small amount")
    )

def pick_template(signal_type: str, state_dict: dict, profile: dict) -> str:
    """
    Selects a deterministic response template and injects dynamic values.
    Pure and deterministic function.
    """
    zone = state_dict.get("zone", "calm")
    
    # Fallback for unknown signals or zones
    if signal_type not in TEMPLATES:
        signal_type = "unknown"
    
    zone_templates = TEMPLATES[signal_type]
    if zone not in zone_templates:
        # Fallback to calm for the signal type if specific zone is missing
        if "calm" in zone_templates:
            zone = "calm"
        else:
            signal_type = "unknown"
            zone = "calm"
            zone_templates = TEMPLATES[signal_type]

    template_list = zone_templates[zone]
    
    # Deterministic selection
    turn = state_dict.get("turn", 0)
    p_id = profile.get("id", "P00")
    # key incorporates identifiers + turn count to ensure variety but determinism
    key = f"{signal_type}:{zone}:{p_id}:{turn}"
    index = deterministic_hash(key) % len(template_list)
    
    selected_template = template_list[index]
    
    return _inject_profile(selected_template, profile, state_dict)
