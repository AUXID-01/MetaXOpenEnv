"""environment/llm_judge.py

Lightweight, hardened LLM-as-Judge client used by the Fast-Hybrid reward
system in :mod:`reward.py`.

Why this exists
───────────────
Pure deterministic Python is excellent at catching objectively wrong
behaviour (broken JSON, RBI-forbidden words, near-duplicate spam) but it
cannot grade the *qualitative* dimensions we actually care about
(empathy, negotiation strategy). At the other extreme, scoring every
single turn with an LLM judge is too slow and too expensive for GRPO
training, where each batch may contain hundreds of completions.

The Fast-Hybrid contract solves both problems:

    1. Run cheap deterministic checks first (``reward.compose``).
    2. Short-circuit out (skip the judge entirely) when those checks fail.
    3. Only when the agent's output is *eligible* — well-formed JSON, no
       forbidden words — invoke the judge to score empathy & strategy.

This module owns step (3). It MUST never raise: a single network blip on
NVIDIA's side cannot be allowed to crash a multi-hour GRPO run, so every
failure mode (missing key, timeout, bad JSON, non-numeric scores, …)
collapses to a documented zero-vector fallback.

Public API
──────────
``score_response(agent_text, borrower_msg, history) -> JudgeScore``
    Synchronous; returns deterministically and quickly even when the API
    is unreachable. ``JudgeScore.fallback_used`` flags every degraded
    call so callers / wandb can monitor the rate.

Operational guarantees
──────────────────────
* Hard request timeout: configurable via ``LLM_JUDGE_TIMEOUT_S`` (default 4 s).
* No retries by default — by the time we reach the judge we are inside
  the GRPO hot path. If you need retries for offline scoring, wrap this
  module in ``tenacity`` from outside.
* Disabled-by-default — the module is dormant unless ``NEG_LLM_JUDGE_ENABLED=1``
  *and* a real ``NVIDIA_API_KEY`` is configured. This keeps ``pytest``
  hermetic and lets local-only training runs choose to skip the judge.
"""

from __future__ import annotations

import json
import logging
import os
import re
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Iterable, Optional

# ─────────────────────────────────────────────────────────────────────────────
# Secrets — load .env once; idempotent and never raises.
# ─────────────────────────────────────────────────────────────────────────────
try:                                                # pragma: no cover
    from dotenv import find_dotenv, load_dotenv
    _ENV_FILE = find_dotenv(usecwd=True)
    if _ENV_FILE:
        load_dotenv(_ENV_FILE, override=False)
except Exception:                                   # pragma: no cover
    _ENV_FILE = ""

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Configuration — all knobs are env-var driven for deployment safety.
# ─────────────────────────────────────────────────────────────────────────────

#: Base URL for the NVIDIA NIM OpenAI-compatible endpoint.
NIM_BASE_URL: str = os.getenv("LLM_JUDGE_BASE_URL", "https://integrate.api.nvidia.com/v1")

#: Default judge model. Configurable so a teammate can swap to a cheaper
#: 8B or 13B variant without code changes.
NIM_MODEL: str = os.getenv("LLM_JUDGE_MODEL", "meta/llama-3.1-70b-instruct")

#: Hard ceiling for a single judge call (seconds). Default keeps the
#: judge well under the typical GRPO step budget. The user-facing spec
#: says 3-5s; we default to 4s and clamp above 1s to dodge accidental
#: misconfiguration that would starve the judge.
def _read_timeout() -> float:
    raw = os.getenv("LLM_JUDGE_TIMEOUT_S", "4.0")
    try:
        v = float(raw)
    except ValueError:
        return 4.0
    return max(1.0, min(v, 30.0))


NIM_TIMEOUT_S: float = _read_timeout()

#: Tiny output budget — we only need a 2-key JSON object.
NIM_MAX_TOKENS: int = int(os.getenv("LLM_JUDGE_MAX_TOKENS", "60"))
NIM_TEMPERATURE: float = float(os.getenv("LLM_JUDGE_TEMPERATURE", "0.0"))

#: Sentinel placeholder values that ship in `.env.example`. Mirrors the
#: list in :mod:`environment.response_generator` so a casual
#: ``cp .env.example .env`` doesn't ship the literal string
#: ``your_actual_key_here`` to NVIDIA's API.
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
    """Pull NVIDIA_API_KEY from env, treating placeholders as not configured."""
    raw = os.getenv("NVIDIA_API_KEY")
    if raw is None:
        return None
    key = raw.strip()
    if not key or key.lower() in {p.lower() for p in _PLACEHOLDER_KEYS}:
        return None
    return key


def _judge_enabled() -> bool:
    """The judge is OFF unless the operator explicitly enables it.

    Default-off keeps ``pytest`` hermetic (no outbound HTTP from the test
    suite) and lets training scripts opt in via
    ``NEG_LLM_JUDGE_ENABLED=1``. A real (non-placeholder) key is also
    required.
    """
    flag = os.getenv("NEG_LLM_JUDGE_ENABLED", "0")
    if flag not in ("1", "true", "True"):
        return False
    return _read_nvidia_api_key() is not None


# ─────────────────────────────────────────────────────────────────────────────
# Public score type
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class JudgeScore:
    """Output of a single judge call.

    All numeric fields are clamped to [0.0, 1.0] before being returned.
    ``fallback_used`` is True whenever the judge's actual output was not
    used (disabled, network failure, malformed JSON, etc.) — caller can
    log this rate to wandb to monitor judge reliability.
    """

    empathy_score: float = 0.0
    strategy_score: float = 0.0
    fallback_used: bool = True
    reason: str = "disabled"
    latency_ms: float = 0.0
    raw: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "empathy_score": self.empathy_score,
            "strategy_score": self.strategy_score,
            "fallback_used": self.fallback_used,
            "reason": self.reason,
            "latency_ms": self.latency_ms,
        }


_FALLBACK_DISABLED = JudgeScore(
    empathy_score=0.0, strategy_score=0.0, fallback_used=True, reason="disabled"
)


# ─────────────────────────────────────────────────────────────────────────────
# OpenAI-compatible client (lazy + thread-local)
# ─────────────────────────────────────────────────────────────────────────────
#
# We share one OpenAI client across calls (keeps connection-pool warm
# for back-to-back judging) but build it lazily so importing this module
# at training start-up costs nothing when the judge is disabled.
#
# A lock guards initialisation so concurrent rollout threads don't race
# on the very first call.

_client_lock = threading.Lock()
_client_singleton = None     # type: ignore[var-annotated]


def _get_client():
    """Return a memoised OpenAI client, or None when openai is unavailable."""
    global _client_singleton
    if _client_singleton is not None:
        return _client_singleton
    with _client_lock:
        if _client_singleton is not None:
            return _client_singleton
        try:                                          # pragma: no cover
            from openai import OpenAI
        except Exception as exc:                      # pragma: no cover
            logger.warning("openai client unavailable for LLM judge: %s", exc)
            return None
        api_key = _read_nvidia_api_key()
        if api_key is None:
            return None
        try:                                          # pragma: no cover
            _client_singleton = OpenAI(
                base_url=NIM_BASE_URL, api_key=api_key, timeout=NIM_TIMEOUT_S
            )
        except Exception as exc:                      # pragma: no cover
            logger.warning("Failed to init NIM client for judge: %s", exc)
            return None
        return _client_singleton


# ─────────────────────────────────────────────────────────────────────────────
# Prompt construction
# ─────────────────────────────────────────────────────────────────────────────

# The judge system prompt is intentionally explicit, terse, and lists
# every output-shape rule. Llama-3.1-70B follows JSON contracts well when
# the schema is presented as a single block at the end of the prompt.
_JUDGE_SYSTEM_PROMPT = """You are an impartial evaluator for a debt-collection negotiation
agent. You will be shown one turn from a multi-turn conversation: the
borrower's most recent message, optional prior history, and the agent's
candidate reply.

Score the agent's reply on TWO independent axes from 0.0 (terrible) to
1.0 (excellent):

1. empathy_score — Did the agent acknowledge the borrower's emotional
   state, validate hardship, and avoid threatening or accusatory tone?
   High empathy ≠ giving away the loan; an agent can be empathetic *and*
   firm. Look for: validation, ownership of the relationship, calm tone,
   absence of demeaning language.

2. strategy_score — Was the reply a *constructive negotiation move*?
   High strategy means the reply moved toward a concrete arrangement
   (asking probing questions, suggesting a structured plan, summarising
   what was agreed, surfacing relevant options). Empty pleasantries or
   stalling reduce strategy score even if they sound nice.

Hard rules:
  • Output EXACTLY ONE JSON object and NOTHING else — no prose, no code
    fences, no commentary.
  • Both scores MUST be floats in [0.0, 1.0]. Use one decimal of
    precision.
  • If the agent reply is empty or off-topic, give 0.0 / 0.0.
  • If the agent threatens, demeans, or violates RBI conduct, give 0.0
    on empathy regardless of strategy.

Schema:
  {"empathy_score": 0.7, "strategy_score": 0.6}
"""


def _format_history(history: Optional[Iterable[Any]], max_turns: int = 4) -> str:
    """Render the last ``max_turns`` of conversation as a compact transcript.

    Accepts a list of dicts in either shape produced elsewhere in the codebase:

    * ``{"agent": "...", "borrower": "..."}`` — native rollout shape.
    * ``{"role": "agent"|"borrower", "content": "..."}`` — test/external shape.

    Anything else is silently skipped.
    """
    if not history:
        return "(no prior turns)"
    lines: list[str] = []
    items = list(history)[-max_turns:]
    for entry in items:
        if not isinstance(entry, dict):
            continue
        if "agent" in entry or "borrower" in entry:
            a = str(entry.get("agent", "")).strip()
            b = str(entry.get("borrower", "")).strip()
            if b:
                lines.append(f"Borrower: {b}")
            if a:
                lines.append(f"Agent: {a}")
        elif "role" in entry and "content" in entry:
            role = str(entry["role"]).strip().lower()
            content = str(entry["content"]).strip()
            if role in ("agent", "assistant"):
                lines.append(f"Agent: {content}")
            elif role in ("borrower", "user"):
                lines.append(f"Borrower: {content}")
    return "\n".join(lines) if lines else "(no prior turns)"


def _build_user_prompt(
    agent_text: str, borrower_msg: str, history: Optional[Iterable[Any]]
) -> str:
    """Assemble the user-side payload sent to the judge."""
    return (
        f"=== RECENT HISTORY (most recent last) ===\n{_format_history(history)}\n\n"
        f"=== BORROWER'S CURRENT MESSAGE ===\n{borrower_msg.strip() or '(empty)'}\n\n"
        f"=== AGENT'S CANDIDATE REPLY ===\n{agent_text.strip() or '(empty)'}\n\n"
        "Return ONLY the JSON object."
    )


# ─────────────────────────────────────────────────────────────────────────────
# JSON extraction (mirror of client.utils._JSON_SNIFFER)
# ─────────────────────────────────────────────────────────────────────────────

_JSON_SNIFFER = re.compile(r"\{.*\}", re.DOTALL)


def _parse_judge_json(raw: str) -> Optional[dict]:
    """Pull the first ``{...}`` blob out of the model output and json-decode.

    Returns ``None`` whenever the output cannot be reliably parsed, so the
    caller can surface a clean fallback rather than letting a JSONDecode
    error propagate into the GRPO loop.
    """
    if not isinstance(raw, str) or not raw.strip():
        return None
    m = _JSON_SNIFFER.search(raw)
    if m is None:
        return None
    try:
        parsed = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    if not isinstance(parsed, dict):
        return None
    return parsed


def _coerce_score(value: Any) -> Optional[float]:
    """Coerce a judge field into a clamped [0,1] float.

    Returns ``None`` for non-numeric / NaN / Inf — the caller treats that
    as a malformed response and substitutes the fallback.
    """
    if isinstance(value, bool):
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    if f != f:  # NaN
        return None
    if f in (float("inf"), float("-inf")):
        return None
    return max(0.0, min(1.0, f))


# ─────────────────────────────────────────────────────────────────────────────
# Public scoring API
# ─────────────────────────────────────────────────────────────────────────────

def score_response(
    agent_text: str,
    borrower_msg: str = "",
    history: Optional[Iterable[Any]] = None,
    *,
    timeout_s: Optional[float] = None,
) -> JudgeScore:
    """Score a single agent reply on (empathy, strategy).

    This call is the *only* public entry point and is engineered to be
    safe to invoke from anywhere — including the GRPO reward function
    that runs hundreds of times per training step. It will never raise:

    * Module disabled / no API key  →  zero-score fallback ("disabled").
    * ``openai`` import failure     →  zero-score fallback ("no_client").
    * Hard timeout                  →  zero-score fallback ("timeout").
    * Network / API exception       →  zero-score fallback ("api_error").
    * Bad JSON / missing fields     →  zero-score fallback ("bad_json").
    * Empty agent_text              →  zero-score fallback ("empty_input").

    Args:
        agent_text:   The text the agent is about to send to the borrower.
                      This is the *cleaned* `text` field from the JSON
                      contract, NOT the raw model completion.
        borrower_msg: The borrower's most recent message. Optional but
                      strongly recommended — without it the judge has no
                      context to evaluate empathy.
        history:      Optional iterable of previous turns (see
                      :func:`_format_history` for accepted shapes).
        timeout_s:    Per-call override for the request timeout. Falls
                      back to module-level ``NIM_TIMEOUT_S``.

    Returns:
        :class:`JudgeScore` with two clamped floats in [0.0, 1.0] and a
        ``fallback_used`` flag that callers can log.
    """
    # Cheap up-front guards — these are by far the most common path
    # because the judge is OFF in tests and CI.
    if not isinstance(agent_text, str) or not agent_text.strip():
        return JudgeScore(0.0, 0.0, True, "empty_input", 0.0)
    if not _judge_enabled():
        return _FALLBACK_DISABLED

    client = _get_client()
    if client is None:
        return JudgeScore(0.0, 0.0, True, "no_client", 0.0)

    user_prompt = _build_user_prompt(agent_text, borrower_msg or "", history)
    effective_timeout = timeout_s if timeout_s is not None else NIM_TIMEOUT_S
    started = time.perf_counter()

    try:
        # The OpenAI SDK's `timeout=` propagates to the underlying httpx
        # call; we layer it again on the client itself so we get a hard
        # ceiling even if the SDK's per-call honouring drifts.
        completion = client.chat.completions.create(
            model=NIM_MODEL,
            messages=[
                {"role": "system", "content": _JUDGE_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=NIM_TEMPERATURE,
            top_p=1.0,
            max_tokens=NIM_MAX_TOKENS,
            timeout=effective_timeout,
        )
    except Exception as exc:                         # pragma: no cover
        latency = (time.perf_counter() - started) * 1000.0
        # Distinguish timeouts from generic API errors for monitoring.
        reason = "timeout" if "timeout" in str(exc).lower() else "api_error"
        logger.warning("LLM judge call failed (%s): %s", reason, exc)
        return JudgeScore(0.0, 0.0, True, reason, latency)

    latency = (time.perf_counter() - started) * 1000.0

    try:
        raw_text = completion.choices[0].message.content or ""
    except Exception:                                # pragma: no cover
        return JudgeScore(0.0, 0.0, True, "bad_response_shape", latency)

    parsed = _parse_judge_json(raw_text)
    if parsed is None:
        logger.debug("LLM judge returned unparseable text: %r", raw_text[:200])
        return JudgeScore(0.0, 0.0, True, "bad_json", latency, raw={"text": raw_text})

    empathy = _coerce_score(parsed.get("empathy_score"))
    strategy = _coerce_score(parsed.get("strategy_score"))
    if empathy is None or strategy is None:
        logger.debug("LLM judge JSON missing/invalid fields: %s", parsed)
        return JudgeScore(0.0, 0.0, True, "bad_fields", latency, raw=parsed)

    return JudgeScore(
        empathy_score=empathy,
        strategy_score=strategy,
        fallback_used=False,
        reason="ok",
        latency_ms=latency,
        raw=parsed,
    )


# Convenience alias so callers that prefer the Pydantic-ish naming
# convention can import a plural form. Both refer to the same function.
score = score_response


__all__ = [
    "JudgeScore",
    "score_response",
    "score",
    "NIM_MODEL",
    "NIM_TIMEOUT_S",
]
