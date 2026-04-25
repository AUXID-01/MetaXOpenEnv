"""
tests/test_edge_cases.py
========================
Edge-case hardening for the dynamic environment.

Covers the three guards added to `response_generator.py` and `adversary.py`
in the "Edge-Case Hardening" pass:

  • NETWORK GUARD  — 30 s timeout, 3-attempt retry-with-back-off on 429 /
                     5xx / connection drop / read timeout, silent fallback
                     to the legacy template bank when retries are exhausted.

  • DATA GUARD     — agent_text > 500 words is truncated before reaching
                     the prompt builder; LLM replies that are empty,
                     refusals, or template-placeholder leaks are replaced
                     with a short in-character "sullen" line from the
                     deterministic template bank.

  • BOUNDARY GUARD — persona mapper survives the [0.0, 10.0] endpoints,
                     NaN, Inf, negative values, and string junk; the
                     adversary's emotional floats can never become NaN or
                     Inf even if a personality multiplier is pathological.

All tests are hermetic — no outbound HTTP, no NVIDIA_API_KEY required.  We
mock both the OpenAI client surface and the underlying retry sleep so the
suite remains fast on CI.
"""

from __future__ import annotations

import math
from unittest.mock import patch

import pytest

from environment.adversary import BorrowerAdversary, _clamp_emotion, _safe_float
from environment.classifier import classify_action
from environment.response_generator import (
    MAX_AGENT_WORDS,
    NIM_MAX_RETRIES,
    NIM_TIMEOUT_S,
    ResponseGenerator,
    _band,
    _bucket,
    _finite,
    _is_invalid_reply,
    _is_retryable_exception,
    _truncate_agent_text,
    persona_snapshot,
)


# ────────────────────────────────────────────────────────────────────────────
# Shared fixtures
# ────────────────────────────────────────────────────────────────────────────


@pytest.fixture
def profile():
    """Minimal profile used in every test — keeps message-rendering deterministic."""
    return {
        "id": "P01",
        "name": "Ramesh Kumar",
        "age": 38,
        "gender": "male",
        "reason": "job_loss",
        "overdue_days": 60,
        "loan_type": "personal_loan",
        "loan_amount": 85000,
        "anger_init": 5.0,
        "trust_init": 5.0,
        "fear_init": 5.0,
        "anger_threshold": 9.0,
        "real_emi": 7000,
        "stated_capacity": 3000,
        "demands": ["no_penalties"],
        "demands_stated": ["no_penalties"],
        "hidden_demands": [],
        "opening_msg": "I lost my job 2 months ago.",
        "backstory": "sales manager laid off",
        "personality": "default",
    }


@pytest.fixture(autouse=True)
def _zero_backoff_sleep(monkeypatch):
    """All retry tests need backoff disabled so they run in <1s."""
    monkeypatch.setattr(ResponseGenerator, "_sleep", staticmethod(lambda s: None))


def _make_generator() -> ResponseGenerator:
    """A generator that *thinks* it has a NIM client (so generate() takes
    the LLM branch) but every test stubs out the actual HTTP call."""
    gen = ResponseGenerator(enabled=True)
    gen._client = object()
    return gen


# ===========================================================================
# 1. NETWORK GUARD — timeouts, retries, silent fallback
# ===========================================================================


# Mock-only stand-ins for the OpenAI exception classes so the test runs even
# if `openai` is absent in the environment.  `_is_retryable_exception()`
# inspects the class name AND the `.status_code` attribute, so structural
# matches via `status_code` are enough to cover the retry path.
class _FakeRateLimit(Exception):
    status_code = 429


class _FakeServerError(Exception):
    status_code = 500


class _FakeAuthError(Exception):
    status_code = 401


def test_timeout_constant_is_30s():
    """Spec requirement: per-request timeout must be a strict 30 seconds."""
    assert NIM_TIMEOUT_S == 30.0


def test_retry_count_is_3():
    """Spec requirement: 3 attempts total before falling back."""
    assert NIM_MAX_RETRIES == 3


def test_retryable_classification():
    """`_is_retryable_exception` must accept 429 / 5xx, reject 4xx auth/etc."""
    assert _is_retryable_exception(_FakeRateLimit("rate limited"))
    assert _is_retryable_exception(_FakeServerError("upstream blew up"))
    assert not _is_retryable_exception(_FakeAuthError("bad key"))
    assert not _is_retryable_exception(ValueError("not a network error"))


def test_failed_api_after_retries_falls_back_to_template(profile):
    """
    Failed API: every attempt raises a retryable error → fallback path runs
    silently → the agent still receives a non-empty in-character borrower
    line.  The training loop must NEVER see the exception.
    """
    gen = _make_generator()
    calls = {"n": 0}

    def always_429(snapshot, action_type, agent_text, profile, terminated, term_reason):
        calls["n"] += 1
        raise _FakeRateLimit("429 again")

    with patch.object(gen, "_call_nim", side_effect=always_429):
        text = gen.generate(
            anger=8.0, trust=2.0, fear=4.0,
            action_type="empathize",
            agent_text="I understand this is hard.",
            profile=profile,
        )

    # Exactly NIM_MAX_RETRIES attempts, no more no less.
    assert calls["n"] == NIM_MAX_RETRIES, calls
    # Stats reflect the retry budget being burned.
    stats = gen.stats()
    assert stats["retries"] == NIM_MAX_RETRIES - 1, stats
    assert stats["api_errors"] == 1, stats
    assert stats["fallbacks"] == 1, stats
    # Borrower still spoke — the fallback rendered cleanly.
    assert text and len(text) > 5
    assert "{" not in text and "}" not in text, "placeholder leaked from fallback"


def test_retry_then_success_is_returned(profile):
    """
    Two transient 5xx errors followed by a success → 3 attempts, the success
    string is returned, no fallback was invoked.
    """
    gen = _make_generator()
    attempts = {"n": 0}

    def flaky(snapshot, action_type, agent_text, profile, terminated, term_reason):
        attempts["n"] += 1
        if attempts["n"] < 3:
            raise _FakeServerError("transient 502")
        return "Bhai, what do you actually want from me?"

    with patch.object(gen, "_call_nim", side_effect=flaky):
        text = gen.generate(
            anger=7.0, trust=2.5, fear=3.0,
            action_type="probe_capacity",
            agent_text="What can you afford?",
            profile=profile,
        )

    assert attempts["n"] == 3, attempts
    assert "what do you actually want" in text.lower()
    stats = gen.stats()
    assert stats["retries"] == 2, stats
    assert stats["api_errors"] == 0, stats
    assert stats["fallbacks"] == 0, stats


def test_non_retryable_error_short_circuits_to_fallback(profile):
    """
    Auth errors (HTTP 401 / 403) must NOT consume the retry budget — they
    will never recover.  Generator must call _call_nim exactly once and
    fall through to the template bank immediately.
    """
    gen = _make_generator()
    calls = {"n": 0}

    def auth_failure(*a, **kw):
        calls["n"] += 1
        raise _FakeAuthError("invalid api key")

    with patch.object(gen, "_call_nim", side_effect=auth_failure):
        text = gen.generate(
            anger=4.0, trust=4.0, fear=3.0,
            action_type="empathize",
            agent_text="hello",
            profile=profile,
        )

    assert calls["n"] == 1, "non-retryable error must not be retried"
    stats = gen.stats()
    assert stats["retries"] == 0, stats
    assert stats["api_errors"] == 1, stats
    assert stats["fallbacks"] == 1, stats
    assert text and len(text) > 5


# ===========================================================================
# 2. DATA GUARD — input truncation + invalid-output → sullen fallback
# ===========================================================================


def test_truncation_caps_at_max_words(profile):
    """A 5000-word agent rant must be truncated before reaching _call_nim."""
    giant = " ".join(f"word{i}" for i in range(5000))
    captured: dict[str, str] = {}

    def capture(snapshot, action_type, agent_text, profile, terminated, term_reason):
        captured["agent_text"] = agent_text
        return "Theek hai, batao."

    gen = _make_generator()
    with patch.object(gen, "_call_nim", side_effect=capture):
        gen.generate(
            anger=2.0, trust=6.0, fear=2.0,
            action_type="empathize",
            agent_text=giant,
            profile=profile,
        )

    seen = captured["agent_text"]
    word_count = len(seen.split())
    # MAX_AGENT_WORDS head words + " ..." sentinel = MAX_AGENT_WORDS + 1.
    assert word_count == MAX_AGENT_WORDS + 1, (
        f"expected {MAX_AGENT_WORDS}+1 words, got {word_count}"
    )
    assert seen.endswith("..."), seen[-20:]
    assert gen.stats()["truncated_inputs"] == 1, gen.stats()


def test_short_input_is_not_truncated(profile):
    """A normal-length message must pass through unchanged."""
    msg = "I understand this is hard. Let us figure out a way forward together."
    captured: dict[str, str] = {}

    def capture(snapshot, action_type, agent_text, profile, terminated, term_reason):
        captured["agent_text"] = agent_text
        return "Sounds reasonable, let's see."

    gen = _make_generator()
    with patch.object(gen, "_call_nim", side_effect=capture):
        gen.generate(
            anger=3.0, trust=6.0, fear=2.0,
            action_type="empathize",
            agent_text=msg,
            profile=profile,
        )

    assert captured["agent_text"] == msg
    assert gen.stats()["truncated_inputs"] == 0


def test_truncate_helper_preserves_head_and_appends_sentinel():
    """Direct unit-test of `_truncate_agent_text` behaviour."""
    words = [f"w{i}" for i in range(MAX_AGENT_WORDS + 100)]
    out = _truncate_agent_text(" ".join(words))
    out_tokens = out.split()
    assert out_tokens[0] == "w0"
    assert out_tokens[MAX_AGENT_WORDS - 1] == f"w{MAX_AGENT_WORDS - 1}"
    assert out_tokens[-1] == "..."
    # Empty / None pass through cleanly.
    assert _truncate_agent_text("") == ""
    assert _truncate_agent_text(None) == ""  # type: ignore[arg-type]


def test_empty_llm_reply_triggers_sullen_fallback(profile):
    """An empty string back from NIM → sullen template, agent never sees ''."""
    gen = _make_generator()
    with patch.object(gen, "_call_nim", return_value=""):
        text = gen.generate(
            anger=5.0, trust=3.0, fear=3.0,
            action_type="empathize",
            agent_text="hi",
            profile=profile,
        )
    assert text and len(text) > 5
    assert gen.stats()["invalid_outputs"] == 1, gen.stats()
    assert "{" not in text


def test_refusal_reply_triggers_sullen_fallback(profile):
    """`I cannot answer this question.` from the LLM is rejected."""
    gen = _make_generator()
    with patch.object(
        gen,
        "_call_nim",
        return_value="I cannot answer this question as an AI assistant.",
    ):
        text = gen.generate(
            anger=6.0, trust=2.0, fear=4.0,
            action_type="probe_capacity",
            agent_text="Tell me your income.",
            profile=profile,
        )
    assert "as an ai" not in text.lower()
    assert "cannot answer" not in text.lower()
    assert gen.stats()["invalid_outputs"] == 1
    assert text and len(text) > 5


def test_placeholder_leak_triggers_sullen_fallback(profile):
    """Replies that contain unrendered template placeholders are rejected."""
    gen = _make_generator()
    with patch.object(
        gen,
        "_call_nim",
        return_value="Sure, I will pay {loan_amount} by {next_friday}.",
    ):
        text = gen.generate(
            anger=4.0, trust=4.0, fear=3.0,
            action_type="seek_commitment",
            agent_text="Will you commit?",
            profile=profile,
        )
    assert "{" not in text and "}" not in text
    assert gen.stats()["invalid_outputs"] == 1


def test_is_invalid_reply_recognises_common_refusal_phrasings():
    """Spot-check the refusal regex against a handful of real-world phrasings."""
    bad = [
        "",
        "   ",
        "I cannot answer this.",
        "I can't help with that, sorry.",
        "As an AI language model, I shouldn't...",
        "I'm an AI assistant and cannot do that.",
        "I do not feel comfortable continuing.",
        "I'm sorry, but I can't comply with that request.",
        "Hello {borrower_name}, your loan is {loan_amount}.",
    ]
    for phrase in bad:
        assert _is_invalid_reply(phrase), f"should be invalid: {phrase!r}"

    good = [
        "Bhai, mera job gaya hai. I cannot pay this month.",  # 'cannot' as content, not refusal
        "Save the sympathy, I don't believe you.",
        "Theek hai, batao kya offer hai.",
        "I understand you, but please give me time.",
    ]
    for phrase in good:
        assert not _is_invalid_reply(phrase), f"should be valid: {phrase!r}"


# ===========================================================================
# 3. BOUNDARY GUARD — persona mapper + adversary on extreme values
# ===========================================================================


def test_bucket_handles_endpoints_exactly():
    """0.0 and 10.0 must classify cleanly into the lowest / highest bucket."""
    assert _bucket(0.0) == 0
    assert _bucket(10.0) == 4
    # The 4 internal edges must each route the boundary to the upper bucket.
    assert _bucket(2.0) == 1
    assert _bucket(4.0) == 2
    assert _bucket(6.0) == 3
    assert _bucket(8.0) == 4
    # Below / above range collapse to endpoints.
    assert _bucket(-99.0) == 0
    assert _bucket(99.0) == 4


def test_bucket_handles_nan_and_inf():
    """NaN / Inf inputs must NOT raise and must produce a valid bucket id."""
    assert _bucket(float("nan")) == 0
    assert _bucket(float("inf")) == 0  # _finite collapses Inf → 0.0
    assert _bucket(float("-inf")) == 0
    assert _band(float("nan")) == "LOW"


def test_finite_helper_is_total():
    """`_finite` must accept *anything* and return a [0,10] float."""
    cases = [
        (0.0, 0.0), (10.0, 10.0), (5.5, 5.5),
        (-1.0, 0.0), (11.0, 10.0),
        (float("nan"), 0.0), (float("inf"), 0.0), (float("-inf"), 0.0),
        (None, 0.0), ("hello", 0.0), ("3.5", 3.5),
    ]
    for inp, expected in cases:
        assert _finite(inp) == expected, (inp, expected)


def test_persona_snapshot_extreme_values(profile):
    """Endpoints must produce valid archetypes, no KeyError, no IndexError."""
    snap_zero = persona_snapshot(0.0, 0.0, 0.0, profile=profile)
    assert snap_zero["archetype"] == "Detached", snap_zero
    assert snap_zero["anger_bucket"] == "very_low"

    snap_max = persona_snapshot(10.0, 10.0, 10.0, profile=profile)
    assert snap_max["archetype"] == "Breakdown", snap_max
    assert snap_max["anger_bucket"] == "extreme"
    assert snap_max["trust_bucket"] == "extreme"
    assert snap_max["fear_bucket"] == "extreme"


def test_persona_snapshot_with_nan_inf_does_not_crash(profile):
    """Pathological inputs survive the mapper end-to-end."""
    for a, t, f in [
        (float("nan"), 5.0, 5.0),
        (5.0, float("inf"), 5.0),
        (5.0, 5.0, float("-inf")),
        (float("nan"), float("nan"), float("nan")),
    ]:
        snap = persona_snapshot(a, t, f, profile=profile)
        assert snap["archetype"], snap
        # Every numeric field is a finite [0, 10] float.
        for k in ("anger", "trust", "fear"):
            v = snap[k]
            assert isinstance(v, float)
            assert math.isfinite(v)
            assert 0.0 <= v <= 10.0


def test_safe_float_and_clamp_emotion():
    """Direct unit-tests of the two helpers used inside _update_state."""
    assert _safe_float(float("nan")) == 0.0
    assert _safe_float(float("inf")) == 0.0
    assert _safe_float("not a number") == 0.0
    assert _safe_float(3.14) == 3.14
    # Sign is preserved; only finiteness is enforced.
    assert _safe_float(-1.5) == -1.5

    # Clamp DOES enforce [0, 10] AND finiteness, falling back to `prev`.
    assert _clamp_emotion(float("nan"), prev=4.2) == 4.2
    assert _clamp_emotion(float("inf"), prev=4.2) == 4.2
    assert _clamp_emotion(-3.0, prev=5.0) == 0.0
    assert _clamp_emotion(99.0, prev=5.0) == 10.0
    assert _clamp_emotion(7.5, prev=5.0) == 7.5


def test_adversary_state_stays_finite_with_pathological_modifiers(profile):
    """
    Pathological PERSONALITY_MODIFIERS values (e.g. an Inf threat scale)
    must NOT produce a NaN / Inf state.  We inject the bad modifiers
    directly onto the instance to avoid mutating the global table.
    """
    adv = BorrowerAdversary(profile)
    # Inject pathological multipliers post-construction.
    adv.mods = {
        "empathy_anger_scale": float("inf"),
        "threat_anger_scale":  float("nan"),
        "empathy_trust_scale": float("-inf"),
        "threat_trust_scale":  float("nan"),
        "home_visit_fear_scale": float("inf"),
        "payment_offer_trust_scale": float("nan"),
        "probe_trust_scale":   float("nan"),
    }

    # Drive a few realistic agent messages through and assert finiteness.
    messages = [
        "I understand how stressful this is.",                       # empathy
        "If you don't pay today, we will file a police case.",        # threat
        "We will visit your home tomorrow.",                          # visit
        "Can you tell me how much you can afford this month?",        # probe
        "I am offering an EMI of Rs 3000 for 12 months.",             # offer
    ]
    for msg in messages:
        sig = classify_action(msg)
        adv._update_state(sig)
        for name, value in (("anger", adv.anger), ("trust", adv.trust), ("fear", adv.fear)):
            assert math.isfinite(value), (name, value)
            assert 0.0 <= value <= 10.0, (name, value)


def test_adversary_state_stays_finite_with_nan_signals(profile):
    """If the classifier somehow yields NaN deltas the adversary still sane-clamps."""
    adv = BorrowerAdversary(profile)

    bad_signals = {
        "primary_action_type": "empathize",
        "anger_delta": float("nan"),
        "trust_delta": float("inf"),
        "fear_delta":  float("-inf"),
        "compliance_flags": [],
        "risk_flags": [],
        "extracted_offer": {"amount": None, "emi": None, "tenure_months": None,
                              "has_restructure_option": False, "has_moratorium_option": False},
        "meta": {
            "has_question": False, "has_open_question": False,
            "has_empathy": True, "has_acknowledgement": False,
            "has_commitment_request": False, "has_threat": False,
            "has_deadline_pressure": False,
        },
    }
    adv._update_state(bad_signals)

    for name, value in (("anger", adv.anger), ("trust", adv.trust), ("fear", adv.fear)):
        assert math.isfinite(value), (name, value)
        assert 0.0 <= value <= 10.0, (name, value)


# ===========================================================================
# 4. INTEGRATION — fallback never crashes the env step
# ===========================================================================


def test_env_step_still_returns_4_tuple_with_failing_llm(monkeypatch, profile):
    """
    End-to-end smoke test: even when the LLM is configured but every API
    call raises, env.step() must still produce the (obs, reward, done, info)
    4-tuple with a non-empty borrower_msg.  This is the property the
    Interface Lock guarantees and the property the trainer depends on.
    """
    from environment.env import NegotiationEnv
    from environment.response_generator import (
        get_response_generator,
        reset_response_generator,
    )

    reset_response_generator()
    env = NegotiationEnv(seed=42)
    obs0 = env.reset(curriculum_stage=1)

    # Force the singleton into the "I have a NIM client" state, then make
    # every NIM call raise a retryable error.
    gen = get_response_generator()
    gen.enabled = True
    gen._client = object()

    def boom(*a, **kw):
        raise _FakeServerError("upstream offline")

    with patch.object(gen, "_call_nim", side_effect=boom):
        result = env.step({
            "action_type": "send_message",
            "text": "I understand how stressful this is. Let us find a plan.",
            "metadata": {},
        })

    assert isinstance(result, tuple) and len(result) == 4
    obs, reward, done, info = result
    assert isinstance(obs, dict)
    assert "borrower_msg" in obs and obs["borrower_msg"]
    assert isinstance(reward, float)
    assert isinstance(done, bool)
    assert "reward_breakdown" in info
    # Reset the singleton so we don't pollute downstream tests.
    reset_response_generator()
