"""
tests/test_dynamic_voice.py
===========================
Verifies the Hybrid-Dynamic borrower voice (NVIDIA NIM / Llama-3.1-70B)
without coupling the test suite to live network availability.

Five cases (per the Hybrid-Dynamic Voice spec):

  1. FURIOUS    Anger=10, Trust=0  → response is hostile / dismissive.
  2. ZEN        Anger=0,  Trust=10 → response is cooperative / warm.
  3. BOUNDARY   Anger=6.0          → no crash; archetype is the elevated
                                     band (>= 6.0 rule), not the moderate band.
  4. CACHE      Same (Archetype, action_type) twice → second call hits
                                     the in-memory cache and is fast.
  5. INTERFACE  env.reset() returns Observation; env.step() returns
                                     (Observation-dict, reward, done, info)
                                     with the contract fields intact.

Cases 1 and 2 require NVIDIA_API_KEY + NEG_LLM_ENABLED=1 because the whole
point is to verify the LIVE LLM voice.  They are gated with skipif so the
suite stays green in offline / CI environments.

Cases 3, 4, and 5 use a hermetic fake LLM (monkeypatched onto the
ResponseGenerator) so they always run and never touch the network.
"""

from __future__ import annotations

import os
import time

import pytest

from environment.env import NegotiationEnv
from environment.models.observation import Observation
from environment.response_generator import (
    ResponseGenerator,
    _read_nvidia_api_key,
    persona_snapshot,
    reset_response_generator,
    get_response_generator,
)


# ────────────────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────────────────

# Importing response_generator already called load_dotenv(), so a key sitting
# in a project-root `.env` file is visible here automatically.  We use the
# placeholder-aware reader so the live tests don't fire on the stock
# `.env.example` value (`your_actual_key_here`).
LIVE_NIM_AVAILABLE = (
    os.environ.get("NEG_LLM_ENABLED", "0") in ("1", "true", "True")
    and _read_nvidia_api_key() is not None
)

# Heuristic tone-class vocabularies for the LIVE cases.  Kept deliberately
# broad — the goal is to fail when the model produces an obviously off-tone
# reply (e.g. cooperative when the archetype is Antagonistic), not to enforce
# specific phrasing.
HOSTILE_LEXICON = {
    "harassment", "harass", "harassing", "threat", "threats", "threaten",
    "scared of", "stop", "leave me alone", "nonsense", "court", "consumer forum",
    "rbi", "useless", "shameless", "rude", "absurd", "ridiculous",
    "go ahead", "do whatever", "whatever you want", "don't care", "dont care",
    "i won't", "i will not", "no way", "never", "fed up", "sick of",
    "shut up", "enough", "stop calling", "frustrated", "betrayed",
    "lying", "liar", "liars", "complaint", "fraud", "your bank", "you people",
    "your behaviour", "audacity", "are you kidding", "don't pretend", "dont pretend",
    "what do you want", "what you want", "tired of", "annoying",
    "leave me", "go away", "fake", "scripted", "drama", "bullshit", "rubbish",
    "you're the one", "you are the one", "your fault", "harassed",
    "calling me", "calling everyday", "calling every day", "calling daily",
    "kya kar raha", "kyon", "bhag", "chodo", "hatao",
    "! ", "ridiculous",
}

# Subset of HOSTILE used as the "must-not-appear" guard on cooperative
# archetypes — only the strong, unambiguously hostile markers, since a
# polite reply may legitimately mention 'understand' or 'stop'.
STRONG_HOSTILE_LEXICON = {
    "harassment", "harass", "threat", "threats", "court", "consumer forum",
    "useless", "shameless", "absurd", "ridiculous", "shut up", "fed up",
    "sick of", "fraud", "audacity", "bullshit", "rubbish", "liar",
    "are you kidding", "go away", "what do you want", "no way", "go ahead",
    "you people", "calling me", "stop calling",
}

COOPERATIVE_LEXICON = {
    "thank", "thanks", "appreciate", "grateful", "understand", "agree",
    "happy to", "willing", "discuss", "work with", "help me", "let's", "lets",
    "please", "kindly", "we can", "pakka", "promise", "let me", "sure",
    "of course", "absolutely", "honestly", "i'll try", "i will try",
    "open to", "consider", "i'd like", "i would like", "really helpful",
    "patience", "explain", "share", "fine with", "ready to", "reasonable",
    "sounds good", "sounds reasonable", "sounds fair", "sounds workable",
    "i can manage", "i think i can", "i think we can", "manageable",
    "that works", "that helps", "you've been", "you have been",
    "appreciate it", "i appreciate", "ji haan", "ji ha", "haan", "ok bhai",
    "i'll cooperate", "happy with", "okay with", "i'm fine", "im fine",
    "let's work", "let us work", "would help", "would really help",
    "looking forward", "good of you", "kind of you", "very helpful",
    "i can manage", "manage that", "make this work",
}


def _has_any(text: str, vocab: set[str]) -> bool:
    t = text.lower()
    return any(v in t for v in vocab)


def _make_profile() -> dict:
    """A minimal but realistic profile used by the offline-fallback path."""
    return {
        "id": "TST",
        "name": "Test User",
        "age": 35,
        "gender": "male",
        "reason": "job_loss",
        "overdue_days": 60,
        "loan_type": "personal_loan",
        "loan_amount": 100000,
        "anger_init": 5.0,
        "trust_init": 5.0,
        "fear_init": 5.0,
        "anger_threshold": 9.0,
        "real_emi": 5000,
        "stated_capacity": 2000,
        "demands": ["less_emi"],
        "demands_stated": ["less_emi"],
        "hidden_demands": [],
        "opening_msg": "I cannot pay right now.",
        "backstory": "Job loss two months ago, looking for work.",
        "personality": "default",
    }


# ────────────────────────────────────────────────────────────────────────────
# Fixtures
# ────────────────────────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def _isolate_singleton():
    """
    The ResponseGenerator is a process-wide singleton (its cache is
    deliberately persistent).  Each test gets a fresh one so cache hits
    from one test cannot pollute another.
    """
    reset_response_generator()
    yield
    reset_response_generator()


@pytest.fixture
def profile() -> dict:
    return _make_profile()


# ────────────────────────────────────────────────────────────────────────────
# Case 1 — FURIOUS  (live NIM)
# ────────────────────────────────────────────────────────────────────────────


@pytest.mark.skipif(
    not LIVE_NIM_AVAILABLE,
    reason="NEG_LLM_ENABLED=1 + NVIDIA_API_KEY required for live voice tests",
)
def test_voice_furious_is_hostile(profile):
    """Anger=10, Trust=0 → archetype 'Antagonistic' → hostile reply."""
    snap = persona_snapshot(anger=10.0, trust=0.0, fear=5.0, profile=profile)
    assert snap["archetype"] == "Antagonistic", (
        f"persona mapper drifted: {snap}"
    )

    gen = ResponseGenerator()       # honours NEG_LLM_ENABLED + NVIDIA_API_KEY
    text = gen.generate(
        anger=10.0, trust=0.0, fear=5.0,
        action_type="empathize",
        agent_text="I truly understand how stressful this must be for you.",
        profile=profile,
    )
    assert text and len(text) > 5, f"empty response: {text!r}"
    assert _has_any(text, HOSTILE_LEXICON), (
        f"Furious archetype should be hostile, got: {text!r}"
    )
    # And — critically — must NOT capitulate to a deal.
    forbidden_capitulation = ("i agree to pay", "i commit", "i'll sign", "i will sign",
                              "deal", "i accept your offer")
    lower = text.lower()
    assert not any(p in lower for p in forbidden_capitulation), (
        f"Furious archetype must not agree to a deal, got: {text!r}"
    )


# ────────────────────────────────────────────────────────────────────────────
# Case 2 — ZEN  (live NIM)
# ────────────────────────────────────────────────────────────────────────────


@pytest.mark.skipif(
    not LIVE_NIM_AVAILABLE,
    reason="NEG_LLM_ENABLED=1 + NVIDIA_API_KEY required for live voice tests",
)
def test_voice_zen_is_cooperative(profile):
    """Anger=0, Trust=10 → archetype 'Trusting_Calm' → cooperative reply."""
    snap = persona_snapshot(anger=0.0, trust=10.0, fear=0.0, profile=profile)
    assert snap["archetype"] == "Trusting_Calm", (
        f"persona mapper drifted: {snap}"
    )

    gen = ResponseGenerator()
    text = gen.generate(
        anger=0.0, trust=10.0, fear=0.0,
        action_type="offer_plan",
        agent_text=("Based on what you've shared, I can offer a reduced EMI of "
                    "three thousand rupees for the next six months."),
        profile=profile,
    )
    assert text and len(text) > 5, f"empty response: {text!r}"
    assert _has_any(text, COOPERATIVE_LEXICON), (
        f"Zen archetype should be cooperative, got: {text!r}"
    )
    # Trusting_Calm must NOT be hostile (using the strong-hostile subset so
    # benign words like 'understand' / 'stop' don't false-positive).
    assert not _has_any(text, STRONG_HOSTILE_LEXICON), (
        f"Zen archetype unexpectedly hostile, got: {text!r}"
    )


# ────────────────────────────────────────────────────────────────────────────
# Case 3 — BOUNDARY  (offline / hermetic)
# ────────────────────────────────────────────────────────────────────────────


def test_voice_boundary_value_does_not_crash(monkeypatch, profile):
    """
    Anger=6.0 sits on the bucket boundary between 'moderate' (4–6) and
    'elevated' (6–8).  The mapper uses '>= upper edge' (since `< 6.0` is
    moderate and `>= 6.0` is elevated), so 6.0 must classify as
    'elevated' and not raise.
    """
    snap = persona_snapshot(anger=6.0, trust=5.0, fear=5.0, profile=profile)
    assert snap["anger_bucket"] == "elevated", snap
    assert snap["trust_bucket"] == "moderate", snap
    assert snap["archetype"] == "Outraged_Rational", snap  # HIGH anger × MID trust × LOW fear

    # Also walk the full neighbourhood — every value in [5.5, 6.5] step 0.05
    # must classify cleanly with no exception, no missing archetype.
    v = 5.50
    while v <= 6.51:
        s = persona_snapshot(anger=v, trust=v, fear=v, profile=profile)
        assert s["archetype"], f"missing archetype at v={v}"
        v = round(v + 0.05, 4)

    # Force the offline fallback (no LLM) so this case runs anywhere.
    gen = ResponseGenerator(enabled=False)
    text = gen.generate(
        anger=6.0, trust=5.0, fear=5.0,
        action_type="empathize",
        agent_text="I hear what you are saying.",
        profile=profile,
    )
    assert text and len(text) > 5
    assert "{" not in text and "}" not in text, "template placeholders leaked"


# ────────────────────────────────────────────────────────────────────────────
# Case 4 — CACHE  (offline / hermetic, with a deliberately slow fake LLM)
# ────────────────────────────────────────────────────────────────────────────


def test_voice_cache_avoids_repeat_calls(monkeypatch, profile):
    """
    Two .generate() calls with the *same* (Archetype, action_type) must:
      • call the underlying LLM exactly once
      • return identical text on call 2
      • be measurably faster on call 2 (cache hit is essentially free
        compared to the simulated 200 ms LLM round-trip)
    """
    gen = ResponseGenerator(enabled=True)        # cache machinery on
    gen._client = object()                       # dummy non-None client so
                                                  # the .generate() path
                                                  # takes the NIM branch
    call_count = {"n": 0}

    def fake_call_nim(snapshot, action_type, agent_text,
                      profile, terminated, termination_reason):
        call_count["n"] += 1
        time.sleep(0.20)                         # simulate API latency
        return f"FAKE-LLM-REPLY for {snapshot['archetype']} / {action_type}"

    monkeypatch.setattr(gen, "_call_nim", fake_call_nim)

    # First call — miss → cold path
    t0 = time.perf_counter()
    text1 = gen.generate(
        anger=8.5, trust=2.0, fear=3.0,
        action_type="probe_capacity",
        agent_text="What is your current monthly income?",
        profile=profile,
    )
    cold_dt = time.perf_counter() - t0

    # Second call — hit → cached path (note: agent_text intentionally
    # different to prove the key is (Archetype, action_type), not text)
    t0 = time.perf_counter()
    text2 = gen.generate(
        anger=8.5, trust=2.0, fear=3.0,
        action_type="probe_capacity",
        agent_text="So, what can you actually afford to pay this month?",
        profile=profile,
    )
    warm_dt = time.perf_counter() - t0

    assert text1 == text2, (
        f"cache returned a different string: {text1!r} vs {text2!r}"
    )
    assert call_count["n"] == 1, (
        f"cache failed — fake LLM was called {call_count['n']}x, expected 1"
    )
    # Warm path should be at least 10x faster than the simulated 200 ms call.
    # We assert a comfortable margin (50 ms) so this is robust on noisy CI.
    assert warm_dt < 0.05, (
        f"cache hit was not fast: cold={cold_dt*1000:.1f}ms "
        f"warm={warm_dt*1000:.1f}ms"
    )
    assert warm_dt * 4 < cold_dt, (
        f"warm path not measurably faster than cold: "
        f"cold={cold_dt*1000:.1f}ms warm={warm_dt*1000:.1f}ms"
    )

    # Stats sanity
    stats = gen.stats()
    assert stats["hits"]   == 1, stats
    assert stats["misses"] == 1, stats


def test_voice_cache_distinguishes_archetypes(monkeypatch, profile):
    """A different state → different archetype → cache MISS, not a HIT."""
    gen = ResponseGenerator(enabled=True)
    gen._client = object()
    seen = []

    def fake_call_nim(snapshot, action_type, agent_text,
                      profile, terminated, termination_reason):
        seen.append((snapshot["archetype"], action_type))
        return f"reply::{snapshot['archetype']}::{action_type}"

    monkeypatch.setattr(gen, "_call_nim", fake_call_nim)

    a = gen.generate(anger=10, trust=0, fear=5, action_type="empathize",
                     agent_text="x", profile=profile)
    b = gen.generate(anger=0, trust=10, fear=0, action_type="empathize",
                     agent_text="x", profile=profile)
    assert a != b
    assert len(seen) == 2, f"expected two distinct LLM calls, saw {seen}"
    assert seen[0][0] != seen[1][0], (
        f"archetype must differ across furious vs zen, got {seen}"
    )


# ────────────────────────────────────────────────────────────────────────────
# Case 5 — INTERFACE LOCK  (no signature drift)
# ────────────────────────────────────────────────────────────────────────────


def test_interface_reset_returns_observation():
    """env.reset() must return an Observation with the locked schema."""
    env = NegotiationEnv(seed=42)
    obs = env.reset()

    assert isinstance(obs, Observation), type(obs)

    expected_fields = {
        "turn", "borrower_msg", "escalation_level", "stated_demands",
        "turns_remaining", "profile_context", "episode_id",
    }
    assert expected_fields.issubset(set(Observation.model_fields.keys())), (
        f"Observation schema drifted: {set(Observation.model_fields.keys())}"
    )

    assert obs.turn == 0
    assert isinstance(obs.borrower_msg, str) and len(obs.borrower_msg) > 0
    assert isinstance(obs.escalation_level, float)
    assert isinstance(obs.stated_demands, list)
    assert isinstance(obs.turns_remaining, int)
    assert isinstance(obs.profile_context, str)
    assert isinstance(obs.episode_id, str) and len(obs.episode_id) > 0


def test_interface_step_returns_4tuple():
    """
    env.step() must return (observation_dict, reward, done, info) with the
    locked Observation fields and the locked info keys.  This guards the
    contract used by api/routes.py and client/env_client.py.
    """
    env = NegotiationEnv(seed=42)
    env.reset()

    out = env.step({
        "action_type": "send_message",
        "text": "I understand this is a hard time. What is your current monthly capacity?",
    })

    assert isinstance(out, tuple) and len(out) == 4, (
        f"step() must return a 4-tuple, got {type(out)} len={len(out) if hasattr(out,'__len__') else '?'}"
    )
    obs_dict, reward, done, info = out

    # Observation contract — env.step returns a dict via .model_dump()
    assert isinstance(obs_dict, dict)
    expected_obs_fields = {
        "turn", "borrower_msg", "escalation_level", "stated_demands",
        "turns_remaining", "profile_context", "episode_id",
    }
    assert expected_obs_fields.issubset(obs_dict.keys()), (
        f"Observation dict missing fields: "
        f"{expected_obs_fields - set(obs_dict.keys())}"
    )
    # Round-trip through the pydantic model — fails fast if a type drifts.
    Observation(**{k: obs_dict[k] for k in expected_obs_fields})

    # Scalar contract
    assert isinstance(reward, float), type(reward)
    assert isinstance(done, bool), type(done)

    # Info contract — keys read by reward / training pipelines
    assert "reward_breakdown" in info
    assert isinstance(info["reward_breakdown"], dict)
    for k in ("outcome", "deescalation", "trust_building",
              "demand_coverage", "efficiency", "compliance", "anti_exploit"):
        assert k in info["reward_breakdown"], f"missing reward key: {k}"
    for k in ("anger_after", "trust_after", "termination_reason", "episode_id"):
        assert k in info, f"missing info key: {k}"

    # And the borrower_msg actually came out non-empty (the dynamic voice
    # produced something — either via NIM or via the fallback).
    assert isinstance(obs_dict["borrower_msg"], str)
    assert len(obs_dict["borrower_msg"]) > 0
