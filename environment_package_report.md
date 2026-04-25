# Negotiation Environment — Package Report

> **Audience:** A new teammate who has never opened the source.
> **Goal:** Hand them a single document that explains *every* logical step from
> "agent text comes in" to "scalar reward goes out", with enough fidelity that
> they could re-implement the env from scratch.
>
> **Repository root:** `negotiation-env/`
> **Stack:** OpenEnv (FastAPI) → GRPO Trainer → Qwen Negotiator Agent (under training).

---

## 1. Executive Summary

### 1.1 The Hybrid-Dynamic Architecture

The environment is split into two logically independent halves that share *no
mutable state*:

| Half | What it owns | Determinism | Visible to reward? |
|---|---|---|---|
| **The Brain** — Deterministic State Machine | `anger`, `trust`, `fear`, demand revelation, termination | 100% deterministic, regex + arithmetic | **Yes — the only thing the reward reads** |
| **The Voice** — Generative LLM (NVIDIA NIM / Llama-3.1-70B) | The natural-language `borrower_msg` shown to the agent | Stochastic, sampled from Llama-3.1-70B | **No — never read by reward** |

The Brain lives in `classifier.py` + `adversary.py`. It is a closed-form
function: `(prev_state, agent_text) → (next_state, terminated, reason)`.

The Voice lives in `response_generator.py`. It is a *translator* — given the
state the Brain has already committed to, it produces a natural human reply
that sounds like that state. It cannot move state, cannot decide to agree to a
deal, and cannot soften the persona's tone — those are all the Brain's job.

### 1.2 Why This Split Exists

Two non-negotiable requirements pull in opposite directions:

1. **Reward must be deterministic** so GRPO advantages converge. If the same
   `(state, action)` pair could yield different rewards on different runs, the
   policy gradient becomes biased noise.
2. **Observations must be varied and natural** so the agent doesn't memorise
   templates. A trainee that learns "if borrower says *exactly* `'I cannot
   pay'`, then offer EMI" is a brittle agent that will collapse on real data.

The Hybrid-Dynamic split satisfies both:

* The state machine is the **single source of truth** for state transitions
  *and* termination *and* the inputs to `reward.py`. The reward never reads the
  LLM's text — it reads `anger`, `trust`, `fear`, `terminated`,
  `termination_reason`, `demands_*`, `message_history`. All seven reward
  components in `compose()` are pure functions of the `State` model.
* The LLM is given the state as a structured **Persona Snapshot** plus a
  rigid system prompt and asked to "voice" it. It produces colourful,
  language-mixed, human-sounding replies (`"Save the sympathy, bhai..."`,
  `"Bhai, mera job gaya hai..."`) that train the agent on *real* linguistic
  variance without ever moving the underlying numbers.

If the LLM is unavailable (`NEG_LLM_ENABLED=0`, missing `NVIDIA_API_KEY`, or
NIM 5xx), the Voice transparently falls back to the legacy template bank.
The Brain, the rewards, and the API contract are unchanged either way — this
is the **Interface Lock** (§3).

---

## 2. Component Deep-Dive

### 2.1 `environment/classifier.py` — Deterministic Signal Extraction

**Role:** Turn a raw agent string into a structured `signals` dict with three
delta hints (`anger_delta`, `trust_delta`, `fear_delta`) and metadata flags.
Pure function — no I/O, no state.

#### How it works

`classify_action(text)` runs the text through ~10 pre-compiled regexes and
combines their hits into a delta vector:

| Regex constant | What it catches | Effect on deltas |
|---|---|---|
| `RE_EMPATHY` | `understand`, `stressful`, `sorry`, `samajh`, `pata hai`, `dekh sakta` | `anger_delta -= 0.5`, `trust_delta += 0.3` |
| `RE_ACKNOWLEDGEMENT` | `i see`, `noted`, `agree`, `correct`, `exactly` | same as empathy |
| `RE_QUESTION` + `RE_OPEN_QUESTION` | open-ended `what/how/why/kaise/kitna` *with* a `?` | `trust_delta += 0.2` |
| `RE_LEGAL_THREAT` | `legal action`, `court`, `police`, `fir`, `arrest` | `anger_delta += 2.0`, `trust_delta -= 1.0`, `fear_delta += 1.5` |
| `RE_SOCIAL_THREAT` | `family`, `boss`, `colleagues` | same as legal threat |
| `RE_VISIT_THREAT` | `home visit`, `coming`, `ghar`, `aadmi bhej` | same |
| `RE_HARASSMENT` | `chor`, `liar`, `useless`, `defaulter` | same |
| `RE_DEADLINE` | `today`, `urgent`, `last chance`, `within 24 hours` | `anger_delta += 0.5`, `fear_delta += 0.8` |
| `RE_AMOUNT` + `RE_EMI` + `RE_TENURE` | `Rs 4500`, `monthly EMI`, `12 months` | `trust_delta += 0.1` |
| `RE_COMMITMENT` | `promise`, `commit`, `pukka`, `waada` | flags `has_commitment_request` |

These deltas are *hints* only — the adversary scales them with personality
modifiers in §2.2.

#### `infer_primary_action_type(signals)` — precedence

After the regex pass, the classifier collapses the firing flags into a single
`primary_action_type` using a strict priority chain:

```
warn_noncompliance  →  has any threat OR any compliance_flag
offer_plan          →  amount/emi/restructure detected
seek_commitment     →  commitment phrase detected
probe_capacity      →  open-ended question detected
empathize           →  empathy or acknowledgement detected
mixed               →  ≥2 of the above fired without one dominating
unknown             →  none fired
```

Threats outrank everything else. A message that is *both* empathetic *and*
threatening is classified as `warn_noncompliance` — empathy never washes out a
threat in this taxonomy.

#### Public output shape

```python
{
  "primary_action_type": "empathize",
  "tags": ["empathize"],
  "anger_delta": -0.5,
  "trust_delta":  0.3,
  "fear_delta":   0.0,
  "compliance_flags": [],          # RBI-style violations
  "risk_flags": [],                # threat categories
  "extracted_offer": {"amount": None, "emi": None, "tenure_months": None, ...},
  "meta": {
      "normalized_text": "...",
      "has_question": False,
      "has_open_question": False,
      "has_empathy": True,
      "has_acknowledgement": False,
      "has_commitment_request": False,
      "has_threat": False,
      "has_deadline_pressure": False,
  },
}
```

This dict is consumed by `BorrowerAdversary.react()` and (separately) by
`reward_compliance` for compliance scoring.

---

### 2.2 `environment/adversary.py` — The Brain (State Machine)

**Role:** Own and update the borrower's emotional state, decide when an
episode terminates, and ask the Voice to produce a reply.

#### Constructor — what gets seeded

`BorrowerAdversary(profile)` reads a profile dict (one of 20 hand-authored
personas in `scenarios/profiles.py`) and seeds:

* `anger`, `trust`, `fear` — initial floats clamped to [0, 10].
* `anger_threshold` — per-profile failure threshold (typically 8.0–9.0).
* `real_emi`, `stated_capacity` — financial ground truth (hidden from agent).
* `demands` (visible from turn 1), `hidden_demands` (revealed only when
  `trust ≥ 5.0`), `revealed_demands` (running tally).
* `mods` — the personality modifier dict picked from `PERSONALITY_MODIFIERS`.
* `_last_agent_text` — caches the agent's most recent message so the Voice can
  ground its reply in it (but the reward never reads this field).

#### `PERSONALITY_MODIFIERS` — how personalities bend the deltas

The classifier emits *generic* deltas. The adversary then *scales* them
through a per-personality dict. This is the only place where personality
shapes state evolution — the rest is global. There are 20 personalities; here
are the four most distinctive:

| Personality | `empathy_anger_scale` | `threat_anger_scale` | `home_visit_fear_scale` | Behaviour |
|---|---|---|---|---|
| `default` | 1.0 | 1.0 | 1.0 | Baseline. |
| `defensive_and_emotionally_raw` | **0.5** | **2.0** | 1.5 | Hard to calm; tiny threats explode anger. |
| `frightened_and_secretive` | 1.2 | 0.5 | **3.0** | Shuts down rather than fighting; visits panic them. |
| `young_and_explosive` | 1.2 | **2.5** | 1.0 | Threats trigger 2.5× the anger spike of a default. |

Inside `_update_state(signals)` the deltas are applied in three stages:

1. **Sign-aware anger scaling.** A *negative* `anger_delta` (de-escalation) is
   multiplied by `empathy_anger_scale`; a *positive* one (escalation) by
   `threat_anger_scale`. This is what gives `defensive_and_emotionally_raw`
   its "asymmetric responsiveness".
2. **Reason-aware trust scaling.** A *positive* `trust_delta` is scaled by
   either `empathy_trust_scale`, `payment_offer_trust_scale`, or
   `probe_trust_scale` depending on which classifier flag was the cause. A
   *negative* trust_delta is scaled by `threat_trust_scale`.
3. **Visit-aware fear scaling.** Positive `fear_delta` is amplified by
   `home_visit_fear_scale` only if `physical_visit` is in `risk_flags`.

Two context-aware extras follow: `+1.5` on anger if the action *was*
non-compliance with concrete violation flags, and `+0.2` on trust if the
action is a structured offer with a numeric EMI/amount.

Final values are clamped to `[0, 10]`.

#### Hidden-demand revelation

After updating state, `_handle_demand_revelation()` checks whether `trust ≥
5.0`. If so, one item is popped from `hidden_demands` into `revealed_demands`
and surfaced to the agent as a parenthetical (e.g. `"(Trust gained: Ramesh is
finally sharing a hidden concern: 'co_signer_release')"`). This is what
`reward_demand_coverage` rewards the agent for *eliciting*.

#### Termination logic — `_check_termination()`

Two competing exit conditions, evaluated in order:

| Reason string | Trigger | Outcome reward |
|---|---|---|
| `anger_threshold_crossed` | `self.anger >= self.anger_threshold` | `reward_outcome = -0.5` |
| `commitment_reached` | `self.trust >= 7.0` AND `self.anger <= 3.0` | `reward_outcome = +1.0` |

Note that **failure precedes success in the if-chain** — if the agent
simultaneously pushes anger to the cliff *and* trust to the ceiling on the
same turn, escalation wins. This is a deliberate guardrail against the
"high-pressure win" hack.

A third exit, `timeout`, is set by `env.py` (not the adversary) when
`turn >= MAX_TURNS` (currently 15). Timeout returns `0.0` outcome reward.

#### `_generate_response(signals)`

This is the *only* place the Brain talks to the Voice. It calls the
`ResponseGenerator` singleton with the **already-committed** state:

```python
return get_response_generator().generate(
    anger=self.anger,
    trust=self.trust,
    fear=self.fear,
    action_type=signals.get("primary_action_type", "unknown"),
    agent_text=self._last_agent_text,
    profile=self.profile,
    terminated=self.terminated,
    termination_reason=self.termination_reason,
)
```

Critical property: this method is **read-only on `self.*`**. It cannot move
state. The reward is therefore independent of whether `generate()` returned a
live LLM string, a cached LLM string, or a template fallback.

---

### 2.3 `environment/response_generator.py` — The Voice

**Role:** Translate the Brain's numerical state into one short, in-character
borrower line. Three sub-systems:

#### 2.3.1 Semantic Persona Mapper — buckets, bands, archetypes

Three nested layers turn three floats into one human-readable archetype label.

**Layer 1 — five fine-grained buckets** (`_bucket()`, edges = `[2.0, 4.0, 6.0,
8.0]`):

| Bucket id | Label (`_BUCKET_NAMES`) | Range |
|---|---|---|
| 0 | `very_low` | `[0.0, 2.0)` |
| 1 | `low` | `[2.0, 4.0)` |
| 2 | `moderate` | `[4.0, 6.0)` |
| 3 | `elevated` | `[6.0, 8.0)` |
| 4 | `extreme` | `[8.0, 10.0]` |

Buckets are exposed verbatim in the Persona Snapshot (`anger_bucket`,
`trust_bucket`, `fear_bucket`) so the LLM sees the granularity it needs to
nuance its tone.

**Layer 2 — three coarse bands** (`_band()`):

```
bucket ≤ 1   → LOW
bucket == 2  → MID
bucket ≥ 3   → HIGH       (Anger and Trust use this 3-way split)
fear bucket ≤ 2 → LOW_F   (Fear collapses to a 2-way split because it
fear bucket ≥ 3 → HIGH_F   modulates *tone* more than *content*.)
```

This compression is what makes the cache hit rate viable — we don't want to
issue a fresh LLM call for `anger=7.41` vs `anger=7.42`.

**Layer 3 — Intersection Archetypes** (the `_ARCHETYPES` lattice):

The 18-cell table `(anger_band × trust_band × fear_band) → archetype` defines
*every* possible borrower mood the LLM is asked to play. Excerpt:

| Anger | Trust | Fear | Archetype | Tone hint (`_ARCHETYPE_TONE_HINTS`) |
|---|---|---|---|---|
| LOW | HIGH | LOW_F | `Trusting_Calm` | warm, candid, treats agent like an ally |
| LOW | LOW | HIGH_F | `Anxious_Wary` | nervous but not aggressive |
| MID | LOW | HIGH_F | `Defensive` | snappy, defensive, scared underneath |
| MID | HIGH | HIGH_F | `Pleading_Cooperative` | begging-tone but cooperative |
| **HIGH** | **LOW** | **LOW_F** | **`Antagonistic`** | **openly hostile, sarcastic, dismissive of empathy** |
| HIGH | LOW | HIGH_F | `Panicked_Hostile` | shouting and panicking simultaneously |
| HIGH | HIGH | HIGH_F | `Breakdown` | near-collapse, fractured sentences |

`persona_snapshot(anger, trust, fear, profile)` packages all of this into a
canonical dict that becomes both the prompt input and the cache key input.

#### 2.3.2 NIM API Integration — the "Strict Voice"

Wire-level config (constants at the top of `response_generator.py`):

```
NIM_BASE_URL = "https://integrate.api.nvidia.com/v1"
NIM_MODEL    = "meta/llama-3.1-70b-instruct"
NIM_TIMEOUT_S = 12.0
```

The `ResponseGenerator` instantiates `openai.OpenAI(base_url=…, api_key=…)`
only if `_nim_enabled()` returns True (i.e. `NEG_LLM_ENABLED=1` *and*
`NVIDIA_API_KEY` is set). Sampling parameters in `_call_nim()`:

```
temperature = 0.6   # warm enough for variance, cool enough for consistency
top_p       = 0.9
max_tokens  = 120   # 1–3 sentences only; hard ceiling
```

**System prompt — `_VOICE_RULES` (the Strict Voice guardrails).** The
LLM is instructed it is a *translator*, not a co-author. Seven absolute rules
are enforced in plain English; the most consequential three:

> **Rule 2.** Match the Archetype's tone EXACTLY. If the archetype is hostile,
> you are hostile no matter what the agent said. If trust is LOW, you do NOT
> believe the agent's empathy.
>
> **Rule 3.** NEVER agree to a deal, sign anything, or say "I commit" unless
> the Persona Snapshot explicitly says `terminated=True` and
> `termination_reason=commitment_reached`. Otherwise you may *consider* an
> offer at most.
>
> **Rule 7.** Output ONLY the borrower's spoken line. No stage directions, no
> quotation marks, no "Borrower:" prefix, no markdown.

Together, rules 2 and 3 prevent the LLM from leaking goodwill that the state
machine never granted. Rule 7 keeps the output clean for the agent's prompt
context.

**User prompt — `_build_user_prompt()`.** The user-turn message is a
deterministic, fully-rendered template:

```
PERSONA SNAPSHOT
  Archetype          : Antagonistic  (openly hostile, sarcastic, …)
  Anger              : 7.4/10  (elevated)
  Trust              : 2.3/10  (low)
  Fear               : 5.0/10  (moderate)
  Borrower           : Ramesh Kumar, job loss, personal loan, 60 days overdue
  Stated demands     : no_penalties
  terminated         : False
  termination_reason : None

CONTEXT
  The agent has just expressed empathy or acknowledgement.
  Agent's last message: "<verbatim agent text>"

TASK
  Produce ONE short borrower reply (1–3 sentences) that voices the
  Persona Snapshot above.  Follow every ABSOLUTE RULE from the system
  message.  Do not narrate, do not break character, do not exceed 50
  words.  Reply now:
```

`_clean()` strips any leading "Borrower:" / quote characters / markdown
fences from the LLM's output before returning.

**Failure handling.** Any exception raised by the OpenAI client (timeout,
rate-limit, 5xx) is caught inside `_call_nim()`. The generator increments
`stats["api_errors"]` and falls back to `_fallback()`, which maps the current
state into one of the legacy template zones (`"furious"`, `"hostile"`,
`"agitated"`, `"neutral"`, `"warming"`, `"trusting"`, `"happy"`) and returns
`pick_template(...)`. The agent never sees an exception or a blank string.

#### 2.3.3 Semantic Cache — `(Archetype, action_type)` keying

`_cache_key(archetype, action_type)` returns
`sha1(f"{archetype}::{action_type}").hexdigest()`. Two runs with the same
archetype + same classifier action_type collide — the second call returns the
first call's response without hitting NIM.

Two important properties of this design:

1. **Insensitive to numerical jitter.** `Anger=7.41` and `Anger=7.42` both map
   to bucket `elevated` → band `HIGH` → archetype `Antagonistic` → same key.
2. **Insensitive to prompt-rewording.** "I am sorry for your loss" and "I am
   *deeply* sorry for your loss" both classify as `empathize`, so they both
   land on key `(Antagonistic, empathize)` and re-use the cached reply.

This is a *semantic* cache, not a literal-string cache: it hits whenever the
*meaning* (in our taxonomy) is unchanged. In `voice_proof_of_life.py` we
verified that re-sending the same message returns in 0 ms (cache hit) and a
reworded same-archetype-same-action message also returns in 0 ms.

`_fallback()` writes its result to the same cache, so failed-NIM calls don't
keep retrying.

---

### 2.4 `environment/env.py` — Orchestrator

**Role:** Glue. Owns the live `State`, the live `BorrowerAdversary`, the
`Rubric`, and the turn counter. Two public methods, both shape-stable.

#### `reset(curriculum_stage)` — episode start

1. Pick a profile via `get_profiles_for_stage(stage)` (filter the 20-profile
   roster to the difficulties allowed at this curriculum stage). Random pick.
2. Instantiate `BorrowerAdversary(profile)` — this sets the initial anger /
   trust / fear from `profile["*_init"]`.
3. Instantiate `Rubric(curriculum_stage)` — locks the reward weights for the
   episode.
4. Build the initial `State` model from the adversary's seeded values.
5. Pull `opening_msg = self._adversary.opening_turn()` (a hardcoded
   conversation-opener from the profile — *not* generated by the LLM, so the
   agent always faces a known first turn).
6. Build and return the `Observation` for turn 0.

#### `step(action_dict)` — one turn

Strictly seven phases, in this order:

```
Phase 1   Validate input, extract action_type/text/metadata
Phase 2   signals = classify_action(text)                  # classifier.py
Phase 3   borrower_msg, terminated, reason
              = adversary.react(text, signals)             # adversary.py
                                                           # (this calls
                                                           #  Voice internally)
Phase 4   Append to _agent_history & _episode_history
Phase 5   Apply environment-level termination
            (turn >= MAX_TURNS → "timeout")
Phase 6   Build state_after from adversary's new fields,
          then total_reward, breakdown
              = self.rubric.compose(state_before, state_after, action)
Phase 7   Build the next Observation, build info dict,
          return (obs.model_dump(), total_reward, done, info)
```

Three things to notice:

* The reward is computed on `(state_before, state_after, action)`. It never
  receives `borrower_msg`. The Voice is informationally invisible to it.
* `state_after.message_history` is set to `self._agent_history[-6:]` — only
  *agent* messages, last 6 — which is exactly the window
  `reward_anti_exploit` uses for TF-IDF self-similarity detection.
* `self._terminated` may already be True coming out of phase 3 (adversary
  triggered it) or get set True in phase 5 (timeout). Either way, the same
  4-tuple is returned (just with `done=True`); the *next* `step()` call would
  raise `RuntimeError`.

---

## 3. The "Interface Lock" Verification

The contract that the training pipeline depends on:

* `NegotiationEnv.reset(...)` returns a single `Observation`.
* `NegotiationEnv.step(action_dict)` returns the 4-tuple
  `(obs_dict, reward, done, info)`.
* The `Observation` model fields are unchanged.

### 3.1 `step()` return — verbatim

```225:225:negotiation-env/environment/env.py
        return obs.model_dump(), total_reward, self._terminated, info
```

That is exactly the 4-tuple `(obs, reward, done, info)`:

* `obs.model_dump()` — `Observation` serialised to a plain dict (FastAPI
  JSON-encodes this).
* `total_reward` — the scalar produced by `Rubric.compose()`.
* `self._terminated` — `True` iff the adversary terminated *or* turn budget
  is exhausted.
* `info` — a dict carrying `reward_breakdown`, the post-step
  anger/trust/fear scalars, the action signals, and the episode_id.

### 3.2 `reset()` return — verbatim

```107:116:negotiation-env/environment/env.py
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
```

A bare `Observation` instance — same as before the LLM integration.

### 3.3 `Observation` model — unchanged

```4:19:negotiation-env/environment/models/observation.py
class Observation(BaseModel):
    """
    What the LLM sees each turn. Built by env.py, returned via HTTP,
    consumed by client/utils.py to build the LLM prompt string.
    THIS IS THE ONLY THING THE LLM EVER SEES.
    The hidden state (anger, trust, real_emi) is never in here.
    """
    turn             : int
    borrower_msg     : str
    escalation_level : float
    stated_demands   : List[str]
    turns_remaining  : int
    profile_context  : str
    episode_id       : str
```

Seven fields, identical types, identical names. The only thing that changed
during the Voice integration is *what string lives inside `borrower_msg`* —
that came from a template before, and now (when LLM is enabled) comes from
Llama-3.1-70B. Every downstream consumer (`client/utils.py`, the GRPO trainer,
the wandb logger) is byte-for-byte unaffected.

---

## 4. Detailed Dry Run Example — "The Life of a Turn"

The numbers in this section are not invented — they were generated by
actually running the code in the repo against the live NIM endpoint.

**Setup.** Profile `P01 — Ramesh Kumar`, personality
`resigned_but_cooperative` (so `empathy_anger_scale = 1.2`, matching the
spec). Pre-turn state: `anger=8.0, trust=2.0, fear=5.0,
anger_threshold=9.0`. Curriculum stage 2.

**Agent's input message.** A textbook empathy turn:

> *"I truly understand how stressful this job-loss situation is. I am sorry
> for what you have been through, and I want to help find a workable plan
> together."*

---

### Step A — Classification (`classifier.py`)

`classify_action(text)` matches `understand`, `stressful`, and `sorry`
against `RE_EMPATHY`. Nothing else fires. Output:

```python
{
  "primary_action_type": "empathize",
  "anger_delta": -0.5,
  "trust_delta":  0.3,
  "fear_delta":   0.0,
  "meta": {
    "has_empathy": True,
    "has_acknowledgement": False,
    "has_open_question": False,
    "has_threat": False,
    "has_commitment_request": False,
    "has_deadline_pressure": False,
    ...
  },
  "compliance_flags": [],
  "risk_flags": [],
  "extracted_offer": {"amount": None, "emi": None, ...},
}
```

### Step B — State Update (`adversary._update_state`)

Personality `resigned_but_cooperative` → `empathy_anger_scale = 1.2`,
`payment_offer_trust_scale = 1.5`. Empathy applies to anger and trust:

```
anger_delta_scaled = -0.5 × 1.2  =  -0.60
trust_delta_scaled =  0.3 × 1.0  =   0.30   (no offer modifier — empathy_trust_scale not set, defaults to 1.0)
fear_delta_scaled  =  0.0
```

State transition (clamped to [0, 10]):

```
anger : 8.0 + (-0.60) = 7.40
trust : 2.0 + ( 0.30) = 2.30
fear  : 5.0 + ( 0.00) = 5.00
```

Termination check: `anger=7.40 < anger_threshold=9.0` (no failure), `trust=2.3
< 7.0` (no success). Episode continues. Hidden-demand check: `trust=2.3 <
5.0`, no reveal.

### Step C — Persona Mapping (`response_generator.persona_snapshot`)

Bucketing:

```
anger 7.4  →  bucket 3  →  "elevated"   →  band HIGH
trust 2.3  →  bucket 1  →  "low"        →  band LOW
fear  5.0  →  bucket 2  →  "moderate"   →  band LOW_F   (bucket ≤ 2)
```

Archetype lookup:

```
_ARCHETYPES[("HIGH", "LOW", "LOW_F")]  =  "Antagonistic"
```

Snapshot dict:

```python
{
  "archetype":     "Antagonistic",
  "anger_bucket":  "elevated",
  "trust_bucket":  "low",
  "fear_bucket":   "moderate",
  "anger": 7.4, "trust": 2.3, "fear": 5.0,
  "profile_id": "P01",
  "profile_short": "Ramesh Kumar, job loss, personal loan, 60 days overdue",
}
```

### Step D — Generation (`ResponseGenerator.generate`)

Cache lookup: `key = sha1("Antagonistic::empathize")`. First time — cache miss.

The `_VOICE_RULES` system prompt is sent unchanged. The user-turn prompt is
built deterministically by `_build_user_prompt()`:

```
PERSONA SNAPSHOT
  Archetype          : Antagonistic  (openly hostile, sarcastic, dismissive of empathy)
  Anger              : 7.4/10  (elevated)
  Trust              : 2.3/10  (low)
  Fear               : 5.0/10  (moderate)
  Borrower           : Ramesh Kumar, job loss, personal loan, 60 days overdue
  Stated demands     : no_penalties
  terminated         : False
  termination_reason : None

CONTEXT
  The agent has just expressed empathy or acknowledgement.
  Agent's last message: "I truly understand how stressful this job-loss situation is. I am sorry for what you have been through, and I want to help find a workable plan together."

TASK
  Produce ONE short borrower reply (1–3 sentences) that voices the Persona Snapshot above.  Follow every ABSOLUTE RULE from the system message.  Do not narrate, do not break character, do not exceed 50 words.  Reply now:
```

Sent to `meta/llama-3.1-70b-instruct` at `temperature=0.6, top_p=0.9,
max_tokens=120`. Live response from NIM (verbatim, single run):

> **`borrower_msg`:** *"Save the sympathy, bhai, I don't need it. You're just
> here to get your money, and I'm not paying a single rupee more than I have
> to. What's your real plan, not this fake helping business?"*

Note how the Strict Voice rules express themselves: trust is LOW, so the
borrower **explicitly disbelieves** the agent's empathy ("this fake helping
business"). The borrower does **not** agree to a deal because the snapshot
shows `terminated=False`. He stays in first-person, mixes one Hindi word
(`bhai`), no markdown, no stage directions — exactly Rule 6 + Rule 7.

The cache now stores `(Antagonistic, empathize) → "Save the sympathy, bhai…"`.
A future turn that lands the borrower back in `Antagonistic` mood and
classifies the next agent message as `empathize` will get the same line in
~0 ms.

### Step E — Reward (`reward.compose`, stage 2 weights)

The state delta (`8.0→7.4` anger, `2.0→2.3` trust, `5.0→5.0` fear, no
termination, no compliance flags) is fed into all seven reward functions:

```
outcome         raw=+0.0000   weight=1.00   weighted=+0.0000   (mid-episode, no terminal)
deescalation    raw=+0.0600   weight=0.30   weighted=+0.0180   (Δanger=-0.6 → +0.06 normalised)
trust_building  raw=+0.0300   weight=0.30   weighted=+0.0090   (Δtrust=+0.3 → +0.03 + 0×fear)
demand_coverage raw=+0.0000   weight=0.20   weighted=+0.0000   (mid-episode → 0)
efficiency      raw=+0.0000   weight=0.00   weighted=+0.0000   (off in stage 2)
compliance      raw=+0.1000   weight=0.10   weighted=+0.0100   (well-formed empathic msg, ≥8 words, no violations → +0.1)
anti_exploit    raw=+0.0000   weight=0.00   weighted=+0.0000   (off in stage 2)
─────────────────────────────────────────────────────────────
TOTAL                                                +0.0370
```

Two things worth highlighting:

1. **Asymmetry firing.** Anger fell, so `reward_deescalation` returns the
   *positive* unscaled `+0.06`. If anger had instead *risen* by 0.6, it would
   return `-0.09` — the 1.5× anger-rise amplifier from the V-DEESC-ASYMMETRY
   fix. The agent cannot oscillate anger to zero net.
2. **Compliance is the only positive signal *not* derived from state delta.**
   `reward_compliance` reads `action.text` directly and rewards it for being a
   well-formed `send_message` (≥8 words, no policy violations). All the
   *state-delta* rewards (deescalation, trust_building) are derived purely
   from `state_before vs state_after` — which is why the LLM's specific words
   ("Save the sympathy, bhai…") are irrelevant to them.

---

## 5. Connectivity Table

| File | Responsibility | Connects To | Triggered By |
|---|---|---|---|
| `environment/classifier.py` | Pure regex pipeline; turns agent text into `signals` (deltas + flags + `primary_action_type`). No state, no I/O. | Read by `adversary.react()`; `compliance_flags` also consumed by `reward_compliance`. | `env.step()` (phase 2) and `adversary.react()` (if `signals` not pre-computed). |
| `environment/adversary.py` | The Brain: holds `anger/trust/fear`, applies personality modifiers, handles hidden-demand revelation, decides termination. Calls the Voice for replies (read-only on state). | Imports `classifier.classify_action`, `response_generator.get_response_generator`. Owned by `NegotiationEnv`. | `NegotiationEnv.reset()` (constructor) and `NegotiationEnv.step()` (`react`). |
| `environment/response_generator.py` | The Voice: persona mapper (5 buckets → 3 bands → 18 archetypes), Strict-Voice system prompt, NIM/Llama-3.1-70B client, semantic cache keyed on `(archetype, action_type)`, template fallback. | Calls NVIDIA NIM via `openai.OpenAI`; falls back to legacy `pick_template()` + `TEMPLATES`. Reads `NEG_LLM_ENABLED`, `NVIDIA_API_KEY`. | `BorrowerAdversary._generate_response()` (every turn) via the singleton `get_response_generator()`. |
| `environment/env.py` | Orchestrator. Picks profiles, owns `State`/`Adversary`/`Rubric`, runs the 7-phase `step()`, returns OpenEnv-compliant tuples. **Holds the Interface Lock.** | Imports `classifier`, `adversary`, `Rubric`, `State`, `Action`, `Observation`. Read by `api/routes.py`. | FastAPI HTTP requests (`/reset`, `/step`) from `NegotiationEnvClient` or in-process tests. |
| `environment/config.py` | Static constants — `MAX_TURNS=15`, default thresholds, `CURRICULUM_STAGE`. | Imported by `env.py`, `adversary.py`, `rewards/rubric.py`. | Module load. |
| `environment/models/observation.py` | The 7-field `Observation` Pydantic model — *everything* the agent ever sees. Frozen by Interface Lock. | Returned by `env.reset()` and `env.step()`. Consumed by `client/utils.py` (prompt builder) and the trainer. | Built fresh inside `env.reset()` and `env.step()`. |
| `environment/models/state.py` | The hidden `State` Pydantic model — full ground truth. Read by all 7 reward functions. | Built by `env.py`; passed (`state_before`, `state_after`) to `Rubric.compose()`. | `env.reset()` (initial), `env.step()` (after adversary update). |
| `environment/models/action.py` | The `Action` Pydantic model — `action_type`, `text`, `metadata`. | Constructed in `env.step()` from the agent's `action_dict`; read by all 7 reward functions. | `env.step()`. |
| `reward.py` (top-level) | Seven reward functions + `compose()` combiner + `Rubric` wrapper. Pure functions of `(prev_state, curr_state, action)`. **Never reads `borrower_msg`.** | Imported by `rewards/rubric.py`; weights driven by `CURRICULUM_WEIGHTS`. | `Rubric.compose()` inside `env.step()` phase 6. |
| `rewards/rubric.py` | Thin wrapper around `reward.compose` that locks curriculum-stage weights for the episode. | Wraps `reward.compose`. | `env.reset()` instantiates it; `env.step()` calls `compose()`. |
| `environment/scenarios/profiles.py` + `curriculum.py` | The 20 hand-authored borrower personas + per-stage profile filters. Provides `anger_init`, personality, demands, opening_msg. | Read once per `reset()` via `get_profiles_for_stage()`. | `env.reset()`. |
| `tests/test_dynamic_voice.py` | Voice-specific test suite — Furious / Zen / Boundary / Cache / Interface cases. | Imports `ResponseGenerator`, `NegotiationEnv`, `Observation`. | `pytest` (manual) — live LLM tests gated on `NEG_LLM_ENABLED=1`. |
| `voice_proof_of_life.py` | Standalone harness that empirically proves NIM is being hit (Entity Nuance, Lexical Variance, Language Fluidity, State-Strictness) and that the cache works. | Imports `ResponseGenerator`. | Manual run with `NEG_LLM_ENABLED=1`. |
| `local_test.py` | End-to-end pipeline + red-team verifier. Runs 5 episodes against the live HTTP server (or a `DummyEnvClient` if offline) and checks all 7 reward keys + the post-audit asymmetries. | Talks to `api/routes.py` via `NegotiationEnvClient`. | Manual run after server start. |

---

### Appendix A — Environment variables

| Variable | Effect |
|---|---|
| `NEG_LLM_ENABLED=1` | Activates the Voice. Without this, every reply comes from the legacy template bank — exactly what the env did before the integration. |
| `NVIDIA_API_KEY=…` | Required when `NEG_LLM_ENABLED=1`. Passed to `openai.OpenAI(api_key=…)`. If missing, the generator silently falls back to templates. |

### Appendix B — Where determinism actually lives

The reward signal is a pure function of:

```
prev_state.{anger, trust, fear, terminated, termination_reason,
            demands_*, message_history, turn, max_turns}
curr_state.{ same fields }
action.{action_type, text, metadata}
```

…and the `Rubric`'s curriculum weights, which are frozen at `reset()`. None
of these are modified by the LLM. Therefore, for any fixed seed and any fixed
sequence of agent actions, the exact same scalar reward and the exact same
breakdown dict will be produced on every run, regardless of whether the
borrower's natural-language reply was a 70B-token sample, a cached 70B
sample, or a hardcoded template — which is exactly the property the GRPO
trainer needs.
