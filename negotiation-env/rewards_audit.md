# Reward System Audit - MetaX Negotiation Environment

## Scope

This audit is focused specifically on the reward subsystem and connected flow:

- `reward.py` (core reward logic and composition)
- `rewards/rubric.py` (runtime compatibility wrapper)
- `training/reward_bridge.py` (GRPO integration bridge)
- `environment/llm_judge.py` (qualitative judge used by reward composer)
- Reward-related contracts and tests (`contracts.py`, `tests/test_env_reward_integration.py`)

---

## Executive Summary

The reward system is a **hybrid deterministic + qualitative design** that is more mature than most RL environment reward stacks. It combines strict, explainable, low-latency deterministic checks with an optional LLM judge for soft-skill dimensions (empathy and strategy), while preserving robust fallback behavior.

Key strengths:

- Highly interpretable componentized reward structure
- Anti-gaming controls (compliance farming, repetition, filler stuffing checks)
- Curriculum-aware weighting for staged learning
- Strong runtime safeguards and fallback logic for LLM-dependent paths
- Detailed per-component telemetry (`raw`, `weight`, `weighted`, judge diagnostics)

Primary gaps:

- Reward module is large and dense, increasing maintenance complexity
- Some interfaces contain legacy shape drift (wrapper/stub style layers)
- No explicit CI-enforced reward regression harness for long-term reward stability

Overall rating for reward subsystem: **8.8 / 10**.

---

## Reward Architecture

## 1) Composition Model

Core entrypoint: `compose(prev_state, curr_state, action, weights=None)` in `reward.py`.

Primary reward components (contract-aligned):

- `outcome`
- `deescalation`
- `trust_building`
- `demand_coverage`
- `efficiency`
- `compliance`
- `anti_exploit`

Extended internal component:

- `format_compliance` (used in total sum and safety gating, though not in `REWARD_BREAKDOWN_KEYS`)

Composition flow:

1. Compute deterministic raw components first.
2. Run short-circuit gate:
   - If `format_compliance < 0` or `compliance < 0`, skip LLM judge scoring.
3. If gate passes, use `llm_judge.score_response()` to populate:
   - `deescalation <- empathy_score`
   - `trust_building <- strategy_score`
4. Compute weighted total and detailed breakdown fields.

This is a strong industry pattern: deterministic safeguards first, expensive qualitative scoring second.

## 2) Curriculum Weighting

`CURRICULUM_WEIGHTS` in `reward.py` defines stage-specific shaping:

- Stage 1: outcome/de-escalation dominated, minimal complexity
- Stage 2: introduces trust and demand coverage
- Stage 3/4: full stack including anti-exploit and efficiency

This staged strategy aligns with modern RL curriculum practice and avoids overwhelming early policy learning with too many competing objectives.

---

## Detailed Component Audit

## `reward_outcome`

Behavior:

- `+1.0` for `commitment_reached`
- `-0.5` for `anger_threshold_crossed`
- `0.0` otherwise (timeout, forbidden, mid-episode)

Assessment:

- Clear terminal anchor signal.
- Good for convergence direction.
- Conservative penalty on escalation may be tuned upward if models learn risky behavior.

## `reward_deescalation`

Behavior:

- Uses anger delta, normalized by 10
- Applies asymmetric penalty amplifier when anger rises

Assessment:

- Good asymmetry design; discourages oscillation tricks.
- Deterministic and stable.

## `reward_trust_building`

Behavior:

- Composite of trust delta + fear-based emotional signal
- Asymmetric penalty for trust drops

Assessment:

- Better than single-axis trust shaping.
- Fear coupling improves realism; still interpretable.

## `reward_demand_coverage`

Behavior:

- Terminal-only coverage score based on addressed vs all demands

Assessment:

- Correctly ties strategic completeness to episode outcome phase.
- Dependence on `demands_addressed` quality means upstream tracking precision is critical.

## `reward_efficiency`

Behavior:

- Bonus only on successful resolution
- Faster successful closure -> higher bonus

Assessment:

- Good antidote to unnecessarily long episodes.
- Balanced correctly by requiring successful resolution first.

## `reward_compliance`

Behavior:

- Validates action type, non-empty text, minimum word count, required metadata, and forbidden/coercive language
- Returns `+0.1` pass, `-0.2` fail

Assessment:

- Strong anti-farming and policy safety control.
- Word-count gate is practical, but can over-penalize concise yet valid responses in some edge contexts.

## `reward_anti_exploit`

Behavior:

- Similarity check between recent messages (TF-IDF cosine, with fallback overlap heuristic)
- Filler phrase stuffing detection with overlap-safe counting
- Additive penalties

Assessment:

- Very good anti-degenerate defense.
- Threshold strategy is explicit and documented.

## `reward_format_compliance`

Behavior:

- Rewards correctly structured JSON output and valid action_type
- Penalizes decode failures or missing required fields

Assessment:

- Crucial for structured action contract reliability.
- Directly supports parser/train stability.

---

## LLM Judge Integration Audit

Files:

- `environment/llm_judge.py`
- Integrated via `reward.py::compose`

Strengths:

- Disabled-by-default for hermetic testability
- Hard timeout bounds, zero-vector fallback on failure
- Strict parse/coerce/clamp behavior
- Does not throw exceptions into hot training loop

Important design win:

- Short-circuiting judge when deterministic guard fails prevents “bad output + empathy bonus” contradictions and reduces cost/latency.

Industry comparison:

- Better than all-LLM judging and better than pure heuristics for soft skills.
- This hybrid architecture is currently a recommended compromise in production RLHF-style systems with strict runtime budgets.

---

## GRPO Reward Bridge Audit

File: `training/reward_bridge.py`

Strengths:

- Explicitly avoids per-completion reset bug (critical fix)
- Clips reward using numeric contract bounds
- Handles multiple completion payload shapes safely

Known limitation (correctly documented in code):

- Per-completion scoring remains approximated temporal credit assignment (single-turn bias risk).

Industry comparison:

- The bridge design is practical and transparent for GRPO constraints.
- For stronger delayed-credit behavior, trajectory-aware objectives or replay-based critics are usually introduced.

---

## Testing Coverage for Rewards

Current direct checks:

- `tests/test_env_reward_integration.py`: integration smoke + monkeypatched compose
- Broad hardening in `tests/test_edge_cases.py` supports upstream stability that reward relies on indirectly

Gap:

- Missing explicit unit test matrix per reward component function with fixed fixtures and expected numeric outputs.

Recommendation:

- Add a deterministic reward golden-suite:
  - fixture pairs of (`prev_state`, `curr_state`, `action`)
  - expected component outputs
  - expected weighted totals per curriculum stage

This prevents subtle behavioral drift as heuristics evolve.

---

## Comparison with Current Industry Practices

## Where this reward system is better

1. **Explainability-first composition:** Detailed breakdown beyond a scalar reward.
2. **Built-in anti-gaming controls:** Compliance and exploit protections are explicit.
3. **Hybrid judge strategy:** Efficient balance of hard rules + soft qualitative scoring.
4. **Runtime resilience:** Non-crashing fallback behavior across judge and parser paths.
5. **Curriculum shaping:** Stage-specific objectives reduce early training confusion.

## Where industry-standard production stacks still go further

1. **Automated reward regression CI:** every PR validated against frozen reward expectations.
2. **Formal reward spec docs versioning:** change-managed reward contracts across teams.
3. **Online calibration dashboards:** monitor component drift and reward hacking signals over training time.
4. **A/B tested reward variants:** systematic experimentation framework for weight and component evolution.

---

## Example Dry Run - Reward Calculation

Scenario:

- Non-terminal step
- Agent sends well-formed empathetic + structured negotiation message
- No compliance violation
- No repetition exploit

Input sketch:

- `prev_state`: anger `6.0`, trust `3.0`, fear `5.0`, turn `3`, not terminated
- `curr_state`: anger `5.2`, trust `3.6`, fear `4.8`, turn `4`, not terminated
- `action`:
  - `action_type`: `offer_emi`
  - `text`: meaningful non-threatening sentence
  - `metadata`: includes `emi_amount`
  - `metadata.raw_text`: valid JSON contract string

Stage: `stage_3` weights.

### Step 1: Deterministic raws

- `outcome = 0.0` (not terminal)
- `demand_coverage = 0.0` (terminal-only)
- `efficiency = 0.0` (terminal success only)
- `compliance = +0.1` (passes checks)
- `anti_exploit = 0.0` (no repetition/stuffing)
- `format_compliance = +0.20` (valid JSON + valid action type)

`deescalation` and `trust_building` temporarily `0.0` placeholders before judge branch.

### Step 2: Short-circuit gate

- Gate condition checks:
  - `format_compliance < 0`? No
  - `compliance < 0`? No
- Judge is allowed to run.

### Step 3: LLM judge mapping

Assume judge returns:

- `empathy_score = 0.7`
- `strategy_score = 0.8`

Then composer sets:

- `deescalation = 0.7`
- `trust_building = 0.8`

### Step 4: Weighted total (illustrative)

Stage 3 weights:

- `outcome 1.0`
- `deescalation 0.3`
- `trust_building 0.3`
- `demand_coverage 0.2`
- `efficiency 0.1`
- `compliance 0.1`
- `anti_exploit 1.0`
- plus internal `format_compliance 0.5`

Total:

- `0.3*0.7 = 0.21`
- `0.3*0.8 = 0.24`
- `0.1*0.1 = 0.01`
- `0.5*0.2 = 0.10`
- others `0.0`

Approx scalar reward = `0.56`.

Breakdown will also include:

- `raw_*`, `weight_*`, `weighted_*`
- `judge_used`, `judge_reason`, `judge_latency_ms`, `judge_fallback_used`, `judge_short_circuited`

### Failure-path dry run (short-circuit case)

If action text includes threat language:

- `compliance = -0.2`
- Short-circuit triggers
- Judge skipped
- `deescalation = 0.0`, `trust_building = 0.0`
- Total driven negative by compliance and possibly anti-exploit/other penalties

This is exactly the intended reward safety behavior.

---

## Priority Recommendations

### P0

- Add deterministic golden tests for each reward function and total compose outputs.
- Add CI pipeline to enforce reward regression checks before merge.

### P1

- Split `reward.py` into modular files (one component per file) while preserving public compose API.
- Formalize reward spec versioning document (`reward_spec_vX.md`) with change log.

### P2

- Add calibration dashboard for reward breakdown trends across training runs.
- Evaluate action-type-specific normalization for more stable cross-policy comparisons.

---

## Final Verdict

The reward subsystem is one of the strongest parts of this repository: transparent, robust, anti-gaming-aware, and operationally practical for RL training at scale. With regression-test formalization and light modular refactor, it would meet a high bar for both research and production-oriented experimentation.
