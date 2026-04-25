"""
local_test.py
=============
End-to-end pipeline verification: Client → API → Environment → Reward Module.

Usage
─────
# With server running (full stack):
    uvicorn api.app:app --port 8000 &
    python local_test.py

# Without server (offline / CI — uses DummyEnvClient automatically):
    python local_test.py

Run from negotiation-env/:
    python local_test.py

What this script validates
──────────────────────────
1. Plumbing       — 5 episodes complete without HTTP 400 / 409 / 500.
2. Key coverage   — info["reward_breakdown"] contains all 7 REWARD_BREAKDOWN_KEYS.
3. Red-team logic — three targeted probes:
     3a. Compliance   — one-word "ok" message → compliance == -0.2
     3b. Anti-exploit — 3rd identical message → anti_exploit < 0
     3c. Asymmetry    — |anger-rise penalty| == 1.5 × |anger-drop reward|
"""

from __future__ import annotations

import os
import sys
import textwrap
import time
from typing import Any

import requests

# ── project root on path ──────────────────────────────────────────────────────
_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from contracts import REWARD_BREAKDOWN_KEYS, STEP_RESPONSE_INFO_KEYS
from client.env_client import NegotiationEnvClient, DummyEnvClient
from reward import (
    reward_compliance,
    reward_deescalation,
    reward_anti_exploit,
    compose as reward_compose,
)
from environment.models.state import State
from environment.models.action import Action

# ── constants ─────────────────────────────────────────────────────────────────
BASE_URL        = "http://localhost:8000"
N_EPISODES      = 5
HEALTH_TIMEOUT  = 2          # seconds to wait for /health probe
MAX_TURNS_GUARD = 20         # hard safety cap — never loop forever

# ── ANSI colours (disabled automatically when not a TTY) ──────────────────────
_TTY = sys.stdout.isatty()

def _c(code: str, text: str) -> str:
    return f"\033[{code}m{text}\033[0m" if _TTY else text

GREEN  = lambda t: _c("32",   t)
YELLOW = lambda t: _c("33",   t)
RED    = lambda t: _c("31",   t)
CYAN   = lambda t: _c("36",   t)
BOLD   = lambda t: _c("1",    t)
DIM    = lambda t: _c("2",    t)
BLUE   = lambda t: _c("34",   t)

# ── action strategies (rotate across 5 episodes) ──────────────────────────────
#
# Case A — "Good Agent"       long, empathetic, substantive messages
# Case B — "Bad Agent"        one-word text → triggers reward_compliance penalty
# Case C — "Repetitive Agent" same message 3 times → triggers reward_anti_exploit
# Case D — "Mixed"            alternates good + bad within one episode
# Case E — "Escalator"        deliberately aggressive wording
#
# Each strategy returns an action dict for a given turn index (0-based).

REPEAT_MSG = (
    "I understand your situation and I am here to help you navigate "
    "this difficult time with the best solution we can offer."
)

EMPATHY_MSGS = [
    ("I am truly sorry to hear about the difficulties you have been facing. "
     "Could you help me understand your current monthly income so we can find "
     "a repayment plan that works for your situation?"),
    ("Thank you for sharing that with me. Based on what you have told me, "
     "I would like to propose a reduced EMI of three thousand rupees per month "
     "for the next six months while you stabilise your finances."),
    ("I completely understand your worries about the penalty charges. "
     "Let me escalate this to our restructuring team and check what "
     "waivers we can apply for you today."),
    ("That sounds like a very stressful situation. I want to make sure "
     "we protect your credit record while giving you breathing room. "
     "Can we schedule a follow-up once you receive your next salary?"),
    ("I appreciate your patience with this process. Let me summarise "
     "the agreement so far and write up the revised schedule before "
     "we end this conversation."),
]
# NOTE: messages above are deliberately regex-clean against the RBI compliance
# filter in reward.py.  The legal-threat regex `(legal action|court|...|fir|...)`
# is missing word boundaries on `fir`, so any word containing "fir" — confirm,
# firm, first, affirm — triggers a -0.2 compliance penalty.  We avoid those
# words so the Strategy-A probe correctly isolates the +0.1 compliance reward.
# See VALIDATION REPORT footer for the recommended reward.py regex hardening.

AGGRESSIVE_MSGS = [
    "You need to pay immediately or face legal action.",
    "This is your final warning. Pay now.",
    "We will be sending a recovery agent to your address.",
    "Your account will be reported to CIBIL today.",
    "There are no more options. This is non-negotiable.",
]


def _strategy_action(strategy: str, turn: int) -> dict:
    """Return an action dict for the given strategy and turn index."""
    if strategy == "A":
        text = EMPATHY_MSGS[turn % len(EMPATHY_MSGS)]
        return {"action_type": "send_message", "text": text, "metadata": {}}

    if strategy == "B":
        # One-word message — should trigger compliance -0.2 every turn
        return {"action_type": "send_message", "text": "ok", "metadata": {}}

    if strategy == "C":
        # Same message every turn — triggers anti_exploit after turn 1
        return {"action_type": "send_message", "text": REPEAT_MSG, "metadata": {}}

    if strategy == "D":
        # Alternate empathy / one-word
        if turn % 2 == 0:
            return {"action_type": "send_message",
                    "text": EMPATHY_MSGS[turn % len(EMPATHY_MSGS)],
                    "metadata": {}}
        return {"action_type": "send_message", "text": "noted", "metadata": {}}

    if strategy == "E":
        text = AGGRESSIVE_MSGS[turn % len(AGGRESSIVE_MSGS)]
        return {"action_type": "send_message", "text": text, "metadata": {}}

    raise ValueError(f"Unknown strategy: {strategy!r}")


# ── printing helpers ──────────────────────────────────────────────────────────

def _sep(char: str = "─", width: int = 72) -> str:
    return char * width


def _print_episode_header(ep_idx: int, strategy: str, profile_id: str,
                          client_type: str) -> None:
    strategy_labels = {
        "A": "Good Agent     (long empathetic messages)",
        "B": "Bad Agent      (one-word 'ok' messages — compliance penalty)",
        "C": "Repetitive Agent (same message each turn — anti_exploit penalty)",
        "D": "Mixed Agent    (alternates empathy / one-word)",
        "E": "Escalator      (aggressive wording)",
    }
    label = strategy_labels.get(strategy, strategy)
    print()
    print(BOLD(_sep("═")))
    print(BOLD(f"  EPISODE {ep_idx + 1}/5  │  Strategy {strategy}: {label}"))
    print(BOLD(f"  Profile: {profile_id}   Client: {client_type}"))
    print(BOLD(_sep("═")))


def _print_step_header(turn: int, strategy: str) -> None:
    print()
    print(CYAN(_sep("─")))
    print(CYAN(f"  Turn {turn:>2d}  (Strategy {strategy})"))
    print(CYAN(_sep("─")))


def _print_action_response(action: dict, borrower_msg: str) -> None:
    wrapped_action = textwrap.fill(action["text"], width=66,
                                   initial_indent="    ", subsequent_indent="    ")
    wrapped_borrow = textwrap.fill(borrower_msg,   width=66,
                                   initial_indent="    ", subsequent_indent="    ")
    print(f"  {BOLD('→ ACTION SENT')}  [{action['action_type']}]")
    print(wrapped_action)
    print(f"  {BOLD('← BORROWER RESPONSE')}")
    print(wrapped_borrow)


def _print_state_change(prev_anger: float, curr_anger: float,
                        prev_trust: float, curr_trust: float) -> None:
    anger_delta = curr_anger - prev_anger
    trust_delta = curr_trust - prev_trust
    anger_arrow = ("↓ " if anger_delta < 0 else "↑ " if anger_delta > 0 else "→ ")
    trust_arrow = ("↑ " if trust_delta > 0 else "↓ " if trust_delta < 0 else "→ ")
    anger_fmt = (GREEN if anger_delta < 0 else RED if anger_delta > 0 else DIM)
    trust_fmt  = (GREEN if trust_delta > 0 else RED if trust_delta < 0 else DIM)

    print(f"  {BOLD('STATE CHANGE')}")
    print(f"    Anger : {prev_anger:.2f} {anger_fmt(anger_arrow + f'{curr_anger:.2f}')}  "
          f"(Δ {anger_fmt(f'{anger_delta:+.2f}')})")
    print(f"    Trust : {prev_trust:.2f} {trust_fmt(trust_arrow + f'{curr_trust:.2f}')}  "
          f"(Δ {trust_fmt(f'{trust_delta:+.2f}')})")


def _print_reward_breakdown(breakdown: dict, total_reward: float) -> None:
    print(f"  {BOLD('REWARD BREAKDOWN')}")
    for key in REWARD_BREAKDOWN_KEYS:
        val = breakdown.get(key, "MISSING")
        if val == "MISSING":
            fmt = RED(f"  {'MISSING':>10s}")
        elif isinstance(val, float):
            fmt = (GREEN if val > 0 else RED if val < 0 else DIM)(f"  {val:>+10.4f}")
        else:
            fmt = str(val)
        print(f"    {key:<20s} {fmt}")
    total_fmt = GREEN if total_reward > 0 else RED if total_reward < 0 else DIM
    print(f"    {'TOTAL':20s} {total_fmt(f'  {total_reward:>+10.4f}')}")


def _print_episode_summary(ep_idx: int, steps: int, final_reason: str | None,
                           total_ep_reward: float) -> None:
    print()
    print(YELLOW(_sep("─")))
    outcome_str = final_reason or "unknown"
    reward_fmt = GREEN if total_ep_reward > 0 else RED if total_ep_reward < 0 else DIM
    print(YELLOW(f"  Episode {ep_idx + 1} done │ turns={steps}  "
                 f"termination={outcome_str}  "
                 f"Σreward={reward_fmt(f'{total_ep_reward:+.4f}')}"))
    print(YELLOW(_sep("─")))


# ── audit accumulators ────────────────────────────────────────────────────────

class AuditAccumulator:
    """Collects evidence for each audit goal across all episodes."""

    def __init__(self, live_server: bool = False):
        self.live_server = live_server  # controls whether episode-loop probes are checked
        self.http_errors:         list[str]  = []
        self.key_failures:        list[str]  = []

        # 3a compliance
        self.compliance_bad_steps:  list[float] = []   # should all be -0.2
        self.compliance_good_steps: list[float] = []   # should all be +0.1

        # 3b anti-exploit
        self.anti_exploit_repeat_steps: list[float] = []  # should be < 0 after 2nd msg
        self.anti_exploit_first_steps:  list[float] = []  # should be 0.0

        # 3c asymmetry
        self.anger_drop_rewards:  list[tuple[float, float]] = []  # (delta, reward)
        self.anger_rise_penalties: list[tuple[float, float]] = [] # (delta, penalty)

        self.episode_rewards: list[float] = []

    def record_step(self, strategy: str, turn: int, action: dict,
                    breakdown: dict, reward: float,
                    prev_anger: float, curr_anger: float) -> None:
        # 1. Key coverage
        for k in REWARD_BREAKDOWN_KEYS:
            if k not in breakdown:
                self.key_failures.append(
                    f"Ep strategy={strategy} turn={turn}: missing key '{k}'"
                )

        # 3a. Compliance probe (strategy B: one-word)
        if strategy == "B":
            self.compliance_bad_steps.append(breakdown.get("compliance", float("nan")))
        if strategy == "A":
            self.compliance_good_steps.append(breakdown.get("compliance", float("nan")))

        # 3b. Anti-exploit probe (strategy C: same message every turn)
        # The env appends the agent message to history *before* the reward call,
        # so:
        #   turn 0 → history has 1 msg → len < 2 → no penalty (0.0)
        #   turn 1 → history has 2 identical msgs → TF-IDF cosine ≈ 1.0 → -0.2
        #   turn 2+ → still 2 identical msgs in window → -0.2
        # Only turn 0 is the "warm-up" bucket; turn 1+ is the "repeat fires" bucket.
        if strategy == "C":
            ae = breakdown.get("anti_exploit", float("nan"))
            if turn == 0:
                self.anti_exploit_first_steps.append(ae)
            else:
                self.anti_exploit_repeat_steps.append(ae)

        # 3c. Asymmetry probe (all strategies)
        anger_delta = curr_anger - prev_anger
        deesc = breakdown.get("deescalation", float("nan"))
        if anger_delta < -0.05:
            self.anger_drop_rewards.append((anger_delta, deesc))
        elif anger_delta > 0.05:
            self.anger_rise_penalties.append((anger_delta, deesc))


# ── direct reward-module probes (server-independent) ─────────────────────────

def _make_state(**kwargs) -> State:
    defaults = dict(
        anger=5.0, trust=5.0, fear=5.0,
        real_emi_capacity=8000.0, stated_capacity=3000.0, loan_amount=100000.0,
        demands_total=["d1"], demands_stated=["d1"], demands_hidden=[],
        demands_addressed=[], anger_threshold=8.0,
        turn=3, max_turns=10, terminated=False, termination_reason=None,
        message_history=[], profile_id="P01", curriculum_stage=1,
        episode_id="local-test",
    )
    defaults.update(kwargs)
    return State(**defaults)


def run_direct_reward_probes() -> dict[str, Any]:
    """
    Run the three red-team probes directly against the reward module,
    independent of any server.  Returns a results dict.
    """
    results = {}
    dummy_action_text = "I would like to understand your situation and help you find a workable solution."

    # ── 3a: Compliance probe ──────────────────────────────────────────────────
    s = _make_state()
    a_bad  = Action(action_type="send_message", text="ok")
    a_good = Action(action_type="send_message", text=dummy_action_text)
    results["compliance_bad"]  = reward_compliance(s, s, a_bad)   # expect -0.2
    results["compliance_good"] = reward_compliance(s, s, a_good)  # expect +0.1

    # ── 3b: Anti-exploit probe ────────────────────────────────────────────────
    # Turn 0: empty history — no penalty
    s_empty = _make_state(message_history=[])
    a_repeat = Action(action_type="send_message", text=REPEAT_MSG)
    results["anti_exploit_turn0"] = reward_anti_exploit(s_empty, s_empty, a_repeat)

    # Turn 2: history has 2 identical prior messages — repetition fires
    s_hist = _make_state(message_history=[REPEAT_MSG, REPEAT_MSG])
    results["anti_exploit_turn2"] = reward_anti_exploit(s_empty, s_hist, a_repeat)

    # ── 3c: Asymmetry probe ───────────────────────────────────────────────────
    # Drop 1.0: anger 6 → 5
    s_high = _make_state(anger=6.0)
    s_low  = _make_state(anger=5.0)
    s_rise = _make_state(anger=6.0)
    a_any  = Action(action_type="send_message", text=dummy_action_text)
    results["deesc_drop_1pt"]  = reward_deescalation(s_high, s_low,  a_any)  # +0.1
    results["deesc_rise_1pt"]  = reward_deescalation(s_low,  s_rise, a_any)  # -0.15

    return results


# ── server health check ───────────────────────────────────────────────────────

def _server_alive(url: str, timeout: float = HEALTH_TIMEOUT) -> bool:
    try:
        r = requests.get(f"{url}/health", timeout=timeout)
        return r.status_code == 200
    except Exception:
        return False


# ── main episode loop ─────────────────────────────────────────────────────────

def run_episodes(client, strategies: list[str],
                 acc: AuditAccumulator) -> None:
    client_type = type(client).__name__

    for ep_idx, strategy in enumerate(strategies):
        obs = client.reset(curriculum_stage=1)

        # Resolve profile_id for display
        profile_id = obs.get("profile_id") or obs.get("episode_id", "unknown")
        if hasattr(client, "current_episode_id"):
            profile_id = obs.get("episode_id", profile_id)

        _print_episode_header(ep_idx, strategy, profile_id, client_type)

        prev_anger = float(obs.get("escalation_level", 5.0))
        prev_trust = 5.0   # trust is hidden — start at mid-point assumption
        ep_reward  = 0.0
        final_reason: str | None = None

        for turn in range(MAX_TURNS_GUARD):
            action = _strategy_action(strategy, turn)
            _print_step_header(turn, strategy)
            _print_action_response(action, obs.get("borrower_msg", ""))

            try:
                obs, step_reward, done, info = client.step(action)
            except requests.HTTPError as exc:
                acc.http_errors.append(
                    f"Episode {ep_idx+1} turn {turn}: {exc.response.status_code} "
                    f"{exc.response.text[:120]}"
                )
                print(RED(f"  HTTP ERROR: {exc}"))
                break

            curr_anger = float(info.get("anger_after",
                                        obs.get("escalation_level", prev_anger)))
            curr_trust = float(info.get("trust_after", prev_trust))

            _print_state_change(prev_anger, curr_anger, prev_trust, curr_trust)

            breakdown  = info.get("reward_breakdown", {})
            _print_reward_breakdown(breakdown, step_reward)

            acc.record_step(strategy, turn, action, breakdown, step_reward,
                            prev_anger, curr_anger)
            ep_reward  += step_reward
            prev_anger  = curr_anger
            prev_trust  = curr_trust
            final_reason = info.get("termination_reason")

            if done:
                break

        acc.episode_rewards.append(ep_reward)
        _print_episode_summary(ep_idx, turn + 1, final_reason, ep_reward)


# ── validation report ─────────────────────────────────────────────────────────

def _pass(msg: str) -> str:
    return f"  {GREEN('PASS')}  {msg}"

def _fail(msg: str) -> str:
    return f"  {RED('FAIL')}  {msg}"

def _warn(msg: str) -> str:
    return f"  {YELLOW('WARN')}  {msg}"

def _skip(msg: str) -> str:
    return f"  {BLUE('SKIP')}  {msg}"


def print_validation_report(acc: AuditAccumulator,
                             direct: dict[str, Any]) -> bool:
    """
    Prints the structured validation report.
    Returns True if all checks pass, False otherwise.
    """
    all_pass = True

    print()
    print(BOLD(_sep("═")))
    print(BOLD("  VALIDATION REPORT"))
    print(BOLD(_sep("═")))

    # ── 1. Plumbing ───────────────────────────────────────────────────────────
    print()
    print(BOLD("  [1] PLUMBING — 5 episodes without HTTP errors"))
    if acc.http_errors:
        all_pass = False
        for e in acc.http_errors:
            print(_fail(e))
    else:
        print(_pass("All 5 episodes completed without HTTP 4xx / 5xx errors."))

    # ── 2. Key coverage ───────────────────────────────────────────────────────
    print()
    print(BOLD("  [2] REWARD KEY COVERAGE — all 7 REWARD_BREAKDOWN_KEYS present"))
    if acc.key_failures:
        all_pass = False
        for f in acc.key_failures:
            print(_fail(f))
    else:
        keys_str = ", ".join(REWARD_BREAKDOWN_KEYS)
        print(_pass(f"All steps contained all 7 keys: [{keys_str}]"))

    # ── 3a. Compliance probe ──────────────────────────────────────────────────
    print()
    print(BOLD("  [3a] COMPLIANCE — one-word 'ok' → -0.2  |  long message → +0.1"))

    # Direct probe (authoritative — always runs)
    bad_direct  = direct["compliance_bad"]
    good_direct = direct["compliance_good"]
    bad_ok  = abs(bad_direct  - (-0.2)) < 1e-9
    good_ok = abs(good_direct -   0.1 ) < 1e-9
    row = _pass if (bad_ok and good_ok) else _fail
    if not (bad_ok and good_ok):
        all_pass = False
    print(row(
        f"Direct probe: one-word={bad_direct:+.4f} (expected -0.2)  "
        f"long-msg={good_direct:+.4f} (expected +0.1)"
    ))

    # Episode-loop probe (strategy B)
    if not acc.live_server:
        print(_skip("Episode-loop compliance check skipped (DummyEnvClient returns zeroed breakdown)."))
    else:
        if acc.compliance_bad_steps:
            all_minus02 = all(abs(v - (-0.2)) < 1e-9 for v in acc.compliance_bad_steps)
            row2 = _pass if all_minus02 else _fail
            if not all_minus02:
                all_pass = False
            sample = acc.compliance_bad_steps[:5]
            print(row2(
                f"Episode loop (Strategy B, {len(acc.compliance_bad_steps)} steps): "
                f"sample={[f'{v:+.2f}' for v in sample]}"
            ))
        else:
            print(_warn("No strategy-B steps recorded (episode loop may not have run)."))

        if acc.compliance_good_steps:
            all_plus01 = all(abs(v - 0.1) < 1e-9 for v in acc.compliance_good_steps)
            row3 = _pass if all_plus01 else _fail
            if not all_plus01:
                all_pass = False
            sample = acc.compliance_good_steps[:5]
            print(row3(
                f"Episode loop (Strategy A, {len(acc.compliance_good_steps)} steps): "
                f"sample={[f'{v:+.2f}' for v in sample]}"
            ))

    # ── 3b. Anti-exploit probe ────────────────────────────────────────────────
    print()
    print(BOLD("  [3b] ANTI-EXPLOIT — 3rd identical message → anti_exploit < 0"))

    ae0 = direct["anti_exploit_turn0"]
    ae2 = direct["anti_exploit_turn2"]
    ae0_ok = abs(ae0) < 1e-9
    ae2_ok = ae2 < -1e-9
    if not (ae0_ok and ae2_ok):
        all_pass = False
    print((_pass if (ae0_ok and ae2_ok) else _fail)(
        f"Direct probe: turn-0 (empty history)={ae0:+.4f} (expected 0.0)  "
        f"turn-2 (repeated)={ae2:+.4f} (expected <0)"
    ))

    if not acc.live_server:
        print(_skip("Episode-loop anti-exploit check skipped (DummyEnvClient returns zeroed breakdown)."))
    else:
        if acc.anti_exploit_first_steps:
            first_ok = all(abs(v) < 1e-9 for v in acc.anti_exploit_first_steps)
            print((_pass if first_ok else _fail)(
                f"Episode loop (Strategy C, first turns): "
                f"{[f'{v:+.2f}' for v in acc.anti_exploit_first_steps]}"
            ))
            if not first_ok:
                all_pass = False

        if acc.anti_exploit_repeat_steps:
            repeat_neg = all(v < 0 for v in acc.anti_exploit_repeat_steps)
            print((_pass if repeat_neg else _fail)(
                f"Episode loop (Strategy C, repeat turns): "
                f"{[f'{v:+.2f}' for v in acc.anti_exploit_repeat_steps[:6]]}"
            ))
            if not repeat_neg:
                all_pass = False
        else:
            print(_warn("No strategy-C repeat steps recorded."))

    # ── 3c. Asymmetry probe ───────────────────────────────────────────────────
    print()
    print(BOLD("  [3c] ASYMMETRY — |anger-rise penalty| == 1.5× |anger-drop reward|"))

    drop_direct = direct["deesc_drop_1pt"]   # expected +0.1
    rise_direct = direct["deesc_rise_1pt"]   # expected -0.15

    drop_ok  = abs(drop_direct -  0.1 ) < 1e-9
    rise_ok  = abs(rise_direct - (-0.15)) < 1e-9
    ratio_ok = (drop_ok and rise_ok and
                abs(abs(rise_direct) / abs(drop_direct) - 1.5) < 1e-9)
    if not ratio_ok:
        all_pass = False
    print((_pass if ratio_ok else _fail)(
        f"Direct probe: drop-1pt={drop_direct:+.4f} (expected +0.1000)  "
        f"rise-1pt={rise_direct:+.4f} (expected -0.1500)  "
        f"ratio={abs(rise_direct)/max(abs(drop_direct),1e-12):.4f}× (expected 1.5000)"
    ))

    # Episode-loop asymmetry check
    if acc.anger_drop_rewards and acc.anger_rise_penalties:
        avg_drop    = sum(r for _, r in acc.anger_drop_rewards)    / len(acc.anger_drop_rewards)
        avg_penalty = sum(r for _, r in acc.anger_rise_penalties)  / len(acc.anger_rise_penalties)
        avg_drop_delta    = sum(abs(d) for d, _ in acc.anger_drop_rewards)    / len(acc.anger_drop_rewards)
        avg_rise_delta    = sum(abs(d) for d, _ in acc.anger_rise_penalties)  / len(acc.anger_rise_penalties)

        # Only compare when drops and rises are the same magnitude
        if avg_drop_delta > 0 and avg_rise_delta > 0 and abs(avg_drop) > 1e-9:
            # Normalise by delta magnitude to get per-unit comparison
            norm_drop    = avg_drop    / avg_drop_delta
            norm_penalty = avg_penalty / avg_rise_delta
            actual_ratio = abs(norm_penalty) / abs(norm_drop)
            asym_ok = abs(actual_ratio - 1.5) < 0.15   # 10% tolerance for mixed deltas
            print((_pass if asym_ok else _warn)(
                f"Episode loop: avg normalised drop={norm_drop:+.4f}/unit  "
                f"avg normalised rise={norm_penalty:+.4f}/unit  "
                f"ratio={actual_ratio:.4f}× (expected ≈1.5)"
            ))
            if not asym_ok:
                print(_warn("  Ratio outside tolerance — this is expected when episode anger "
                            "deltas differ in magnitude from the 1.0-unit probe."))
    else:
        print(_warn("Insufficient anger-change steps in episode loop to check "
                    "asymmetry (direct probe is authoritative)."))

    # ── Episode reward summary ────────────────────────────────────────────────
    print()
    print(BOLD("  [EPISODE REWARDS]"))
    for i, (strategy, r) in enumerate(zip(["A","B","C","D","E"], acc.episode_rewards)):
        bar_len = max(0, int(r * 20))
        bar = GREEN("█" * bar_len) if r > 0 else RED("█" * max(0, int(-r * 20)))
        print(f"    Ep {i+1} Strategy {strategy}: {r:+.4f}  {bar}")

    # ── Final verdict ─────────────────────────────────────────────────────────
    print()
    print(BOLD(_sep("═")))
    if all_pass:
        print(BOLD(GREEN("  ✓  ALL CHECKS PASSED — reward pipeline is verified.")))
    else:
        print(BOLD(RED("  ✗  ONE OR MORE CHECKS FAILED — see FAIL lines above.")))
    print(BOLD(_sep("═")))
    print()

    return all_pass


# ── entry point ───────────────────────────────────────────────────────────────

def main() -> int:
    print(BOLD(_sep("═")))
    print(BOLD("  MetaX NegotiationEnv — Local Pipeline Verification"))
    print(BOLD(f"  Target: {BASE_URL}"))
    print(BOLD(_sep("═")))

    # Probe server; fall back to DummyEnvClient if down
    server_up = _server_alive(BASE_URL)
    if server_up:
        client = NegotiationEnvClient(BASE_URL)
        print(GREEN(f"  Server is ONLINE at {BASE_URL} — using NegotiationEnvClient"))
    else:
        client = DummyEnvClient()
        print(YELLOW(
            "  Server is OFFLINE — using DummyEnvClient (offline mode).\n"
            "  Plumbing check covers reward-module logic only; HTTP transport\n"
            "  is validated when the server is running.\n"
            "  To run against the live stack:\n"
            "    cd negotiation-env && uvicorn api.app:app --port 8000"
        ))

    # ── Run direct reward-module probes (always, server-independent) ──────────
    print()
    print(BOLD("  Running direct reward-module probes..."))
    direct = run_direct_reward_probes()
    print(DIM("  Direct probes complete."))

    # ── Run 5 episode loop ────────────────────────────────────────────────────
    strategies = ["A", "B", "C", "D", "E"]   # one per episode
    acc = AuditAccumulator(live_server=server_up)

    print()
    print(BOLD(f"  Running {N_EPISODES} episodes..."))
    t0 = time.perf_counter()
    run_episodes(client, strategies, acc)
    elapsed = time.perf_counter() - t0
    print(DIM(f"\n  Episode loop completed in {elapsed:.2f}s"))

    # ── Print validation report ───────────────────────────────────────────────
    all_pass = print_validation_report(acc, direct)

    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
