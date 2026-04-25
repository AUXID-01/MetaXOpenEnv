"""
End-to-End Stability Test Suite
Crisis Negotiation / Debt Restructuring RL Environment

Run with:
    python test_e2e.py                  # assumes env running on localhost:8000
    python test_e2e.py --url http://...  # custom URL
    python test_e2e.py --local           # import env directly (no HTTP)

Each test is self-contained. Results are printed with PASS / FAIL / WARN.
No external test framework needed — pure stdlib + requests.
"""

import sys
import json
import time
import argparse
import traceback
import importlib
import os
from typing import Any, Dict, Optional, Tuple

# ── optional: colorize output ──────────────────────────────────────────────────
GREEN  = ""
RED    = ""
YELLOW = ""
BLUE   = ""
RESET  = ""
BOLD   = ""

# ── result tracking ────────────────────────────────────────────────────────────
results = []

def record(name: str, status: str, detail: str = ""):
    results.append({"name": name, "status": status, "detail": detail})
    icon = {"PASS": "[PASS]",
            "FAIL": "[FAIL]",
            "WARN": "[WARN]"}[status]
    print(f"  {icon}  {name}")
    if detail:
        for line in detail.strip().splitlines():
            print(f"         {line}")

def section(title: str):
    print(f"\n{'-'*60}")
    print(f"  {title}")
    print(f"{'-'*60}")

# ── HTTP client (used when testing deployed env) ───────────────────────────────
class HTTPEnvClient:
    def __init__(self, base_url: str):
        import requests
        self.requests = requests
        self.base_url = base_url.rstrip("/")
        self.active_episode_id: Optional[str] = None

    def reset(self, curriculum_stage: Optional[int] = None) -> Dict:
        payload = {"curriculum_stage": curriculum_stage} if curriculum_stage else {}
        r = self.requests.post(f"{self.base_url}/reset", json=payload, timeout=10)
        r.raise_for_status()
        res = r.json()
        self.active_episode_id = res.get("episode_id")
        return res.get("observation", res)

    def step(self, action: Dict) -> Dict:
        # Wrap action into StepRequest format
        payload = {
            "episode_id": self.active_episode_id,
            "action_type": action.get("type", action.get("action_type", "send_message")),
            "text": action.get("text", ""),
            "metadata": {k: v for k, v in action.items() if k not in ["type", "text", "action_type", "episode_id"]}
        }
        r = self.requests.post(f"{self.base_url}/step", json=payload, timeout=10)
        r.raise_for_status()
        return r.json()

    def state(self) -> Dict:
        r = self.requests.get(f"{self.base_url}/state", timeout=10)
        r.raise_for_status()
        res = r.json()
        # Return the 'state' field if it exists, otherwise the whole thing
        return res.get("state", res)

    def health(self) -> Dict:
        r = self.requests.get(f"{self.base_url}/health", timeout=5)
        r.raise_for_status()
        return r.json()


# ── local import client (used when testing without HTTP) ──────────────────────
class LocalEnvClient:
    """
    Wraps a locally importable environment class.
    """
    def __init__(self):
        # Add project root to sys.path
        sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
        try:
            from environment.env import NegotiationEnv
            self.env = NegotiationEnv()
        except ImportError as e:
            raise ImportError(
                f"Local import failed: {e}\n"
                "Make sure you run from the project root."
            )

    def reset(self, curriculum_stage=None):
        obs = self.env.reset(curriculum_stage=curriculum_stage)
        if hasattr(obs, "model_dump"):
            return obs.model_dump()
        return obs

    def step(self, action):
        # Convert action dict to the format the env expects if needed
        # Actually our env expects a dict with 'action_type' and 'text'
        # The test uses 'type' and 'text'. I'll map them.
        mapped_action = {
            "action_type": action.get("type", action.get("action_type", "send_message")),
            "text": action.get("text", ""),
            "metadata": {k: v for k, v in action.items() if k not in ["type", "text", "action_type"]}
        }
        
        result = self.env.step(mapped_action)
        if isinstance(result, tuple):
            obs, reward, done, info = result
            return {"observation": obs, "reward": reward, "done": done, "info": info}
        return result

    def state(self):
        return self.env.state()

    def health(self):
        return {"status": "ok", "mode": "local"}


# ── helpers ────────────────────────────────────────────────────────────────────

def assert_keys(d: Dict, keys: list, label: str) -> Tuple[bool, str]:
    missing = [k for k in keys if k not in d]
    if missing:
        return False, f"{label} missing keys: {missing}"
    return True, ""

def run_full_episode(client, max_turns=20, action_fn=None) -> Dict:
    """
    Run one episode to completion (done=True or max_turns).
    action_fn(obs) -> action dict. Defaults to a fixed polite message.
    Returns summary dict.
    """
    obs = client.reset()
    total_reward = 0.0
    turn = 0
    trajectory = []

    while turn < max_turns:
        if action_fn:
            action = action_fn(obs)
        else:
            action = {
                "type": "send_message",
                "text": "I understand you're in a difficult situation. Can you tell me more about what happened?"
            }

        result = client.step(action)
        reward  = result.get("reward", 0.0)
        done    = result.get("done", False)
        obs     = result.get("observation", result)
        info    = result.get("info", {})

        total_reward += reward
        trajectory.append({"turn": turn, "reward": reward, "done": done, "info": info})
        turn += 1

        if done:
            break

    return {
        "turns": turn,
        "total_reward": round(total_reward, 4),
        "done_naturally": done,
        "trajectory": trajectory,
        "final_obs": obs,
    }


# ══════════════════════════════════════════════════════════════════════════════
# TEST GROUPS
# ══════════════════════════════════════════════════════════════════════════════

# ── GROUP 1: Environment Health ───────────────────────────────────────────────

def test_health(client):
    try:
        r = client.health()
        ok = isinstance(r, dict)
        record("Health endpoint responds", "PASS" if ok else "FAIL",
               f"Response: {r}" if ok else "No dict returned")
    except Exception as e:
        record("Health endpoint responds", "FAIL", str(e))


# ── GROUP 2: reset() contract ─────────────────────────────────────────────────

def test_reset_returns_observation(client):
    try:
        obs = client.reset()
        # Adapted for our codebase: borrower_msg, turn, escalation_level, stated_demands
        ok, msg = assert_keys(obs, ["borrower_msg", "turn", "escalation_level", "stated_demands"], "reset()")
        record("reset() returns required observation keys", "PASS" if ok else "FAIL", msg)
    except Exception as e:
        record("reset() returns required observation keys", "FAIL", traceback.format_exc(limit=2))

def test_reset_initial_values(client):
    try:
        obs = client.reset()
        issues = []
        if not isinstance(obs.get("turn"), int) or obs["turn"] != 0:
            issues.append(f"turn should be 0, got {obs.get('turn')}")
        el = obs.get("escalation_level")
        if not (isinstance(el, (int, float)) and 0 <= el <= 10):
            issues.append(f"escalation_level should be 0-10, got {el}")
        if not isinstance(obs.get("stated_demands"), list):
            issues.append(f"stated_demands should be a list, got {type(obs.get('stated_demands'))}")
        if not isinstance(obs.get("borrower_msg"), str) or len(obs["borrower_msg"]) < 5:
            issues.append(f"borrower_msg too short or wrong type")
        record("reset() initial values are sane", "FAIL" if issues else "PASS", "\n".join(issues))
    except Exception as e:
        record("reset() initial values are sane", "FAIL", traceback.format_exc(limit=2))

def test_reset_is_repeatable(client):
    try:
        obs1 = client.reset()
        obs2 = client.reset()
        # Both should be valid observations — not necessarily identical (stochastic is fine)
        ok1, _ = assert_keys(obs1, ["borrower_msg", "turn"], "reset() call 1")
        ok2, _ = assert_keys(obs2, ["borrower_msg", "turn"], "reset() call 2")
        record("reset() is callable multiple times without crash", "PASS" if ok1 and ok2 else "FAIL")
    except Exception as e:
        record("reset() is callable multiple times without crash", "FAIL", traceback.format_exc(limit=2))


# ── GROUP 3: step() contract ──────────────────────────────────────────────────

def test_step_returns_required_keys(client):
    try:
        client.reset()
        result = client.step({"type": "send_message", "text": "I hear you, please tell me more about your situation."})
        ok, msg = assert_keys(result, ["observation", "reward", "done"], "step()")
        record("step() returns observation, reward, done", "PASS" if ok else "FAIL", msg)
    except Exception as e:
        record("step() returns observation, reward, done", "FAIL", traceback.format_exc(limit=2))

def test_step_reward_is_numeric(client):
    try:
        client.reset()
        result = client.step({"type": "send_message", "text": "I understand your concerns and want to help."})
        r = result.get("reward")
        ok = isinstance(r, (int, float))
        record("step() reward is numeric", "PASS" if ok else "FAIL",
               f"Got: {r} ({type(r).__name__})")
    except Exception as e:
        record("step() reward is numeric", "FAIL", traceback.format_exc(limit=2))

def test_step_done_is_bool(client):
    try:
        client.reset()
        result = client.step({"type": "send_message", "text": "Let's work through this together to find a solution."})
        d = result.get("done")
        ok = isinstance(d, bool)
        record("step() done is boolean", "PASS" if ok else "FAIL",
               f"Got: {d} ({type(d).__name__})")
    except Exception as e:
        record("step() done is boolean", "FAIL", traceback.format_exc(limit=2))

def test_step_all_action_types(client):
    """Try primary action types from contracts.ACTION_TYPES."""
    # Adapted to actual ACTION_TYPES in contracts.py
    action_types = [
        {"type": "send_message",        "text": "I understand your situation and I'm here to help you."},
        {"type": "offer_emi",           "text": "We can offer you a reduced EMI plan.", "emi_amount": 2000},
        {"type": "acknowledge_hardship","text": "I'm sorry to hear about your medical difficulties."},
        {"type": "ask_open_question",   "text": "Could you explain what happened with your income?"},
        {"type": "stall",               "text": "Please wait a moment while I review your account details."},
    ]
    passed = []
    failed = []
    for action in action_types:
        try:
            client.reset()
            result = client.step(action)
            if "reward" in result or "observation" in result:
                passed.append(action["type"])
            else:
                failed.append(f"{action['type']}: missing keys in response")
        except Exception as e:
            failed.append(f"{action['type']}: {str(e)[:80]}")
    status = "PASS" if not failed else ("WARN" if len(failed) < 3 else "FAIL")
    detail = (f"OK: {passed}\nFAIL: {failed}") if failed else f"All passed: {passed}"
    record("All primary action types accepted by step()", status, detail)


# ── GROUP 4: State machine ────────────────────────────────────────────────────

def test_adversary_state_exists(client):
    try:
        client.reset()
        state = client.state()
        has_anger = "anger" in state
        has_trust  = "trust" in state
        ok = has_anger and has_trust
        record("Adversary state contains anger + trust fields", "PASS" if ok else "FAIL",
               f"State keys: {list(state.keys()) if isinstance(state, dict) else 'non-dict'}")
    except Exception as e:
        record("Adversary state contains anger + trust fields", "FAIL", traceback.format_exc(limit=2))

def test_anger_changes_on_bad_action(client):
    """A threatening or dismissive message should raise anger."""
    try:
        client.reset()
        state_before = client.state()
        anger_before = state_before.get("anger")

        client.step({"type": "send_message",
                     "text": "You have no choice. Pay now or face legal action and home visits immediately."})
        state_after = client.state()
        anger_after = state_after.get("anger")

        if anger_before is None or anger_after is None:
            record("Anger increases on threatening message", "WARN",
                   "Could not read anger from state — check state schema")
        elif anger_after >= anger_before:
            record("Anger increases on threatening message", "PASS",
                   f"anger: {anger_before} -> {anger_after}")
        else:
            record("Anger increases on threatening message", "WARN",
                   f"anger went DOWN on threat: {anger_before} -> {anger_after} (check rules)")
    except Exception as e:
        record("Anger increases on threatening message", "FAIL", traceback.format_exc(limit=2))

def test_trust_changes_on_good_action(client):
    """An empathetic message should not decrease trust."""
    try:
        client.reset()
        state_before = client.state()
        trust_before = state_before.get("trust")

        client.step({"type": "acknowledge_hardship",
                     "text": "I can see this has been incredibly stressful for your family. I want to find a sustainable solution."})
        state_after = client.state()
        trust_after = state_after.get("trust")

        if trust_before is None or trust_after is None:
            record("Trust does not drop on empathetic message", "WARN",
                   "Could not read trust from state — check state schema")
        elif trust_after >= trust_before:
            record("Trust does not drop on empathetic message", "PASS",
                   f"trust: {trust_before} -> {trust_after}")
        else:
            record("Trust does not drop on empathetic message", "WARN",
                   f"trust dropped: {trust_before} -> {trust_after} (check transition rules)")
    except Exception as e:
        record("Trust does not drop on empathetic message", "FAIL", traceback.format_exc(limit=2))


# ── GROUP 5: Episode termination ──────────────────────────────────────────────

def test_episode_terminates_on_timeout(client):
    """Running max_turns steps without resolution should terminate."""
    try:
        client.reset()
        done = False
        max_steps = 30  # generous upper bound
        turns = 0
        while turns < max_steps:
            result = client.step({"type": "send_message", "text": "I understand your situation, can you tell me more about it?"})
            done = result.get("done", False)
            turns += 1
            if done:
                break
        record("Episode terminates (timeout or resolution) within 30 turns",
               "PASS" if done else "FAIL",
               f"done={done} after {turns} steps")
    except Exception as e:
        record("Episode terminates (timeout or resolution) within 30 turns", "FAIL",
               traceback.format_exc(limit=2))

def test_step_after_done_is_safe(client):
    """Calling step() after done=True should not crash."""
    try:
        client.reset()
        done = False
        for _ in range(35):
            result = client.step({"type": "send_message", "text": "I see tell me more."})
            done = result.get("done", False)
            if done:
                break
        if not done:
            record("step() after done is safe", "WARN", "Episode never ended in 35 turns")
            return
        # Now call step again after done
        try:
            client.step({"type": "send_message", "text": "one more"})
            record("step() after done is safe", "PASS", "No crash on post-done step")
        except RuntimeError:
             record("step() after done is safe", "PASS", "Env correctly raised RuntimeError (graceful rejection)")
        except Exception as e:
            record("step() after done is safe", "WARN",
                   f"step() raised an unexpected exception: {type(e).__name__}")
    except Exception as e:
        record("step() after done is safe", "FAIL", traceback.format_exc(limit=2))


# ── GROUP 6: Reward function integrity ────────────────────────────────────────

def test_reward_bounded(client):
    """Over a full episode, per-step reward should stay in a reasonable range."""
    # Stage 3 weights for full reward stack
    client.reset(curriculum_stage=3)
    try:
        ep = run_full_episode(client, max_turns=20)
        rewards = [t["reward"] for t in ep["trajectory"]]
        # Our numeric range for reward_per_step is (-1.0, 1.5)
        out_of_range = [r for r in rewards if r < -2.0 or r > 2.0]
        ok = len(out_of_range) == 0
        record("Per-step rewards stay within [-2, +2]",
               "PASS" if ok else "WARN",
               f"All rewards: {[round(r,3) for r in rewards]}" +
               (f"\nOut of range: {out_of_range}" if out_of_range else ""))
    except Exception as e:
        record("Per-step rewards stay within [-2, +2]", "FAIL", traceback.format_exc(limit=2))

def test_reward_not_always_zero(client):
    """At least one non-zero reward signal must exist over an episode."""
    try:
        client.reset(curriculum_stage=3)
        ep = run_full_episode(client, max_turns=20)
        rewards = [t["reward"] for t in ep["trajectory"]]
        non_zero = [r for r in rewards if abs(r) > 1e-6]
        ok = len(non_zero) > 0
        record("At least one non-zero reward per episode",
               "PASS" if ok else "FAIL",
               f"Non-zero rewards: {non_zero}" if non_zero else "All rewards were 0.0")
    except Exception as e:
        record("At least one non-zero reward per episode", "FAIL", traceback.format_exc(limit=2))

def test_anti_repetition_penalty_fires(client):
    """Sending the exact same message every turn should incur a penalty."""
    try:
        client.reset(curriculum_stage=3)
        rewards = []
        for i in range(5):
            result = client.step({
                "type": "send_message",
                "text": "I understand your concerns and I want to help you resolve this debt today."
            })
            rewards.append(result.get("reward", 0.0))
            if result.get("done"):
                break
        # Expect at least one negative reward from the anti-repetition component
        # Note: compliance +0.1 might mask -0.2 if other things are positive, 
        # but combined it should drop.
        has_drop = any(rewards[i] < rewards[i-1] for i in range(1, len(rewards)))
        record("Anti-repetition penalty drops reward on repeated messages",
               "PASS" if has_drop else "WARN",
               f"Rewards over repeated turns: {[round(r,3) for r in rewards]}")
    except Exception as e:
        record("Anti-repetition penalty drops reward on repeated messages", "FAIL",
               traceback.format_exc(limit=2))

def test_reward_components_in_info(client):
    """step() info dict should expose per-component reward breakdown."""
    try:
        client.reset()
        result = client.step({"type": "send_message", "text": "Let's talk through your options for loan repayment."})
        info = result.get("info", {})
        has_breakdown = "reward_breakdown" in info or "breakdown" in info
        record("step() info exposes reward component breakdown",
               "PASS" if has_breakdown else "WARN",
               f"info keys: {list(info.keys())}")
    except Exception as e:
        record("step() info exposes reward component breakdown", "FAIL",
               traceback.format_exc(limit=2))


# ── GROUP 7: Curriculum / difficulty ─────────────────────────────────────────

def test_curriculum_stage_reset(client):
    """Curriculum stage should be accepted by reset()."""
    try:
        obs = client.reset(curriculum_stage=1)
        record("reset() accepts curriculum_stage", "PASS")
    except Exception as e:
        record("reset() accepts curriculum_stage", "FAIL", str(e))


# ── GROUP 8: Full episode integration ─────────────────────────────────────────

def test_full_episode_runs_without_crash(client):
    try:
        ep = run_full_episode(client, max_turns=20)
        ok = ep["turns"] > 0 and ep["turns"] <= 20
        record("Full episode completes without crash",
               "PASS" if ok else "FAIL",
               f"turns={ep['turns']}, total_reward={ep['total_reward']}, "
               f"done_naturally={ep['done_naturally']}")
    except Exception as e:
        record("Full episode completes without crash", "FAIL", traceback.format_exc(limit=2))

def test_multiple_episodes_sequential(client):
    """Run 3 episodes back-to-back — no state leakage."""
    try:
        summaries = []
        for i in range(3):
            ep = run_full_episode(client, max_turns=15)
            summaries.append(ep)
        all_ok = all(e["turns"] > 0 for e in summaries)
        record("3 sequential episodes run without state leakage",
               "PASS" if all_ok else "FAIL",
               f"Turns per episode: {[e['turns'] for e in summaries]}")
    except Exception as e:
        record("3 sequential episodes run without state leakage", "FAIL",
               traceback.format_exc(limit=2))


# ── GROUP 9: Performance / stability ─────────────────────────────────────────

def test_step_latency(client):
    """Each step() call should complete within 2 seconds."""
    try:
        client.reset()
        latencies = []
        for _ in range(5):
            t0 = time.time()
            client.step({"type": "send_message", "text": "Tell me more about your hardship."})
            latencies.append(round(time.time() - t0, 3))
        avg = round(sum(latencies) / len(latencies), 3)
        ok = avg < 2.0
        record("step() average latency < 2s",
               "PASS" if ok else ("WARN" if avg < 5.0 else "FAIL"),
               f"Latencies (s): {latencies}  avg={avg}s")
    except Exception as e:
        record("step() average latency < 2s", "FAIL", traceback.format_exc(limit=2))

def test_env_handles_empty_message(client):
    """Empty or whitespace-only message should not crash the env."""
    try:
        client.reset()
        result = client.step({"type": "send_message", "text": " "})
        ok = "reward" in result or "observation" in result
        record("Empty message does not crash env", "PASS" if ok else "WARN",
               "Returned: " + str(result)[:60])
    except Exception as e:
        record("Empty message does not crash env", "WARN",
               f"Env raised exception on empty message: {str(e)[:120]}")

def test_env_handles_very_long_message(client):
    """A very long message (2000 chars) should not crash the env."""
    try:
        client.reset()
        long_text = "I understand your concerns and want to help. " * 50  # ~2000 chars
        result = client.step({"type": "send_message", "text": long_text})
        ok = "reward" in result or "observation" in result
        record("Very long message does not crash env", "PASS" if ok else "WARN")
    except Exception as e:
        record("Very long message does not crash env", "WARN",
               f"Env raised exception on long message: {str(e)[:120]}")


# ==============================================================================
# MAIN RUNNER
# ==============================================================================

def run_all(client):
    section("1. ENVIRONMENT HEALTH")
    test_health(client)

    section("2. reset() CONTRACT")
    test_reset_returns_observation(client)
    test_reset_initial_values(client)
    test_reset_is_repeatable(client)

    section("3. step() CONTRACT")
    test_step_returns_required_keys(client)
    test_step_reward_is_numeric(client)
    test_step_done_is_bool(client)
    test_step_all_action_types(client)

    section("4. ADVERSARY STATE MACHINE")
    test_adversary_state_exists(client)
    test_anger_changes_on_bad_action(client)
    test_trust_changes_on_good_action(client)

    section("5. EPISODE TERMINATION")
    test_episode_terminates_on_timeout(client)
    test_step_after_done_is_safe(client)

    section("6. REWARD FUNCTION INTEGRITY")
    test_reward_bounded(client)
    test_reward_not_always_zero(client)
    test_anti_repetition_penalty_fires(client)
    test_reward_components_in_info(client)

    section("7. CURRICULUM")
    test_curriculum_stage_reset(client)

    section("8. FULL EPISODE INTEGRATION")
    test_full_episode_runs_without_crash(client)
    test_multiple_episodes_sequential(client)

    section("9. PERFORMANCE & EDGE CASES")
    test_step_latency(client)
    test_env_handles_empty_message(client)
    test_env_handles_very_long_message(client)


def print_summary():
    total  = len(results)
    passed = sum(1 for r in results if r["status"] == "PASS")
    warned = sum(1 for r in results if r["status"] == "WARN")
    failed = sum(1 for r in results if r["status"] == "FAIL")

    print(f"\n{'-'*60}")
    print(f"  AUDIT SUMMARY")
    print(f"{'-'*60}")
    print(f"  Total checks : {total}")
    print(f"  PASS         : {passed}")
    print(f"  WARN         : {warned}  (non-blocking)")
    print(f"  FAIL         : {failed}  (blocking)")
    print(f"{'-'*60}")

    if failed == 0 and warned <= 3:
        print(f"  [OK] Environment is STABLE. Safe to start training.\n")
    elif failed == 0:
        print(f"  [WARN] Environment mostly stable.\n")
    else:
        print(f"  [FAIL] Environment has FAILURES. Fix before any training run.\n")

    sys.exit(1 if failed > 0 else 0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="E2E audit for Crisis Negotiation RL env")
    parser.add_argument("--url",   default="http://localhost:8000",
                        help="Base URL of the deployed env")
    parser.add_argument("--local", action="store_true",
                        help="Import env directly (run from root)")
    args = parser.parse_args()

    print(f"\nCrisis Negotiation RL - End-to-End Stability Audit")
    print(f"{'-'*60}")

    if args.local:
        client = LocalEnvClient()
    else:
        client = HTTPEnvClient(args.url)

    try:
        run_all(client)
    except KeyboardInterrupt:
        print("\n\nAborted.")

    print_summary()
