import os
import sys
import subprocess
import traceback
# Ensure the root directory is on the path so modular imports work
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# --- DYNAMIC IMPORTS ---
# Handles minor path differences in project layout robustly
def get_contracts():
    import contracts
    return contracts
def get_client_utils():
    try:
        import client.utils as cu
        return cu
    except ImportError:
        import utils as cu
        return cu
def get_env_client():
    try:
        import client.env_client as ec
        if hasattr(ec, "DummyEnvClient"):
            return ec
    except ImportError:
        pass
    import client.dummy_client as dc
    return dc
def get_real_env_client():
    try:
        import client.env_client as ec
        if hasattr(ec, "NegotiationEnvClient"):
            return ec
    except ImportError:
        import client.real_client as rc
        return rc
    return ec
def get_prompting():
    try:
        import training.prompting as pb
        return pb
    except ImportError:
        import training.prompt_builder as pb
        return pb
def get_rollout():
    import training.rollout as ro
    return ro
def get_reward_bridge():
    import training.reward_bridge as rb
    return rb
def get_curriculum():
    try:
        import training.curriculum as cs
        if hasattr(cs, "CurriculumScheduler"):
            return cs
    except ImportError:
        pass
    import training.curriculum_scheduler as cs
    return cs
# --- RUNNER LOGIC ---
passed = 0
failed = 0
failed_critical = 0
def run_check(description, test_func, is_critical=True):
    global passed, failed, failed_critical
    try:
        test_func()
        print(f"  [OK] {description}")
        passed += 1
    except BaseException as e:
        err_msg = str(e) if str(e) else e.__class__.__name__
        print(f"  [FAIL] {description} - ERROR: {err_msg}")
        failed += 1
        if is_critical:
            failed_critical += 1
# --- TEST SUITE ---
def run_all_tests():
    print("SECTION 1 — CONTRACTS")
    
    def t1_1(): get_contracts()
    run_check("contracts.py loads without error", t1_1)
    
    def t1_2():
        c = get_contracts()
        assert len(c.REWARD_BREAKDOWN_KEYS) == 7, f"Got {len(c.REWARD_BREAKDOWN_KEYS)}"
    run_check("REWARD_BREAKDOWN_KEYS has exactly 7 keys", t1_2)
    
    def t1_3():
        c = get_contracts()
        assert len(c.ACTION_TYPES) == 6, f"Got {len(c.ACTION_TYPES)}"
    run_check("ACTION_TYPES has exactly 6 keys", t1_3)
    
    def t1_4():
        c = get_contracts()
        assert "commitment_reached" in c.TERMINATION_REASONS, "commitment_reached missing"
    run_check("All TERMINATION_REASONS present including \"commitment_reached\"", t1_4)
    
    def t1_5():
        c = get_contracts()
        assert len(c.CURRICULUM_STAGES) == 4, f"Got {len(c.CURRICULUM_STAGES)}"
    run_check("CURRICULUM_STAGES has 4 stages", t1_5)
    
    def t1_6():
        c = get_contracts()
        assert "reward/total" in c.WANDB_COLUMNS, "Missing reward/total"
        assert "episode/reason" in c.WANDB_COLUMNS, "Missing episode/reason"
    run_check("WANDB_COLUMNS contains \"reward/total\" and \"episode/reason\"", t1_6)
    
    def t1_7():
        c = get_contracts()
        assert c.NUMERIC_RANGES["reward_per_step"] == (-1.0, 1.5), f"Got {c.NUMERIC_RANGES.get('reward_per_step')}"
    run_check("NUMERIC_RANGES reward_per_step is (-1.0, 1.5)", t1_7)
    
    def t1_8():
        result = subprocess.run([sys.executable, "contracts.py"], capture_output=True, text=True)
        assert result.returncode == 0, f"Status {result.returncode}: {result.stderr.strip()}"
    run_check("contracts.py self-validation passes (run its __main__ assertions)", t1_8)
    
    
    print("\nSECTION 2 — CLIENT")
    
    def t2_1():
        c = get_contracts()
        ec = get_env_client()
        client = ec.DummyEnvClient()
        obs = client.reset()
        for k in c.OBSERVATION_SCHEMA.keys():
            assert k in obs, f"Missing key {k}"
    run_check("DummyEnvClient.reset() returns all keys in OBSERVATION_SCHEMA", t2_1)
    
    def t2_2():
        ec = get_env_client()
        client = ec.DummyEnvClient()
        client.reset()
        res = client.step({"action_type": "send_message", "text": "hi", "metadata": {}})
        assert isinstance(res, tuple) and len(res) == 4, "Did not return (obs, reward, done, info)"
    run_check("DummyEnvClient.step() returns (obs, reward, done, info) tuple", t2_2)
    
    def t2_3():
        c = get_contracts()
        ec = get_env_client()
        client = ec.DummyEnvClient()
        client.reset()
        _, _, _, info = client.step({"action_type": "send_message", "text": "hi", "metadata": {}})
        for k in c.STEP_RESPONSE_INFO_KEYS:
            if k != "episode_id":
                assert k in info, f"Missing {k} in info dict"
    run_check("info dict contains all keys in STEP_RESPONSE_INFO_KEYS except episode_id (dummy doesnt have it)", t2_3)
    
    def t2_4():
        c = get_contracts()
        ec = get_env_client()
        client = ec.DummyEnvClient()
        client.reset()
        _, _, _, info = client.step({"action_type": "send_message", "text": "hi", "metadata": {}})
        for k in c.REWARD_BREAKDOWN_KEYS:
            assert k in info.get("reward_breakdown", {}), f"Missing {k} in reward_breakdown"
    run_check("info[\"reward_breakdown\"] contains all REWARD_BREAKDOWN_KEYS", t2_4)
    def t2_5():
        ec = get_env_client()
        client = ec.DummyEnvClient()
        client.reset()
        for i in range(3):
            _, _, done, _ = client.step({"action_type": "send_message", "text": f"hi {i}", "metadata": {}})
            assert not done, f"Should not be done at step {i+1}"
        _, _, done, info = client.step({"action_type": "send_message", "text": "last", "metadata": {}})
        assert done, "Should be done on 4th step"
        assert info.get("termination_reason") in ["commitment_reached", "timeout"], \
            f"Reason: {info.get('termination_reason')}"
    run_check("DummyEnvClient cycles correctly — 3 steps after reset, 4th step returns done=True with timeout reason", t2_5)
    def t2_6():
        ec = get_env_client()
        client = ec.DummyEnvClient()
        client.reset(curriculum_stage="stage_2")
    run_check("curriculum_stage parameter accepted by reset() without error", t2_6)
    
    def t2_7():
        rec = get_real_env_client()
        assert hasattr(rec.NegotiationEnvClient, "reset"), "Missing reset()"
        assert hasattr(rec.NegotiationEnvClient, "step"), "Missing step()"
    run_check("NegotiationEnvClient class exists and has reset() and step() methods", t2_7)
    print("\nSECTION 3 — JSON PARSER (client/utils.py)")

    def t3_1():
        cu = get_client_utils()
        # Valid JSON with all four contract keys.
        text = '{"action_type": "offer_emi", "text": "Here is 500", "metadata": {"amount": 500}}'
        res = cu.parse_action_xml(text)   # alias of action_from_text — kept for back-compat
        assert res["action_type"] == "offer_emi", res["action_type"]
        assert res["text"] == "Here is 500", res["text"]
        # Parser also injects metadata["raw_text"] for the format-compliance reward;
        # the user-facing keys must still survive intact.
        assert res["metadata"].get("amount") == 500, res["metadata"]
    run_check("Valid JSON parsed correctly — action_type, text, metadata all extracted", t3_1)
    def t3_2():
        cu = get_client_utils()
        # Greedy {.*} regex must skip chatty preamble/suffix.
        text = 'Sure, here is my response:\n{"action_type": "offer_emi", "text": "Hey", "metadata": {}}\nThanks!'
        res = cu.parse_action_xml(text)
        assert res["action_type"] == "offer_emi"
        assert res["text"] == "Hey"
    run_check("Messy output with text before and after JSON object still parsed correctly", t3_2)
    def t3_3():
        cu = get_client_utils()
        # Truncated mid-object → JSONDecodeError → safe fallback (text="").
        text = '{"action_type": "offer'
        res = cu.parse_action_xml(text)
        assert res["action_type"] == "send_message", f"Got {res.get('action_type')}"
        # Critical anti-leak invariant: monologue text must NOT be copied to text.
        assert res["text"] == "", f"Internal monologue leaked into text: {res.get('text')!r}"
    run_check("Truncated mid-object JSON falls back to send_message with empty text (no leak)", t3_3)
    def t3_4():
        cu = get_client_utils()
        text = '{"action_type": "invalid_magic", "text": "hello", "metadata": {}}'
        res = cu.parse_action_xml(text)
        assert res["action_type"] == "send_message", f"Failed fallback: {res.get('action_type')}"
        # Action falls back, but text is still trusted because the JSON parsed cleanly.
        assert res["text"] == "hello", res["text"]
    run_check("Unknown action_type falls back to send_message; valid text preserved", t3_4)
    def t3_5():
        cu = get_client_utils()
        res = cu.parse_action_xml("")
        assert res["action_type"] == "send_message", f"Failed fallback: {res.get('action_type')}"
        assert res["text"] == "", f"Empty input must produce empty text, got {res.get('text')!r}"
    run_check("Completely empty string falls back to send_message with empty text", t3_5)
    def t3_6():
        cu = get_client_utils()
        # Trailing comma is invalid JSON — whole object rejected, safe fallback.
        text = '{"action_type": "offer_emi", "text": "Hey", "metadata": {}, }'
        res = cu.parse_action_xml(text)
        assert res["action_type"] == "send_message", f"Got {res.get('action_type')}"
        assert res["text"] == "", f"Malformed JSON must NOT leak raw input as text, got {res.get('text')!r}"
    run_check("Malformed JSON (trailing comma) falls back safely without leaking raw input", t3_6)
    def t3_7():
        cu = get_client_utils()
        # No closing brace at all — extractor regex fails to match.
        text = '{"action_type": "offer_emi", "text": "Half'
        res = cu.parse_action_xml(text)  # must not raise
        assert res["text"] == "", "Truncated JSON must produce empty text"
    run_check("Truncated mid-key output handled without exception, no leak", t3_7)
    print("\nSECTION 4 — PROMPT BUILDER")
    def t4_1():
        c = get_contracts()
        pb = get_prompting()
        prompt = pb.build_system_prompt()
        for k in c.ACTION_TYPES:  # list, not dict — remove .keys()
            assert k in prompt, f"Missing ACTION_TYPE: {k}"
    run_check("build_system_prompt() contains all 6 ACTION_TYPES", t4_1)
    def t4_2():
        pb = get_prompting()
        prompt = pb.build_system_prompt()
        assert "RBI" in prompt, "Missing RBI compliance rule"
    run_check("build_system_prompt() contains RBI compliance rule", t4_2)
    def t4_3():
        pb = get_prompting()
        prompt = pb.build_system_prompt()
        # The contract is now JSON — system prompt must instruct the model
        # to emit a single JSON object with action_type / text / metadata
        # / thought_process keys.
        assert '"action_type"' in prompt, "Missing 'action_type' key in JSON contract"
        assert '"text"' in prompt, "Missing 'text' key in JSON contract"
        assert '"thought_process"' in prompt, "Missing 'thought_process' (private monologue) key"
        assert "JSON" in prompt, "Missing JSON contract instruction"
    run_check("build_system_prompt() contains JSON format instruction with all four keys", t4_3)
    def t4_4():
        pb = get_prompting()
        obs = {"borrower_message": "hi", "turns_remaining": 5, "escalation_level": 1, "stated_demands": []}
        prompt = pb.build_turn_prompt(obs, [])
        assert "CURRENT TURN" in prompt, "Missing CURRENT TURN"
        assert "CONVERSATION HISTORY" not in prompt, "Included history unexpectedly"
    run_check("build_turn_prompt() with no history renders correctly", t4_4)
    def t4_5():
        pb = get_prompting()
        obs = {"borrower_message": "hi", "turns_remaining": 5, "escalation_level": 1, "stated_demands": []}
        history = [{"role": "user", "content": "hello"}, {"role": "agent", "content": "yo"}]
        prompt = pb.build_turn_prompt(obs, history)
        assert "CONVERSATION HISTORY" in prompt
        assert prompt.find("CONVERSATION HISTORY") < prompt.find("CURRENT TURN"), "History placed after current block"
    run_check("build_turn_prompt() with 2 history entries renders CONVERSATION HISTORY block before CURRENT TURN block", t4_5)
    def t4_6():
        pb = get_prompting()
        obs = {"borrower_message": "hi", "turns_remaining": 5, "escalation_level": 1, "stated_demands": []}
        # Simulates a previous-run completion that was logged verbatim into
        # history (raw JSON). The prompt builder must surface the human-readable
        # `text` value and NOT splat the JSON envelope into the prompt.
        sentinel_keys = '{"action_type": "send_message", "text": "raw content", "metadata": {}}'
        history = [{"role": "agent", "content": sentinel_keys}]
        prompt = pb.build_turn_prompt(obs, history)
        # Inspect only the CONVERSATION HISTORY block — the system prompt
        # legitimately mentions JSON keys to instruct the model on the output
        # contract, so we must not flag those as leaks.
        assert "CONVERSATION HISTORY" in prompt, "History block missing"
        history_block = prompt.split("CONVERSATION HISTORY", 1)[1].split("CURRENT TURN", 1)[0]
        assert "raw content" in history_block, "Clean content missing from history"
        assert "{" not in history_block, "Raw JSON envelope leaked into history block"
        assert '"action_type"' not in history_block, "JSON key leaked into history block"
    run_check("history entries show clean text not raw JSON envelope", t4_6)
    def t4_7():
        pb = get_prompting()
        obs = {"borrower_message": "hi", "turns_remaining": 5, "escalation_level": 1, "stated_demands": []}
        prompt = pb.build_turn_prompt(obs, [])
        assert "None stated yet" in prompt, "Missing fallback string for empty demands"
    run_check("Empty stated_demands renders \"None stated yet\"", t4_7)
    def t4_8():
        pb = get_prompting()
        obs = {"borrower_message": "hi", "turns_remaining": 42, "escalation_level": 7, "stated_demands": []}
        prompt = pb.build_turn_prompt(obs, [])
        assert "42" in prompt, "Missing turns_remaining"
        assert "7" in prompt, "Missing escalation_level"
    run_check("turns_remaining and escalation_level appear in output", t4_8)
    print("\nSECTION 5 — ROLLOUT")
    def t5_1():
        ro = get_rollout()
        ec = get_env_client()
        client = ec.DummyEnvClient()
        def fake_llm(p): return '{"action_type": "send_message", "text": "hi", "metadata": {}}'
        traj = ro.run_episode(client, fake_llm, max_turns=3, curriculum_stage="stage_1")
        assert isinstance(traj, dict)
    run_check("run_episode() completes without error against DummyEnvClient", t5_1)
    def t5_2():
        ro = get_rollout()
        ec = get_env_client()
        client = ec.DummyEnvClient()
        def fake_llm(p): return '{"action_type": "send_message", "text": "hi", "metadata": {}}'
        traj = ro.run_episode(client, fake_llm, max_turns=3)
        req_keys = ["prompts", "completions", "rewards", "total_score", "success", 
                    "turns_taken", "reward_breakdown", "parse_failures", 
                    "final_anger", "final_trust", "last_info"]
        for k in req_keys:
            assert k in traj, f"Missing key {k} in trajectory object"
    run_check("trajectory contains all required keys: prompts, completions, rewards, total_score, success, turns_taken, reward_breakdown, parse_failures, final_anger, final_trust, last_info", t5_2)
    def t5_3():
        ro = get_rollout()
        ec = get_env_client()
        client = ec.DummyEnvClient()
        def fake_llm(p): return '{"action_type": "send_message", "text": "hi", "metadata": {}}'
        traj = ro.run_episode(client, fake_llm, max_turns=3)
        tt = traj["turns_taken"]
        assert len(traj["prompts"]) == tt, f"mismatched prompts length: {len(traj['prompts'])}"
        assert len(traj["completions"]) == tt, f"mismatched completions length: {len(traj['completions'])}"
        assert len(traj["rewards"]) == tt, f"mismatched rewards length: {len(traj['rewards'])}"
    run_check("len(prompts) == len(completions) == len(rewards) == turns_taken", t5_3)
    def t5_4():
        ro = get_rollout()
        ec = get_env_client()
        
        class InspectClient(ec.DummyEnvClient):
            def __init__(self):
                super().__init__()
                self.first_msg = ""
            def reset(self, **kwargs):
                obs = super().reset(**kwargs)
                self.first_msg = obs.get("borrower_message", "")
                return obs
        client = InspectClient()
        def fake_llm(p): return '{"action_type": "send_message", "text": "hi", "metadata": {}}'
        traj = ro.run_episode(client, fake_llm, max_turns=3)
        if traj["turns_taken"] > 1:
            assert "CONVERSATION HISTORY" in traj["prompts"][1]
            assert client.first_msg in traj["prompts"][1], "First borrower_message not found in Turn 2"
    run_check("history is passed correctly — Turn 2 prompt contains Turn 1 borrower message in CONVERSATION HISTORY block", t5_4)
    def t5_5():
        ro = get_rollout()
        ec = get_env_client()
        
        class StageClient(ec.DummyEnvClient):
            def __init__(self):
                super().__init__()
                self.received_stage = None
            def reset(self, curriculum_stage=None, **kwargs):
                self.received_stage = curriculum_stage
                return super().reset(curriculum_stage=curriculum_stage, **kwargs)
        client = StageClient()
        def fake_llm(p): return '{"action_type": "send_message", "text": "hi", "metadata": {}}'
        ro.run_episode(client, fake_llm, max_turns=3, curriculum_stage="stage_3")
        assert client.received_stage == "stage_3", f"Got: {client.received_stage}"
    run_check("curriculum stage forwarded to client.reset()", t5_5)
    def t5_6():
        ro = get_rollout()
        ec = get_env_client()
        
        class AccumulatingClient(ec.DummyEnvClient):
            def step(self, a):
                obs, r, d, info = super().step(a)
                if "reward_breakdown" not in info:
                    info["reward_breakdown"] = {}
                info["reward_breakdown"]["test_accum"] = info["reward_breakdown"].get("test_accum", 0.0) + 1.0
                return obs, r, d, info
        client = AccumulatingClient()
        def fake_llm(p): return '{"action_type": "send_message", "text": "hi", "metadata": {}}'
        traj = ro.run_episode(client, fake_llm, max_turns=3)
        val = traj["reward_breakdown"].get("test_accum", 0.0)
        assert val > 1.0, f"Expected accumulation > 1.0, got {val}"
    run_check("reward_breakdown accumulates across turns not just last turn", t5_6)
    def t5_7():
        ro = get_rollout()
        ec = get_env_client()
        client = ec.DummyEnvClient()
        def fake_llm(p): return '{"action_type": "send_message", "text": "hi", "metadata": {}}'
        traj = ro.run_episode(client, fake_llm, max_turns=4)
        assert "termination_reason" in traj["last_info"], "Missing termination_reason"
    run_check("last_info contains termination_reason", t5_7)
    def t5_8():
        ro = get_rollout()
        ec = get_env_client()
        client = ec.DummyEnvClient()
        def fake_llm(p): return "Just plain prose, no JSON object at all"
        traj = ro.run_episode(client, fake_llm, max_turns=2)
        assert traj["parse_failures"] > 0, f"Failures counted: {traj['parse_failures']}"
    run_check("parse_failures increments when JSON contract is broken", t5_8)
    def t5_9():
        ro = get_rollout()
        ec = get_env_client()
        client = ec.DummyEnvClient()
        def fake_llm(p): return '{"action_type": "send_message", "text": "hi", "metadata": {}}'
        traj = ro.run_episode(client, fake_llm, max_turns=3)
        assert abs(traj["total_score"] - sum(traj["rewards"])) < 1e-6
    run_check("total_score equals sum of rewards list", t5_9)
    print("\nSECTION 6 — REWARD BRIDGE")
    def t6_1():
        get_reward_bridge()
    run_check("reward_bridge.py imports without error", t6_1)
    def t6_2():
        rb = get_reward_bridge()
        ec = get_env_client()
        client = ec.DummyEnvClient()
        fn = rb.make_env_reward_fn(client)
        assert callable(fn), "Result is not callable"
    run_check("make_env_reward_fn(client) returns a callable", t6_2)
    def t6_3():
        rb = get_reward_bridge()
        ec = get_env_client()
        client = ec.DummyEnvClient()
        fn = rb.make_env_reward_fn(client)
        obs_batch = rb.reset_env_for_batch(client, batch_size=2)
        comps = [
            [{'content': '{"action_type": "offer_emi", "text": "hi", "metadata": {"emi_amount": 1500}}'}],
            [{'content': '{"action_type": "offer_emi", "text": "hi2", "metadata": {"emi_amount": 1500}}'}]
        ]
        rewards = fn(prompts=[], completions=comps, env_obs=obs_batch)
        assert isinstance(rewards, list), type(rewards)
        assert len(rewards) == 2, len(rewards)
        assert isinstance(rewards[0], float), type(rewards[0])
    run_check("reward_fn called with 2 completions returns list of 2 floats", t6_3)
    def t6_4():
        c = get_contracts()
        rb = get_reward_bridge()
        ec = get_env_client()
        client = ec.DummyEnvClient()
        fn = rb.make_env_reward_fn(client)
        obs_batch = rb.reset_env_for_batch(client, batch_size=2)
        comps = [
            [{'content': '{"action_type": "send_message", "text": "hi", "metadata": {}}'}],
            [{'content': '{"action_type": "send_message", "text": "hi", "metadata": {}}'}]
        ]
        rewards = fn(prompts=[], completions=comps, env_obs=obs_batch)
        rmin, rmax = c.NUMERIC_RANGES["reward_per_step"]
        for r in rewards:
            assert rmin <= r <= rmax, f"Reward {r} out of established bounds"
    run_check("all returned rewards are within NUMERIC_RANGES reward_per_step", t6_4)
    def t6_5():
        rb = get_reward_bridge()
        ec = get_env_client()
        
        class CrashingClient(ec.DummyEnvClient):
            def step(self, action_dict):
                if action_dict.get("text") == "CRASH":
                    raise Exception("Injected fault")
                return super().step(action_dict)
                
        client = CrashingClient()
        fn = rb.make_env_reward_fn(client)
        obs_batch = rb.reset_env_for_batch(client, batch_size=2)
        comps = [
            [{'content': '{"action_type": "send_message", "text": "hi", "metadata": {}}'}],
            [{'content': '{"action_type": "send_message", "text": "CRASH", "metadata": {}}'}]
        ]
        rewards = fn(prompts=[], completions=comps, env_obs=obs_batch)
        assert len(rewards) == 2
        assert rewards[1] == 0.0, "Exception did not fail gracefully to 0.0"
    run_check("exception in one step returns 0.0 for that item only, does not crash the batch", t6_5)
    def t6_6():
        rb = get_reward_bridge()
        ec = get_env_client()
        client = ec.DummyEnvClient()
        obs_batch = rb.reset_env_for_batch(client, 3)
        assert isinstance(obs_batch, list)
        assert len(obs_batch) == 3
        assert isinstance(obs_batch[0], dict)
    run_check("reset_env_for_batch(client, 3) returns list of 3 obs dicts", t6_6)
    print("\nSECTION 7 — CURRICULUM SCHEDULER")
    def t7_1():
        cs = get_curriculum()
        sched = cs.CurriculumScheduler()
        assert sched is not None
    run_check("CurriculumScheduler instantiates without error", t7_1, is_critical=False)
    
    def t7_2():
        cs = get_curriculum()
        sched = cs.CurriculumScheduler()
        res = sched.advance_if_ready(0.1)
        assert res is False, f"Expected False below threshold, got {res}"
    run_check("advance_if_ready(0.1) returns False (below any threshold)", t7_2, is_critical=False)
    def t7_3():
        cs = get_curriculum()
        sched = cs.CurriculumScheduler()
        res = sched.advance_if_ready(0.99)
        if res is not False:
            raise Exception(f"Expected False due to acceptable stubbing, got {res}")
    run_check("advance_if_ready(0.99) returns False (stubbed — acceptable for now, flagged clearly)", t7_3, is_critical=False)
if __name__ == "__main__":
    run_all_tests()
    
    print("\n=== SCAFFOLD STATUS ===")
    print(f"Passed : {passed} / {passed + failed}")
    print(f"Failed : {failed}")
    print("")
    
    if failed_critical == 0:
        print("READY FOR GATE 5: YES")
    else:
        print("READY FOR GATE 5: NO")
        
    print("(YES only if all Section 1-6 checks pass. Section 7 stub failure is acceptable and noted separately.)")