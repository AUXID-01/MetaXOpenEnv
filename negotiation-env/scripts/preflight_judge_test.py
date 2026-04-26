import os
import sys
import json
# Ensure we can import from the project root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from environment.models.state import State
from environment.models.action import Action
from client.utils import action_from_text
from reward import compose
def create_dummy_states():
    """Helper to create basic State objects for reward calculation.

    All seven required-but-not-shown-in-spec fields (real_emi_capacity,
    stated_capacity, loan_amount, anger_threshold, profile_id,
    curriculum_stage, episode_id) get sensible defaults that exercise the
    deterministic rewards without triggering termination or saturation.
    """
    state_kwargs = dict(
        turn=1, max_turns=15,
        anger=5.0, trust=5.0, fear=5.0,
        real_emi_capacity=2000.0, stated_capacity=1500.0, loan_amount=50000.0,
        anger_threshold=8.0,
        terminated=False, termination_reason=None,
        demands_total=["settlement"], demands_hidden=["settlement"], demands_stated=[],
        demands_addressed=[], message_history=["I hear you and want to help."],
        profile_id="P01", curriculum_stage=1, episode_id="EP_PREFLIGHT",
    )
    prev_state = State(**state_kwargs)
    curr_state = State(**state_kwargs)
    return prev_state, curr_state
def run_edge_case(name: str, raw_llm_output: str, expect_short_circuit: bool):
    print(f"\n{'='*60}")
    print(f"TEST CASE: {name}")
    print(f"{'='*60}")

    # 1. Parse the output (Simulating client.utils)
    action_dict = action_from_text(raw_llm_output)

    # Mirror exactly what env.step() does when the parser returns empty text
    # on a parse failure (see environment/env.py:131-132). The Action model
    # enforces non-empty `text`; the env substitutes "..." so the deterministic
    # reward functions still get a valid Action object to look at, while
    # format_compliance still penalises the original raw_text under the hood.
    if not action_dict.get("text", "").strip():
        action_dict["text"] = "..."

    action = Action(**action_dict)

    # Inject conversational history into metadata so the Judge has context (as per Phase 3 spec)
    action.metadata["borrower_msg"] = "I can't pay right now."
    action.metadata["conversation_history"] = []

    # 2. Run the Fast-Hybrid Reward Combiner
    prev_state, curr_state = create_dummy_states()
    total_score, breakdown = compose(prev_state, curr_state, action)

    # 3. Print Results
    print(f"Raw Text Input:\n{raw_llm_output.strip()}\n")
    print(f"Parsed Action Type : {action.action_type}")
    print(f"Parsed Text        : '{action.text}'")

    print("\nREWARD BREAKDOWN:")
    print(f"  Format Compliance : {breakdown.get('raw_format_compliance')}")
    print(f"  RBI Compliance    : {breakdown.get('raw_compliance')}")
    print(f"  Anti-Exploit      : {breakdown.get('raw_anti_exploit')}")
    print(f"  Total Score       : {total_score:.4f}")

    print("\nLLM JUDGE DIAGNOSTICS:")
    is_short_circuited = breakdown.get('judge_short_circuited', False)
    print(f"  Short Circuited?  : {is_short_circuited} (Expected: {expect_short_circuit})")
    print(f"  Judge Used?       : {breakdown.get('judge_used')}")
    print(f"  Judge Reason      : {breakdown.get('judge_reason')}")
    print(f"  Judge Empathy     : {breakdown.get('judge_empathy')}")
    print(f"  Judge Strategy    : {breakdown.get('judge_strategy')}")

    # Assertions
    if expect_short_circuit:
        assert is_short_circuited, f"FAILED: Expected {name} to short-circuit, but it didn't!"
        assert breakdown.get('raw_deescalation', 0.0) == 0.0, "FAILED: Short-circuit should set deescalation to 0.0"
    else:
        assert not is_short_circuited, f"FAILED: Expected {name} to reach the judge, but it short-circuited!"
# =====================================================================
# EDGE CASE RUNS
# =====================================================================
if __name__ == "__main__":
    print("Starting Pre-Flight Fast-Hybrid Reward Checks...")
    # CASE 1: Pure Prose (The formatting leak risk)
    # Expectation: Format penalty, short-circuit = TRUE, no LLM judge called.
    run_edge_case(
        name="1. Pure Prose / No JSON",
        raw_llm_output="I think I should offer an EMI. How about 5000?",
        expect_short_circuit=True
    )
    # CASE 2: Broken JSON (Missing Text Key)
    # Expectation: Format penalty, short-circuit = TRUE, no LLM judge called.
    run_edge_case(
        name="2. Missing Text Key in JSON",
        raw_llm_output='{"thought_process": "Oops, forgot text", "action_type": "send_message"}',
        expect_short_circuit=True
    )
    # CASE 3: RBI Compliance Violation (Legal Threat)
    # Expectation: Format PASSES, RBI Compliance FAILS, short-circuit = TRUE.
    run_edge_case(
        name="3. Legal Threat (RBI Violation)",
        raw_llm_output="""```json
        {
            "thought_process": "Borrower is stubborn. Time to scare them.",
            "action_type": "send_message",
            "text": "If you do not pay, I will call the police immediately."
        }
        ```""",
        expect_short_circuit=True
    )
    # CASE 4: The Happy Path (Perfect JSON, Polite)
    # Expectation: Format PASSES, RBI PASSES, short-circuit = FALSE. Judge IS called.
    run_edge_case(
        name="4. The Happy Path",
        raw_llm_output="""
        {
            "thought_process": "The borrower is stressed. I will show empathy.",
            "action_type": "acknowledge_hardship",
            "text": "I understand that losing your job is incredibly stressful. Let's find a way to manage this together."
        }
        """,
        expect_short_circuit=False
    )

    print("\nAll assertions passed! The fast-hybrid reward system is working flawlessly.")
