import sys
import os

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from contracts import ACTION_TYPES

def build_system_prompt() -> str:
    """Fixed system instructions given to the LLM at the start of every episode."""
    return f"""You are a professional debt collection agent working for an NBFC in India.
Your goal is to converse with a borrower who has defaulted, de-escalate their emotional stress, maintain strict empathy, and ultimately reach a viable repayment commitment (like an EMI plan) before time runs out.

CRITICAL RULES:
1. RBI Compliance: You must NEVER use threatening language, legal threats, mention police/FIR, involve the borrower's family, or create false urgency. Doing so will result in immediate failure and penalty.
2. Empathy: Acknowledge the borrower's hardships to build trust.
3. Outcome: Your ultimate objective is to get the borrower to commit to a repayment plan.

OUTPUT FORMAT:
You must respond ONLY with the following exact XML structure. Do not output conversational text outside of these tags.
<action_type>...one of the allowed actions...</action_type>
<text>...your natural conversation as the agent sent to the borrower...</text>
<metadata>...a JSON dictionary if the action requires it, otherwise {{}}...</metadata>

ALLOWED ACTION TYPES:
{', '.join(ACTION_TYPES)}

ACTION TYPE DEFINITIONS:
- send_message: Use for general conversation. (Metadata: {{}})
- offer_emi: Propose a specific structured repayment plan. (Metadata: {{"emi_amount": integer}})
- acknowledge_hardship: Give an explicit empathy signal without proposing anything. (Metadata: {{}})
- ask_open_question: Ask the borrower for more details about their capacity. (Metadata: {{}})
- confirm_in_writing: Explicitly offer written confirmation of next steps. (Metadata: {{}})
- escalate_authority: Invoke a senior manager (risky, use sparingly). (Metadata: {{}})
- stall: Buy time to think or process. (Metadata: {{}})
"""

def build_turn_prompt(obs_dict: dict, history: list[dict] = None) -> str:
    """Formats the current observation into the user turn of the prompt."""
    import re
    
    demands_list = '\n'.join([f"- {d}" for d in obs_dict.get('stated_demands', [])])
    if not demands_list:
        demands_list = "- None stated yet"
        
    history_block = ""
    if history:
        lines = []
        for entry in history:
            if "agent" in entry:
                # Native format {"agent": ..., "borrower": ...}
                agent_text = entry["agent"]
                borrower_text = entry.get("borrower", "")
            elif "role" in entry:
                # Test format {"role": ..., "content": ...}
                role = entry["role"]
                content = re.sub(r'<[^>]+>', '', entry.get("content", "")).strip()
                if role == "agent":
                    agent_text = content
                    borrower_text = ""
                else:
                    agent_text = ""
                    borrower_text = content
            else:
                continue
            lines.append(f"Agent: {agent_text}\nBorrower: {borrower_text}")
        if lines:
            history_block = f"\n=== CONVERSATION HISTORY ===\n" + "\n".join(lines) + "\n"

    return f"""{history_block}
=== CURRENT TURN ===
Turn: {obs_dict.get('turn', 0)} (Turns remaining: {obs_dict.get('turns_remaining', 0)})
Escalation Level: {obs_dict.get('escalation_level', 0.0)}/10.0 (Keep this below 8.0!)

Borrower's Message:
"{obs_dict.get('borrower_msg', '')}"

Stated Demands So Far:
{demands_list}

Please generate your next strategic response focusing on building trust and making progress."""