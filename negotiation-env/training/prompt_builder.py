import sys
import os

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from contracts import ACTION_TYPES

MAX_HISTORY_TURNS = 6

def build_system_prompt() -> str:
    """Fixed system instructions given to the LLM at the start of every episode."""
    return f"""You are a professional bank collections agent in India.
Your job is to negotiate a repayment plan with a defaulting borrower.
You must respond in EXACTLY this XML format every single turn. No exceptions.

<action_type>send_message</action_type>
<text>Your message to the borrower here.</text>

Allowed values for action_type:
- send_message       (default conversational turn)
- offer_emi          (propose a specific EMI amount)
- acknowledge_hardship  (explicit empathy signal)
- ask_open_question  (needs assessment)
- confirm_in_writing (promise written confirmation)
- stall              (buy time)

Rules:
1. ALWAYS include both XML tags. Never skip them.
2. NEVER put anything outside the XML tags.
3. NEVER use JSON or nested tags inside <text>.
4. Keep your message under 3 sentences.
5. Never threaten legal action, mention police, or contact family members.
6. Build trust before proposing any repayment plan.

Example of a correct response:
<action_type>acknowledge_hardship</action_type>
<text>I understand that losing your job has been incredibly stressful. 
I am not here to add pressure. Can you tell me what your situation 
looks like right now?</text>"""

def build_turn_prompt(obs_dict: dict, history: list[dict] = None) -> str:
    """Formats the current observation into the user turn of the prompt."""
    import re
    
    demands_list = '\n'.join([f"- {d}" for d in obs_dict.get('stated_demands', [])])
    if not demands_list:
        demands_list = "- None stated yet"
        
    history_block = ""
    if history:
        # Only include the last MAX_HISTORY_TURNS
        trimmed_history = history[-MAX_HISTORY_TURNS:]
        lines = []
        for entry in trimmed_history:
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

    sys_prompt = build_system_prompt()
    return f"""{sys_prompt}
{history_block}
=== CURRENT TURN ===
Turn: {obs_dict.get('turn', 0)} (Turns remaining: {obs_dict.get('turns_remaining', 0)})
Escalation Level: {obs_dict.get('escalation_level', 0.0)}/10.0 (Keep this below 8.0!)
Stated Demands So Far:
{demands_list}

Borrower's Message:
"{obs_dict.get('borrower_msg', '')}"

Please generate your next strategic response focusing on building trust and making progress."""