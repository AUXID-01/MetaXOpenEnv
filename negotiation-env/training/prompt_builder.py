import sys
import os

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from contracts import ACTION_TYPES

MAX_HISTORY_TURNS = 6

def build_system_prompt() -> str:
    """Fixed system instructions given to the LLM at the start of every episode.

    The contract: the model must emit EXACTLY ONE valid JSON object per turn.
    Anything outside the JSON object will be discarded by the parser. The
    `thought_process` key is the model's private scratchpad — it is logged
    for analysis but is *never* sent to the borrower, so the model is free
    to reason about strategy, emotional state, and tactics without those
    words leaking into the dialogue and tripping the anti-exploit penalty.
    """
    return """You are a professional bank collections agent in India.
Your job is to negotiate a repayment plan with a defaulting borrower.

═══════════════════════════════════════════════════════════════════════════
OUTPUT CONTRACT — READ CAREFULLY
═══════════════════════════════════════════════════════════════════════════
On every turn you MUST return EXACTLY ONE valid JSON object and NOTHING
ELSE. The schema is:

{
  "thought_process": "Your private reasoning. Plan strategy, analyse the
                      borrower's emotional state, decide what action_type
                      is best, draft the message, and review it. The
                      borrower NEVER sees this field — it is internal.",
  "action_type": "send_message",
  "text": "The actual sentence(s) the borrower will read.",
  "metadata": {}
}

Always populate `thought_process` BEFORE you decide on `text` — use it to
think first, then write the message. Keep `text` to ≤3 sentences and make
it sound natural and human.

Allowed values for "action_type":
  - send_message         (default conversational turn)
  - offer_emi            (propose a specific EMI amount; put the figure
                          in metadata as {"emi_amount": <number>})
  - acknowledge_hardship (explicit empathy signal)
  - ask_open_question    (needs assessment)
  - confirm_in_writing   (promise written confirmation)
  - stall                (buy time)
  - escalate_authority   (bring in a senior — last resort)

Hard rules:
  1. Output ONE JSON object. No prose before or after, no second JSON.
  2. Do NOT wrap the JSON in markdown fences (```json ... ```). If you
     accidentally do, the parser will strip them, but plain JSON is
     preferred.
  3. Never put XML tags anywhere — they will be ignored.
  4. Never threaten legal action, mention police, contact family or
     neighbours, or use abusive language. RBI Fair Practices Code applies.
  5. Build trust before proposing any repayment plan.
  6. Keep `text` under 3 sentences.

Example of a correct response (emit JSON exactly like this — no extra
formatting):
{"thought_process": "Borrower sounds stressed about job loss; anger is high (~7) and trust is low. I should lead with empathy before any numbers, otherwise I'll spike anger further. Acknowledge the hardship explicitly and invite them to share more — this also lets me probe for hidden demands.", "action_type": "acknowledge_hardship", "text": "I'm really sorry you're going through this — losing a job is incredibly hard. I'm not here to add pressure. Could you tell me a bit about what your situation looks like right now?", "metadata": {}}"""

def _clean_history_content(content: str) -> str:
    """Surface the human-readable text from a raw history entry.

    History rows can land here in three shapes:

    1. Plain prose ("Hello, can we talk?")  →  returned as-is.
    2. A raw JSON envelope from an earlier turn
       (``{"action_type": "...", "text": "Hello", ...}``)  →  the value of
       the ``"text"`` key is extracted and returned, so the JSON envelope
       does not contaminate the next prompt and inflate the TF-IDF
       repetition signal.
    3. Legacy XML envelopes from older trajectory logs  →  XML tags are
       stripped (kept as a no-cost back-compat with archived data).
    """
    import json as _json
    import re as _re

    if not isinstance(content, str):
        return ""

    s = content.strip()

    # Try JSON first — if it parses and exposes a "text" key, use that.
    if s.startswith("{"):
        try:
            parsed = _json.loads(s)
            if isinstance(parsed, dict) and isinstance(parsed.get("text"), str):
                return parsed["text"].strip()
        except _json.JSONDecodeError:
            pass

    # Otherwise strip any XML envelopes that might still be floating around.
    return _re.sub(r"<[^>]+>", "", s).strip()


def build_turn_prompt(obs_dict: dict, history: list[dict] = None) -> str:
    """Formats the current observation into the user turn of the prompt."""
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
                # Native format {"agent": ..., "borrower": ...} — these are
                # already plain text strings produced by rollout.py.
                agent_text = entry["agent"]
                borrower_text = entry.get("borrower", "")
            elif "role" in entry:
                # Test / external format {"role": ..., "content": ...} —
                # may contain a raw JSON envelope from an earlier completion.
                role = entry["role"]
                content = _clean_history_content(entry.get("content", ""))
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