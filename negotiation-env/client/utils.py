import re
import json
import logging
import sys
import os

# Add project root to sys.path to allow absolute imports from contracts
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from contracts import ACTION_TYPES

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

# Greedy JSON sniffer: grabs from the first '{' to the LAST '}' in the
# completion. The greedy `.*` (with re.DOTALL) is intentional — we want
# the outermost object, even when the model wraps it in markdown fences
# or emits chatty preamble like "Sure, here's my response:\n```json\n{...}\n```".
# json.loads itself is the real validator; this regex is just the extractor.
_JSON_SNIFFER = re.compile(r"\{.*\}", re.DOTALL)

_VALID_ACTIONS = set(ACTION_TYPES)


def obs_to_prompt(obs_dict: dict, system_prompt: str) -> str:
    """
    Converts observation dict into the full prompt string sent to the LLM.
    Will be populated in Phase 2 alongside prompt_builder.
    """
    pass


def action_from_text(generated_text: str) -> dict:
    """Parse an LLM completion into the action dict the env expects.

    Expected LLM output format (JSON — see ``prompt_builder.build_system_prompt``):

        {
          "thought_process": "Internal reasoning, never sent to borrower.",
          "action_type": "send_message",
          "text": "What the borrower actually reads.",
          "metadata": {}
        }

    The parser is deliberately lenient about *wrapping* (markdown fences,
    leading prose) but strict about *content*: it never lets the model's
    raw chain-of-thought leak into ``action_dict["text"]`` because that
    would be sent verbatim to the adversary and instantly trip the TF-IDF
    anti-exploit penalty (long internal monologues are extremely
    self-similar across turns).

    Returns
    -------
    dict
        ``{"action_type": str, "text": str, "metadata": dict}`` — always
        well-formed even on parse failure. ``metadata["raw_text"]`` always
        holds the original completion (so :func:`reward_format_compliance`
        can grade it). On a successful parse, ``metadata["thought_process"]``
        is also populated.

    Critical edge case
    ------------------
    On any parse failure the fallback is::

        {"action_type": "send_message", "text": "", "metadata": {"raw_text": ...}}

    We deliberately do NOT use ``generated_text`` as the fallback ``text``
    — that would leak the model's internal monologue to the borrower and
    contaminate the TF-IDF anti-exploit signal across turns.
    """
    fallback = {
        "action_type": "send_message",
        "text": "",
        "metadata": {"raw_text": generated_text},
    }

    if not isinstance(generated_text, str) or not generated_text.strip():
        return fallback

    match = _JSON_SNIFFER.search(generated_text)
    if not match:
        logger.warning("action_from_text: no JSON object found in completion.")
        return fallback

    candidate = match.group(0)

    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError as exc:
        logger.warning(f"action_from_text: JSON decode failed ({exc.msg}).")
        return fallback

    if not isinstance(parsed, dict):
        logger.warning(f"action_from_text: parsed JSON is {type(parsed).__name__}, not object.")
        return fallback

    raw_action_type = parsed.get("action_type")
    if isinstance(raw_action_type, str) and raw_action_type.strip() in _VALID_ACTIONS:
        action_type = raw_action_type.strip()
    else:
        if raw_action_type is not None:
            logger.warning(f"action_from_text: invalid action_type '{raw_action_type}'; falling back to send_message.")
        action_type = "send_message"

    raw_text = parsed.get("text")
    text = raw_text.strip() if isinstance(raw_text, str) else ""

    metadata: dict = {"raw_text": generated_text}

    parsed_meta = parsed.get("metadata")
    if isinstance(parsed_meta, dict):
        for k, v in parsed_meta.items():
            if k == "raw_text":
                continue  # never let the model overwrite our raw_text mirror
            metadata[k] = v

    thought = parsed.get("thought_process")
    if isinstance(thought, str) and thought.strip():
        metadata["thought_process"] = thought.strip()

    return {
        "action_type": action_type,
        "text": text,
        "metadata": metadata,
    }


# Back-compat alias used by older call sites (and the smoke fixtures in
# reward_bridge.py's ``__main__``). The function is JSON-first now, but
# the public name is preserved.
parse_action_xml = action_from_text
