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

def obs_to_prompt(obs_dict: dict, system_prompt: str) -> str:
    """
    Converts observation dict into the full prompt string sent to the LLM.
    Will be populated in Phase 2 alongside prompt_builder.
    """
    pass

def action_from_text(generated_text: str) -> dict:
    """
    Parses LLM output into action dict.
    
    Expected LLM output format (XML tags — NOT JSON):
    <action_type>offer_emi</action_type>
    <text>I understand your situation and want to help...</text>
    
    Fallback on parse failure:
    {"action_type": "send_message", "text": <raw text>, "metadata": {}}
    """
    action_dict = {
        "action_type": "send_message",
        "text": generated_text.strip(),
        "metadata": {"raw_text": generated_text} # Addresses: Problem 1
    }
    
    # 1. Parse Action Type
    action_match = re.search(r'<action_type>(.*?)</action_type>', generated_text, re.DOTALL | re.IGNORECASE)
    if action_match:
        parsed_type = action_match.group(1).strip()
        if parsed_type in ACTION_TYPES:
            action_dict["action_type"] = parsed_type
        else:
            logger.warning(f"Invalid action_type parsed: '{parsed_type}'. Falling back to 'send_message'.")
            action_dict["action_type"] = "send_message"
    else:
        logger.warning("Missing <action_type> tag. Falling back to 'send_message'.")
        action_dict["action_type"] = "send_message"

    # 2. Parse Text
    text_match = re.search(r'<text>(.*?)</text>', generated_text, re.DOTALL | re.IGNORECASE)
    if text_match:
        action_dict["text"] = text_match.group(1).strip()
        
    return action_dict
parse_action_xml = action_from_text