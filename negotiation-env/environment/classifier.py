# environment/classifier.py
import re
from typing import Dict, List, Optional, Any

"""
HOLDS: deterministic text signal classification logic.
RUNS: called inside adversary.react() on every LLM output.
CONNECTS TO: adversary.py (uses signals to update state), rewards/compliance.py.
"""

# ─── REGEX PATTERNS ──────────────────────────────────────────────────────────

# Empathy & Acknowledgement
RE_EMPATHY = re.compile(
    r"(understand|stressful|sorry|difficult|tough|empathize|samajh|pata hai|dekh sakta|clear)", 
    re.IGNORECASE
)
RE_ACKNOWLEDGEMENT = re.compile(
    r"(i see|got it|acknowledged|correct|exactly|agree|noted)", 
    re.IGNORECASE
)

# Questions
RE_QUESTION = re.compile(r"\?|can you|could you|please tell", re.IGNORECASE)
RE_OPEN_QUESTION = re.compile(
    r"(what|how|why|when|describe|tell me|explain|elaborate|kaise|kitna|kyon)", 
    re.IGNORECASE
)

# Financial / Payment Context
RE_AMOUNT = re.compile(r"(?:rs\.?|inr|rupees?|₹)?\s*(\d{3,}(?:,\d+)*)", re.IGNORECASE)
RE_EMI = re.compile(r"\bemi\b|monthly|installment", re.IGNORECASE)
RE_TENURE = re.compile(r"(\d+)\s*(month|year)", re.IGNORECASE)
RE_RESTRUCTURE = re.compile(r"restructure|re-arrange|flexible|new plan", re.IGNORECASE)
RE_MORATORIUM = re.compile(r"moratorium|pause|hold|freeze|break", re.IGNORECASE)

# Commitment Requests
RE_COMMITMENT = re.compile(
    r"(promise|commit|agree|confirm|final|guarantee|pukka|waada)", 
    re.IGNORECASE
)

# Threats & Compliance Violations (High Risk)
RE_LEGAL_THREAT = re.compile(r"(legal action|court|case|judge|arbitration|notice|police|fir|arrest|jail)", re.IGNORECASE)
RE_SOCIAL_THREAT = re.compile(r"(family|neighbours?|neighbors?|neighborhood|boss|relative|parents|office|colleagues)", re.IGNORECASE)
RE_VISIT_THREAT = re.compile(r"(visit|home|house|place|coming|ghar|aadmi bhej)", re.IGNORECASE)
RE_HARASSMENT = re.compile(r"(shame|useless|defaulter|fraud|cheater|chor|liar)", re.IGNORECASE)

# Coercive Urgency / Deadlines
RE_DEADLINE = re.compile(
    r"(immediately|right now|today|urgent|by evening|within \d+ hours|last chance|warning)", 
    re.IGNORECASE
)

# ─── HELPERS ───────────────────────────────────────────────────────────────

def normalize_text(text: str) -> str:
    """Lowercase and collapse whitespace while preserving major punctuation."""
    if not text:
        return ""
    text = text.lower()
    text = re.sub(r"\s+", " ", text).strip()
    return text

def detect_empathy(text: str) -> bool:
    return bool(RE_EMPATHY.search(text))

def detect_acknowledgement(text: str) -> bool:
    return bool(RE_ACKNOWLEDGEMENT.search(text))

def detect_open_question(text: str) -> bool:
    # Must have both a question marker and an open-ended keyword
    return bool(RE_QUESTION.search(text)) and bool(RE_OPEN_QUESTION.search(text))

def detect_payment_offer(text: str) -> dict:
    """Extracts numerical amounts and plan intentions."""
    res = {
        "amount": None,
        "emi": None,
        "tenure_months": None,
        "has_restructure_option": bool(RE_RESTRUCTURE.search(text)),
        "has_moratorium_option": bool(RE_MORATORIUM.search(text)),
    }
    
    amount_match = RE_AMOUNT.search(text)
    if amount_match:
        val = int(amount_match.group(1).replace(",", ""))
        if bool(RE_EMI.search(text)):
            res["emi"] = val
        else:
            res["amount"] = val
            
    tenure_match = RE_TENURE.search(text)
    if tenure_match:
        count = int(tenure_match.group(1))
        unit = tenure_match.group(2)
        res["tenure_months"] = count if "month" in unit else count * 12
        
    return res

def detect_commitment_request(text: str) -> bool:
    return bool(RE_COMMITMENT.search(text))

def detect_threats(text: str) -> list[str]:
    threats = []
    if RE_LEGAL_THREAT.search(text): threats.append("legal_action")
    if RE_SOCIAL_THREAT.search(text): threats.append("social_exposure")
    if RE_VISIT_THREAT.search(text): threats.append("physical_visit")
    if RE_HARASSMENT.search(text): threats.append("personal_attack")
    return threats

def detect_deadline_pressure(text: str) -> bool:
    return bool(RE_DEADLINE.search(text))

def detect_compliance_flags(text: str) -> list[str]:
    """RBI / Regulatory safety violations."""
    flags = []
    if RE_LEGAL_THREAT.search(text): flags.append("coercive_legal_threat")
    if RE_SOCIAL_THREAT.search(text): flags.append("third_party_disclosure")
    if RE_VISIT_THREAT.search(text): flags.append("home_visit_intimidation")
    if RE_HARASSMENT.search(text): flags.append("abusive_language")
    if RE_DEADLINE.search(text): flags.append("false_urgency")
    return flags

def infer_primary_action_type(signals: dict) -> str:
    """Precedence logic to determine the dominant intent."""
    meta = signals["meta"]
    
    if meta["has_threat"] or signals["compliance_flags"]:
        return "warn_noncompliance"
    
    offer = signals["extracted_offer"]
    if offer["amount"] or offer["emi"] or offer["has_restructure_option"]:
        return "offer_plan"
    
    if meta["has_commitment_request"]:
        return "seek_commitment"
    
    if meta["has_open_question"]:
        return "probe_capacity"
    
    if meta["has_empathy"] or meta["has_acknowledgement"]:
        return "empathize"
        
    # Check for multiple firing signals
    count = sum([meta["has_empathy"], meta["has_open_question"], meta["has_commitment_request"]])
    if count > 1:
        return "mixed"
        
    return "unknown"

# ─── PUBLIC API ────────────────────────────────────────────────────────────

def classify_action(text: str) -> Dict[str, Any]:
    """
    Main entry point for determining intent and signals from agent text.
    PURE function: deterministic, no side effects.
    """
    norm = normalize_text(text)
    
    # Run detectors
    empathy = detect_empathy(norm)
    ack = detect_acknowledgement(norm)
    open_q = detect_open_question(norm)
    offer = detect_payment_offer(norm)
    commitment = detect_commitment_request(norm)
    threats = detect_threats(norm)
    deadline = detect_deadline_pressure(norm)
    compliance = detect_compliance_flags(norm)
    
    # Partial delta hints (heuristics only)
    anger_delta = 0.0
    trust_delta = 0.0
    fear_delta = 0.0
    
    if empathy or ack:
        anger_delta -= 0.5
        trust_delta += 0.3
    if open_q:
        trust_delta += 0.2
    if threats:
        anger_delta += 2.0
        trust_delta -= 1.0
        fear_delta += 1.5
    if deadline:
        anger_delta += 0.5
        fear_delta += 0.8
    if offer["emi"] or offer["amount"]:
        trust_delta += 0.1
    
    # Assemble result
    signals = {
        "primary_action_type": "unknown",
        "tags": threats + (["offer"] if (offer["amount"] or offer["emi"]) else []),
        "anger_delta": round(anger_delta, 2),
        "trust_delta": round(trust_delta, 2),
        "fear_delta": round(fear_delta, 2),
        "compliance_flags": compliance,
        "risk_flags": threats,
        "extracted_offer": offer,
        "meta": {
            "normalized_text": norm,
            "has_question": bool(RE_QUESTION.search(norm)),
            "has_open_question": open_q,
            "has_empathy": empathy,
            "has_acknowledgement": ack,
            "has_commitment_request": commitment,
            "has_threat": len(threats) > 0,
            "has_deadline_pressure": deadline,
        }
    }
    
    signals["primary_action_type"] = infer_primary_action_type(signals)
    
    # Add tags based on action type for easier downstream filtering
    if signals["primary_action_type"] not in signals["tags"]:
        signals["tags"].append(signals["primary_action_type"])
        
    return signals
