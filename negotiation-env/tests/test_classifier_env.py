# tests/test_classifier.py
import pytest
from environment.classifier import classify_action

def test_classifier_is_deterministic():
    text = "I really understand your problem and want to help."
    result1 = classify_action(text)
    result2 = classify_action(text)
    assert result1 == result2

def test_empathy_detection():
    # English
    res = classify_action("I am sorry to hear about your job loss and I understand the stress.")
    assert res["meta"]["has_empathy"] is True
    assert res["anger_delta"] < 0
    assert res["primary_action_type"] == "empathize"
    
    # Hinglish
    res_h = classify_action("Main aapki situation samajh sakta hoon.")
    assert res_h["meta"]["has_empathy"] is True

def test_threat_detection():
    res = classify_action("Pay now or I will send the police to your house and call your neighbors.")
    assert res["meta"]["has_threat"] is True
    assert "legal_action" in res["risk_flags"]
    assert "social_exposure" in res["risk_flags"]
    assert res["anger_delta"] >= 2.0
    assert "coercive_legal_threat" in res["compliance_flags"]
    assert res["primary_action_type"] == "warn_noncompliance"

def test_financial_extraction():
    # EMI
    res = classify_action("Can you pay an EMI of 5000 Rupees?")
    assert res["extracted_offer"]["emi"] == 5000
    assert res["primary_action_type"] == "offer_plan"
    
    # Total amount
    res2 = classify_action("I need the full payment of 50000 INR immediately.")
    assert res2["extracted_offer"]["amount"] == 50000
    assert res2["meta"]["has_deadline_pressure"] is True

def test_open_question():
    res = classify_action("What is the main reason for your delay?")
    assert res["meta"]["has_open_question"] is True
    assert res["primary_action_type"] == "probe_capacity"

def test_empty_string():
    res = classify_action("")
    assert res["primary_action_type"] == "unknown"
    assert res["meta"]["normalized_text"] == ""
