# tests/test_adversary.py
import pytest
from environment.adversary import BorrowerAdversary

@pytest.fixture
def profile_cooperative():
    return {
        "id": "P01",
        "name": "Ramesh",
        "personality": "cooperative_and_responsible",
        "anger_init": 1.0,
        "trust_init": 2.0,
        "fear_init": 1.0,
        "anger_threshold": 8.0,
        "real_emi": 5000,
        "stated_capacity": 3000,
        "demands": ["low interest"],
        "hidden_demands": ["no office calls"],
        "opening_msg": "I am worried.",
        "reason": "job loss"
    }

@pytest.fixture
def profile_angry():
    return {
        "id": "P02",
        "name": "Suresh",
        "personality": "defensive_and_emotionally_raw",
        "anger_init": 5.0,
        "trust_init": 1.0,
        "fear_init": 2.0,
        "anger_threshold": 7.0,
        "real_emi": 2000,
        "stated_capacity": 1000,
        "demands": ["waiver"],
        "hidden_demands": ["police case"],
        "opening_msg": "Why are you calling me?",
        "reason": "hospital bill"
    }

def test_adversary_init(profile_cooperative):
    adv = BorrowerAdversary(profile_cooperative)
    assert adv.anger == 1.0
    assert adv.trust == 2.0
    assert adv.real_emi == 5000

def test_opening_turn(profile_cooperative):
    adv = BorrowerAdversary(profile_cooperative)
    assert adv.opening_turn() == "I am worried."

def test_react_empathy_cooperative(profile_cooperative):
    adv = BorrowerAdversary(profile_cooperative)
    # Mock symbols
    signals = {
        "primary_action_type": "empathize",
        "anger_delta": -0.5,
        "trust_delta": 0.3,
        "fear_delta": 0.0,
        "meta": {"has_empathy": True, "has_open_question": False},
        "extracted_offer": {"amount": None, "emi": None},
        "compliance_flags": [],
        "risk_flags": []
    }
    reply, done, reason = adv.react("I understand your situation.", signals=signals)
    assert adv.anger < 1.0
    assert adv.trust > 2.0
    assert done is False

def test_react_threat_angry_scaling(profile_angry):
    adv = BorrowerAdversary(profile_angry)
    signals = {
        "primary_action_type": "warn_noncompliance",
        "anger_delta": 2.0,
        "trust_delta": -1.0,
        "fear_delta": 1.0,
        "meta": {"has_empathy": False, "has_open_question": False},
        "extracted_offer": {"amount": None, "emi": None},
        "compliance_flags": ["coercive_legal_threat"],
        "risk_flags": ["legal_action"]
    }
    # anger_delta 2.0 * threat_anger_scale (2.0 for defensive) + 1.5 extra for compliance flag = 5.5 increase
    reply, done, reason = adv.react("Pay or go to jail.", signals=signals)
    assert adv.anger >= 7.0
    assert done is True
    assert reason == "anger_threshold_crossed"

def test_hidden_demand_reveal(profile_cooperative):
    adv = BorrowerAdversary(profile_cooperative)
    adv.trust = 4.9
    signals = {
        "primary_action_type": "empathize",
        "anger_delta": 0.0,
        "trust_delta": 0.5, 
        "fear_delta": 0.0,
        "meta": {"has_empathy": True, "has_open_question": False},
        "extracted_offer": {"amount": None, "emi": None},
        "compliance_flags": [],
        "risk_flags": []
    }
    assert "no office calls" not in adv.revealed_demands
    adv.react("I care about you.", signals=signals)
    assert adv.trust >= 5.0
    assert "no office calls" in adv.revealed_demands
    assert "no office calls" in adv.demands

def test_successful_commitment(profile_cooperative):
    adv = BorrowerAdversary(profile_cooperative)
    adv.trust = 6.9
    adv.anger = 2.0
    signals = {
        "primary_action_type": "offer_plan",
        "anger_delta": -0.5,
        "trust_delta": 0.5,
        "fear_delta": 0.0,
        "meta": {"has_empathy": False, "has_open_question": False},
        "extracted_offer": {"amount": None, "emi": 5000},
        "compliance_flags": [],
        "risk_flags": []
    }
    reply, done, reason = adv.react("Here is a plan.", signals=signals)
    assert adv.trust >= 7.0
    assert adv.anger <= 3.0
    assert done is True
    assert reason == "commitment_reached"
