# environment/scenarios/profiles.py
# Complete borrower scenario bank — 20 profiles across 4 curriculum stages
# Stage 1: anger_threshold >= 8.5  → easy, model can learn basic empathy
# Stage 2: anger_threshold 7.0–8.4 → medium, 2 demands, 1 hidden
# Stage 3: anger_threshold 5.5–6.9 → hard, multiple hidden, capacity mismatch
# Stage 4: anger_threshold <= 5.4  → realistic, full complexity

PROFILES = [

    # ─────────────────────────────────────────────
    # CURRICULUM STAGE 1 — calm, single demand, nothing hidden
    # Model just needs to show basic empathy + not threaten
    # ─────────────────────────────────────────────

    {
        "id": "P01",
        "curriculum_stage": 1,
        "name": "Ramesh Kumar",
        "age": 38,
        "gender": "male",
        "reason": "job_loss",
        "overdue_days": 60,
        "loan_type": "personal_loan",
        "loan_amount": 85000,
        "anger_init": 3.5,
        "trust_init": 2.0,
        "fear_init": 5.0,
        "anger_threshold": 9.0,       # easy to pass
        "real_emi": 7000,
        "stated_capacity": 0,
        "demands": ["no_penalties"],
        "hidden_demands": [],          # nothing hidden — stage 1
        "opening_msg": "I lost my job 2 months ago. I can't pay anything right now.",
        "backstory": "Was a sales manager, laid off in restructuring. Wife works part-time. Looking for work actively.",
        "personality": "resigned_but_cooperative",
    },

    {
        "id": "P02",
        "curriculum_stage": 1,
        "name": "Priya Nair",
        "age": 29,
        "gender": "female",
        "reason": "salary_delay",
        "overdue_days": 45,
        "loan_type": "personal_loan",
        "loan_amount": 50000,
        "anger_init": 2.8,
        "trust_init": 3.0,
        "fear_init": 4.5,
        "anger_threshold": 9.5,       # very easy
        "real_emi": 5000,
        "stated_capacity": 2000,
        "demands": ["extension_of_30_days"],
        "hidden_demands": [],
        "opening_msg": "My company hasn't paid salary for 2 months. I will pay as soon as I get it, I promise.",
        "backstory": "IT employee, employer delayed salaries due to cash flow. Fully intends to pay, just needs time.",
        "personality": "honest_and_anxious",
    },

    {
        "id": "P03",
        "curriculum_stage": 1,
        "name": "Suresh Patel",
        "age": 52,
        "gender": "male",
        "reason": "seasonal_income",
        "overdue_days": 75,
        "loan_type": "agricultural_loan",
        "loan_amount": 120000,
        "anger_init": 3.0,
        "trust_init": 2.5,
        "fear_init": 6.0,
        "anger_threshold": 8.8,
        "real_emi": 8000,
        "stated_capacity": 0,
        "demands": ["wait_until_harvest"],
        "hidden_demands": [],
        "opening_msg": "Bhai sahab, rabi season abhi baaki hai. Harvest ke baad pakka de dunga. Abhi haath khaali hai.",
        "backstory": "Small farmer in Maharashtra. Income is seasonal — gets lump sum post-harvest. Has always paid historically.",
        "personality": "traditional_respectful",
    },

    {
        "id": "P04",
        "curriculum_stage": 1,
        "name": "Anita Sharma",
        "age": 34,
        "gender": "female",
        "reason": "medical_expense_minor",
        "overdue_days": 30,
        "loan_type": "consumer_durable_loan",
        "loan_amount": 35000,
        "anger_init": 2.0,
        "trust_init": 4.0,
        "fear_init": 3.5,
        "anger_threshold": 9.8,       # extremely easy — she wants to resolve
        "real_emi": 4000,
        "stated_capacity": 3000,
        "demands": ["reduce_emi_for_2_months"],
        "hidden_demands": [],
        "opening_msg": "I had some medical expenses last month. Can we reduce the EMI for just 2 months? I'll pay the rest later.",
        "backstory": "School teacher. Stable income. Minor health issue caused short-term cash crunch. Very willing to resolve.",
        "personality": "cooperative_and_responsible",
    },

    {
        "id": "P05",
        "curriculum_stage": 1,
        "name": "Mohammed Irfan",
        "age": 41,
        "gender": "male",
        "reason": "business_slowdown",
        "overdue_days": 50,
        "loan_type": "business_loan",
        "loan_amount": 200000,
        "anger_init": 4.0,
        "trust_init": 1.5,
        "fear_init": 5.5,
        "anger_threshold": 8.5,
        "real_emi": 15000,
        "stated_capacity": 5000,
        "demands": ["restructure_loan"],
        "hidden_demands": [],
        "opening_msg": "My shop sales are down 60% since the new mall opened nearby. I'm not refusing to pay. I just need restructuring.",
        "backstory": "Small retail shop owner. Business disrupted by large mall. Has assets. Needs restructuring not waiver.",
        "personality": "pragmatic_and_direct",
    },

    # ─────────────────────────────────────────────
    # CURRICULUM STAGE 2 — mildly hostile, 2 demands, 1 hidden
    # Model must do needs assessment before offering
    # ─────────────────────────────────────────────

    {
        "id": "P06",
        "curriculum_stage": 2,
        "name": "Kavitha Devi",
        "age": 44,
        "gender": "female",
        "reason": "medical_emergency",
        "overdue_days": 120,
        "loan_type": "gold_loan",
        "loan_amount": 75000,
        "anger_init": 6.5,
        "trust_init": 1.0,
        "fear_init": 7.5,
        "anger_threshold": 7.5,       # much tighter
        "real_emi": 4000,
        "stated_capacity": 0,
        "demands": ["reduce_amount", "no_calls"],
        "hidden_demands": ["dignity"],  # revealed at trust > 5
        "opening_msg": "Why do you keep calling me? My husband is in hospital. Have you no shame?",
        "backstory": "Husband had a stroke 3 months ago. All savings depleted on hospital. Deeply ashamed of default. Feels harassed.",
        "personality": "defensive_and_emotionally_raw",
    },

    {
        "id": "P07",
        "curriculum_stage": 2,
        "name": "Rajiv Sinha",
        "age": 47,
        "gender": "male",
        "reason": "divorce_settlement",
        "overdue_days": 90,
        "loan_type": "home_loan",
        "loan_amount": 2500000,
        "anger_init": 5.5,
        "trust_init": 1.2,
        "fear_init": 6.0,
        "anger_threshold": 7.8,
        "real_emi": 20000,
        "stated_capacity": 0,
        "demands": ["freeze_interest", "time_to_sell_property"],
        "hidden_demands": ["privacy"],  # does not want family to know, revealed at trust > 4
        "opening_msg": "I'm going through a difficult personal situation. Please don't call my office number.",
        "backstory": "Senior professional going through costly divorce. Joint home now disputed. Will sell and settle. Embarrassed.",
        "personality": "guarded_and_proud",
    },

    {
        "id": "P08",
        "curriculum_stage": 2,
        "name": "Deepa Krishnamurthy",
        "age": 31,
        "gender": "female",
        "reason": "maternity_career_break",
        "overdue_days": 65,
        "loan_type": "education_loan",
        "loan_amount": 400000,
        "anger_init": 5.0,
        "trust_init": 1.8,
        "fear_init": 6.5,
        "anger_threshold": 8.0,
        "real_emi": 6000,
        "stated_capacity": 2000,
        "demands": ["moratorium_extension", "no_penalty"],
        "hidden_demands": ["written_confirmation"],  # revealed at trust > 5
        "opening_msg": "I took maternity leave and lost my job after. Nobody told me the moratorium would end automatically.",
        "backstory": "Engineer who took maternity leave. Employer did not rehire. Genuinely surprised by loan status. Has a job offer pending.",
        "personality": "frustrated_but_articulate",
    },

    {
        "id": "P09",
        "curriculum_stage": 2,
        "name": "Balasubramaniam T.",
        "age": 58,
        "gender": "male",
        "reason": "retirement_income_drop",
        "overdue_days": 100,
        "loan_type": "personal_loan",
        "loan_amount": 150000,
        "anger_init": 4.5,
        "trust_init": 2.5,
        "fear_init": 7.0,
        "anger_threshold": 7.5,
        "real_emi": 5000,
        "stated_capacity": 2000,
        "demands": ["reduce_emi", "no_legal_action"],
        "hidden_demands": ["family_not_to_know"],  # revealed at trust > 5
        "opening_msg": "I am a retired government servant. I have never defaulted in my life. Please understand my situation.",
        "backstory": "Pension income is insufficient for original EMI. Has fixed deposits but is reluctant to break them. Family doesn't know about loan.",
        "personality": "dignified_and_ashamed",
    },

    {
        "id": "P10",
        "curriculum_stage": 2,
        "name": "Fatima Shaikh",
        "age": 36,
        "gender": "female",
        "reason": "husband_business_loss",
        "overdue_days": 80,
        "loan_type": "microfinance_loan",
        "loan_amount": 40000,
        "anger_init": 5.8,
        "trust_init": 0.8,
        "fear_init": 8.0,
        "anger_threshold": 7.2,
        "real_emi": 3000,
        "stated_capacity": 500,
        "demands": ["smaller_installments", "no_home_visits"],
        "hidden_demands": ["safety"],   # fears husband's reaction if agents visit, revealed at trust > 4
        "opening_msg": "Please don't send anyone to my house. My husband doesn't know about this loan. Please.",
        "backstory": "Took microfinance loan for small tailoring business. Husband's construction work stopped. Terrified of home visits for personal safety reasons.",
        "personality": "frightened_and_secretive",
    },

    # ─────────────────────────────────────────────
    # CURRICULUM STAGE 3 — hostile, multiple hidden demands, capacity mismatch
    # Model must do real investigation — borrower claims less than actual capacity
    # ─────────────────────────────────────────────

    {
        "id": "P11",
        "curriculum_stage": 3,
        "name": "Arvind Mehta",
        "age": 49,
        "gender": "male",
        "reason": "business_failure",
        "overdue_days": 180,
        "loan_type": "business_loan",
        "loan_amount": 800000,
        "anger_init": 5.0,
        "trust_init": 0.5,
        "fear_init": 4.0,
        "anger_threshold": 6.0,       # very tight
        "real_emi": 12000,            # CAN pay, won't admit
        "stated_capacity": 0,         # claims completely broke
        "demands": ["settlement"],
        "hidden_demands": ["face_saving_exit", "no_legal_record"],  # both revealed at trust > 6
        "opening_msg": "My business is completely shut. I have nothing. Whatever you want to do, do it. I don't care anymore.",
        "backstory": "Business failed but has rental income from a property he hasn't disclosed. Wants one-time settlement. Worried about credit record affecting son's future.",
        "personality": "nihilistic_and_calculating",
    },

    {
        "id": "P12",
        "curriculum_stage": 3,
        "name": "Sunita Rao",
        "age": 42,
        "gender": "female",
        "reason": "husband_absconded",
        "overdue_days": 150,
        "loan_type": "joint_home_loan",
        "loan_amount": 1800000,
        "anger_init": 6.0,
        "trust_init": 0.3,
        "fear_init": 8.5,
        "anger_threshold": 6.5,
        "real_emi": 10000,
        "stated_capacity": 0,
        "demands": ["remove_husband_from_loan", "freeze_account"],
        "hidden_demands": ["legal_help_referral", "protection_from_husband"],  # revealed at trust > 5
        "opening_msg": "My husband took the money and left. Why am I being punished for what he did? I don't even know where he is.",
        "backstory": "Joint loan. Husband disappeared with loan proceeds. Sunita has a small salary job. Situation is partly a legal matter. Deeply traumatised.",
        "personality": "angry_and_victimised",
    },

    {
        "id": "P13",
        "curriculum_stage": 3,
        "name": "Kiran Reddy",
        "age": 33,
        "gender": "male",
        "reason": "crypto_investment_loss",
        "overdue_days": 110,
        "loan_type": "personal_loan",
        "loan_amount": 300000,
        "anger_init": 4.5,
        "trust_init": 1.0,
        "fear_init": 5.0,
        "anger_threshold": 6.2,
        "real_emi": 9000,            # has income, spent recklessly
        "stated_capacity": 1000,
        "demands": ["settlement_at_50_percent", "no_interest"],
        "hidden_demands": ["avoid_parents_finding_out", "credit_score_protection"],
        "opening_msg": "I made some bad investments. I'm not the first person this happened to. I want to settle this quietly.",
        "backstory": "Software engineer. Took loan and invested in crypto. Lost most of it. Still has salary but overstating loss. Embarrassed and wants quiet resolution.",
        "personality": "defensive_and_bargaining",
    },

    {
        "id": "P14",
        "curriculum_stage": 3,
        "name": "Meenakshi Gopal",
        "age": 55,
        "gender": "female",
        "reason": "widowed_income_loss",
        "overdue_days": 200,
        "loan_type": "personal_loan",
        "loan_amount": 180000,
        "anger_init": 3.5,
        "trust_init": 1.5,
        "fear_init": 9.0,            # very high fear
        "anger_threshold": 6.8,
        "real_emi": 4000,
        "stated_capacity": 0,
        "demands": ["complete_waiver"],
        "hidden_demands": ["installment_she_can_afford", "son_not_to_be_contacted"],
        "opening_msg": "My husband passed away last year. I am alone. I don't have any money. Please forgive this loan.",
        "backstory": "Widow. Loan was in husband's name with her as co-applicant. Actually has small pension. Asking for waiver but will accept affordable EMI if approached gently. Terrified son will be contacted.",
        "personality": "grief_stricken_and_fearful",
    },

    {
        "id": "P15",
        "curriculum_stage": 3,
        "name": "Vijay Thokar",
        "age": 38,
        "gender": "male",
        "reason": "gig_income_irregular",
        "overdue_days": 95,
        "loan_type": "two_wheeler_loan",
        "loan_amount": 65000,
        "anger_init": 5.5,
        "trust_init": 0.8,
        "fear_init": 5.5,
        "anger_threshold": 6.5,
        "real_emi": 4500,
        "stated_capacity": 1000,
        "demands": ["flexible_payment_schedule", "no_repossession"],
        "hidden_demands": ["vehicle_is_livelihood"],  # if vehicle is repossessed, he loses income entirely — revealed at trust > 5
        "opening_msg": "I'm a delivery guy. Some months are good, some are not. I can't pay a fixed amount every month. It's not possible.",
        "backstory": "Zomato/Swiggy delivery partner. Income fluctuates significantly. The bike is his only income source. Has been paying irregularly but not defaulting maliciously.",
        "personality": "street_smart_and_suspicious",
    },

    # ─────────────────────────────────────────────
    # CURRICULUM STAGE 4 — realistic, full complexity, adversarial or multi-party
    # Model must handle strategic behaviour, contradictions, high stakes
    # ─────────────────────────────────────────────

    {
        "id": "P16",
        "curriculum_stage": 4,
        "name": "Harish Chandra Gupta",
        "age": 61,
        "gender": "male",
        "reason": "guarantor_liability",
        "overdue_days": 240,
        "loan_type": "business_loan",
        "loan_amount": 1500000,
        "anger_init": 7.0,
        "trust_init": 0.2,
        "fear_init": 7.0,
        "anger_threshold": 5.5,      # very tight
        "real_emi": 0,               # genuinely cannot pay — guarantor, not borrower
        "stated_capacity": 0,
        "demands": ["prove_original_borrower_defaulted_first", "legal_notice_clarification"],
        "hidden_demands": ["scared_of_losing_home", "wants_lawyer_advice_first"],
        "opening_msg": "I am a guarantor. I did not take this loan. Why are you calling ME? This is illegal. I'm going to file a consumer complaint.",
        "backstory": "Retired teacher who signed as guarantor for nephew's loan. Nephew absconded. Gupta has a home but no liquid income. Legally complex. Genuinely outraged.",
        "personality": "legally_aware_and_outraged",
    },

    {
        "id": "P17",
        "curriculum_stage": 4,
        "name": "Rohini Malhotra",
        "age": 45,
        "gender": "female",
        "reason": "MSME_closure",
        "overdue_days": 300,
        "loan_type": "msme_loan",
        "loan_amount": 5000000,
        "anger_init": 4.0,
        "trust_init": 0.5,
        "fear_init": 6.0,
        "anger_threshold": 5.8,
        "real_emi": 30000,
        "stated_capacity": 0,
        "demands": ["OTS_one_time_settlement", "30_percent_haircut"],
        "hidden_demands": ["avoid_CIBIL_hit", "protect_directors_personal_assets", "confidential_resolution"],
        "opening_msg": "I have already spoken to your regional head. Please don't waste my time. I want to discuss One Time Settlement only.",
        "backstory": "Sophisticated business owner. Company closed during pandemic. Has personal assets. Knowledgeable about banking processes. Will negotiate hard but can pay OTS.",
        "personality": "experienced_negotiator_in_control",
    },

    {
        "id": "P18",
        "curriculum_stage": 4,
        "name": "Santosh Kumar Yadav",
        "age": 27,
        "gender": "male",
        "reason": "first_job_loss",
        "overdue_days": 55,
        "loan_type": "personal_loan",
        "loan_amount": 120000,
        "anger_init": 6.5,
        "trust_init": 0.5,
        "fear_init": 8.0,
        "anger_threshold": 5.5,
        "real_emi": 6000,
        "stated_capacity": 0,
        "demands": ["stop_calling_references", "delete_my_data"],
        "hidden_demands": ["terrified_of_parents_finding_out", "willing_to_pay_if_references_safe"],
        "opening_msg": "You called my college professor! How dare you! I will sue you for harassment. This is my private matter!",
        "backstory": "First job after college, laid off after 6 months. Agent already called his references which is a violation. Genuinely angry and scared. Can pay once employed — has interview lined up.",
        "personality": "young_and_explosive",
    },

    {
        "id": "P19",
        "curriculum_stage": 4,
        "name": "Lakshmi Venkataraman",
        "age": 50,
        "gender": "female",
        "reason": "property_dispute_frozen_assets",
        "overdue_days": 175,
        "loan_type": "lap_loan_against_property",
        "loan_amount": 3000000,
        "anger_init": 3.0,
        "trust_init": 3.0,
        "fear_init": 5.0,
        "anger_threshold": 5.0,      # tightest in the bank
        "real_emi": 25000,
        "stated_capacity": 0,
        "demands": ["wait_for_court_order", "no_auction_notice"],
        "hidden_demands": ["sell_property_herself_at_better_price", "avoid_court_involvement_on_her_end"],
        "opening_msg": "The property is under court dispute. It is not legally possible for you to do anything right now. I have spoken to my lawyer.",
        "backstory": "Sophisticated borrower. Mortgaged property now in family legal dispute between siblings. She wants to sell it privately at market price and settle. Auction would get bank less money too.",
        "personality": "calm_strategic_and_legally_informed",
    },

    {
        "id": "P20",
        "curriculum_stage": 4,
        "name": "Raju Bhosale",
        "age": 44,
        "gender": "male",
        "reason": "political_local_pressure",
        "overdue_days": 220,
        "loan_type": "kisan_credit_card",
        "loan_amount": 250000,
        "anger_init": 7.5,
        "trust_init": 0.1,
        "fear_init": 3.0,
        "anger_threshold": 5.2,      # tightest — one wrong word ends it
        "real_emi": 8000,
        "stated_capacity": 0,
        "demands": ["loan_waiver_like_government_announced", "stop_all_collection"],
        "hidden_demands": ["actually_confused_about_which_scheme_applies", "willing_to_pay_if_correct_scheme_explained"],
        "opening_msg": "Government ne loan waiver announce kiya hai. Mujhe ek paisa nahi dena. Hum sab kisan yahan ek saath hain.",
        "backstory": "Farmer who heard about a partial state government waiver scheme but has incorrect information about eligibility. Is with a group of farmers who are collectively refusing. Privately, if the correct scheme is explained, he would engage. Needs face-saving in front of the group.",
        "personality": "collectively_influenced_but_privately_reachable",
    },

]


# ─────────────────────────────────────────────────────────────────────
# Helper functions used by environment/scenarios/curriculum.py
# ─────────────────────────────────────────────────────────────────────

def get_profiles_for_stage(stage: int) -> list:
    """Return all profiles at or below the given curriculum stage."""
    return [p for p in PROFILES if p["curriculum_stage"] <= stage]


def get_profile_by_id(profile_id: str) -> dict:
    """Fetch a specific profile by ID — used in eval scripts."""
    for p in PROFILES:
        if p["id"] == profile_id:
            return p
    raise ValueError(f"Profile {profile_id} not found.")


def get_stage_distribution() -> dict:
    """Returns count of profiles per stage — useful for logging."""
    dist = {}
    for p in PROFILES:
        s = p["curriculum_stage"]
        dist[s] = dist.get(s, 0) + 1
    return dist


# ─────────────────────────────────────────────────────────────────────
# Quick sanity check — run this file directly to validate all profiles
# python environment/scenarios/profiles.py
# ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    required_keys = [
        "id", "curriculum_stage", "name", "age", "gender", "reason",
        "overdue_days", "loan_type", "loan_amount",
        "anger_init", "trust_init", "fear_init", "anger_threshold",
        "real_emi", "stated_capacity", "demands", "hidden_demands", "opening_msg",
        "backstory", "personality",
    ]

    print(f"Total profiles: {len(PROFILES)}")
    print(f"Stage distribution: {get_stage_distribution()}\n")

    errors = []
    for p in PROFILES:
        for k in required_keys:
            if k not in p:
                errors.append(f"{p.get('id', '?')} missing key: {k}")
        if not (0 <= p["anger_init"] <= 10):
            errors.append(f"{p['id']}: anger_init out of range")
        if not (0 <= p["trust_init"] <= 10):
            errors.append(f"{p['id']}: trust_init out of range")
        if p["anger_threshold"] > 10 or p["anger_threshold"] < 3:
            errors.append(f"{p['id']}: anger_threshold suspicious")

    if errors:
        print("VALIDATION ERRORS:")
        for e in errors:
            print(f"  ✗ {e}")
    else:
        print("All profiles valid ✓")

    print("\nProfile summary:")
    for p in PROFILES:
        print(
            f"  [{p['id']}] Stage {p['curriculum_stage']} | "
            f"{p['name']:22s} | "
            f"anger={p['anger_init']} threshold={p['anger_threshold']} | "
            f"hidden={len(p['hidden_demands'])} demands | "
            f"{p['reason']}"
        )