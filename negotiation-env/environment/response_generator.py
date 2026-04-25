# environment/response_generator.py
from typing import Dict, List, Any

"""
HOLDS: deterministic template bank for borrower dialogue.
RUNS: called by adversary.py to generate responses.
CONNECTS TO: adversary.py.
"""

# ─────────────────────────────────────────────────────────────────────────────
# TEMPLATES BANK
# Organized by [Signal Type][Emotion Zone]
# ─────────────────────────────────────────────────────────────────────────────

TEMPLATES: Dict[str, Dict[str, List[str]]] = {
    "empathize": {
        "calm": [
            "I appreciate you listening to me, {name}. It's been hard since the {reason}.",
            "It's rare to find someone at the bank who actually listens. My {loan_type} has been a lot of stress.",
            "Thank you for saying that. I really am trying my best to manage everything.",
            "I hope you mean that. I've been very worried about my {backstory_short} situation.",
            "It helps to feel understood. My priority is to resolve this {loan_type} as soon as I can.",
            "I'm glad we can talk like this. Most people just demand money without hearing the reason."
        ],
        "agitated": [
            "You say you understand, but the {loan_type} is still overdue and the pressure is real.",
            "Words are fine, but my situation with {reason} isn't going away.",
            "I hear you, but I'm still getting three calls a day. It doesn't feel like you understand.",
            "If you understood, you'd know why I'm asking for {demands_str}.",
            "I appreciate the words, but I need a real solution for my {loan_amount} loan.",
            "I'm trying to stay calm, but it's hard when {backstory_short} is constantly on my mind."
        ],
        "angry": [
            "Don't give me your scripted empathy. If you understood, you'd stop the harassment!",
            "You 'understand'? Then why are your agents calling my family about {reason}?",
            "Save your pity. Just tell me what can be done about the {loan_type} interest.",
            "You say this is tough? Try being in my shoes with {backstory_short} and no help!",
            "Stop pretending to care. You just want the {loan_amount} paid, that's all.",
            "I don't need your 'understanding.' I need you to listen to my {demands_str}!"
        ],
        "panicked": [
            "Please... if you understand, then help me. I'm so scared about the {loan_type}.",
            "I don't know what to do... my {reason} has ruined everything. Please don't be harsh.",
            "I'm at my limit. If you really see my struggle, please give me some time.",
            "Everything is falling apart... {backstory_short}... I can't breathe with this debt.",
            "Please... just don't tell my family. I'll do anything to fix this.",
            "I'm lost. I just need one person to actually help me with this {loan_amount}."
        ]
    },
    "clarify": {
        "calm": [
            "Let me explain. Since the {reason}, my monthly income hasn't been enough for {loan_type}.",
            "To be clear, I'm not refusing to pay. My {backstory_short} makes it complicated.",
            "The main issue is my {demands_str}. If we solve that, I can pay the {loan_amount}.",
            "I want to ensure you have the full picture of why the {loan_type} is delayed.",
            "The situation started when {backstory_short}. That's why I'm in this position.",
            "I'm trying to be transparent. The {reason} was unexpected for my family."
        ],
        "agitated": [
            "I've told you already, the {reason} changed everything! I can't pay the full EMI.",
            "Listen, my {backstory_short} is the reason the {loan_type} is stuck. It's simple.",
            "I'm telling you the truth. My {demands_str} are what I need addressed first.",
            "Can't you see? My {loan_amount} loan is overdue because of the {reason}!",
            "I'm trying to explain, but it feels like you're not listening to the {backstory_short} part.",
            "My situation is {reason}. I don't know how else to clarify it for the bank."
        ],
        "angry": [
            "How many times do I have to repeat about the {reason}? Are you even recording this?",
            "My {backstory_short} is why I can't pay! Stop asking the same questions!",
            "I'm being clear: I need {demands_str} or there is no way forward with this {loan_amount}.",
            "You keep pushing for the {loan_type} payment, but you ignore the {reason} entirely!",
            "I've clarified everything! My {backstory_short} is a fact, not an excuse!",
            "Read your notes! I've explained the {reason} fifty times already!"
        ],
        "panicked": [
            "I'm trying to tell you... {backstory_short}... please, it's just so hard.",
            "The {reason} happened and I lost control of the {loan_type}. I'm so sorry.",
            "Please, listen carefully... my {demands_str} are all I care about right now.",
            "I don't know how else to say it... {backstory_short}... I'm desperate.",
            "The {loan_amount} is too much for me now because of {reason}. Please believe me.",
            "I'm so confused... {backstory_short}... please tell me you understand now."
        ]
    },
    "probe_capacity": {
        "calm": [
            "You're asking about my capacity. Truthfully, {stated_capacity} is all I can manage for now.",
            "I've looked at my finances. With {reason}, I can only commit to {stated_capacity} per month.",
            "I can't pay the full {loan_amount}, but I can start with small amounts if you help.",
            "My current capacity is low because of {backstory_short}. Let's discuss a realistic plan.",
            "I want to be honest. I can pay {stated_capacity} now and more later when things improve.",
            "My priority is {demands_str}, but I can put some money toward the {loan_type} too."
        ],
        "agitated": [
            "Why are you asking about my income again? I told you, {stated_capacity} is my limit!",
            "I can't magic money out of nowhere. {backstory_short} has left me with nothing.",
            "I'm trying to stay afloat. Even {stated_capacity} is a stretch with the {reason}.",
            "You keep probing, but the answer remains {stated_capacity}. I can't do more.",
            "My {loan_amount} is a burden. Asking about my capacity won't change the {reason}!",
            "I don't have hidden money. {backstory_short} is my current reality."
        ],
        "angry": [
            "Are you calling me a liar? I said {stated_capacity} and I meant it!",
            "Stop digging into my life! The {reason} happened, and that's all you need to know!",
            "You want to know my capacity? It's zero if you keep harassing me about the {loan_type}!",
            "My {backstory_short} is none of your business beyond the {loan_amount} I owe!",
            "I'm sick of these questions. {stated_capacity} is the final word.",
            "Why should I tell you anything? Your bank didn't help when the {reason} hit!"
        ],
        "panicked": [
            "I don't have anything... {backstory_short}... please don't pressure me.",
            "Even {stated_capacity} is so hard to find right now. My {reason} was a disaster.",
            "I'm scared... I don't know how I'll even pay {stated_capacity}. Please help me.",
            "The {loan_type} is 150 days overdue and I'm drowning. {backstory_short} is too much.",
            "Please... I'm doing my best. {stated_capacity} is all I can possibly offer.",
            "I'm so worried... if you ask for more than {stated_capacity}, I won't be able to eat."
        ]
    },
    "offer_plan": {
        "calm": [
            "A plan for the {loan_type}? I'm listening. If it addresses my {demands_str}, I'm interested.",
            "I appreciate the offer. Let me see if the EMI fits my current {stated_capacity}.",
            "If the restructuring helps with the {reason} situation, I'll take it.",
            "Finally, a realistic approach. I want to settle the {loan_amount} properly.",
            "Let's discuss. I need the plan to be manageable given my {backstory_short}.",
            "Small EMIs would really help. {stated_capacity} is what I can realistically commit to."
        ],
        "agitated": [
            "Is the EMI high? Because my {reason} has made me very tight for cash.",
            "I need this plan to be final. No more changes for the {loan_type} later.",
            "Will this stop the calls? Because my {backstory_short} is already enough stress.",
            "I'll listen, but it has to be better than a simple demand for the {loan_amount}.",
            "Show me the numbers. My {stated_capacity} is fixed, so the plan must fit.",
            "I hope this plan includes my request for {demands_str}."
        ],
        "angry": [
            "Another 'plan'? Is this just another trick to get the {loan_amount} out of me?",
            "Unless this plan solves the {reason} issue, I don't want to hear it!",
            "You're making an offer now? After all the harassment about the {loan_type}?",
            "It better be a deep discount. My {backstory_short} deserves some consideration.",
            "I'm not signing anything until I'm sure my {demands_str} are met!",
            "Why should I trust your plan? You've been pushing me since day one!"
        ],
        "panicked": [
            "Please make it a small amount... I'm so scared of the {loan_type} debt.",
            "Will this fix everything? Will the {reason} pressure finally stop?",
            "I'll try to pay... just make the EMI very low. {stated_capacity} is all I have.",
            "Oh god, please let this work. My {backstory_short} situation is so fragile.",
            "I'm desperate for a way out of the {loan_amount}. Please, tell me the plan.",
            "Please don't take my home... I'll agree to a plan if it's manageable."
        ]
    },
    "seek_commitment": {
        "calm": [
            "I can commit to {stated_capacity} by next week. I give you my word.",
            "Yes, I will pay. My {reason} is resolving and I want the {loan_type} settled.",
            "I am a man of my word. I will ensure the {loan_amount} is paid as agreed.",
            "You have my commitment. I just need you to respect my {demands_str} in return.",
            "I'll pay. My {backstory_short} won't stand in the way of my word.",
            "Pukka promise. I'll make the first payment on the {loan_type} by Monday."
        ],
        "agitated": [
            "I'll try my best. But you have to understand the {reason} is still fresh.",
            "I'm promising, but don't hold the {loan_type} over my head if there's a day's delay.",
            "I'll give you {stated_capacity}, but I need that written confirmation we discussed.",
            "It's a promise, okay? Just stop the pressure about the {loan_amount} for a while.",
            "I'm committing, but my {backstory_short} is still a daily struggle.",
            "Fine, I agree. But please ensure no one visits my home for the {loan_type}."
        ],
        "angry": [
            "I'll pay when I can! My word is better than your bank's threats anyway!",
            "You want a commitment? Fix the {reason} interest and then we'll talk!",
            "I'm not promising anything until I see my {demands_str} in writing!",
            "Stop pushing for a 'promise'. My {backstory_short} is real, and life is unpredictable!",
            "I'll pay the {loan_amount} when I have it! That's my commitment, take it or leave it!",
            "You have my word that I won't forget how your bank treated me during {reason}!"
        ],
        "panicked": [
            "I promise! Just please don't tell my neighbors about the {loan_type}!",
            "I'll pay whatever I can... {stated_capacity}... just please help me.",
            "I give you my word, just stop the calls! My {reason} is too much to bear.",
            "I'll sign anything, just don't take the {loan_amount} to court. Please.",
            "Pukka waada. I'll pay. Please just give me one more chance.",
            "I'm so sorry... I'll commit to the {loan_type}. Please, I'm trying."
        ]
    },
    "warn_noncompliance": {
        "calm": [
            "You are mentioning legal action. I know my rights under the RBI guidelines.",
            "Threats won't help resolve the {loan_amount}. Let's be professional.",
            "I'm aware of the consequences. But my {reason} is a real hardship.",
            "If you go legal, it will just take longer for everyone. {loan_type} is best settled here.",
            "I don't appreciate the tone. My {backstory_short} doesn't make me a criminal.",
            "We should avoid escalation. My {demands_str} are very reasonable."
        ],
        "agitated": [
            "Why are you threatening me? My {reason} wasn't my fault!",
            "This is harassment! I will report this to the consumer forum immediately.",
            "You're pushing me too far. My {backstory_short} is stress enough without your threats.",
            "Legal action? For a {loan_amount} loan during a family crisis?",
            "I'm not scared of your 'consequences'. My {loan_type} will be paid when I'm able.",
            "Stop the threats. It won't make the money appear any faster."
        ],
        "angry": [
            "Go to court then! See if the judge cares about your bank more than my {reason}!",
            "I'm going to file an FIR for this harassment! You can't threaten me like this!",
            "You think I'm a thief? My {backstory_short} is a tragedy, and you're making it worse!",
            "Tell your senior I'm not paying a paisa if this attitude continues!",
            "I'll call the police myself! Your agents are visiting my neighborhood for {loan_type}!",
            "Shut up with the threats! I'll see you in the consumer forum, you wait and watch!"
        ],
        "panicked": [
            "Please... no legal action. I'll pay the {loan_amount}. Please don't do this.",
            "Oh god, police? No, please, my family... {backstory_short}... I'll fix it.",
            "I'm begging you, don't escalate. I'll find the money for {loan_type} somehow.",
            "I'll be ruined... {reason}... please, just one more week. Don't go to court.",
            "Please... I'm so scared. I'll agree to anything, just no legal notices.",
            "Don't visit my house... my {backstory_short}... I can't let them know."
        ]
    },
    "unknown": {
        "calm": [
            "I'm not sure what you're asking. Can we focus on the {loan_type}?",
            "I hear you, but my priority is resolving the {loan_amount} debt.",
            "Let's get back to the point. My {reason} is the main issue here.",
            "I'm a bit confused by that statement. How does it help with my {demands_str}?",
            "Can we stay focused? I want to settle this {loan_type} and move on.",
            "I didn't quite catch that. Let's talk about the next steps for my EMI."
        ]
    }
}

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def deterministic_hash(string: str) -> int:
    """A cross-run deterministic integer hash."""
    h = 0
    for char in string:
        h = (h * 31 + ord(char)) & 0xFFFFFFFF
    return h

def _inject_profile(template: str, profile: dict, state_dict: dict) -> str:
    """Replaces placeholders in the template with actual profile and state values."""
    backstory = profile.get("backstory", "")
    backstory_short = backstory[:80] + ("..." if len(backstory) > 80 else "")
    
    demands = profile.get("demands_stated", profile.get("demands", []))
    demands_str = ", ".join(demands) if demands else "no specific demands"
    
    return template.format(
        name=profile.get("name", "Customer"),
        age=profile.get("age", "N/A"),
        reason=profile.get("reason", "financial hardship"),
        loan_type=profile.get("loan_type", "loan"),
        loan_amount=profile.get("loan_amount", "total amount"),
        backstory_short=backstory_short,
        demands_str=demands_str,
        anger=round(state_dict.get("anger", 0), 1),
        trust=round(state_dict.get("trust", 0), 1),
        fear=round(state_dict.get("fear", 0), 1),
        stated_capacity=profile.get("stated_capacity", "a small amount")
    )

def pick_template(signal_type: str, state_dict: dict, profile: dict) -> str:
    """
    Selects a deterministic response template and injects dynamic values.
    Pure and deterministic function.
    """
    zone = state_dict.get("zone", "calm")
    
    # Fallback for unknown signals or zones
    if signal_type not in TEMPLATES:
        signal_type = "unknown"
    
    zone_templates = TEMPLATES[signal_type]
    if zone not in zone_templates:
        # Fallback to calm for the signal type if specific zone is missing
        if "calm" in zone_templates:
            zone = "calm"
        else:
            signal_type = "unknown"
            zone = "calm"
            zone_templates = TEMPLATES[signal_type]

    template_list = zone_templates[zone]
    
    # Deterministic selection
    turn = state_dict.get("turn", 0)
    p_id = profile.get("id", "P00")
    # key incorporates identifiers + turn count to ensure variety but determinism
    key = f"{signal_type}:{zone}:{p_id}:{turn}"
    index = deterministic_hash(key) % len(template_list)
    
    selected_template = template_list[index]
    
    return _inject_profile(selected_template, profile, state_dict)
