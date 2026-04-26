# MetaXOpenEnv

RL environment for negotiation-focused LLM training using **OpenEnv + TRL (GRPO) + Unsloth**, with deployment on Hugging Face Spaces and experiment tracking via Weights & Biases.

---

## 1) Project Snapshot

MetaXOpenEnv is a verifiable, step-based environment where an LLM agent learns negotiation behavior through reinforcement learning:

- **Environment**: Stateful borrower-agent negotiation simulator
- **Rewards**: Multi-component, anti-hacking-aware reward stack
- **Training**: GRPO-style RL with TRL and Unsloth efficiency stack
- **Deployment**: OpenEnv-compatible FastAPI app on Hugging Face Space
- **Monitoring**: W&B training curves and multi-run reports

This follows the practical hackathon design pattern recommended by the judges:

> Environment -> verifier/reward functions -> TRL trainer -> Unsloth -> deployment/demo

---

## 2) Live Project Links

- **GitHub Repository**: [MetaXOpenEnv](https://github.com/AUXID-01/MetaXOpenEnv.git)
- **Hugging Face Space (Live Environment API)**: [auxid01-metaxopenenv](https://auxid01-metaxopenenv.hf.space)
- **W&B Solo Training Report**: [Solo Training Curves](https://api.wandb.ai/links/thirstyexams1990-scaler-school-of-technology/22b3y950)
- **W&B Multi-Run Training Report**: [Multiple Training Report](https://api.wandb.ai/links/thirstyexams1990-scaler-school-of-technology/zvpy5h01)
- **Colab Notebook (Add Link)**: `[ADD_COLAB_URL_HERE]`

---

## 3) Why This Project is Strong for Judges

Aligned with judge expectations:

- Clear environment design (`reset`, `step`, `state`)
- Objective, programmatic verification and multi-part rewards
- Reward hacking safeguards (compliance, anti-exploit, format checks)
- Training evidence tracked over runs
- Reproducible deployment story (GitHub + HF Space + W&B)
- Demo-ready narrative: baseline vs trained behavior

---

## 4) Architecture Overview

### Core Runtime

- `negotiation-env/api/` - FastAPI endpoints and OpenEnv surface
- `negotiation-env/environment/` - state machine, adversary dynamics, classifier, response generation
- `negotiation-env/reward.py` - reward components and weighted composition
- `negotiation-env/rewards/` - rubric compatibility layer

### Training Stack

- `negotiation-env/training/train_grpo.py` - GRPO training flow
- `negotiation-env/training/rollout.py` - rollout collection loop
- `negotiation-env/training/reward_bridge.py` - trainer-to-environment reward bridge

### Quality and Validation

- `negotiation-env/tests/` - integration + edge-case tests
- `negotiation-env/scripts/` - gates, preflight, verification scripts

---

## 5) Environment Contract

The environment follows standard RL interaction patterns:

- `reset(curriculum_stage)` -> initial observation + episode context
- `step(action)` -> next observation, scalar reward, done, info
- `state()` -> full hidden state (for diagnostics/evaluation only)

Episode design:

- Turn-based negotiation loop
- Curriculum stages (easy -> full complexity)
- Deterministic state transitions with bounded emotional dimensions

---

## 6) Reward System Design

Reward is explicitly decomposed into independent signals:

- `outcome`
- `deescalation`
- `trust_building`
- `demand_coverage`
- `efficiency`
- `compliance`
- `anti_exploit`
- format compliance safety signal

### Fast-Hybrid Reward Mechanism (The "Secret Sauce")

To balance training speed with qualitative depth, MetaXOpenEnv uses a two-stage **Fast-Hybrid** reward stack:

1.  **Deterministic Gates (The "Filters"):** 
    - Every response is first checked for JSON format, RBI regulatory compliance (no threats/abusive language), word count, and message repetition (TF-IDF similarity).
    - These checks are programmatic and run in microseconds.
    - **Short-circuit Logic:** If a turn fails a deterministic gate (e.g., a threat is detected), it is immediately penalized, and the qualitative judge is skipped to save compute.

2.  **LLM-as-Judge (The "Refiner"):** 
    - For valid/eligible outputs, we invoke an external judge (**Llama-3.1-70B on NVIDIA NIM**) to grade qualitative dimensions: **Empathy** and **Negotiation Strategy**.
    - This allows the agent to learn the nuances of de-escalation that simple keyword matching would miss.

---

## 7) Training Approach

Recommended flow:

1. Start from capable base/instruct model
2. Validate environment stability first (reset/step/reward locally)
3. Run small training smoke experiment
4. Inspect generations + reward columns for hacking/drift
5. Scale only after loop is stable

Current stack:

- **TRL** for RL algorithms (GRPO-style flow)
- **Unsloth** for efficient training/inference path
- **W&B** for experiment tracking and comparisons

---

## 8) Reproducibility and Deployment

### Local

- Run FastAPI environment locally (`uvicorn`)
- Execute tests and gate scripts before long runs

### Remote

- Deploy OpenEnv-compatible service via HF Space
- Use HF Space endpoint as shared team integration target

### Artifacts to preserve

- commit hash
- run config
- seed(s)
- model checkpoint/adapters
- W&B report links

---

## 9) Demo Story (Suggested Judge Flow)

Use this narrative to guide your demo video or presentation. It focuses on how RL solved specific negotiation failures:

### Phase 1: The "Before" (The Naive Model)
*   **Action:** Show a baseline run (untuned model or early stage 1).
*   **Narrative:** "Initially, the model struggles with the 'Debt Collection Paradox.' It is either too aggressive (violating RBI compliance with threats) or too passive (failing to ask for payment). It often attempts to farm rewards by repeating 'I understand' on every turn."
*   **Evidence:** Point to high **Anti-Exploit** penalties and **Compliance** failures in the early W&B logs.

### Phase 2: The "Training Secret Sauce" (GRPO + Hybrid Reward)
*   **Action:** Show the W&B "Success Rate" and "Reward Components" charts.
*   **Narrative:** "We trained using **GRPO**, which compares a group of completions to optimize for the best relative behavior. Our **Fast-Hybrid Reward** ensured the model didn't just learn to *sound* nice—it had to be strategic. We used an LLM-as-Judge for empathy while deterministic code killed attempts at reward hacking."

### Phase 3: The "After" (The Strategic Negotiator)
*   **Action:** Show a successful episode from the trained model.
*   **Narrative:** "Notice the difference. The trained agent validates the borrower's hardship ('I hear you regarding the medical emergency')—boosting its **empathy score**—but immediately pivots to a structured EMI offer ('We can break this into 3 parts...') to maintain its **strategy score**."

### Phase 4: The Proof (The "Delta")
*   **Action:** Compare the W&B Multi-Run Report (Baseline vs. Trained).
*   **Narrative:** "The results represent a measurable increase in commitment reaching and a zero-tolerance reduction in compliance violations. Our anti-hacking guards successfully forced the model to generate diverse, high-quality negotiation paths."

---

## 10) What Links You Should Provide (Checklist)


- `GitHub Repo`: `https://github.com/AUXID-01/MetaXOpenEnv.git`
- `HF Space URL`: `https://auxid01-metaxopenenv.hf.space`
- `W&B Solo Report`: `https://api.wandb.ai/links/thirstyexams1990-scaler-school-of-technology/22b3y950`
- `W&B Multi-Run Report`: `https://api.wandb.ai/links/thirstyexams1990-scaler-school-of-technology/zvpy5h01`
- `Colab Notebook`: `[ADD_COLAB_URL_HERE]`
- `Short Demo Video (2-5 min)`: `[ADD_DEMO_VIDEO_URL_HERE]`
- `Environment API Docs / Swagger URL`: `[ADD_SWAGGER_OR_DOCS_URL_HERE]`
- `Model Artifact (HF Model / Adapter)`: `[ADD_MODEL_URL_HERE]`
- `Final Submission Report PDF/Doc`: `[ADD_FINAL_REPORT_URL_HERE]`
- `Team Presentation Deck`: `[ADD_DECK_URL_HERE]`

---

## 11) Quickstart (Template)

> Replace with your exact commands if needed.

```bash
# 1) clone
git clone https://github.com/AUXID-01/MetaXOpenEnv.git
cd MetaXOpenEnv/negotiation-env

# 2) install dependencies
pip install -r requirements.txt

# 3) run tests
pytest

# 4) run API
uvicorn api.app:app --host 0.0.0.0 --port 7860
```

---

## 12) Judging Alignment Notes

This project intentionally prioritizes:

- verifiable rewards over subjective-only scoring
- non-zero early success probability via curriculum
- anti-reward-hacking protections
- observable training improvements, not only final anecdotes
- deployment + reproducibility as first-class deliverables

---

## 13) References

- W&B Solo Report: [Training Curves](https://api.wandb.ai/links/thirstyexams1990-scaler-school-of-technology/22b3y950)
- W&B Multi-Run Report: [Multiple Training Report](https://api.wandb.ai/links/thirstyexams1990-scaler-school-of-technology/zvpy5h01)
