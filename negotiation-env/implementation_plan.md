# Training Scaffolding (Person C) for Debt Restructuring Environment

This plan focuses purely on building out the **Person C (Training)** component in a decoupled way so it is fully prepared and mocked while Person A and B stabilize the environment and reward logic. The focus remains exclusively on the **Debt Restructuring/NBFC** domain.

## Schema & Contract Audit (Debt/NBFC alignment check)

I've reviewed your current schemas and contracts (`contracts.py`, `api/schemas.py`, `rewards/rubric_input.py`). 
*   **Alignment:** They are perfectly aligned with the Debt domain. Your defined `ACTION_TYPES` (e.g., `offer_emi`, `acknowledge_hardship`), reward breakdown (e.g., `compliance` explicitly checking RBI rules), and variable naming (`borrower_msg` instead of `adversary_msg`) directly support the loan default narrative. No "crisis/hostage negotiation" traces remain in the contracts. 
*   **WandB Logs:** The `info` dict returning `reward_breakdown` with scalar components exactly matches the `contracts.py` specifications.

## Proposed Changes

We will scaffold the following files within the `training/` and `client/` directories.

### `training/prompt_builder.py` 

#### [NEW] `prompt_builder.py`
We will implement two critical functions independent of the actual OpenEnv.
*   **`build_system_prompt()`**: A rigid system instructions template. It will tell the LLM:
    *   **Role**: Debt collection agent for an NBFC in India.
    *   **Goal**: Reach a sensible repayment commitment, de-escalate stress, maintain empathy, without violating RBI guidelines (no threats/predatory behavior).
    *   **Output Format Guide**: Must output an explicitly requested action (e.g., `offer_emi`, `send_message`) with the conversational text and required metadata (e.g., `emi_amount`).
*   **`obs_to_prompt(obs_dict)`**: A parser that takes the exact keys from `api.schemas.Observation` (`turn`, `borrower_msg`, `escalation_level`, `stated_demands`) and cleanly maps them into the string presented to the LLM on every step.

### `client/env_client.py` & `client/utils.py`

#### [NEW] `client/env_client.py`
*   Create a clean HTTP wrapper (`NegotiationEnvClient`) pointing at a base URL (ready for local Uvicorn or HF Space).
*   Implement a `DummyEnvClient` class mapping the exact same API signature (`reset()`, `step()`) that returns hard-coded payloads using your Pydantic schemas. This prevents us from being blocked by the environment team.

#### [NEW] `client/utils.py`
*   **`action_from_text(generated_text)`**: Rigorous regex parsing or JSON extraction to pull the LLM's text output back into the `{ "action_type": ..., "text": ..., "metadata": ... }` shape expected by the HTTP `/step` endpoint. Will include defaults for parsing errors.

### `training/rollout.py`

#### [NEW] `rollout.py`
*   **`run_episode()` loop**: Will connect a HuggingFace Model (or dummy completion engine during testing) wrapped with our prompt builder to the `env_client`. 
*   **Logic Flow**: Loop through `num_max_turns`. Send prompt -> Parse generation -> Hit `client.step()` -> Accumulate `(prompt, response, reward)` -> Check `done` flag -> Break if terminated.
*   This encapsulates an entire episode's trajectory data needed by the `GRPOTrainer`.

### `training/train_grpo.ipynb` (Skeleton)

#### [MODIFY] `train_grpo.ipynb`
Set up the fundamental shell of the training notebook intended for Google Colab/local Jupyter.
*   **Installation Cell**: `!pip install trl unsloth bitsandbytes wandb`
*   **Imports & WandB Setup**: Login scaffolding mapping `info["reward_breakdown"]` components to independent WandB logged columns.
*   **Model Initialisation**: Boilerplate for loading Unsloth 4-bit `FastLanguageModel`.
*   **GRPO Skeleton**: Hook the `run_episode` rollout into TRL's `GRPOTrainer`. Includes placeholder Hyperparameters aligned with 1.5B/3B models (e.g., learning rates, micro-batch sizes).

## Open Questions

> [!WARNING]
> Parsing robustness is often the biggest failure point for LLMs in environments. Are we expecting perfectly structured JSON responses from the base model, or should we use `<action>send_message</action>` XML tags in the output? XML tagging has much higher success rates for base (instruct) models than complex JSON parsing.

## Verification Plan

1.  **Prompt Unit Test**: I will write simple dummy observations directly into `prompt_builder.py` to print out what the LLM will see, without starting an environment.
2.  **Mock Rollout execution**: By instantiating the `DummyEnvClient`, we will execute a 5-step mocked episode loop using `rollout.py` without needing the FastAPI server, ensuring data flow is smooth, dicts pack into schemas correctly, and the episode loop doesn't throw KeyErrors.
