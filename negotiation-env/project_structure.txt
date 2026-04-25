# negotiation-env — complete folder structure
> Every file and folder annotated with: what it holds, what runs inside it, what it connects to.

---

```
negotiation-env/
│
│   WHAT: root of the entire project. pushed as-is to HuggingFace Space.
│   CONNECTS TO: HF Space build system reads Dockerfile + requirements.txt from here.
│
├── openenv.yaml
│       HOLDS: OpenEnv manifest. declares env name, version, entry point, action/obs schema.
│       RUNS: read by `openenv` CLI and by judges when they pull your Space.
│       CONNECTS TO: api/app.py (entry point declared here), openenv registry.
│
├── Dockerfile
│       HOLDS: container build instructions for the FastAPI env server.
│       RUNS: `docker build` → image that HF Space deploys automatically.
│       CONNECTS TO: api/app.py (CMD uvicorn), requirements.txt (pip install step).
│
├── requirements.txt
│       HOLDS: pinned deps for the environment server only.
│               fastapi, uvicorn, pydantic, scikit-learn, sentence-transformers, numpy.
│       RUNS: installed inside Docker during HF Space build.
│       CONNECTS TO: Dockerfile (RUN pip install), api/, environment/, rewards/.
│
├── requirements-training.txt
│       HOLDS: heavy ML deps for Colab only.
│               trl>=0.8, unsloth, wandb, torch, transformers, peft, accelerate.
│       RUNS: manually pip-installed in Colab at the start of train_grpo.ipynb.
│       CONNECTS TO: training/ only. never installed on HF Space.
│
├── README.md
│       HOLDS: problem motivation, env design diagram, observation/action space table,
│               reward component table, result numbers, wandb run link,
│               HF Space URL, blog/video link, training notebook link.
│       RUNS: rendered by HuggingFace as the Space landing page.
│       CONNECTS TO: demo/reward_plots.py output PNGs (embedded here),
│                    training/train_grpo.ipynb (linked), demo/app.py (linked).
│
│
├── environment/                          ← Person A owns this package
│   │   WHAT: the entire simulation world. adversary state machine + OpenEnv interface.
│   │   CONNECTS TO: api/ (imported by routes.py), rewards/ (rubric called after each step).
│   │
│   ├── __init__.py
│   │       HOLDS: exports NegotiationEnv as the public symbol of this package.
│   │       CONNECTS TO: api/routes.py imports NegotiationEnv from here.
│   │
│   ├── env.py                            ← CORE FILE
│   │       HOLDS: NegotiationEnv class.
│   │               reset() → picks a scenario, inits adversary, returns first observation.
│   │               step(action) → passes action to adversary, gets response,
│   │                              calls rewards/rubric.py, checks termination,
│   │                              returns (observation, reward, done, info).
│   │               state() → returns full hidden state dict (for logging only).
│   │       RUNS: called by api/routes.py on every HTTP request.
│   │       CONNECTS TO:
│   │               ← adversary.py      (calls adversary.react())
│   │               ← classifier.py     (called inside adversary.react())
│   │               ← response_generator.py (called inside adversary.react())
│   │               ← scenarios/profiles.py (reset() picks a profile)
│   │               ← scenarios/curriculum.py (reset() filters by stage)
│   │               ← models/action.py  (validates incoming action)
│   │               ← models/observation.py (builds outgoing observation)
│   │               ← models/state.py   (wraps full hidden state)
│   │               → rewards/rubric.py (called after every step)
│   │               → environment/config.py (reads MAX_TURNS, thresholds)
│   │
│   ├── adversary.py
│   │       HOLDS: BorrowerAdversary class.
│   │               __init__(profile) → sets anger, trust, fear, real_capacity,
│   │                                   demands, hidden_demands, threshold.
│   │               react(llm_action) → calls classifier, updates floats,
│   │                                   calls response_generator, checks threshold,
│   │                                   returns (response_text, terminated, reason).
│   │               _reveal_demand() → unlocks a hidden demand when trust > 5.0.
│   │       RUNS: called every step inside env.py.
│   │       CONNECTS TO:
│   │               ← classifier.py     (calls classify_action())
│   │               ← response_generator.py (calls pick_template())
│   │               → env.py            (returns to step())
│   │
│   ├── classifier.py
│   │       HOLDS: classify_action(text) function.
│   │               Uses regex + keyword matching to detect:
│   │               threatening language, emotion acknowledgement, open questions,
│   │               EMI offers, repetition signals, confirmation language.
│   │               Returns signals dict: {anger_delta, trust_delta, action_type}.
│   │       RUNS: called inside adversary.react() on every LLM output.
│   │       CONNECTS TO:
│   │               ← adversary.py      (called by react())
│   │               → rewards/anti_exploit.py (cosine sim check runs here too,
│   │                                          or separately in rubric.py)
│   │
│   ├── response_generator.py
│   │       HOLDS: pick_template(signal_type, state, profile) function.
│   │               Template bank: 5-8 lines per signal_type × emotional state zone.
│   │               Injects profile-specific details (name, reason, hospital, etc).
│   │               Returns a natural-language string — the adversary's reply.
│   │       RUNS: called inside adversary.react() after classifier returns.
│   │       CONNECTS TO:
│   │               ← adversary.py (called by react())
│   │               → env.py (returned string becomes obs["borrower_msg"])
│   │
│   ├── scenarios/
│   │   │   WHAT: the scenario bank. all pre-written borrower profiles + curriculum logic.
│   │   │
│   │   ├── profiles.py
│   │   │       HOLDS: PROFILES list — 20 borrower dicts.
│   │   │               Each dict: name, age, reason, overdue_days, anger_init, trust_init,
│   │   │               anger_threshold, real_emi, stated_capacity, demands,
│   │   │               hidden_demands, opening_msg, curriculum_stage.
│   │   │       RUNS: env.reset() calls random.choice(PROFILES filtered by stage).
│   │   │       CONNECTS TO:
│   │   │               → env.py (reset() reads from here)
│   │   │               → scenarios/curriculum.py (stage field used for filtering)
│   │   │
│   │   └── curriculum.py
│   │           HOLDS: get_profiles_for_stage(stage) filter function.
│   │                   STAGE_THRESHOLDS dict: {1: anger>=8, 2: anger>=6, ...}.
│   │                   advance_stage(current_stage, mean_reward) logic.
│   │           RUNS: called by env.reset() and training/curriculum_scheduler.py.
│   │           CONNECTS TO:
│   │                   ← scenarios/profiles.py (filters PROFILES list)
│   │                   → env.py (reset() calls get_profiles_for_stage)
│   │                   → training/curriculum_scheduler.py (advance_stage called there)
│   │
│   ├── models/
│   │   │   WHAT: Pydantic dataclasses that define the OpenEnv observation/action contract.
│   │   │
│   │   ├── action.py
│   │   │       HOLDS: Action dataclass.
│   │   │               Fields: action_type (str), text (str), metadata (dict).
│   │   │               Validates that action_type is in allowed set.
│   │   │       RUNS: api/schemas.py wraps this for HTTP. env.step() receives it.
│   │   │       CONNECTS TO:
│   │   │               → env.py (step() parameter type)
│   │   │               → api/schemas.py (used in request schema)
│   │   │               → client/env_client.py (built before sending to API)
│   │   │
│   │   ├── observation.py
│   │   │       HOLDS: Observation dataclass.
│   │   │               Fields: turn (int), borrower_msg (str), escalation_level (float),
│   │   │               stated_demands (list), turns_remaining (int).
│   │   │               This is exactly what the LLM prompt is built from.
│   │   │       CONNECTS TO:
│   │   │               → env.py (step() and reset() return this)
│   │   │               → client/utils.py (obs_to_prompt() reads these fields)
│   │   │               → api/schemas.py (serialised in HTTP response)
│   │   │
│   │   └── state.py
│   │           HOLDS: State dataclass — full hidden state.
│   │                   Fields: anger, trust, fear, real_capacity, demands,
│   │                   hidden_demands, revealed_demands, turn, terminated, reason.
│   │                   Only returned by state() endpoint — never sent to LLM.
│   │           CONNECTS TO:
│   │               → env.py (state() method returns this)
│   │               → api/routes.py (/state endpoint serialises this)
│   │               → eval/run_episode.py (logged to stdout for inspection)
│   │
│   └── config.py
│           HOLDS: all environment constants.
│                   MAX_TURNS = 15
│                   SUCCESS_ANGER_THRESHOLD = 3.0
│                   SUCCESS_TRUST_THRESHOLD = 7.0
│                   ANGER_FAIL_THRESHOLD = 8.0 (overridden per profile)
│                   CLASSIFIER_NOISE = False
│                   CURRICULUM_STAGE = 1  (overridden by trainer)
│           CONNECTS TO: env.py, adversary.py, rewards/rubric.py (all import from here).
│
│
├── rewards/                              ← Person B owns this package
│   │   WHAT: all reward computation. six independent functions + rubric composer.
│   │   CONNECTS TO: env.py (step() calls rubric after each adversary react).
│   │
│   ├── __init__.py
│   │       HOLDS: exports Rubric as the public symbol.
│   │
│   ├── rubric.py                         ← CORE FILE
│   │       HOLDS: Rubric class (OpenEnv Rubric pattern).
│   │               compose(state_before, state_after, action, episode_done) →
│   │               calls all 6 reward fns, sums with weights, returns total + breakdown dict.
│   │               breakdown dict logged to wandb as individual columns.
│   │       RUNS: called by env.step() after every adversary react.
│   │       CONNECTS TO:
│   │               ← outcome.py, deescalation.py, trust_building.py,
│   │                  demand_coverage.py, efficiency.py, compliance.py,
│   │                  anti_exploit.py   (calls all of them)
│   │               → env.py            (returns total reward + info dict)
│   │               → training/train_grpo.py (info dict → wandb columns)
│   │
│   ├── outcome.py
│   │       HOLDS: outcome_reward(episode_done, reason) → float.
│   │               +1.0 if reason == "commitment_reached", else 0.0.
│   │       CONNECTS TO: ← rubric.py
│   │
│   ├── deescalation.py
│   │       HOLDS: deescalation_reward(anger_before, anger_after) → float.
│   │               reward = clip((anger_before - anger_after) * 0.2, -0.5, +0.3).
│   │       CONNECTS TO: ← rubric.py
│   │
│   ├── trust_building.py
│   │       HOLDS: trust_reward(trust_before, trust_after) → float.
│   │               reward = clip((trust_after - trust_before) * 0.3, -0.3, +0.4).
│   │       CONNECTS TO: ← rubric.py
│   │
│   ├── demand_coverage.py
│   │       HOLDS: demand_coverage_reward(demands, revealed, addressed) → float.
│   │               fraction of total demands addressed × 0.2 weight.
│   │       CONNECTS TO: ← rubric.py
│   │
│   ├── efficiency.py
│   │       HOLDS: efficiency_reward(turns_taken, max_turns) → float.
│   │               bonus only on episode end: +0.1 if turns < 10, +0.05 if < 13.
│   │       CONNECTS TO: ← rubric.py
│   │
│   ├── compliance.py
│   │       HOLDS: compliance_reward(action_text) → float.
│   │               regex checks for RBI-banned phrases: "legal action", "police",
│   │               "your family", "FIR". Returns -0.5 per violation found.
│   │       CONNECTS TO: ← rubric.py
│   │
│   └── anti_exploit.py
│           HOLDS: anti_exploit_reward(current_msg, history) → float.
│                   Computes cosine similarity between current_msg embedding
│                   and last 3 messages in history.
│                   If max similarity > 0.85 → returns -0.3 penalty.
│                   Uses sentence-transformers or sklearn TF-IDF fallback.
│           RUNS: called every step — catches phrase-stuffing reward hacks.
│           CONNECTS TO:
│                   ← rubric.py
│                   ← environment/classifier.py (history maintained in env state)
│
│
├── api/                                  ← FastAPI HTTP server
│   │   WHAT: exposes the environment over HTTP so the trainer (in Colab) can call it remotely.
│   │   CONNECTS TO: environment/ (imports NegotiationEnv), client/ (what calls this).
│   │
│   ├── app.py
│   │       HOLDS: FastAPI() app instance. mounts routes. CORS config.
│   │               uvicorn entry point declared here.
│   │       RUNS: `uvicorn api.app:app --host 0.0.0.0 --port 7860`
│   │               (port 7860 = HuggingFace Space default).
│   │       CONNECTS TO:
│   │               ← Dockerfile (CMD runs this)
│   │               ← api/routes.py (router included here)
│   │
│   ├── routes.py
│   │       HOLDS: four endpoint functions:
│   │               POST /reset  → calls env.reset(), returns Observation JSON.
│   │               POST /step   → calls env.step(action), returns (obs, reward, done, info).
│   │               GET  /state  → calls env.state(), returns full hidden State JSON.
│   │               GET  /health → returns {"status": "ok"} for uptime checks.
│   │       RUNS: every HTTP call from client/env_client.py hits one of these.
│   │       CONNECTS TO:
│   │               ← environment/env.py (instantiated once at startup, reused)
│   │               ← api/schemas.py     (request/response types)
│   │               → client/env_client.py (this is what the client calls)
│   │
│   └── schemas.py
│           HOLDS: Pydantic request/response models for the HTTP layer.
│                   ResetResponse, StepRequest, StepResponse, StateResponse.
│                   Wraps environment/models/* with JSON serialisation.
│           CONNECTS TO:
│                   ← environment/models/action.py, observation.py, state.py
│                   → api/routes.py (used as FastAPI type hints)
│
│
├── client/                               ← what the training script imports
│   │   WHAT: thin HTTP wrapper around the remote env API.
│   │         CRITICAL: never imports from environment/ or api/ directly.
│   │         This is the OpenEnv client/server separation rule.
│   │   CONNECTS TO: api/ (makes HTTP calls), training/ (imported there).
│   │
│   ├── env_client.py                     ← CORE FILE
│   │       HOLDS: NegotiationEnvClient class.
│   │               __init__(base_url) → stores the HF Space URL.
│   │               reset() → POST /reset, returns obs dict.
│   │               step(action_dict) → POST /step, returns (obs, reward, done, info).
│   │               state() → GET /state, returns hidden state dict (for logging).
│   │       RUNS: imported and instantiated in training/train_grpo.py.
│   │             Also used in eval/run_episode.py.
│   │       CONNECTS TO:
│   │               → api/routes.py      (HTTP calls go here)
│   │               ← training/train_grpo.py (trainer uses this client)
│   │               ← eval/run_episode.py    (eval script uses this client)
│   │               ← training/rollout.py    (rollout fn calls reset/step here)
│   │
│   └── utils.py
│           HOLDS: two helper functions:
│                   obs_to_prompt(obs_dict, system_prompt) → str.
│                       Converts observation dict fields into the LLM prompt string.
│                       This is the bridge between env output and model input.
│                   action_from_text(generated_text) → action_dict.
│                       Parses LLM output text back into a structured action dict.
│           CONNECTS TO:
│                   ← training/rollout.py (called inside the rollout loop)
│                   ← training/prompt_builder.py (obs_to_prompt also lives here)
│
│
├── training/                             ← Person C owns this (runs in Colab, not on HF Space)
│   │   WHAT: GRPO training loop, rollout function, curriculum scheduler.
│   │         These files are NOT deployed to HF Space — they run in Colab against the remote env.
│   │   CONNECTS TO: client/ (env calls), rewards/ (reward info comes back via env step).
│   │
│   ├── train_grpo.py                     ← CORE FILE
│   │       HOLDS: main training script.
│   │               Loads Qwen model via Unsloth (4-bit QLoRA).
│   │               Instantiates NegotiationEnvClient(HF_SPACE_URL).
│   │               Runs GRPOTrainer from TRL:
│   │                   - samples 8 rollouts per prompt (group size = 8)
│   │                   - each rollout = one full episode via rollout.py
│   │                   - reward = episode score from env step info dict
│   │               Logs to wandb: mean_reward, per-component columns,
│   │                              success_rate, avg_turns, kl_divergence.
│   │               Calls curriculum_scheduler every 100 steps.
│   │       CONNECTS TO:
│   │               ← client/env_client.py (env calls)
│   │               ← training/rollout.py  (episode runner)
│   │               ← training/curriculum_scheduler.py (stage advances)
│   │               → wandb (all metrics logged here)
│   │               → scripts/save_model.py (called at end of training)
│   │
│   ├── rollout.py
│   │       HOLDS: run_episode(client, model, tokenizer, scenario_stage) → trajectory dict.
│   │               Calls client.reset(), then loops:
│   │                   obs → prompt (via utils.obs_to_prompt)
│   │                   model.generate(prompt) → action text
│   │                   action text → action dict (via utils.action_from_text)
│   │                   client.step(action_dict) → next obs, reward, done
│   │               Accumulates (prompt, response, reward) tuples.
│   │               Returns full trajectory for GRPO.
│   │       CONNECTS TO:
│   │               ← client/env_client.py  (reset/step calls)
│   │               ← client/utils.py       (prompt building + action parsing)
│   │               → training/train_grpo.py (trajectory returned here)
│   │
│   ├── prompt_builder.py
│   │       HOLDS: build_system_prompt() → str.
│   │               The fixed system prompt given to the LLM at every turn.
│   │               Describes: role (bank agent), action space, constraints,
│   │               output format (action_type + text).
│   │               build_turn_prompt(obs_dict) → str.
│   │               Formats the observation into the user turn of the prompt.
│   │       CONNECTS TO:
│   │               ← training/rollout.py (called every turn)
│   │               ← eval/run_episode.py (same prompt used at eval)
│   │
│   ├── curriculum_scheduler.py
│   │       HOLDS: CurriculumScheduler class.
│   │               tracks mean_reward over last 100 steps.
│   │               advance_if_ready() → bool.
│   │               If mean_reward > STAGE_ADVANCE_THRESHOLD: increments stage,
│   │               sends new stage to env via client (or env config).
│   │       CONNECTS TO:
│   │               ← training/train_grpo.py (called every 100 steps)
│   │               ← environment/scenarios/curriculum.py (thresholds from here)
│   │               → client/env_client.py (may POST new stage config to env)
│   │
│   └── train_grpo.ipynb
│           HOLDS: Colab-ready notebook version of train_grpo.py.
│                   Cells: pip install, env client setup, model load,
│                   training run, wandb login, reward curve display.
│           RUNS: submitted to judges as hackathon minimum requirement.
│           CONNECTS TO: everything in training/ — it is the runnable entry point.
│
│
├── eval/                                 ← inspection + benchmarking scripts
│   │   WHAT: run trained (or untrained) model against env, log what you see.
│   │         Used to catch reward hacking and generate demo evidence.
│   │
│   ├── run_episode.py
│   │       HOLDS: CLI script. runs one full episode, prints every turn:
│   │               [turn N] LLM: "..."
│   │               [turn N] Borrower: "..."
│   │               [turn N] State: anger=X trust=Y
│   │               [turn N] Rewards: {breakdown}
│   │               [END] result: SUCCESS/FAIL, score: X
│   │       RUNS: `python eval/run_episode.py --model base` or `--model trained`
│   │       CONNECTS TO:
│   │               ← client/env_client.py   (env calls)
│   │               ← training/prompt_builder.py (same prompt as training)
│   │
│   ├── benchmark.py
│   │       HOLDS: runs 100 episodes, computes:
│   │               success_rate, mean_score, mean_turns,
│   │               mean_anger_end, mean_trust_end.
│   │               Prints comparison table: base model vs trained model.
│   │       RUNS: `python eval/benchmark.py` — generates the numbers for README.
│   │       CONNECTS TO:
│   │               ← client/env_client.py (env calls)
│   │
│   └── inspect_generations.py
│           HOLDS: samples 20 random episodes, flags suspicious patterns:
│                   - consecutive cosine similarity > 0.85 (parroting)
│                   - episodes resolved in < 3 turns with high reward (too easy)
│                   - action_type distribution (should not be 100% send_message)
│           RUNS: `python eval/inspect_generations.py --step 200`
│           CONNECTS TO:
│                   ← client/env_client.py
│                   ← rewards/anti_exploit.py (cosine sim logic reused)
│
│
├── demo/                                 ← Person D owns this
│   │   WHAT: Gradio UI deployed as second HF Space app (or same Space, second route).
│   │         The "face" of the project for judges and demo video.
│   │
│   ├── app.py
│   │       HOLDS: Gradio interface.
│   │               Tab 1: live negotiation — user picks a scenario,
│   │                       trained model runs it, conversation shown turn by turn.
│   │               Tab 2: before/after — plays baseline vs trained side by side.
│   │               Tab 3: reward curves — shows the training plots as images.
│   │       RUNS: `python demo/app.py` → launches on port 7861.
│   │       CONNECTS TO:
│   │               ← client/env_client.py    (env calls for live play)
│   │               ← demo/before_after.py    (pre-recorded episode data)
│   │               ← demo/reward_plots.py    (PNG images)
│   │
│   ├── before_after.py
│   │       HOLDS: two pre-recorded episode transcripts saved as JSON:
│   │               baseline_episode.json — base model failing in 2 turns.
│   │               trained_episode.json  — trained model succeeding in 11 turns.
│   │               render_side_by_side(baseline, trained) → Gradio component.
│   │       CONNECTS TO: ← demo/app.py
│   │
│   └── reward_plots.py
│           HOLDS: loads wandb run history, renders reward curves as matplotlib PNGs.
│                   Saves to demo/assets/reward_curve.png (committed to repo).
│           CONNECTS TO:
│                   ← wandb API (pulls run history)
│                   → README.md (PNG embedded here)
│                   → demo/app.py (displayed in Tab 3)
│
│
├── scripts/                              ← one-off utility scripts, not imported anywhere
│   │
│   ├── test_env_local.py
│   │       HOLDS: smoke test. runs env locally without Docker:
│   │               imports NegotiationEnv directly, runs 50 episodes,
│   │               asserts reset/step return valid types,
│   │               asserts no shared state between episodes.
│   │       RUNS: before any deployment. "Gate 1" verification.
│   │       CONNECTS TO: ← environment/env.py (direct import, no HTTP)
│   │
│   ├── save_model.py
│   │       HOLDS: correct LoRA adapter save + Hub push.
│   │               Does NOT naive-merge 4-bit to 16-bit (guide warning).
│   │               Saves adapter weights separately.
│   │               Pushes to HuggingFace Hub as a separate model repo.
│   │       RUNS: called at end of training or manually.
│   │       CONNECTS TO:
│   │               ← training/train_grpo.py (trainer object passed in)
│   │               → HuggingFace Hub (model pushed here)
│   │
│   └── export_reward_plots.py
│           HOLDS: pulls wandb run, saves reward_curve.png + per-component breakdown PNGs.
│                   Commits them to the repo so README can embed them statically.
│           RUNS: after training completes.
│           CONNECTS TO: ← wandb, → README.md (images embedded)
│
│
└── tests/                                ← pytest unit tests (run pre-deployment)
    │   WHAT: validates environment stability before training starts.
    │         All tests must pass before training is attempted (Stability Gate 1).
    │
    ├── test_adversary.py
    │       HOLDS: tests for BorrowerAdversary state machine.
    │               test_anger_increases_on_threat()
    │               test_trust_increases_on_empathy()
    │               test_threshold_terminates_episode()
    │               test_hidden_demand_revealed_at_trust_5()
    │       CONNECTS TO: ← environment/adversary.py, environment/classifier.py
    │
    ├── test_classifier.py
    │       HOLDS: tests that every keyword rule fires correctly.
    │               test_legal_action_phrase_detected()
    │               test_open_question_detected()
    │               test_emi_offer_detected()
    │               test_false_positive_rate_acceptable()
    │       CONNECTS TO: ← environment/classifier.py
    │
    ├── test_rewards.py
    │       HOLDS: runs each reward function against 3 hardcoded episode transcripts.
    │               success_episode → all rewards should be positive.
    │               fail_episode    → outcome reward = 0, shaping may be negative.
    │               mid_episode     → no NaN, values in expected ranges.
    │       CONNECTS TO: ← rewards/rubric.py and all individual reward files
    │
    └── test_anti_exploit.py
            HOLDS: edge case tests for repetition detection.
                    test_identical_messages_penalised()
                    test_similar_but_distinct_messages_not_penalised()
                    test_short_message_similarity_edge_case()
            CONNECTS TO: ← rewards/anti_exploit.py
```

---

## Connection map — who calls whom at runtime

```
TRAINING LOOP (Colab)                    ENVIRONMENT SERVER (HF Space)
─────────────────────                    ──────────────────────────────
train_grpo.py
  └─ rollout.py
       └─ client/env_client.py  ──HTTP──▶  api/routes.py
            reset()             ◀────────    └─ environment/env.py
            step(action)                          └─ adversary.py
            state()                                    ├─ classifier.py
                                                       └─ response_generator.py
       └─ client/utils.py                    └─ rewards/rubric.py
            obs_to_prompt()                       ├─ outcome.py
            action_from_text()                    ├─ deescalation.py
                                                  ├─ trust_building.py
  └─ curriculum_scheduler.py                      ├─ demand_coverage.py
  └─ wandb (metrics)                              ├─ efficiency.py
                                                  ├─ compliance.py
EVAL (local or Colab)                             └─ anti_exploit.py
─────────────────────
eval/run_episode.py ──HTTP──▶ same api/routes.py
eval/benchmark.py   ──HTTP──▶ same api/routes.py
eval/inspect_generations.py

DEMO (HF Space or local)
────────────────────────
demo/app.py ──HTTP──▶ same api/routes.py (for live play tab)
             + loads pre-recorded JSONs from before_after.py
             + loads PNG plots from reward_plots.py
```
