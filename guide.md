# Local Server Testing Guide

This guide walks you through standing up the **MetaX NegotiationEnv** FastAPI
server on your own machine, exercising the `/health`, `/reset`, and `/step`
endpoints by hand, and finally running the automated verification suite end to
end.

It assumes a working POSIX shell (Linux / macOS / WSL2) and Python 3.10+.

---

## 0. One-time prerequisites

```bash
git clone <this-repo> MetaXOpenEnv
cd MetaXOpenEnv
python3 -m venv .venv
source .venv/bin/activate
pip install -r negotiation-env/requirements.txt
```

> **Note.** `requirements.txt` and `requirements-training.txt` currently pin
> the *same* set of API/runtime dependencies. Heavy ML libraries
> (`trl`, `unsloth`, `bitsandbytes`, `wandb`, `datasets`) are **not** in either
> file — they are installed separately on the GPU box (see the audit report's
> "Training Pipeline Readiness" section). For just running the FastAPI server
> + LLM voice, the lightweight `requirements.txt` is enough.

---

## 1. Configure secrets (`.env`)

The project uses **`python-dotenv`** to load credentials at process start.
The real secrets file (`.env`) is git-ignored; `.env.example` is the tracked
template.

```bash
cp .env.example .env
$EDITOR .env
```

Set the two variables that control LLM voice generation:

```env
# .env  (lives at the repo root, NOT inside negotiation-env/)
NVIDIA_API_KEY=nvapi-XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
NEG_LLM_ENABLED=1
```

Validation rules (enforced by `environment/response_generator.py`):

| Setting | Behaviour |
|---|---|
| `NEG_LLM_ENABLED` unset / `0` | Hermetic mode — borrower replies come from the deterministic template bank. `NVIDIA_API_KEY` is **not** required. |
| `NEG_LLM_ENABLED=1` + valid `NVIDIA_API_KEY` | Live NIM mode — Llama-3.1-70B-Instruct generates borrower replies, with retry/fallback to templates on transient errors. |
| `NEG_LLM_ENABLED=1` + missing/empty/`your_actual_key_here` | `ResponseGenerator.__init__` raises `ValueError` with a clear remediation message. **The server will refuse to start.** |

> Variables already exported in your shell take precedence over `.env`
> (dotenv only fills in *unset* variables). If you want to test the
> `ValueError` path while a real key is still in your shell, run
> `unset NVIDIA_API_KEY` first.

---

## 2. Start the local FastAPI server

From the repo root:

```bash
cd negotiation-env
uvicorn api.app:app --reload --port 8000
```

Or, equivalent on a single line:

```bash
PYTHONPATH=. uvicorn api.app:app --host 127.0.0.1 --port 8000 --reload
```

For container / HF Space parity (binds all interfaces):

```bash
uvicorn api.app:app --host 0.0.0.0 --port 8000
```

You should see:

```
INFO:     Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
INFO:     Application startup complete.
```

> **Module path.** The ASGI app object lives at `api.app:app` — that's
> `negotiation-env/api/app.py`'s top-level `app = FastAPI(...)`. Don't confuse
> it with `api/routes.py` (which only defines `router`, mounted by `app.py`).

---

## 3. Manual probes

### 3a. `curl` — three-step happy path

```bash
# 1. Liveness probe
curl -s http://localhost:8000/health
# → {"status":"ok"}

# 2. Start an episode (curriculum_stage=1 = "easy" profiles P01..P05)
curl -s -X POST http://localhost:8000/reset \
     -H "Content-Type: application/json" \
     -d '{"curriculum_stage": 1}' | tee /tmp/reset.json
# → {"observation": {...}, "episode_id": "ep-...", "profile_id": "P03",
#    "curriculum_stage": 1}

# 3. Take one step. We need the episode_id from the previous response.
EPISODE_ID=$(python3 -c "import json,sys; print(json.load(open('/tmp/reset.json'))['episode_id'])")
curl -s -X POST http://localhost:8000/step \
     -H "Content-Type: application/json" \
     -d "{
       \"episode_id\": \"$EPISODE_ID\",
       \"action_type\": \"acknowledge_hardship\",
       \"text\": \"I really hear how stressful this has been for you. Let's work together to find an EMI plan that you can actually manage.\",
       \"metadata\": {}
     }"
# → {"observation": {...}, "reward": 0.42, "done": false,
#    "info": {"reward_breakdown": {...}, "termination_reason": null,
#             "anger_after": 6.2, "trust_after": 3.4, "episode_id": "ep-..."}}
```

### 3b. Python `requests` snippet — same flow, programmatic

Save as `manual_probe.py` and run with `python manual_probe.py`:

```python
import json
import requests

BASE = "http://localhost:8000"

print("/health         →", requests.get(f"{BASE}/health").json())

reset = requests.post(
    f"{BASE}/reset", json={"curriculum_stage": 1}
).json()
print("/reset          → episode_id =", reset["episode_id"])
print("                  profile_id =", reset["profile_id"])
print("                  borrower   =", reset["observation"]["borrower_msg"][:80], "…")

step = requests.post(
    f"{BASE}/step",
    json={
        "episode_id":  reset["episode_id"],
        "action_type": "acknowledge_hardship",
        "text":        "I really hear how stressful this has been. Let's find an EMI plan that fits your situation.",
        "metadata":    {},
    },
).json()
print("/step           → reward     =", step["reward"])
print("                  done       =", step["done"])
print("                  borrower   =", step["observation"]["borrower_msg"][:80], "…")
print("reward_breakdown:", json.dumps(step["info"]["reward_breakdown"], indent=2))
```

### 3c. LLM-on vs LLM-off — a concrete comparison

Same agent message, two different `.env` configurations, two very different
borrower replies:

| Mode | `.env` | Sample `borrower_msg` |
|---|---|---|
| **LLM-off** (template) | `NEG_LLM_ENABLED=` (unset) | `"I appreciate that you understand. Maybe we can talk about a plan."` |
| **LLM-on** (NIM/Llama-3.1-70B) | `NEG_LLM_ENABLED=1` + key | `"Honestly that does mean a lot — most agents just keep pushing. If you can actually drop the EMI to something I can carry month to month, I'm willing to commit on paper."` |

The deterministic state machine (anger / trust / fear deltas, reward
breakdown, termination logic) is **identical** in both modes — only the
natural-language surface differs. This is the core "Hybrid-Dynamic"
guarantee.

---

## 4. Programmatic verification

### 4a. Five-episode end-to-end test (`local_test.py`)

This script auto-detects whether `localhost:8000` is up. If yes it uses the
real `NegotiationEnvClient`; if no it falls back to `DummyEnvClient`.

```bash
cd negotiation-env
python local_test.py
```

It prints, for every step: action sent → borrower reply → state delta
(anger/trust) → full `reward_breakdown` dict, and at the end runs three
red-team probes:

* **Compliance:** verifies a one-word `"ok"` message returns
  `compliance == -0.2`.
* **Anti-exploit:** verifies the third identical message yields
  `anti_exploit < 0`.
* **Asymmetry:** verifies the deescalation penalty for a 1.0 anger *rise* is
  `1.5×` larger than the reward for a 1.0 anger *drop*.

A successful run terminates with `ALL CHECKS PASSED`.

### 4b. Full preflight suite (`scripts/preflight_check.py`)

This is the gate you run **before pushing to a Hugging Face Space**. It is
deliberately strict — any failing stage exits non-zero.

```bash
cd negotiation-env
python scripts/preflight_check.py
```

| Stage | What it gates |
|-------|---------------|
| **Pytest suite** | All unit tests under `tests/` — including `test_dynamic_voice.py` (LLM voice) and `test_edge_cases.py` (network/data/boundary guards). |
| **Gate 1** (`scripts/gate1_test.py`) | Deterministic state machine — `adversary.py` reacts to canned signals correctly. |
| **Gate 2** (`scripts/gate2_test.py`) | Reward module — all 7 components fire on the right transitions. |
| **Gate 3** (`scripts/gate3_test.py`) | Curriculum loader — every profile in `profiles_*.yaml` parses. |
| **Gate 4** (`scripts/gate4_test.py`) | Prompt parser — `action_from_text()` round-trips XML correctly, fallback path triggers on malformed input. |
| **Local env integration** (`scripts/test_env_local.py`) | `NegotiationEnv.reset/step` — interface lock holds, observation schema unchanged. |
| **API smoke** (in-process `TestClient`) | FastAPI app responds 200 to `/health`, `/reset`, and a single `/step`. Does **not** require a running uvicorn — uses `fastapi.testclient.TestClient` against the imported `app` object. |

A green preflight ends with:

```
[SUCCESS] All preflight checks passed. Safe to deploy.
```

---

## 5. Common failure modes

| Symptom | Likely cause | Fix |
|---------|--------------|-----|
| `ValueError: NVIDIA_API_KEY is required when NEG_LLM_ENABLED=1` at startup | Placeholder or empty key in `.env` while LLM mode is on | Either fill in a real `nvapi-…` key, or set `NEG_LLM_ENABLED=` (empty / unset). |
| HTTP `401 Unauthorized` from NIM mid-episode (silently retried, then template fallback) | Key has expired or hit quota | Rotate the key on `build.nvidia.com`, redeploy. The training loop will not crash — it falls back to templates. |
| HTTP `400` on `/step`: *"Call /reset before /step."* | The server has no active episode (cold start, or server restarted between requests) | Call `/reset` first; capture `episode_id`. |
| HTTP `409` on `/step`: *"episode_id does not match active episode."* | Two clients racing on the same server, or you cached a stale `episode_id` | Re-`/reset` and use the freshly returned `episode_id`. |
| `OSError: [Errno 98] Address already in use` on `uvicorn` start | Port `8000` already taken (perhaps by a previous `uvicorn` you forgot to kill) | `lsof -i :8000` then `kill <pid>`, or pass `--port 8001`. |
| `ModuleNotFoundError: No module named 'environment'` when running `uvicorn` | Started from the wrong directory | `cd negotiation-env` first; the imports are relative to that root. |
| Tests pass but `/step` returns the **same** template borrower reply every time | LLM is silently disabled (placeholder key, `NEG_LLM_ENABLED=` unset, or transient NIM error) | `curl http://localhost:8000/state` and inspect; check server logs for retry messages; verify `.env` was loaded (placeholders are treated as "no key"). |
| Live NIM tests in `tests/test_dynamic_voice.py` skipped | `_read_nvidia_api_key()` returned `None` (placeholder or unset) | Set a real key in `.env`; rerun. Hermetic tests still run regardless. |

---

## 6. Quick reference

```bash
# 1. Configure
cp .env.example .env && $EDITOR .env

# 2. Start server
cd negotiation-env && uvicorn api.app:app --reload --port 8000

# 3. Probe (separate shell)
curl -s http://localhost:8000/health

# 4. Verify end-to-end
python local_test.py
python scripts/preflight_check.py
```

If preflight is green and the manual probes return sensible LLM borrower
replies, you are cleared to push to the Hugging Face Space.
