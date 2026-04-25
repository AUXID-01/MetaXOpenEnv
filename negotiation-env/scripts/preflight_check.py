"""Run pre-deployment quality gates before pushing to Space.

This script is intentionally strict: any failing check exits non-zero.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent


def run_step(name: str, command: list[str]) -> None:
    print(f"\n[STEP] {name}")
    print(f"[CMD ] {' '.join(command)}")
    completed = subprocess.run(command, cwd=ROOT, check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"{name} failed with exit code {completed.returncode}")
    print(f"[OK  ] {name}")


def run_api_smoke() -> None:
    print("\n[STEP] API smoke check")
    python_code = (
        "from fastapi.testclient import TestClient; "
        "from api.app import app; "
        "c=TestClient(app); "
        "r=c.get('/health'); "
        "assert r.status_code==200 and r.json().get('status')=='ok'; "
        "rr=c.post('/reset', json={'curriculum_stage':1}); "
        "assert rr.status_code==200; "
        "data=rr.json(); "
        "rs=c.post('/step', json={"
        "'episode_id':data['episode_id'],"
        "'action_type':'send_message',"
        "'text':'I understand your situation and we can work on a plan together.',"
        "'metadata':{}"
        "}); "
        "assert rs.status_code==200; "
        "print('api-smoke-ok')"
    )
    completed = subprocess.run([sys.executable, "-c", python_code], cwd=ROOT, check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"API smoke check failed with exit code {completed.returncode}")
    print("[OK  ] API smoke check")


def main() -> int:
    print("=== Preflight Deployment Checks ===")
    print(f"Repo root: {ROOT}")

    try:
        run_step("Pytest suite", [sys.executable, "-m", "pytest"])
        run_step("Gate 1", [sys.executable, "scripts/gate1_test.py"])
        run_step("Gate 2", [sys.executable, "scripts/gate2_test.py"])
        run_step("Gate 3", [sys.executable, "scripts/gate3_test.py"])
        run_step("Gate 4", [sys.executable, "scripts/gate4_test.py"])
        run_step("Local environment integration", [sys.executable, "scripts/test_env_local.py"])
        run_api_smoke()
    except RuntimeError as exc:
        print(f"\n[FAIL] {exc}")
        print("Preflight failed. Do not deploy yet.")
        return 1

    print("\n[SUCCESS] All preflight checks passed. Safe to deploy.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
