"""
IncidentMind — Pre-Submission Check
Run this before submitting. All 7 checks must be green.

Usage:
    python pre_submit_check.py
    python pre_submit_check.py --url https://your-space.hf.space
"""

import argparse
import json
import os
import sys
import time

import requests
from dotenv import load_dotenv

# Load .env variables for Check #7 to pass locally
load_dotenv()

parser = argparse.ArgumentParser()
parser.add_argument("--url", default="http://localhost:7860", help="Environment base URL")
args = parser.parse_args()

BASE = args.url.rstrip("/")

PASS = "[PASS]"
FAIL = "[FAIL]"
WARN = "[WARN]"

results = []


def check(name: str, fn):
    try:
        ok, detail = fn()
        status = PASS if ok else FAIL
        print(f"  {status} {name}")
        if detail:
            print(f"       {detail}")
        results.append(ok)
        return ok
    except Exception as e:
        print(f"  {FAIL} {name}")
        print(f"       ERROR: {e}")
        results.append(False)
        return False


print(f"\n{'='*55}")
print(f"  IncidentMind — Pre-Submission Check")
print(f"  Target: {BASE}")
print(f"{'='*55}\n")

# CHECK 1 — Health endpoint
def c1():
    r = requests.get(f"{BASE}/health", timeout=10)
    ok = r.status_code == 200 and r.json().get("status") == "ok"
    return ok, f"status={r.status_code} body={r.json()}"

check("GET /health returns 200 OK", c1)

# CHECK 2 — reset() for all difficulties
def c2():
    scores = {}
    for d in ["easy", "medium", "hard"]:
        r = requests.post(f"{BASE}/reset?difficulty={d}", timeout=10)
        if r.status_code != 200:
            return False, f"reset({d}) returned {r.status_code}"
        obs = r.json()
        required = ["scenario_id", "difficulty", "alerts", "logs_available",
                    "logs_seen", "clarifications_remaining", "steps_remaining",
                    "confidence_signal", "blast_radius", "resolved"]
        missing = [f for f in required if f not in obs]
        if missing:
            return False, f"reset({d}) missing fields: {missing}"
        scores[d] = obs["scenario_id"]
    return True, f"scenarios={scores}"

check("POST /reset for easy, medium, hard — all fields present", c2)

# CHECK 3 — step() returns correct structure
def c3():
    # Reset first, then take a step with "escalate" — works for any scenario
    reset_resp = requests.post(f"{BASE}/reset?difficulty=easy", timeout=10)
    if reset_resp.status_code != 200:
        return False, f"reset() returned {reset_resp.status_code}"
    obs = reset_resp.json()
    # Use first available service for investigation test
    first_service = obs.get("logs_available", [None])[0]
    if not first_service:
        return False, "No services in logs_available"
    r = requests.post(f"{BASE}/step",
                      json={"action_type": "investigate", "target": first_service},
                      timeout=10)
    if r.status_code != 200:
        return False, f"step() returned {r.status_code}"
    result = r.json()
    required = ["observation", "reward", "done", "info"]
    missing = [f for f in required if f not in result]
    if missing:
        return False, f"step() missing fields: {missing}"
    if not isinstance(result["reward"], (int, float)):
        return False, f"reward is not a number: {result['reward']}"
    return True, f"reward={result['reward']} done={result['done']}  service={first_service}"

check("POST /step returns observation + reward + done + info", c3)

# CHECK 4 — state() endpoint
def c4():
    r = requests.get(f"{BASE}/state", timeout=10)
    ok = r.status_code == 200
    return ok, f"status={r.status_code}"

check("GET /state returns 200", c4)

# CHECK 5 — tasks endpoint
def c5():
    r = requests.get(f"{BASE}/tasks", timeout=10)
    if r.status_code != 200:
        return False, f"status={r.status_code}"
    data = r.json()
    tasks = data.get("tasks", [])
    if len(tasks) < 9:
        return False, f"Expected 9 tasks, got {len(tasks)}"
    return True, f"{len(tasks)} tasks listed"

check("GET /tasks returns 3+ tasks", c5)

# CHECK 6 — graders return strictly between 0.0 and 1.0
def c6():
    from server.graders import grade_episode
    import json
    with open("server/scenarios.json") as f:
        scenarios = json.load(f)

    all_ok = True
    detail_parts = []
    for scenario_id, scenario in scenarios.items():
        # Simulate a completed state
        mock_state = {
            "scenario_id": scenario_id,
            "difficulty": scenario["difficulty"],
            "step": 3,
            "max_steps": scenario["max_steps"],
            "logs_seen": {scenario["root_cause_service"]: ["log line"]},
            "actions_taken": [],
            "clarifications_remaining": scenario["clarification_budget"],
            "blast_radius": 0,
            "resolved": True,
            "resolution_action": scenario["correct_action"],
            "done": True,
        }
        score = grade_episode(mock_state, scenario)
        # Meta OpenEnv Requirement: STRICTLY between 0 and 1
        in_range = 0.0 < score < 1.0 
        if not in_range:
            all_ok = False
        detail_parts.append(f"{scenario['difficulty']}={score:.3f}")

    return all_ok, " | ".join(detail_parts)

check("Graders return scores strictly in (0.0, 1.0)", c6)

# CHECK 7 — environment variables set
def c7():
    missing = []
    for var in ["API_BASE_URL", "MODEL_NAME", "HF_TOKEN"]:
        if not os.environ.get(var):
            missing.append(var)
    if missing:
        return False, f"Missing env vars: {missing} — set these in HF Space secrets before submitting"
    return True, "All 3 env vars present"

check("Environment variables set (API_BASE_URL, MODEL_NAME, HF_TOKEN)", c7)

# ── Summary ──────────────────────────────────────────────────────────────────

passed = sum(results)
total = len(results)

print(f"\n{'='*55}")
print(f"  RESULT: {passed}/{total} checks passed")
print(f"{'='*55}")

if passed == total:
    print(f"\n  {PASS} All checks passed. Ready to submit.\n")
    sys.exit(0)
elif passed >= total - 1 and not results[-1]:
    print(f"\n  {WARN} {total-1}/{total} passed. Only env vars missing.")
    print(f"  Set them in HF Space secrets and you are ready to submit.\n")
    sys.exit(0)
else:
    failed = [i+1 for i, r in enumerate(results) if not r]
    print(f"\n  {FAIL} Checks {failed} failed. Fix before submitting.\n")
    sys.exit(1)
