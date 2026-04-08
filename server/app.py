"""
IncidentMind — Ambiguity-Aware Production Incident Response Environment
OpenEnv-compliant FastAPI server implementing step() / reset() / state()

Meta PyTorch OpenEnv Hackathon 2026
"""

import json
import os
import random
import uuid
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from .graders import grade_episode

# ─── Load Scenarios ───────────────────────────────────────────────────────────

SCENARIOS_PATH = os.path.join(os.path.dirname(__file__), "scenarios.json")
with open(SCENARIOS_PATH, "r") as f:
    ALL_SCENARIOS = json.load(f)

SCENARIOS_BY_DIFFICULTY = {
    "easy":   [s for s in ALL_SCENARIOS.values() if s["difficulty"] == "easy"],
    "medium": [s for s in ALL_SCENARIOS.values() if s["difficulty"] == "medium"],
    "hard":   [s for s in ALL_SCENARIOS.values() if s["difficulty"] == "hard"],
}

# ─── Pydantic Models ──────────────────────────────────────────────────────────

class Alert(BaseModel):
    service: str
    severity: str
    message: str


class Action(BaseModel):
    action_type: str          # investigate | ask_clarification | resolve | rollback | escalate
    target: Optional[str] = None
    resolution_action: Optional[str] = None


class Observation(BaseModel):
    scenario_id: str
    difficulty: str
    description: str
    alerts: List[Alert]
    logs_available: List[str]
    logs_seen: Dict[str, List[str]]
    actions_taken: List[Dict[str, Any]]
    clarifications_remaining: int
    steps_remaining: int
    confidence_signal: float
    blast_radius: int
    resolved: bool
    possible_actions: List[str] = []
    clarification_keys: List[str] = []   # valid keys for ask_clarification


class StepResult(BaseModel):
    observation: Observation
    reward: float
    done: bool
    info: Dict[str, Any]


class TaskSpec(BaseModel):
    task_id: str
    difficulty: str
    description: str
    max_steps: int
    success_threshold: float


# ─── App & State ──────────────────────────────────────────────────────────────

app = FastAPI(
    title="IncidentMind",
    description="Ambiguity-Aware Production Incident Response RL Environment",
    version="1.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global episode state
_state: Dict[str, Any] = {}
_current_scenario: Dict[str, Any] = {}


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _compute_confidence_signal(state: Dict, scenario: Dict) -> float:
    """
    Mirrors OPTI-FAB's confidence gate: rises as agent gathers useful evidence.
    0.0 = no evidence, 1.0 = root cause confirmed, no wrong actions.
    """
    logs_seen = state.get("logs_seen", {})
    services_available = scenario.get("services_available", [])
    root_cause = scenario.get("root_cause_service", "")
    blast_radius = state.get("blast_radius", 0)

    if not services_available:
        return 0.0

    # 50% weight: coverage (how many services investigated)
    coverage = len(logs_seen) / len(services_available)

    # 50% weight: whether root cause has been investigated
    root_investigated = 1.0 if root_cause in logs_seen else 0.0

    signal = (0.5 * coverage) + (0.5 * root_investigated)

    # Blast radius degrades confidence
    signal -= blast_radius * 0.15
    return round(max(0.0, min(1.0, signal)), 3)


def _build_observation(state: Dict, scenario: Dict) -> Observation:
    return Observation(
        scenario_id=state["scenario_id"],
        difficulty=state["difficulty"],
        description=scenario["description"],
        alerts=[Alert(**a) for a in scenario["alerts"]],
        logs_available=scenario["services_available"],
        logs_seen=state["logs_seen"],
        actions_taken=state["actions_taken"],
        clarifications_remaining=state["clarifications_remaining"],
        steps_remaining=max(0, state["max_steps"] - state["step"]),
        confidence_signal=_compute_confidence_signal(state, scenario),
        blast_radius=state["blast_radius"],
        resolved=state["resolved"],
        possible_actions=scenario.get("possible_actions", []),
        clarification_keys=list(scenario.get("clarification_map", {}).keys()),
    )


def _initial_state(scenario: Dict) -> Dict:
    return {
        "episode_id": str(uuid.uuid4()),
        "scenario_id": scenario["scenario_id"],
        "difficulty": scenario["difficulty"],
        "step": 0,
        "max_steps": scenario["max_steps"],
        "logs_seen": {},
        "actions_taken": [],
        "clarifications_remaining": scenario["clarification_budget"],
        "blast_radius": 0,
        "resolved": False,
        "resolution_action": None,
        "cumulative_reward": 0.0,
        "done": False,
    }


# ─── Routes ───────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
def root():
    """Landing page for the IncidentMind environment."""
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>IncidentMind | OpenEnv RL Environment</title>
        <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
        <style>
            :root {
                --bg-primary: #0a0a0f;
                --bg-secondary: #12121a;
                --bg-card: #1a1a24;
                --text-main: #f3f4f6;
                --text-muted: #9ca3af;
                --accent: #3b82f6;
                --accent-hover: #60a5fa;
                --border: #272733;
                --success: #10b981;
            }
            body {
                font-family: 'Inter', system-ui, -apple-system, sans-serif;
                background-color: var(--bg-primary);
                color: var(--text-main);
                margin: 0;
                min-height: 100vh;
                display: flex;
                flex-direction: column;
                align-items: center;
                padding: 4rem 2rem;
                line-height: 1.5;
            }
            .container {
                max-width: 800px;
                width: 100%;
            }
            header {
                text-align: center;
                margin-bottom: 3.5rem;
            }
            .icon {
                display: inline-flex;
                align-items: center;
                justify-content: center;
                width: 64px;
                height: 64px;
                background: rgba(59, 130, 246, 0.1);
                color: var(--accent);
                border-radius: 16px;
                margin-bottom: 1.5rem;
            }
            .icon svg {
                width: 32px;
                height: 32px;
                stroke-width: 2;
            }
            h1 {
                font-size: 2.5rem;
                font-weight: 700;
                margin: 0 0 0.75rem 0;
                letter-spacing: -0.02em;
            }
            .subtitle {
                font-size: 1.125rem;
                color: var(--text-muted);
                margin: 0 0 1.5rem 0;
            }
            .status-badge {
                display: inline-flex;
                align-items: center;
                gap: 0.5rem;
                background: rgba(16, 185, 129, 0.1);
                color: var(--success);
                padding: 0.5rem 1rem;
                border-radius: 9999px;
                font-weight: 500;
                font-size: 0.875rem;
                border: 1px solid rgba(16, 185, 129, 0.2);
            }
            .status-badge::before {
                content: '';
                width: 8px;
                height: 8px;
                background-color: var(--success);
                border-radius: 50%;
                box-shadow: 0 0 8px var(--success);
            }
            .grid {
                display: grid;
                grid-template-columns: repeat(3, 1fr);
                gap: 1.5rem;
                margin-bottom: 3rem;
            }
            .card {
                background-color: var(--bg-card);
                border: 1px solid var(--border);
                border-radius: 12px;
                padding: 1.5rem;
                text-align: center;
                transition: transform 0.2s, border-color 0.2s;
            }
            .card:hover {
                transform: translateY(-2px);
                border-color: var(--accent);
            }
            .card-value {
                font-size: 2rem;
                font-weight: 700;
                margin-bottom: 0.25rem;
            }
            .card-label {
                font-size: 0.875rem;
                color: var(--text-muted);
                font-weight: 500;
                text-transform: uppercase;
                letter-spacing: 0.05em;
            }
            .api-section {
                background-color: var(--bg-secondary);
                border: 1px solid var(--border);
                border-radius: 12px;
                overflow: hidden;
                margin-bottom: 3rem;
            }
            .api-header {
                padding: 1.25rem 1.5rem;
                border-bottom: 1px solid var(--border);
                font-weight: 600;
                display: flex;
                justify-content: space-between;
                align-items: center;
            }
            .endpoint {
                padding: 1rem 1.5rem;
                display: flex;
                align-items: center;
                border-bottom: 1px solid var(--border);
            }
            .endpoint:last-child {
                border-bottom: none;
            }
            .method {
                font-family: 'SFMono-Regular', Consolas, monospace;
                font-size: 0.875rem;
                font-weight: 600;
                width: 60px;
            }
            .method.get { color: #3b82f6; }
            .method.post { color: #10b981; }
            .path {
                font-family: 'SFMono-Regular', Consolas, monospace;
                font-size: 0.9rem;
                color: #e5e7eb;
                flex: 1;
            }
            .desc {
                color: var(--text-muted);
                font-size: 0.9rem;
            }
            footer {
                text-align: center;
                color: var(--text-muted);
                font-size: 0.875rem;
                border-top: 1px solid var(--border);
                padding-top: 2rem;
            }
            .btn-docs {
                display: inline-flex;
                align-items: center;
                gap: 0.5rem;
                background-color: var(--accent);
                color: white;
                text-decoration: none;
                padding: 0.5rem 1.25rem;
                border-radius: 6px;
                font-weight: 500;
                font-size: 0.875rem;
                transition: background-color 0.2s;
            }
            .btn-docs:hover {
                background-color: var(--accent-hover);
            }
        </style>
    </head>
    <body>
        <div class="container">
            <header>
                <div class="icon">
                    <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path stroke-linecap="round" stroke-linejoin="round" d="M9 12l2 2 4-4m5.618-4.016A11.955 11.955 0 0112 2.944a11.955 11.955 0 01-8.618 3.04A12.02 12.02 0 003 9c0 5.591 3.824 10.29 9 11.622 5.176-1.332 9-6.03 9-11.622 0-1.042-.133-2.052-.382-3.016z" />
                    </svg>
                </div>
                <h1>IncidentMind</h1>
                <p class="subtitle">Ambiguity-Aware Production Incident Response Environment</p>
                <div class="status-badge">System Operational — v1.1.0</div>
            </header>

            <div class="grid">
                <div class="card">
                    <div class="card-value">9</div>
                    <div class="card-label">Scenarios</div>
                </div>
                <div class="card">
                    <div class="card-value">3</div>
                    <div class="card-label">Difficulties</div>
                </div>
                <div class="card">
                    <div class="card-value" style="display:flex;align-items:baseline;justify-content:center;gap:2px;">
                        <span>0.0</span><span style="color:var(--text-muted);font-weight:400;font-size:1.5rem">/</span><span>1.0</span>
                    </div>
                    <div class="card-label">Score Range</div>
                </div>
            </div>

            <div class="api-section">
                <div class="api-header">
                    <span>API Reference</span>
                    <a href="/docs" class="btn-docs">Open Interactive Docs</a>
                </div>
                <div class="endpoint">
                    <span class="method get">GET</span>
                    <span class="path">/health</span>
                    <span class="desc">System health check</span>
                </div>
                <div class="endpoint">
                    <span class="method post">POST</span>
                    <span class="path">/reset?difficulty={level}</span>
                    <span class="desc">Initialize scenario</span>
                </div>
                <div class="endpoint">
                    <span class="method post">POST</span>
                    <span class="path">/step</span>
                    <span class="desc">Execute single action</span>
                </div>
                <div class="endpoint">
                    <span class="method get">GET</span>
                    <span class="path">/state</span>
                    <span class="desc">Retrieve complete state</span>
                </div>
                <div class="endpoint">
                    <span class="method get">GET</span>
                    <span class="path">/tasks</span>
                    <span class="desc">List valid scenarios</span>
                </div>
            </div>

            <footer>
                Meta PyTorch OpenEnv Hackathon 2026
            </footer>
        </div>
    </body>
    </html>
    """


@app.get("/health")
def health():
    return {"status": "ok", "environment": "incidentmind", "version": "1.1.0"}


@app.get("/tasks")
def list_tasks():
    """Enumerate all tasks with metadata for the OpenEnv validator."""
    tasks = []
    for scenario_id, scenario in ALL_SCENARIOS.items():
        tasks.append(
            TaskSpec(
                task_id=scenario["scenario_id"],
                difficulty=scenario["difficulty"],
                description=scenario["description"],
                max_steps=scenario["max_steps"],
                success_threshold=scenario["success_threshold"],
            )
        )
    return {"tasks": tasks}


@app.post("/reset", response_model=Observation)
def reset(difficulty: str = Query("easy", pattern="^(easy|medium|hard)$")):
    """
    Start a fresh episode for the given difficulty.
    Returns the initial observation.
    """
    global _state, _current_scenario

    scenarios = SCENARIOS_BY_DIFFICULTY.get(difficulty, [])
    if not scenarios:
        raise HTTPException(status_code=400, detail=f"No scenarios for difficulty: {difficulty}")

    _current_scenario = random.choice(scenarios)
    _state = _initial_state(_current_scenario)

    return _build_observation(_state, _current_scenario)


@app.post("/step", response_model=StepResult)
def step(action: Action):
    """
    Agent takes one action. Returns observation + reward + done + info.
    """
    global _state, _current_scenario

    if not _state:
        raise HTTPException(status_code=400, detail="Episode not started. Call /reset first.")
    if _state["done"]:
        raise HTTPException(status_code=400, detail="Episode already done. Call /reset to start a new one.")

    reward = 0.0
    info: Dict[str, Any] = {"action": action.model_dump()}
    done = False

    # Increment step and apply step penalty
    _state["step"] += 1
    reward -= 0.2

    action_type = action.action_type

    # ── INVESTIGATE ──────────────────────────────────────────────────────────
    if action_type == "investigate":
        target = action.target
        if not target:
            reward -= 0.5
            info["error"] = "investigate requires a target service"
        elif target not in _current_scenario["services_available"]:
            reward -= 0.5
            info["error"] = f"Service '{target}' does not exist"
        elif target in _state["logs_seen"]:
            reward -= 0.3
            info["result"] = f"Already investigated {target} — redundant investigation"
        else:
            logs = _current_scenario["logs"].get(target, ["[INFO] No logs available for this service."])
            _state["logs_seen"][target] = logs
            info["result"] = f"Fetched {len(logs)} log lines from {target}"
            info["logs"] = logs

    # ── ASK CLARIFICATION ────────────────────────────────────────────────────
    elif action_type == "ask_clarification":
        if _state["clarifications_remaining"] <= 0:
            reward -= 2.0
            info["error"] = "Clarification budget exhausted"
        else:
            question_key = action.target
            answer = _current_scenario["clarification_map"].get(
                question_key,
                "No information available for that question."
            )
            _state["clarifications_remaining"] -= 1
            reward -= 0.5
            info["result"] = answer
            info["clarifications_remaining"] = _state["clarifications_remaining"]

    # ── RESOLVE ──────────────────────────────────────────────────────────────
    elif action_type == "resolve":
        resolution = action.resolution_action
        if not resolution:
            reward -= 0.5
            info["error"] = "resolve requires a resolution_action"
        else:
            _state["resolution_action"] = resolution
            _state["resolved"] = True
            done = True

            if resolution == _current_scenario["correct_action"]:
                # Correct fix
                steps_remaining = max(0, _state["max_steps"] - _state["step"])
                efficiency_bonus = round((steps_remaining / _state["max_steps"]) * 3.5, 2)
                reward += 10.0 + efficiency_bonus
                info["result"] = "CORRECT — incident resolved successfully"
                info["efficiency_bonus"] = efficiency_bonus
            else:
                # Wrong fix — blast radius
                reward -= 3.0
                _state["blast_radius"] += 1
                reward -= 2.0 * _state["blast_radius"]
                info["result"] = "WRONG FIX — incorrect root cause targeted"
                info["blast_radius"] = _state["blast_radius"]

    # ── ROLLBACK ────────────────────────────────────────────────────────────
    elif action_type == "rollback":
        reward -= 1.0
        info["result"] = f"Rolled back service: {action.target}"

    # ── ESCALATE ────────────────────────────────────────────────────────────
    elif action_type == "escalate":
        reward -= 2.0
        done = True
        info["result"] = "Escalated to human on-call — episode ended"

    else:
        reward -= 0.5
        info["error"] = f"Unknown action_type: {action_type}"

    # ── TIMEOUT CHECK ────────────────────────────────────────────────────────
    if _state["step"] >= _state["max_steps"] and not done:
        reward -= 5.0
        done = True
        info["timeout"] = True
        info["result"] = "Episode timed out — all steps exhausted"

    # ── RECORD & FINALIZE ────────────────────────────────────────────────────
    _state["actions_taken"].append({
        "step": _state["step"],
        "action_type": action_type,
        "target": action.target,
        "resolution_action": action.resolution_action,
        "reward": round(reward, 3),
    })
    _state["cumulative_reward"] = round(_state["cumulative_reward"] + reward, 3)
    _state["done"] = done

    # Compute final grade if episode is done
    if done:
        final_score = grade_episode(_state, _current_scenario)
        info["final_score"] = final_score
        info["cumulative_reward"] = _state["cumulative_reward"]

    observation = _build_observation(_state, _current_scenario)

    return StepResult(
        observation=observation,
        reward=round(reward, 3),
        done=done,
        info=info,
    )


@app.get("/state")
def get_state():
    """Returns full current episode state as JSON."""
    if not _state:
        return {"status": "no_active_episode"}
    return {
        **_state,
        "confidence_signal": _compute_confidence_signal(_state, _current_scenario),
        "scenario_id": _current_scenario.get("scenario_id"),
        "root_cause": _current_scenario.get("root_cause_service"),
        "possible_actions": _current_scenario.get("possible_actions", []),
    }


def main():
    """Entry point for openenv multi-mode deployment."""
    import uvicorn
    uvicorn.run(
        "server.app:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", 7860)),
        reload=False,
    )


if __name__ == "__main__":
    main()
