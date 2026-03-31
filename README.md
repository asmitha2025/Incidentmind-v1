---
title: IncidentMind
emoji: 🚨
colorFrom: red
colorTo: blue
sdk: docker
pinned: false
tags:
  - openenv
  - reinforcement-learning
  - incident-response
---

# IncidentMind
## Ambiguity-Aware Production Incident Response RL Environment
**Meta PyTorch OpenEnv Hackathon 2026 | Hariharan M & Asmitha M**

> *"Production AI systems don't fail because they lack intelligence. They fail because they act under uncertainty without knowing what they don't know."*

---

## What Is This?

IncidentMind is an [OpenEnv](https://huggingface.co/openenv)-compliant Reinforcement Learning environment that trains AI agents to diagnose and resolve production software incidents under real-world conditions — noisy logs, red herrings, cascading failures, and time pressure.

The core insight: frontier models hit **82–97% false positive rates** in production incident response (OpenSec, 2026). Not because they lack intelligence — but because they act without gathering enough evidence first. IncidentMind trains agents to stop guessing and start investigating.

---

## The OPTI-FAB Origin

This environment is built on the confidence-gating architecture of **OPTI-FAB** — our semiconductor wafer defect inspection system that reduced classification latency 63.9% by deciding *when partial data is sufficient to act*.

| OPTI-FAB Concept | IncidentMind Equivalent |
|-----------------|------------------------|
| Confidence gate ≥ 0.85 | Agent gathers evidence before resolving |
| Entropy threshold ≤ 0.35 | Clarification budget — ask before acting |
| Early exit at 50–68% frame | Efficiency bonus for early correct resolution |
| False reject = wasted cost | Wrong fix = blast radius cascade |
| Circular buffer streaming | Agent sees partial logs, investigates incrementally |

Same architecture. Same math. Different domain.

---

## The Nine Tasks

Each difficulty level has 3 scenarios — agents are randomly assigned one per episode, forcing generalization across incident patterns.

| Task | Difficulty | Scenario | Red Herrings | Max Steps | Pass Threshold |
|------|-----------|----------|-------------|-----------|---------------|
| easy_001 | Easy | Postgres connection pool exhausted | 1 (Redis) | 8 | 0.70 |
| easy_002 | Easy | Worker OOM kill — memory leak in order processor | 1 (Message queue) | 8 | 0.70 |
| easy_003 | Easy | Disk full — WAL archiving failure on postgres | 1 (Log aggregator) | 8 | 0.70 |
| medium_001 | Medium | ML inference GPU saturation — cascading timeouts | 1 (Product catalog) | 12 | 0.60 |
| medium_002 | Medium | DNS misconfiguration — intermittent resolution failures | 1 (Health checker) | 12 | 0.60 |
| medium_003 | Medium | TLS certificate expiry on internal billing API | 1 (Cert monitor) | 12 | 0.60 |
| hard_001 | Hard | Feature flag activates model with missing embeddings | 3 (Payment GW, Session Store, Auth) | 15 | 0.50 |
| hard_002 | Hard | Certificate rotation breaking mutual TLS across mesh | 3 (API GW, DB Proxy, Monitoring) | 15 | 0.50 |
| hard_003 | Hard | Schema migration race condition — silent data corruption | 3 (Inventory, Reconciliation, Pricing) | 15 | 0.50 |

### Baseline vs Trained Agent Scores

| Task | Baseline Agent (no training) | Trained Agent (target) | Delta |
|------|------------------------------|----------------------|-------|
| Easy | ~0.80 | 0.90+ | +0.10 |
| Medium | ~0.65 | 0.80+ | +0.15 |
| Hard | **~0.30** | **0.60+** | **+0.30** |

The hard tasks are specifically designed so untrained frontier models score 0.10–0.35 by acting on the most visible alert. A trained agent learns to ignore red herrings, trace the dependency chain, and apply the correct fix. With 3 scenarios per difficulty, agents must **generalize** — they can't memorize a single incident pattern. **That diversity is the value of this environment.**

---

## OpenEnv API

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check — returns `{"status": "ok"}` |
| `/reset?difficulty={easy\|medium\|hard}` | POST | Start fresh episode |
| `/step` | POST | Take one action |
| `/state` | GET | Full current state as JSON |
| `/tasks` | GET | Enumerate all 9 tasks with metadata |

### Example Episode

```python
import requests

BASE = "https://your-space.hf.space"

# Start episode
obs = requests.post(f"{BASE}/reset?difficulty=hard").json()

# Take action
result = requests.post(f"{BASE}/step", json={
    "action_type": "investigate",
    "target": "fraud-detection"
}).json()

print(result["reward"])          # -0.2 (step penalty)
print(result["observation"]["confidence_signal"])  # rises as you find evidence
print(result["done"])            # False — keep going
```

### Action Space

```json
{"action_type": "investigate",       "target": "service-name"}
{"action_type": "ask_clarification", "target": "question_key"}
{"action_type": "resolve",           "resolution_action": "your-fix"}
{"action_type": "escalate"}
```

---

## Reward Function

| Signal | Value | What It Teaches |
|--------|-------|----------------|
| Correct resolution | +10.0 | The primary objective |
| Efficiency bonus | +0.5 to +3.5 | Resolve faster after enough evidence |
| Step penalty | -0.2 / step | Be efficient — don't over-investigate |
| Wrong resolution | -3.0 | Investigate before acting |
| Blast radius | -2.0 / service | Wrong actions cascade |
| Redundant investigation | -0.3 | Don't check the same service twice |
| Clarification exhausted | -2.0 | Use budget wisely |
| Timeout | -5.0 | Biggest penalty — running out of time |

The `confidence_signal` field (0.0–1.0) in every observation is the direct RL analogue of OPTI-FAB's confidence gate — rising as the agent builds evidence, falling when wrong actions increase blast radius.

---

## Running Locally

```bash
# Clone and install
git clone https://github.com/YOUR_USERNAME/incidentmind
cd incidentmind
pip install -r requirements.txt

# Set environment variables
cp .env.example .env
# Edit .env with your HF token

# Start environment
uvicorn incident_env:app --host 0.0.0.0 --port 7860

# Run baseline agent (in another terminal)
python inference.py

# Run pre-submission check
python pre_submit_check.py
```

### With Docker

```bash
docker build -t incidentmind .
docker run -p 7860:7860 \
  -e API_BASE_URL=https://router.huggingface.co/v1 \
  -e MODEL_NAME=meta-llama/Llama-3.3-70B-Instruct \
  -e HF_TOKEN=your_token \
  incidentmind
```

---

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `API_BASE_URL` | Yes | `https://router.huggingface.co/v1` |
| `MODEL_NAME` | Yes | `meta-llama/Llama-3.3-70B-Instruct` |
| `HF_TOKEN` | Yes | Your HuggingFace token |
| `ENV_URL` | No | Environment URL (default: `http://localhost:7860`) |

---

## Project Structure

```
incidentmind/
├── incident_env.py      # Core FastAPI environment — step/reset/state
├── graders.py           # Deterministic graders — pure Python state checks
├── scenarios.json       # 9 incident scenarios (3 easy / 3 medium / 3 hard)
├── inference.py         # Baseline agent — OpenAI client + timeout guard
├── openenv.yaml         # OpenEnv spec metadata
├── Dockerfile           # Container for HF Spaces
├── requirements.txt     # Python dependencies
├── pre_submit_check.py  # Pre-submission validation
└── .env.example         # Environment variable template
```

---

## Key Research Context

| Stat | Value | Source |
|------|-------|--------|
| GPT-5 false positive rate | 82.5% | OpenSec paper, 2026 |
| Agent success drop with ambiguity | 90% → 40% | OpenSec production study |
| Gaia2 lowest-scoring capability | Ambiguity Resolution | Gaia2 benchmark, Meta 2025 |
| OPTI-FAB latency reduction | 63.9% | Our benchmark, RTX 4050 |

---

*Hariharan M & Asmitha M | IncidentMind | Meta PyTorch OpenEnv Hackathon 2026*
