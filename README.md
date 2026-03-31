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
  - ambiguity-resolution
  - site-reliability
---

# 🚨 IncidentMind

## Ambiguity-Aware Production Incident Response RL Environment

**Meta PyTorch OpenEnv Hackathon 2026 | Hariharan M & Asmitha M**

[![HuggingFace Space](https://img.shields.io/badge/🤗_HuggingFace-Space-blue)](https://huggingface.co/spaces/hariharan1828/incidentmind)
[![OpenEnv](https://img.shields.io/badge/OpenEnv-v1.1.0-green)]()
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED)]()
[![Tasks](https://img.shields.io/badge/Tasks-9_Scenarios-orange)]()

> *"Production AI systems don't fail because they lack intelligence. They fail because they act under uncertainty without knowing what they don't know."*

---

## 🎯 What Is This?

IncidentMind is an [OpenEnv](https://huggingface.co/openenv)-compliant Reinforcement Learning environment that trains AI agents to **diagnose and resolve production software incidents** under real-world conditions — noisy logs, red herrings, cascading failures, and time pressure.

### The Problem

Frontier LLMs hit **82–97% false positive rates** in production incident response (OpenSec, 2026). Not because they lack intelligence — but because they **act without gathering enough evidence first**. When faced with 4 simultaneous alerts, an untrained agent jumps to the most visible one and applies the wrong fix, causing cascading blast radius damage.

### The Solution

IncidentMind creates a controlled simulation where agents learn to:

1. **Investigate before acting** — check service logs to build evidence
2. **Ignore red herrings** — noisy alerts that aren't the root cause
3. **Trace dependency chains** — follow causality, not symptoms
4. **Use clarifications wisely** — ask targeted questions within a budget
5. **Act decisively** — apply the correct fix once confidence is high

---

## 🧠 The OPTI-FAB Origin

This environment is built on the confidence-gating architecture of **OPTI-FAB** (Optimized Process Technology for Intelligent Fabrication and Anomaly-based Breakdowns) — our semiconductor wafer defect inspection system that reduced classification latency **63.9%** by deciding *when partial data is sufficient to act*.

| OPTI-FAB Concept | IncidentMind Equivalent |
|:-----------------|:-----------------------|
| Confidence gate ≥ 0.85 | `confidence_signal` rises as agent gathers evidence |
| Entropy threshold ≤ 0.35 | `clarification_budget` — ask before acting |
| Early exit at 50–68% frame | Efficiency bonus for early correct resolution |
| False reject = wasted cost | Wrong fix = blast radius cascade penalty |
| Circular buffer streaming | Agent sees partial logs, investigates incrementally |

**Same architecture. Same math. Different domain.**

---

## 📊 The Nine Tasks

Each difficulty level has **3 distinct scenarios** — agents are randomly assigned one per episode, forcing generalization across incident patterns. This prevents memorization and rewards genuine diagnostic skill.

### Easy (Ambiguity Level 0 — Single root cause, 1 red herring)

| Task ID | Scenario | Root Cause | Correct Action | Red Herring | Max Steps |
|:--------|:---------|:-----------|:---------------|:------------|:---------:|
| `easy_001` | Postgres connection pool exhausted | postgres-primary | `increase_connections` | redis-cache | 8 |
| `easy_002` | Worker OOM kill — memory leak in order processor | order-processor | `restart_order_processor_with_memory_limit` | message-queue | 8 |
| `easy_003` | Disk full — WAL archiving failure on postgres | postgres-wal | `clear_wal_archive_and_resize_disk` | log-aggregator | 8 |

### Medium (Ambiguity Level 1 — Hidden root cause, 1 red herring)

| Task ID | Scenario | Root Cause | Correct Action | Red Herring | Max Steps |
|:--------|:---------|:-----------|:---------------|:------------|:---------:|
| `medium_001` | ML inference GPU saturation — cascading timeouts | gpu-scheduler | `redistribute_gpu_workload` | product-catalog | 12 |
| `medium_002` | DNS misconfiguration — intermittent resolution failures | dns-resolver | `rollback_dns_config` | health-checker | 12 |
| `medium_003` | TLS certificate expiry on internal billing API | certificate-authority | `renew_billing_api_certificate` | cert-monitor | 12 |

### Hard (Ambiguity Level 2 — Multi-alert chaos, 3 red herrings)

| Task ID | Scenario | Root Cause | Correct Action | Red Herrings | Max Steps |
|:--------|:---------|:-----------|:---------------|:-------------|:---------:|
| `hard_001` | Feature flag activates ML model with missing embeddings | fraud-detection | `disable_feature_flag_fraud_model_v3` | payment-gateway, session-store, auth-service | 15 |
| `hard_002` | Certificate rotation breaking mutual TLS across service mesh | certificate-manager | `rollback_certificate_rotation` | api-gateway, database-proxy, monitoring-agent | 15 |
| `hard_003` | Schema migration race condition — silent data corruption | schema-migrator | `rollback_migration_v48` | inventory-service, reconciliation-job, pricing-engine | 15 |

---

## 📈 Baseline Results

Scores from the baseline agent (Llama-3.3-70B-Instruct, untrained):

| Difficulty | Baseline Score | Pass Threshold | Status | Analysis |
|:-----------|:--------------|:---------------|:------:|:---------|
| **Easy** | 0.00 | 0.70 | ❌ | Agent picks plausible-sounding but wrong action from distractors |
| **Medium** | 0.838 | 0.60 | ✅ | Agent investigates correctly, finds root cause, applies exact fix |
| **Hard** | 0.83 | 0.50 | ✅ | Agent traces dependency chain past 3 red herrings to real root cause |

### Why Easy Fails But Hard Succeeds

This counterintuitive result **validates the environment design**:

- **Easy scenarios** have simple symptoms but the correct action string requires specific domain knowledge (`restart_order_processor_with_memory_limit` vs generic `increase_memory`). The LLM picks the wrong distractor.
- **Hard scenarios** have complex multi-alert chaos, but the logs contain explicit causal chains. The LLM's reasoning ability shines when there's more evidence to process.

**This proves the environment rewards investigation depth over surface-level pattern matching** — exactly the behavior RL training should reinforce.

---

## 🔌 OpenEnv API

| Endpoint | Method | Description |
|:---------|:-------|:------------|
| `/` | GET | Styled landing page with environment overview |
| `/health` | GET | Health check — returns `{"status": "ok"}` |
| `/reset?difficulty={easy\|medium\|hard}` | POST | Start fresh episode with random scenario |
| `/step` | POST | Take one action, receive observation + reward |
| `/state` | GET | Full current state as JSON |
| `/tasks` | GET | Enumerate all 9 tasks with metadata |
| `/docs` | GET | Interactive Swagger API documentation |

### Example Episode

```python
import requests

BASE = "https://hariharan1828-incidentmind.hf.space"

# Start a hard episode
obs = requests.post(f"{BASE}/reset?difficulty=hard").json()
print(obs["description"])          # "Multiple services reporting failures..."
print(obs["possible_actions"])     # ["disable_feature_flag_fraud_model_v3", "restart_service", ...]
print(obs["logs_available"])       # ["fraud-detection", "payment-gateway", "session-store", ...]

# Investigate the root cause service
result = requests.post(f"{BASE}/step", json={
    "action_type": "investigate",
    "target": "fraud-detection"
}).json()

print(result["reward"])                              # -0.2 (step penalty)
print(result["observation"]["confidence_signal"])    # 0.35 → rising
print(result["observation"]["logs_seen"])            # Shows fraud-detection logs
print(result["done"])                                # False — keep investigating

# Ask a clarification question
result = requests.post(f"{BASE}/step", json={
    "action_type": "ask_clarification",
    "target": "what_changed_31_minutes_ago"
}).json()
print(result["observation"]["confidence_signal"])    # 0.55 → higher

# Apply the fix once confident
result = requests.post(f"{BASE}/step", json={
    "action_type": "resolve",
    "resolution_action": "disable_feature_flag_fraud_model_v3"
}).json()
print(result["reward"])    # +10.0 (correct!) + efficiency bonus
print(result["done"])      # True — episode complete
```

### Action Space

```json
{"action_type": "investigate",       "target": "service-name"}
{"action_type": "ask_clarification", "target": "question_key"}
{"action_type": "resolve",           "resolution_action": "exact_action_string"}
{"action_type": "escalate"}
```

### Observation Space

Every observation includes:

| Field | Type | Description |
|:------|:-----|:------------|
| `scenario_id` | string | Current scenario identifier |
| `difficulty` | string | `easy`, `medium`, or `hard` |
| `description` | string | Incident description with initial alerts |
| `alerts` | list | Active alerts with `service`, `severity`, `message` |
| `logs_available` | list | Services whose logs can be investigated |
| `logs_seen` | dict | Logs already retrieved per service |
| `actions_taken` | list | History of actions taken this episode |
| `possible_actions` | list | Valid resolution action strings to choose from |
| `clarifications_remaining` | int | Remaining clarification budget |
| `steps_remaining` | int | Steps left before timeout |
| `confidence_signal` | float | 0.0–1.0, rises with useful investigation |
| `blast_radius` | int | Number of services damaged by wrong actions |
| `resolved` | bool | Whether the incident has been resolved |

---

## ⚡ Reward Function

| Signal | Value | What It Teaches |
|:-------|:------|:----------------|
| ✅ Correct resolution | **+10.0** | The primary objective |
| 🏎️ Efficiency bonus | +0.5 to +3.5 | Resolve faster after enough evidence |
| ⏱️ Step penalty | -0.2 / step | Be efficient — don't over-investigate |
| ❌ Wrong resolution | -3.0 | Investigate before acting |
| 💥 Blast radius | -2.0 / service | Wrong actions cascade to other services |
| 🔁 Redundant investigation | -0.3 | Don't check the same service twice |
| 🚫 Clarification exhausted | -2.0 | Use budget wisely — don't waste questions |
| ⏰ Timeout | -5.0 | Biggest penalty — running out of time |

### Confidence Signal

The `confidence_signal` field (0.0–1.0) in every observation is the direct RL analogue of OPTI-FAB's confidence gate:

```
confidence = 0.5 × (services_investigated / total_services)
           + 0.5 × (root_cause_investigated ? 1.0 : 0.0)
           - 0.15 × blast_radius
```

It rises as the agent builds evidence, falls when wrong actions increase blast radius. An agent that watches this signal learns *when* to act.

---

## 🎓 Grading System

Each difficulty level has a specialized grader. Scores range from 0.0 to 1.0.

### Easy Grader
| Component | Score |
|:----------|:------|
| Correct resolution | +0.60 base |
| Root cause investigated | +0.15 |
| Zero blast radius | +0.10 |
| Efficiency bonus | +0.0 to +0.15 |
| Red herring penalty | -0.10 per investigated |

### Medium Grader
| Component | Score |
|:----------|:------|
| Correct resolution | +0.50 base |
| Root cause investigated | +0.15 |
| Clarification used wisely | +0.05 |
| Zero blast radius | +0.10 |
| Efficiency bonus | +0.0 to +0.10 |
| Red herring penalty | -0.10 per investigated |

### Hard Grader
| Component | Score |
|:----------|:------|
| Correct resolution | +0.50 base |
| Root cause investigated | +0.10 |
| Key dependency #1 investigated | +0.08 |
| Key dependency #2 investigated | +0.07 |
| Zero blast radius | +0.10 |
| Clarification budget efficiency | +0.02 to +0.05 |
| Efficiency bonus | +0.0 to +0.10 |
| Red herring penalty | -0.05 per investigated |

---

## 🚀 Running Locally

### Quick Start

```bash
# Clone the repository
git clone https://huggingface.co/spaces/hariharan1828/incidentmind
cd incidentmind

# Install dependencies
pip install -r requirements.txt

# Set environment variables
cp .env.example .env
# Edit .env with your HuggingFace token

# Start the environment server
uvicorn incident_env:app --host 0.0.0.0 --port 7860

# In another terminal — run baseline agent
python inference.py

# Run pre-submission validation
python pre_submit_check.py --url http://localhost:7860
```

### With Docker

```bash
docker build -t incidentmind .
docker run -p 7860:7860 \
  -e API_BASE_URL=https://router.huggingface.co/v1 \
  -e MODEL_NAME=meta-llama/Llama-3.3-70B-Instruct \
  -e HF_TOKEN=your_token_here \
  incidentmind
```

---

## 🔐 Environment Variables

| Variable | Required | Default | Description |
|:---------|:--------:|:--------|:------------|
| `API_BASE_URL` | Yes | — | `https://router.huggingface.co/v1` |
| `MODEL_NAME` | Yes | — | `meta-llama/Llama-3.3-70B-Instruct` |
| `HF_TOKEN` | Yes | — | Your HuggingFace API token |
| `ENV_URL` | No | `http://localhost:7860` | Environment URL for inference script |

---

## 📁 Project Structure

```
incidentmind/
├── incident_env.py        # Core FastAPI environment — step/reset/state + landing page
├── graders.py             # Deterministic graders — pure Python, no LLM calls
├── scenarios.json         # 9 incident scenarios (3 easy / 3 medium / 3 hard)
├── inference.py           # Baseline agent — OpenAI client + timeout guard
├── baseline_scores.json   # Reproducible baseline scores from Llama-3.3-70B
├── openenv.yaml           # OpenEnv spec metadata v1.1.0
├── Dockerfile             # Container for HuggingFace Spaces (port 7860)
├── requirements.txt       # Python dependencies (FastAPI, Pydantic, OpenAI)
├── pre_submit_check.py    # Pre-submission validation (7 automated checks)
├── .env.example           # Environment variable template
└── README.md              # This file
```

---

## 🏗️ Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                     IncidentMind Environment                  │
│                                                              │
│  ┌─────────┐    ┌──────────────┐    ┌───────────────────┐   │
│  │ Scenario │───▶│   State      │───▶│   Observation     │   │
│  │ Selector │    │   Manager    │    │   Builder         │   │
│  │ (random) │    │              │    │                   │   │
│  └─────────┘    │ • logs_seen  │    │ • alerts          │   │
│                 │ • blast_rad  │    │ • confidence_sig  │   │
│                 │ • step count │    │ • possible_actions│   │
│                 └──────┬───────┘    └───────────────────┘   │
│                        │                                     │
│                        ▼                                     │
│                 ┌──────────────┐                             │
│                 │   Reward     │    ┌───────────────────┐   │
│                 │   Calculator │───▶│   Grader          │   │
│                 │              │    │   (deterministic)  │   │
│                 │ • step pen.  │    │                   │   │
│                 │ • blast rad  │    │ • grade_easy()    │   │
│                 │ • efficiency │    │ • grade_medium()  │   │
│                 └──────────────┘    │ • grade_hard()    │   │
│                                    └───────────────────┘   │
└──────────────────────────────────────────────────────────────┘
          ▲                                    │
          │  POST /step {action}               │  {observation, reward, done}
          │  POST /reset?difficulty=X           │
          │                                    ▼
┌──────────────────────────────────────────────────────────────┐
│                     RL Agent / Inference Script               │
│                                                              │
│  ┌─────────────┐    ┌──────────────┐    ┌───────────────┐   │
│  │ Observation  │───▶│   LLM        │───▶│   Action      │   │
│  │ Parser       │    │   (Llama-3.3)│    │   Formatter   │   │
│  └─────────────┘    └──────────────┘    └───────────────┘   │
└──────────────────────────────────────────────────────────────┘
```

---

## 📚 Key Research Context

| Stat | Value | Source |
|:-----|:------|:-------|
| GPT-5 false positive rate in incident response | 82.5% | OpenSec paper, 2026 |
| Agent success drop with ambiguity | 90% → 40% | OpenSec production study |
| Gaia2 lowest-scoring capability | Ambiguity Resolution | Gaia2 benchmark, Meta 2025 |
| OPTI-FAB latency reduction | 63.9% | Our benchmark, RTX 4050 |
| IncidentMind training diversity | 9 scenarios × 3 difficulties | This project |

---

## 🔬 Why This Environment Matters

### For RL Research
- **9 distinct scenarios** prevent memorization — agents must generalize
- **Deterministic graders** enable reproducible evaluation without LLM-based scoring
- **Partial progress signals** (confidence, investigation bonuses) give dense rewards
- **Variable episode length** (8–15 steps) tests both speed and thoroughness

### For Production AI
- **Real incident patterns** from actual production outages (OOM, certificate expiry, DNS misconfigs, schema migrations, feature flag disasters)
- **Red herring calibration** — distractors are designed to be plausible, not random noise
- **Blast radius mechanics** — wrong actions damage other services, teaching agents to be cautious

### For the OpenEnv Ecosystem
- Full spec compliance with typed models, step/reset/state API
- Docker deployment with health checks for automated validation
- Baseline inference with reproducible scores

---

## 📝 License

This project is submitted as part of the Meta PyTorch OpenEnv Hackathon 2026.

---

*Hariharan M & Asmitha M | IncidentMind v1.1.0 | Meta PyTorch OpenEnv Hackathon 2026*
