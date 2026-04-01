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

<div align="center">

# 🚨 IncidentMind

### Ambiguity-Aware Production Incident Response RL Environment

**Meta PyTorch OpenEnv Hackathon 2026**

*Hariharan M & Asmitha M*

[![HuggingFace Space](https://img.shields.io/badge/🤗%20HuggingFace-Live%20Demo-blue)](https://huggingface.co/spaces/hariharan1828/incidentmind)
[![OpenEnv](https://img.shields.io/badge/OpenEnv-Compliant-brightgreen)]()
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED)]()
[![Scenarios](https://img.shields.io/badge/Scenarios-9-orange)]()
[![Grading](https://img.shields.io/badge/Grading-Deterministic-purple)]()

> *"Production AI systems don't fail because they lack intelligence.*
> *They fail because they act under uncertainty without knowing what they don't know."*

</div>

---

## Overview

IncidentMind is an [OpenEnv](https://github.com/pytorch/openenv)-compliant reinforcement learning environment that trains AI agents to **diagnose and resolve production software incidents** under real-world conditions: noisy alerts, red herrings, cascading failures, and time pressure.

### The Problem

Frontier LLMs achieve **82–97% false positive rates** in production incident response (OpenSec, 2026). The failure mode is consistent: when faced with multiple simultaneous alerts, untrained agents jump to the most visible symptom and apply the wrong fix — causing cascading blast-radius damage across dependent services.

### The Solution

IncidentMind provides a simulation where agents learn to:

- **Investigate before acting** — retrieve service logs to build an evidence base
- **Ignore red herrings** — distinguish symptoms from root causes
- **Trace dependency chains** — follow causality upstream through service graphs
- **Use clarifications strategically** — ask targeted questions within a limited budget
- **Act decisively** — apply the correct resolution once confidence is high

---

## Research Foundation: OPTI-FAB

IncidentMind is built on the confidence-gating architecture of **OPTI-FAB** (Optimized Process Technology for Intelligent Fabrication and Anomaly-based Breakdowns) — our semiconductor wafer inspection system that achieved a **63.9% reduction in classification latency** by learning *when partial evidence is sufficient to act*.

| OPTI-FAB Concept | IncidentMind Mapping |
|:--|:--|
| Confidence gate ≥ 0.85 | `confidence_signal` — rises as the agent gathers relevant evidence |
| Entropy threshold ≤ 0.35 | `clarification_budget` — bounded question-asking under uncertainty |
| Early exit at 50–68% of pipeline | Efficiency bonus — resolve earlier with fewer steps for higher reward |
| False reject = wasted manufacturing cost | Wrong fix = blast-radius cascade across dependent services |
| Circular buffer streaming | Incremental log investigation — agent sees partial data, requests more |

**Same architecture. Same principles. Different domain.**

---

## Scenario Design

Nine scenarios across three difficulty tiers. Each episode randomly selects one scenario per difficulty, preventing memorization and rewarding genuine diagnostic reasoning.

### Easy — Single Root Cause (Ambiguity Level 0)

| ID | Incident | Root Cause Service | Resolution | Red Herring | Steps |
|:--|:--|:--|:--|:--|:--:|
| `easy_001` | Database connection pool exhausted | `postgres-primary` | `increase_connections` | redis-cache | 8 |
| `easy_002` | Worker OOM kill — cursor leak in batch import | `order-processor` | `restart_order_processor_with_memory_limit` | message-queue | 8 |
| `easy_003` | Disk full — WAL archiving failure | `postgres-replica` | `clear_wal_archive_and_resize_disk` | log-aggregator | 8 |

### Medium — Cascading Failure with Hidden Root Cause (Ambiguity Level 1)

| ID | Incident | Root Cause Service | Resolution | Red Herring | Steps |
|:--|:--|:--|:--|:--|:--:|
| `medium_001` | ML inference GPU saturation — cascading timeouts | `ml-inference` | `rollback_model_reload` | product-catalog | 12 |
| `medium_002` | DNS misconfiguration — intermittent resolution failures | `dns-resolver` | `rollback_dns_config` | health-checker | 12 |
| `medium_003` | TLS certificate expiry on internal billing API | `billing-api` | `renew_billing_api_certificate` | cert-monitor | 12 |

### Hard — Multi-Alert Chaos (Ambiguity Level 2)

| ID | Incident | Root Cause Service | Resolution | Red Herrings | Steps |
|:--|:--|:--|:--|:--|:--:|
| `hard_001` | Feature flag activates model requiring missing embeddings | `fraud-detection` | `disable_feature_flag_fraud_model_v3` | payment-gateway, session-store, auth-service | 15 |
| `hard_002` | Certificate rotation with incomplete CA propagation | `certificate-manager` | `rollback_certificate_rotation` | api-gateway, database-proxy, monitoring-agent | 15 |
| `hard_003` | Schema migration race condition — silent data corruption | `schema-migrator` | `rollback_migration_v48` | inventory-service, reconciliation-job, pricing-engine | 15 |

---

## Baseline Performance

Baseline agent: **Llama-3.3-70B-Instruct** (untrained, zero-shot via HuggingFace Inference API).

| Difficulty | Score | Threshold | Result |
|:--|:--:|:--:|:--:|
| **Easy** | 0.825 | 0.70 | ✅ Pass |
| **Medium** | 0.838 | 0.60 | ✅ Pass |
| **Hard** | 0.830 | 0.50 | ✅ Pass |

The baseline demonstrates that IncidentMind is **solvable by capable LLMs** while remaining challenging enough for RL training to produce meaningful improvements. The `possible_actions` list in each observation ensures agents select from valid resolution strings rather than hallucinating action names.

---

## API Reference

IncidentMind exposes a standard OpenEnv-compliant REST API.

| Endpoint | Method | Description |
|:--|:--|:--|
| `/` | `GET` | Landing page with environment overview |
| `/health` | `GET` | Health check — returns `{"status": "ok"}` |
| `/reset?difficulty={easy\|medium\|hard}` | `POST` | Start a fresh episode; returns initial observation |
| `/step` | `POST` | Submit an action; returns observation, reward, done, info |
| `/state` | `GET` | Full current episode state as JSON |
| `/tasks` | `GET` | Enumerate all 9 tasks with metadata and thresholds |
| `/docs` | `GET` | Interactive Swagger/OpenAPI documentation |

### Action Space

```json
{"action_type": "investigate",       "target": "service-name"}
{"action_type": "ask_clarification", "target": "question_key"}
{"action_type": "resolve",           "resolution_action": "exact_action_string"}
{"action_type": "rollback",          "target": "service-name"}
{"action_type": "escalate"}
```

> **Important:** The `resolution_action` must be one of the exact strings from the `possible_actions` field in the observation. Free-text descriptions will not match.

### Observation Space

| Field | Type | Description |
|:--|:--|:--|
| `scenario_id` | `string` | Current scenario identifier |
| `difficulty` | `string` | `easy`, `medium`, or `hard` |
| `description` | `string` | Incident description with context |
| `alerts` | `list[Alert]` | Active alerts — each has `service`, `severity`, `message` |
| `logs_available` | `list[str]` | Services whose logs can be investigated |
| `logs_seen` | `dict[str, list[str]]` | Logs already retrieved, keyed by service |
| `actions_taken` | `list[dict]` | Full history of actions taken this episode |
| `possible_actions` | `list[str]` | Valid resolution strings for the current scenario |
| `clarifications_remaining` | `int` | Remaining clarification budget |
| `steps_remaining` | `int` | Steps left before timeout |
| `confidence_signal` | `float` | 0.0–1.0 — rises with useful investigation, falls with blast radius |
| `blast_radius` | `int` | Number of services damaged by incorrect actions |
| `resolved` | `bool` | Whether the incident has been correctly resolved |

### Example Episode

```python
import requests

BASE = "https://hariharan1828-incidentmind.hf.space"

# 1. Start a hard episode
obs = requests.post(f"{BASE}/reset?difficulty=hard").json()
print(obs["alerts"])              # 5 simultaneous alerts — which is the real cause?
print(obs["possible_actions"])    # ["disable_feature_flag_fraud_model_v3", ...]

# 2. Investigate a suspect service
result = requests.post(f"{BASE}/step", json={
    "action_type": "investigate",
    "target": "fraud-detection"
}).json()
print(result["info"]["logs"])     # Reveals KeyError on user_embedding_v3
print(result["observation"]["confidence_signal"])  # 0.35 → rising

# 3. Ask a clarification question
result = requests.post(f"{BASE}/step", json={
    "action_type": "ask_clarification",
    "target": "what_changed_31_minutes_ago"
}).json()
print(result["info"]["result"])   # "Feature flag fraud_model_v3 was enabled..."

# 4. Resolve with the correct action
result = requests.post(f"{BASE}/step", json={
    "action_type": "resolve",
    "resolution_action": "disable_feature_flag_fraud_model_v3"
}).json()
print(result["reward"])           # +10.0 base + efficiency bonus
print(result["info"]["final_score"])  # 0.83
print(result["done"])             # True — episode complete
```

---

## Reward Function

| Signal | Value | Purpose |
|:--|:--|:--|
| ✅ Correct resolution | **+10.0** | Primary objective — apply the right fix |
| 🏎️ Efficiency bonus | **+0.0 to +3.5** | Scaled by `steps_remaining / max_steps` |
| ⏱️ Step penalty | **−0.2** per step | Encourage efficiency — don't over-investigate |
| ❌ Wrong resolution | **−3.0** | Penalize premature or incorrect fixes |
| 💥 Blast radius | **−2.0** per service | Wrong actions cascade to dependent services |
| 🔁 Redundant investigation | **−0.3** | Don't investigate the same service twice |
| ❓ Clarification cost | **−0.5** per question | Information has a price — ask wisely |
| 🚫 Budget exhausted | **−2.0** | Attempting clarification with zero remaining |
| ⏰ Timeout | **−5.0** | Largest penalty — running out of steps |

### Confidence Signal

The `confidence_signal` (0.0–1.0) mirrors OPTI-FAB's confidence gate:

```
confidence = 0.5 × (services_investigated / total_services)
           + 0.5 × (root_cause_investigated ? 1.0 : 0.0)
           − 0.15 × blast_radius
```

This provides a dense signal: it rises as the agent gathers relevant evidence and falls when incorrect actions increase blast radius.

---

## Grading System

All grading is **deterministic** — no LLM judge, no randomness. Scores are reproducible given identical episode state.

### Easy Grader (threshold: 0.70)

| Component | Score | Condition |
|:--|:--:|:--|
| Correct resolution | +0.70 | Applied `correct_action` while `resolved=True` |
| Root cause investigated | +0.15 | Investigated `root_cause_service` before resolving |
| Efficiency bonus | +0.00 to +0.15 | `0.15 × (steps_remaining / max_steps)` |

### Medium Grader (threshold: 0.60)

| Component | Score | Condition |
|:--|:--:|:--|
| Correct resolution | +0.65 | Applied `correct_action` |
| Root cause investigated | +0.10 | Investigated `root_cause_service` |
| Investigation discipline | +0.10 | Investigated root cause *before* any red herring |
| Efficiency bonus | +0.00 to +0.15 | `0.15 × (steps_remaining / max_steps)` |
| Partial credit | 0.20 | Wrong action but investigated root cause |

### Hard Grader (threshold: 0.50)

| Component | Score | Condition |
|:--|:--:|:--|
| Correct resolution | +0.50 | Applied `correct_action` |
| Root cause investigated | +0.10 | Investigated `root_cause_service` |
| Key dependency #1 | +0.08 | Investigated first `key_investigation_service` |
| Key dependency #2 | +0.07 | Investigated second `key_investigation_service` |
| Zero blast radius | +0.10 | No incorrect actions during the episode |
| Clarification efficiency | +0.02 to +0.05 | Strategic use of 1–2 clarifications |
| Efficiency bonus | +0.00 to +0.10 | `0.10 × (steps_remaining / max_steps)` |
| Red herring penalty | −0.05 each | Per red-herring service investigated |
| Blast radius penalty | −0.10 each | Per service damaged by wrong actions |

---

## Getting Started

### Local Development

```bash
# Clone the repository
git clone https://huggingface.co/spaces/hariharan1828/incidentmind
cd incidentmind

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env with your HuggingFace token and model settings

# Start the environment server
uvicorn incident_env:app --host 0.0.0.0 --port 7860

# Run the baseline agent (in a separate terminal)
python inference.py

# Run pre-submission validation
python pre_submit_check.py --url http://localhost:7860
```

### Docker Deployment

```bash
docker build -t incidentmind .
docker run -p 7860:7860 \
  -e API_BASE_URL=https://router.huggingface.co/v1 \
  -e MODEL_NAME=meta-llama/Llama-3.3-70B-Instruct \
  -e HF_TOKEN=your_token_here \
  incidentmind
```

### Environment Variables

| Variable | Required | Default | Description |
|:--|:--:|:--|:--|
| `API_BASE_URL` | Yes | — | LLM API endpoint (e.g., `https://router.huggingface.co/v1`) |
| `MODEL_NAME` | Yes | — | Model identifier (e.g., `meta-llama/Llama-3.3-70B-Instruct`) |
| `HF_TOKEN` | Yes | — | HuggingFace API token for model access |
| `ENV_URL` | No | `http://localhost:7860` | Environment URL used by the inference script |

---

## Project Structure

```
incidentmind/
├── incident_env.py          # FastAPI environment — step / reset / state / landing page
├── graders.py               # Deterministic graders — pure state-based, no LLM judge
├── scenarios.json            # 9 incident scenarios (3 per difficulty tier)
├── inference.py              # Baseline agent — Llama-3.3-70B with timeout guard
├── baseline_scores.json      # Reproducible scores from baseline evaluation
├── openenv.yaml              # OpenEnv specification metadata (v1.1.0)
├── pre_submit_check.py       # Pre-submission validator (7 automated checks)
├── Dockerfile                # Container configuration for HuggingFace Spaces
├── requirements.txt          # Python dependencies
├── .env.example              # Environment variable template
└── README.md
```

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                     IncidentMind Environment                     │
│                                                                  │
│  ┌───────────┐    ┌──────────────┐    ┌──────────────────────┐  │
│  │  Scenario  │───▶│    State     │───▶│    Observation        │  │
│  │  Selector  │    │    Manager   │    │    Builder            │  │
│  │ (random)   │    │              │    │                      │  │
│  └───────────┘    │ • logs_seen  │    │ • alerts             │  │
│                   │ • blast_rad  │    │ • confidence_signal  │  │
│                   │ • step_count │    │ • possible_actions   │  │
│                   └──────┬───────┘    └──────────────────────┘  │
│                          │                                       │
│                          ▼                                       │
│                   ┌──────────────┐    ┌──────────────────────┐  │
│                   │   Reward     │───▶│    Grader             │  │
│                   │   Calculator │    │   (deterministic)     │  │
│                   │              │    │                      │  │
│                   │ • +10 correct│    │ • grade_easy()       │  │
│                   │ • -3  wrong  │    │ • grade_medium()     │  │
│                   │ • -0.2/step  │    │ • grade_hard()       │  │
│                   └──────────────┘    └──────────────────────┘  │
└──────────────────────────────────────────────────────────────────┘
          ▲                                      │
          │  POST /step {action}                 │  {observation, reward, done}
          │  POST /reset?difficulty=X             │
          │                                      ▼
┌──────────────────────────────────────────────────────────────────┐
│                     RL Agent / Inference Client                   │
│                                                                  │
│  ┌─────────────┐    ┌──────────────┐    ┌──────────────────┐   │
│  │ Observation  │───▶│     LLM      │───▶│   Action         │   │
│  │ Parser       │    │ (Llama-3.3)  │    │   Formatter      │   │
│  └─────────────┘    └──────────────┘    └──────────────────┘   │
└──────────────────────────────────────────────────────────────────┘
```

---

## Research Context

| Metric | Value | Source |
|:--|:--|:--|
| GPT-5 false positive rate in incident triage | 82–97% | OpenSec, 2026 |
| Agent success rate drop under ambiguity | 90% → 40% | OpenSec production study |
| Lowest-scoring capability on Gaia2 | Ambiguity Resolution | Meta, 2025 |
| OPTI-FAB latency reduction | 63.9% | Our benchmark (RTX 4050) |
| IncidentMind scenario diversity | 9 scenarios × 3 difficulty tiers | This project |

### Why This Environment Matters

**For RL Research:**
- 9 distinct scenarios prevent memorization — agents must generalize across incident patterns
- Deterministic graders enable reproducible evaluation without LLM-based scoring variance
- Confidence signal provides dense intermediate rewards — not just sparse episode-end scoring
- Variable episode lengths (8–15 steps) test both speed and thoroughness

**For Production AI Safety:**
- Scenarios model real-world outage patterns: OOM kills, certificate expiry, DNS misconfigurations, schema migration failures, feature flag disasters
- Red herrings are calibrated to be plausible — they are real concurrent issues, not random noise
- Blast-radius mechanics teach agents that wrong actions have downstream consequences

**For the OpenEnv Ecosystem:**
- Full specification compliance with typed Pydantic models and step/reset/state API
- Docker deployment with health checks for automated validation pipelines
- Reproducible baseline with saved scores and pre-submission validation scripts

---

## License

This project is submitted as part of the **Meta PyTorch OpenEnv Hackathon 2026**.

---

<div align="center">

*Built by **Hariharan M & Asmitha M** · IncidentMind v1.1.0 · Powered by OPTI-FAB confidence-gating architecture*

</div>
