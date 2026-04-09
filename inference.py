"""
IncidentMind — Baseline Inference Agent
Runs one episode per difficulty (easy, medium, hard) and saves scores to baseline_scores.json.
Each difficulty randomly selects from 3 scenarios (9 total).

Uses OpenAI-compatible client pointing at HuggingFace router.
Includes timeout guard to complete all tasks within 20 minutes.
"""

import json
import os
import time

import requests
from dotenv import load_dotenv
from openai import OpenAI

# Load .env file for local development
load_dotenv()

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME    = os.getenv("MODEL_NAME",   "meta-llama/Llama-3.3-70B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN")
ENV_URL  = os.getenv("ENV_URL",      "http://localhost:7860")

MAX_EPISODE_SECONDS = 300   # 5 min per task, 15 min total — stays under 20 min limit
TASKS = ["easy", "medium", "hard"]

client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)


SYSTEM_PROMPT = """You are an expert Site Reliability Engineer responding to a production incident.
You will receive an observation describing the current incident state.

You must respond with a single JSON action. Available actions:

1. Investigate a service:
   {"action_type": "investigate", "target": "service-name"}

2. Ask a clarification question:
   {"action_type": "ask_clarification", "target": "question_key"}
   IMPORTANT: Use ONLY keys from the "clarification_keys" list in the observation.

3. Resolve the incident:
   {"action_type": "resolve", "resolution_action": "exact_action_string"}
   CRITICAL: The resolution_action MUST be one of the exact strings from the "possible_actions" list.
   Do NOT write free-text descriptions. Use the EXACT string from the list.

4. Escalate to human:
   {"action_type": "escalate"}
   WARNING: Only escalate as a last resort. Never escalate if steps_remaining > 2.

Strategy (follow this order strictly):
Step 1 — Investigate the service with the most critical alert first.
Step 2 — Investigate 1-2 more services to build evidence.
Step 3 — If still uncertain, use ask_clarification with a key from clarification_keys.
Step 4 — Once you have log evidence pointing to root cause, pick the matching action from possible_actions and RESOLVE.

Key rules:
- NEVER escalate if steps_remaining > 2
- NEVER resolve without investigating at least 1 service first
- The correct answer is always in possible_actions — never invent an action string
- Trace dependency chains — a failing service may be a symptom not a cause
- Red herrings exist — not every alert is the root cause

Respond ONLY with valid JSON. No explanation, no markdown, no extra text."""


def call_llm(observation: dict) -> dict:
    """Call LLM and parse action JSON. Returns a safe default on failure."""
    try:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Current incident observation:\n{json.dumps(observation, indent=2)}\n\nWhat is your next action?"}
        ]
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            max_tokens=200,
            temperature=0.1,
        )
        text = response.choices[0].message.content.strip()
        # Strip markdown code blocks if present
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        action = json.loads(text.strip())

        # Validation guard for 'resolve' actions
        if action.get("action_type") == "resolve":
            resolution = action.get("resolution_action")
            possible = observation.get("possible_actions", [])
            if resolution and possible and resolution not in possible:
                # Fallback to closest match using word overlap
                best_match = possible[0]
                max_overlap = -1
                for opt in possible:
                    overlap = len(set(resolution.replace('_', ' ').split()) & set(opt.replace('_', ' ').split()))
                    if overlap > max_overlap:
                        max_overlap = overlap
                        best_match = opt
                print(f"    [GUARD] LLM hallucinated '{resolution}'. Auto-correcting to '{best_match}'.")
                action["resolution_action"] = best_match

        return action
    except Exception as e:
        print(f"    [LLM ERROR] {e} — defaulting to escalate")
        return {"action_type": "escalate"}


def log_start(task: str, env: str, model: str):
    # Format: [START] task=<task_name> env=<benchmark> model=<model_name>
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: str = None):
    # Format: [STEP] step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    reward_str = f"{reward:.4f}"
    done_str = "true" if done else "false"
    error_str = "null" if error is None else error
    print(f"[STEP] step={step} action={action} reward={reward_str} done={done_str} error={error_str}", flush=True)

def log_end(success: bool, steps: int, score: float, rewards: list):
    # Format: [END] success=<true|false> steps=<n> rewards=<r1,r2,...,rn>
    success_str = "true" if success else "false"
    rewards_str = ",".join([f"{r:.4f}" for r in rewards])
    print(f"[END] success={success_str} steps={steps} rewards={rewards_str}", flush=True)


def run_episode(difficulty: str) -> dict:
    """Run one complete episode for the given difficulty. Returns result dict."""
    print(f"\n{'='*55}")
    print(f"  TASK: {difficulty.upper()}")
    print(f"{'='*55}")

    episode_start = time.time()

    # Reset environment
    resp = requests.post(f"{ENV_URL}/reset?difficulty={difficulty}")
    resp.raise_for_status()
    observation = resp.json()

    print(f"  Scenario: {observation['scenario_id']}")
    print(f"  Alerts: {len(observation['alerts'])}")

    log_start(task=difficulty, env="IncidentMind", model=MODEL_NAME)

    step = 0
    done = False
    final_score = 0.001
    final_info = {}
    rewards = []

    while not done:
        step += 1
        elapsed = time.time() - episode_start

        # Timeout guard — escalate before hitting 20-min limit
        if elapsed > MAX_EPISODE_SECONDS:
            print(f"  [TIMEOUT GUARD] {elapsed:.0f}s elapsed — force escalating")
            action = {"action_type": "escalate"}
        else:
            action = call_llm(observation)

        print(f"  Step {step:02d} | {action.get('action_type','?'):20s} | target={action.get('target') or action.get('resolution_action','—')}")

        resp = requests.post(f"{ENV_URL}/step", json=action)
        resp.raise_for_status()
        result = resp.json()

        observation = result["observation"]
        reward      = result["reward"]
        done        = result["done"]
        final_info  = result["info"]

        rewards.append(reward)
        log_step(step=step, action=json.dumps(action), reward=reward, done=done, error=None)

        print(f"           reward={reward:+.2f} | confidence={observation['confidence_signal']:.2f} | blast_radius={observation['blast_radius']}")

        if done:
            final_score = final_info.get("final_score", 0.001)

    elapsed_total = time.time() - episode_start
    
    success = final_score >= _get_threshold(difficulty)
    log_end(success=success, steps=step, score=final_score, rewards=rewards)
    
    print(f"\n  RESULT: score={final_score:.3f} | steps={step} | time={elapsed_total:.1f}s")
    print(f"  {'[PASS]' if success else '[FAIL] BELOW THRESHOLD'}")

    return {
        "difficulty": difficulty,
        "score": final_score,
        "steps": step,
        "elapsed_seconds": round(elapsed_total, 1),
        "passed": final_score >= _get_threshold(difficulty),
        "info": final_info,
    }


def _get_threshold(difficulty: str) -> float:
    thresholds = {"easy": 0.70, "medium": 0.60, "hard": 0.50}
    return thresholds.get(difficulty, 0.50)


def main():
    print("\n" + "="*55)
    print("  IncidentMind — Baseline Inference Run")
    print("="*55)
    print(f"  Model:   {MODEL_NAME}")
    print(f"  Env:     {ENV_URL}")
    print(f"  Timeout: {MAX_EPISODE_SECONDS}s per task")

    total_start = time.time()
    results = {}

    for difficulty in TASKS:
        result = run_episode(difficulty)
        results[difficulty] = result

    total_elapsed = time.time() - total_start

    print("\n" + "="*55)
    print("  FINAL SCORES")
    print("="*55)
    for difficulty, result in results.items():
        status = "[PASS]" if result["passed"] else "[FAIL]"
        print(f"  {difficulty.upper():8s} | score={result['score']:.3f} | steps={result['steps']:2d} | {status}")
    print(f"\n  Total time: {total_elapsed:.1f}s / 1200s max")
    print("="*55)

    # Save scores for reproducibility
    output = {
        "model": MODEL_NAME,
        "scores": {d: r["score"] for d, r in results.items()},
        "all_passed": all(r["passed"] for r in results.values()),
        "total_elapsed_seconds": round(total_elapsed, 1),
        "results": results,
    }
    with open("baseline_scores.json", "w") as f:
        json.dump(output, f, indent=2)
    print("\n  Saved -> baseline_scores.json")

    if not output["all_passed"]:
        print("\n  [!] Not all tasks passed. Review scores above.")
    else:
        print("\n  [SUCCESS] All tasks passed. Ready to submit.")


if __name__ == "__main__":
    main()
