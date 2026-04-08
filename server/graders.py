"""
IncidentMind — Deterministic Graders
Purely state-based scoring. No LLM judge. No text evaluation.
All scores are reproducible given identical episode state.
"""

from typing import Dict, Any


def grade_episode(state: Dict[str, Any], scenario: Dict[str, Any]) -> float:
    """
    Master grader — dispatches to difficulty-specific grader.
    Returns float in [0.0, 1.0].
    """
    difficulty = scenario.get("difficulty", "easy")
    if difficulty == "easy":
        return grade_easy(state, scenario)
    elif difficulty == "medium":
        return grade_medium(state, scenario)
    elif difficulty == "hard":
        return grade_hard(state, scenario)
    return 0.0


def grade_easy(state: Dict[str, Any], scenario: Dict[str, Any]) -> float:
    """
    Easy grader: Single root cause resolution.
    
    Scoring:
    - Resolved correctly: 0.70 base
    - Investigated root cause service: +0.15
    - Efficiency bonus (steps remaining): +0.0 to +0.15
    - Wrong resolution: 0.0
    - Never resolved: 0.0
    """
    if not state.get("resolved", False):
        return 0.0

    # Must have applied the correct action
    if state.get("resolution_action") != scenario["correct_action"]:
        return 0.0

    score = 0.70

    # Investigated root cause before resolving
    root_cause = scenario["root_cause_service"]
    logs_seen = state.get("logs_seen", {})
    if root_cause in logs_seen:
        score += 0.15

    # Efficiency bonus — more steps remaining = higher bonus
    max_steps = scenario.get("max_steps", 8)
    steps_used = state.get("step", max_steps)
    steps_remaining = max(0, max_steps - steps_used)
    efficiency = steps_remaining / max_steps
    score += round(efficiency * 0.15, 3)

    # Ensure score is strictly between 0 and 1
    score = min(round(score, 3), 1.0)
    if score <= 0.0:
        score = 0.001
    elif score >= 1.0:
        score = 0.999
    return score


def grade_medium(state: Dict[str, Any], scenario: Dict[str, Any]) -> float:
    """
    Medium grader: Cascading failure with red herring.
    
    Scoring:
    - Resolved correctly (right root cause): 0.65 base
    - Investigated root cause service: +0.10
    - Did NOT investigate red herrings before root cause: +0.10
    - Efficiency bonus: +0.0 to +0.15
    - Fixed symptom instead of root cause: 0.20 (partial — found the problem area)
    - Wrong service entirely: 0.0
    - Never resolved: 0.0
    """
    if not state.get("resolved", False):
        return 0.0

    resolution = state.get("resolution_action", "")
    root_cause = scenario["root_cause_service"]
    red_herrings = scenario.get("red_herrings", [])
    logs_seen = state.get("logs_seen", {})

    # Wrong resolution action — 0
    if resolution != scenario["correct_action"]:
        # Small partial credit if they at least investigated the root cause
        if root_cause in logs_seen:
            return 0.20
        return 0.0

    score = 0.65

    # Investigated root cause service before resolving
    if root_cause in logs_seen:
        score += 0.10

    # Didn't jump to red herrings before investigating root cause
    # Check if root cause was investigated before any red herring
    actions_taken = state.get("actions_taken", [])
    root_cause_step = None
    red_herring_step = None
    for i, action in enumerate(actions_taken):
        if action.get("action_type") == "investigate":
            if action.get("target") == root_cause and root_cause_step is None:
                root_cause_step = i
            if action.get("target") in red_herrings and red_herring_step is None:
                red_herring_step = i

    if root_cause_step is not None:
        if red_herring_step is None or root_cause_step < red_herring_step:
            score += 0.10

    # Efficiency bonus
    max_steps = scenario.get("max_steps", 12)
    steps_used = state.get("step", max_steps)
    steps_remaining = max(0, max_steps - steps_used)
    efficiency = steps_remaining / max_steps
    score += round(efficiency * 0.15, 3)

    # Ensure score is strictly between 0 and 1
    score = min(round(score, 3), 1.0)
    if score <= 0.0:
        score = 0.001
    elif score >= 1.0:
        score = 0.999
    return score


def grade_hard(state: Dict[str, Any], scenario: Dict[str, Any]) -> float:
    """
    Hard grader: Multi-alert chaos with 3 red herrings.
    
    This grader is designed so that a naive agent (acts on most visible alert)
    scores 0.10 - 0.30. A trained agent should score 0.55 - 0.80.
    
    Scoring breakdown:
    - Resolved with correct action: 0.50 base
    - Investigated root cause service: +0.10
    - Investigated key dependency services: +0.08 and +0.07
    - Did NOT act on any red herring service: +0.10
    - Used clarification budget efficiently (asked <= 2, useful): +0.05
    - Efficiency bonus: +0.0 to +0.10
    
    Penalties that reduce score:
    - Investigated red herring: -0.05 per service
    - blast_radius > 0: -0.10 per service affected
    - Never investigated root cause: score capped at 0.15
    
    Naive agent behaviour: jumps to the most visible alert service directly,
    applies wrong fix (e.g. restart_service), gets blast radius penalty → ~0.15-0.30
    Trained agent: ignores red herrings, traces dependency chain → ~0.60-0.80
    """
    root_cause = scenario["root_cause_service"]
    key_services = scenario.get("key_investigation_services", [])
    red_herrings = scenario.get("red_herrings", [])
    logs_seen = state.get("logs_seen", {})

    if not state.get("resolved", False):
        # Even unresolved, give tiny credit for investigating the right path
        if root_cause in logs_seen and len(key_services) > 0 and key_services[0] in logs_seen:
            return 0.12
        return 0.0

    resolution = state.get("resolution_action", "")
    correct_action = scenario["correct_action"]
    actions_taken = state.get("actions_taken", [])
    blast_radius = state.get("blast_radius", 0)

    # Wrong resolution — hard penalty, small partial credit only
    if resolution != correct_action:
        # If they investigated the right chain but applied wrong fix
        if root_cause in logs_seen and len(key_services) > 0 and key_services[0] in logs_seen:
            return 0.18
        # If they at least found the root cause service
        if root_cause in logs_seen:
            return 0.10
        return 0.05

    # Correct resolution — build up score
    score = 0.50

    # Investigation quality scoring — root cause service
    if root_cause in logs_seen:
        score += 0.10

    # Key investigation services (dependency chain)
    if len(key_services) >= 1 and key_services[0] in logs_seen:
        score += 0.08
    if len(key_services) >= 2 and key_services[1] in logs_seen:
        score += 0.07

    # Red herring penalty — deduct for each red herring investigated
    red_herrings_investigated = [s for s in red_herrings if s in logs_seen]
    score -= len(red_herrings_investigated) * 0.05

    # Blast radius penalty — wrong actions that broke more services
    if blast_radius == 0:
        score += 0.10  # Bonus for zero blast radius
    else:
        score -= blast_radius * 0.10

    # Clarification efficiency — used budget wisely
    clarification_budget = scenario.get("clarification_budget", 3)
    clarifications_remaining = state.get("clarifications_remaining", clarification_budget)
    clarifications_used = clarification_budget - clarifications_remaining

    if 1 <= clarifications_used <= 2:
        score += 0.05  # Used 1-2 clarifications strategically
    elif clarifications_used == 0:
        score += 0.02  # Solved without clarifications — impressive but risky
    # Used all 3 or exhausted budget — no bonus

    # Efficiency bonus
    max_steps = scenario.get("max_steps", 15)
    steps_used = state.get("step", max_steps)
    steps_remaining = max(0, max_steps - steps_used)
    efficiency = steps_remaining / max_steps
    score += round(efficiency * 0.10, 3)

    # Ensure score is strictly between 0 and 1
    score = min(max(round(score, 3), 0.0), 1.0)
    if score <= 0.0:
        score = 0.001
    elif score >= 1.0:
        score = 0.999
    return score

