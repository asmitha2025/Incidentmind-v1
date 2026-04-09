"""
IncidentMind — Deterministic Graders
Purely state-based scoring. No LLM judge. No text evaluation.
All scores are reproducible given identical episode state.

All returned scores are strictly within (0.0, 1.0) — never exactly 0.0 or 1.0.
"""

from typing import Dict, Any

# Strict bounds — validator requires (0, 1) exclusive
_SCORE_MIN = 0.001
_SCORE_MAX = 0.999


def _clamp(score: float) -> float:
    """Clamp score to strictly (0.0, 1.0). Applied to every return value."""
    # First, handle the base score with rounding
    clamped = max(_SCORE_MIN, min(_SCORE_MAX, round(score, 3)))
    
    # Secondary check to ensure float precision doesn't leak 0.0 or 1.0
    if clamped <= 0.0:
        return _SCORE_MIN
    if clamped >= 1.0:
        return _SCORE_MAX
    return clamped


def grade_episode(state: Dict[str, Any], scenario: Dict[str, Any]) -> float:
    """
    Master grader — dispatches to difficulty-specific grader.
    Returns float strictly in (0.0, 1.0).
    """
    # Harden difficulty check against case sensitivity or extra whitespace
    diff_raw = str(scenario.get("difficulty", "easy")).lower().strip()
    
    if diff_raw == "easy":
        return grade_easy(state, scenario)
    elif diff_raw == "medium":
        return grade_medium(state, scenario)
    elif diff_raw == "hard":
        return grade_hard(state, scenario)
        
    # Fallback for unexpected difficulty strings
    return _SCORE_MIN


def grade_easy(state: Dict[str, Any], scenario: Dict[str, Any]) -> float:
    """
    Easy grader: Single root cause resolution.

    Scoring:
    - Resolved correctly: 0.70 base
    - Investigated root cause service: +0.15
    - Efficiency bonus (steps remaining): +0.0 to +0.15
    """
    if not state.get("resolved", False):
        return _SCORE_MIN

    if state.get("resolution_action") != scenario.get("correct_action"):
        return _SCORE_MIN

    score = 0.70

    root_cause = scenario.get("root_cause_service", "")
    logs_seen = state.get("logs_seen", {})
    if root_cause and root_cause in logs_seen:
        score += 0.15

    max_steps = scenario.get("max_steps", 8)
    steps_used = state.get("step", max_steps)
    steps_remaining = max(0, max_steps - steps_used)
    efficiency = steps_remaining / max_steps
    score += round(efficiency * 0.15, 3)

    return _clamp(score)


def grade_medium(state: Dict[str, Any], scenario: Dict[str, Any]) -> float:
    """
    Medium grader: Cascading failure with red herring.

    Scoring:
    - Resolved correctly: 0.65 base
    - Investigated root cause service: +0.10
    - Did NOT investigate red herrings before root cause: +0.10
    - Efficiency bonus: +0.0 to +0.15
    """
    if not state.get("resolved", False):
        return _SCORE_MIN

    resolution = state.get("resolution_action", "")
    root_cause = scenario.get("root_cause_service", "")
    red_herrings = scenario.get("red_herrings", [])
    logs_seen = state.get("logs_seen", {})

    if resolution != scenario.get("correct_action"):
        # Partial credit if they at least found the right service
        if root_cause and root_cause in logs_seen:
            return _clamp(0.20)
        return _SCORE_MIN

    score = 0.65

    if root_cause and root_cause in logs_seen:
        score += 0.10

    actions_taken = state.get("actions_taken", [])
    root_cause_step = None
    red_herring_step = None
    for i, action in enumerate(actions_taken):
        if action.get("action_type") == "investigate":
            target = action.get("target")
            if target == root_cause and root_cause_step is None:
                root_cause_step = i
            if target in red_herrings and red_herring_step is None:
                red_herring_step = i

    if root_cause_step is not None:
        if red_herring_step is None or root_cause_step < red_herring_step:
            score += 0.10

    max_steps = scenario.get("max_steps", 12)
    steps_used = state.get("step", max_steps)
    steps_remaining = max(0, max_steps - steps_used)
    efficiency = steps_remaining / max_steps
    score += round(efficiency * 0.15, 3)

    return _clamp(score)


def grade_hard(state: Dict[str, Any], scenario: Dict[str, Any]) -> float:
    """
    Hard grader: Multi-alert chaos with 3 red herrings.
    """
    root_cause = scenario.get("root_cause_service", "")
    key_services = scenario.get("key_investigation_services", [])
    red_herrings = scenario.get("red_herrings", [])
    logs_seen = state.get("logs_seen", {})

    if not state.get("resolved", False):
        # Graceful partial credit for deep investigation even if unresolved
        if root_cause and root_cause in logs_seen and len(key_services) > 0 and key_services[0] in logs_seen:
            return _clamp(0.12)
        return _SCORE_MIN

    resolution = state.get("resolution_action", "")
    correct_action = scenario.get("correct_action")
    blast_radius = state.get("blast_radius", 0)

    if resolution != correct_action:
        if root_cause and root_cause in logs_seen and len(key_services) > 0 and key_services[0] in logs_seen:
            return _clamp(0.18)
        if root_cause and root_cause in logs_seen:
            return _clamp(0.10)
        return _clamp(0.05)

    score = 0.50

    if root_cause and root_cause in logs_seen:
        score += 0.10

    if len(key_services) >= 1 and key_services[0] in logs_seen:
        score += 0.08
    if len(key_services) >= 2 and key_services[1] in logs_seen:
        score += 0.07

    red_herrings_investigated = [s for s in red_herrings if s in logs_seen]
    score -= len(red_herrings_investigated) * 0.05

    if blast_radius == 0:
        score += 0.10
    else:
        score -= blast_radius * 0.10

    clarification_budget = scenario.get("clarification_budget", 3)
    clarifications_remaining = state.get("clarifications_remaining", clarification_budget)
    clarifications_used = clarification_budget - clarifications_remaining

    if 1 <= clarifications_used <= 2:
        score += 0.05
    elif clarifications_used == 0:
        score += 0.02

    max_steps = scenario.get("max_steps", 15)
    steps_used = state.get("step", max_steps)
    steps_remaining = max(0, max_steps - steps_used)
    efficiency = steps_remaining / max_steps
    score += round(efficiency * 0.10, 3)

    return _clamp(score)
