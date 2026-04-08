"""
Graders for the Smart Meeting Scheduler environment.

Each grader receives an EnvironmentState and returns a score in [0.0, 1.0].
Graders are deterministic and programmatic — no LLM needed.
"""
from __future__ import annotations

from typing import Callable, Dict

from models import CalendarEvent, EnvironmentState, PRIORITY_WEIGHT

WORKING_START = 9 * 60
WORKING_END = 18 * 60


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _count_overlaps(events: list[CalendarEvent]) -> int:
    count = 0
    for i, a in enumerate(events):
        for b in events[i + 1:]:
            if a.overlaps(b):
                count += 1
    return count


def _all_within_hours(events: list[CalendarEvent]) -> bool:
    return all(
        e.start_minutes() >= WORKING_START and e.end_minutes() <= WORKING_END
        for e in events
    )


def _completion_ratio(state: EnvironmentState) -> float:
    total = len(state.completed_request_ids) + len(state.pending_requests)
    if total == 0:
        return 1.0
    return len(state.completed_request_ids) / total


def _overlap_score(events: list[CalendarEvent]) -> float:
    """1.0 if no overlaps, degrades by 0.2 per overlap."""
    overlaps = _count_overlaps(events)
    return max(0.0, 1.0 - overlaps * 0.2)


# ---------------------------------------------------------------------------
# Task-specific graders
# ---------------------------------------------------------------------------

def grade_easy(state: EnvironmentState) -> float:
    """
    Easy task: 3 meetings, no conflicts expected.

    Score breakdown:
      - 60% completion ratio (were all 3 requests scheduled?)
      - 30% no overlaps
      - 10% all within working hours
    """
    evs = state.scheduled_events

    completion = _completion_ratio(state)               # 0–1
    no_overlap = _overlap_score(evs)                    # 0–1
    hours_ok = 1.0 if _all_within_hours(evs) else 0.5  # 0.5 or 1

    score = 0.6 * completion + 0.3 * no_overlap + 0.1 * hours_ok
    return round(min(1.0, max(0.0, score)), 4)


def grade_medium(state: EnvironmentState) -> float:
    """
    Medium task: 5 meetings, priority conflicts must be resolved.

    Score breakdown:
      - 50% completion ratio
      - 25% no overlaps
      - 15% high/critical meetings scheduled (priority adherence)
      - 10% working hours adherence
    """
    evs = state.scheduled_events

    completion = _completion_ratio(state)

    no_overlap = _overlap_score(evs)

    # Priority adherence: were high/critical requests completed?
    high_reqs = [
        r for r in state.pending_requests
        if PRIORITY_WEIGHT[r.priority] >= 3
    ]
    critical_total = sum(
        1 for rid in (state.completed_request_ids + [r.request_id for r in state.pending_requests])
        if any(r.request_id == rid and PRIORITY_WEIGHT[r.priority] >= 3
               for r in state.pending_requests)
    )
    # simpler: fraction of high-priority reqs still pending (bad)
    high_pending_penalty = len(high_reqs) * 0.15
    priority_score = max(0.0, 1.0 - high_pending_penalty)

    hours_ok = 1.0 if _all_within_hours(evs) else 0.6

    score = (
        0.50 * completion
        + 0.25 * no_overlap
        + 0.15 * priority_score
        + 0.10 * hours_ok
    )
    return round(min(1.0, max(0.0, score)), 4)


def grade_hard(state: EnvironmentState) -> float:
    """
    Hard task: 8 meetings — completion + priority + efficiency + gap minimisation.

    Score breakdown:
      - 40% completion ratio
      - 20% no overlaps
      - 20% high/critical meetings scheduled first (all non-pending)
      - 10% working hours
      - 10% gap efficiency (less idle time = higher score)
    """
    evs = state.scheduled_events

    completion = _completion_ratio(state)

    no_overlap = _overlap_score(evs)

    # Priority: count critical/high requests still pending
    high_still_pending = sum(
        1 for r in state.pending_requests if PRIORITY_WEIGHT[r.priority] >= 3
    )
    priority_score = max(0.0, 1.0 - high_still_pending * 0.25)

    hours_ok = 1.0 if _all_within_hours(evs) else 0.5

    # Gap efficiency
    if evs:
        sorted_evs = sorted(evs, key=lambda e: e.start_minutes())
        total_meeting_minutes = sum(e.duration_minutes() for e in evs)
        working_minutes = WORKING_END - WORKING_START  # 540
        # Ideal: meetings back-to-back, minimal gaps
        scheduled_span = (
            sorted_evs[-1].end_minutes() - sorted_evs[0].start_minutes()
            if evs else 0
        )
        gaps = max(0, scheduled_span - total_meeting_minutes)
        gap_ratio = gaps / working_minutes
        gap_score = max(0.0, 1.0 - gap_ratio)
    else:
        gap_score = 0.0

    score = (
        0.40 * completion
        + 0.20 * no_overlap
        + 0.20 * priority_score
        + 0.10 * hours_ok
        + 0.10 * gap_score
    )
    return round(min(1.0, max(0.0, score)), 4)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

GRADERS: Dict[str, Callable[[EnvironmentState], float]] = {
    "easy": grade_easy,
    "medium": grade_medium,
    "hard": grade_hard,
}


def grade(state: EnvironmentState) -> float:
    """Grade the current state for the active task_id."""
    grader = GRADERS.get(state.task_id)
    if grader is None:
        raise ValueError(f"No grader for task_id {state.task_id!r}")
    return grader(state)
