"""
Smart Meeting Scheduler — core environment.

Implements a Gymnasium-style reset() / step() interface compatible with
the OpenEnv standard.
"""
from __future__ import annotations

import json
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple

from models import (
    Action, CalendarEvent, CancelEvent, CreateEvent, Done,
    EnvironmentState, FreeSlot, MeetingRequest, Observation, Priority,
    QueryFreeSlots, RescheduleEvent, PRIORITY_WEIGHT,
)


# ---------------------------------------------------------------------------
# Task definitions
# ---------------------------------------------------------------------------

WORKING_START = 9 * 60   # 09:00 in minutes
WORKING_END = 18 * 60    # 18:00 in minutes


def _t(hhmm: str) -> int:
    """Convert HH:MM to minutes since midnight."""
    h, m = hhmm.split(":")
    return int(h) * 60 + int(m)


def _fmt(minutes: int) -> str:
    """Convert minutes since midnight to HH:MM."""
    return f"{minutes // 60:02d}:{minutes % 60:02d}"


TASKS: Dict[str, Dict[str, Any]] = {
    "easy": {
        "description": (
            "Schedule 3 simple meetings with clear, non-overlapping time windows. "
            "No conflicts expected if the agent places them correctly."
        ),
        "max_steps": 20,
        "requests": [
            MeetingRequest(
                request_id="r1", title="Team Standup", duration_minutes=30,
                priority=Priority.HIGH, preferred_start="09:00",
                attendees=["alice", "bob"], flexible=False,
            ),
            MeetingRequest(
                request_id="r2", title="Product Review", duration_minutes=60,
                priority=Priority.MEDIUM, preferred_start="11:00",
                attendees=["carol"], flexible=True,
            ),
            MeetingRequest(
                request_id="r3", title="1-on-1 with Manager", duration_minutes=30,
                priority=Priority.HIGH, preferred_start="15:00",
                attendees=["dave"], flexible=True,
            ),
        ],
    },
    "medium": {
        "description": (
            "Schedule 5 meetings with mixed priorities. Some preferred windows overlap, "
            "so the agent must resolve conflicts by priority or rescheduling."
        ),
        "max_steps": 25,
        "requests": [
            MeetingRequest(
                request_id="r1", title="Executive Briefing", duration_minutes=60,
                priority=Priority.CRITICAL, preferred_start="10:00",
                attendees=["ceo", "cto"], flexible=False,
            ),
            MeetingRequest(
                request_id="r2", title="Design Sync", duration_minutes=45,
                priority=Priority.MEDIUM, preferred_start="10:30",
                attendees=["ux_team"], flexible=True,
            ),
            MeetingRequest(
                request_id="r3", title="Sprint Planning", duration_minutes=90,
                priority=Priority.HIGH, preferred_start="13:00",
                attendees=["dev_team"], flexible=True,
            ),
            MeetingRequest(
                request_id="r4", title="Budget Review", duration_minutes=60,
                priority=Priority.HIGH, preferred_start="14:00",
                attendees=["finance"], flexible=True,
            ),
            MeetingRequest(
                request_id="r5", title="Casual Coffee Chat", duration_minutes=30,
                priority=Priority.LOW, preferred_start="16:00",
                attendees=["eve"], flexible=True,
            ),
        ],
    },
    "hard": {
        "description": (
            "Schedule 8 meetings with tight time windows. High-priority meetings must "
            "be placed first; minimize gaps in the workday for maximum efficiency."
        ),
        "max_steps": 35,
        "requests": [
            MeetingRequest(
                request_id="r1", title="Board Meeting", duration_minutes=120,
                priority=Priority.CRITICAL, preferred_start="09:00",
                attendees=["board"], flexible=False,
            ),
            MeetingRequest(
                request_id="r2", title="All-Hands", duration_minutes=60,
                priority=Priority.CRITICAL, preferred_start="11:00",
                attendees=["all_staff"], flexible=False,
            ),
            MeetingRequest(
                request_id="r3", title="Investor Call", duration_minutes=45,
                priority=Priority.HIGH, preferred_start="13:00",
                attendees=["investors"], flexible=True,
            ),
            MeetingRequest(
                request_id="r4", title="Product Roadmap", duration_minutes=60,
                priority=Priority.HIGH, preferred_start="14:00",
                attendees=["pm_team"], flexible=True,
            ),
            MeetingRequest(
                request_id="r5", title="Engineering Review", duration_minutes=45,
                priority=Priority.HIGH, preferred_start="14:30",
                attendees=["eng_leads"], flexible=True,
            ),
            MeetingRequest(
                request_id="r6", title="Marketing Sync", duration_minutes=30,
                priority=Priority.MEDIUM, preferred_start="15:30",
                attendees=["marketing"], flexible=True,
            ),
            MeetingRequest(
                request_id="r7", title="Hiring Interviews", duration_minutes=60,
                priority=Priority.MEDIUM, preferred_start="16:00",
                attendees=["hr", "candidates"], flexible=True,
            ),
            MeetingRequest(
                request_id="r8", title="Team Retrospective", duration_minutes=45,
                priority=Priority.LOW, preferred_start="17:00",
                attendees=["dev_team"], flexible=True,
            ),
        ],
    },
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _compute_free_slots(
    events: List[CalendarEvent],
    min_duration: int = 0,
) -> List[FreeSlot]:
    """Return free time windows within working hours."""
    sorted_events = sorted(events, key=lambda e: e.start_minutes())
    slots: List[FreeSlot] = []
    cursor = WORKING_START

    for ev in sorted_events:
        s = ev.start_minutes()
        if s > cursor:
            dur = s - cursor
            if dur >= min_duration:
                slots.append(FreeSlot(
                    start_time=_fmt(cursor),
                    end_time=_fmt(s),
                    duration_minutes=dur,
                ))
        cursor = max(cursor, ev.end_minutes())

    # Trailing slot
    if cursor < WORKING_END:
        dur = WORKING_END - cursor
        if dur >= min_duration:
            slots.append(FreeSlot(
                start_time=_fmt(cursor),
                end_time=_fmt(WORKING_END),
                duration_minutes=dur,
            ))
    return slots


def _has_conflict(new_event: CalendarEvent, existing: List[CalendarEvent]) -> bool:
    return any(new_event.overlaps(ev) for ev in existing)


def _within_working_hours(event: CalendarEvent) -> bool:
    return (
        event.start_minutes() >= WORKING_START
        and event.end_minutes() <= WORKING_END
        and event.start_minutes() < event.end_minutes()
    )


# ---------------------------------------------------------------------------
# Environment class
# ---------------------------------------------------------------------------

class MeetingSchedulerEnv:
    """
    OpenEnv-compatible Meeting Scheduler environment.
    """

    def __init__(self) -> None:
        self.state: EnvironmentState = EnvironmentState(task_id="")

    # ------------------------------------------------------------------
    def reset(self, task_id: str) -> Observation:
        task = TASKS.get(task_id)
        if task is None:
            raise ValueError(f"Unknown task_id {task_id!r}. Choose from {list(TASKS)}")

        requests = deepcopy(task["requests"])
        self.state = EnvironmentState(
            task_id=task_id,
            pending_requests=requests,
            max_steps=task["max_steps"],
        )
        return self._make_observation()

    # ------------------------------------------------------------------
    def step(self, action: Action) -> Tuple[Observation, float, bool, Dict[str, Any]]:
        """
        Returns (observation, reward, done, info).
        """
        if self.state.done:
            obs = self._make_observation()
            return obs, 0.0, True, {"error": "Episode already done."}

        self.state.step_number += 1
        reward = 0.0
        info: Dict[str, Any] = {}

        # ---- Dispatch ------------------------------------------------
        act_type = action.action_type  # type: ignore[union-attr]

        if act_type == "create_event":
            reward, info = self._handle_create(action)  # type: ignore[arg-type]

        elif act_type == "cancel_event":
            reward, info = self._handle_cancel(action)  # type: ignore[arg-type]

        elif act_type == "reschedule_event":
            reward, info = self._handle_reschedule(action)  # type: ignore[arg-type]

        elif act_type == "query_free_slots":
            reward, info = self._handle_query(action)  # type: ignore[arg-type]

        elif act_type == "done":
            reward, info = self._handle_done()
            self.state.done = True

        else:
            reward = -1.0
            info = {"error": f"Unknown action type: {act_type}"}

        # ---- Step limit ----------------------------------------------
        if self.state.step_number >= self.state.max_steps and not self.state.done:
            self.state.done = True
            info["timeout"] = True

        self.state.total_reward += reward
        self.state.action_log.append({"step": self.state.step_number, "action": act_type, "reward": reward})

        obs = self._make_observation(last_result=info.get("message", str(info)))
        return obs, reward, self.state.done, info

    # ------------------------------------------------------------------
    # Action handlers
    # ------------------------------------------------------------------

    def _handle_create(self, action: CreateEvent) -> Tuple[float, Dict]:
        # Build candidate event
        try:
            candidate = CalendarEvent(
                title=action.title,
                start_time=action.start_time,
                end_time=action.end_time,
                priority=action.priority,
                attendees=action.attendees,
            )
        except Exception as exc:
            return -1.0, {"error": str(exc), "message": f"Invalid event parameters: {exc}"}

        # Working hours check
        if not _within_working_hours(candidate):
            return -1.0, {"message": "⚠ Scheduled outside working hours (09:00–18:00). -1 penalty."}

        # Conflict check
        if _has_conflict(candidate, self.state.scheduled_events):
            return -5.0, {"message": f"❌ Conflict detected for '{action.title}'. -5 penalty."}

        # Accept
        self.state.scheduled_events.append(candidate)

        # Reward: base + priority bonus
        priority_bonus = PRIORITY_WEIGHT[candidate.priority]
        reward = 2.0 + priority_bonus

        # Preferred time adherence bonus
        linked_req = next(
            (r for r in self.state.pending_requests if r.request_id == action.request_id),
            None,
        )
        if linked_req:
            if linked_req.preferred_start and candidate.start_time == linked_req.preferred_start:
                reward += 1.0  # exact preferred time
            self.state.pending_requests.remove(linked_req)
            self.state.completed_request_ids.append(linked_req.request_id)

        msg = (
            f"✅ Created '{candidate.title}' ({candidate.start_time}–{candidate.end_time}). "
            f"Reward +{reward:.1f}"
        )
        return reward, {"message": msg, "event_id": candidate.event_id}

    def _handle_cancel(self, action: CancelEvent) -> Tuple[float, Dict]:
        target = next((e for e in self.state.scheduled_events if e.event_id == action.event_id), None)
        if target is None:
            return -0.5, {"message": f"Event ID {action.event_id!r} not found."}

        self.state.scheduled_events.remove(target)
        self.state.cancelled_event_ids.append(action.event_id)
        # Cancelling a high-priority event incurs a penalty
        penalty = -PRIORITY_WEIGHT[target.priority] * 0.5
        msg = f"🗑 Cancelled '{target.title}'. Reward {penalty:.1f}"
        return penalty, {"message": msg}

    def _handle_reschedule(self, action: RescheduleEvent) -> Tuple[float, Dict]:
        target = next((e for e in self.state.scheduled_events if e.event_id == action.event_id), None)
        if target is None:
            return -0.5, {"message": f"Event ID {action.event_id!r} not found."}

        if not target.flexible:
            return -2.0, {"message": f"'{target.title}' is not flexible — cannot reschedule. -2 penalty."}

        try:
            updated = target.model_copy(
                update={"start_time": action.new_start_time, "end_time": action.new_end_time}
            )
        except Exception as exc:
            return -1.0, {"error": str(exc), "message": str(exc)}

        if not _within_working_hours(updated):
            return -1.0, {"message": "⚠ New slot outside working hours."}

        others = [e for e in self.state.scheduled_events if e.event_id != target.event_id]
        if _has_conflict(updated, others):
            return -5.0, {"message": f"❌ Conflict on reschedule of '{target.title}'. -5 penalty."}

        self.state.scheduled_events.remove(target)
        self.state.scheduled_events.append(updated)
        return 1.0, {"message": f"🔄 Rescheduled '{target.title}' to {action.new_start_time}–{action.new_end_time}. +1"}

    def _handle_query(self, action: QueryFreeSlots) -> Tuple[float, Dict]:
        slots = _compute_free_slots(self.state.scheduled_events, action.min_duration_minutes)
        msg = f"🔍 {len(slots)} free slot(s) of ≥{action.min_duration_minutes} min found."
        return 0.0, {"message": msg, "free_slots": [s.model_dump() for s in slots]}

    def _handle_done(self) -> Tuple[float, Dict]:
        """Final reward calculation when agent calls done."""
        n_pending = len(self.state.pending_requests)
        n_total = n_pending + len(self.state.completed_request_ids)
        n_scheduled = len(self.state.scheduled_events)

        # Overlap penalty (shouldn't happen but double-check)
        overlap_penalty = 0.0
        evs = self.state.scheduled_events
        for i, a in enumerate(evs):
            for b in evs[i + 1:]:
                if a.overlaps(b):
                    overlap_penalty += 5.0

        # Bonus: all requests scheduled
        completion_bonus = 10.0 if n_pending == 0 else 0.0

        # Efficiency bonus: fewer steps = more bonus (max 3)
        steps_used = self.state.step_number
        steps_limit = self.state.max_steps
        efficiency_bonus = max(0.0, 3.0 * (1 - steps_used / steps_limit))

        # Gap penalty for hard task (encourage dense packing)
        gap_penalty = 0.0
        if self.state.task_id == "hard" and evs:
            slots = _compute_free_slots(evs, 0)
            total_gap = sum(s.duration_minutes for s in slots)
            gap_penalty = total_gap / 60.0 * 0.1  # 0.1 per hour of gaps

        reward = completion_bonus + efficiency_bonus - overlap_penalty - gap_penalty
        msg = (
            f"📋 Done! Scheduled {n_scheduled}/{n_total} meetings. "
            f"Completion bonus: +{completion_bonus}, Efficiency: +{efficiency_bonus:.2f}, "
            f"Overlap penalty: -{overlap_penalty}, Gap penalty: -{gap_penalty:.2f}. "
            f"Final step reward: {reward:.2f}"
        )
        return reward, {"message": msg, "scheduled": n_scheduled, "total": n_total}

    # ------------------------------------------------------------------
    def _make_observation(self, last_result: str = "") -> Observation:
        free = _compute_free_slots(self.state.scheduled_events, 0)
        return Observation(
            scheduled_events=list(self.state.scheduled_events),
            pending_requests=list(self.state.pending_requests),
            current_time=_fmt(
                min(WORKING_START + self.state.step_number * 10, WORKING_END)
            ),
            working_hours_start="09:00",
            working_hours_end="18:00",
            step_number=self.state.step_number,
            max_steps=self.state.max_steps,
            free_slots=free,
            last_action_result=last_result,
            total_reward_so_far=self.state.total_reward,
        )

    # ------------------------------------------------------------------
    def get_state(self) -> EnvironmentState:
        return deepcopy(self.state)
