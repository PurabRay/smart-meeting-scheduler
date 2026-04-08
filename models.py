"""
Typed Pydantic v2 models for the Smart Meeting Scheduler environment.
Defines Observation, Action, State, and all sub-types.
"""
from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Union
from uuid import uuid4

from pydantic import BaseModel, Field, field_validator


class Priority(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


PRIORITY_WEIGHT: Dict[Priority, int] = {
    Priority.LOW: 1,
    Priority.MEDIUM: 2,
    Priority.HIGH: 3,
    Priority.CRITICAL: 4,
}


class CalendarEvent(BaseModel):
    """A scheduled event on the calendar."""
    event_id: str = Field(default_factory=lambda: str(uuid4())[:8])
    title: str
    start_time: str          #"HH:MM"  (24-hour,within working day)
    end_time: str            #"HH:MM"
    priority: Priority = Priority.MEDIUM
    attendees: List[str] = Field(default_factory=list)
    date: str = "2024-01-15"  #ISO date string;single-day env

    @field_validator("start_time", "end_time")
    @classmethod
    def validate_time_format(cls, v: str) -> str:
        parts = v.split(":")
        if len(parts) != 2:
            raise ValueError(f"Time must be HH:MM, got {v!r}")
        h, m = int(parts[0]), int(parts[1])
        if not (0 <= h <= 23 and 0 <= m <= 59):
            raise ValueError(f"Invalid time {v!r}")
        return v

    def start_minutes(self) -> int:
        h, m = self.start_time.split(":")
        return int(h) * 60 + int(m)

    def end_minutes(self) -> int:
        h, m = self.end_time.split(":")
        return int(h) * 60 + int(m)

    def duration_minutes(self) -> int:
        return self.end_minutes() - self.start_minutes()

    def overlaps(self, other: "CalendarEvent") -> bool:
        return self.start_minutes() < other.end_minutes() and self.end_minutes() > other.start_minutes()


class MeetingRequest(BaseModel):
    """An incoming meeting request that the agent must handle."""
    request_id: str = Field(default_factory=lambda: str(uuid4())[:8])
    title: str
    duration_minutes: int
    priority: Priority = Priority.MEDIUM
    preferred_start: Optional[str] = None   #"HH:MM" or None
    preferred_end: Optional[str] = None     #latest possible end "HH:MM"
    attendees: List[str] = Field(default_factory=list)
    flexible: bool = True                   


class FreeSlot(BaseModel):
    start_time: str
    end_time: str
    duration_minutes: int




class Observation(BaseModel):
    """Everything the agent can see at each step."""
    scheduled_events: List[CalendarEvent] = Field(default_factory=list)
    pending_requests: List[MeetingRequest] = Field(default_factory=list)
    current_time: str = "09:00"       #simulated clock
    working_hours_start: str = "09:00"
    working_hours_end: str = "18:00"
    step_number: int = 0
    max_steps: int = 30
    free_slots: List[FreeSlot] = Field(default_factory=list)
    last_action_result: str = ""
    total_reward_so_far: float = 0.0



class CreateEvent(BaseModel):
    action_type: Literal["create_event"] = "create_event"
    title: str
    start_time: str  
    end_time: str     
    priority: Priority = Priority.MEDIUM
    attendees: List[str] = Field(default_factory=list)
    request_id: Optional[str] = None   


class CancelEvent(BaseModel):
    action_type: Literal["cancel_event"] = "cancel_event"
    event_id: str


class RescheduleEvent(BaseModel):
    action_type: Literal["reschedule_event"] = "reschedule_event"
    event_id: str
    new_start_time: str
    new_end_time: str


class QueryFreeSlots(BaseModel):
    action_type: Literal["query_free_slots"] = "query_free_slots"
    min_duration_minutes: int = 30


class Done(BaseModel):
    action_type: Literal["done"] = "done"
    message: str = "Schedule complete."



Action = Union[CreateEvent, CancelEvent, RescheduleEvent, QueryFreeSlots, Done]




class EnvironmentState(BaseModel):
    task_id: str
    scheduled_events: List[CalendarEvent] = Field(default_factory=list)
    pending_requests: List[MeetingRequest] = Field(default_factory=list)
    completed_request_ids: List[str] = Field(default_factory=list)
    cancelled_event_ids: List[str] = Field(default_factory=list)
    step_number: int = 0
    max_steps: int = 30
    done: bool = False
    total_reward: float = 0.0
    working_hours_start: str = "09:00"
    working_hours_end: str = "18:00"
    action_log: List[Dict[str, Any]] = Field(default_factory=list)
