"""
FastAPI server for the Smart Meeting Scheduler OpenEnv environment.

Endpoints:
  GET  /                  → health check / metadata
  POST /reset             → reset environment to a task
  POST /step              → take an action
  GET  /state             → introspect current state
  POST /grade             → compute final score
  GET  /tasks             → list available tasks
"""
from __future__ import annotations

import traceback
from typing import Any, Dict, List, Optional, Union

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from environment import MeetingSchedulerEnv, TASKS
from graders import grade
from models import (
    Action, CancelEvent, CreateEvent, Done,
    Observation, QueryFreeSlots, RescheduleEvent,
)

app = FastAPI(
    title="Smart Meeting Scheduler",
    description=(
        "An OpenEnv-compatible environment where an AI agent acts as a "
        "personal calendar assistant, scheduling, rescheduling, and cancelling "
        "meetings within working hours (09:00–18:00) without conflicts."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global environment instance (single-session for HF Space demo)
_env = MeetingSchedulerEnv()


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------

class ResetRequest(BaseModel):
    task_id: str  # "easy" | "medium" | "hard"


class StepRequest(BaseModel):
    action_type: str
    # CreateEvent fields
    title: Optional[str] = None
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    priority: Optional[str] = None
    attendees: Optional[List[str]] = None
    request_id: Optional[str] = None
    # CancelEvent
    event_id: Optional[str] = None
    # RescheduleEvent
    new_start_time: Optional[str] = None
    new_end_time: Optional[str] = None
    # QueryFreeSlots
    min_duration_minutes: Optional[int] = 30
    # Done
    message: Optional[str] = "Schedule complete."


class StepResponse(BaseModel):
    observation: Dict[str, Any]
    reward: float
    done: bool
    info: Dict[str, Any]


class GradeResponse(BaseModel):
    task_id: str
    score: float
    total_reward: float
    steps_used: int
    scheduled_count: int
    pending_count: int


# ---------------------------------------------------------------------------
# Helper: parse action from flat StepRequest
# ---------------------------------------------------------------------------

def _parse_action(req: StepRequest) -> Action:
    at = req.action_type
    if at == "create_event":
        return CreateEvent(
            title=req.title or "Untitled",
            start_time=req.start_time or "09:00",
            end_time=req.end_time or "10:00",
            priority=req.priority or "medium",
            attendees=req.attendees or [],
            request_id=req.request_id,
        )
    elif at == "cancel_event":
        return CancelEvent(event_id=req.event_id or "")
    elif at == "reschedule_event":
        return RescheduleEvent(
            event_id=req.event_id or "",
            new_start_time=req.new_start_time or "09:00",
            new_end_time=req.new_end_time or "10:00",
        )
    elif at == "query_free_slots":
        return QueryFreeSlots(min_duration_minutes=req.min_duration_minutes or 30)
    elif at == "done":
        return Done(message=req.message or "Schedule complete.")
    else:
        raise ValueError(f"Unknown action_type: {at!r}")


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/")
def root() -> Dict[str, Any]:
    return {
        "name": "Smart Meeting Scheduler",
        "version": "1.0.0",
        "status": "ok",
        "tasks": list(TASKS.keys()),
        "description": (
            "AI calendar assistant environment. The agent receives meeting "
            "requests and must schedule them without conflicts, within "
            "working hours (09:00–18:00), prioritising by urgency."
        ),
    }


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.get("/tasks")
def list_tasks() -> Dict[str, Any]:
    return {
        task_id: {
            "description": task["description"],
            "num_requests": len(task["requests"]),
            "max_steps": task["max_steps"],
        }
        for task_id, task in TASKS.items()
    }


@app.post("/reset")
def reset(req: ResetRequest) -> Dict[str, Any]:
    if req.task_id not in TASKS:
        raise HTTPException(status_code=400, detail=f"Unknown task_id. Choose from {list(TASKS)}")
    obs = _env.reset(req.task_id)
    return obs.model_dump()


@app.post("/step")
def step(req: StepRequest) -> StepResponse:
    try:
        action = _parse_action(req)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    try:
        obs, reward, done, info = _env.step(action)
    except Exception as exc:
        tb = traceback.format_exc()
        raise HTTPException(status_code=500, detail=f"{exc}\n{tb}")

    return StepResponse(
        observation=obs.model_dump(),
        reward=reward,
        done=done,
        info=info,
    )


@app.get("/state")
def get_state() -> Dict[str, Any]:
    return _env.get_state().model_dump()


@app.post("/grade")
def grade_current() -> GradeResponse:
    state = _env.get_state()
    if not state.task_id:
        raise HTTPException(status_code=400, detail="No active task. Call /reset first.")
    score = grade(state)
    return GradeResponse(
        task_id=state.task_id,
        score=score,
        total_reward=state.total_reward,
        steps_used=state.step_number,
        scheduled_count=len(state.scheduled_events),
        pending_count=len(state.pending_requests),
    )


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=7860, reload=False)
