"""
inference.py — Smart Meeting Scheduler
=======================================
Hackathon-required inference script.
Runs a task against the environment using OpenAI Client.
Produces exact log format: [START], [STEP], [END].

Usage:
    export HF_TOKEN=hf_token
    export API_BASE_URL=https://router.huggingface.co/v1
    export MODEL_NAME=Qwen/Qwen2.5-72B-Instruct
    export MY_ENV_V4_TASK=easy   # or medium / hard / all
    python inference.py
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
from typing import Any, Dict, List, Optional

import requests
from openai import OpenAI


API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME   = os.getenv("MODEL_NAME",   "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN     = os.getenv("HF_TOKEN")     or os.getenv("API_KEY")
TASK_NAME    = os.getenv("MY_ENV_V4_TASK",      "easy")   
BENCHMARK    = os.getenv("MY_ENV_V4_BENCHMARK",  "smart-meeting-scheduler")


BASE_URL = os.getenv("ENV_BASE_URL", "http://localhost:7860")


client = OpenAI(
    base_url=API_BASE_URL,
    api_key=HF_TOKEN or "no-key",   
)


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val  = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )

def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.2f} rewards={rewards_str}",
        flush=True,
    )


SYSTEM_PROMPT = """You are an expert personal calendar assistant AI agent.
Your job is to schedule meetings on a calendar within working hours (09:00–18:00)
without any overlaps or conflicts.

You receive an observation each step containing:
- scheduled_events: already placed meetings (event_id, title, start_time, end_time, priority)
- pending_requests: meetings still needing to be scheduled (request_id, title, duration_minutes, priority, preferred_start)
- free_slots: available time windows
- last_action_result: feedback from your previous action

You MUST respond with a single valid JSON action object. Available actions:

1. Schedule a meeting:
{"action_type": "create_event", "title": "...", "start_time": "HH:MM", "end_time": "HH:MM", "priority": "low|medium|high|critical", "attendees": ["..."], "request_id": "..."}

2. Cancel a meeting:
{"action_type": "cancel_event", "event_id": "..."}

3. Reschedule a meeting:
{"action_type": "reschedule_event", "event_id": "...", "new_start_time": "HH:MM", "new_end_time": "HH:MM"}

4. Query free time slots:
{"action_type": "query_free_slots", "min_duration_minutes": 30}

5. Finish scheduling:
{"action_type": "done", "message": "All meetings scheduled."}

Strategy:
- Schedule HIGH and CRITICAL priority meetings first.
- Always check free_slots before placing a meeting.
- Calculate end_time as start_time + duration_minutes.
- Call done when all pending_requests are scheduled or cannot be scheduled.
- Prefer the preferred_start time when possible, but find alternatives if blocked.
- NEVER schedule outside 09:00–18:00.
- NEVER overlap two meetings.

Respond ONLY with the JSON object. No explanations, no markdown fences, no extra text."""



def api_reset(task_id: str) -> Dict[str, Any]:
    r = requests.post(f"{BASE_URL}/reset", json={"task_id": task_id}, timeout=30)
    r.raise_for_status()
    return r.json()

def api_step(action: Dict[str, Any]) -> Dict[str, Any]:
    r = requests.post(f"{BASE_URL}/step", json=action, timeout=30)
    r.raise_for_status()
    return r.json()

def api_grade() -> Dict[str, Any]:
    r = requests.post(f"{BASE_URL}/grade", timeout=30)
    r.raise_for_status()
    return r.json()



def get_model_action(obs: Dict[str, Any], step: int, conversation: List[Dict]) -> Dict[str, Any]:
    user_msg = (
        f"STEP {step}\n\n"
        f"OBSERVATION:\n{json.dumps(obs, indent=2)}\n\n"
        f"Pending requests to schedule: {len(obs.get('pending_requests', []))}\n"
        f"Scheduled events: {len(obs.get('scheduled_events', []))}\n"
        f"Free slots: {len(obs.get('free_slots', []))}\n\n"
        "Respond with your next action as a JSON object."
    )
    # Keep only last 6 turns to stay within memory limits on 8GB machines
    trimmed = conversation[-6:] if len(conversation) > 6 else conversation
    conversation.append({"role": "user", "content": user_msg})

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            max_tokens=512,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                *trimmed,
                {"role": "user", "content": user_msg},
            ],
        )
        raw = (response.choices[0].message.content or "").strip()
    except Exception as exc:
        print(f"[DEBUG] Model call failed: {exc}", flush=True)
        raw = '{"action_type": "done", "message": "Model error."}'

    conversation.append({"role": "assistant", "content": raw})

    try:
        clean = re.sub(r"```(?:json)?|```", "", raw).strip()
        return json.loads(clean)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except Exception:
                pass
    return {"action_type": "done", "message": "Parse error — stopping."}



def run_task(task_id: str) -> None:
    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)

    obs = api_reset(task_id)

    conversation: List[Dict] = []
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    try:
        for step in range(1, 200):   
            action = get_model_action(obs, step, conversation)
            action_str = json.dumps(action, separators=(",", ":"))

            result     = api_step(action)
            reward     = float(result.get("reward", 0.0))
            done       = bool(result.get("done", False))
            obs        = result.get("observation", {})
            error_msg  = result.get("info", {}).get("error") or None

            rewards.append(reward)
            steps_taken = step

            log_step(step=step, action=action_str, reward=reward, done=done, error=error_msg)

            if done:
                break

        grade_result = api_grade()
        score   = float(grade_result.get("score", 0.0))
        score   = min(max(score, 0.0), 1.0)
        success = score > 0.0

    except Exception as exc:
        print(f"[DEBUG] run_task error: {exc}", flush=True)

    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)



def main() -> None:
    
    for attempt in range(30):
        try:
            r = requests.get(f"{BASE_URL}/health", timeout=5)
            if r.status_code == 200:
                break
        except Exception:
            pass
        print(f"Waiting for server {BASE_URL}... ({attempt + 1}/30)", flush=True)
        time.sleep(2)
    else:
        print("ERROR: Server did not start in time.", file=sys.stderr)
        sys.exit(1)

    tasks_to_run = ["easy", "medium", "hard"] if TASK_NAME == "all" else [TASK_NAME]
    for task_id in tasks_to_run:
        run_task(task_id)


if __name__ == "__main__":
    main()
