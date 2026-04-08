"""
inference.py — Smart Meeting Scheduler
=======================================
Hackathon-required inference script.
Runs all 3 tasks against the environment using Groq (Llama) as the agent.
Produces exact log format: [START], [STEP], [END].

Usage:
    python inference.py [--task easy|medium|hard|all] [--base-url http://localhost:7860]

Requirements:
    pip install groq requests
    export GROQ_API_KEY=your_key_here
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from typing import Any, Dict, List

import requests
from groq import Groq

BASE_URL = "http://localhost:7860"
MODEL = "llama-3.3-70b-versatile"   

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



def api_reset(task_id: str, base_url: str) -> Dict[str, Any]:
    r = requests.post(f"{base_url}/reset", json={"task_id": task_id})
    r.raise_for_status()
    return r.json()


def api_step(action: Dict[str, Any], base_url: str) -> Dict[str, Any]:
    r = requests.post(f"{base_url}/step", json=action)
    r.raise_for_status()
    return r.json()


def api_grade(base_url: str) -> Dict[str, Any]:
    r = requests.post(f"{base_url}/grade")
    r.raise_for_status()
    return r.json()




def run_task(task_id: str, base_url: str) -> Dict[str, Any]:
    """Run a single task and return results."""
    client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

    print(f"\n[START] task_id={task_id} model={MODEL} timestamp={int(time.time())}")
    sys.stdout.flush()

    # Reset
    obs = api_reset(task_id, base_url)
    print(f"[START] observation={json.dumps(obs, separators=(',', ':'))}")
    sys.stdout.flush()

    conversation: List[Dict[str, str]] = []
    step_num = 0
    total_reward = 0.0
    done = False

    while not done:
        step_num += 1

       
        user_msg = (
            f"STEP {step_num}\n\n"
            f"OBSERVATION:\n{json.dumps(obs, indent=2)}\n\n"
            f"Pending requests to schedule: {len(obs.get('pending_requests', []))}\n"
            f"Scheduled events: {len(obs.get('scheduled_events', []))}\n"
            f"Free slots: {len(obs.get('free_slots', []))}\n\n"
            "Respond with your next action as a JSON object."
        )
        conversation.append({"role": "user", "content": user_msg})

        
        response = client.chat.completions.create(
            model=MODEL,
            max_tokens=512,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                *conversation,
            ],
        )
        raw_action = response.choices[0].message.content.strip()

        
        conversation.append({"role": "assistant", "content": raw_action})

        
        try:
            clean = re.sub(r"```(?:json)?|```", "", raw_action).strip()
            action = json.loads(clean)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", raw_action, re.DOTALL)
            if match:
                action = json.loads(match.group())
            else:
                action = {"action_type": "done", "message": "Parse error — stopping."}

        
        result = api_step(action, base_url)
        step_reward = result.get("reward", 0.0)
        total_reward += step_reward
        done = result.get("done", False)
        obs = result.get("observation", {})
        info = result.get("info", {})

        print(
            f"[STEP] step={step_num} "
            f"action={json.dumps(action, separators=(',', ':'))} "
            f"reward={step_reward:.3f} "
            f"done={done} "
            f"info={json.dumps(info, separators=(',', ':'))}"
        )
        sys.stdout.flush()

        if done:
            break

    
    grade_result = api_grade(base_url)
    final_score = grade_result.get("score", 0.0)

    print(
        f"[END] task_id={task_id} "
        f"final_score={final_score:.4f} "
        f"total_reward={total_reward:.3f} "
        f"steps={step_num} "
        f"grade={json.dumps(grade_result, separators=(',', ':'))}"
    )
    sys.stdout.flush()

    return {
        "task_id": task_id,
        "final_score": final_score,
        "total_reward": total_reward,
        "steps": step_num,
        **grade_result,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Smart Meeting Scheduler inference")
    parser.add_argument("--task", default="all", choices=["easy", "medium", "hard", "all"])
    parser.add_argument("--base-url", default=BASE_URL)
    args = parser.parse_args()

    if not os.environ.get("GROQ_API_KEY"):
        print("ERROR: GROQ_API_KEY environment variable not set.", file=sys.stderr)
        print("Get a free key at https://console.groq.com", file=sys.stderr)
        sys.exit(1)

   
    for attempt in range(30):
        try:
            r = requests.get(f"{args.base_url}/health", timeout=5)
            if r.status_code == 200:
                break
        except Exception:
            pass
        print(f"Waiting for server... ({attempt + 1}/30)")
        time.sleep(2)
    else:
        print("ERROR: Server did not start in time.", file=sys.stderr)
        sys.exit(1)

    tasks_to_run = ["easy", "medium", "hard"] if args.task == "all" else [args.task]
    results = []

    for task_id in tasks_to_run:
        try:
            result = run_task(task_id, args.base_url)
            results.append(result)
        except Exception as exc:
            print(f"[ERROR] task_id={task_id} error={exc}", file=sys.stderr)
            results.append({"task_id": task_id, "final_score": 0.0, "error": str(exc)})

    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for r in results:
        score = r.get("final_score", 0.0)
        steps = r.get("steps", "N/A")
        print(f"  {r['task_id']:8s}  score={score:.4f}  steps={steps}")

    avg = sum(r.get("final_score", 0.0) for r in results) / max(len(results), 1)
    print(f"\n  AVERAGE SCORE: {avg:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
