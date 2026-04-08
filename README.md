Smart Meeting Scheduler — OpenEnv Environment

> A realistic AI calendar assistant environment for the OpenEnv Hackathon.
> The agent schedules, reschedules, and cancels meetings across 3 difficulty levels
> while respecting working hours and avoiding conflicts.



To Run locally
bash
1. Clone the repo
git clone https://github.com/PurabRay/smart-meeting-scheduler
cd smart-meeting-scheduler

2. Install dependencies
pip install -r requirements.txt

3. Start the environment server
uvicorn main:app --host 0.0.0.0 --port 7860

4. Run inference (in another terminal)
export ANTHROPIC_API_KEY=your_key_here
python inference.py --task all --base-url http://localhost:7860

To Run with Docker
docker build -t smart-meeting-scheduler .
docker run -p 7860:7860 -e ANTHROPIC_API_KEY=your_key smart-meeting-scheduler

Environment Design

Core Concept

An AI agent acts as a **personal calendar assistant**. At each step it receives
an observation of the current calendar state and must choose one action to
manage meetings until all pending requests are handled.

### Working Hours
All meetings must fall within **09:00 – 18:00** (9 AM – 6 PM).

Tasks

| Task | Meetings | Difficulty | Max Steps | Key Challenge |
|------|----------|-----------|-----------|---------------|
| `easy` | 3 | ⭐ Easy | 20 | Place 3 non-conflicting meetings |
| `medium` | 5 | ⭐⭐ Medium | 25 | Resolve priority conflicts, mixed windows |
| `hard` | 8 | ⭐⭐⭐ Hard | 35 | Tight packing, critical-first, minimal gaps |

 Observation Space

{
  "scheduled_events": [
    { "event_id": "abc123", "title": "Team Standup",
      "start_time": "09:00", "end_time": "09:30",
      "priority": "high", "attendees": ["alice", "bob"] }
  ],
  "pending_requests": [
    { "request_id": "r2", "title": "Product Review",
      "duration_minutes": 60, "priority": "medium",
      "preferred_start": "11:00", "flexible": true }
  ],
  "free_slots": [
    { "start_time": "09:30", "end_time": "11:00", "duration_minutes": 90 }
  ],
  "current_time": "09:00",
  "working_hours_start": "09:00",
  "working_hours_end": "18:00",
  "step_number": 1,
  "max_steps": 20,
  "last_action_result": "✅ Created 'Team Standup' (09:00–09:30). Reward +5.0",
  "total_reward_so_far": 5.0
}
Action Space

// Create a meeting
{ "action_type": "create_event", "title": "Sprint Planning",
  "start_time": "13:00", "end_time": "14:30",
  "priority": "high", "attendees": ["dev_team"], "request_id": "r3" }

// Cancel a meeting
{ "action_type": "cancel_event", "event_id": "abc123" }

// Reschedule a meeting
{ "action_type": "reschedule_event", "event_id": "abc123",
  "new_start_time": "14:00", "new_end_time": "14:30" }

// Query free time
{ "action_type": "query_free_slots", "min_duration_minutes": 60 }

// Finish
{ "action_type": "done", "message": "All meetings scheduled." }


Reward Function
| Event | Reward |
|-------|--------|
| Schedule meeting (no conflict) | +2 + priority_weight (3–6 total) |
| Preferred time matched | +1 |
| All requests completed | **+10 bonus** |
| Efficiency (fewer steps) | up to +3 |
| Conflict / double-booking | **−5** |
| Outside working hours | −1 |
| Cancel meeting | −0.5 × priority_weight |
| Gap penalty (hard task) | −0.1 per idle hour |

Grader Logic

Each task returns a **score in [0.0 – 1.0]**:
Easy

score = 0.60 × completion + 0.30 × no_overlap + 0.10 × hours_ok

### Medium
score = 0.50 × completion + 0.25 × no_overlap
      + 0.15 × priority_adherence + 0.10 × hours_ok

### Hard
score = 0.40 × completion + 0.20 × no_overlap
      + 0.20 × priority_adherence + 0.10 × hours_ok
      + 0.10 × gap_efficiency

API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Metadata |
| `GET` | `/health` | Health check (used by HF Space ping) |
| `GET` | `/tasks` | List all 3 tasks |
| `POST` | `/reset` | Start a task `{ "task_id": "easy" }` |
| `POST` | `/step` | Execute an action |
| `GET` | `/state` | Inspect full environment state |
| `POST` | `/grade` | Get final score `{ "score": 0.95 }` |

Project Structure

smart-meeting-scheduler/
├── main.py            # FastAPI server (entry point)
├── environment.py     # reset() / step() logic
├── models.py          # Pydantic v2 typed models
├── graders.py         # Deterministic graders for all 3 tasks
├── inference.py       # Hackathon inference script ([START]/[STEP]/[END])
├── openenv.yaml       # OpenEnv configuration
├── requirements.txt   # Python dependencies
├── Dockerfile         # HF Spaces compatible Docker image
└── README.md          # This file

 Links

- Hugging Face Space: `https://huggingface.co/spaces/YOUR_USERNAME/smart-meeting-scheduler`
- GitHub Repo: `https://github.com/YOUR_USERNAME/smart-meeting-scheduler`
- OpenEnv Docs: `https://openenv.ai`
 License
