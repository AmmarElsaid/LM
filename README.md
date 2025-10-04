# LearnMate — Starter Kit (MVP)

This starter kit gives you a minimal, working prototype for **LearnMate** with a simple multi-agent workflow:
**Planner → Tutor → Web Research → Validator → UI (Streamlit)**

## What you can demo
- Type a question like: *"Explain recursion in Python and find a recent tutorial."*
- The app will:
  1) Plan the subtasks,
  2) Generate an explanation + code (using OpenAI if available, or the offline dataset),
  3) Search the web (using DuckDuckGo Search library) and summarize links,
  4) Merge everything and show results in the UI.
- Every run logs **entropy metrics** to `logs/entropy.jsonl` (prompts used, timings, failures).

## Quickstart

### 1) Install Python packages
```bash
python -m venv .venv
# Windows PowerShell:
.\.venv\Scripts\activate
# macOS/Linux:
# source .venv/bin/activate

pip install -r requirements.txt
```

### 2) (Optional) Add API keys
Create a `.env` file (copy `.env.example`):
```
OPENAI_API_KEY=sk-...
```
> If no key is provided, the system falls back to the offline dataset for explanations and uses DuckDuckGo search for web supplementation.

### 3) Run the app
```bash
streamlit run app.py
```

Open the local URL shown by Streamlit.

## Files
- `app.py` — Streamlit UI.
- `agents.py` — Planner, Tutor, Web, Validator agents + Orchestrator.
- `utils/logging_utils.py` — Entropy logger.
- `data/seed_lessons.csv` — Offline mini-knowledge base.
- `requirements.txt` — Dependencies.
- `.env.example` — API key template.
- `logs/entropy.jsonl` — Run-time entropy metrics.

## How the agents work

**Planner Agent**  
Splits the user prompt into concrete subtasks (explain → code → find tutorial).

**Tutor Agent**  
- If `OPENAI_API_KEY` exists: asks GPT for explanation + code.  
- Otherwise: falls back to `data/seed_lessons.csv` and returns a best-match explanation and code.

**Web Agent**  
Uses `duckduckgo-search` to find 1–3 relevant links and short summaries. If offline, it returns helpful hints instead.

**Validator Agent**  
Checks that we have an explanation, a runnable code block, and at least one resource link; merges into a single lesson payload.

**Orchestrator**  
Calls the agents in order, measures timings, counts prompts, and logs to `logs/entropy.jsonl`.

## Entropy metrics
Each run appends a JSON line similar to:
```json
{
  "feature": "lesson_pipeline",
  "prompt": "Explain recursion in Python and find a recent tutorial.",
  "prompts_used": 1,
  "timing_sec": {"planner": 0.01, "tutor": 1.23, "web": 0.76, "validator": 0.02},
  "agent_failures": 0,
  "completed": true,
  "timestamp": "2025-08-21T00:00:00Z"
}
```

## Suggested team roles
- **Lead**: Run meetings, keep the scope focused on MVP, review logs.
- **Backend 1**: Tutor Agent & offline dataset.
- **Backend 2**: Planner + Validator + Orchestrator.
- **Frontend**: Streamlit UI.
- **Research**: Curate dataset topics, test coverage.

## Next steps (after MVP)
- Add code execution (Pyodide or a local Python sandbox) to run examples safely in-browser.
- Expand dataset; switch to a vector store (Chroma/FAISS) for better offline retrieval.
- Add settings (e.g., difficulty, tone, preferred sources).
- Track richer entropy metrics: retries per agent, error types, token usage.
