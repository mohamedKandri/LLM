# Distill

Self-improving local LLM trainer with a two-phase graduation system:
validate the pipeline against a **free** teacher (Phase 1), then scale
with a **paid** teacher (Phase 2) by flipping `teacher_tier` in
`config.yaml` — no code changes. Once the local model matches the paid
teacher on the gold set, "Go local" disables API calls entirely.

## Layout

```
config.yaml               # teacher_tier is THE phase switch; budgets, models, targets
backend/
  app/
    config.py             # typed config loader (single source of truth)
    teacher_client.py     # OpenRouter client: rate limit, hard budget cap (SQLite), retries
    evaluator.py          # code tasks: run tests; general tasks: LLM judge (RLAIF)
    prompt_generator.py   # seeds -> new tasks via teacher (self-instruct, validated JSONL)
    dataset_manager.py    # SQLite store, dedup (hashed trigrams), JSONL export, gold-set guard
    trainer.py            # stub — LoRA fine-tune (transformers/peft/trl, CPU-sized model)
    benchmark.py          # student vs active teacher on the gold set, per-tier history, cached teacher answers
    orchestrator.py       # background loop (generate->answer->evaluate->save->retrain hook), start/pause/stop, graduate()/go_local()
    main.py               # FastAPI server for the Tauri UI (/run/start,pause,resume,stop, /graduate, /go-local)
  tests/                  # no-network unit tests (53 passing; a couple mock the network to test retry behavior)
  requirements.txt
data/
  seeds.jsonl             # human-written seed tasks feeding prompt_generator
  gold_set/gold.jsonl     # benchmark set (20 entries, AI-drafted — needs your review, see its README) — NEVER trains the model
frontend/                 # Tauri + React (scaffolded after backend pipeline works)
```

## Quick start (backend)

```powershell
py -3.12 -m venv backend\.venv
backend\.venv\Scripts\python -m pip install -r backend\requirements.txt
$env:OPENROUTER_API_KEY = "sk-or-..."
cd backend
.venv\Scripts\python -m uvicorn app.main:app --port 8791
```

Tests: `cd backend; .venv\Scripts\python -m pytest tests -q`

## Safety rails

- **Budget cap enforced in code**: `BudgetLedger` (SQLite) checks USD +
  call totals before every API call; hitting the cap raises, persists
  across restarts.
- **Gold set never enters training data** — `dataset_manager.add()` checks every
  prompt against the gold set before accepting it.
- Every API call and training example is tagged with the tier
  (`free`/`paid`) active when it was created — full per-phase audit log.
