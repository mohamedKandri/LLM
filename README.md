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
    dataset_manager.py    # stub — SQLite store, dedup, JSONL export
    trainer.py            # stub — LoRA fine-tune (Unsloth/peft/trl)
    benchmark.py          # stub — gold-set eval, per-tier score history
    orchestrator.py       # stub — main loop + graduate()/go_local()
    main.py               # FastAPI server for the Tauri UI
  tests/                  # no-network unit tests (14 passing)
  requirements.txt
data/
  seeds.jsonl             # human-written seed tasks feeding prompt_generator
  gold_set/               # human-verified benchmark — NEVER trains the model
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
- **Gold set never enters training data.**
- Every API call and training example is tagged with the tier
  (`free`/`paid`) active when it was created — full per-phase audit log.
