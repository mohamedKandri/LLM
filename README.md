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
    trainer.py            # LoRA fine-tune (transformers/peft/trl), CPU-only, GGUF export via llama.cpp (optional)
    benchmark.py          # student vs active teacher on the gold set, per-tier history, cached teacher answers
    orchestrator.py       # background loop (generate->answer->evaluate->save->retrain hook), start/pause/stop, graduate()/go_local()
    main.py               # FastAPI server for the Tauri UI (/run/start,pause,resume,stop, /graduate, /go-local)
  tests/                  # no-network unit tests (61 passing; a couple mock the network to test retry behavior)
  requirements.txt
data/
  seeds.jsonl             # human-written seed tasks feeding prompt_generator
  gold_set/gold.jsonl     # benchmark set (20 entries, AI-drafted — needs your review, see its README) — NEVER trains the model
  checkpoints/            # LoRA adapter checkpoints, one per generation (gitignored)
frontend/                 # Tauri + React (scaffolded after backend pipeline works)
```

## Quick start (backend)

```powershell
py -3.12 -m venv backend\.venv
backend\.venv\Scripts\python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
backend\.venv\Scripts\python -m pip install -r backend\requirements.txt
$env:OPENROUTER_API_KEY = "sk-or-..."
cd backend
.venv\Scripts\python -m uvicorn app.main:app --port 8791
```

Tests: `cd backend; .venv\Scripts\python -m pytest tests -q`
(Trainer unit tests don't need torch/transformers installed — those heavy
imports are deferred into `Trainer.train()`, only exercised live via
`scripts/smoke_test_trainer.py`.)

## Training on CPU — set expectations

This dev machine has no discrete GPU, so `trainer.py` runs LoRA
fine-tuning on CPU via plain transformers+peft+trl (not Unsloth, which
requires CUDA), against `Qwen2.5-0.5B-Instruct`. It works — verified live
via `scripts/smoke_test_trainer.py`, real adapter weights get written —
but it's **slow**: ~3 minutes per optimizer step even on this small
model. A real retrain at the default `retrain_every_n_accepted: 200`
(2 epochs) is on the order of tens of minutes to a few hours, not
seconds. If that's too slow in practice, lower `retrain_every_n_accepted`
or `student.train.epochs` in `config.yaml`, or move training to a GPU
machine later (only `student.base_model` needs to change — the code
doesn't hardcode model size).

## Safety rails

- **Budget cap enforced in code**: `BudgetLedger` (SQLite) checks USD +
  call totals before every API call; hitting the cap raises, persists
  across restarts.
- **Gold set never enters training data** — `dataset_manager.add()` checks every
  prompt against the gold set before accepting it.
- Every API call and training example is tagged with the tier
  (`free`/`paid`) active when it was created — full per-phase audit log.
