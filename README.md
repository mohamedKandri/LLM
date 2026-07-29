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
    local_inference.py    # loads a trained checkpoint (base + LoRA adapter) and generates real completions
    retrain_flow.py        # wires dataset export -> trainer -> local_inference -> benchmark into orchestrator's retrain hook
    benchmark.py          # student vs active teacher on the gold set, per-tier history, cached teacher answers
    orchestrator.py       # background loop (generate->answer->evaluate->save->retrain), start/pause/stop, graduate()/go_local()
    settings_store.py     # read/write for the Settings UI: teacher model + budget, API key (.env), seed CRUD
    main.py               # FastAPI server for the UI — /status, /run/*, /graduate, /go-local, /settings, /seeds, /benchmark/history
  tests/                  # no-network unit tests + FastAPI TestClient integration tests (100 passing)
  requirements.txt
data/
  seeds.jsonl             # human-written seed tasks feeding prompt_generator
  gold_set/gold.jsonl     # benchmark set (20 entries, AI-drafted — needs your review, see its README) — NEVER trains the model
  checkpoints/            # LoRA adapter checkpoints, one per generation (gitignored)
frontend/                 # Tauri + React + TypeScript desktop UI
  src/
    api.ts, types.ts      # typed client for the FastAPI backend
    components/           # Dashboard (run controls, spend, benchmark chart), Settings, SeedEditor
  src-tauri/               # Rust shell (native window, packaging)
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
(Trainer/local-inference unit tests don't need torch/transformers
installed — those heavy imports are deferred into `Trainer.train()` /
`LocalStudent.generate()`, only exercised live via
`scripts/smoke_test_trainer.py` and
`scripts/smoke_test_local_inference.py`.)

## Quick start (frontend)

Needs Node.js (already have it if `node -v` works) and Rust (for the
Tauri native shell — install via https://rustup.rs; needs MSVC Build
Tools on Windows, `Desktop development with C++` in the VS Installer).

```powershell
cd frontend
npm install
npm run dev          # browser dev server at http://localhost:1420, talks to :8791
# or, once Rust is installed:
npm run tauri dev    # native desktop window
```

The backend must be running separately (`uvicorn app.main:app --port 8791`,
see above) — the UI is just an HTTP client against it, whether run as a
browser tab or a Tauri window.

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

## The retrain loop, end to end

`orchestrator.py`'s retrain hook (fired every
`student.train.retrain_every_n_accepted` accepted examples) is wired to
`retrain_flow.make_retrain_fn()`, which does the full cycle for real:
`dataset_manager.export_jsonl()` -> `trainer.train()` -> load the new
checkpoint via `local_inference.LocalStudent` -> `benchmark.run()`
against the active teacher's gold-set answers. This runs synchronously
inside the orchestrator's own background thread — a retrain is an
intentional pause of generation while it's in progress, not something
that needs its own thread pool.

Verified live via `scripts/smoke_test_local_inference.py`: trained a
tiny checkpoint, reloaded it fresh, and asked it "What is the capital
of France?" — it answered "The capital of France is Paris." Real
weights, real generation, no API calls.

## Safety rails

- **Budget cap enforced in code**: `BudgetLedger` (SQLite) checks USD +
  call totals before every API call; hitting the cap raises, persists
  across restarts.
- **Gold set never enters training data** — `dataset_manager.add()` checks every
  prompt against the gold set before accepting it.
- Every API call and training example is tagged with the tier
  (`free`/`paid`) active when it was created — full per-phase audit log.
