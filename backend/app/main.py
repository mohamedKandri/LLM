"""FastAPI server the Tauri UI talks to. Run: uvicorn app.main:app --port 8791"""

from __future__ import annotations

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from . import settings_store
from .benchmark import read_history
from .config import load_config
from .dataset_manager import DatasetManager
from .orchestrator import Orchestrator
from .teacher_client import BudgetLedger

app = FastAPI(title="Distill", version="0.1.0")

# The UI runs as either a Tauri webview or a plain Vite dev server
# (`npm run dev` in a browser) — both are separate origins from this
# API. Both talk to localhost only; nothing here is exposed externally.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Single long-lived orchestrator for the process, built lazily so
# importing this module (e.g. for /status in tests) never spins up a
# TeacherClient/PromptGenerator/DatasetManager unless a run is actually
# started or its live state is queried.
_orchestrator: Orchestrator | None = None


def _get_orchestrator() -> Orchestrator:
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = Orchestrator(load_config())
    return _orchestrator


def _reset_orchestrator() -> None:
    """Force the next access to rebuild from the (just-changed) config
    file — needed after graduate()/settings edits change which teacher
    model TeacherClient should be pointed at."""
    global _orchestrator
    _orchestrator = None


@app.get("/health")
def health():
    return {"ok": True}


@app.get("/status")
def status():
    """Dashboard payload: phase, tier, spend, call counts."""
    cfg = load_config()
    ledger = BudgetLedger(cfg.db_path)
    spent, calls = ledger.totals(cfg.teacher_tier, "teacher")
    judge_spent, judge_calls = ledger.totals(cfg.teacher_tier, "judge")
    return {
        "phase": cfg.phase,
        "teacher_tier": cfg.teacher_tier,
        "teacher_model": cfg.teacher.model,
        "local_only": cfg.local_only,
        "teacher_spend_usd": spent,
        "teacher_calls": calls,
        "judge_spend_usd": judge_spent,
        "judge_calls": judge_calls,
        "budget": {
            "max_usd": cfg.teacher.budget.max_usd,
            "max_calls": cfg.teacher.budget.max_calls,
        },
        "benchmark_target": cfg.benchmark_target,
        "dataset_counts": DatasetManager(cfg).counts(),
    }


@app.get("/run/status")
def run_status():
    """Live orchestrator state: running/paused, counters, spend."""
    return _get_orchestrator().snapshot()


@app.post("/run/start")
def run_start():
    _get_orchestrator().start()
    return _get_orchestrator().snapshot()


@app.post("/run/pause")
def run_pause():
    _get_orchestrator().pause()
    return _get_orchestrator().snapshot()


@app.post("/run/resume")
def run_resume():
    _get_orchestrator().resume()
    return _get_orchestrator().snapshot()


@app.post("/run/stop")
def run_stop():
    _get_orchestrator().stop()
    return _get_orchestrator().snapshot()


@app.post("/graduate")
def graduate():
    try:
        _get_orchestrator().graduate()
    except RuntimeError as e:
        raise HTTPException(400, str(e)) from e
    # New tier -> new teacher model; rebuild so the next snapshot (this
    # response, and every request after it) reflects it.
    _reset_orchestrator()
    return _get_orchestrator().snapshot()


@app.post("/go-local")
def go_local():
    try:
        _get_orchestrator().go_local()
    except RuntimeError as e:
        raise HTTPException(400, str(e)) from e
    return _get_orchestrator().snapshot()


# ---- Settings ------------------------------------------------------


class SettingsUpdate(BaseModel):
    teacher_model: str | None = None
    budget_max_usd: float | None = None
    budget_max_calls: int | None = None


class ApiKeyUpdate(BaseModel):
    key: str


@app.get("/settings")
def get_settings():
    return settings_store.get_settings(load_config())


@app.post("/settings")
def update_settings(body: SettingsUpdate):
    if _get_orchestrator().status.state == "running":
        raise HTTPException(400, "Stop or pause the run before changing settings")
    cfg = load_config()
    try:
        result = settings_store.update_settings(
            cfg,
            teacher_model=body.teacher_model,
            budget_max_usd=body.budget_max_usd,
            budget_max_calls=body.budget_max_calls,
        )
    except ValueError as e:
        raise HTTPException(400, str(e)) from e
    _reset_orchestrator()
    return result


@app.post("/settings/api-key")
def set_api_key(body: ApiKeyUpdate):
    try:
        settings_store.set_api_key(body.key)
    except ValueError as e:
        raise HTTPException(400, str(e)) from e
    return settings_store.get_settings(load_config())


@app.delete("/settings/api-key")
def remove_api_key():
    settings_store.remove_api_key()
    return settings_store.get_settings(load_config())


# ---- Seeds -----------------------------------------------------------


class SeedCreate(BaseModel):
    type: str
    prompt: str
    tests: str | None = None


@app.get("/seeds")
def list_seeds():
    return settings_store.list_seeds(load_config())


@app.post("/seeds")
def add_seed(body: SeedCreate):
    try:
        return settings_store.add_seed(load_config(), body.type, body.prompt, body.tests)
    except (ValueError, SyntaxError) as e:
        raise HTTPException(400, str(e)) from e


@app.delete("/seeds/{seed_id}")
def delete_seed(seed_id: str):
    if not settings_store.delete_seed(load_config(), seed_id):
        raise HTTPException(404, f"Seed {seed_id!r} not found")
    return {"deleted": seed_id}


# ---- Benchmark ---------------------------------------------------------


@app.get("/benchmark/history")
def benchmark_history(tier: str | None = None):
    cfg = load_config()
    return read_history(cfg.db_path, tier=tier)
