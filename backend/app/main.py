"""FastAPI server the Tauri UI talks to. Run: uvicorn app.main:app --port 8791"""

from __future__ import annotations

from fastapi import FastAPI, HTTPException

from .config import load_config
from .dataset_manager import DatasetManager
from .orchestrator import Orchestrator
from .teacher_client import BudgetLedger

app = FastAPI(title="Distill", version="0.1.0")

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
    return _get_orchestrator().snapshot()


@app.post("/go-local")
def go_local():
    try:
        _get_orchestrator().go_local()
    except RuntimeError as e:
        raise HTTPException(400, str(e)) from e
    return _get_orchestrator().snapshot()
