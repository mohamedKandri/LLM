"""FastAPI server the Tauri UI talks to. Run: uvicorn app.main:app --port 8791"""

from __future__ import annotations

from fastapi import FastAPI, HTTPException

from .config import load_config
from .teacher_client import BudgetLedger

app = FastAPI(title="Distill", version="0.1.0")


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
    }


# --- to be wired to Orchestrator once it exists -------------------------
@app.post("/run/start")
def run_start():
    raise HTTPException(501, "Orchestrator not implemented yet")


@app.post("/run/stop")
def run_stop():
    raise HTTPException(501, "Orchestrator not implemented yet")


@app.post("/graduate")
def graduate():
    raise HTTPException(501, "Implemented with Orchestrator.graduate()")


@app.post("/go-local")
def go_local():
    raise HTTPException(501, "Implemented with Orchestrator.go_local()")
