"""Main loop: generate -> teacher -> evaluate -> save -> (retrain -> benchmark).

Owns the SQLite-backed BudgetLedger, TeacherClient (teacher + judge),
Evaluator, PromptGenerator, and DatasetManager. Every artifact produced
during a run is tagged with whatever teacher_tier was active when that
artifact was created — reading cfg.teacher_tier fresh at run time, not
once at construction, so a graduate() mid-app-lifetime is picked up.

trainer.py and local inference don't exist yet, so retraining is an
optional hook (`retrain_fn: () -> None`, called every
student.train.retrain_every_n_accepted accepted examples): the
generate/evaluate/save loop runs standalone without it. Benchmark runs
are the retrain_fn's responsibility to trigger (it has the student
inference function this module doesn't) — wiring that in later requires
no changes here.

Runs in a background thread so start()/pause()/stop() can be called from
FastAPI request handlers without blocking them. Never crashes the
thread silently: any exception (BudgetExceededError, LocalOnlyModeError,
or otherwise) stops the loop cleanly and is recorded in status.last_error.
"""

from __future__ import annotations

import threading
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from .config import DistillConfig, save_config
from .dataset_manager import DatasetManager
from .evaluator import Evaluator
from .prompt_generator import PromptGenerator
from .teacher_client import BudgetExceededError, BudgetLedger, LocalOnlyModeError, TeacherClient

RETRAIN_KEY = ("student", "train", "retrain_every_n_accepted")


@dataclass
class OrchestratorStatus:
    state: str = "idle"  # idle | running | paused | stopped
    teacher_tier: str | None = None
    iterations: int = 0
    tasks_generated: int = 0
    accepted: int = 0
    rejected: int = 0
    accepted_since_retrain: int = 0
    last_error: str | None = None
    stop_reason: str | None = None


class Orchestrator:
    def __init__(
        self,
        cfg: DistillConfig,
        teacher: TeacherClient | None = None,
        judge: TeacherClient | None = None,
        evaluator: Evaluator | None = None,
        prompt_generator: PromptGenerator | None = None,
        dataset: DatasetManager | None = None,
        retrain_fn: Callable[[], None] | None = None,
        batch_size: int = 5,
        poll_interval_s: float = 1.0,
        config_path: Path | None = None,
    ):
        self.cfg = cfg
        self.ledger = BudgetLedger(cfg.db_path)
        self.teacher = teacher or TeacherClient(cfg, cfg.teacher, self.ledger)
        self.judge = judge or TeacherClient(cfg, cfg.judge, self.ledger)
        self.evaluator = evaluator or Evaluator(cfg, judge_client=self.judge)
        self.prompt_generator = prompt_generator or PromptGenerator(cfg, self.teacher)
        self.dataset = dataset or DatasetManager(cfg)
        self.retrain_fn = retrain_fn
        self.batch_size = batch_size
        self.poll_interval_s = poll_interval_s
        # None -> save_config's own default (the real config.yaml). Tests
        # MUST override this so pytest runs never rewrite the real file.
        self._config_path = config_path

        self.status = OrchestratorStatus()
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._pause_event = threading.Event()

    # ------------------------------------------------------------------
    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._pause_event.clear()
        self.status = OrchestratorStatus(state="running", teacher_tier=self.cfg.teacher_tier)
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()

    def pause(self) -> None:
        self._pause_event.set()
        if self.status.state == "running":
            self.status.state = "paused"

    def resume(self) -> None:
        self._pause_event.clear()
        if self.status.state == "paused":
            self.status.state = "running"

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=10)
        self.status.state = "stopped"

    # ------------------------------------------------------------------
    def _run_loop(self) -> None:
        try:
            while not self._stop_event.is_set():
                if self._pause_event.is_set():
                    time.sleep(self.poll_interval_s)
                    continue
                self._step()
                self.status.iterations += 1
                time.sleep(self.poll_interval_s)
        except BudgetExceededError as e:
            self.status.state = "stopped"
            self.status.stop_reason = f"budget_exceeded: {e}"
        except LocalOnlyModeError as e:
            self.status.state = "stopped"
            self.status.stop_reason = f"local_only: {e}"
        except Exception as e:  # loop must never die silently
            self.status.state = "stopped"
            self.status.stop_reason = "error"
            self.status.last_error = f"{e}\n{traceback.format_exc()}"

    def _step(self) -> None:
        tasks = self.prompt_generator.generate(n=self.batch_size)
        self.status.tasks_generated += len(tasks)

        for task in tasks:
            if self._stop_event.is_set():
                return
            reply = self.teacher.complete(
                [{"role": "user", "content": task["prompt"]}], purpose="answer"
            )
            result = self.evaluator.evaluate(task, reply.text)
            if not result.passed:
                self.status.rejected += 1
                continue
            add_result = self.dataset.add(task["prompt"], reply.text, result, reply.tier, reply.model)
            if add_result.accepted:
                self.status.accepted += 1
                self.status.accepted_since_retrain += 1
            else:
                self.status.rejected += 1

        threshold = self.cfg.raw
        for key in RETRAIN_KEY:
            threshold = threshold[key]
        if self.retrain_fn and self.status.accepted_since_retrain >= threshold:
            self.retrain_fn()
            self.status.accepted_since_retrain = 0

    # ------------------------------------------------------------------
    def snapshot(self) -> dict:
        """Full run + spend state for the dashboard."""
        s = self.status
        spent, calls = self.ledger.totals(self.cfg.teacher_tier, "teacher")
        judge_spent, judge_calls = self.ledger.totals(self.cfg.teacher_tier, "judge")
        return {
            "state": s.state,
            "teacher_tier": s.teacher_tier or self.cfg.teacher_tier,
            "iterations": s.iterations,
            "tasks_generated": s.tasks_generated,
            "accepted": s.accepted,
            "rejected": s.rejected,
            "accepted_since_retrain": s.accepted_since_retrain,
            "last_error": s.last_error,
            "stop_reason": s.stop_reason,
            "dataset_counts": self.dataset.counts(),
            "teacher_spend_usd": spent,
            "teacher_calls": calls,
            "judge_spend_usd": judge_spent,
            "judge_calls": judge_calls,
        }

    # ------------------------------------------------------------------
    def graduate(self) -> None:
        """Phase 1 -> 2: free teacher -> paid teacher. Only flips
        phase/teacher_tier in config.yaml — no code changes. Rebuilds
        the teacher client + prompt generator so the new tier's model
        takes effect immediately (TeacherClient snapshots its endpoint
        at construction, it doesn't re-read config per call)."""
        if self.status.state == "running":
            raise RuntimeError("Stop or pause the run before graduating")
        if self.cfg.teacher_tier != "free":
            raise RuntimeError(f"Already on tier {self.cfg.teacher_tier!r}")
        self.cfg.raw["phase"] = 2
        self.cfg.raw["teacher_tier"] = "paid"
        save_config(self.cfg, self._config_path)
        self.teacher = TeacherClient(self.cfg, self.cfg.teacher, self.ledger)
        self.prompt_generator = PromptGenerator(self.cfg, self.teacher)

    def go_local(self) -> None:
        """Disable all API calls; inference must come from the local
        fine-tuned model from here on (enforced inside TeacherClient,
        which reads cfg.local_only fresh on every call — no rebuild
        needed here)."""
        if self.status.state == "running":
            raise RuntimeError("Stop or pause the run before going local")
        self.cfg.raw["inference"]["local_only"] = True
        save_config(self.cfg, self._config_path)
