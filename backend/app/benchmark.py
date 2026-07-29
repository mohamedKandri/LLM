"""Gold-set benchmark: student vs the ACTIVE teacher tier.

Runs every gold-set task through both the student (via a caller-supplied
inference function — trainer.py/local inference doesn't exist yet, so
this stays decoupled and is testable with a fake) and the active teacher,
scores both answers with the SAME Evaluator used to accept training
data, and stores the ratio (student/teacher) per run.

Scores are stored per tier in SQLite (`benchmark_runs.teacher_tier`) so
phase-1 (vs free) and phase-2 (vs paid) histories coexist — graduating
changes which target `cfg.benchmark_target` returns, never overwrites
the history.

Teacher gold-set answers are cached (`benchmark_teacher_cache`, keyed by
gold task id + teacher model, generated at temperature 0 for
reproducibility) so re-running the benchmark after another fine-tune
doesn't re-spend budget re-asking the teacher the same fixed questions.

Interface:
    b = Benchmark(cfg, teacher_client, evaluator)
    result = b.run(student_infer_fn)   # student_infer_fn: str -> str
    b.history(tier="free")             # chart data for the UI
    b.target_met()                     # latest run for the active tier cleared cfg.benchmark_target
"""

from __future__ import annotations

import json
import sqlite3
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from .config import DistillConfig
from .evaluator import Evaluator
from .teacher_client import TeacherClient

StudentInferFn = Callable[[str], str]


def read_history(db_path: str | Path, tier: str | None = None) -> list[dict]:
    """Standalone reader for the API/UI: no gold set, teacher, or
    evaluator needed, just the DB. Safe to call before any Benchmark has
    ever run in this DB (table may not exist yet -> empty list)."""
    conn = sqlite3.connect(str(db_path))
    try:
        conn.execute("PRAGMA journal_mode=WAL")
        exists = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='benchmark_runs'"
        ).fetchone()
        if not exists:
            return []
        query = (
            "SELECT id, teacher_tier, teacher_model, student_score, teacher_score, "
            "ratio, target, target_met, created_at FROM benchmark_runs"
        )
        params: tuple = ()
        if tier:
            query += " WHERE teacher_tier = ?"
            params = (tier,)
        query += " ORDER BY id"
        rows = conn.execute(query, params).fetchall()
    finally:
        conn.close()
    return [
        {
            "id": r[0],
            "teacher_tier": r[1],
            "teacher_model": r[2],
            "student_score": r[3],
            "teacher_score": r[4],
            "ratio": r[5],
            "target": r[6],
            "target_met": bool(r[7]),
            "created_at": r[8],
        }
        for r in rows
    ]


def _load_gold(path: Path) -> list[dict]:
    tasks = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                tasks.append(json.loads(line))
    return tasks


@dataclass
class BenchmarkResult:
    id: int
    teacher_tier: str
    teacher_model: str
    student_score: float
    teacher_score: float
    ratio: float  # student_score / teacher_score; 0.0 if teacher scored 0
    target: float
    target_met: bool
    per_item: list[dict]


class Benchmark:
    def __init__(self, cfg: DistillConfig, teacher: TeacherClient, evaluator: Evaluator):
        self.cfg = cfg
        self.teacher = teacher
        self.evaluator = evaluator
        self._lock = threading.Lock()
        self.db_path = str(cfg.db_path)
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self.gold = _load_gold(cfg.gold_set_path)
        if not self.gold:
            raise ValueError(f"No gold-set entries found at {cfg.gold_set_path}")
        with self._conn() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS benchmark_runs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    teacher_tier TEXT NOT NULL,
                    teacher_model TEXT NOT NULL,
                    student_score REAL NOT NULL,
                    teacher_score REAL NOT NULL,
                    ratio REAL NOT NULL,
                    target REAL NOT NULL,
                    target_met INTEGER NOT NULL,
                    per_item TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT (datetime('now'))
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS benchmark_teacher_cache (
                    gold_id TEXT NOT NULL,
                    teacher_model TEXT NOT NULL,
                    response TEXT NOT NULL,
                    PRIMARY KEY (gold_id, teacher_model)
                )
                """
            )

    @contextmanager
    def _conn(self):
        conn = sqlite3.connect(self.db_path)
        conn.execute("PRAGMA journal_mode=WAL")
        try:
            with conn:
                yield conn
        finally:
            conn.close()

    # ------------------------------------------------------------------
    def run(self, student_infer_fn: StudentInferFn) -> BenchmarkResult:
        tier = self.cfg.teacher_tier
        model = self.teacher.endpoint.model
        per_item = []
        student_scores = []
        teacher_scores = []

        for task in self.gold:
            student_answer = student_infer_fn(task["prompt"])
            teacher_answer = self._teacher_answer(task, model)

            student_eval = self.evaluator.evaluate(task, student_answer)
            teacher_eval = self.evaluator.evaluate(task, teacher_answer)

            student_scores.append(student_eval.score)
            teacher_scores.append(teacher_eval.score)
            per_item.append(
                {
                    "id": task["id"],
                    "type": task["type"],
                    "student_score": student_eval.score,
                    "teacher_score": teacher_eval.score,
                }
            )

        student_avg = sum(student_scores) / len(student_scores)
        teacher_avg = sum(teacher_scores) / len(teacher_scores)
        ratio = (student_avg / teacher_avg) if teacher_avg > 0 else 0.0
        target = self.cfg.benchmark_target
        met = ratio >= target

        with self._lock, self._conn() as conn:
            cur = conn.execute(
                "INSERT INTO benchmark_runs "
                "(teacher_tier, teacher_model, student_score, teacher_score, ratio, target, target_met, per_item) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (tier, model, student_avg, teacher_avg, ratio, target, int(met), json.dumps(per_item)),
            )
            run_id = cur.lastrowid

        return BenchmarkResult(run_id, tier, model, student_avg, teacher_avg, ratio, target, met, per_item)

    def _teacher_answer(self, task: dict, model: str) -> str:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT response FROM benchmark_teacher_cache WHERE gold_id = ? AND teacher_model = ?",
                (task["id"], model),
            ).fetchone()
        if row:
            return row[0]
        reply = self.teacher.complete(
            [{"role": "user", "content": task["prompt"]}],
            purpose="benchmark",
            temperature=0.0,
        )
        with self._lock, self._conn() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO benchmark_teacher_cache (gold_id, teacher_model, response) VALUES (?, ?, ?)",
                (task["id"], model, reply.text),
            )
        return reply.text

    # ------------------------------------------------------------------
    def history(self, tier: str | None = None) -> list[dict]:
        query = (
            "SELECT id, teacher_tier, teacher_model, student_score, teacher_score, "
            "ratio, target, target_met, created_at FROM benchmark_runs"
        )
        params: tuple = ()
        if tier:
            query += " WHERE teacher_tier = ?"
            params = (tier,)
        query += " ORDER BY id"
        with self._conn() as conn:
            rows = conn.execute(query, params).fetchall()
        return [
            {
                "id": r[0],
                "teacher_tier": r[1],
                "teacher_model": r[2],
                "student_score": r[3],
                "teacher_score": r[4],
                "ratio": r[5],
                "target": r[6],
                "target_met": bool(r[7]),
                "created_at": r[8],
            }
            for r in rows
        ]

    def target_met(self) -> bool:
        """Whether the MOST RECENT run for the currently active tier
        cleared cfg.benchmark_target — i.e. graduation-eligible."""
        runs = self.history(tier=self.cfg.teacher_tier)
        return bool(runs) and runs[-1]["target_met"]
