"""Wires dataset -> trainer -> local inference -> benchmark into the
single no-arg callable orchestrator.retrain_fn expects.

Kept separate from orchestrator.py so that module stays generic (it
doesn't need to know Trainer/LocalStudent/Benchmark exist) — this is
just the concrete implementation main.py plugs in.

Runs synchronously, inside the orchestrator's own background thread,
same as every other step. CPU training can take a long time (see the
README) — that's an intentional pause of generation/evaluation while a
retrain is in progress, not a bug to fix with concurrency; keeping this
single-threaded keeps the whole pipeline's behavior easy to reason about.

Trainer/LocalStudent/Benchmark classes are injectable for testing
(make_retrain_fn(orch, trainer_cls=FakeTrainer, ...)) so the wiring
logic can be unit tested in milliseconds without doing real training.
"""

from __future__ import annotations

import logging
from typing import Callable

from .benchmark import Benchmark
from .local_inference import LocalStudent
from .trainer import Trainer

logger = logging.getLogger(__name__)


def make_retrain_fn(
    orchestrator,
    trainer_cls=Trainer,
    student_cls=LocalStudent,
    benchmark_cls=Benchmark,
) -> Callable[[], None]:
    """`orchestrator` supplies cfg, dataset, teacher, evaluator — already
    built for the generate/evaluate loop, reused here rather than
    duplicating construction."""

    def _retrain() -> None:
        cfg = orchestrator.cfg
        export_path = cfg.db_path.parent / "exports" / "train.jsonl"
        n = orchestrator.dataset.export_jsonl(export_path)
        if n == 0:
            logger.info("retrain skipped: no accepted examples to train on")
            return

        trainer = trainer_cls(cfg)
        result = trainer.train(export_path)
        logger.info("trained generation %s on %d examples", result.generation, result.n_examples)

        student = student_cls(cfg, result.checkpoint_dir)
        bench = benchmark_cls(cfg, orchestrator.teacher, orchestrator.evaluator)
        bench_result = bench.run(student.generate)
        logger.info(
            "benchmark: %.1f%% of %s teacher (target %.0f%%, met=%s)",
            bench_result.ratio * 100,
            bench_result.teacher_tier,
            bench_result.target * 100,
            bench_result.target_met,
        )

    return _retrain
