"""Gold-set benchmark: student vs the ACTIVE teacher tier.

STUB — needs the gold set (human-verified, lives in data/gold_set/,
NEVER exported to training data; dataset_manager must refuse any
prompt that hashes to a gold-set entry).

Scores are stored per tier in SQLite (table `benchmark_runs` with a
teacher_tier column) so phase-1 (vs free) and phase-2 (vs paid)
histories coexist — graduation resets the target, not the history.

Planned interface:
    b = Benchmark(cfg, teacher_client)
    result = b.run(student_infer_fn)   # -> {"student": s, "teacher": t, "ratio": s/t}
    b.history(tier="free")             # chart data for the UI
    b.target_met()                     # ratio >= cfg.benchmark_target -> graduation eligible
"""

from __future__ import annotations

from .config import DistillConfig


class Benchmark:
    def __init__(self, cfg: DistillConfig, teacher):
        self.cfg = cfg
        raise NotImplementedError
