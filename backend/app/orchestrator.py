"""Main loop: generate -> teacher -> evaluate -> save -> retrain -> benchmark.

STUB — assembled last, once every module it drives exists.

Responsibilities:
- Reads phase/teacher_tier from config at loop start; tags every
  artifact (examples, ledger rows, benchmark runs) with the active tier.
- Start / pause / stop controlled from the FastAPI endpoints.
- Stops cleanly on BudgetExceededError instead of crashing.
- graduate(): free -> paid tier switch in config.yaml (no code change).
- go_local(): sets inference.local_only = true, killing all API paths.
"""

from __future__ import annotations

from .config import DistillConfig


class Orchestrator:
    def __init__(self, cfg: DistillConfig):
        self.cfg = cfg
        raise NotImplementedError
