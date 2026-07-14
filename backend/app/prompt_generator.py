"""Generates training prompts from seed tasks via the teacher.

STUB — waiting on the user's seed task examples before implementing
the variation logic (the seeds determine task types, difficulty axes,
and how "tests" for code tasks are derived).

Planned interface:

    gen = PromptGenerator(cfg, teacher_client, seeds)
    tasks = gen.generate(n=10)   # -> list of task dicts:
    # {"type": "code"|"general", "prompt": str, "tests": str|None,
    #  "seed_id": str, "teacher_tier": cfg.teacher_tier}
"""

from __future__ import annotations

from .config import DistillConfig
from .teacher_client import TeacherClient


class PromptGenerator:
    def __init__(self, cfg: DistillConfig, teacher: TeacherClient, seeds: list[dict]):
        self.cfg = cfg
        self.teacher = teacher
        self.seeds = seeds

    def generate(self, n: int = 10) -> list[dict]:
        raise NotImplementedError("Awaiting seed task examples from the user")
