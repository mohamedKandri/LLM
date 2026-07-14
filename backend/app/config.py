"""Typed loader for config.yaml.

Everything reads config through here so teacher_tier stays the single
source of truth for which teacher/budget/benchmark-target is active.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

# Repo root = two levels above backend/app/
ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "config.yaml"

VALID_TIERS = ("free", "paid")


@dataclass
class Budget:
    max_usd: float
    max_calls: int


@dataclass
class ModelEndpoint:
    """One callable model: the active teacher, or the judge."""

    model: str
    rate_limit_per_min: int
    budget: Budget
    role: str  # "teacher" | "judge" — used to namespace the spend ledger


@dataclass
class DistillConfig:
    raw: dict[str, Any] = field(repr=False, default_factory=dict)

    @property
    def phase(self) -> int:
        return int(self.raw["phase"])

    @property
    def teacher_tier(self) -> str:
        tier = self.raw["teacher_tier"]
        if tier not in VALID_TIERS:
            raise ValueError(f"teacher_tier must be one of {VALID_TIERS}, got {tier!r}")
        return tier

    @property
    def teacher(self) -> ModelEndpoint:
        t = self.raw["teachers"][self.teacher_tier]
        return ModelEndpoint(
            model=t["model"],
            rate_limit_per_min=int(t["rate_limit_per_min"]),
            budget=Budget(float(t["budget"]["max_usd"]), int(t["budget"]["max_calls"])),
            role="teacher",
        )

    @property
    def judge(self) -> ModelEndpoint:
        j = self.raw["judge"]
        return ModelEndpoint(
            model=j["model"],
            rate_limit_per_min=int(j["rate_limit_per_min"]),
            budget=Budget(float(j["budget"]["max_usd"]), int(j["budget"]["max_calls"])),
            role="judge",
        )

    @property
    def judge_min_score(self) -> int:
        return int(self.raw["judge"]["min_score"])

    @property
    def openrouter(self) -> dict[str, Any]:
        return self.raw["openrouter"]

    @property
    def api_key(self) -> str | None:
        return os.environ.get(self.openrouter["api_key_env"])

    @property
    def local_only(self) -> bool:
        return bool(self.raw["inference"]["local_only"])

    @property
    def db_path(self) -> Path:
        return ROOT / self.raw["storage"]["db_path"]

    @property
    def generation(self) -> dict[str, Any]:
        return self.raw["generation"]

    @property
    def evaluation(self) -> dict[str, Any]:
        return self.raw["evaluation"]

    @property
    def benchmark_target(self) -> float:
        return float(self.raw["benchmark"]["targets"][self.teacher_tier])

    @property
    def gold_set_path(self) -> Path:
        return ROOT / self.raw["benchmark"]["gold_set_path"]


def load_config(path: Path | None = None) -> DistillConfig:
    with open(path or CONFIG_PATH, encoding="utf-8") as f:
        return DistillConfig(raw=yaml.safe_load(f))


def save_config(cfg: DistillConfig, path: Path | None = None) -> None:
    """Persist config changes (used by graduate / go-local flows)."""
    with open(path or CONFIG_PATH, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg.raw, f, sort_keys=False, allow_unicode=True)
