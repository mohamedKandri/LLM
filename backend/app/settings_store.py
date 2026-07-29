"""Read/write helpers backing the Settings UI: the active tier's teacher
model + budget, the OpenRouter API key (.env), and the seed task list
(data/seeds.jsonl).

Kept separate from config.py, which stays a read-only loader — writes
here are explicit, narrow, and never a side effect of a GET. All write
paths accept an override (config_path / env_path / seeds_path) so tests
never touch the real config.yaml/.env/seeds.jsonl — same rule
orchestrator.py follows for graduate()/go_local().
"""

from __future__ import annotations

import json
from pathlib import Path

from .config import ROOT, DistillConfig, save_config


def get_settings(cfg: DistillConfig) -> dict:
    return {
        "teacher_tier": cfg.teacher_tier,
        "teacher_model": cfg.teacher.model,
        "budget_max_usd": cfg.teacher.budget.max_usd,
        "budget_max_calls": cfg.teacher.budget.max_calls,
        "local_only": cfg.local_only,
        "api_key_set": cfg.api_key is not None,
    }


def update_settings(
    cfg: DistillConfig,
    teacher_model: str | None = None,
    budget_max_usd: float | None = None,
    budget_max_calls: int | None = None,
    config_path: Path | None = None,
) -> dict:
    """Applies to the block for the CURRENTLY ACTIVE tier only — Settings
    edits "what the app is using right now", not the other tier's config."""
    tier_block = cfg.raw["teachers"][cfg.teacher_tier]
    if teacher_model is not None:
        if not teacher_model.strip():
            raise ValueError("teacher_model cannot be empty")
        tier_block["model"] = teacher_model.strip()
    if budget_max_usd is not None:
        if budget_max_usd < 0:
            raise ValueError("budget_max_usd cannot be negative")
        tier_block["budget"]["max_usd"] = budget_max_usd
    if budget_max_calls is not None:
        if budget_max_calls < 0:
            raise ValueError("budget_max_calls cannot be negative")
        tier_block["budget"]["max_calls"] = int(budget_max_calls)
    save_config(cfg, config_path)
    return get_settings(cfg)


# ---- API key (.env) --------------------------------------------------

_API_KEY_NAME = "OPENROUTER_API_KEY"


def set_api_key(key: str, env_path: Path | None = None) -> None:
    if not key or not key.strip():
        raise ValueError("API key cannot be empty")
    _write_env_value(_API_KEY_NAME, key.strip(), env_path or (ROOT / ".env"))


def remove_api_key(env_path: Path | None = None) -> None:
    _delete_env_value(_API_KEY_NAME, env_path or (ROOT / ".env"))


def _write_env_value(name: str, value: str, env_file: Path) -> None:
    lines = env_file.read_text(encoding="utf-8").splitlines() if env_file.exists() else []
    prefix = f"{name}="
    out, found = [], False
    for line in lines:
        if line.strip().startswith(prefix):
            out.append(f"{name}={value}")
            found = True
        else:
            out.append(line)
    if not found:
        out.append(f"{name}={value}")
    env_file.write_text("\n".join(out) + "\n", encoding="utf-8")


def _delete_env_value(name: str, env_file: Path) -> None:
    if not env_file.exists():
        return
    prefix = f"{name}="
    kept = [l for l in env_file.read_text(encoding="utf-8").splitlines() if not l.strip().startswith(prefix)]
    env_file.write_text("\n".join(kept) + ("\n" if kept else ""), encoding="utf-8")


# ---- seeds CRUD --------------------------------------------------------


def list_seeds(cfg: DistillConfig, seeds_path: Path | None = None) -> list[dict]:
    path = seeds_path or (ROOT / cfg.generation["seeds_path"])
    seeds = []
    if path.exists():
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    seeds.append(json.loads(line))
    return seeds


def add_seed(
    cfg: DistillConfig,
    type_: str,
    prompt: str,
    tests: str | None = None,
    seeds_path: Path | None = None,
) -> dict:
    if type_ not in ("code", "general"):
        raise ValueError("type must be 'code' or 'general'")
    if not prompt or len(prompt.strip()) < 10:
        raise ValueError("prompt must be at least 10 characters")
    if type_ == "code":
        if not tests or "assert" not in tests:
            raise ValueError("code seeds need a 'tests' field with at least one assert")
        compile(tests, "<tests>", "exec")

    path = seeds_path or (ROOT / cfg.generation["seeds_path"])
    seeds = list_seeds(cfg, seeds_path)
    existing_ids = {s.get("id") for s in seeds}
    n = len(seeds) + 1
    seed_id = f"seed-{n:02d}"
    while seed_id in existing_ids:
        n += 1
        seed_id = f"seed-{n:02d}"

    entry: dict = {"id": seed_id, "type": type_, "prompt": prompt.strip()}
    if tests:
        entry["tests"] = tests.strip()

    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")
    return entry


def delete_seed(cfg: DistillConfig, seed_id: str, seeds_path: Path | None = None) -> bool:
    path = seeds_path or (ROOT / cfg.generation["seeds_path"])
    seeds = list_seeds(cfg, seeds_path)
    remaining = [s for s in seeds if s.get("id") != seed_id]
    if len(remaining) == len(seeds):
        return False
    with open(path, "w", encoding="utf-8") as f:
        for s in remaining:
            f.write(json.dumps(s) + "\n")
    return True
