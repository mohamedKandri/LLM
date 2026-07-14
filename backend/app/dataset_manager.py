"""SQLite training-set store + JSONL export.

STUB — schema is final, logic lands after teacher_client/evaluator
are exercised end to end.

Schema (table `examples`):
    id, prompt, response, score REAL, method TEXT,
    teacher_tier TEXT,      -- "free"|"paid": phase provenance, the audit tag
    teacher_model TEXT,
    source TEXT,            -- "self_generated" | "human_verified"
    embedding BLOB,         -- for dedup (cosine sim > storage.dedup_similarity_threshold)
    created_at TEXT

Planned interface:
    dm = DatasetManager(cfg)
    dm.add(task, response_text, eval_result)      # rejects if dup or below threshold
    dm.export_jsonl(path, tiers=("free","paid"))  # chat-format JSONL for SFTTrainer
    dm.counts()                                   # per-tier accepted counts for the UI
"""

from __future__ import annotations

from .config import DistillConfig


class DatasetManager:
    def __init__(self, cfg: DistillConfig):
        self.cfg = cfg
        raise NotImplementedError
