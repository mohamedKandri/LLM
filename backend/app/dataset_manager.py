"""SQLite training-set store: accepted (prompt, response, score) pairs,
deduplicated by embedding similarity, exported to JSONL for SFTTrainer.

Dedup uses a pluggable embedder (see storage.dedup_similarity_threshold
in config.yaml):
- Default: `HashingEmbedder`, a dependency-free hashed character-trigram
  vector. Not semantic, but it's exactly what catches what a self-instruct
  loop actually produces — near-identical reworded repeats of the same
  task — without pulling in sentence-transformers before it's needed
  (that dep is still commented out in requirements.txt).
- A real embedding model can be swapped in later via `DatasetManager(cfg,
  embedder=...)`; the schema and call sites don't change.

Schema (table `examples`):
    id, prompt, response, score REAL, method TEXT,
    teacher_tier TEXT,      -- "free"|"paid": phase provenance, the audit tag
    teacher_model TEXT,
    source TEXT,            -- "self_generated" | "human_verified"
    embedding BLOB,          -- packed float32 vector, for dedup
    created_at TEXT

`add()` takes prompt/response/eval_result/tier/model directly rather than
a task dict — the task that GENERATED a prompt and the call that ANSWERED
it are separate teacher calls, so the tier tag has to come from the
answering call, not from prompt_generator's metadata.

Gold-set prompts (benchmark.gold_set_path) are refused here so they can
never leak into training data — this is the model-collapse guard, not a
suggestion.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import sqlite3
import threading
from array import array
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from .config import DistillConfig
from .evaluator import EvalResult


def _normalize(text: str) -> str:
    """Casefolded, punctuation-free — cheap exact/near-exact dup key."""
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


class HashingEmbedder:
    """Bag-of-character-trigrams, hashed into a fixed-size signed vector
    and L2-normalized, so cosine similarity is a plain dot product."""

    dims = 256

    def embed(self, text: str) -> list[float]:
        norm_text = re.sub(r"\s+", " ", text.lower()).strip()
        grams = [norm_text[i : i + 3] for i in range(max(len(norm_text) - 2, 1))]
        vec = [0.0] * self.dims
        for g in grams or [norm_text]:
            h = int.from_bytes(hashlib.blake2b(g.encode("utf-8"), digest_size=8).digest(), "big")
            vec[h % self.dims] += 1.0 if (h // self.dims) % 2 == 0 else -1.0
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]


@dataclass
class AddResult:
    accepted: bool
    reason: str  # "accepted" | "below_threshold" | "duplicate" | "in_gold_set"
    id: int | None = None


def _pack(vec: list[float]) -> bytes:
    return array("f", vec).tobytes()


def _unpack(blob: bytes) -> list[float]:
    a = array("f")
    a.frombytes(blob)
    return list(a)


def _cosine(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))  # both already L2-normalized


class DatasetManager:
    def __init__(self, cfg: DistillConfig, embedder=None):
        self.cfg = cfg
        self.embedder = embedder or HashingEmbedder()
        self.threshold = float(cfg.raw["storage"]["dedup_similarity_threshold"])
        self._lock = threading.Lock()
        self.db_path = str(cfg.db_path)
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS examples (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    prompt TEXT NOT NULL,
                    response TEXT NOT NULL,
                    score REAL NOT NULL,
                    method TEXT NOT NULL,
                    teacher_tier TEXT NOT NULL,
                    teacher_model TEXT NOT NULL,
                    source TEXT NOT NULL,
                    embedding BLOB NOT NULL,
                    created_at TEXT NOT NULL DEFAULT (datetime('now'))
                )
                """
            )
        # Small-scale dataset assumption: cache all embeddings in memory
        # for dedup checks. Revisit (e.g. an ANN index) if the accepted
        # set grows into the tens of thousands.
        self._cache: list[tuple[int, list[float]]] = []
        with self._conn() as conn:
            for row_id, blob in conn.execute("SELECT id, embedding FROM examples"):
                self._cache.append((row_id, _unpack(blob)))
        self._gold_prompts = self._load_gold_prompts()

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def _load_gold_prompts(self) -> set[str]:
        path = self.cfg.gold_set_path
        prompts: set[str] = set()
        if path.exists():
            with open(path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        prompts.add(_normalize(json.loads(line)["prompt"]))
        return prompts

    # ------------------------------------------------------------------
    def add(
        self,
        prompt: str,
        response: str,
        eval_result: EvalResult,
        teacher_tier: str,
        teacher_model: str,
        source: str = "self_generated",
    ) -> AddResult:
        if not eval_result.passed:
            return AddResult(False, "below_threshold")
        if _normalize(prompt) in self._gold_prompts:
            return AddResult(False, "in_gold_set")

        vec = self.embedder.embed(prompt)
        with self._lock:
            for _, existing in self._cache:
                if _cosine(vec, existing) >= self.threshold:
                    return AddResult(False, "duplicate")

            with self._conn() as conn:
                cur = conn.execute(
                    "INSERT INTO examples "
                    "(prompt, response, score, method, teacher_tier, teacher_model, source, embedding) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        prompt,
                        response,
                        eval_result.score,
                        eval_result.method,
                        teacher_tier,
                        teacher_model,
                        source,
                        _pack(vec),
                    ),
                )
                row_id = cur.lastrowid
            self._cache.append((row_id, vec))
        return AddResult(True, "accepted", row_id)

    def add_human_verified(
        self, prompt: str, response: str, teacher_tier: str, teacher_model: str = "human"
    ) -> AddResult:
        """Import hand-written examples so the training set isn't 100%
        self-generated. Still subject to the gold-set and dedup guards."""
        result = EvalResult(score=1.0, passed=True, method="human_verified")
        return self.add(prompt, response, result, teacher_tier, teacher_model, source="human_verified")

    def import_human_verified_jsonl(self, path: Path) -> list[AddResult]:
        """Bulk import: one {"prompt", "response", "teacher_tier"} object per line."""
        results = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                results.append(
                    self.add_human_verified(
                        row["prompt"], row["response"], row.get("teacher_tier", "free")
                    )
                )
        return results

    # ------------------------------------------------------------------
    def counts(self) -> dict[str, int]:
        """Accepted-example counts per teacher tier, for the dashboard."""
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT teacher_tier, COUNT(*) FROM examples GROUP BY teacher_tier"
            ).fetchall()
        return {tier: n for tier, n in rows}

    def export_jsonl(self, path: str | Path, tiers: Iterable[str] = ("free", "paid")) -> int:
        """Write accepted examples as chat-format JSONL for SFTTrainer.
        Returns the number of examples written."""
        tiers = tuple(tiers)
        placeholders = ",".join("?" * len(tiers))
        with self._conn() as conn:
            rows = conn.execute(
                f"SELECT prompt, response FROM examples "
                f"WHERE teacher_tier IN ({placeholders}) ORDER BY id",
                tiers,
            ).fetchall()

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            for prompt, response in rows:
                record = {
                    "messages": [
                        {"role": "user", "content": prompt},
                        {"role": "assistant", "content": response},
                    ]
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        return len(rows)
