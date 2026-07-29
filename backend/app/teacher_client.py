"""OpenRouter client for the teacher and judge models.

Design constraints (see config.yaml):
- Model name always comes from config — never hardcoded. Swapping
  teacher_tier free→paid must require zero code changes here.
- Hard budget cap (USD + call count) enforced in code, persisted in
  SQLite so it survives restarts. Exceeding it raises, it doesn't warn.
- Sliding-window rate limiter sized to the tier's req/min limit.
- Exponential backoff with jitter on 429/5xx/timeouts, honoring
  Retry-After when present.
- local_only=true (the "Go local" switch) hard-disables all API calls.
"""

from __future__ import annotations

import json
import random
import sqlite3
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

import requests

from .config import DistillConfig, ModelEndpoint

_RETRYABLE_STATUS = {408, 429, 500, 502, 503, 504}


class TeacherClientError(Exception):
    """Base error for teacher client failures."""


class BudgetExceededError(TeacherClientError):
    """Hard budget cap hit — the run must stop, not degrade."""


class LocalOnlyModeError(TeacherClientError):
    """API calls are disabled because the app has gone local."""


class MissingAPIKeyError(TeacherClientError):
    pass


@dataclass
class TeacherResponse:
    text: str
    model: str
    tier: str
    prompt_tokens: int
    completion_tokens: int
    cost_usd: float
    raw: dict[str, Any]


class _RateLimiter:
    """Sliding-window limiter: at most `per_min` requests in any 60s span."""

    def __init__(self, per_min: int):
        self.per_min = per_min
        self._timestamps: list[float] = []
        self._lock = threading.Lock()

    def acquire(self) -> None:
        while True:
            with self._lock:
                now = time.monotonic()
                self._timestamps = [t for t in self._timestamps if now - t < 60.0]
                if len(self._timestamps) < self.per_min:
                    self._timestamps.append(now)
                    return
                wait = 60.0 - (now - self._timestamps[0]) + 0.05
            time.sleep(max(wait, 0.05))


class BudgetLedger:
    """Persistent spend/call ledger, one row per API call.

    The cap check reads accumulated totals from SQLite before every
    call, so restarts, crashes, or multiple client instances can't
    reset or bypass the cap. Totals are namespaced by (tier, role):
    teacher spend in phase 1 vs phase 2 stays separately auditable.
    """

    def __init__(self, db_path):
        self.db_path = str(db_path)
        self._lock = threading.Lock()
        with self._conn() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS api_ledger (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ts TEXT NOT NULL DEFAULT (datetime('now')),
                    tier TEXT NOT NULL,
                    role TEXT NOT NULL,
                    model TEXT NOT NULL,
                    purpose TEXT NOT NULL,
                    prompt_tokens INTEGER NOT NULL DEFAULT 0,
                    completion_tokens INTEGER NOT NULL DEFAULT 0,
                    cost_usd REAL NOT NULL DEFAULT 0.0
                )
                """
            )

    @contextmanager
    def _conn(self):
        # Nested `with conn:` commits/rolls back the transaction; the
        # outer try/finally guarantees the file handle is actually
        # closed too — sqlite3.Connection's own context manager only
        # does the former, which leaks handles on long-running loops
        # (and on Windows, blocks deleting the db file at all).
        conn = sqlite3.connect(self.db_path)
        conn.execute("PRAGMA journal_mode=WAL")
        try:
            with conn:
                yield conn
        finally:
            conn.close()

    def totals(self, tier: str, role: str) -> tuple[float, int]:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT COALESCE(SUM(cost_usd), 0), COUNT(*) FROM api_ledger "
                "WHERE tier = ? AND role = ?",
                (tier, role),
            ).fetchone()
        return float(row[0]), int(row[1])

    def check(self, tier: str, role: str, budget) -> None:
        spent, calls = self.totals(tier, role)
        if calls >= budget.max_calls:
            raise BudgetExceededError(
                f"Call cap hit for {role}/{tier}: {calls}/{budget.max_calls} calls used"
            )
        if spent > budget.max_usd + 1e-9:
            raise BudgetExceededError(
                f"Budget cap hit for {role}/{tier}: ${spent:.4f} spent, cap ${budget.max_usd:.2f}"
            )

    def record(
        self,
        tier: str,
        role: str,
        model: str,
        purpose: str,
        prompt_tokens: int,
        completion_tokens: int,
        cost_usd: float,
    ) -> None:
        with self._lock, self._conn() as conn:
            conn.execute(
                "INSERT INTO api_ledger "
                "(tier, role, model, purpose, prompt_tokens, completion_tokens, cost_usd) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (tier, role, model, purpose, prompt_tokens, completion_tokens, cost_usd),
            )


class TeacherClient:
    """One instance per endpoint role: the active teacher, or the judge.

    >>> cfg = load_config()
    >>> teacher = TeacherClient(cfg, cfg.teacher)
    >>> judge = TeacherClient(cfg, cfg.judge)
    """

    def __init__(self, cfg: DistillConfig, endpoint: ModelEndpoint, ledger: BudgetLedger | None = None):
        self.cfg = cfg
        self.endpoint = endpoint
        self.ledger = ledger or BudgetLedger(cfg.db_path)
        self._limiter = _RateLimiter(endpoint.rate_limit_per_min)
        self._or = cfg.openrouter

    # ------------------------------------------------------------------
    def complete(
        self,
        messages: list[dict[str, str]],
        purpose: str = "generate",
        **overrides: Any,
    ) -> TeacherResponse:
        """Chat completion with rate limiting, budget enforcement, retries.

        `purpose` tags the ledger row ("generate", "judge", "benchmark", ...)
        so the audit log shows what every dollar/call was spent on.
        """
        if self.cfg.local_only:
            raise LocalOnlyModeError(
                "inference.local_only is true — API calls are disabled. "
                "Flip it in config.yaml to re-enable."
            )
        api_key = self.cfg.api_key
        if not api_key:
            raise MissingAPIKeyError(
                f"Set the {self._or['api_key_env']} environment variable "
                "(Settings > API key in the UI)."
            )

        # Enforced BEFORE the call: once the cap is hit nothing else goes out.
        self.ledger.check(self.cfg.teacher_tier, self.endpoint.role, self.endpoint.budget)

        gen = self.cfg.generation
        payload: dict[str, Any] = {
            "model": self.endpoint.model,
            "messages": messages,
            "temperature": overrides.get("temperature", gen.get("temperature", 0.7)),
            "max_tokens": overrides.get("max_tokens", gen.get("max_tokens", 2048)),
            # Ask OpenRouter to report actual cost in the response.
            "usage": {"include": True},
        }

        self._limiter.acquire()
        data = self._post_with_retries(payload, api_key)

        try:
            text = data["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError) as e:
            raise TeacherClientError(f"Malformed OpenRouter response: {data}") from e

        usage = data.get("usage") or {}
        resp = TeacherResponse(
            text=text,
            model=data.get("model", self.endpoint.model),
            tier=self.cfg.teacher_tier,
            prompt_tokens=int(usage.get("prompt_tokens", 0)),
            completion_tokens=int(usage.get("completion_tokens", 0)),
            cost_usd=float(usage.get("cost", 0.0)),
            raw=data,
        )
        self.ledger.record(
            tier=resp.tier,
            role=self.endpoint.role,
            model=resp.model,
            purpose=purpose,
            prompt_tokens=resp.prompt_tokens,
            completion_tokens=resp.completion_tokens,
            cost_usd=resp.cost_usd,
        )
        return resp

    # ------------------------------------------------------------------
    def _post_with_retries(self, payload: dict[str, Any], api_key: str) -> dict[str, Any]:
        url = self._or["base_url"].rstrip("/") + "/chat/completions"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://localhost/distill",
            "X-Title": "Distill",
        }
        max_retries = int(self._or.get("max_retries", 5))
        backoff = float(self._or.get("backoff_base_s", 2.0))
        timeout = float(self._or.get("timeout_s", 120))

        last_err: Exception | None = None
        for attempt in range(max_retries + 1):
            try:
                r = requests.post(url, headers=headers, json=payload, timeout=timeout)
                if r.status_code == 200:
                    data = r.json()
                    if "error" not in data:
                        return data
                    # Some upstream failures come back as HTTP 200 with an
                    # {"error": {...}} body instead of a real error status
                    # (seen on OpenRouter's free tier under provider load)
                    # — treat it exactly like a retryable HTTP status.
                    if attempt < max_retries:
                        time.sleep(backoff * (2**attempt) + random.uniform(0, 1))
                        continue
                    raise TeacherClientError(f"OpenRouter embedded error: {data['error']}")
                if r.status_code in _RETRYABLE_STATUS and attempt < max_retries:
                    retry_after = r.headers.get("Retry-After")
                    delay = (
                        float(retry_after)
                        if retry_after and retry_after.replace(".", "", 1).isdigit()
                        else backoff * (2**attempt) + random.uniform(0, 1)
                    )
                    time.sleep(delay)
                    continue
                raise TeacherClientError(
                    f"OpenRouter HTTP {r.status_code}: {r.text[:500]}"
                )
            except (requests.Timeout, requests.ConnectionError) as e:
                last_err = e
                if attempt < max_retries:
                    time.sleep(backoff * (2**attempt) + random.uniform(0, 1))
                    continue
                raise TeacherClientError(f"Network failure after {attempt + 1} attempts") from e
            except json.JSONDecodeError as e:
                raise TeacherClientError("OpenRouter returned non-JSON body") from e
        raise TeacherClientError("Retries exhausted") from last_err
