"""Generates new training tasks from seed tasks via the teacher.

Self-instruct style: each teacher call shows `few_shot_k` random seeds
of one type and asks for `tasks_per_call` NEW tasks in strict JSONL.
Generated tasks are validated (schema, compiling tests for code tasks)
and deduplicated against the seeds and each other before being handed
to the orchestrator. Invalid or duplicate items are dropped, not fixed —
a bad code task just wastes one candidate; its broken tests would make
the teacher's answer fail evaluation anyway.

Task dict shape (consumed by evaluator + dataset_manager):
    {"type": "code"|"general", "prompt": str, "tests": str (code only),
     "seed_ids": [str], "teacher_tier": str, "teacher_model": str}
"""

from __future__ import annotations

import json
import random
import re
from pathlib import Path
from typing import Any

from .config import ROOT, DistillConfig
from .teacher_client import TeacherClient

CODE_SYSTEM_PROMPT = """\
You create Python programming exercises for training a coding assistant.
Given example tasks, produce NEW tasks that differ in topic and difficulty —
do not rephrase the examples. Mix easy/medium/hard; include some debugging
tasks ("fix this buggy function") like the examples show.

Rules for every task:
- Solvable with the Python standard library only. No I/O, no network,
  no randomness, deterministic output.
- "tests" is plain Python `assert` lines that run AFTER the solution code
  in the same file. Test the function/class named in the prompt. 2-4 asserts,
  including at least one edge case. The asserts must be consistent with a
  correct solution — verify them mentally before writing them.

Output ONLY a JSON array, no prose, no markdown fences. Each element:
{"type": "code", "prompt": "...", "tests": "assert ...\\nassert ..."}"""

GENERAL_SYSTEM_PROMPT = """\
You create technical questions for training a programming/CS assistant.
Given example questions, produce NEW questions that differ in topic —
do not rephrase the examples. Cover CS fundamentals, Python, networking,
architecture, and developer practice. Vary difficulty and tone: some
formal, some casual like a colleague asking in chat.

Output ONLY a JSON array, no prose, no markdown fences. Each element:
{"type": "general", "prompt": "..."}"""

_FENCE_RE = re.compile(r"```(?:json|jsonl)?\s*\n(.*?)```", re.DOTALL)


def load_seeds(cfg: DistillConfig) -> list[dict[str, Any]]:
    path = ROOT / cfg.generation["seeds_path"]
    seeds = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                seeds.append(json.loads(line))
    return seeds


def _normalize(prompt: str) -> str:
    """Cheap dedup key: casefolded, punctuation-free."""
    return re.sub(r"[^a-z0-9]+", " ", prompt.lower()).strip()


class PromptGenerator:
    def __init__(self, cfg: DistillConfig, teacher: TeacherClient, seeds: list[dict] | None = None):
        self.cfg = cfg
        self.teacher = teacher
        self.seeds = seeds if seeds is not None else load_seeds(cfg)
        if not self.seeds:
            raise ValueError("No seed tasks loaded")
        self._seen = {_normalize(s["prompt"]) for s in self.seeds}

    # ------------------------------------------------------------------
    def generate(self, n: int = 10) -> list[dict[str, Any]]:
        """Return up to `n` validated, deduplicated new tasks.

        May return fewer if the teacher keeps producing invalid or
        duplicate output — the call budget per invocation is capped so
        a misbehaving (typically free) teacher can't burn the ledger.
        """
        gen = self.cfg.generation
        per_call = int(gen.get("tasks_per_call", 5))
        code_fraction = float(gen.get("code_fraction", 0.5))

        n_code = round(n * code_fraction)
        wanted = {"code": n_code, "general": n - n_code}
        # Hard cap on teacher calls: 2x the ideal number, minimum 2.
        max_calls = max(2, 2 * -(-n // per_call))

        out: list[dict[str, Any]] = []
        calls = 0
        for task_type in ("code", "general"):
            while wanted[task_type] > 0 and calls < max_calls:
                calls += 1
                batch = self._one_batch(task_type, min(per_call, wanted[task_type]))
                for task in batch:
                    if wanted[task_type] <= 0:
                        break
                    out.append(task)
                    wanted[task_type] -= 1
        random.shuffle(out)
        return out

    # ------------------------------------------------------------------
    def _one_batch(self, task_type: str, count: int) -> list[dict[str, Any]]:
        few_shot_k = int(self.cfg.generation.get("few_shot_k", 3))
        pool = [s for s in self.seeds if s["type"] == task_type]
        shots = random.sample(pool, min(few_shot_k, len(pool)))

        examples = json.dumps(
            [{k: s[k] for k in ("type", "prompt", "tests") if k in s} for s in shots],
            indent=2,
        )
        system = CODE_SYSTEM_PROMPT if task_type == "code" else GENERAL_SYSTEM_PROMPT
        messages = [
            {"role": "system", "content": system},
            {
                "role": "user",
                "content": f"Example tasks:\n{examples}\n\nProduce {count} new tasks now.",
            },
        ]
        reply = self.teacher.complete(messages, purpose="generate_prompts")

        valid = []
        for item in self._parse_tasks(reply.text):
            task = self._validate(item, task_type)
            if task is None:
                continue
            key = _normalize(task["prompt"])
            if key in self._seen:
                continue
            self._seen.add(key)
            task["seed_ids"] = [s.get("id", "?") for s in shots]
            task["teacher_tier"] = reply.tier
            task["teacher_model"] = reply.model
            valid.append(task)
        return valid

    # ------------------------------------------------------------------
    @staticmethod
    def _parse_tasks(text: str) -> list[dict[str, Any]]:
        """Accept a JSON array, optionally fenced, or bare JSONL lines."""
        m = _FENCE_RE.search(text)
        if m:
            text = m.group(1)
        text = text.strip()
        try:
            data = json.loads(text)
            if isinstance(data, list):
                return [d for d in data if isinstance(d, dict)]
            if isinstance(data, dict):
                return [data]
        except json.JSONDecodeError:
            pass
        tasks = []
        for line in text.splitlines():
            line = line.strip().rstrip(",")
            if not line.startswith("{"):
                continue
            try:
                d = json.loads(line)
                if isinstance(d, dict):
                    tasks.append(d)
            except json.JSONDecodeError:
                continue
        return tasks

    @staticmethod
    def _validate(item: dict[str, Any], expected_type: str) -> dict[str, Any] | None:
        prompt = item.get("prompt")
        if item.get("type") != expected_type or not isinstance(prompt, str) or len(prompt) < 15:
            return None
        task: dict[str, Any] = {"type": expected_type, "prompt": prompt.strip()}
        if expected_type == "code":
            tests = item.get("tests")
            if not isinstance(tests, str) or "assert" not in tests:
                return None
            try:
                compile(tests, "<tests>", "exec")
            except SyntaxError:
                return None
            task["tests"] = tests.strip()
        return task
