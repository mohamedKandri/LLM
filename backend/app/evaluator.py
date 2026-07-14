"""Scores teacher responses before they're allowed into the training set.

Two paths, dispatched on the task's type:

- "code":    extract the code block, run the task's tests in a
             subprocess with a timeout. Pass = 1.0, fail = 0.0.
- "general": second LLM call as judge (RLAIF), 1-10 rubric score,
             normalized to 0-1.

Only results at or above evaluation.accept_threshold get saved by
dataset_manager. Scores are always normalized 0-1 so the threshold
means the same thing for both paths.
"""

from __future__ import annotations

import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .config import DistillConfig
from .teacher_client import TeacherClient, TeacherClientError

_CODE_BLOCK_RE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.DOTALL)
_JUDGE_SCORE_RE = re.compile(r"SCORE:\s*(\d{1,2})")

JUDGE_SYSTEM_PROMPT = """\
You are a strict evaluator of assistant responses. Score the RESPONSE to the
PROMPT on a 1-10 scale:
1-3  = wrong, harmful, or ignores the prompt
4-6  = partially correct, incomplete, or poorly explained
7-8  = correct and clear, minor flaws
9-10 = correct, complete, well-structured, nothing to fix

Judge only what is written. Do not reward length. Reply with a one-sentence
justification, then on the final line exactly: SCORE: <n>"""


@dataclass
class EvalResult:
    score: float                 # normalized 0-1
    passed: bool                 # score >= accept threshold
    method: str                  # "code_exec" | "llm_judge"
    details: dict[str, Any] = field(default_factory=dict)


class Evaluator:
    def __init__(self, cfg: DistillConfig, judge_client: TeacherClient | None = None):
        self.cfg = cfg
        self.threshold = float(cfg.evaluation["accept_threshold"])
        self.exec_timeout = float(cfg.evaluation["code_exec_timeout_s"])
        # Judge client is injected so orchestrator can share one budget ledger.
        self.judge_client = judge_client

    def evaluate(self, task: dict[str, Any], response_text: str) -> EvalResult:
        """`task` needs "type" ("code"|"general"), "prompt", and for code
        tasks a "tests" field: Python source that asserts against the
        solution (runs in the same file, after the extracted code)."""
        if task.get("type") == "code":
            return self._eval_code(task, response_text)
        return self._eval_with_judge(task, response_text)

    # -- code path -----------------------------------------------------
    def _eval_code(self, task: dict[str, Any], response_text: str) -> EvalResult:
        code = self._extract_code(response_text)
        if code is None:
            return EvalResult(0.0, False, "code_exec", {"error": "no code block found"})

        tests = task.get("tests", "")
        if not tests:
            return EvalResult(0.0, False, "code_exec", {"error": "task has no tests"})

        # Solution + tests in one file; tests are plain asserts / raises.
        program = f"{code}\n\n# --- tests ---\n{tests}\n"
        with tempfile.TemporaryDirectory() as tmp:
            script = Path(tmp) / "candidate.py"
            script.write_text(program, encoding="utf-8")
            try:
                proc = subprocess.run(
                    [sys.executable, "-I", str(script)],  # -I: isolated, ignores env/site
                    capture_output=True,
                    text=True,
                    timeout=self.exec_timeout,
                    cwd=tmp,
                )
            except subprocess.TimeoutExpired:
                return EvalResult(0.0, False, "code_exec", {"error": "timeout"})

        passed = proc.returncode == 0
        return EvalResult(
            score=1.0 if passed else 0.0,
            passed=passed,
            method="code_exec",
            details={
                "returncode": proc.returncode,
                "stderr": proc.stderr[-2000:],
                "stdout": proc.stdout[-500:],
            },
        )

    @staticmethod
    def _extract_code(text: str) -> str | None:
        blocks = _CODE_BLOCK_RE.findall(text)
        if blocks:
            # Concatenate: teachers often split imports and solution.
            return "\n\n".join(b.strip() for b in blocks)
        # Bare-code fallback: response that compiles as Python is accepted as-is.
        try:
            compile(text, "<candidate>", "exec")
            return text
        except SyntaxError:
            return None

    # -- judge path (RLAIF) ---------------------------------------------
    def _eval_with_judge(self, task: dict[str, Any], response_text: str) -> EvalResult:
        if self.judge_client is None:
            raise TeacherClientError("Evaluator needs a judge_client for general tasks")

        messages = [
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"PROMPT:\n{task['prompt']}\n\nRESPONSE:\n{response_text}",
            },
        ]
        # One re-ask if the judge doesn't follow the SCORE format.
        for attempt in range(2):
            reply = self.judge_client.complete(messages, purpose="judge", temperature=0.0)
            raw_score = self._parse_score(reply.text)
            if raw_score is not None:
                min_score = self.cfg.judge_min_score
                score = raw_score / 10.0
                return EvalResult(
                    score=score,
                    passed=raw_score >= min_score and score >= self.threshold,
                    method="llm_judge",
                    details={"raw_score": raw_score, "justification": reply.text[:1000]},
                )
            messages.append({"role": "assistant", "content": reply.text})
            messages.append(
                {"role": "user", "content": "Reply again ending with exactly: SCORE: <1-10>"}
            )
        return EvalResult(0.0, False, "llm_judge", {"error": "unparseable judge reply"})

    @staticmethod
    def _parse_score(text: str) -> int | None:
        m = _JUDGE_SCORE_RE.search(text)
        if not m:
            return None
        score = int(m.group(1))
        return score if 1 <= score <= 10 else None
