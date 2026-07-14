"""Evaluator behavior: code execution path + judge parsing. No network."""

import pytest

from app.config import load_config
from app.evaluator import EvalResult, Evaluator


@pytest.fixture
def evaluator():
    return Evaluator(load_config())


CODE_TASK = {
    "type": "code",
    "prompt": "Write add(a, b) returning the sum.",
    "tests": "assert add(2, 3) == 5\nassert add(-1, 1) == 0",
}


def test_code_pass(evaluator):
    resp = "Here you go:\n```python\ndef add(a, b):\n    return a + b\n```"
    r = evaluator.evaluate(CODE_TASK, resp)
    assert r.passed and r.score == 1.0 and r.method == "code_exec"


def test_code_fail(evaluator):
    resp = "```python\ndef add(a, b):\n    return a - b\n```"
    r = evaluator.evaluate(CODE_TASK, resp)
    assert not r.passed and r.score == 0.0


def test_code_infinite_loop_times_out(evaluator):
    evaluator.exec_timeout = 2
    resp = "```python\ndef add(a, b):\n    while True: pass\n```"
    r = evaluator.evaluate(CODE_TASK, resp)
    assert not r.passed and r.details["error"] == "timeout"


def test_no_code_block_and_not_python(evaluator):
    r = evaluator.evaluate(CODE_TASK, "I cannot help with that :) :) (")
    assert not r.passed and "no code block" in r.details["error"]


def test_multiple_blocks_concatenated(evaluator):
    resp = "```python\nimport math\n```\nthen\n```python\ndef add(a, b):\n    return a + b\n```"
    r = evaluator.evaluate(CODE_TASK, resp)
    assert r.passed


def test_judge_score_parsing():
    assert Evaluator._parse_score("Good answer.\nSCORE: 8") == 8
    assert Evaluator._parse_score("SCORE: 10") == 10
    assert Evaluator._parse_score("SCORE: 11") is None
    assert Evaluator._parse_score("I'd rate it eight.") is None


def test_judge_path_uses_client(evaluator):
    class FakeReply:
        text = "Solid.\nSCORE: 9"

    class FakeJudge:
        def complete(self, messages, **kw):
            return FakeReply()

    evaluator.judge_client = FakeJudge()
    r = evaluator.evaluate({"type": "general", "prompt": "Explain DNS."}, "DNS is...")
    assert r.method == "llm_judge" and r.passed and r.score == 0.9


def test_judge_below_min_score_rejected(evaluator):
    class FakeReply:
        text = "Weak.\nSCORE: 5"

    class FakeJudge:
        def complete(self, messages, **kw):
            return FakeReply()

    evaluator.judge_client = FakeJudge()
    r = evaluator.evaluate({"type": "general", "prompt": "Explain DNS."}, "no")
    assert not r.passed and r.score == 0.5
