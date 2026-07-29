"""Benchmark: student-vs-teacher scoring, caching, history, target_met.
Fake teacher + fake student inference — no network."""

import json

import pytest

from app.benchmark import Benchmark, read_history
from app.config import load_config
from app.evaluator import Evaluator


class FakeEndpoint:
    model = "fake-teacher/v1"


class FakeReply:
    def __init__(self, text):
        self.text = text


class FakeTeacher:
    """Always answers code tasks correctly; general tasks get a fixed
    'good' answer. Counts calls so the cache can be verified."""

    endpoint = FakeEndpoint()

    def __init__(self):
        self.calls = 0

    def complete(self, messages, **kw):
        self.calls += 1
        prompt = messages[-1]["content"]
        if "gcd" in prompt:
            return FakeReply("```python\ndef gcd(a, b):\n    while b:\n        a, b = b, a % b\n    return a\n```")
        return FakeReply("A deadlock is a cycle of processes each waiting on a resource held by the next.")


class FakeJudge:
    """Judges any response as 9/10 — good enough for a passing benchmark."""

    def complete(self, messages, **kw):
        return FakeReply("Solid answer.\nSCORE: 9")


@pytest.fixture
def gold_path(tmp_path):
    path = tmp_path / "gold.jsonl"
    entries = [
        {"id": "g-code", "type": "code", "prompt": "Write gcd(a, b).",
         "tests": "assert gcd(12, 8) == 4\nassert gcd(7, 5) == 1"},
        {"id": "g-general", "type": "general", "prompt": "What is a deadlock?",
         "reference": "A cycle of waiting processes."},
    ]
    path.write_text("\n".join(json.dumps(e) for e in entries) + "\n", encoding="utf-8")
    return path


@pytest.fixture
def cfg(tmp_path, gold_path):
    c = load_config()
    c.raw["storage"]["db_path"] = str(tmp_path / "test.db")
    c.raw["benchmark"]["gold_set_path"] = str(gold_path)
    return c


@pytest.fixture
def bench(cfg):
    teacher = FakeTeacher()
    evaluator = Evaluator(cfg, judge_client=FakeJudge())
    return Benchmark(cfg, teacher, evaluator), teacher


def perfect_student(prompt: str) -> str:
    if "gcd" in prompt:
        return "```python\ndef gcd(a, b):\n    while b:\n        a, b = b, a % b\n    return a\n```"
    return "A deadlock is a cycle of processes each waiting on a resource held by the next."


def broken_student(prompt: str) -> str:
    if "gcd" in prompt:
        return "```python\ndef gcd(a, b):\n    return a + b\n```"  # wrong
    return "I don't know."


def test_run_matches_teacher_when_student_is_as_good(bench):
    b, _ = bench
    result = b.run(perfect_student)
    assert result.student_score == pytest.approx(result.teacher_score)
    assert result.ratio == pytest.approx(1.0)
    assert result.target_met  # 1.0 >= cfg default target (0.9 for free tier)


def test_run_flags_target_not_met_for_weak_student(bench):
    b, _ = bench
    result = b.run(broken_student)
    assert result.student_score < result.teacher_score
    assert not result.target_met


def test_teacher_answers_are_cached_across_runs(bench):
    b, teacher = bench
    b.run(perfect_student)
    calls_after_first = teacher.calls
    b.run(perfect_student)
    # Second run should hit the cache — same number of teacher calls
    assert teacher.calls == calls_after_first


def test_history_filters_by_tier_and_orders(bench):
    b, _ = bench
    b.run(perfect_student)
    b.run(broken_student)
    hist = b.history(tier="free")
    assert len(hist) == 2
    assert hist[0]["id"] < hist[1]["id"]
    assert all(h["teacher_tier"] == "free" for h in hist)
    assert b.history(tier="paid") == []


def test_target_met_reflects_most_recent_run(bench):
    b, _ = bench
    assert not b.target_met()  # no runs yet
    b.run(broken_student)
    assert not b.target_met()
    b.run(perfect_student)
    assert b.target_met()  # latest run is what matters


def test_missing_gold_set_raises(tmp_path):
    cfg = load_config()
    cfg.raw["storage"]["db_path"] = str(tmp_path / "test.db")
    cfg.raw["benchmark"]["gold_set_path"] = str(tmp_path / "does_not_exist.jsonl")
    with pytest.raises(FileNotFoundError):
        Benchmark(cfg, FakeTeacher(), Evaluator(cfg, judge_client=FakeJudge()))


def test_read_history_empty_before_any_table_exists(tmp_path):
    # No Benchmark has ever run against this DB — table doesn't exist yet.
    assert read_history(tmp_path / "fresh.db") == []


def test_read_history_matches_benchmark_history(bench, cfg):
    b, _ = bench
    b.run(perfect_student)
    assert read_history(cfg.db_path) == b.history()
    assert read_history(cfg.db_path, tier="paid") == []
