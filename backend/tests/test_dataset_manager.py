"""DatasetManager: accept/reject rules, dedup, gold-set guard, export."""

import json

import pytest

from app.config import load_config
from app.dataset_manager import DatasetManager
from app.evaluator import EvalResult

PASS = EvalResult(score=1.0, passed=True, method="code_exec")
FAIL = EvalResult(score=0.2, passed=False, method="code_exec")


@pytest.fixture
def dm(tmp_path):
    cfg = load_config()
    cfg.raw["storage"]["db_path"] = str(tmp_path / "test.db")
    cfg.raw["benchmark"]["gold_set_path"] = str(tmp_path / "gold.jsonl")
    return DatasetManager(cfg)


def test_accept_valid_example(dm):
    r = dm.add("Write a function that reverses a string.", "def rev(s): return s[::-1]",
               PASS, "free", "modelA")
    assert r.accepted and r.reason == "accepted" and r.id is not None
    assert dm.counts() == {"free": 1}


def test_reject_below_threshold(dm):
    r = dm.add("Write a broken function.", "def f(): pass", FAIL, "free", "modelA")
    assert not r.accepted and r.reason == "below_threshold"
    assert dm.counts() == {}


def test_reject_exact_duplicate(dm):
    dm.add("Write a function that reverses a string.", "def rev(s): return s[::-1]",
           PASS, "free", "modelA")
    r = dm.add("Write a function that reverses a string.", "def rev(s): return s[::-1]",
               PASS, "free", "modelA")
    assert not r.accepted and r.reason == "duplicate"
    assert dm.counts() == {"free": 1}


def test_reject_near_duplicate(dm):
    dm.add("Write a function that reverses a string in python please",
           "def rev(s): return s[::-1]", PASS, "free", "modelA")
    # Same text, trivially reworded — trigram cosine sim should clear
    # the default 0.92 threshold.
    r = dm.add("Write a function that reverses a string in python please.",
               "def rev(s): return s[::-1]", PASS, "free", "modelA")
    assert not r.accepted and r.reason == "duplicate"


def test_distinct_prompts_both_accepted(dm):
    r1 = dm.add("Write a function that reverses a string.", "...", PASS, "free", "modelA")
    r2 = dm.add("Explain what a hash table is.", "...", PASS, "free", "modelA")
    assert r1.accepted and r2.accepted
    assert dm.counts() == {"free": 2}


def test_gold_set_prompt_rejected(tmp_path):
    gold_path = tmp_path / "gold.jsonl"
    gold_path.write_text(
        json.dumps({"type": "general", "prompt": "What is a deadlock?"}) + "\n",
        encoding="utf-8",
    )
    cfg = load_config()
    cfg.raw["storage"]["db_path"] = str(tmp_path / "test.db")
    cfg.raw["benchmark"]["gold_set_path"] = str(gold_path)
    dm = DatasetManager(cfg)
    r = dm.add("What is a deadlock?", "A deadlock is...", PASS, "free", "modelA")
    assert not r.accepted and r.reason == "in_gold_set"


def test_counts_split_by_tier(dm):
    dm.add("Task one about lists.", "...", PASS, "free", "modelA")
    dm.add("Task two about dicts.", "...", PASS, "paid", "modelB")
    dm.add("Task three about sets.", "...", PASS, "paid", "modelB")
    assert dm.counts() == {"free": 1, "paid": 2}


def test_export_jsonl_writes_chat_format_and_filters_tier(dm, tmp_path):
    dm.add("Task one.", "Answer one.", PASS, "free", "modelA")
    dm.add("Task two.", "Answer two.", PASS, "paid", "modelB")
    out = tmp_path / "export.jsonl"

    n = dm.export_jsonl(out, tiers=("free",))
    assert n == 1
    lines = out.read_text(encoding="utf-8").splitlines()
    record = json.loads(lines[0])
    assert record["messages"] == [
        {"role": "user", "content": "Task one."},
        {"role": "assistant", "content": "Answer one."},
    ]

    n_both = dm.export_jsonl(out, tiers=("free", "paid"))
    assert n_both == 2


def test_human_verified_import(dm, tmp_path):
    src = tmp_path / "human.jsonl"
    src.write_text(
        json.dumps({"prompt": "Handwritten task.", "response": "Handwritten answer.", "teacher_tier": "free"}) + "\n",
        encoding="utf-8",
    )
    results = dm.import_human_verified_jsonl(src)
    assert results[0].accepted
    assert dm.counts() == {"free": 1}


def test_dedup_persists_across_manager_instances(tmp_path):
    cfg = load_config()
    cfg.raw["storage"]["db_path"] = str(tmp_path / "test.db")
    cfg.raw["benchmark"]["gold_set_path"] = str(tmp_path / "gold.jsonl")
    dm1 = DatasetManager(cfg)
    dm1.add("A task about queues.", "...", PASS, "free", "modelA")

    dm2 = DatasetManager(cfg)  # fresh instance, same DB — cache must reload
    r = dm2.add("A task about queues.", "...", PASS, "free", "modelA")
    assert not r.accepted and r.reason == "duplicate"
