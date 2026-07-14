"""PromptGenerator parsing/validation/dedup — fake teacher, no network."""

import json

import pytest

from app.config import load_config
from app.prompt_generator import PromptGenerator, load_seeds


class FakeReply:
    tier = "free"
    model = "fake/model"

    def __init__(self, text):
        self.text = text


class FakeTeacher:
    """Returns queued replies; records how many calls were made."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = 0

    def complete(self, messages, **kw):
        self.calls += 1
        if self.replies:
            return FakeReply(self.replies.pop(0))
        return FakeReply("[]")


@pytest.fixture
def cfg():
    return load_config()


@pytest.fixture
def seeds(cfg):
    return load_seeds(cfg)


CODE_TASK = {
    "type": "code",
    "prompt": "Write a Python function `gcd(a, b)` returning the greatest common divisor.",
    "tests": "assert gcd(12, 8) == 4\nassert gcd(7, 5) == 1",
}
GENERAL_TASK = {"type": "general", "prompt": "What is a deadlock and how do you avoid one?"}


def test_seeds_load_and_are_well_formed(seeds):
    assert len(seeds) == 15
    for s in seeds:
        assert s["type"] in ("code", "general")
        if s["type"] == "code":
            compile(s["tests"], "<t>", "exec")  # every seed's tests must be valid Python


def test_generate_tags_metadata(cfg, seeds):
    teacher = FakeTeacher([json.dumps([CODE_TASK]), json.dumps([GENERAL_TASK])])
    gen = PromptGenerator(cfg, teacher, seeds)
    tasks = gen.generate(n=2)
    assert len(tasks) == 2
    for t in tasks:
        assert t["teacher_tier"] == "free"
        assert t["teacher_model"] == "fake/model"
        assert len(t["seed_ids"]) > 0


def test_parses_fenced_json_and_jsonl():
    fenced = "Here you go:\n```json\n" + json.dumps([GENERAL_TASK]) + "\n```"
    assert PromptGenerator._parse_tasks(fenced) == [GENERAL_TASK]
    jsonl = json.dumps(GENERAL_TASK) + "\n" + json.dumps(CODE_TASK)
    assert len(PromptGenerator._parse_tasks(jsonl)) == 2
    assert PromptGenerator._parse_tasks("Sorry, I can't do that.") == []


def test_validate_rejects_bad_items():
    v = PromptGenerator._validate
    assert v({"type": "code", "prompt": "Write gcd function properly", "tests": "no asserts here"}, "code") is None
    assert v({"type": "code", "prompt": "Write gcd function properly", "tests": "assert gcd(1,"}, "code") is None
    assert v({"type": "general", "prompt": "short"}, "general") is None
    assert v({"type": "general", "prompt": GENERAL_TASK["prompt"]}, "code") is None
    assert v(CODE_TASK, "code") is not None


def test_dedup_against_seeds_and_self(cfg, seeds):
    dup_of_seed = {
        "type": "general",
        "prompt": "What's the difference between authentication and authorization?",
    }
    teacher = FakeTeacher(
        [
            json.dumps([dup_of_seed, GENERAL_TASK, GENERAL_TASK]),  # seed dup + self dup
        ]
    )
    gen = PromptGenerator(cfg, teacher, seeds)
    cfg.raw["generation"]["code_fraction"] = 0.0
    tasks = gen.generate(n=3)
    prompts = [t["prompt"] for t in tasks]
    assert prompts.count(GENERAL_TASK["prompt"]) == 1
    assert dup_of_seed["prompt"] not in prompts


def test_call_budget_capped_on_garbage_teacher(cfg, seeds):
    teacher = FakeTeacher([])  # always returns empty array
    gen = PromptGenerator(cfg, teacher, seeds)
    tasks = gen.generate(n=10)
    assert tasks == []
    assert teacher.calls <= 2 * -(-10 // int(cfg.generation["tasks_per_call"])) + 1
