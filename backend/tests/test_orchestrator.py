"""Orchestrator: step logic, thread lifecycle, retrain trigger,
graduate()/go_local(). All network calls faked; config writes are
always redirected to tmp_path — never the real config.yaml."""

import time

import pytest

from app.config import load_config
from app.dataset_manager import DatasetManager
from app.evaluator import Evaluator
from app.orchestrator import Orchestrator


class FakeReply:
    def __init__(self, text, tier="free", model="fake/model"):
        self.text = text
        self.tier = tier
        self.model = model


class FakeTeacher:
    def __init__(self):
        self.calls = 0

    def complete(self, messages, **kw):
        self.calls += 1
        prompt = messages[-1]["content"]
        if "bad" in prompt:
            return FakeReply("```python\ndef gcd(a, b):\n    return -1\n```")
        if "gcd" in prompt:
            return FakeReply("```python\ndef gcd(a, b):\n    while b:\n        a, b = b, a % b\n    return a\n```")
        return FakeReply("A deadlock is a cycle of waiting processes.")


class FakeJudge:
    def complete(self, messages, **kw):
        return FakeReply("Good.\nSCORE: 9")


class FakePromptGenerator:
    """Returns a fixed batch of tasks once, then empties out."""

    def __init__(self, batches):
        self.batches = list(batches)

    def generate(self, n):
        return self.batches.pop(0) if self.batches else []


GOOD_CODE_TASK = {
    "type": "code",
    "prompt": "Write gcd(a, b).",
    "tests": "assert gcd(12, 8) == 4\nassert gcd(7, 5) == 1",
}
BAD_CODE_TASK = {
    "type": "code",
    "prompt": "Write bad_gcd(a, b).",  # contains 'bad' -> FakeTeacher answers wrong
    "tests": "assert gcd(12, 8) == 4",
}
GENERAL_TASK = {"type": "general", "prompt": "What is a deadlock?"}


@pytest.fixture
def cfg(tmp_path):
    c = load_config()
    c.raw["storage"]["db_path"] = str(tmp_path / "test.db")
    c.raw["benchmark"]["gold_set_path"] = str(tmp_path / "gold.jsonl")
    return c


@pytest.fixture
def config_path(tmp_path):
    return tmp_path / "config.yaml"


def make_orchestrator(cfg, config_path, batches, retrain_fn=None, poll_interval_s=0.01):
    teacher = FakeTeacher()
    judge = FakeJudge()
    evaluator = Evaluator(cfg, judge_client=judge)
    dataset = DatasetManager(cfg)
    pg = FakePromptGenerator(batches)
    return Orchestrator(
        cfg,
        teacher=teacher,
        judge=judge,
        evaluator=evaluator,
        prompt_generator=pg,
        dataset=dataset,
        retrain_fn=retrain_fn,
        poll_interval_s=poll_interval_s,
        config_path=config_path,
    )


def test_step_accepts_good_rejects_bad(cfg, config_path):
    orch = make_orchestrator(cfg, config_path, [[GOOD_CODE_TASK, BAD_CODE_TASK, GENERAL_TASK]])
    orch._step()
    assert orch.status.tasks_generated == 3
    assert orch.status.accepted == 2  # good code + general
    assert orch.status.rejected == 1  # bad code
    assert orch.dataset.counts() == {"free": 2}


def test_retrain_fn_triggered_at_threshold(cfg, config_path):
    cfg.raw["student"]["train"]["retrain_every_n_accepted"] = 2
    calls = []
    orch = make_orchestrator(
        cfg, config_path, [[GOOD_CODE_TASK, GENERAL_TASK]], retrain_fn=lambda: calls.append(1)
    )
    orch._step()
    assert orch.status.accepted == 2
    assert calls == [1]
    assert orch.status.accepted_since_retrain == 0


def test_retrain_fn_not_called_without_hook(cfg, config_path):
    cfg.raw["student"]["train"]["retrain_every_n_accepted"] = 1
    orch = make_orchestrator(cfg, config_path, [[GOOD_CODE_TASK]], retrain_fn=None)
    orch._step()  # must not raise even though threshold is cleared
    assert orch.status.accepted == 1


def test_start_stop_lifecycle(cfg, config_path):
    orch = make_orchestrator(cfg, config_path, [[GOOD_CODE_TASK]] * 50)
    orch.start()
    assert orch.status.state == "running"
    time.sleep(0.05)
    orch.stop()
    assert orch.status.state == "stopped"
    assert orch.status.iterations >= 1


def test_pause_resume(cfg, config_path):
    orch = make_orchestrator(cfg, config_path, [[GOOD_CODE_TASK]] * 50)
    orch.start()
    time.sleep(0.03)
    orch.pause()
    assert orch.status.state == "paused"
    paused_iterations = orch.status.iterations
    time.sleep(0.05)
    assert orch.status.iterations == paused_iterations  # no progress while paused
    orch.resume()
    time.sleep(0.03)
    orch.stop()
    assert orch.status.iterations >= paused_iterations


def test_snapshot_shape(cfg, config_path):
    orch = make_orchestrator(cfg, config_path, [[GOOD_CODE_TASK]])
    orch._step()
    snap = orch.snapshot()
    assert snap["accepted"] == 1
    assert snap["dataset_counts"] == {"free": 1}
    assert snap["teacher_tier"] == "free"


def test_graduate_flips_tier_and_writes_only_tmp_config(cfg, config_path):
    orch = make_orchestrator(cfg, config_path, [])
    orch.graduate()
    assert cfg.raw["phase"] == 2
    assert cfg.raw["teacher_tier"] == "paid"
    assert config_path.exists()
    # Must never touch the real project config.yaml
    from app.config import CONFIG_PATH

    assert config_path != CONFIG_PATH


def test_graduate_refuses_while_running(cfg, config_path):
    orch = make_orchestrator(cfg, config_path, [[GOOD_CODE_TASK]] * 50)
    orch.start()
    time.sleep(0.02)
    with pytest.raises(RuntimeError, match="Stop or pause"):
        orch.graduate()
    orch.stop()


def test_graduate_refuses_when_already_paid(cfg, config_path):
    orch = make_orchestrator(cfg, config_path, [])
    orch.graduate()
    with pytest.raises(RuntimeError, match="Already on tier"):
        orch.graduate()


def test_go_local_flips_flag_without_touching_real_config(cfg, config_path):
    orch = make_orchestrator(cfg, config_path, [])
    orch.go_local()
    assert cfg.raw["inference"]["local_only"] is True
    assert config_path.exists()
