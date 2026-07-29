"""retrain_flow: wiring logic only — dataset export -> trainer ->
student -> benchmark, with fakes standing in for the real (heavy) ML
classes. Verifies the glue, not real training/inference."""

import pytest

from app.config import load_config
from app.dataset_manager import DatasetManager
from app.evaluator import Evaluator
from app.orchestrator import Orchestrator
from app.retrain_flow import make_retrain_fn


class FakeReply:
    tier = "free"
    model = "fake/model"

    def __init__(self, text="fake answer"):
        self.text = text


class FakeTeacher:
    def complete(self, messages, **kw):
        return FakeReply()


class FakeJudge:
    def complete(self, messages, **kw):
        return FakeReply("Good.\nSCORE: 9")


class FakeTrainResult:
    def __init__(self, checkpoint_dir):
        self.checkpoint_dir = checkpoint_dir
        self.generation = 1
        self.n_examples = 3


class FakeTrainer:
    calls: list = []

    def __init__(self, cfg):
        self.cfg = cfg

    def train(self, jsonl_path):
        FakeTrainer.calls.append(jsonl_path)
        return FakeTrainResult(checkpoint_dir="/fake/checkpoint/gen_0001")


class FakeStudent:
    instances: list = []

    def __init__(self, cfg, checkpoint_dir):
        self.cfg = cfg
        self.checkpoint_dir = checkpoint_dir
        FakeStudent.instances.append(self)

    def generate(self, prompt):
        return "fake student answer"


class FakeBenchmarkResult:
    ratio = 0.42
    teacher_tier = "free"
    target = 0.9
    target_met = False


class FakeBenchmark:
    calls: list = []

    def __init__(self, cfg, teacher, evaluator):
        self.cfg = cfg
        self.teacher = teacher
        self.evaluator = evaluator

    def run(self, student_infer_fn):
        FakeBenchmark.calls.append(student_infer_fn)
        return FakeBenchmarkResult()


@pytest.fixture(autouse=True)
def _reset_fakes():
    FakeTrainer.calls = []
    FakeStudent.instances = []
    FakeBenchmark.calls = []


@pytest.fixture
def orch(tmp_path):
    cfg = load_config()
    cfg.raw["storage"]["db_path"] = str(tmp_path / "test.db")
    evaluator = Evaluator(cfg, judge_client=FakeJudge())
    dataset = DatasetManager(cfg)
    return Orchestrator(
        cfg,
        teacher=FakeTeacher(),
        judge=FakeJudge(),
        evaluator=evaluator,
        prompt_generator=object(),  # unused by retrain_fn
        dataset=dataset,
    )


def test_retrain_skipped_when_dataset_empty(orch):
    retrain = make_retrain_fn(orch, FakeTrainer, FakeStudent, FakeBenchmark)
    retrain()
    assert FakeTrainer.calls == []
    assert FakeStudent.instances == []
    assert FakeBenchmark.calls == []


def test_retrain_wires_trainer_student_benchmark_in_order(orch):
    from app.evaluator import EvalResult

    orch.dataset.add("What is 2+2?", "4", EvalResult(1.0, True, "llm_judge"), "free", "fake/model")

    retrain = make_retrain_fn(orch, FakeTrainer, FakeStudent, FakeBenchmark)
    retrain()

    assert len(FakeTrainer.calls) == 1  # trained on the exported JSONL
    assert len(FakeStudent.instances) == 1
    assert FakeStudent.instances[0].checkpoint_dir == "/fake/checkpoint/gen_0001"  # from trainer's result
    assert len(FakeBenchmark.calls) == 1
    assert FakeBenchmark.calls[0] == FakeStudent.instances[0].generate  # student.generate passed through


def test_retrain_reuses_orchestrators_own_teacher_and_evaluator(orch, monkeypatch):
    from app.evaluator import EvalResult

    orch.dataset.add("What is 2+2?", "4", EvalResult(1.0, True, "llm_judge"), "free", "fake/model")

    captured = {}
    real_init = FakeBenchmark.__init__

    def spy_init(self, cfg, teacher, evaluator):
        captured["teacher"] = teacher
        captured["evaluator"] = evaluator
        real_init(self, cfg, teacher, evaluator)

    monkeypatch.setattr(FakeBenchmark, "__init__", spy_init)
    retrain = make_retrain_fn(orch, FakeTrainer, FakeStudent, FakeBenchmark)
    retrain()

    assert captured["teacher"] is orch.teacher
    assert captured["evaluator"] is orch.evaluator
