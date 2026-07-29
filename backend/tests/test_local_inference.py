"""LocalStudent: non-ML logic only (missing-checkpoint guard, lazy
loading). Real generation is exercised live via
scripts/smoke_test_local_inference.py — no torch/transformers/peft
needed here, those imports are deferred into _ensure_loaded()."""

import pytest

from app.config import load_config
from app.local_inference import LocalStudent


def test_missing_checkpoint_raises_immediately(tmp_path):
    cfg = load_config()
    with pytest.raises(FileNotFoundError, match="Checkpoint directory not found"):
        LocalStudent(cfg, tmp_path / "does_not_exist")


def test_construction_does_not_load_model(tmp_path):
    """Construction must be cheap — the model loads lazily on first
    .generate() call, not at __init__ time."""
    cfg = load_config()
    ckpt = tmp_path / "gen_0001"
    ckpt.mkdir()
    student = LocalStudent(cfg, ckpt)
    assert student._model is None
    assert student._tokenizer is None
