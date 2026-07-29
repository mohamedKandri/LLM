"""Trainer: checkpoint numbering, JSONL loading, GGUF-guard logic.
No torch/transformers/peft/trl needed — those are only imported inside
train()/merge_and_save(), which real model training tests exercise
separately (see scripts/smoke_test_trainer.py, not part of this suite)."""

import json

import pytest

from app.config import load_config
from app.trainer import Trainer


@pytest.fixture
def cfg(tmp_path):
    c = load_config()
    c.raw["student"]["export"]["checkpoints_dir"] = str(tmp_path / "checkpoints")
    return c


def test_checkpoints_dir_created(cfg, tmp_path):
    Trainer(cfg)
    assert (tmp_path / "checkpoints").is_dir()


def test_next_generation_starts_at_one(cfg):
    trainer = Trainer(cfg)
    assert trainer.next_generation() == 1


def test_next_generation_increments_past_existing(cfg):
    trainer = Trainer(cfg)
    (trainer.checkpoints_dir / "gen_0001").mkdir()
    (trainer.checkpoints_dir / "gen_0002").mkdir()
    assert trainer.next_generation() == 3


def test_next_generation_ignores_malformed_dirs(cfg):
    trainer = Trainer(cfg)
    (trainer.checkpoints_dir / "gen_0001").mkdir()
    (trainer.checkpoints_dir / "not_a_gen_dir").mkdir()
    (trainer.checkpoints_dir / "gen_oops").mkdir()
    assert trainer.next_generation() == 2


def test_load_jsonl_skips_blank_lines(cfg, tmp_path):
    trainer = Trainer(cfg)
    path = tmp_path / "data.jsonl"
    path.write_text(
        json.dumps({"messages": [{"role": "user", "content": "hi"}]}) + "\n\n"
        + json.dumps({"messages": [{"role": "user", "content": "bye"}]}) + "\n",
        encoding="utf-8",
    )
    examples = trainer._load_jsonl(path)
    assert len(examples) == 2


def test_train_raises_on_empty_dataset(cfg, tmp_path):
    trainer = Trainer(cfg)
    empty = tmp_path / "empty.jsonl"
    empty.write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="No training examples"):
        trainer.train(empty)


def test_export_gguf_raises_actionable_error_without_llama_cpp(cfg, tmp_path):
    cfg.raw["trainer"] = {"llama_cpp_convert_script": None}
    trainer = Trainer(cfg)
    with pytest.raises(FileNotFoundError, match="llama.cpp"):
        trainer.export_gguf(tmp_path / "merged", tmp_path / "out.gguf")


def test_export_gguf_raises_if_configured_script_missing(cfg, tmp_path):
    cfg.raw["trainer"] = {"llama_cpp_convert_script": str(tmp_path / "does_not_exist.py")}
    trainer = Trainer(cfg)
    with pytest.raises(FileNotFoundError, match="llama.cpp"):
        trainer.export_gguf(tmp_path / "merged", tmp_path / "out.gguf")
