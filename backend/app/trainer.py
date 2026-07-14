"""LoRA fine-tuning of the student model (Unsloth + peft + trl).

STUB — implemented once the dataset pipeline produces real JSONL.

Planned interface:
    trainer = Trainer(cfg)
    ckpt = trainer.train(jsonl_path)       # returns checkpoint dir, one per generation
    trainer.export_gguf(ckpt)              # for Ollama / llama.cpp inference
Triggered by orchestrator every student.train.retrain_every_n_accepted
accepted examples, or manually from the UI.
"""

from __future__ import annotations

from .config import DistillConfig


class Trainer:
    def __init__(self, cfg: DistillConfig):
        self.cfg = cfg
        raise NotImplementedError
