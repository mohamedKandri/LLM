"""LoRA fine-tuning of the student model.

CPU-sized by config.yaml's student.base_model (this dev machine has no
discrete GPU) — plain HuggingFace transformers + peft (LoRA) + trl
(SFTTrainer), not Unsloth (which requires CUDA). Trains on the JSONL
dataset_manager.export_jsonl() produces, saves a LoRA adapter checkpoint
per generation (data/checkpoints/gen_NNNN/), and can merge the adapter
into the base model as input for GGUF export.

Heavy ML imports (torch/transformers/peft/trl) are deferred into train()
so importing this module — e.g. from orchestrator.py, which imports it
whether or not a retrain is ever triggered — never requires those
packages installed unless a training run actually happens.

GGUF conversion (for Ollama/llama.cpp local inference) is intentionally
NOT reimplemented here — it shells out to llama.cpp's own
convert_hf_to_gguf.py, the standard, correctness-tested tool for mapping
a model's architecture into GGUF's metadata format. Configure its path
via trainer.llama_cpp_convert_script in config.yaml; export_gguf() fails
with an actionable error if that isn't set up. Training works without it.

Interface:
    trainer = Trainer(cfg)
    result = trainer.train(jsonl_path)             # -> TrainResult
    merged = trainer.merge_and_save(result.checkpoint_dir, out_dir)
    trainer.export_gguf(merged, out_path)          # requires llama.cpp configured
Triggered by orchestrator.retrain_fn every
student.train.retrain_every_n_accepted accepted examples, or manually.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from .config import DistillConfig


@dataclass
class TrainResult:
    checkpoint_dir: Path
    generation: int
    n_examples: int
    train_loss: float | None


class Trainer:
    def __init__(self, cfg: DistillConfig):
        self.cfg = cfg
        self.checkpoints_dir = cfg.checkpoints_dir
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    def next_generation(self) -> int:
        nums = []
        for p in self.checkpoints_dir.glob("gen_*"):
            try:
                nums.append(int(p.name.split("_")[1]))
            except (IndexError, ValueError):
                continue
        return (max(nums) + 1) if nums else 1

    def train(self, jsonl_path: str | Path) -> TrainResult:
        examples = self._load_jsonl(Path(jsonl_path))
        if not examples:
            raise ValueError(f"No training examples found in {jsonl_path}")

        import torch
        from datasets import Dataset
        from peft import LoraConfig, get_peft_model
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from trl import SFTConfig, SFTTrainer

        student_cfg = self.cfg.raw["student"]
        base_model = student_cfg["base_model"]
        lora_cfg = student_cfg["lora"]
        train_cfg = student_cfg["train"]

        tokenizer = AutoTokenizer.from_pretrained(base_model)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            base_model, torch_dtype=torch.float32, device_map="cpu"
        )
        model = get_peft_model(
            model,
            LoraConfig(
                r=lora_cfg["r"],
                lora_alpha=lora_cfg["alpha"],
                lora_dropout=lora_cfg["dropout"],
                target_modules=list(lora_cfg["target_modules"]),
                task_type="CAUSAL_LM",
            ),
        )

        texts = [
            tokenizer.apply_chat_template(ex["messages"], tokenize=False, add_generation_prompt=False)
            for ex in examples
        ]
        dataset = Dataset.from_dict({"text": texts})

        generation = self.next_generation()
        out_dir = self.checkpoints_dir / f"gen_{generation:04d}"
        out_dir.mkdir(parents=True, exist_ok=True)

        sft_config = SFTConfig(
            output_dir=str(out_dir),
            num_train_epochs=train_cfg["epochs"],
            per_device_train_batch_size=train_cfg["batch_size"],
            gradient_accumulation_steps=train_cfg["grad_accum"],
            learning_rate=train_cfg["lr"],
            logging_steps=max(1, len(dataset) // 10),
            save_strategy="no",  # we save explicitly below, once, at the end
            report_to=[],
            dataset_text_field="text",
            use_cpu=True,
        )
        sft_trainer = SFTTrainer(
            model=model,
            args=sft_config,
            train_dataset=dataset,
            processing_class=tokenizer,
        )
        train_output = sft_trainer.train()

        sft_trainer.model.save_pretrained(out_dir)
        tokenizer.save_pretrained(out_dir)
        train_loss = getattr(train_output, "training_loss", None)
        meta = {
            "generation": generation,
            "base_model": base_model,
            "n_examples": len(examples),
            "created_at": datetime.now(timezone.utc).isoformat(),
            "train_loss": train_loss,
        }
        (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

        return TrainResult(
            checkpoint_dir=out_dir,
            generation=generation,
            n_examples=len(examples),
            train_loss=train_loss,
        )

    @staticmethod
    def _load_jsonl(path: Path) -> list[dict]:
        examples = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    examples.append(json.loads(line))
        return examples

    # ------------------------------------------------------------------
    def merge_and_save(self, checkpoint_dir: str | Path, out_dir: str | Path) -> Path:
        """Merge a LoRA adapter checkpoint into the base model weights,
        producing a standalone HF model directory — llama.cpp's converter
        needs full weights, it doesn't understand LoRA adapters directly."""
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer

        checkpoint_dir = Path(checkpoint_dir)
        out_dir = Path(out_dir)
        base_model = self.cfg.raw["student"]["base_model"]

        base = AutoModelForCausalLM.from_pretrained(base_model, torch_dtype=torch.float32, device_map="cpu")
        merged = PeftModel.from_pretrained(base, checkpoint_dir).merge_and_unload()
        out_dir.mkdir(parents=True, exist_ok=True)
        merged.save_pretrained(out_dir)
        AutoTokenizer.from_pretrained(checkpoint_dir).save_pretrained(out_dir)
        return out_dir

    def export_gguf(self, merged_model_dir: str | Path, out_path: str | Path) -> Path:
        """Convert a merged HF model directory to GGUF. Requires a
        llama.cpp checkout; see trainer.llama_cpp_convert_script in
        config.yaml."""
        script = self.cfg.raw.get("trainer", {}).get("llama_cpp_convert_script")
        if not script or not Path(script).exists():
            raise FileNotFoundError(
                "GGUF export needs llama.cpp's convert_hf_to_gguf.py. Clone "
                "https://github.com/ggerganov/llama.cpp and set "
                "trainer.llama_cpp_convert_script in config.yaml to its path."
            )
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["python", script, str(merged_model_dir), "--outfile", str(out_path), "--outtype", "q8_0"],
            check=True,
        )
        return out_path
