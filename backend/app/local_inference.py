"""Local inference from a trained LoRA checkpoint — the "student" side
of benchmark.py's student_infer_fn.

Loads the base model + a LoRA adapter via transformers+peft (the same
stack trainer.py trains with), not GGUF/Ollama — that's a separate,
optional export path (Trainer.export_gguf) for a lighter, Python-free
end-user runtime once the app is ready to go fully local. This module
is what lets the pipeline actually SCORE a checkpoint today.

Heavy imports (torch/transformers/peft) are deferred into
_ensure_loaded() so importing this module never requires them
installed unless local inference is actually used — same pattern as
trainer.py.

Loading is expensive (model + adapter weights); a LocalStudent loads
once on first .generate() call and reuses the loaded model for every
call after that. Construct one instance and reuse it for an entire
benchmark run or orchestrator session — don't build a fresh one per call.
"""

from __future__ import annotations

from pathlib import Path

from .config import DistillConfig


class LocalStudent:
    def __init__(self, cfg: DistillConfig, checkpoint_dir: str | Path, max_new_tokens: int = 512):
        self.cfg = cfg
        self.checkpoint_dir = Path(checkpoint_dir)
        if not self.checkpoint_dir.exists():
            raise FileNotFoundError(f"Checkpoint directory not found: {self.checkpoint_dir}")
        self.max_new_tokens = max_new_tokens
        self._model = None
        self._tokenizer = None

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer

        base_model = self.cfg.raw["student"]["base_model"]
        tokenizer = AutoTokenizer.from_pretrained(base_model)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        base = AutoModelForCausalLM.from_pretrained(base_model, dtype=torch.float32, device_map="cpu")
        model = PeftModel.from_pretrained(base, self.checkpoint_dir)
        model.eval()

        self._model = model
        self._tokenizer = tokenizer

    def generate(self, prompt: str) -> str:
        """Matches benchmark.StudentInferFn: str -> str."""
        self._ensure_loaded()
        import torch

        text = self._tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True
        )
        inputs = self._tokenizer(text, return_tensors="pt")
        with torch.no_grad():
            output_ids = self._model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,  # greedy — benchmark scoring should be reproducible
                pad_token_id=self._tokenizer.pad_token_id,
            )
        new_tokens = output_ids[0][inputs["input_ids"].shape[1] :]
        return self._tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
