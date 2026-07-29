"""Live check of real LoRA training on this CPU-only machine: a tiny
synthetic JSONL dataset, one training step, verify a checkpoint with
real adapter weights gets written. Downloads the base model
(Qwen2.5-0.5B-Instruct, ~1GB) from Hugging Face on first run.
Run from backend/:  .venv\\Scripts\\python ..\\scripts\\smoke_test_trainer.py
"""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from app.config import load_config
from app.trainer import Trainer


def main():
    cfg = load_config()
    # Route checkpoints to a throwaway dir so this script can be re-run
    # without accumulating gen_* dirs under the real data/checkpoints.
    scratch = Path(__file__).resolve().parent / "tmp_trainer_smoke"
    cfg.raw["student"]["export"]["checkpoints_dir"] = str(scratch / "checkpoints")
    # Keep the run fast: 1 epoch, tiny batch, no grad accumulation.
    cfg.raw["student"]["train"]["epochs"] = 1
    cfg.raw["student"]["train"]["batch_size"] = 1
    cfg.raw["student"]["train"]["grad_accum"] = 1

    dataset_path = scratch / "tiny_train.jsonl"
    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    examples = [
        {"messages": [{"role": "user", "content": "What is 2+2?"}, {"role": "assistant", "content": "4"}]},
        {"messages": [{"role": "user", "content": "Say hello."}, {"role": "assistant", "content": "Hello!"}]},
    ]
    with open(dataset_path, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")

    print(f"base model: {cfg.raw['student']['base_model']}")
    print("downloading base model + training on 2 tiny examples (may take a few minutes on first run)...")
    t0 = time.time()
    trainer = Trainer(cfg)
    result = trainer.train(dataset_path)
    dt = time.time() - t0

    print(f"\ndone in {dt:.1f}s")
    print(f"checkpoint: {result.checkpoint_dir}")
    print(f"generation: {result.generation}, n_examples: {result.n_examples}, train_loss: {result.train_loss}")

    adapter_files = list(result.checkpoint_dir.glob("adapter_*"))
    meta_file = result.checkpoint_dir / "meta.json"
    ok = bool(adapter_files) and meta_file.exists()
    print(f"adapter files found: {[f.name for f in adapter_files]}")
    print("\nTRAINER SMOKE TEST:", "PASS" if ok else "FAIL (no adapter weights written)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
