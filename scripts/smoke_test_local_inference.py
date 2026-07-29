"""Live check: train a tiny checkpoint, then load it back via
LocalStudent and generate a REAL completion from it. No API calls —
this only proves the student half of the pipeline (trainer.py's output
-> local_inference.py's input) actually works.
Run from backend/:  .venv\\Scripts\\python ..\\scripts\\smoke_test_local_inference.py
"""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from app.config import load_config
from app.local_inference import LocalStudent
from app.trainer import Trainer


def main():
    cfg = load_config()
    scratch = Path(__file__).resolve().parent / "tmp_local_inference_smoke"
    cfg.raw["student"]["export"]["checkpoints_dir"] = str(scratch / "checkpoints")
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

    print(f"[1] training a tiny checkpoint ({cfg.raw['student']['base_model']})...")
    t0 = time.time()
    trainer = Trainer(cfg)
    result = trainer.train(dataset_path)
    print(f"    done in {time.time() - t0:.1f}s -> {result.checkpoint_dir}")

    print("\n[2] loading the checkpoint via LocalStudent and generating...")
    student = LocalStudent(cfg, result.checkpoint_dir, max_new_tokens=40)
    t0 = time.time()
    answer = student.generate("What is the capital of France?")
    dt = time.time() - t0

    print(f"    done in {dt:.1f}s")
    print(f"    prompt:  What is the capital of France?")
    print(f"    answer:  {answer!r}")

    ok = isinstance(answer, str) and len(answer) > 0
    print("\nLOCAL INFERENCE SMOKE TEST:", "PASS" if ok else "FAIL (empty/invalid output)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
