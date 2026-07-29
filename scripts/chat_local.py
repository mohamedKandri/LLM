"""Interactive chat with the local student model — no API, no cost.

Loads the base model as configured (student.base_model in config.yaml).
If a trained checkpoint exists (data/checkpoints/gen_NNNN/), loads that
checkpoint's LoRA adapter on top so you're talking to the fine-tuned
model instead of the untrained base. Multi-turn: conversation history
is kept for the session.

Run from backend/:  .venv\\Scripts\\python ..\\scripts\\chat_local.py
Type 'quit' or Ctrl+C to exit.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from app.config import load_config


def latest_checkpoint(checkpoints_dir: Path) -> Path | None:
    gens = sorted(checkpoints_dir.glob("gen_*"), key=lambda p: p.name)
    return gens[-1] if gens else None


def main():
    cfg = load_config()
    base_model = cfg.raw["student"]["base_model"]
    ckpt = latest_checkpoint(cfg.checkpoints_dir)

    print(f"Loading {base_model}" + (f" + adapter {ckpt.name}" if ckpt else " (untrained base model)") + "...")

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(base_model, dtype=torch.float32, device_map="cpu")

    if ckpt:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, ckpt)
    model.eval()

    print("Ready. Type a message (or 'quit' to exit).\n")

    history: list[dict] = []
    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit"):
            break

        history.append({"role": "user", "content": user_input})
        text = tokenizer.apply_chat_template(history, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt")
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=200,
                do_sample=True,
                temperature=0.7,
                pad_token_id=tokenizer.pad_token_id,
            )
        reply = tokenizer.decode(
            output_ids[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
        ).strip()
        print(f"Model: {reply}\n")
        history.append({"role": "assistant", "content": reply})


if __name__ == "__main__":
    main()
