"""Probe OpenRouter :free models with a tiny request each; report which respond.

Standalone (raw requests, no ledger) — this is an ops tool, not pipeline code.
Run from backend/:  .venv\\Scripts\\python ..\\scripts\\probe_free_models.py
"""

import sys
import time
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))
from app.config import load_config

CANDIDATES = [
    "qwen/qwen3-next-80b-a3b-instruct:free",
    "nvidia/nemotron-3-super-120b-a12b:free",
    "openai/gpt-oss-20b:free",
    "meta-llama/llama-3.3-70b-instruct:free",
    "google/gemma-4-31b-it:free",
    "tencent/hy3:free",
    "nousresearch/hermes-3-llama-3.1-405b:free",
    "qwen/qwen3-coder:free",
]


def main():
    cfg = load_config()
    url = cfg.openrouter["base_url"].rstrip("/") + "/chat/completions"
    headers = {"Authorization": f"Bearer {cfg.api_key}", "Content-Type": "application/json"}
    alive = []
    for model in CANDIDATES:
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": "Reply with exactly: OK"}],
            "max_tokens": 8,
        }
        t0 = time.time()
        try:
            r = requests.post(url, headers=headers, json=payload, timeout=45)
            dt = time.time() - t0
            if r.status_code == 200:
                text = (r.json()["choices"][0]["message"]["content"] or "").strip()
                print(f"  OK   {model}  ({dt:.1f}s) -> {text[:40]!r}")
                alive.append(model)
            else:
                err = r.json().get("error", {}).get("message", r.text[:80])
                print(f"  {r.status_code}  {model}  ({dt:.1f}s) {err[:90]}")
        except requests.RequestException as e:
            print(f"  ERR  {model}  {type(e).__name__}")
        time.sleep(3)  # stay friendly to the free tier
    print("\nresponding models:", alive if alive else "NONE")


if __name__ == "__main__":
    main()
