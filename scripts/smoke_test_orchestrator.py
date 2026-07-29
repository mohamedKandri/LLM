"""Live check of the actual background Orchestrator loop (not fakes):
start() -> real generate/answer/evaluate/save iterations against the
free-tier teacher -> stop(). Does NOT call graduate()/go_local() (those
rewrite config.yaml) — this only proves start/pause/stop threading and
the loop body work against the real API.
Run from backend/:  .venv\\Scripts\\python ..\\scripts\\smoke_test_orchestrator.py
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from app.config import load_config
from app.orchestrator import Orchestrator


def main():
    cfg = load_config()
    orch = Orchestrator(cfg, batch_size=2, poll_interval_s=0.5)
    print(f"before: {orch.snapshot()}")

    orch.start()
    print("started, letting it run for ~20s...")
    time.sleep(20)
    orch.stop()

    snap = orch.snapshot()
    print(f"\nafter: {snap}")
    ok = snap["state"] == "stopped" and (snap["accepted"] + snap["rejected"]) > 0
    print("\nORCHESTRATOR SMOKE TEST:", "PASS" if ok else "FAIL (no tasks processed)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
