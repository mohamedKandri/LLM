"""Live end-to-end smoke test of the Phase-1 mechanics (~5 free-tier calls).

generate -> teacher answers -> evaluate (code: run tests / general: LLM judge)
-> save (dataset_manager) -> ledger audit.
Run from backend/:  .venv\\Scripts\\python ..\\scripts\\smoke_test.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from app.config import load_config
from app.dataset_manager import DatasetManager
from app.evaluator import Evaluator
from app.prompt_generator import PromptGenerator
from app.teacher_client import BudgetLedger, TeacherClient


def clip(text, n=160):
    text = " ".join(text.split())
    return text[:n] + ("..." if len(text) > n else "")


def main():
    cfg = load_config()
    print(f"phase={cfg.phase} tier={cfg.teacher_tier} teacher={cfg.teacher.model}")
    ledger = BudgetLedger(cfg.db_path)
    teacher = TeacherClient(cfg, cfg.teacher, ledger)
    judge = TeacherClient(cfg, cfg.judge, ledger)
    evaluator = Evaluator(cfg, judge_client=judge)
    dataset = DatasetManager(cfg)

    # 1) Seeds -> new tasks via teacher
    print("\n[1] generating tasks from seeds...")
    gen = PromptGenerator(cfg, teacher)
    tasks = gen.generate(n=2)
    if not tasks:
        print("FAIL: teacher produced no valid tasks")
        return 1
    for t in tasks:
        print(f"  - ({t['type']}) {clip(t['prompt'])}")

    # 2) Teacher answers each task, evaluator scores it
    results = []
    for t in tasks:
        print(f"\n[2] teacher answering ({t['type']}): {clip(t['prompt'], 80)}")
        reply = teacher.complete(
            [{"role": "user", "content": t["prompt"]}], purpose="answer"
        )
        r = evaluator.evaluate(t, reply.text)
        results.append(r)
        print(f"    -> method={r.method} score={r.score:.2f} passed={r.passed}")
        if not r.passed:
            print(f"    details: {clip(str(r.details), 200)}")
            continue
        add_result = dataset.add(t["prompt"], reply.text, r, reply.tier, reply.model)
        print(f"    -> dataset_manager: {add_result.reason} (id={add_result.id})")

    # 3) Dataset + ledger audit
    print(f"\n[3] dataset counts by tier: {dataset.counts()}")
    print("[4] budget ledger:")
    for role in ("teacher", "judge"):
        spent, calls = ledger.totals(cfg.teacher_tier, role)
        print(f"    {role}/{cfg.teacher_tier}: {calls} calls, ${spent:.4f}")

    ok = any(r.passed for r in results)
    print("\nSMOKE TEST:", "PASS (pipeline mechanics work end to end)" if ok else
          "MECHANICS OK, but no answer passed evaluation — inspect above")
    return 0


if __name__ == "__main__":
    sys.exit(main())
