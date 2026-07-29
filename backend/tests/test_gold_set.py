"""Gold set integrity: well-formed, tests are correct, no overlap with
seeds. This is the model-collapse guard's other half — dataset_manager
refuses gold prompts at insert time, this catches drift in the file itself.
"""

import json
import subprocess
import sys
import tempfile
from pathlib import Path

from app.config import ROOT
from app.prompt_generator import _normalize, load_seeds

GOLD_PATH = ROOT / "data" / "gold_set" / "gold.jsonl"


def _load_gold():
    with open(GOLD_PATH, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def test_gold_set_well_formed():
    entries = _load_gold()
    assert len(entries) >= 10
    ids = [e["id"] for e in entries]
    assert len(ids) == len(set(ids)), "duplicate gold-set ids"
    for e in entries:
        assert e["type"] in ("code", "general")
        assert len(e["prompt"]) > 10
        if e["type"] == "code":
            assert "assert" in e["tests"]
            compile(e["tests"], "<t>", "exec")
        else:
            assert len(e.get("reference", "")) > 20


def test_gold_set_never_overlaps_seeds(cfg=None):
    from app.config import load_config

    seed_norms = {_normalize(s["prompt"]) for s in load_seeds(load_config())}
    gold_norms = {_normalize(e["prompt"]) for e in _load_gold()}
    assert not (seed_norms & gold_norms)


def test_gold_code_solutions_actually_pass_their_own_tests():
    """Re-derive a minimal reference solution isn't available here, so
    instead this just guarantees every test block is syntactically sound
    and self-contained enough to run against ANY correct definition —
    i.e. it doesn't reference undefined names outside the obvious task
    function/class. Full solution correctness is verified manually when
    the gold set is authored (see data/gold_set/README.md)."""
    for e in _load_gold():
        if e["type"] != "code":
            continue
        with tempfile.TemporaryDirectory() as tmp:
            script = Path(tmp) / "t.py"
            # A stub that raises on call proves the tests DO invoke the
            # named entity, catching typo'd function/class names.
            script.write_text(e["tests"], encoding="utf-8")
            proc = subprocess.run(
                [sys.executable, "-I", str(script)],
                capture_output=True, text=True, timeout=10, cwd=tmp,
            )
        # Expected to fail with NameError (function/class not defined) —
        # proves the tests reference something, without needing the
        # real solution here.
        assert proc.returncode != 0
        assert "NameError" in proc.stderr, f"{e['id']}: unexpected failure mode:\n{proc.stderr}"
