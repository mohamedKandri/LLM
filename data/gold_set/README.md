# Gold set

Benchmark examples (`gold.jsonl`, one task per line) that `benchmark.py`
runs the local model and the active teacher against to score graduation
readiness.

**This data must NEVER enter the training set.** It is the fixed yardstick
that detects model collapse; `dataset_manager.add()` refuses any training
example whose prompt matches a gold-set entry, and it never overlaps with
`data/seeds.jsonl` (checked — no shared prompts).

**Status: AI-drafted, needs your sign-off.** The current 20 entries (10
code, 10 general) were drafted by Claude, not hand-written. Every code
task's `tests` field was independently executed against a real solution
before being added, so they're mechanically correct — but "gold" also
means *you* trust the task is well-posed and the difficulty is right for
what this project should target. Read through `gold.jsonl` before relying
on benchmark scores; edit or replace anything that doesn't look right.

Format per line:
```json
{"id": "...", "type": "code|general", "prompt": "...", "tests": "... (code tasks, must compile and contain asserts)", "reference": "... (general tasks, a model answer used for teacher-vs-student comparison)"}
```
