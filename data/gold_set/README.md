# Gold set

Human-verified benchmark examples (`gold.jsonl`, one task per line).

**This data must NEVER enter the training set.** It is the fixed yardstick
that detects model collapse; `dataset_manager` will refuse any training
example whose prompt matches a gold-set entry.

Format per line:
```json
{"type": "code|general", "prompt": "...", "tests": "... (code tasks)", "reference": "... (general tasks)"}
```
