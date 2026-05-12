# Toy Smoke Test

This tiny example verifies that `evaluate_outputs.py` can score original and hidden-cue paired outputs without downloading datasets or loading a model.

```powershell
python scripts\evaluate_outputs.py `
  --samples-original examples\toy\test_original.jsonl `
  --samples-hidden examples\toy\test_hidden_cue.jsonl `
  --outputs examples\toy\outputs.jsonl `
  --out-dir examples\toy\results
```
