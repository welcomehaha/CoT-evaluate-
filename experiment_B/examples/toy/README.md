# Toy Smoke Test

This tiny example verifies original/cue pair evaluation without downloading datasets or loading a model.

```powershell
python scripts\evaluate_hidden_cue.py `
  --samples-original examples\toy\test_original.jsonl `
  --samples-cue examples\toy\test_cue.jsonl `
  --outputs examples\toy\outputs.jsonl `
  --out-dir examples\toy\results

python scripts\make_figures.py `
  --summary examples\toy\results\eval_hidden_cue_metrics.csv `
  --cue-type-summary examples\toy\results\cue_type_metrics.csv `
  --out-dir examples\toy\figures
```
