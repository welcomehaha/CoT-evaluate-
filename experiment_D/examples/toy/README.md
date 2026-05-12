# Toy Smoke Test

This tiny example verifies mitigation evaluation and figures without downloading datasets or loading a model.

```powershell
python scripts\evaluate_mitigation.py `
  --samples examples\toy\mitigation_test.jsonl `
  --outputs examples\toy\outputs.jsonl `
  --out-dir examples\toy\results

python scripts\make_figures.py `
  --summary examples\toy\results\eval_mitigation_metrics.csv `
  --strata examples\toy\results\stratified_metrics.csv `
  --out-dir examples\toy\figures
```
