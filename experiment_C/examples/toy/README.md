# Toy Smoke Test

This tiny reward-hacking example verifies evaluation and figures without downloading datasets or loading a model.

```powershell
python scripts\evaluate_hacking.py `
  --samples examples\toy\hacking_test.jsonl `
  --outputs examples\toy\outputs.jsonl `
  --out-dir examples\toy\results

python scripts\make_figures.py `
  --summary examples\toy\results\hacking_metrics.csv `
  --monitor-summary examples\toy\results\monitor_input_metrics.csv `
  --out-dir examples\toy\figures
```
