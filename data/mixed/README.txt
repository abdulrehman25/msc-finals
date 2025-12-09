Dummy Web Log Bundle
====================
Files generated for anomaly detection evaluation (benign + injected attacks).

Apache Combined format:
- access_eval_mix_2000.log (+ .labels.txt)   # ~22% attacks
- access_eval_small_500.log (+ .labels.txt)  # ~18% attacks

Nginx-style JSON log:
- nginx_json_eval_800.log (+ .labels.txt)    # ~25% attacks

Each .labels.txt aligns line-by-line:
  0 = benign, 1 = attack/suspicious

Suggested commands (PowerShell):

  python .\scripts\evaluate_file.py --log "./data/mixed/access_eval_mix_2000.log" --labels "./data/mixed/access_eval_mix_2000.log.labels.txt"
  python .\scripts\evaluate_file.py --log "./data/mixed/access_eval_small_500.log"  --labels "./data/mixed/access_eval_small_500.log.labels.txt"
  python .\scripts\evaluate_file.py --log "./data/mixed/nginx_json_eval_800.log" --labels "./data/mixed/nginx_json_eval_800.log.labels.txt"
