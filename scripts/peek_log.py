import os, sys
from featurization.parse import parse_line

path = sys.argv[1] if len(sys.argv)>1 else None
if not path or not os.path.exists(path):
    print("[ERR] File not found:", path); sys.exit(1)

total = ok = 0
samples = []
with open(path, encoding="utf-8", errors="ignore") as f:
    for ln in f:
        if not ln.strip(): continue
        total += 1
        d = parse_line(ln)
        if d: 
            ok += 1
            if len(samples) < 5: samples.append((ln.rstrip(), d))
        if total >= 1000: break

print(f"[INFO] Previewed {total} lines; parsed ok: {ok} ({(ok/total*100) if total else 0:.1f}%)")
for i,(raw, d) in enumerate(samples, 1):
    print(f"\n--- sample {i} ---\nRAW: {raw}\nPARSED: {d}")
