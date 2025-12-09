import re, sys
pat = re.compile(r'^(\S+) \S+ \S+ \[([^\]]+)\] "([^"]+)" (\d{3}) (\S+)$')
with open(sys.argv[1], encoding="utf-8", errors="ignore") as f, open(sys.argv[2], "w", encoding="utf-8") as out:
    for ln in f:
        m = pat.match(ln.strip())
        if not m: continue
        ip, ts, req, status, size = m.groups()
        out.write(f'{ip} - - [{ts}] "{req}" {status} {size} "-" "-"\n')
