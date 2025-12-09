import re, argparse, sys, os

# Simple, explainable rules. Modify freely.
PATTERNS = [
    r"/\.\./", r"%2fetc%2fpasswd", r"/etc/passwd", r"/etc/shadow", r"\.env\b",
    r"/phpmyadmin\b", r"/wp-login\.php\b", r"/\.git/config\b", r"\bserver-status\b",
    r"(['\"]\s*or\s*1=1)|(%27\s*or\s*1=1)|(; *drop\s+table)", r"<script>.*</script>",
    r"\bjndi:ldap://", r"[?&]cmd=", r"/cgi-bin/", r"/admin\b", r"/\.DS_Store\b",
]
UA_BAD = [r"sqlmap", r"nikto", r"dirb", r"wpscan", r"acunetix", r"nessus", r"curl/7\.", r"python-requests"]

STATUS_POS = {401, 403, 404, 429, 500, 502, 503}
LONG_URL = 180  # very long URLs often indicate payloads
METHOD_BAD = {"TRACE", "OPTIONS"}  # tune as needed

_req = re.compile(r'"([A-Z]+)\s+([^"]*?)\s+HTTP/[\d.]+"')
_ua = re.compile(r'"[^"]*"$')  # last quoted field is usually UA (Combined format)

def is_attack(line: str) -> int:
    s = line.strip()
    ls = s.lower()
    # path/payload signatures
    if any(re.search(p, ls) for p in PATTERNS):
        return 1
    # UA signatures
    if any(re.search(p, ls) for p in UA_BAD):
        return 1
    # method + path heuristics
    m = _req.search(s)
    if m:
        method, path = m.group(1), m.group(2)
        if method in METHOD_BAD:
            return 1
        if len(path) > LONG_URL:
            return 1
        # crude extension flags
        if any(ext in path.lower() for ext in [".php", ".cgi"]) and ("?" in path or "=" in path):
            return 1
    # status code heuristic (if present)
    parts = s.split('"')
    # pattern: ip ... "REQ" status size "ref" "ua"
    try:
        tail = parts[2].strip()  # after closing quote
        status = int(tail.split()[0])
        if status in STATUS_POS:
            return 1
    except Exception:
        pass
    return 0

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True)
    ap.add_argument("--out", help="Output labels path (default: <log>.labels.txt)")
    args = ap.parse_args()
    if not os.path.exists(args.log):
        print("[ERR] file not found:", args.log); sys.exit(1)
    out = args.out or (args.log + ".labels.txt")
    total = pos = 0
    with open(args.log, encoding="utf-8", errors="ignore") as f, open(out, "w", encoding="utf-8") as g:
        for ln in f:
            if not ln.strip(): continue
            y = is_attack(ln)
            pos += y
            total += 1
            g.write(str(y) + "\n")
    print(f"[OK] wrote {out}  (total={total}, positives={pos}, rate={pos/(total or 1):.3f})")

if __name__ == "__main__":
    main()
