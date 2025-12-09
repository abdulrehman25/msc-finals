import re, argparse

ATTACK_PATTERNS = [
    r"/\.\./", r"%2fetc%2fpasswd", r"/etc/passwd", r"/etc/shadow", r"\.env\b",
    r"/phpmyadmin", r"/wp-login\.php", r"/\.git/config", r"server-status",
    r"(['\"]\s*or\s*1=1)|(%27\s*or\s*1=1)|(;drop\s+table)", r"<script>.*</script>",
    r"\bjndi:ldap://", r"\?cmd=", r"cgi-bin", r"/admin\b"
]
UA_ATTACK = [r"sqlmap", r"nikto", r"dirb", r"wpscan"]

def is_attack(line):
    ls = line.lower()
    if any(re.search(p, ls) for p in ATTACK_PATTERNS):
        return 1
    if any(re.search(p, ls) for p in UA_ATTACK):
        return 1
    return 0

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True)
    ap.add_argument("--out", help="labels output path (defaults to <log>.labels.txt)")
    args = ap.parse_args()
    out = args.out or (args.log + ".labels.txt")
    with open(args.log, encoding="utf-8", errors="ignore") as f, open(out, "w", encoding="utf-8") as g:
        for ln in f:
            if not ln.strip(): continue
            g.write(str(is_attack(ln))+"\n")
    print("Wrote", out)

if __name__ == "__main__":
    main()
