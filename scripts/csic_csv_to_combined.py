import csv, sys, os, re, random
from datetime import datetime, timedelta, timezone

CANDIDATES = {
    "method": ["method","http_method","req_method"],
    "url":    ["url","path","request","resource","uri","request_uri"],
    "status": ["status","status_code","resp_status","code"],
    "ua":     ["user-agent","user_agent","ua","http_user_agent"],
    "ref":    ["referrer","referer","http_referer","ref"],
    "label":  ["label","target","class","attack","is_attack","anomaly","malicious","y","type","attack-label","attack_label","attacklabel"],
    "ip":     ["ip","client_ip","remote_addr","host","src_ip","source_ip"],
    "size":   ["size","bytes","resp_bytes","body_bytes_sent","bytes_sent","length"]
}

ATTACK_VALUES = {"1","attack","anomaly","anomalous","malicious","true","yes","bad"}  # case-insensitive

def find_col(header, keys):
    hlow = [h.strip().lower() for h in header]
    for k in keys:
        if k in hlow:
            return hlow.index(k)
    return None

def pick_col(header, names):
    for name in names:
        idx = find_col(header, [name])
        if idx is not None:
            return idx
    return None

def rand_ip():
    return "{}.{}.{}.{}".format(
        random.randint(1,254), random.randint(0,255),
        random.randint(0,255), random.randint(1,254)
    )

def to_int(x, default=0):
    try: return int(x)
    except: return default

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Input CSIC CSV file")
    ap.add_argument("--out-log", required=True, help="Output Combined-like log path")
    ap.add_argument("--out-labels", help="Output labels path (default: <out-log>.labels.txt)")
    ap.add_argument("--label-all", type=int, choices=[0,1], help="Force label for all rows (use when CSV has no label col)")
    ap.add_argument("--http-version", default="HTTP/1.1")
    ap.add_argument("--start", default="2025-01-10T12:00:00+00:00", help="Start timestamp for synthetic dates (ISO8601)")
    ap.add_argument("--step-ms", type=int, default=250, help="Time step between rows (ms)")
    ap.add_argument("--attack-values", nargs="*", help="Extra strings treated as positive labels (case-insensitive)")
    args = ap.parse_args()

    if not os.path.exists(args.csv):
        sys.exit(f"[ERR] file not found: {args.csv}")

    if args.attack_values:
        for v in args.attack_values:
            ATTACK_VALUES.add(str(v).lower())

    out_labels = args.out_labels or (args.out_log + ".labels.txt")

    # read header
    with open(args.csv, newline="", encoding="utf-8", errors="ignore") as f:
        sniffer = csv.Sniffer()
        sample = f.read(4096)
        f.seek(0)
        dialect = sniffer.sniff(sample) if sample else csv.excel
        rdr = csv.reader(f, dialect)
        header = next(rdr, None)
        if not header:
            sys.exit("[ERR] empty CSV")
        hlow = [h.strip().lower() for h in header]

        # auto-detect columns
        idx_method = pick_col(hlow, CANDIDATES["method"])   # may be None
        idx_url    = pick_col(hlow, CANDIDATES["url"])      # preferred
        idx_status = pick_col(hlow, CANDIDATES["status"])
        idx_ua     = pick_col(hlow, CANDIDATES["ua"])
        idx_ref    = pick_col(hlow, CANDIDATES["ref"])
        idx_label  = pick_col(hlow, CANDIDATES["label"])
        idx_ip     = pick_col(hlow, CANDIDATES["ip"])
        idx_size   = pick_col(hlow, CANDIDATES["size"])

        print("[INFO] column mapping:")
        print("  method:", header[idx_method] if idx_method is not None else "(missing -> UNK)")
        print("  url   :", header[idx_url]    if idx_url    is not None else "(missing)")
        print("  status:", header[idx_status] if idx_status is not None else "(missing -> 200)")
        print("  ua    :", header[idx_ua]     if idx_ua     is not None else "(missing -> '-')")
        print("  ref   :", header[idx_ref]    if idx_ref    is not None else "(missing -> '-')")
        print("  label :", header[idx_label]  if idx_label  is not None else ("(forced -> "+str(args.label_all)+")" if args.label_all is not None else "(missing)"))
        print("  ip    :", header[idx_ip]     if idx_ip     is not None else "(missing -> synthetic)")
        print("  size  :", header[idx_size]   if idx_size   is not None else "(missing -> approx)")

        if idx_url is None:
            sys.exit("[ERR] could not find a URL/path column. Add one to CSV or rename to 'url'/'path'/'request'.")

        if idx_label is None and args.label_all is None:
            print("[WARN] no label column found. Use --label-all 0 or --label-all 1 if this file is all normal or all attacks.")
            print("[WARN] proceeding with label=0 for all rows.")
            forced_label = 0
        else:
            forced_label = args.label_all

        # write outputs
        start_dt = datetime.fromisoformat(args.start)
        cur = start_dt
        step = timedelta(milliseconds=args.step_ms)

        n, pos = 0, 0
        with open(args.out_log, "w", encoding="utf-8") as g, open(out_labels, "w", encoding="utf-8") as glab:
            for row in rdr:
                if not row or len(row) < len(header):  # skip broken
                    continue

                method = (row[idx_method] if idx_method is not None else "UNK") or "UNK"
                url    = (row[idx_url] or "/") if idx_url is not None else "/"
                status = to_int(row[idx_status], 200) if idx_status is not None else 200
                ua     = (row[idx_ua] if idx_ua is not None else "-") or "-"
                ref    = (row[idx_ref] if idx_ref is not None else "-") or "-"
                ip     = (row[idx_ip] if idx_ip is not None else rand_ip()) or rand_ip()
                # size: rough approximation if missing
                size   = to_int(row[idx_size], 0) if idx_size is not None else max(0, min(20000, len(url)*3))

                # label
                if forced_label is not None:
                    y = forced_label
                elif idx_label is not None:
                    val = str(row[idx_label]).strip().lower()
                    # common encodings: 0/1 or strings
                    if val in ATTACK_VALUES:
                        y = 1
                    elif val in {"0","normal","benign","false","no","good"}:
                        y = 0
                    else:
                        # fallback: anything nonzero becomes 1
                        try:
                            y = 1 if int(val) != 0 else 0
                        except:
                            y = 0
                else:
                    y = 0

                # timestamp
                ts = cur.strftime("%d/%b/%Y:%H:%M:%S %z")
                cur += step

                # write Combined-style line
                combined = f'{ip} - - [{ts}] "{method} {url} {args.http_version}" {status} {size} "{ref}" "{ua}"'
                g.write(combined + "\n")
                glab.write(str(y) + "\n")
                n += 1
                if y == 1: pos += 1

        print(f"[OK] wrote {args.out_log} and {out_labels}  (rows={n}, positives={pos})")

if __name__ == "__main__":
    main()
