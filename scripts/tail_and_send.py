import argparse, io, os, time, gzip, json, sys
import requests

def open_text(path):
    # tailing real-time logs are almost never gz; handle anyway if user points to one
    if path.lower().endswith(".gz"):
        return io.TextIOWrapper(gzip.open(path, "rb"), encoding="utf-8", errors="ignore")
    return open(path, "r", encoding="utf-8", errors="ignore")

def file_stat_signature(path):
    try:
        st = os.stat(path)
        # tuple lets us detect rotation (inode differs on *nix; size shrink on Windows)
        return (st.st_ino if hasattr(st, "st_ino") else None, st.st_size, st.st_mtime)
    except FileNotFoundError:
        return None

def tail_and_send(path, api, sleep, from_start, batch, status_every):
    sess = requests.Session()
    sig = None
    f = None
    sent = ok = bad = 0

    def reopen(seek_end=True):
        nonlocal f, sig
        if f: 
            try: f.close()
            except: pass
            f = None
        # wait until file exists
        while not os.path.exists(path):
            time.sleep(sleep)
        f = open_text(path)
        if seek_end:
            # seek to end for live tail
            try:
                f.seek(0, os.SEEK_END)
            except Exception:
                pass
        sig = file_stat_signature(path)

    reopen(seek_end=not from_start)

    buffer = []
    last_report = time.time()

    while True:
        # detect rotation/truncation
        cur_sig = file_stat_signature(path)
        if sig and cur_sig:
            same_inode = (sig[0] == cur_sig[0]) if (sig[0] is not None and cur_sig[0] is not None) else True
            shrunk = cur_sig[1] < sig[1]
            if (not same_inode) or shrunk:
                # rotated or truncated
                reopen(seek_end=False)
                buffer.clear()

        line = f.readline()
        if not line:
            time.sleep(sleep)
        else:
            if not line.strip():
                continue
            buffer.append(line)
            if len(buffer) >= batch:
                okb, badb = flush(api, sess, buffer)
                ok += okb; bad += badb; sent += (okb + badb)
                buffer.clear()

        # periodic flush/report even if batch not full
        now = time.time()
        if (buffer and now - last_report >= 1.0) or (now - last_report >= status_every):
            okb, badb = flush(api, sess, buffer)
            ok += okb; bad += badb; sent += (okb + badb)
            buffer.clear()
            last_report = now
            if now - last_report >= status_every:
                print(f"[TAIL] sent={sent} ok={ok} bad={bad}", flush=True)

def flush(api, sess, lines):
    ok = bad = 0
    for ln in lines:
        try:
            r = sess.post(api, json={"line": ln}, timeout=10)
            if r.ok:
                ok += 1
            else:
                bad += 1
                print(f"[WARN] {r.status_code} {r.text[:200]}", flush=True)
        except Exception as e:
            bad += 1
            print(f"[ERR] {e}", flush=True)
    return ok, bad

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True, help="Path to live log (plain or .gz)")
    ap.add_argument("--api", default="http://127.0.0.1:8000/score")
    ap.add_argument("--sleep", type=float, default=0.25, help="Idle sleep between polls")
    ap.add_argument("--from-start", action="store_true", help="Read file from beginning")
    ap.add_argument("--batch", type=int, default=50, help="Post lines in batches up to this size")
    ap.add_argument("--status-every", type=float, default=5.0, help="Seconds between progress prints")
    args = ap.parse_args()

    if not os.path.exists(args.log):
        print(f"[ERR] file not found: {args.log}", file=sys.stderr); sys.exit(1)
    print(f"[TAIL] streaming {args.log} -> {args.api}", flush=True)
    try:
        tail_and_send(args.log, args.api, args.sleep, args.from_start, args.batch, args.status_every)
    except KeyboardInterrupt:
        print("\n[TAIL] stopped by user")

if __name__ == "__main__":
    main()
