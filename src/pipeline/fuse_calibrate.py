import argparse, json, os

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", choices=["percentile"], default="percentile")
    ap.add_argument("--p", type=float, default=95.0)
    args = ap.parse_args()
    os.makedirs("artifacts/fusion", exist_ok=True)
    json.dump({"method": args.method, "p": args.p, "weights": {"content":0.5, "session":0.5}},
              open("artifacts/fusion/config.json","w"), indent=2)
    print("Saved fusion config.")
if __name__ == "__main__":
    main()
