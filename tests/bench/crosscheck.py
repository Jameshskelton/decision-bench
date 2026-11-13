"""Compare two endpoints on the same items: do they make the same decisions?

d3's card verified vLLM against its own AutoModel code on AMD only. Run this on the H200
before trusting vLLM numbers:

  python -m bench.crosscheck --a http://127.0.0.1:8000/v1/systemone --b http://127.0.0.1:8001/v1/systemone \
      --model vllm-sr/d3 --n 200
"""
import argparse

import numpy as np

from .client import SystemOneClient
from .common import load_config, read_jsonl
from .questions import questions_for


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--n", type=int, default=200)
    a = ap.parse_args()
    cfg = load_config()
    rows = (read_jsonl(cfg["run"]["data_dir"] / "d1_test.jsonl")[: a.n // 2]
            + read_jsonl(cfg["run"]["data_dir"] / "d2_test.jsonl")[: a.n // 2])
    ca = SystemOneClient({"url": a.a, "model": a.model}, 120, 2)
    cb = SystemOneClient({"url": a.b, "model": a.model}, 120, 2)
    diffs, flips, cat_flips = [], 0, 0
    for row in rows:
        q = questions_for(row["dataset"])
        ra, rb = ca.decide(row["state"], q), cb.decide(row["state"], q)
        pa = ra["answers"].get("block", {}).get("p_yes")
        pb = rb["answers"].get("block", {}).get("p_yes")
        if pa is None or pb is None:
            print(f"{row['id']}: missing answer (a={ra['status']}, b={rb['status']})")
            continue
        diffs.append(abs(pa - pb))
        flips += int((pa >= 0.5) != (pb >= 0.5))
        if "category" in q:
            cat_flips += int(ra["answers"].get("category", {}).get("choice")
                             != rb["answers"].get("category", {}).get("choice"))
    d = np.array(diffs)
    print(f"compared {len(d)}: median |Δp| {np.median(d):.4f}, max {d.max():.4f}, "
          f"block flips {flips}, category flips {cat_flips}")
    print("PASS" if flips <= max(1, len(d) // 200) else "INVESTIGATE: more flips than near-ties explain")


if __name__ == "__main__":
    main()
