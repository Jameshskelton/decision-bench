"""Run one system against the prepared data.

  python -m bench.run smoke    --system kev-4b
  python -m bench.run accuracy --system kev-4b --datasets d2_calib d2_test d1_test --repeats 3
  python -m bench.run load     --system kev-4b

Accuracy results: results/<system>/accuracy/<dataset>_r<k>.jsonl (resumable; rerun to fill gaps).
Load results:     results/<system>/load/c<N>.jsonl plus c<N>.summary.json.
"""
import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

from .client import make_client
from .common import append_jsonl, load_config, read_jsonl
from .questions import SCHEMA_VERSION, questions_for


def _qs(row):
    return questions_for(row["dataset"])


def _record(row, r, run_idx, concurrency):
    return {
        "id": row["id"],
        "run": run_idx,
        "concurrency": concurrency,
        "ts": datetime.now(timezone.utc).isoformat(),
        "status": r["status"],
        "error": r["error"],
        "attempts": r["attempts"],
        "latency_ms": r["latency_ms"],
        "total_ms": r["total_ms"],
        "answers": r["answers"],
        "usage": r["usage"],
        "rate_limit": r["rate_limit"] or None,
        "schema": SCHEMA_VERSION,
        "raw": r["body"],
    }


def warmup(client, rows, n):
    for row in rows[:n]:
        client.decide(row["state"], _qs(row))


def cmd_smoke(cfg, system, client):
    rows = read_jsonl(cfg["run"]["data_dir"] / "d1_test.jsonl")[:2] + \
        read_jsonl(cfg["run"]["data_dir"] / "d2_test.jsonl")[:2]
    ok = True
    for row in rows:
        r = client.decide(row["state"], _qs(row))
        lat = f"{r['latency_ms']:.0f} ms" if r["latency_ms"] else "-"
        print(f"{row['id']}: HTTP {r['status']} {lat} label={row['labels']['block']} -> {r['answers']}")
        if r["status"] != 200 or "block" not in r["answers"]:
            ok = False
            print("   body:", json.dumps(r["body"])[:600])
    raise SystemExit(0 if ok else 1)


def cmd_accuracy(cfg, system, client, datasets, repeats, concurrency):
    base = cfg["run"]["results_dir"] / system / "accuracy"
    base.mkdir(parents=True, exist_ok=True)
    data = {d: read_jsonl(cfg["run"]["data_dir"] / f"{d}.jsonl") for d in datasets}
    warm = read_jsonl(cfg["run"]["data_dir"] / "d2_calib.jsonl")
    print(f"[{system}] warmup {cfg['run']['warmup_requests']} requests")
    warmup(client, warm, cfg["run"]["warmup_requests"])
    for d, rows in data.items():
        for k in range(1, repeats + 1):
            path = base / f"{d}_r{k}.jsonl"
            done = set()
            if path.exists():
                done = {r["id"] for r in read_jsonl(path) if r["status"] == 200 and r["answers"].get("block")}
            todo = [r for r in rows if r["id"] not in done]
            print(f"[{system}] {d} run {k}: {len(todo)} to do ({len(done)} done)")
            t0 = time.perf_counter()
            with open(path, "a") as fh, ThreadPoolExecutor(concurrency) as pool:
                for row, r in zip(todo, pool.map(lambda x: client.decide(x["state"], _qs(x)), todo)):
                    append_jsonl(fh, _record(row, r, k, concurrency))
            print(f"[{system}] {d} run {k}: {time.perf_counter() - t0:.1f}s")


def cmd_load(cfg, system, client):
    lc = cfg["load"]
    rows = read_jsonl(cfg["run"]["data_dir"] / f"{lc['dataset']}.jsonl")
    base = cfg["run"]["results_dir"] / system / "load"
    base.mkdir(parents=True, exist_ok=True)
    n = lc["requests_per_level"]
    work = [rows[i % len(rows)] for i in range(n)]
    for c in lc["concurrency_levels"]:
        warmup_rows = rows[-max(c * 4, cfg["run"]["warmup_requests"]):]
        with ThreadPoolExecutor(c) as pool:
            list(pool.map(lambda x: client.decide(x["state"], _qs(x)), warmup_rows))
        t0 = time.perf_counter()
        with ThreadPoolExecutor(c) as pool:
            results = list(pool.map(lambda x: (x, time.perf_counter(), client.decide(x["state"], _qs(x))), work))
        wall = time.perf_counter() - t0
        with open(base / f"c{c}.jsonl", "w") as fh:
            for row, start, r in results:
                rec = _record(row, r, 0, c)
                rec["start_s"] = start - t0
                rec.pop("raw")
                append_jsonl(fh, rec)
        ok = sum(1 for _, _, r in results if r["status"] == 200)
        summary = {"concurrency": c, "requests": n, "ok": ok, "wall_s": wall, "rps": ok / wall}
        (base / f"c{c}.summary.json").write_text(json.dumps(summary, indent=2))
        print(f"[{system}] c={c}: {ok}/{n} ok, {ok / wall:.1f} req/s")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["smoke", "accuracy", "load"])
    ap.add_argument("--system", required=True)
    ap.add_argument("--datasets", nargs="+", default=["d2_calib", "d2_test", "d1_test"])
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--concurrency", type=int, default=1)
    ap.add_argument("--config")
    a = ap.parse_args()
    cfg = load_config(a.config)
    if a.system not in cfg["systems"]:
        raise SystemExit(f"unknown system {a.system}; choose from {list(cfg['systems'])}")
    client = make_client(cfg["systems"][a.system], cfg["run"]["request_timeout_s"], cfg["run"]["max_retries"])
    if a.mode == "smoke":
        cmd_smoke(cfg, a.system, client)
    elif a.mode == "accuracy":
        cmd_accuracy(cfg, a.system, client, a.datasets, a.repeats, a.concurrency)
    else:
        cmd_load(cfg, a.system, client)


if __name__ == "__main__":
    main()
