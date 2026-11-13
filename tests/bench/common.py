import json
import os
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent


def load_config(path: str | os.PathLike | None = None) -> dict:
    path = Path(path or os.environ.get("BENCH_CONFIG") or ROOT / "config.yaml")
    with open(path) as f:
        cfg = yaml.safe_load(f)
    for key in ("results_dir", "data_dir", "report_dir"):
        p = Path(cfg["run"][key])
        cfg["run"][key] = p if p.is_absolute() else ROOT / p
    return cfg


def read_jsonl(path) -> list[dict]:
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path, rows) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def append_jsonl(fh, row) -> None:
    fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    fh.flush()
