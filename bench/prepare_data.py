"""Download the two datasets and write normalized JSONL files.

Outputs, under data/:
  d1_test.jsonl    562 real agent tool calls (safeagents-judge-bench, test split)
  d2_calib.jsonl   500 calls from agent-action-safety train, used only to fit calibration
                   and thresholds
  d2_test.jsonl    stratified sample of agent-action-safety validation
  data_manifest.json  dataset revisions, counts and skipped rows

Every row: {"id", "dataset", "state", "labels": {...}, "group"}.
`group` is the unit the bootstrap resamples (a trajectory for d1, a generation batch for d2),
so correlated calls move together.

Usage: python -m bench.prepare_data
"""
import json
from collections import Counter

import numpy as np

from .common import load_config, write_jsonl


def _maybe_json(s):
    if isinstance(s, str):
        try:
            return json.loads(s)
        except (ValueError, TypeError):
            return s
    return s


def normalize_d1(row: dict) -> dict:
    findings = [
        f"{f.get('title', f.get('id', ''))} ({f.get('severity', '?')})"
        for f in (row.get("findings") or [])
    ]
    state = {
        "user_task": row["task_text"],
        "proposed_tool_call": {"tool": row["tool_name"], "arguments": _maybe_json(row["tool_args"])},
        "scanner_findings": findings,
    }
    return {
        "id": f"d1:{row['id']}",
        "dataset": "d1",
        "state": state,
        "labels": {
            "block": int(row["label"] == "block"),
            "category": row["category"],
            "label_source": row.get("label_source"),
        },
        "group": row.get("trial") or row["id"],
        "meta": {"task": row.get("task"), "cohort": row.get("cohort")},
    }


D2_STATE_FIELDS = [
    ("context", "agent_context"),
    ("constraints", "constraints"),
    ("trigger", "trigger"),
    ("original_goal", "original_goal"),
    ("user_message", "user_message"),
    ("conversation", "conversation"),
    ("action", "proposed_tool_call"),
]


def normalize_d2(row: dict, split: str) -> dict | None:
    """Rows without a flat `action` field (the repo mixes in a chat-format schema) are skipped."""
    if not row.get("action") or row.get("is_safe") is None:
        return None
    state = {out: row[src] for src, out in D2_STATE_FIELDS if row.get(src) not in (None, "", [])}
    return {
        "id": f"d2:{split}:{row['id']}",
        "dataset": "d2",
        "state": state,
        "labels": {
            "block": int(not row["is_safe"]),
            "property": row.get("property") or "none",
            "severity": row.get("severity"),
            "scenario": row.get("scenario"),
            "injection": int(row.get("property") == "injection"),
        },
        "group": row.get("batch_id") or row["id"],
        "meta": {"domain": row.get("domain"), "input_tier": row.get("input_tier"), "source": row.get("source")},
    }


def stratified_sample(rows, n, key, rng):
    if n is None or n >= len(rows):
        return list(rows)
    strata = {}
    for r in rows:
        strata.setdefault(key(r), []).append(r)
    out = []
    for k, members in sorted(strata.items(), key=lambda kv: str(kv[0])):
        take = max(1, round(n * len(members) / len(rows)))
        idx = rng.choice(len(members), size=min(take, len(members)), replace=False)
        out.extend(members[i] for i in sorted(idx))
    rng.shuffle(out)
    return out[:n]


def _read_any(path):
    import gzip

    import pandas as pd

    if path.endswith(".parquet"):
        return pd.read_parquet(path).to_dict("records")
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt") as f:
        text = f.read().strip()
    if not text:
        return []
    if text[0] == "[":
        return json.loads(text)
    return [json.loads(line) for line in text.splitlines() if line.strip()]


DATA_EXT = (".jsonl", ".json", ".parquet", ".jsonl.gz", ".json.gz")
SPLIT_ALIASES = {
    "train": ["train", "training"],
    "validation": ["validation", "valid", "val", "dev", "eval", "evaluation"],
    "test": ["test", "testing", "holdout"],
}


def _card_split_patterns(card_configs, split):
    """Split -> glob patterns from the dataset card's YAML `configs:` block, if it has one."""
    out = []
    for c in card_configs or []:
        for df in (c.get("data_files") or []) if isinstance(c, dict) else []:
            if isinstance(df, dict) and df.get("split") == split:
                paths = df.get("path")
                out += [paths] if isinstance(paths, str) else list(paths or [])
    return out


def resolve_split_files(files, split, card_configs=None):
    """Pick a split's data files: the card's own mapping first, then name aliases as whole tokens
    (so 'val' does not match 'interval'), checked against path segments and file stems."""
    import fnmatch
    import re

    data = [f for f in files if f.lower().endswith(DATA_EXT) and not f.split("/")[-1].startswith(".")]
    pats = _card_split_patterns(card_configs, split)
    if pats:
        hit = [f for f in data if any(fnmatch.fnmatch(f, p) for p in pats)]
        if hit:
            return sorted(hit), f"dataset card configs: {pats}"
    aliases = SPLIT_ALIASES.get(split, [split])
    hit = []
    for f in data:
        tokens = set(re.split(r"[^a-z0-9]+", f.lower()))
        if tokens & set(aliases):
            hit.append(f)
    return sorted(hit), f"filename aliases {aliases}"


def load_d2_split(repo: str, split: str, files_override=None):
    """The repo's files have two schemas, so `datasets.load_dataset` fails; read files directly."""
    from huggingface_hub import HfApi, hf_hub_download

    api = HfApi()
    info = api.dataset_info(repo)
    all_files = api.list_repo_files(repo, repo_type="dataset", revision=info.sha)
    if files_override:
        files, how = files_override, "config override"
    else:
        try:
            card_configs = info.card_data.get("configs") if info.card_data is not None else None
        except AttributeError:
            card_configs = None
        files, how = resolve_split_files(all_files, split, card_configs)
    if not files:
        listing = "\n  ".join(all_files)
        raise SystemExit(
            f"No data files found for split '{split}' in {repo}.\nRepo files:\n  {listing}\n"
            f"Set datasets.d2.files.{split} in config.yaml to the right file(s) and rerun."
        )
    print(f"d2 {split}: {files} (matched by {how})")
    rows, skipped = [], Counter()
    for f in files:
        for raw in _read_any(hf_hub_download(repo, f, repo_type="dataset", revision=info.sha)):
            norm = normalize_d2(raw, split)
            if norm is None:
                skipped[f] += 1
            else:
                rows.append(norm)
    return rows, {"files": files, "matched_by": how, "skipped": dict(skipped), "sha": info.sha}


def main():
    cfg = load_config()
    out_dir = cfg["run"]["data_dir"]
    rng = np.random.default_rng(cfg["datasets"]["seed"])
    manifest = {}

    # d1: safeagents-judge-bench loads cleanly with `datasets`.
    from datasets import load_dataset
    from huggingface_hub import HfApi

    d1cfg = cfg["datasets"]["d1"]
    ds = load_dataset(d1cfg["repo"], split=d1cfg["split"])
    d1 = [normalize_d1(r) for r in ds]
    write_jsonl(out_dir / "d1_test.jsonl", d1)
    manifest["d1"] = {
        "repo": d1cfg["repo"],
        "sha": HfApi().dataset_info(d1cfg["repo"]).sha,
        "rows": len(d1),
        "block": sum(r["labels"]["block"] for r in d1),
        "categories": Counter(r["labels"]["category"] for r in d1),
    }

    # d2: calibration from train, test from validation.
    d2cfg = cfg["datasets"]["d2"]
    overrides = d2cfg.get("files") or {}
    train, tinfo = load_d2_split(d2cfg["repo"], d2cfg["calib_from_split"], overrides.get(d2cfg["calib_from_split"]))
    val, vinfo = load_d2_split(d2cfg["repo"], d2cfg["test_from_split"], overrides.get(d2cfg["test_from_split"]))
    if set(tinfo["files"]) & set(vinfo["files"]):
        raise SystemExit(f"train and validation resolved to the same files: {tinfo['files']}")
    val_actions = {json.dumps(r["state"].get("proposed_tool_call")) for r in val}
    train = [r for r in train if json.dumps(r["state"].get("proposed_tool_call")) not in val_actions]

    calib = stratified_sample(train, d2cfg["calib_size"], lambda r: r["labels"]["block"], rng)
    test = stratified_sample(
        val, d2cfg["test_size"], lambda r: (r["labels"]["block"], r["labels"]["property"]), rng
    )
    write_jsonl(out_dir / "d2_calib.jsonl", calib)
    write_jsonl(out_dir / "d2_test.jsonl", test)
    for name, rows, info in (("d2_calib", calib, tinfo), ("d2_test", test, vinfo)):
        manifest[name] = {
            "repo": d2cfg["repo"], **info, "rows": len(rows),
            "block": sum(r["labels"]["block"] for r in rows),
            "properties": Counter(r["labels"]["property"] for r in rows),
        }

    with open(out_dir / "data_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2, default=dict)
    for k, v in manifest.items():
        print(f"{k}: {v['rows']} rows, {v['block']} block")


if __name__ == "__main__":
    main()
