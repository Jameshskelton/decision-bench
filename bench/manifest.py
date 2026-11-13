"""Record what was run: GPU, driver, model revisions, serving runtime, config. Run once per system,
while its server is up, right before its accuracy pass.

  python -m bench.manifest --system kev-4b --runtime "kev @ <git sha>"
"""
import argparse
import json
import platform
import subprocess
from datetime import datetime, timezone

from .common import load_config
from .questions import SCHEMA_VERSION


def sh(cmd):
    try:
        return subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=20).stdout.strip()
    except Exception as e:  # noqa: BLE001
        return f"unavailable: {e!r}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--system", required=True)
    ap.add_argument("--runtime", default="", help="serving stack and version, e.g. 'vllm 0.32.0rc1.dev190'")
    a = ap.parse_args()
    cfg = load_config()
    scfg = cfg["systems"][a.system]
    m = {
        "system": a.system,
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "schema": SCHEMA_VERSION,
        "endpoint": {k: v for k, v in scfg.items() if k != "api_key_env"},
        "runtime": a.runtime,
        "client_host": platform.node(),
        "gpu": sh("nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader"),
        "cuda": sh("nvidia-smi | grep -o 'CUDA Version: [0-9.]*'"),
        "harness_git": sh("git rev-parse HEAD"),
    }
    if scfg.get("hf_repo"):
        try:
            from huggingface_hub import model_info
            m["hf_revision"] = model_info(scfg["hf_repo"]).sha
        except Exception as e:  # noqa: BLE001
            m["hf_revision"] = f"unavailable: {e!r}"
    out = cfg["run"]["results_dir"] / a.system / "manifest.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(m, indent=2))
    print(json.dumps(m, indent=2))


if __name__ == "__main__":
    main()
