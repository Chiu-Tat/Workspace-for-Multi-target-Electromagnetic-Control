#!/usr/bin/env python3
"""Run the Section 6/7 analyses and write the results bundle.

    python -m analysis.sections_67.run --studies 6.1,6.2,6.3,6.4,7.1,7.2,7.3,7.4
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import hashlib
import importlib.metadata
import sys
import time
from pathlib import Path

from analysis.sections_67.common import OUTPUT_ROOT, output_dirs, dump_json
from analysis.sections_67.sec6 import run_6_1, run_6_2, run_6_3, run_6_4
from analysis.sections_67.sec7 import run_7_1, run_7_2, run_7_3, run_7_4


STUDIES = {
    "6.1": run_6_1,
    "6.2": run_6_2,
    "6.3": run_6_3,
    "6.4": run_6_4,
    "7.1": run_7_1,
    "7.2": run_7_2,
    "7.3": run_7_3,
    "7.4": run_7_4,
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--studies", default="6.1,6.2,6.3,6.4,7.1,7.2,7.3,7.4")
    parser.add_argument("--output", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--rerun-2mm-ablation", action="store_true")
    args = parser.parse_args(argv)

    paths = output_dirs(args.output)
    selected = [item.strip() for item in args.studies.split(",") if item.strip()]
    unknown = [item for item in selected if item not in STUDIES]
    if unknown:
        raise SystemExit(f"unknown studies: {unknown}")

    previous_log = paths["root"] / "run_log.json"
    ledger = json.loads(previous_log.read_text()) if previous_log.exists() else {"studies": {}}
    run_id = str(time.time_ns())
    ledger["started_unix_s"] = time.time()
    repo = Path(__file__).resolve().parents[2]
    sources = list((repo / "analysis/sections_67").glob("*.py")) + list((repo / "hybrid_region").glob("*.py"))
    sources += list((repo / "emns").rglob("*.py"))
    provenance = {
        "run_id": run_id, "command": sys.argv,
        "python": sys.version, "platform": platform.platform(),
        "machine": platform.machine(),
        "packages": {k: importlib.metadata.version(k) for k in ("numpy", "scipy", "matplotlib")},
        "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip(),
        "source_sha256": {str(p.relative_to(repo)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
        "evidence_status": "nominal floating point; studies 6.4 and 7.4 reuse archived outputs",
    }
    cpu = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True)
    provenance["cpu"] = cpu.stdout.strip() if cpu.returncode == 0 else platform.processor()
    (paths["validation"] / f"run_{run_id}.json").write_text(json.dumps(provenance, indent=2))
    for name in selected:
        print(f"==> {name}", flush=True)
        started = time.perf_counter()
        if name == "6.4":
            result = run_6_4(paths, rerun_2mm=args.rerun_2mm_ablation)
        else:
            result = STUDIES[name](paths)
        elapsed = time.perf_counter() - started
        summary = _summarise(result)
        summary["elapsed_s"] = elapsed
        summary["run_id"] = run_id
        ledger["studies"][name] = summary
        dump = paths["data"] / f"study_{name.replace('.', '_')}_summary.json"
        dump_json(dump, summary)
        dump_json(paths["root"] / "run_log.json", ledger)
        print(f"    done in {elapsed:.1f} s", flush=True)

    ledger["finished_unix_s"] = time.time()
    dump_json(paths["root"] / "run_log.json", ledger)
    return 0


def _summarise(result):
    if isinstance(result, dict):
        compact = {}
        for key, value in result.items():
            if key in {"figure", "metrics", "cases", "tfw", "current_limits", "inclusion", "sanity", "headings"}:
                compact[key] = value
            elif isinstance(value, (str, int, float, bool)) or value is None:
                compact[key] = value
        return compact or {"keys": list(result)}
    return {"type": type(result).__name__}


if __name__ == "__main__":
    sys.exit(main())
