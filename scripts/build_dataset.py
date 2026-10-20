"""Build the simulated dataset, run the nine gates, release only if all pass (SIMULATED DATA).

Usage:  python scripts/build_dataset.py [--full-check] [--out data/<name from config>]
The data are built in a temporary directory next to the target and moved into place only when
every automated gate passes. --full-check also rebuilds everything a second time and compares hashes.
"""
from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vsensor.data.card import write_card, write_quality_report  # noqa: E402
from vsensor.data.config import load_dataset_config  # noqa: E402
from vsensor.data.generate import build_dataset  # noqa: E402
from vsensor.data.quality import run_all  # noqa: E402
from vsensor.params import load_registry  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    ap.add_argument("--full-check", action="store_true")
    a = ap.parse_args()
    cfg, reg = load_dataset_config(), load_registry()
    out = Path(a.out) if a.out else ROOT / "data" / cfg.name
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix="build_", dir=out.parent))
    try:
        man = build_dataset(cfg, reg, tmp)
        full = None
        if a.full_check:
            tmp2 = Path(tempfile.mkdtemp(prefix="rebuild_", dir=out.parent))
            try:
                full = build_dataset(cfg, reg, tmp2, progress=False)["dataset_hash"]
            finally:
                shutil.rmtree(tmp2, ignore_errors=True)
        write_card(cfg, reg, tmp)
        gates = run_all(cfg, reg, tmp, full_hash=full)
        write_quality_report(gates, cfg, tmp)
        for g in gates:
            print(f"Gate {g.number} {g.name}: {g.summary}")
            for d in g.details:
                print("    " + d)
        if not all(g.passed for g in gates):
            print("NOT RELEASED: at least one gate failed.")
            return 1
        if out.exists():
            shutil.rmtree(out)
        tmp.rename(out)
        print(f"Released {out}  dataset hash {man['dataset_hash']}")
        return 0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
