"""Fit the four baselines on train, evaluate on VALIDATION only, check the pre-registered ordering.

python scripts/run_baselines_validation.py [--no-boot]     (all data simulated)
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from vsensor.data.config import load_dataset_config
from vsensor.eval.harness import load_eval, lag_note, noise_floor, report
from vsensor.models.baselines import LoadLookup, M0, M0b, PerFuelConstant, bootstrap_params, crossfit_m0b
from vsensor.params import load_registry

ROOT = Path(__file__).resolve().parents[1]


def main():
    cfg, reg = load_dataset_config(), load_registry()
    d = ROOT / "data" / cfg.name
    out = ROOT / "reports" / "baselines"; out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    tr, va = load_eval(d, "train"), load_eval(d, "validation")
    models = [PerFuelConstant(), LoadLookup(reg), M0(reg), M0b(reg)]
    for m in models:
        m.fit(tr); print(f"fitted {m.name} ({time.time()-t0:.0f}s)", flush=True)
    m0b = models[-1]
    lines = ["# Baselines on VALIDATION (simulated data)", "",
             f"Dataset: {cfg.name}. Fitted on train only. Test and stress splits not touched.", "",
             f"M0 delivery ratio: {models[2].theta[0]:.4f}",
             f"M0b a, b (best start): {m0b.theta[0]:.4f}, {m0b.theta[1]:.4f}; all starts (params, RMSE vs measured on fit rows): {m0b.starts_result}", ""]
    if "--no-boot" not in sys.argv:
        bp = bootstrap_params(tr)
        lo, hi = np.percentile(bp, [2.5, 97.5], axis=0)
        lines.append(f"M0b episode-bootstrap (20 resamples) 95 % interval: a [{lo[0]:.4f}, {hi[0]:.4f}], b [{lo[1]:.4f}, {hi[1]:.4f}]")
        print(lines[-1], flush=True)
    preds = {m.name: m.predict(va) for m in models}
    rep = report(va, preds)
    rep.to_csv(out / "validation_metrics.csv", index=False)
    nf, lg = noise_floor(va), lag_note(va)
    n_meas = int(va["t_exh_meas"].notna().sum())
    lines += [f"Noise floor (RMS measured - true): {nf:.2f} K", f"Lag (true vs steady): RMS {lg['rms']:.2f} K, p99 {lg['p99']:.2f} K",
              f"Rows: {len(va)}; with measured exhaust value: {n_meas}; dropouts: {len(va)-n_meas}", "",
              "## RMSE vs measured [K] (95 % episode-bootstrap interval)", ""]
    order = [m.name for m in models]
    ok = True
    lines.append("| group | " + " | ".join(order) + " |"); lines.append("|---|" + "---|" * len(order))
    for gk, gv in [("all", "all"), ("fuel", "diesel_like"), ("fuel", "methanol")] + [("load_bin", b) for b in sorted(va["load_bin"].unique())]:
        cells, vals = [], []
        for n in order:
            r = rep[(rep.model == n) & (rep.group == gk) & (rep.value == gv) & (rep.reference == "t_exh_meas")].iloc[0]
            cells.append(f"{r.rmse:.2f} [{r.rmse_lo:.2f}, {r.rmse_hi:.2f}]"); vals.append(r.rmse)
        lines.append(f"| {gv} | " + " | ".join(cells) + " |")
        if gk == "fuel" and not (vals[0] > vals[1] > vals[2] > vals[3]): ok = False
    lines += ["", f"Pre-registered ordering (constant > lookup > M0 > M0b, per fuel): {'HOLDS' if ok else 'DOES NOT HOLD - run the defined checks, report as is, do not tune'}"]
    (out / "validation_summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines)); print(f"done in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
