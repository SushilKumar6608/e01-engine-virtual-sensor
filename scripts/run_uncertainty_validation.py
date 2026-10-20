"""Stage 4: M3 + conformal calibration + OOD score + abstain rule. Fit on train, calibrate on calibration,
CHECK on validation. No test or stress split is read. python scripts/run_uncertainty_validation.py (simulated data)
"""
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from vsensor.data.config import load_dataset_config
from vsensor.eval.harness import load_eval
from vsensor.features.build import build_features
from vsensor.models.baselines import M0b, crossfit_m0b
from vsensor.models.uncertainty import (ALPHA, OOD_QUANTILE, WIDTH_TOLERANCE_K, AbstainRule, OODScorer,
                                        QuantileResidualModel, conformal_correction)
from vsensor.params import load_registry

ROOT = Path(__file__).resolve().parents[1]


def coverage_table(df, lo, hi, ref):
    y = df[ref].to_numpy(); ok = np.isfinite(y)
    inside = ((y >= lo) & (y <= hi)) & ok
    ep = pd.DataFrame({"ep": df["episode_id"].to_numpy()[ok], "c": inside[ok]}).groupby("ep")["c"].mean()
    return dict(coverage=inside[ok].mean(), ep_below80=float((ep < 0.8).mean()), ep_worst_decile=float(np.percentile(ep, 10)))


def main():
    cfg, reg = load_dataset_config(), load_registry()
    d = ROOT / "data" / cfg.name; out = ROOT / "reports" / "uncertainty"; out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    tr, ca, va = (load_eval(d, s) for s in ("train", "calibration", "validation"))
    m0b = M0b(reg).fit(tr); nom_tr = crossfit_m0b(tr, 5, reg); nom = {"calibration": m0b.predict(ca), "validation": m0b.predict(va)}
    fill = float(tr["n_tc"].median())
    F = {"train": build_features(tr, nom_tr, reg, fill), "calibration": build_features(ca, nom["calibration"], reg, fill),
         "validation": build_features(va, nom["validation"], reg, fill)}
    fit = (tr["t_exh_meas"].notna() & (tr["t_s"] % 5 == 0)).to_numpy()
    q = QuantileResidualModel("C").fit(F["train"][fit], tr["t_exh_meas"].to_numpy()[fit], nom_tr[fit])
    print(f"M3 fitted ({time.time()-t0:.0f}s)", flush=True)
    raw = {k: q.predict_raw(F[k], nom[k]) for k in ("calibration", "validation")}
    qhat = conformal_correction(*raw["calibration"], ca["t_exh_meas"].to_numpy(), ALPHA)
    # per-fuel variant (reported beside, not the headline)
    qhat_f = {c: conformal_correction(raw["calibration"][0][ca.fuel_code == c], raw["calibration"][1][ca.fuel_code == c],
                                      ca["t_exh_meas"].to_numpy()[ca.fuel_code == c]) for c in (0, 1)}
    lo, hi = raw["validation"][0] - qhat, raw["validation"][1] + qhat
    lo_f = raw["validation"][0] - va.fuel_code.map(qhat_f).to_numpy(); hi_f = raw["validation"][1] + va.fuel_code.map(qhat_f).to_numpy()
    lines = ["# M3 + split conformal + abstain rule on VALIDATION (simulated data; no test or stress split read)", "",
             f"alpha {ALPHA}; conformal correction (global) {qhat:.3f} K; per fuel {qhat_f[0]:.3f} (diesel-like), {qhat_f[1]:.3f} (methanol)", "",
             "## Coverage of the 90 % interval", "", "| split | interval | reference | coverage | episodes <80 % | worst-decile episode |", "|---|---|---|---|---|---|"]
    # calibration (by construction ~90 %) and validation (the independent check)
    clo, chi = raw["calibration"][0] - qhat, raw["calibration"][1] + qhat
    for name, df, a, b in (("calibration", ca, clo, chi), ("validation", va, lo, hi), ("validation (per-fuel correction)", va, lo_f, hi_f),
                           ("validation (uncorrected quantiles)", va, raw["validation"][0], raw["validation"][1])):
        for ref in ("t_exh_meas", "t_exh_true"):
            c = coverage_table(df, a, b, ref)
            lines.append(f"| {name} | 90 % | {ref} | {c['coverage']*100:.1f} % | {c['ep_below80']*100:.1f} % | {c['ep_worst_decile']*100:.1f} % |")
    lines += ["", "## Coverage and width by group on validation (global correction, vs measured)", "", "| group | coverage | mean width [K] | p95 width [K] |", "|---|---|---|---|"]
    w = hi - lo; y = va["t_exh_meas"].to_numpy(); inside = (y >= lo) & (y <= hi)
    for g, m in [("all", np.ones(len(va), bool)), ("diesel_like", (va.fuel == "diesel_like").to_numpy()), ("methanol", (va.fuel == "methanol").to_numpy())] + \
                [(f"load {b}", (va.load_bin == b).to_numpy()) for b in sorted(va.load_bin.unique())]:
        lines.append(f"| {g} | {inside[m].mean()*100:.1f} % | {w[m].mean():.1f} | {np.percentile(w[m],95):.1f} |")
    # OOD
    ref_mask = (tr["t_s"] % 10 == 0).to_numpy()
    ood = OODScorer().fit(F["train"], tr.fuel_code.to_numpy(), ref_mask)
    s_ca, s_va = ood.score(F["calibration"], ca.fuel_code.to_numpy()), ood.score(F["validation"], va.fuel_code.to_numpy())
    thr = float(np.quantile(s_ca, OOD_QUANTILE)); rule = AbstainRule(thr)
    dec = rule.decide(s_va, w, F["validation"])
    lines += ["", "## Abstain rule on validation (in-distribution, so every abstain is a false abstain)", "",
              f"OOD threshold (99th percentile of calibration scores) {thr:.3f}; width tolerance {WIDTH_TOLERANCE_K} K (assumed)", "",
              "| reason | share of rows |", "|---|---|"] + [f"| {c} | {dec[c].mean()*100:.2f} % |" for c in dec.columns]
    kept = ~dec["abstain"].to_numpy()
    c_all, c_kept = inside.mean(), inside[kept].mean()
    lines += ["", f"Coverage of rows answered: {c_kept*100:.1f} % (all rows {c_all*100:.1f} %). Row-wise numbers; rows within an episode are not independent."]
    # input-rule unit check on corrupted validation inputs (not model results)
    bad = va.copy(); bad.loc[bad.index[:1000], "p_scav"] = 9.0
    Fb = build_features(bad, nom["validation"], reg, fill); db = rule.decide(ood.score(Fb, bad.fuel_code.to_numpy()), w, Fb)
    lines.append(f"Corrupted copy (first 1000 rows p_scav = 9 bar): input_out_of_range flagged {db['input_out_of_range'].iloc[:1000].mean()*100:.0f} % of them, {db['input_out_of_range'].iloc[1000:].mean()*100:.1f} % of the rest.")
    (out / "uncertainty_validation.md").write_text("\n".join(lines) + "\n")
    json.dump(dict(qhat=qhat, qhat_per_fuel=qhat_f, ood_threshold=thr), open(out / "calibration_constants.json", "w"), indent=1)
    print("\n".join(lines)); print(f"done {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
