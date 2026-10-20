"""Stage 3 part 2: feature variants A-D, M1 (ridge) and M2 (gradient-boosted trees) on VALIDATION only.

M1/M2 for B-D predict the residual of the cross-fitted M0b estimate; variant A has no nominal estimate and predicts
the measured temperature directly. Tuning uses validation only; test and stress splits are not touched.
python scripts/run_ablation_validation.py     (all data simulated)
"""
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from vsensor.data.config import load_dataset_config
from vsensor.eval.harness import load_eval, metrics
from vsensor.features.build import VARIANTS, build_features
from vsensor.models.baselines import M0, M0b, crossfit_m0b
from vsensor.params import load_registry

ROOT = Path(__file__).resolve().parents[1]
GRID = [dict(max_leaf_nodes=15, min_samples_leaf=200), dict(max_leaf_nodes=31, min_samples_leaf=200),
        dict(max_leaf_nodes=15, min_samples_leaf=1000), dict(max_leaf_nodes=31, min_samples_leaf=1000)]


def main():
    cfg, reg = load_dataset_config(), load_registry()
    d = ROOT / "data" / cfg.name; out = ROOT / "reports" / "models"; out.mkdir(parents=True, exist_ok=True)
    t0 = time.time(); tr, va = load_eval(d, "train"), load_eval(d, "validation")
    m0b = M0b(reg).fit(tr); m0 = M0(reg).fit(tr)
    nom_tr = crossfit_m0b(tr, 5, reg); nom_va = m0b.predict(va)
    print(f"nominal ready ({time.time()-t0:.0f}s)", flush=True)
    fill = float(tr["n_tc"].median())
    Ftr, Fva = build_features(tr, nom_tr, reg, fill), build_features(va, nom_va, reg, fill)
    print(f"features ready ({time.time()-t0:.0f}s)", flush=True)
    fit_mask = (tr["t_exh_meas"].notna() & (tr["t_s"] % 5 == 0)).to_numpy()
    y_tr = tr["t_exh_meas"].to_numpy(); y_va = va["t_exh_meas"].to_numpy()
    thr = float(np.percentile(np.abs(Ftr["m_fuel_slope30"].to_numpy()), 90))   # fixed on train: top 10 % fuel-flow slope = "ramp"
    ramp = np.abs(Fva["m_fuel_slope30"].to_numpy()) >= thr
    rows, preds, log = [], {}, {"grid": []}
    def score(name, p):
        r = {"model": name}
        for g, m in (("all", np.ones(len(va), bool)), ("diesel_like", (va.fuel == "diesel_like").to_numpy()),
                     ("methanol", (va.fuel == "methanol").to_numpy()), ("load>=0.85", (va.load_fraction >= 0.85).to_numpy()),
                     ("ramp (top 10 % |fuel slope|)", ramp), ("not ramp", ~ramp)):
            s = metrics(va[m], p[m], "t_exh_meas", 200); r[g] = f"{s['rmse']:.2f} [{s['rmse_lo']:.2f},{s['rmse_hi']:.2f}]"
        rows.append(r); preds[name] = p
    score("M0", m0.predict(va)); score("M0b", nom_va)
    best_m2 = {}
    for v, cols in VARIANTS.items():
        base_tr = np.zeros(len(tr)) if v == "A" else nom_tr
        base_va = np.zeros(len(va)) if v == "A" else nom_va
        Xtr, Xva = Ftr[cols].to_numpy()[fit_mask], Fva[cols].to_numpy()
        ytr = (y_tr - base_tr)[fit_mask]
        sc = StandardScaler().fit(Xtr)                      # scaler fitted on train only
        ridge = Ridge(alpha=1.0).fit(sc.transform(Xtr), ytr)
        score(f"M1-{v}", base_va + ridge.predict(sc.transform(Xva)))
        best = None
        for g in GRID:
            gb = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.08, l2_regularization=1.0, random_state=0, **g).fit(Xtr, ytr)
            p = base_va + gb.predict(Xva); rmse = float(np.sqrt(np.mean((p - y_va) ** 2)))
            log["grid"].append(dict(variant=v, **g, val_rmse=rmse))
            if best is None or rmse < best[0]: best = (rmse, g, gb, p)
        score(f"M2-{v}", best[3]); best_m2[v] = (best[1], best[2], cols)
        print(f"variant {v} done ({time.time()-t0:.0f}s) M2 grid-best {best[1]} rmse {best[0]:.2f}", flush=True)
    tab = pd.DataFrame(rows); tab.to_csv(out / "ablation_validation.csv", index=False)
    # importance (permutation, 20k validation rows) for the best-variant M2, against physical expectation
    bv = min("BCD", key=lambda v: min(x["val_rmse"] for x in log["grid"] if x["variant"] == v))
    g, gb, cols = best_m2[bv]; idx = np.random.default_rng(0).choice(len(va), 20000, replace=False)
    pi = permutation_importance(gb, Fva[cols].to_numpy()[idx], (y_va - nom_va)[idx], n_repeats=3, random_state=0, scoring="neg_root_mean_squared_error")
    imp = pd.Series(pi.importances_mean, index=cols).sort_values(ascending=False)
    imp.to_csv(out / f"importance_M2_{bv}.csv", header=["rmse_increase_K"])
    log["best_variant_for_importance"] = bv; (out / "ablation_log.json").write_text(json.dumps(log, indent=1))
    md = ["# Feature ablation A-D and models M1, M2 on VALIDATION (simulated data)", "",
          "RMSE against measured [K], 95 % episode-bootstrap interval. B-D predict the residual of cross-fitted M0b; A predicts the target directly.",
          f"Ramp rows: fuel-flow slope over 30 s in the top 10 % of train (threshold {thr:.5f} kg/s per s). The 10 s large-change rule of the feature 'since_change' hits only 60 validation rows in 1 episode, so it is not used for reporting.",
          "Fitted on every 5th second of train, scalers on train only. Test and stress splits not touched.", "",
          "| model | " + " | ".join(tab.columns[1:]) + " |", "|---|" + "---|" * (len(tab.columns) - 1)]
    md += ["| " + " | ".join(str(x) for x in r) + " |" for r in tab.itertuples(index=False)]
    md += ["", f"Permutation importance, M2 variant {bv} (RMSE increase when the feature is shuffled, K):", "", imp.round(3).to_string(), "",
           "Hyperparameter grid (validation RMSE, all rows): " + json.dumps(log["grid"])]
    (out / "ablation_summary.md").write_text("\n".join(md) + "\n"); print("\n".join(md)); print(f"done {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
