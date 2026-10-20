"""Evaluation harness (SIMULATION PROJECT; all data are simulated).

Implements the protocol fixed in docs/MODEL_CHOICES.md section 10d:
* headline error is against the MEASURED exhaust temperature; error against the TRUE value is
  shown beside it for diagnosis only;
* a noise-floor row (RMS of measured minus true) on every split; dropout row counts printed;
* confidence intervals by resampling EPISODES, never rows;
* test_* and stress_* splits are refused unless unsealed explicitly; each unseal is logged.
"""
from __future__ import annotations

import csv
import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd

from vsensor.data.generate import load_split

SEALED_PREFIXES = ("test_", "stress_")
UNSEAL_LOG = Path(__file__).resolve().parents[3] / "reports" / "unseal_log.csv"
LOAD_BINS = [0.0, 0.40, 0.55, 0.70, 0.85, 1.01]


class SealedSplitError(RuntimeError):
    pass


def load_eval(data_dir, split: str, unseal_reason: str | None = None) -> pd.DataFrame:
    """Signals joined with truth for one split. Sealed splits need a reason, which is logged."""
    if split.startswith(SEALED_PREFIXES):
        if not unseal_reason:
            raise SealedSplitError(f"split '{split}' is sealed; pass unseal_reason (it is logged)")
        UNSEAL_LOG.parent.mkdir(parents=True, exist_ok=True)
        new = not UNSEAL_LOG.exists()
        with UNSEAL_LOG.open("a", newline="") as f:
            w = csv.writer(f)
            if new:
                w.writerow(["utc_time", "split", "reason"])
            w.writerow([dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"), split, unseal_reason])
    sig = load_split(data_dir, split, "signals")
    tru = load_split(data_dir, split, "truth")
    cols = ["episode_id", "t_s", "t_exh_true", "t_exh_steady", "load_fraction", "foul_tc", "foul_cool", "fault_active"]
    df = sig.merge(tru[cols], on=["episode_id", "t_s"], how="left", validate="1:1")
    df["fuel"] = np.where(df["fuel_code"] == 0, "diesel_like", "methanol")
    df["load_bin"] = pd.cut(df["load_fraction"], LOAD_BINS, right=False).astype(str)
    return df


def _ep_sums(df: pd.DataFrame, pred: np.ndarray, ref: str) -> pd.DataFrame:
    e = pred - df[ref].to_numpy()
    ok = np.isfinite(e)
    g = pd.DataFrame({"ep": df["episode_id"].to_numpy(), "n": ok.astype(float),
                      "s": np.where(ok, e, 0.0), "s2": np.where(ok, e * e, 0.0),
                      "a": np.where(ok, np.abs(e), 0.0)})
    return g.groupby("ep", sort=True).sum()


def metrics(df: pd.DataFrame, pred: np.ndarray, ref: str = "t_exh_meas", n_boot: int = 500, seed: int = 0) -> dict:
    """RMSE, MAE, bias with 95 % episode-bootstrap intervals."""
    g = _ep_sums(df, pred, ref)
    n, s, s2, a = (g[c].to_numpy() for c in ("n", "s", "s2", "a"))
    if n.sum() == 0:
        return dict(rows=0, episodes=len(g))
    def f(w):
        nn = (w * n).sum()
        return np.sqrt((w * s2).sum() / nn), (w * a).sum() / nn, (w * s).sum() / nn
    point = f(np.ones(len(g)))
    rng = np.random.default_rng(seed)
    boots = np.array([f(rng.multinomial(len(g), np.full(len(g), 1 / len(g))).astype(float)) for _ in range(n_boot)])
    lo, hi = np.percentile(boots, [2.5, 97.5], axis=0)
    out = dict(rows=int(n.sum()), episodes=len(g))
    for i, k in enumerate(("rmse", "mae", "bias")):
        out[k] = point[i]; out[k + "_lo"] = lo[i]; out[k + "_hi"] = hi[i]
    return out


def noise_floor(df: pd.DataFrame) -> float:
    d = (df["t_exh_meas"] - df["t_exh_true"]).to_numpy()
    return float(np.sqrt(np.nanmean(d * d)))


def lag_note(df: pd.DataFrame) -> dict:
    d = np.abs((df["t_exh_true"] - df["t_exh_steady"]).to_numpy())
    return dict(rms=float(np.sqrt(np.mean(d * d))), p99=float(np.percentile(d, 99)))


def report(df: pd.DataFrame, preds: dict[str, np.ndarray], n_boot: int = 300) -> pd.DataFrame:
    """One long table: model x group (all / fuel / load bin) x reference (measured / true)."""
    rows = []
    groups = [("all", "all", np.ones(len(df), bool))]
    for f in sorted(df["fuel"].unique()):
        groups.append(("fuel", f, (df["fuel"] == f).to_numpy()))
    for b in sorted(df["load_bin"].unique()):
        groups.append(("load_bin", b, (df["load_bin"] == b).to_numpy()))
    for name, p in preds.items():
        for gk, gv, m in groups:
            sub = df[m]
            for ref in ("t_exh_meas", "t_exh_true"):
                r = metrics(sub, p[m], ref, n_boot)
                r.update(model=name, group=gk, value=gv, reference=ref)
                rows.append(r)
    return pd.DataFrame(rows)
