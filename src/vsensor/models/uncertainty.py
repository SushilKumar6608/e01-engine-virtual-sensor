"""Quantile model M3, split conformal calibration, out-of-range score and abstain rule (SIMULATION PROJECT).

Rules fixed in docs/MODEL_CHOICES.md section 10g before any result existed.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from vsensor.features.build import VARIANTS

ALPHA = 0.10
OOD_FEATURES = ["n_rpm", "m_fuel", "p_scav", "t_scav", "t_amb", "boost_ratio", "cooler_dt"]
OOD_K = 10
OOD_QUANTILE = 0.99
WIDTH_TOLERANCE_K = 30.0  # assumed engineering tolerance (full interval width), not sourced
M2_SETTINGS = dict(max_iter=300, learning_rate=0.08, l2_regularization=1.0, max_leaf_nodes=15, min_samples_leaf=1000, random_state=0)


class QuantileResidualModel:
    """5 % and 95 % quantiles of (measured - nominal estimate) with feature variant C."""
    def __init__(self, variant: str = "C", alpha: float = ALPHA):
        self.cols = VARIANTS[variant]; self.alpha = alpha

    def fit(self, F: pd.DataFrame, y: np.ndarray, nominal: np.ndarray):
        X, r = F[self.cols].to_numpy(), y - nominal
        self.lo = HistGradientBoostingRegressor(loss="quantile", quantile=self.alpha / 2, **M2_SETTINGS).fit(X, r)
        self.hi = HistGradientBoostingRegressor(loss="quantile", quantile=1 - self.alpha / 2, **M2_SETTINGS).fit(X, r)
        return self

    def predict_raw(self, F: pd.DataFrame, nominal: np.ndarray):
        X = F[self.cols].to_numpy()
        a, b = nominal + self.lo.predict(X), nominal + self.hi.predict(X)
        return np.minimum(a, b), np.maximum(a, b)  # no crossing


def conformal_correction(lo: np.ndarray, hi: np.ndarray, y: np.ndarray, alpha: float = ALPHA) -> float:
    """CQR: ceil((n+1)(1-alpha))/n quantile of max(lo - y, y - hi). Rows with a missing target are skipped."""
    ok = np.isfinite(y)
    score = np.maximum(lo[ok] - y[ok], y[ok] - hi[ok])
    n = len(score)
    k = int(np.ceil((n + 1) * (1 - alpha)))
    return float(np.inf if k > n else np.sort(score)[k - 1])


class OODScorer:
    """Mean distance to the k nearest training points, per fuel, in a standardised input space."""
    def __init__(self, k: int = OOD_K):
        self.k = k

    def fit(self, F: pd.DataFrame, fuel_code: np.ndarray, step_mask: np.ndarray):
        self.sc, self.nn = {}, {}
        for code in (0, 1):
            m = (fuel_code == code) & step_mask
            X = F.loc[m, OOD_FEATURES].to_numpy()
            self.sc[code] = StandardScaler().fit(X)
            self.nn[code] = NearestNeighbors(n_neighbors=self.k).fit(self.sc[code].transform(X))
        return self

    def score(self, F: pd.DataFrame, fuel_code: np.ndarray) -> np.ndarray:
        out = np.empty(len(F))
        for code in (0, 1):
            m = fuel_code == code
            if m.any():
                d, _ = self.nn[code].kneighbors(self.sc[code].transform(F.loc[m, OOD_FEATURES].to_numpy()))
                out[m] = d.mean(axis=1)
        return out


@dataclass
class AbstainRule:
    ood_threshold: float
    width_tolerance: float = WIDTH_TOLERANCE_K

    def decide(self, ood: np.ndarray, width: np.ndarray, F: pd.DataFrame) -> pd.DataFrame:
        r = pd.DataFrame({
            "input_out_of_range": F["f_oor"].to_numpy() > 0, "input_stuck": F["f_stuck"].to_numpy() > 0,
            "far_from_training": ood > self.ood_threshold, "interval_too_wide": width > self.width_tolerance})
        r["abstain"] = r.any(axis=1)
        return r
