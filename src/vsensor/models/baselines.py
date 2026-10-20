"""Baselines fitted on TRAIN episodes only (SIMULATION PROJECT). See docs/MODEL_CHOICES.md section 10 and 10d.

Units: the dataset stores pressures in bar; the nominal model works in Pa. ``to_si`` is the ONE place
where the conversion happens (tests/test_baselines.py checks it).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from vsensor.params import Registry, load_registry
from vsensor.sim.nominal import NominalModel
from vsensor.sim.plant import Plant

BAR = 1.0e5


def to_si(df: pd.DataFrame) -> dict[str, np.ndarray]:
    """Dataset columns (bar, K, rpm, kg/s) to the nominal model's SI inputs (Pa)."""
    return dict(n_rpm=df["n_rpm"].to_numpy(float), m_fuel=df["m_fuel"].to_numpy(float),
                p_scav=df["p_scav"].to_numpy(float) * BAR, t_scav=df["t_scav"].to_numpy(float),
                fuel_code=df["fuel_code"].to_numpy(float))


def fit_rows(df: pd.DataFrame, step: int = 10) -> pd.DataFrame:
    """Training rows used for fitting: measured target present, every ``step``-th second."""
    m = df["t_exh_meas"].notna() & (df["t_s"] % step == 0)
    return df[m]


class Base:
    name = "base"
    def fit(self, train: pd.DataFrame): return self
    def predict(self, df: pd.DataFrame) -> np.ndarray: raise NotImplementedError


class PerFuelConstant(Base):
    name = "per_fuel_constant"
    def fit(self, train):
        d = train[train["t_exh_meas"].notna()]
        self.c = d.groupby("fuel_code")["t_exh_meas"].mean().to_dict()
        return self
    def predict(self, df):
        return df["fuel_code"].map(self.c).to_numpy(float)


class LoadLookup(Base):
    """Load proxy = fuel flow x lower heating value over the rated fuel-energy rate (not a measured signal)."""
    name = "load_lookup"
    def __init__(self, reg: Registry | None = None, bins: int = 20):
        reg = reg or load_registry()
        plant = Plant(reg)
        self.q_rated = plant.q_ref
        self.lhv = {0: plant.fuels["diesel_like"].lhv_liquid, 1: plant.fuels["methanol"].lhv_liquid}
        self.bins = bins
    def _x(self, df):
        return df["m_fuel"].to_numpy(float) * df["fuel_code"].map(self.lhv).to_numpy(float) / self.q_rated
    def fit(self, train):
        d = train[train["t_exh_meas"].notna()]
        x = self._x(d)
        self.tab = {}
        for code in (0, 1):
            m = (d["fuel_code"] == code).to_numpy()
            edges = np.quantile(x[m], np.linspace(0, 1, self.bins + 1))
            idx = np.clip(np.searchsorted(edges, x[m], side="right") - 1, 0, self.bins - 1)
            xc = np.array([x[m][idx == i].mean() for i in range(self.bins)])
            yc = np.array([d["t_exh_meas"].to_numpy()[m][idx == i].mean() for i in range(self.bins)])
            self.tab[code] = (xc, yc)
        return self
    def predict(self, df):
        x = self._x(df); out = np.empty(len(df))
        for code, (xc, yc) in self.tab.items():
            m = (df["fuel_code"] == code).to_numpy()
            out[m] = np.interp(x[m], xc, yc)  # clamps outside the training range
        return out


class M0(Base):
    """Nominal heat balance: constant air delivery ratio refitted on train; exhaust share from the rated heat balance."""
    name = "M0"
    def __init__(self, reg: Registry | None = None):
        self.nm = NominalModel(reg)
    def _pred(self, df, lam):
        s = to_si(df); self.nm.lam = lam
        return self.nm.predict_by_code(s["n_rpm"], s["m_fuel"], s["p_scav"], s["t_scav"], s["fuel_code"])
    def fit(self, train):
        d = fit_rows(train); y = d["t_exh_meas"].to_numpy()
        r = least_squares(lambda x: self._pred(d, x[0]) - y, [self.nm.reg["nominal.lambda_del"]])
        self.theta = r.x
        return self
    def predict(self, df):
        return self._pred(df, self.theta[0])


class M0b(M0):
    """Delivery ratio = a + b * (p_scav / p_amb). Multi-start; only a and b are fitted."""
    name = "M0b"
    def _pred(self, df, ab):
        s = to_si(df); r = df["p_scav"].to_numpy(float) / df["p_amb"].to_numpy(float)
        lam = ab[0] + ab[1] * r; out = np.empty(len(df))
        for code, fid in ((0, "diesel_like"), (1, "methanol")):
            m = s["fuel_code"] == code
            self.nm.lam = lam[m]
            out[m] = self.nm.predict(s["n_rpm"][m], s["m_fuel"][m], s["p_scav"][m], s["t_scav"][m], fid)
        self.nm.lam = self.nm.reg["nominal.lambda_del"]
        return out
    def fit(self, train, starts=None, step=10):
        d = fit_rows(train, step); y = d["t_exh_meas"].to_numpy()
        lam0 = self.nm.reg["nominal.lambda_del"]
        starts = starts or [[lam0, 0.0], [lam0 + 0.2, -0.05], [lam0 - 0.2, 0.05]]
        fits = [least_squares(lambda x: self._pred(d, x) - y, s0) for s0 in starts]
        self.starts_result = [(f.x.tolist(), float(np.sqrt(np.mean(f.fun ** 2)))) for f in fits]
        self.theta = min(fits, key=lambda f: f.cost).x
        return self
    def predict(self, df):
        return self._pred(df, self.theta)


def bootstrap_params(train: pd.DataFrame, n_boot: int = 20, step: int = 30, seed: int = 0, reg=None) -> np.ndarray:
    """Episode-level bootstrap of the M0b parameters (identifiability check)."""
    eps = train["episode_id"].unique(); rng = np.random.default_rng(seed); out = []
    grouped = {e: g for e, g in fit_rows(train, step).groupby("episode_id")}
    for _ in range(n_boot):
        pick = rng.choice(eps, len(eps))
        d = pd.concat([grouped[e] for e in pick if e in grouped])
        out.append(M0b(reg).fit(d, starts=[[0.9, 0.0]], step=1).theta)
    return np.array(out)


def crossfit_m0b(train: pd.DataFrame, folds: int = 5, reg=None, seed: int = 0) -> np.ndarray:
    """Out-of-fold M0b predictions on train (5 episode-grouped folds), for the features stage."""
    eps = np.array(sorted(train["episode_id"].unique())); rng = np.random.default_rng(seed); rng.shuffle(eps)
    fold_of = {e: i % folds for i, e in enumerate(eps)}
    f = train["episode_id"].map(fold_of).to_numpy(); out = np.empty(len(train))
    for k in range(folds):
        m = M0b(reg).fit(train[f != k]); out[f == k] = m.predict(train[f == k])
    return out
