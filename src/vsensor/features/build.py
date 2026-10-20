"""Feature construction (SIMULATION PROJECT). One function for training and for the later inference path.

Design document section 8. Every feature uses only the CURRENT and PAST values of the input signals
(causal windows, per episode). No hidden plant state and no exhaust-temperature history is used.

Variants (section 8.1):  A raw signals | B = A + physics features + nominal estimate |
                         C = B + dynamic features | D = C + input-quality flags
The nominal estimate is passed in (from M0b); for training rows it must be the cross-fitted one.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from vsensor.data.quality import RANGES
from vsensor.params import Registry, load_registry
from vsensor.sim.plant import Plant

RAW = ["n_rpm", "m_fuel", "p_scav", "t_scav", "n_tc", "t_amb", "p_amb", "fuel_code"]
PHYS = ["nominal_est", "lambda_proxy", "fuel_power", "heat_per_rev", "boost_ratio", "n_tc_corr", "cooler_dt"]
DYN = ["m_fuel_mean10", "m_fuel_mean60", "m_fuel_slope30", "n_rpm_mean10", "n_rpm_slope30", "since_change"]
FLAGS = ["f_n_tc_missing", "f_oor", "f_stuck"]
VARIANTS = {"A": RAW, "B": RAW + PHYS, "C": RAW + PHYS + DYN, "D": RAW + PHYS + DYN + FLAGS}
CHANGE_THRESHOLD = 0.03  # kg/s fuel-flow change over 10 s counted as a "large change" (about 5-10 % of the range)
MAX_SINCE = 600.0


def _g(df, col):
    return df.groupby("episode_id", sort=False)[col]


def build_features(df: pd.DataFrame, nominal_est: np.ndarray, reg: Registry | None = None,
                   n_tc_fill: float | None = None) -> pd.DataFrame:
    """df: signals sorted by (episode_id, t_s) at 1 Hz. Returns one row per input row, all variants' columns."""
    reg = reg or load_registry()
    plant = Plant(reg)
    f = pd.DataFrame(index=df.index)
    d = df.reset_index(drop=True)
    miss = d["n_tc"].isna().to_numpy()
    d["n_tc"] = _g(d, "n_tc").ffill()                      # causal fill with the last good value
    d["n_tc"] = d["n_tc"].fillna(n_tc_fill if n_tc_fill is not None else 12000.0)
    out = {c: d[c].to_numpy(float) for c in RAW}
    # --- B: physics
    lhv = d["fuel_code"].map({0: plant.fuels["diesel_like"].lhv_liquid, 1: plant.fuels["methanol"].lhv_liquid}).to_numpy(float)
    q = d["m_fuel"].to_numpy(float) * lhv
    rho = d["p_scav"].to_numpy(float) * 1e5 / (plant.r_air * d["t_scav"].to_numpy(float))
    lam_del = reg["nominal.lambda_del"]
    m_air = lam_del * rho * plant.v_disp * d["n_rpm"].to_numpy(float) / 60.0
    o2_air = plant.air.moles(1.0)["O2"]
    need = np.empty(len(d))
    for code, fid in ((0, "diesel_like"), (1, "methanol")):
        m = (d["fuel_code"] == code).to_numpy(); fu = plant.fuels[fid]
        need[m] = fu.o2_per_fuel * d["m_fuel"].to_numpy(float)[m] / fu.molar_mass
    out["nominal_est"] = np.asarray(nominal_est, float)
    out["lambda_proxy"] = o2_air * m_air / need
    out["fuel_power"] = q
    out["heat_per_rev"] = q / d["n_rpm"].to_numpy(float)
    out["boost_ratio"] = d["p_scav"].to_numpy(float) / d["p_amb"].to_numpy(float)
    out["n_tc_corr"] = d["n_tc"].to_numpy(float) / np.sqrt(d["t_scav"].to_numpy(float) / 288.15)
    out["cooler_dt"] = d["t_scav"].to_numpy(float) - d["t_amb"].to_numpy(float)
    # --- C: dynamic (past-only)
    def roll(col, w):
        return _g(d, col).rolling(w, min_periods=1).mean().reset_index(level=0, drop=True).sort_index().to_numpy()
    def slope(col, w):
        past = _g(d, col).shift(w)
        first = _g(d, col).transform("first")
        return ((d[col] - past.fillna(first)) / w).to_numpy()
    out["m_fuel_mean10"], out["m_fuel_mean60"] = roll("m_fuel", 10), roll("m_fuel", 60)
    out["m_fuel_slope30"] = slope("m_fuel", 30)
    out["n_rpm_mean10"], out["n_rpm_slope30"] = roll("n_rpm", 10), slope("n_rpm", 30)
    ch = (d["m_fuel"] - _g(d, "m_fuel").shift(10).fillna(_g(d, "m_fuel").transform("first"))).abs().to_numpy() > CHANGE_THRESHOLD
    last = pd.Series(np.where(ch, d["t_s"].to_numpy(float), np.nan), index=d.index)
    last = last.groupby(d["episode_id"], sort=False).ffill()
    out["since_change"] = np.minimum(d["t_s"].to_numpy(float) - last.fillna(-MAX_SINCE).to_numpy(), MAX_SINCE)
    # --- D: input-quality flags (inputs only)
    out["f_n_tc_missing"] = miss.astype(float)
    oor = np.zeros(len(d), bool)
    for col in ("n_rpm", "m_fuel", "p_scav", "t_scav", "t_amb", "p_amb"):
        lo, hi = RANGES[col]; v = d[col].to_numpy(float); oor |= (v < lo) | (v > hi)
    out["f_oor"] = oor.astype(float)
    stuck = np.zeros(len(d), bool)
    for col in ("m_fuel", "n_rpm", "p_scav", "n_tc"):
        sd = _g(d, col).rolling(60, min_periods=60).std().reset_index(level=0, drop=True).sort_index().to_numpy()
        stuck |= np.nan_to_num(sd, nan=1.0) == 0.0
    out["f_stuck"] = stuck.astype(float)
    res = pd.DataFrame(out, index=df.index)
    return res
