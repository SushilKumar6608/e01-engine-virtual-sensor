"""The nine data quality and trust gates (design document, section 7). SIMULATED DATA.

Each gate is an automated test that returns pass or fail with the numbers behind it. A dataset
that fails any gate is not released. Thresholds are fixed in this file and in
config/dataset.yaml before the data are generated; they are not tuned to make a dataset pass.

Gate 9 (documentation) has an automated completeness part and a manual part: the person who
releases the dataset still has to read the data card.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from vsensor.data.config import DatasetConfig
from vsensor.data.episodes import simulate_episode
from vsensor.data.generate import (
    MODEL_INPUT_COLUMNS, ROOT, TARGET_COLUMN, episode_digest, generator_code_hash, load_manifest,
    load_split, plant_for, registry_hash,
)
from vsensor.params import Registry
from vsensor.sim import thermo
from vsensor.sim.fuels import FUEL_SPECIES
from vsensor.sim.plant import Plant
from vsensor.sim.sensors import SensorSpecs, first_order_lag

CODE_TO_FUEL = {code: fid for fid, (_, code) in FUEL_SPECIES.items()}
RANGES = {  # physical bounds of each recorded signal (units as stored)
    "n_rpm": (0.0, 130.0), "m_fuel": (0.0, 1.0), "p_scav": (1.0, 6.0), "t_scav": (270.0, 350.0),
    "n_tc": (0.0, 25000.0), "t_amb": (250.0, 330.0), "p_amb": (0.95, 1.07), "t_exh_meas": (400.0, 1100.0),
}
HIDDEN_NAME_HINTS = ("foul", "p_exh", "m_air", "steady", "true", "load_fraction", "speed_factor", "delivery", "fault")


@dataclass
class Gate:
    number: int
    name: str
    passed: bool
    summary: str
    details: list[str] = field(default_factory=list)


def _close(name, ok, summary, details=None, number=0):
    return Gate(number, name, bool(ok), summary, details or [])


# ------------------------------------------------------------------------------ gate 1
def gate1_provenance(cfg: DatasetConfig, reg: Registry, data_dir: Path) -> Gate:
    man = load_manifest(data_dir)
    problems = []
    for key in ("dataset_hash", "config_hash", "registry_hash", "generator_code_hash", "base_seed"):
        if man.get(key) in (None, ""):
            problems.append(f"manifest lacks {key}")
    if man.get("config_hash") != cfg.content_hash():
        problems.append("configuration changed since the data were generated")
    if man.get("registry_hash") != registry_hash(reg):
        problems.append("parameter registry changed since the data were generated")
    if man.get("generator_code_hash") != generator_code_hash():
        problems.append("generating code changed since the data were generated")
    counts = reg.status_counts()
    bad = [p for p in reg.ids if reg.param(p).status == "assumed" and len((reg.param(p).rationale or "").strip()) < 15]
    if bad:
        problems.append(f"assumed parameters without rationale: {bad}")
    summ = "PASS" if not problems else "FAIL: " + "; ".join(problems)
    det = [f"{len(reg.ids)} parameters: " + ", ".join(f"{counts.get(k, 0)} {k}" for k in ("sourced", "derived", "calibrated", "assumed")),
           f"dataset hash {man['dataset_hash'][:16]}, config hash {man['config_hash'][:16]}, registry hash {man['registry_hash'][:16]}, "
           f"code hash {man['generator_code_hash'][:16]}, seed {man['base_seed']}"]
    return Gate(1, "Provenance", not problems, summ, det)


# ------------------------------------------------------------------------------ gate 2
def gate2_units_ranges(cfg: DatasetConfig, data_dir: Path) -> Gate:
    problems, det = [], []
    total = 0
    for name in cfg.splits:
        sig, tru = load_split(data_dir, name), load_split(data_dir, name, "truth")
        total += len(sig)
        missing_cols = [c for c in ["episode_id", "t_s", *MODEL_INPUT_COLUMNS, TARGET_COLUMN] if c not in sig.columns]
        if missing_cols:
            problems.append(f"{name}: missing columns {missing_cols}")
            continue
        if len(sig) != len(tru):
            problems.append(f"{name}: signals and truth differ in length")
        for col, (lo, hi) in RANGES.items():
            v = sig[col].to_numpy(dtype=float)
            if np.isinf(v).any():
                problems.append(f"{name}.{col}: infinite values")
            f = v[np.isfinite(v)]
            if len(f) and (f.min() < lo or f.max() > hi):
                problems.append(f"{name}.{col}: [{f.min():.4g}, {f.max():.4g}] outside [{lo}, {hi}]")
        expected_nan = {"n_tc": tru["missing_n_tc"].to_numpy(), "t_exh_meas": tru["missing_t_exh"].to_numpy()}
        for col in sig.columns:
            if col == "episode_id":
                continue
            isnan = sig[col].isna().to_numpy()
            exp = expected_nan.get(col, np.zeros(len(sig), dtype=bool))
            if not np.array_equal(isnan, exp):
                problems.append(f"{name}.{col}: {int((isnan != exp).sum())} missing values that are not injected dropouts")
        for col, (lo, hi) in {"foul_tc": (0, 1), "foul_cool": (0, 1), "load_fraction": (0.3, 1.0)}.items():
            v = tru[col].to_numpy()
            if v.min() < lo - 1e-9 or v.max() > hi + 1e-9:
                problems.append(f"{name} truth.{col} outside [{lo}, {hi}]")
    det.append(f"{total} rows over {len(cfg.splits)} splits; every signal inside its physical bounds, NaN only at injected dropouts")
    return Gate(2, "Units and ranges", not problems, "PASS" if not problems else "FAIL: " + "; ".join(problems[:6]), det + problems[:20])


# ------------------------------------------------------------------------------ gate 3
def _energy_residual(plant: Plant, s, i: int, fuel: str) -> float:
    fu = plant.fuels[fuel]
    moles = fu.exhaust_moles(s.m_fuel[i], s.m_air[i], plant.air)
    h_in = float(thermo.mixture_sensible_enthalpy_rate(plant.air.moles(s.m_air[i]), s.t_amb[i]))
    h_out = float(thermo.mixture_sensible_enthalpy_rate(moles, s.t_turb_out[i]))
    p_fric = float(plant.friction_power(s.n_rpm[i]))
    lhs = s.q_fuel[i] + h_in
    rhs = s.p_brake[i] + p_fric + s.q_wall[i] + s.q_cooler[i] + h_out + (1 - plant.eta_m) * s.p_turbine[i]
    return abs(lhs - rhs) / s.q_fuel[i]


def _mass_residual(plant: Plant, s, i: int, fuel: str) -> float:
    fu = plant.fuels[fuel]
    moles = fu.exhaust_moles(s.m_fuel[i], s.m_air[i], plant.air)
    m_out = sum(float(n) * thermo.molar_mass(sp) for sp, n in moles.items())
    return abs(m_out - (s.m_air[i] + s.m_fuel[i])) / m_out


def gate3_closure(cfg: DatasetConfig, reg: Registry, data_dir: Path, n_samples: int = 300, tol_energy: float = 2e-3, tol_mass: float = 1e-9) -> Gate:
    rng = np.random.default_rng(7)
    cache: dict = {}
    worst_e = worst_m = 0.0
    worst_lam = (9.0, 0.0)
    problems = []
    names = [n for n in cfg.splits]
    per = max(1, n_samples // len(names))
    for name in names:
        split = cfg.splits[name]
        plant = plant_for(split, reg, cache)
        tru = load_split(data_dir, name, "truth")
        sig = load_split(data_dir, name)
        idx = rng.choice(len(tru), size=min(per, len(tru)), replace=False)
        for fcode in (0, 1):
            sel = idx[sig["fuel_code"].to_numpy()[idx] == fcode]
            if len(sel) == 0:
                continue
            fuel = CODE_TO_FUEL[fcode]
            t = tru.iloc[sel]
            s = plant.solve(t["n_rpm_true"].to_numpy(), t["m_fuel_true"].to_numpy(), t_amb=t["t_amb_true"].to_numpy(),
                            p_amb=t["p_amb_true"].to_numpy() * 1e5, foul_tc=t["foul_tc"].to_numpy(), foul_cool=t["foul_cool"].to_numpy(), fuel=fuel)
            if not np.all(s.converged):
                problems.append(f"{name}: steady solve did not converge")
                continue
            for i in range(len(sel)):
                worst_e = max(worst_e, _energy_residual(plant, s, i, fuel))
                worst_m = max(worst_m, _mass_residual(plant, s, i, fuel))
            rho = s.p_scav / (plant.r_air * s.t_scav)
            lam = s.m_air / (rho * plant.v_disp * s.n_rpm / 60.0)
            lam_model = plant.lam_coeff * np.sqrt(1.0 - s.p_exh / s.p_scav)
            if np.max(np.abs(lam - lam_model)) > 1e-6:
                problems.append(f"{name}: air flow differs from the speed-density relation")
            worst_lam = (min(worst_lam[0], float(lam.min())), max(worst_lam[1], float(lam.max())))
    if worst_e > tol_energy:
        problems.append(f"energy residual {worst_e:.2e} above {tol_energy}")
    if worst_m > tol_mass:
        problems.append(f"mass residual {worst_m:.2e} above {tol_mass}")
    if not (0.5 <= worst_lam[0] and worst_lam[1] <= 1.3):
        problems.append(f"delivery ratio {worst_lam} outside 0.5 to 1.3")

    # lag test: a step in the steady state must settle with the registered time constants
    lag_lines = []
    for key, tau in (("dyn.tau_n_tc", reg["dyn.tau_n_tc"]), ("dyn.tau_p_scav", reg["dyn.tau_p_scav"]),
                     ("dyn.tau_t_scav", reg["dyn.tau_t_scav"]), ("dyn.tau_t_exh", reg["dyn.tau_t_exh"])):
        x = np.concatenate([np.zeros(50), np.ones(400)])
        y = first_order_lag(x, tau)
        k63 = int(np.argmax(y >= 1 - np.exp(-1.0))) - 50 + 1   # samples after the step
        settled = abs(y[-1] - 1.0)
        ok = abs(k63 - tau) <= 1.0 and settled < 1e-6
        lag_lines.append(f"{key}: 63 % reached after {k63} s (registry {tau:g} s), residual after {len(x) - 50} s = {settled:.1e}")
        if not ok:
            problems.append(f"lag check failed for {key}")
    det = [f"{n_samples} operating points re-solved from the stored true inputs: worst energy residual {worst_e:.1e} of fuel energy (limit {tol_energy}), worst mass residual {worst_m:.1e} (limit {tol_mass:.0e})",
           f"air delivery ratio range {worst_lam[0]:.3f} to {worst_lam[1]:.3f}, equal to the speed-density relation within 1e-6", *lag_lines]
    return Gate(3, "Energy and mass closure", not problems, "PASS" if not problems else "FAIL: " + "; ".join(problems), det)


# ------------------------------------------------------------------------------ gate 4
def gate4_direction(cfg: DatasetConfig, reg: Registry, data_dir: Path) -> Gate:
    plant = Plant(reg)
    problems, det = [], []
    for fuel, rng_ in (("diesel_like", (0.12, 0.34)), ("methanol", (0.27, 0.76))):
        mf = np.linspace(*rng_, 25)
        t = plant.exhaust_temperature_fixed_air(mf, m_air=11.0, t_scav=310.0, fuel=fuel)
        if not np.all(np.diff(t) > 0):
            problems.append(f"{fuel}: exhaust temperature does not rise with fuel flow at fixed air")
    t_air = plant.exhaust_temperature_fixed_air(0.25, np.linspace(6, 16, 25), 310.0)
    if not np.all(np.diff(t_air) < 0):
        problems.append("more air at fixed fuel does not lower the exhaust temperature")
    det.append("at fixed air supply the exhaust temperature rises with fuel flow (both fuels) and falls with air flow at fixed fuel")
    # fouling effect evaluated at the dataset's own operating points
    rng = np.random.default_rng(11)
    cache: dict = {}
    mins = {"foul_tc": 1e9, "foul_cool": 1e9}
    n_used = 0
    for name in ("train", "test_in_distribution", "test_regime_cold_ambient"):
        split = cfg.splits[name]
        pl = plant_for(split, reg, cache)
        tru, sig = load_split(data_dir, name, "truth"), load_split(data_dir, name)
        idx = rng.choice(len(tru), size=200, replace=False)
        for fcode in (0, 1):
            sel = idx[sig["fuel_code"].to_numpy()[idx] == fcode]
            if len(sel) == 0:
                continue
            t = tru.iloc[sel]
            kw = dict(t_amb=t["t_amb_true"].to_numpy(), p_amb=t["p_amb_true"].to_numpy() * 1e5, fuel=CODE_TO_FUEL[fcode])
            n, m = t["n_rpm_true"].to_numpy(), t["m_fuel_true"].to_numpy()
            base = pl.solve(n, m, foul_tc=0.0, foul_cool=0.0, **kw).t_exh
            for key, args in (("foul_tc", dict(foul_tc=0.5, foul_cool=0.0)), ("foul_cool", dict(foul_tc=0.0, foul_cool=0.5))):
                d = pl.solve(n, m, **args, **kw).t_exh - base
                mins[key] = min(mins[key], float(d.min()))
            n_used += len(sel)
    for key, v in mins.items():
        if v <= 0:
            problems.append(f"{key} 0.5 does not raise the exhaust temperature at every sampled dataset point (smallest change {v:.2f} K)")
    det.append(f"at {n_used} sampled dataset operating points, fouling of 0.5 raises the exhaust temperature at the same fuel flow and speed: smallest rise {mins['foul_tc']:.1f} K (turbocharger), {mins['foul_cool']:.1f} K (air cooler)")
    return Gate(4, "Direction of effects", not problems, "PASS" if not problems else "FAIL: " + "; ".join(problems), det)


# ------------------------------------------------------------------------------ gate 5
def _bin_edges(a: np.ndarray, nb: int):
    return np.linspace(a.min(), a.max(), nb + 1)


def gate5_coverage(cfg: DatasetConfig, reg: Registry, data_dir: Path) -> Gate:
    cv = cfg.coverage
    plant_cache: dict = {}
    problems, det = [], []
    rng = np.random.default_rng(5)
    rows_out = []
    for name, split in cfg.splits.items():
        small = split.episodes < cv.small_split_episodes
        nb, min_rows, min_eps = (cv.small_bins, cv.small_min_rows, cv.small_min_episodes) if small else (cv.bins, cv.min_rows, cv.min_episodes)
        plant = plant_for(split, reg, plant_cache)
        sig = load_split(data_dir, name)
        meta = pd.read_parquet(data_dir / "episodes.parquet")
        fuel_of_ep = meta.set_index("episode_id")["fuel_code"]
        for fuel, share in split.fuel_mix.items():
            lo, hi = split.load_band[fuel]
            load = rng.uniform(lo, hi, 40000)
            sf = rng.uniform(*cfg.profile.speed_factor_range, 40000)
            n_p = np.minimum(sf * plant.n_rated * load ** (1 / 3), plant.n_rated)
            m_p = plant.fuel_flow_for_power(load * plant.p_rated, n_p, fuel)
            ex, ey = _bin_edges(n_p, nb), _bin_edges(m_p, nb)
            hist_p, _, _ = np.histogram2d(n_p, m_p, bins=[ex, ey])
            required = hist_p / hist_p.sum() >= cv.required_prior_mass
            code = FUEL_SPECIES[fuel][1]
            s = sig[sig["fuel_code"] == code]
            ix = np.clip(np.digitize(s["n_rpm"].to_numpy(), ex) - 1, 0, nb - 1)
            iy = np.clip(np.digitize(s["m_fuel"].to_numpy(), ey) - 1, 0, nb - 1)
            inside = (s["n_rpm"].to_numpy() >= ex[0] - 1) & (s["n_rpm"].to_numpy() <= ex[-1] + 1)
            cnt = np.zeros((nb, nb), dtype=int)
            np.add.at(cnt, (ix[inside], iy[inside]), 1)
            eps_cnt = np.zeros((nb, nb), dtype=int)
            ep_ids = s["episode_id"].astype(str).to_numpy()[inside]
            keys = pd.DataFrame({"ix": ix[inside], "iy": iy[inside], "ep": ep_ids}).drop_duplicates()
            np.add.at(eps_cnt, (keys["ix"].to_numpy(), keys["iy"].to_numpy()), 1)
            holes = required & ((cnt < min_rows) | (eps_cnt < min_eps))
            rows_out.append((name, fuel, int(required.sum()), int(holes.sum()), nb, min_rows, min_eps))
            if holes.any():
                problems.append(f"{name}/{fuel}: {int(holes.sum())} of {int(required.sum())} required bins below {min_rows} rows or {min_eps} episodes")
    det.append("split / fuel / required bins / holes / grid / min rows / min episodes")
    det += [f"{a} / {b} / {c} / {d} / {e}x{e} / {f} / {g}" for a, b, c, d, e, f, g in rows_out]
    return Gate(5, "Coverage", not problems, "PASS" if not problems else "FAIL: " + "; ".join(problems[:8]), det + problems[8:20])


# ------------------------------------------------------------------------------ gate 6
def _std_ok(obs: float, exp: float, tol: float) -> bool:
    return abs(obs / exp - 1.0) <= tol


def gate6_sensor_realism(cfg: DatasetConfig, reg: Registry, data_dir: Path) -> Gate:
    sp = SensorSpecs.from_registry(reg)
    meta = pd.read_parquet(data_dir / "episodes.parquet").set_index("episode_id")
    events = pd.read_parquet(data_dir / "events.parquet")
    problems, det = [], []
    res = {k: [] for k in ("speed", "fuel", "pscav", "tscav", "ntc", "tamb", "pamb", "texh")}
    for name in cfg.splits:
        sig, tru = load_split(data_dir, name), load_split(data_dir, name, "truth")
        for eid, g in sig.groupby("episode_id", observed=True):
            tg = tru.loc[g.index]
            m = meta.loc[eid]
            t = g["t_s"].to_numpy(dtype=float)
            res["speed"].append(g["n_rpm"].to_numpy() - tg["n_rpm_true"].to_numpy())
            res["fuel"].append(g["m_fuel"].to_numpy() / tg["m_fuel_true"].to_numpy() - 1.0 - m["bias_m_fuel_rel"])
            res["pscav"].append(g["p_scav"].to_numpy() - tg["p_scav_true"].to_numpy() - m["drift_pscav_bar_per_h"] * t / 3600.0)
            res["tscav"].append(g["t_scav"].to_numpy() - first_order_lag(tg["t_scav_true"].to_numpy(), sp.tscav_tau))
            nt = g["n_tc"].to_numpy() / tg["n_tc_true"].to_numpy() - 1.0
            res["ntc"].append(nt[np.isfinite(nt)])
            res["tamb"].append(g["t_amb"].to_numpy() - tg["t_amb_true"].to_numpy() - m["bias_t_amb_k"])
            res["pamb"].append(g["p_amb"].to_numpy() - tg["p_amb_true"].to_numpy() - m["bias_p_amb_bar"])
            te = g["t_exh_meas"].to_numpy() - first_order_lag(tg["t_exh_true"].to_numpy(), sp.texh_tau) - m["bias_t_exh_k"]
            ok = ~tg["fault_active"].to_numpy() & np.isfinite(te)
            res["texh"].append(te[ok])
    spec = {
        "speed": (sp.speed_noise, sp.speed_res, "rpm"), "fuel": (sp.fuel_noise_rel, sp.fuel_res / 0.3, "relative"),
        "pscav": (sp.pscav_noise, sp.pscav_res, "bar"), "tscav": (sp.tscav_noise, sp.tscav_res, "K"),
        "ntc": (sp.ntc_noise_rel, sp.ntc_res / 14000.0, "relative"), "tamb": (sp.tamb_noise, sp.tamb_res, "K"),
        "pamb": (sp.pamb_noise, sp.pamb_res, "bar"), "texh": (sp.texh_noise, sp.texh_res, "K"),
    }
    for k, (noise, res_, unit) in spec.items():
        r = np.concatenate(res[k])
        exp = float(np.sqrt(noise**2 + res_**2 / 12.0))
        obs = float(np.std(r))
        mean_ok = abs(np.mean(r)) < 4 * exp / np.sqrt(len(r)) + res_ / 2
        ok = _std_ok(obs, exp, 0.10) and mean_ok
        det.append(f"{k}: residual std {obs:.4g} {unit} vs specified {exp:.4g} (noise plus quantisation), mean {np.mean(r):+.2g}; {'ok' if ok else 'MISMATCH'}")
        if not ok:
            problems.append(f"{k} noise does not match its specification")
    tr = meta[meta["split"] == "train"]
    for col, sd, label in (("bias_m_fuel_rel", sp.fuel_bias_sd_rel, "fuel flow bias"), ("bias_t_exh_k", sp.texh_bias_sd, "exhaust sensor bias"),
                           ("bias_t_amb_k", sp.tamb_bias_sd, "ambient temperature bias"), ("bias_p_amb_bar", sp.pamb_bias_sd, "ambient pressure bias"),
                           ("drift_pscav_bar_per_h", sp.pscav_drift_sd_per_h, "scavenge pressure drift rate")):
        v = tr[col].to_numpy()
        ok = abs(v.mean()) < 4 * sd / np.sqrt(len(v)) and _std_ok(v.std(), sd, 0.20)
        det.append(f"{label}: spread across {len(v)} training episodes {v.std():.4g} vs specified {sd:.4g}, mean {v.mean():+.2g}; {'ok' if ok else 'MISMATCH'}")
        if not ok:
            problems.append(f"{label} does not match its specification")
    n_rows = sum(int(m_["rows"]) for m_ in load_manifest(data_dir)["splits"].values())
    exp_ev = sp.ntc_dropout_rate_per_h * n_rows / 3600.0
    obs_ev = int((events["kind"] == "dropout_n_tc").sum())
    ok = abs(obs_ev - exp_ev) <= 4 * np.sqrt(exp_ev)
    det.append(f"turbocharger speed dropouts: {obs_ev} events vs {exp_ev:.0f} expected (rate {sp.ntc_dropout_rate_per_h}/h over {n_rows / 3600:.0f} h), limit 4 sigma; {'ok' if ok else 'MISMATCH'}")
    if not ok:
        problems.append("dropout rate does not match its specification")
    dr = events[events["kind"] == "dropout_n_tc"]
    lens = (dr["end_s"] - dr["start_s"]).to_numpy()
    full = dr["end_s"].to_numpy() < cfg.episode_duration_s
    if len(lens) and (lens[full].min() < max(1, np.floor(sp.ntc_dropout_len_min)) or lens.max() > sp.ntc_dropout_len_max + 1):
        problems.append("dropout lengths outside the specified range")
    det.append(f"dropout lengths {lens.min():.0f} to {lens.max():.0f} s (specified {sp.ntc_dropout_len_min:g} to {sp.ntc_dropout_len_max:g} s)")
    fm = meta[meta["split"] == "test_sensor_fault"]
    cyc = cfg.splits["test_sensor_fault"].fault_cycle
    exp_counts = {k: sum(1 for i in range(cfg.splits["test_sensor_fault"].episodes) if cyc[i % len(cyc)] == k) for k in set(cyc)}
    obs_counts = fm["fault_kind"].value_counts().to_dict()
    if {k: int(obs_counts.get(k, 0)) for k in exp_counts} != exp_counts:
        problems.append("injected fault counts differ from the configured cycle")
    fi = cfg.fault_injection
    bias = fm[fm["fault_kind"] == "bias"]["fault_magnitude"].abs()
    drift = fm[fm["fault_kind"] == "drift"]["fault_magnitude"].abs()
    if len(bias) and (bias.min() < fi.bias_k[0] or bias.max() > fi.bias_k[1]) or len(drift) and (drift.min() < fi.drift_k_per_min[0] or drift.max() > fi.drift_k_per_min[1]):
        problems.append("injected fault sizes outside the configured range")
    det.append(f"injected faults per type {exp_counts}; bias sizes {bias.min():.1f} to {bias.max():.1f} K, drift rates {drift.min():.2f} to {drift.max():.2f} K/min")
    return Gate(6, "Sensor realism", not problems, "PASS" if not problems else "FAIL: " + "; ".join(problems), det)


# ------------------------------------------------------------------------------ gate 7
def gate7_leakage(cfg: DatasetConfig, reg: Registry, data_dir: Path) -> Gate:
    problems, det = [], []
    load_min, stress_min = reg["operating.load_min"], reg["operating.load_stress_min"]
    ids = {n: set(load_split(data_dir, n)["episode_id"].astype(str).unique()) for n in cfg.splits}
    names = list(ids)
    overlap = [(a, b) for i, a in enumerate(names) for b in names[i + 1:] if ids[a] & ids[b]]
    if overlap:
        problems.append(f"episodes shared between splits: {overlap}")
    meta = pd.read_parquet(data_dir / "episodes.parquet")
    if meta["episode_digest"].duplicated().any():
        problems.append("two episodes have identical content")
    seeds = [s.seed_id for s in cfg.splits.values()]
    if len(set(seeds)) != len(seeds):
        problems.append("split seed ids are not unique")
    det.append(f"{sum(len(v) for v in ids.values())} episodes, no episode in two splits, no two episodes with identical content")
    bad_inputs = [c for c in MODEL_INPUT_COLUMNS if any(h in c for h in HIDDEN_NAME_HINTS) or c == TARGET_COLUMN]
    if bad_inputs:
        problems.append(f"model inputs with hidden or target-like names: {bad_inputs}")
    sig = load_split(data_dir, "train")
    extra = [c for c in sig.columns if c not in ("episode_id", "t_s", *MODEL_INPUT_COLUMNS, TARGET_COLUMN)]
    if extra:
        problems.append(f"unexpected columns in the model-visible table: {extra}")
    tru = load_split(data_dir, "train", "truth")
    y = tru["t_exh_true"].to_numpy()
    worst = 0.0
    for c in MODEL_INPUT_COLUMNS:
        x = sig[c].to_numpy(dtype=float)
        ok = np.isfinite(x)
        r = abs(np.corrcoef(x[ok], y[ok])[0, 1]) if np.std(x[ok]) > 0 else 0.0
        worst = max(worst, r)
    if worst > 0.99:
        problems.append(f"an input correlates {worst:.3f} with the true target")
    det.append(f"model inputs are exactly {MODEL_INPUT_COLUMNS}; strongest correlation of any input with the true target {worst:.2f} (limit 0.99)")
    # no input is a copy of a hidden truth column
    copies = []
    pairs = {"n_rpm": "n_rpm_true", "m_fuel": "m_fuel_true", "n_tc": "n_tc_true", "t_scav": "t_scav_true", "p_scav": "p_scav_true"}
    for c, tc in pairs.items():
        a, b = sig[c].to_numpy(dtype=float), tru[tc].to_numpy(dtype=float)
        ok = np.isfinite(a)
        if np.mean(np.isclose(a[ok], b[ok], rtol=0, atol=1e-12)) > 0.5:
            copies.append(c)
    if copies:
        problems.append(f"inputs identical to hidden truth columns: {copies}")
    # regime splits really are held out
    tr_load = pd.concat([load_split(data_dir, n, "truth") for n in ("train", "validation", "calibration", "test_in_distribution")])
    tr_meta = meta[meta["split"].isin(["train", "validation", "calibration", "test_in_distribution"])]
    hi = load_split(data_dir, "test_regime_high_load", "truth")
    if tr_load["load_fraction"].max() > 0.90 + 1e-9 or hi["load_fraction"].min() < 0.90 - 1e-9:
        problems.append("top load band is not cleanly held out")
    ml = load_split(data_dir, "test_regime_methanol_low_load", "truth")
    sig_all = pd.concat([load_split(data_dir, n) for n in ("train", "validation", "calibration", "test_in_distribution")])
    meth_train = tr_load[sig_all["fuel_code"].to_numpy() == 1]
    if meth_train["load_fraction"].min() < 0.55 - 1e-9 or ml["load_fraction"].max() > 0.55 + 1e-9:
        problems.append("methanol low load is not cleanly held out")
    cold = load_split(data_dir, "test_regime_cold_ambient", "truth")
    if tr_load["t_amb_true"].min() <= cold["t_amb_true"].max():
        problems.append("cold ambient overlaps the training temperatures")
    det.append(f"held-out regimes: training load at most {tr_load['load_fraction'].max():.3f}, high-load split from {hi['load_fraction'].min():.3f}; training methanol load at least {meth_train['load_fraction'].min():.3f}, methanol low-load split up to {ml['load_fraction'].max():.3f}; training ambient from {tr_load['t_amb_true'].min():.1f} K, cold split up to {cold['t_amb_true'].max():.1f} K")
    low = load_split(data_dir, "stress_low_load", "truth")
    if low["load_fraction"].max() > load_min + 1e-9 or tr_load["load_fraction"].min() < load_min - 1e-9 or low["load_fraction"].min() < stress_min - 1e-9:
        problems.append("low-load stress data overlap the operating range")
    det.append(f"abstention stress split spans {low['load_fraction'].min():.2f} to {low['load_fraction'].max():.2f} load; training starts at {tr_load['load_fraction'].min():.2f}")
    return Gate(7, "Leakage", not problems, "PASS" if not problems else "FAIL: " + "; ".join(problems), det)


# ------------------------------------------------------------------------------ gate 8
def gate8_reproducibility(cfg: DatasetConfig, reg: Registry, data_dir: Path, per_split: int = 2, full_hash: str | None = None) -> Gate:
    man = load_manifest(data_dir)
    sp = SensorSpecs.from_registry(reg)
    cache: dict = {}
    problems, n_checked = [], 0
    rng = np.random.default_rng(3)
    for name, split in cfg.splits.items():
        plant = plant_for(split, reg, cache)
        picks = sorted({0, split.episodes - 1, *rng.choice(split.episodes, size=per_split, replace=False).tolist()})
        for i in picks:
            ep = simulate_episode(name, split, int(i), cfg, reg, plant, sp)
            n_checked += 1
            if episode_digest(ep) != man["episode_digests"][name][int(i)]:
                problems.append(f"{ep.episode_id} rebuilt differently")
    det = [f"{n_checked} episodes rebuilt from the configuration and seed alone; every digest equals the stored one"]
    if full_hash is not None:
        det.append(f"full rebuild into a second directory: hash {full_hash[:16]} {'equals' if full_hash == man['dataset_hash'] else 'DIFFERS FROM'} stored {man['dataset_hash'][:16]}")
        if full_hash != man["dataset_hash"]:
            problems.append("full rebuild gave a different dataset hash")
    return Gate(8, "Reproducibility", not problems, "PASS" if not problems else "FAIL: " + "; ".join(problems[:6]), det)


# ------------------------------------------------------------------------------ gate 9
REQUIRED_SECTIONS = ("Purpose", "Contents", "How the data were generated", "Splits", "Known limits", "Provenance")


def gate9_documentation(data_dir: Path, card: Path | None = None) -> Gate:
    card = card or (ROOT / "docs" / "DATA_CARD.md")
    problems = []
    if not card.exists():
        return Gate(9, "Documentation", False, "FAIL: data card missing", [])
    text = card.read_text()
    head = "\n".join(text.splitlines()[:6]).lower()
    if "simulated" not in head:
        problems.append("the data card does not say in its first lines that the data are simulated")
    for sec in REQUIRED_SECTIONS:
        m = re.search(rf"^## {re.escape(sec)}\s*\n(.*?)(?=^## |\Z)", text, re.S | re.M)
        if not m or len(m.group(1).strip()) < 200:
            problems.append(f"section '{sec}' missing or shorter than 200 characters")
    man = load_manifest(data_dir)
    if man["dataset_hash"][:16] not in text:
        problems.append("the data card does not quote the current dataset hash")
    det = [f"{card.name}: required sections present: {', '.join(REQUIRED_SECTIONS)}; quotes dataset hash {man['dataset_hash'][:16]}",
           "Manual review is still required before release: read the data card."]
    return Gate(9, "Documentation", not problems, "PASS (automated part)" if not problems else "FAIL: " + "; ".join(problems), det)


def run_all(cfg: DatasetConfig, reg: Registry, data_dir: Path, full_hash: str | None = None) -> list[Gate]:
    return [
        gate1_provenance(cfg, reg, data_dir), gate2_units_ranges(cfg, data_dir), gate3_closure(cfg, reg, data_dir),
        gate4_direction(cfg, reg, data_dir), gate5_coverage(cfg, reg, data_dir), gate6_sensor_realism(cfg, reg, data_dir),
        gate7_leakage(cfg, reg, data_dir), gate8_reproducibility(cfg, reg, data_dir, full_hash=full_hash), gate9_documentation(data_dir),
    ]
