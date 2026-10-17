"""One simulated voyage segment ("episode") at 1 Hz (SIMULATED DATA).

Plain-language summary
----------------------
An episode is 30 minutes of an engine running with its own weather, fuel and health state.

1. Draw the episode's fixed properties: fuel, ambient air, how fouled the turbocharger and the
   air cooler are at the start and how fast the fouling grows.
2. Draw the operator's plan: a sequence of power levels held for a few minutes and joined by
   ramps, plus a speed that sits above or below the propeller-law speed for the same power
   (heavy weather, light running) and a small fluctuation.
3. Ask the steady-state plant what the engine would settle at for each second's inputs.
4. Make the real engine slower than that: each output follows its steady-state value through a
   first-order lag (quasi-steady model plus lags, an assumption, not a transient physics model).
5. Pass the true values through the sensor model to get what a data logger would record.

Random numbers: one generator per episode, seeded from (base seed, split id, episode index), so
an episode does not depend on how many other episodes were generated or in what order.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import gaussian_filter1d

from vsensor.data.config import DatasetConfig, Split
from vsensor.params import Registry
from vsensor.sim.fuels import FUEL_SPECIES
from vsensor.sim.plant import Plant
from vsensor.sim.sensors import FAULT_CODE, FaultSpec, SensorSpecs, first_order_lag, measure


def episode_rng(base_seed: int, seed_id: int, index: int) -> np.random.Generator:
    return np.random.Generator(np.random.PCG64(np.random.SeedSequence([base_seed, seed_id, index])))


def build_profile(rng: np.random.Generator, cfg: DatasetConfig, band: tuple[float, float]):
    """Load (fraction of rated power) and speed factor versus time, both inside their ranges."""
    n = cfg.episode_duration_s
    pr = cfg.profile
    knots_t, knots_l, knots_s = [0.0], [rng.uniform(*band)], [rng.uniform(*pr.speed_factor_range)]
    time = 0.0
    while True:
        time += rng.uniform(*pr.hold_s)
        knots_t.append(time); knots_l.append(knots_l[-1]); knots_s.append(knots_s[-1])
        if time >= n:
            break
        time += rng.uniform(*pr.ramp_s)
        knots_t.append(time); knots_l.append(rng.uniform(*band)); knots_s.append(rng.uniform(*pr.speed_factor_range))
        if time >= n:
            break
    t = np.arange(n, dtype=float)
    load = np.interp(t, knots_t, knots_l)
    sf = np.interp(t, knots_t, knots_s)
    load = gaussian_filter1d(load, pr.smoothing_sigma_s, mode="nearest")
    sf = gaussian_filter1d(sf, pr.smoothing_sigma_s, mode="nearest")
    # operator and sea-state fluctuation: AR(1) noise on power
    phi = np.exp(-1.0 / pr.wiggle_tau_s)
    w = np.empty(n)
    w[0] = rng.normal(0.0, pr.wiggle_sd)
    eps = rng.normal(0.0, pr.wiggle_sd * np.sqrt(1.0 - phi**2), n)
    for k in range(1, n):
        w[k] = phi * w[k - 1] + eps[k]
    load = np.clip(load + w, band[0], band[1])
    sf = np.clip(sf, *pr.speed_factor_range)
    return load, sf


@dataclass
class Episode:
    episode_id: str
    split: str
    meta: dict
    signals: dict[str, np.ndarray]      # what a data logger records (model-visible, incl. measured target)
    truth: dict[str, np.ndarray]        # true values and hidden states (never given to a model)
    events: list[dict]


def simulate_episode(
    split_name: str, split: Split, index: int, cfg: DatasetConfig, reg: Registry, plant: Plant, sp: SensorSpecs
) -> Episode:
    rng = episode_rng(cfg.base_seed, split.seed_id, index)
    n = cfg.episode_duration_s
    t = np.arange(n, dtype=float)
    eid = f"{split_name}-{index:04d}"

    # 1. fixed properties ------------------------------------------------------------------
    fuels = list(split.fuel_mix)
    fuel = fuels[int(np.searchsorted(np.cumsum([split.fuel_mix[f] for f in fuels]), rng.random(), side="right").clip(0, len(fuels) - 1))]
    band = split.load_band[fuel]
    t_amb0 = rng.uniform(*split.t_amb_k)
    p_amb0 = float(np.clip(rng.normal(reg["amb.p_mean"], reg["amb.p_sd"]), reg["amb.p_mean"] - 3 * reg["amb.p_sd"], reg["amb.p_mean"] + 3 * reg["amb.p_sd"]))
    t_amp, t_per, t_ph = rng.uniform(0.0, reg["amb.t_drift_amp_max"]), rng.uniform(1500.0, 3600.0), rng.uniform(0.0, 2 * np.pi)
    p_amp, p_per, p_ph = rng.uniform(0.0, reg["amb.p_drift_amp_max"]), rng.uniform(1500.0, 3600.0), rng.uniform(0.0, 2 * np.pi)
    ftc0 = reg["health.tc.max"] * rng.beta(reg["health.tc.beta_a"], reg["health.tc.beta_b"])
    fcl0 = reg["health.cool.max"] * rng.beta(reg["health.cool.beta_a"], reg["health.cool.beta_b"])
    ftc_rate = abs(rng.normal(0.0, reg["health.drift_per_h_sd"]))
    fcl_rate = abs(rng.normal(0.0, reg["health.drift_per_h_sd"]))

    # 2. operator plan ---------------------------------------------------------------------
    load, sf = build_profile(rng, cfg, band)
    t_amb = t_amb0 + t_amp * np.sin(2 * np.pi * t / t_per + t_ph)
    p_amb = p_amb0 + p_amp * np.sin(2 * np.pi * t / p_per + p_ph)
    foul_tc = np.clip(ftc0 + ftc_rate * t / 3600.0, 0.0, 1.0)
    foul_cool = np.clip(fcl0 + fcl_rate * t / 3600.0, 0.0, 1.0)
    n_rpm = np.minimum(sf * plant.n_rated * load ** (1.0 / 3.0), plant.n_rated)
    m_fuel = plant.fuel_flow_for_power(load * plant.p_rated, n_rpm, fuel)

    # 3. steady state for each second -----------------------------------------------------------
    ss = plant.solve(n_rpm, m_fuel, t_amb=t_amb, p_amb=p_amb, foul_tc=foul_tc, foul_cool=foul_cool, fuel=fuel)
    if not np.all(ss.converged):
        raise RuntimeError(f"plant did not converge in {eid}")

    # 4. real dynamics: first-order lags ------------------------------------------------------
    true = {
        "n_rpm": n_rpm,
        "m_fuel": m_fuel,
        "p_scav": first_order_lag(ss.p_scav, reg["dyn.tau_p_scav"]),
        "t_scav": first_order_lag(ss.t_scav, reg["dyn.tau_t_scav"]),
        "n_tc": first_order_lag(ss.n_tc, reg["dyn.tau_n_tc"]),
        "t_amb": t_amb,
        "p_amb": p_amb,
        "t_exh": first_order_lag(ss.t_exh, reg["dyn.tau_t_exh"]),
    }

    # 5. sensors and faults -----------------------------------------------------------------------
    kind = "none"
    if split.fault_cycle:
        kind = split.fault_cycle[index % len(split.fault_cycle)]
    fault = FaultSpec()
    if kind != "none":
        fi = cfg.fault_injection
        start = float(np.floor(rng.uniform(*fi.start_s)))
        sign = float(rng.choice([-1.0, 1.0]))
        mag = {"bias": rng.uniform(*fi.bias_k), "drift": rng.uniform(*fi.drift_k_per_min), "dropout": rng.uniform(*fi.dropout_s)}[kind]
        fault = FaultSpec(kind, start, float(mag), sign)
    m = measure(true, sp, rng, fault)

    fuel_code = FUEL_SPECIES[fuel][1]
    signals = {
        "t_s": t.astype(np.int32),
        "fuel_code": np.full(n, fuel_code, dtype=np.int8),
        "n_rpm": m.signals["n_rpm"],
        "m_fuel": m.signals["m_fuel"],
        "p_scav": m.signals["p_scav"],
        "t_scav": m.signals["t_scav"],
        "n_tc": m.signals["n_tc"],
        "t_amb": m.signals["t_amb"],
        "p_amb": m.signals["p_amb"],
        "t_exh_meas": m.signals["t_exh"],
    }
    truth = {
        "t_s": t.astype(np.int32),
        "t_exh_true": true["t_exh"],
        "t_exh_steady": ss.t_exh,
        "p_scav_true": true["p_scav"] / 1e5,
        "t_scav_true": true["t_scav"],
        "n_tc_true": true["n_tc"],
        "n_rpm_true": n_rpm,
        "m_fuel_true": m_fuel,
        "t_amb_true": t_amb,
        "p_amb_true": p_amb / 1e5,
        "load_fraction": load,
        "speed_factor": sf,
        "foul_tc": foul_tc,
        "foul_cool": foul_cool,
        "p_exh_steady": ss.p_exh / 1e5,
        "m_air_steady": ss.m_air,
        "delivery_ratio_steady": ss.delivery_ratio,
        "fault_code": m.fault_kind_code,
        "fault_active": m.fault_active,
        "missing_t_exh": m.texh_missing,
        "missing_n_tc": m.ntc_missing,
    }
    events = [{"kind": "dropout_n_tc", "start_s": a, "end_s": b, "magnitude": float(b - a)} for a, b in m.ntc_dropouts]
    events += m.fault_events
    meta = {
        "episode_id": eid, "split": split_name, "index": index, "fuel": fuel, "fuel_code": fuel_code,
        "load_lo": band[0], "load_hi": band[1], "t_amb0": t_amb0, "p_amb0": p_amb0,
        "foul_tc0": ftc0, "foul_cool0": fcl0, "foul_tc_rate_per_h": ftc_rate, "foul_cool_rate_per_h": fcl_rate,
        "fault_kind": kind, "fault_start_s": fault.start_s, "fault_magnitude": fault.magnitude * fault.sign if kind in ("bias", "drift") else fault.magnitude,
        "bias_m_fuel_rel": m.biases["m_fuel"], "bias_t_amb_k": m.biases["t_amb"], "bias_p_amb_bar": m.biases["p_amb"],
        "bias_t_exh_k": m.biases["t_exh"], "drift_pscav_bar_per_h": m.drift_rate_pscav_bar_per_h,
        "eta_scale_methanol": split.eta_scale_methanol,
    }
    return Episode(eid, split_name, meta, signals, truth, events)
