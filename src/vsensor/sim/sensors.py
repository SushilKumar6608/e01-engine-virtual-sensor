"""Sensor model: how a true physical value becomes a measured value (SIMULATED DATA).

Plain-language summary
----------------------
The plant knows the true pressures and temperatures. Real instruments never report the truth:
they respond slowly (lag), carry a small constant offset (bias), wander slowly (drift), jitter
(noise), report in steps (quantisation) and sometimes lose the signal (dropout). This module
adds those imperfections, signal by signal, following design document section 6.3.

All sizes come from the parameter registry and are *assumptions*: typical orders of magnitude
for industrial instruments, with no datasheet behind them. The model therefore says nothing
about the noise of any real engine's instruments.

Faults are different from the normal imperfections. A fault is an injected event on the
exhaust temperature sensor (a bias step, a drifting ramp or a dropout) whose start, kind and
size are known exactly, so fault detection can be scored.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import lfilter, lfilter_zi

from vsensor.params import Registry

FAULT_CODE = {"none": 0, "bias": 1, "drift": 2, "dropout": 3}


def first_order_lag(x: np.ndarray, tau: float, dt: float = 1.0) -> np.ndarray:
    """First-order lag y' = (x - y) / tau with exact discretisation; starts settled at x[0]."""
    x = np.asarray(x, dtype=float)
    a = 1.0 - np.exp(-dt / tau)
    b, den = [a], [1.0, -(1.0 - a)]
    y, _ = lfilter(b, den, x, zi=lfilter_zi(b, den) * x[0])
    return y


def quantise(x: np.ndarray, step: float) -> np.ndarray:
    return np.round(np.asarray(x, dtype=float) / step) * step


@dataclass(frozen=True)
class SensorSpecs:
    speed_noise: float
    speed_res: float
    fuel_noise_rel: float
    fuel_bias_sd_rel: float
    fuel_res: float
    pscav_noise: float
    pscav_drift_sd_per_h: float
    pscav_res: float
    tscav_noise: float
    tscav_tau: float
    tscav_res: float
    ntc_noise_rel: float
    ntc_res: float
    ntc_dropout_rate_per_h: float
    ntc_dropout_len_min: float
    ntc_dropout_len_max: float
    tamb_noise: float
    tamb_bias_sd: float
    tamb_res: float
    pamb_noise: float
    pamb_bias_sd: float
    pamb_res: float
    texh_noise: float
    texh_bias_sd: float
    texh_tau: float
    texh_res: float

    @classmethod
    def from_registry(cls, r: Registry) -> "SensorSpecs":
        return cls(
            speed_noise=r["sens.speed.noise"], speed_res=r["sens.speed.res"],
            fuel_noise_rel=r["sens.fuel.noise_rel"], fuel_bias_sd_rel=r["sens.fuel.bias_sd_rel"], fuel_res=r["sens.fuel.res"],
            pscav_noise=r["sens.pscav.noise"], pscav_drift_sd_per_h=r["sens.pscav.drift_sd_per_h"], pscav_res=r["sens.pscav.res"],
            tscav_noise=r["sens.tscav.noise"], tscav_tau=r["sens.tscav.tau"], tscav_res=r["sens.tscav.res"],
            ntc_noise_rel=r["sens.ntc.noise_rel"], ntc_res=r["sens.ntc.res"],
            ntc_dropout_rate_per_h=r["sens.ntc.dropout_rate_per_h"],
            ntc_dropout_len_min=r["sens.ntc.dropout_len_min"], ntc_dropout_len_max=r["sens.ntc.dropout_len_max"],
            tamb_noise=r["sens.tamb.noise"], tamb_bias_sd=r["sens.tamb.bias_sd"], tamb_res=r["sens.tamb.res"],
            pamb_noise=r["sens.pamb.noise"], pamb_bias_sd=r["sens.pamb.bias_sd"], pamb_res=r["sens.pamb.res"],
            texh_noise=r["sens.texh.noise"], texh_bias_sd=r["sens.texh.bias_sd"], texh_tau=r["sens.texh.tau"], texh_res=r["sens.texh.res"],
        )


@dataclass(frozen=True)
class FaultSpec:
    kind: str = "none"          # none | bias | drift | dropout
    start_s: float = 0.0
    magnitude: float = 0.0      # K for bias, K per minute for drift, seconds for dropout
    sign: float = 1.0


@dataclass
class Measured:
    """Measured signals plus the exact record of what was done to them."""
    signals: dict[str, np.ndarray]
    biases: dict[str, float]
    drift_rate_pscav_bar_per_h: float
    ntc_dropouts: list[tuple[int, int]]
    fault_kind_code: np.ndarray       # per row, FAULT_CODE of an active fault else 0
    fault_active: np.ndarray          # per row bool
    fault_events: list[dict]
    texh_missing: np.ndarray          # per row bool (injected dropout)
    ntc_missing: np.ndarray           # per row bool


def measure(true: dict[str, np.ndarray], sp: SensorSpecs, rng: np.random.Generator, fault: FaultSpec) -> Measured:
    """Apply the sensor model to true signals (1 Hz). ``true`` keys:
    n_rpm, m_fuel (kg/s), p_scav (Pa), t_scav (K), n_tc (rpm), t_amb (K), p_amb (Pa), t_exh (K).
    The order of random draws is fixed, so a seed always gives the same measurements."""
    n = len(true["n_rpm"])
    t = np.arange(n, dtype=float)
    out: dict[str, np.ndarray] = {}
    biases: dict[str, float] = {}

    out["n_rpm"] = quantise(true["n_rpm"] + rng.normal(0.0, sp.speed_noise, n), sp.speed_res)

    biases["m_fuel"] = float(rng.normal(0.0, sp.fuel_bias_sd_rel))
    out["m_fuel"] = quantise(true["m_fuel"] * (1.0 + biases["m_fuel"] + rng.normal(0.0, sp.fuel_noise_rel, n)), sp.fuel_res)

    drift_rate = float(rng.normal(0.0, sp.pscav_drift_sd_per_h))
    out["p_scav"] = quantise(true["p_scav"] / 1e5 + drift_rate * t / 3600.0 + rng.normal(0.0, sp.pscav_noise, n), sp.pscav_res)

    out["t_scav"] = quantise(first_order_lag(true["t_scav"], sp.tscav_tau) + rng.normal(0.0, sp.tscav_noise, n), sp.tscav_res)

    ntc = quantise(true["n_tc"] * (1.0 + rng.normal(0.0, sp.ntc_noise_rel, n)), sp.ntc_res)
    ntc_missing = np.zeros(n, dtype=bool)
    ntc_events: list[tuple[int, int]] = []
    for _ in range(int(rng.poisson(sp.ntc_dropout_rate_per_h * n / 3600.0))):
        a = int(rng.integers(0, n))
        length = int(round(rng.uniform(sp.ntc_dropout_len_min, sp.ntc_dropout_len_max)))
        b = min(n, a + max(length, 1))
        ntc_missing[a:b] = True
        ntc_events.append((a, b))
    ntc = ntc.copy()
    ntc[ntc_missing] = np.nan
    out["n_tc"] = ntc

    biases["t_amb"] = float(rng.normal(0.0, sp.tamb_bias_sd))
    out["t_amb"] = quantise(true["t_amb"] + biases["t_amb"] + rng.normal(0.0, sp.tamb_noise, n), sp.tamb_res)
    biases["p_amb"] = float(rng.normal(0.0, sp.pamb_bias_sd))
    out["p_amb"] = quantise(true["p_amb"] / 1e5 + biases["p_amb"] + rng.normal(0.0, sp.pamb_noise, n), sp.pamb_res)

    biases["t_exh"] = float(rng.normal(0.0, sp.texh_bias_sd))
    texh = first_order_lag(true["t_exh"], sp.texh_tau) + biases["t_exh"] + rng.normal(0.0, sp.texh_noise, n)

    code = np.zeros(n, dtype=np.int8)
    active = np.zeros(n, dtype=bool)
    texh_missing = np.zeros(n, dtype=bool)
    events: list[dict] = []
    if fault.kind != "none":
        s = int(fault.start_s)
        if fault.kind == "bias":
            texh = texh.copy()
            texh[s:] += fault.sign * fault.magnitude
            end, mag = n, fault.sign * fault.magnitude
        elif fault.kind == "drift":
            texh = texh.copy()
            texh[s:] += fault.sign * fault.magnitude * (t[s:] - s) / 60.0
            end, mag = n, fault.sign * fault.magnitude
        elif fault.kind == "dropout":
            end, mag = min(n, s + int(round(fault.magnitude))), fault.magnitude
            texh_missing[s:end] = True
        else:
            raise ValueError(fault.kind)
        code[s:end] = FAULT_CODE[fault.kind]
        active[s:end] = True
        events.append({"kind": f"fault_{fault.kind}", "start_s": s, "end_s": end, "magnitude": mag})
    texh = quantise(texh, sp.texh_res)
    texh[texh_missing] = np.nan
    out["t_exh"] = texh

    return Measured(out, biases, drift_rate, ntc_events, code, active, events, texh_missing, ntc_missing)
