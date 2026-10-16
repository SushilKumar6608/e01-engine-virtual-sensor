"""Calibration helpers shared by scripts and tests (SIMULATION PROJECT).

Three calibrations exist, all documented in the registry as status "calibrated". They run in
this order because each one uses the result of the previous one:

1. Efficiency curve (plant): three constants are fitted exactly to three published
   fuel-consumption points (50/75/100 % load) of a comparable MAN engine.
2. Rated point (plant): five air-path constants are tuned so that the clean, ISO-ambient,
   diesel-like, rated-power operating point hits five targets.
3. Nominal model: its exhaust share comes from the rated heat balance; its air delivery
   ratio is fitted by least squares to the *healthy* plant over a load/speed grid.

tests/test_calibration.py re-runs all three and fails if the values stored in
config/params.yaml are stale.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares, root

from vsensor.params import Registry
from vsensor.sim.nominal import NominalModel
from vsensor.sim.plant import Plant

YAML_PATH = Path(__file__).resolve().parents[2] / "config" / "params.yaml"

EFFICIENCY_IDS = ["engine.eta_ind_max", "engine.eta_ind_curv", "engine.eta_ind_x_opt"]
RATED_IDS = ["scav.lambda_coeff", "tc.k_turbine", "cooler.k_loss", "cooler.eps", "tc.d_comp"]
NOMINAL_IDS = ["nominal.lambda_del", "nominal.phi_exhaust"]
SFOC_LOADS = np.array([0.50, 0.75, 1.00])


# ------------------------------------------------------------- 0. efficiency curve
def sfoc_target_efficiencies(reg: Registry) -> np.ndarray:
    """Brake efficiencies implied by the published fuel-consumption points.

    eta = 3.6e6 J per kWh / (SFOC [kg per kWh] * LHV [J/kg]).
    """
    sfoc = np.array([reg["calib.sfoc.load_50"], reg["calib.sfoc.load_75"], reg["calib.sfoc.load_100"]])  # g/kWh
    return 3.6e6 / (sfoc * 1e-3 * reg["calib.sfoc.lhv"])


def brake_efficiency_on_propeller_law(plant: Plant, loads=SFOC_LOADS, fuel: str = "diesel_like") -> np.ndarray:
    n, mf = plant.propeller_law(loads, fuel)
    return np.asarray(plant.brake_power(mf, n, fuel)) / (mf * plant.fuels[fuel].lhv_liquid)


def calibrate_efficiency(reg0: Registry) -> Registry:
    target = sfoc_target_efficiencies(reg0)

    def build(x):
        return reg0.with_overrides(engine__eta_ind_max=x[0], engine__eta_ind_curv=x[1], engine__eta_ind_x_opt=x[2])

    def resid(x):
        return brake_efficiency_on_propeller_law(Plant(build(x))) - target

    x0 = np.array([reg0["engine.eta_ind_max"], reg0["engine.eta_ind_curv"], reg0["engine.eta_ind_x_opt"]])
    sol = root(resid, x0, method="hybr", tol=1e-13)
    if not sol.success or np.max(np.abs(resid(sol.x))) > 1e-9:
        raise RuntimeError(f"efficiency calibration failed: {sol.message}")
    if not (sol.x[1] > 0 and 0.4 < sol.x[2] < 1.3):
        raise RuntimeError(f"efficiency calibration gave an unphysical curve: {sol.x}")
    return build(sol.x)



# ---------------------------------------------------------------- 2. rated point
def rated_state(reg: Registry):
    plant = Plant(reg)
    mf = plant.fuel_flow_for_power(plant.p_rated, plant.n_rated)
    return plant, plant.solve(plant.n_rated, mf)


def calibrate_rated(reg0: Registry) -> Registry:
    t_tgt, p_tgt = reg0["calib.target.t_exh_rated"], reg0["calib.target.p_scav_rated"]
    dp_tgt, n_tgt = reg0["calib.target.cooler_dp_rated"], reg0["calib.target.n_tc_rated"]
    dt_tgt = reg0["calib.target.cooler_dt_rated"]

    def build(x):
        return reg0.with_overrides(
            scav__lambda_coeff=x[0], tc__k_turbine=x[1] * 1e-3, cooler__k_loss=x[2] * 10.0, cooler__eps=x[3]
        )

    def resid(x):
        reg = build(x)
        _, s = rated_state(reg)
        if not s.converged:
            return np.ones(4)
        return np.array(
            [
                (float(s.t_exh) - t_tgt) / t_tgt,
                (float(s.p_scav) - p_tgt) / p_tgt,
                (float(s.dp_cooler) - dp_tgt) / dp_tgt,
                (float(s.t_scav) - float(s.t_amb) - reg["cooler.t_coolant_offset"] - dt_tgt) / dt_tgt,
            ]
        )

    x0 = np.array([reg0["scav.lambda_coeff"], reg0["tc.k_turbine"] * 1e3, reg0["cooler.k_loss"] / 10.0, 0.92])
    sol = root(resid, x0, method="hybr", tol=1e-12)
    if not sol.success or np.max(np.abs(resid(sol.x))) > 1e-8:
        raise RuntimeError(f"rated calibration failed: {sol.message}")
    reg1 = build(sol.x)
    # Speed is inversely proportional to wheel diameter at fixed specific work.
    _, s = rated_state(reg1)
    return reg1.with_overrides(tc__d_comp=reg1["tc.d_comp"] * float(s.n_tc) / n_tgt)


# ------------------------------------------------------------- 3. nominal model
def healthy_grid(plant: Plant, fuel: str = "diesel_like"):
    """Commissioning-style grid: load 30-100 % at three speed offsets from the propeller law.

    Speed is capped at rated speed. Returns the converged healthy ISO plant solution.
    """
    loads = np.linspace(0.30, 1.0, 15)
    pts_n, pts_p = [], []
    for s in (0.93, 1.0, 1.07):
        n_prop = plant.n_rated * loads ** (1.0 / 3.0)
        pts_n.append(np.minimum(s * n_prop, plant.n_rated))
        pts_p.append(loads * plant.p_rated)
    n = np.concatenate(pts_n)
    p = np.concatenate(pts_p)
    mf = plant.fuel_flow_for_power(p, n, fuel)
    sol = plant.solve(n, mf, fuel=fuel)
    if not sol.converged:
        raise RuntimeError("healthy grid did not converge")
    return sol


def nominal_phi_from_heat_balance(plant: Plant) -> float:
    """Share of fuel energy leaving with the exhaust, from the rated-point heat balance.

    A shop test gives fuel energy, brake power and the heat taken by cooling water and oil
    at the rated point. Here the same numbers are taken from the plant:
    phi = 1 - indicated efficiency at the rated fuel-energy rate - wall-loss fraction.
    """
    return float(1.0 - plant.eta_indicated(1.0) - plant.f_wall)


def calibrate_nominal(reg: Registry) -> Registry:
    """Exhaust share from the rated heat balance (closed form); air delivery ratio by least squares.

    Fitting both by least squares was tried first and diverged: more air and a larger
    exhaust share give almost the same temperature, so the pair is not identifiable.
    """
    plant = Plant(reg)
    phi = nominal_phi_from_heat_balance(plant)
    g = healthy_grid(plant)

    def resid(x):
        nm = NominalModel(reg.with_overrides(nominal__lambda_del=x[0], nominal__phi_exhaust=phi))
        return nm.predict(g.n_rpm, g.m_fuel, g.p_scav, g.t_scav) - g.t_exh

    sol = least_squares(resid, x0=[0.95], bounds=([0.3], [1.5]), xtol=1e-14, ftol=1e-14)
    return reg.with_overrides(nominal__lambda_del=sol.x[0], nominal__phi_exhaust=phi)


# ------------------------------------------------------------------- yaml writer
def write_yaml_values(values: dict[str, float], path: Path = YAML_PATH) -> None:
    """Replace the ``value:`` line of the named parameters in the registry file."""
    text = path.read_text()
    for pid, val in values.items():
        pattern = re.compile(rf"(^  {re.escape(pid)}:\n(?:    .*\n)*?    value: )([^\n]+)", re.M)
        if not pattern.search(text):
            raise KeyError(pid)
        text = pattern.sub(rf"\g<1>{val:.6g}", text, count=1)
    path.write_text(text)


def calibrate_all(reg0: Registry) -> Registry:
    """Run the three calibrations in dependency order."""
    reg = calibrate_efficiency(reg0)
    reg = calibrate_rated(reg)
    return calibrate_nominal(reg)


ALL_IDS = EFFICIENCY_IDS + RATED_IDS + NOMINAL_IDS
