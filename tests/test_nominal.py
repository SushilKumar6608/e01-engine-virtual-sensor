"""Nominal model: honest inputs only, correct heat balance, not omniscient (SIMULATION PROJECT)."""
from __future__ import annotations

import ast
import inspect
from pathlib import Path

import numpy as np
import pytest

from vsensor.params import load_registry
from vsensor.sim import nominal as nominal_module
from vsensor.sim.fuels import FUEL_SPECIES
from vsensor.sim.nominal import CODE_TO_FUEL, NominalModel
from vsensor.sim.plant import Plant

REG = load_registry()
PLANT = Plant(REG)
NOMINAL = NominalModel(REG)

HIDDEN_NAMES = {
    "foul_tc", "foul_cool", "p_exh", "t_exh", "t_turb_out", "delivery_ratio", "lambda_total",
    "Plant", "SteadyState", "hidden", "signals",
}


def test_predict_signature_is_exactly_the_measured_inputs():
    params = list(inspect.signature(NominalModel.predict).parameters)
    assert params == ["self", "n_rpm", "m_fuel", "p_scav", "t_scav", "fuel"]


def test_nominal_code_never_touches_hidden_plant_state():
    """Gate 7 (leakage), structural part: look at the identifiers in the code, not the docstrings."""
    tree = ast.parse(Path(nominal_module.__file__).read_text())
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            names.add(node.id)
        elif isinstance(node, ast.Attribute):
            names.add(node.attr)
        elif isinstance(node, ast.arg):
            names.add(node.arg)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            mod = getattr(node, "module", "") or ""
            assert "plant" not in mod, f"nominal model must not import the plant: {mod}"
            names |= {a.name for a in node.names}
    assert not names & HIDDEN_NAMES, names & HIDDEN_NAMES


def test_heat_balance_is_implemented_correctly():
    """If the nominal model is given the plant's own air flow and exhaust share at one operating
    point, it must reproduce the plant's exhaust temperature. The only remaining difference is the
    air enthalpy (constant specific heat in the plant, gas tables in the nominal model)."""
    for fuel in FUEL_SPECIES:
        for load in (0.4, 0.7, 1.0):
            n, mf = PLANT.propeller_law(load, fuel)
            s = PLANT.solve(n, mf, fuel=fuel, foul_tc=0.3, t_amb=285.0)
            rho = float(s.p_scav / (PLANT.r_air * s.t_scav))
            lam_true = float(s.m_air) / (rho * PLANT.v_disp * float(s.n_rpm) / 60.0)
            phi_true = float(PLANT.heat_to_exhaust(s.m_fuel, fuel) / s.q_fuel)
            nm = NominalModel(REG.with_overrides(nominal__lambda_del=lam_true, nominal__phi_exhaust=phi_true))
            est = float(nm.predict(s.n_rpm, s.m_fuel, s.p_scav, s.t_scav, fuel))
            assert abs(est - float(s.t_exh)) < 0.5, (fuel, load, est, float(s.t_exh))


def test_by_code_matches_per_fuel_calls():
    n, mf = PLANT.propeller_law(np.array([0.5, 0.8]))
    s = PLANT.solve(n, mf)
    for code, fid in CODE_TO_FUEL.items():
        a = NOMINAL.predict(s.n_rpm, s.m_fuel, s.p_scav, s.t_scav, fid)
        b = NOMINAL.predict_by_code(s.n_rpm, s.m_fuel, s.p_scav, s.t_scav, np.full(2, code))
        assert np.allclose(a, b)
    mixed = NOMINAL.predict_by_code(s.n_rpm, s.m_fuel, s.p_scav, s.t_scav, np.array([0, 1]))
    assert mixed[0] == pytest.approx(float(NOMINAL.predict(s.n_rpm[0], s.m_fuel[0], s.p_scav[0], s.t_scav[0], "diesel_like")))


def test_nominal_is_not_omniscient_it_degrades_with_hidden_fouling():
    """The baseline must have something left to learn: hidden health states that its inputs only
    partly reveal."""
    loads = np.linspace(0.5, 1.0, 11)
    n, mf = PLANT.propeller_law(loads)

    def mean_abs_error(**kw):
        s = PLANT.solve(n, mf, **kw)
        return float(np.mean(np.abs(NOMINAL.predict(s.n_rpm, s.m_fuel, s.p_scav, s.t_scav) - s.t_exh)))

    healthy = mean_abs_error()
    assert mean_abs_error(foul_tc=1.0) > healthy + 5.0
    assert mean_abs_error(foul_cool=1.0) > healthy + 5.0


def test_characterisation_ambient_shift_is_mostly_absorbed_by_the_nominal_model():
    """RECORDED MODEL BEHAVIOUR with a consequence for the evaluation plan.

    Ambient temperature acts on the exhaust almost entirely through air density at the engine inlet,
    and the nominal model sees that through the measured scavenge pressure and temperature. So a
    cold-ambient test set is NOT a hard regime shift for the physics baseline or for a residual model
    built on it (the design document lists cold ambient as a held-out regime in section 9).
    """
    loads = np.linspace(0.5, 1.0, 11)
    n, mf = PLANT.propeller_law(loads)

    def mean_abs_error(**kw):
        s = PLANT.solve(n, mf, **kw)
        return float(np.mean(np.abs(NOMINAL.predict(s.n_rpm, s.m_fuel, s.p_scav, s.t_scav) - s.t_exh)))

    healthy = mean_abs_error()
    for t_amb in (268.15, 278.15, 308.15, 318.15):
        assert abs(mean_abs_error(t_amb=t_amb) - healthy) < 10.0, t_amb


def test_nominal_is_reasonable_on_the_healthy_engine_above_half_load():
    """Sanity, not a performance claim: within 25 K of the plant from 50 % load on a healthy engine."""
    loads = np.linspace(0.5, 1.0, 11)
    n, mf = PLANT.propeller_law(loads)
    s = PLANT.solve(n, mf)
    err = NOMINAL.predict(s.n_rpm, s.m_fuel, s.p_scav, s.t_scav) - s.t_exh
    assert np.max(np.abs(err)) < 25.0, err
