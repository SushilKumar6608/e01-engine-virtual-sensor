"""Calibrated registry values must reproduce their targets and must not be stale (SIMULATION PROJECT)."""
from __future__ import annotations

import numpy as np
import pytest

from vsensor.calibration import (
    ALL_IDS,
    SFOC_LOADS,
    brake_efficiency_on_propeller_law,
    calibrate_all,
    healthy_grid,
    nominal_phi_from_heat_balance,
    rated_state,
    sfoc_target_efficiencies,
)
from vsensor.params import load_registry
from vsensor.sim.nominal import NominalModel
from vsensor.sim.plant import Plant

REG = load_registry()


def test_efficiency_curve_reproduces_published_fuel_consumption():
    plant = Plant(REG)
    eta = brake_efficiency_on_propeller_law(plant)
    assert np.allclose(eta, sfoc_target_efficiencies(REG), atol=2e-6)
    sfoc = 3.6e6 / (eta * REG["calib.sfoc.lhv"]) * 1e3  # g/kWh
    assert np.allclose(sfoc, [165.5, 163.0, 167.0], atol=0.02)
    assert np.allclose(SFOC_LOADS, [0.5, 0.75, 1.0])


def test_efficiency_curve_is_a_sensible_parabola():
    assert REG["engine.eta_ind_curv"] > 0
    assert 0.4 < REG["engine.eta_ind_x_opt"] < 1.3
    assert 0.4 < REG["engine.eta_ind_max"] < 0.6


def test_rated_point_hits_its_targets():
    plant, s = rated_state(REG)
    assert s.converged
    assert abs(float(s.load_fraction) - 1.0) < 1e-9
    assert abs(float(s.t_exh) - REG["calib.target.t_exh_rated"]) < 0.05
    assert abs(float(s.p_scav) / REG["calib.target.p_scav_rated"] - 1.0) < 1e-4
    assert abs(float(s.n_tc) / REG["calib.target.n_tc_rated"] - 1.0) < 1e-4
    assert abs(float(s.dp_cooler) / REG["calib.target.cooler_dp_rated"] - 1.0) < 1e-3
    assert abs(float(s.t_scav - s.t_amb) - REG["calib.target.cooler_dt_rated"]) < 0.01


def test_rated_air_flow_and_excess_air_are_in_a_plausible_range():
    """Order-of-magnitude check only (no source for the 4T50ME-X): about 14 kg/s, excess air ratio below 3."""
    _, s = rated_state(REG)
    assert 10.0 < float(s.m_air) < 18.0
    assert 2.0 < float(s.lambda_total) < 3.2


def test_implied_mean_effective_pressure_is_close_to_published_class_value():
    """Stroke is an assumption [registry]. It implies a rated brake mean effective pressure that should be
    of the same order as the 21.0 bar quoted at L1 for the larger G60ME-C9.5 [S10]."""
    plant = Plant(REG)
    bmep_bar = plant.p_rated / (plant.v_disp * plant.n_rated / 60.0) / 1e5
    assert 18.0 < bmep_bar < 22.0, bmep_bar


def test_nominal_exhaust_share_equals_rated_heat_balance():
    plant = Plant(REG)
    assert abs(REG["nominal.phi_exhaust"] - nominal_phi_from_heat_balance(plant)) < 1e-5


def test_stored_calibrations_are_not_stale():
    """Re-run all three calibrations from the stored registry; the answer must not move."""
    fresh = calibrate_all(REG)
    for pid in ALL_IDS:
        assert fresh[pid] == pytest.approx(REG[pid], rel=2e-4), pid


def test_nominal_fit_error_on_the_healthy_grid_is_recorded():
    """Regression guard on the commissioning baseline's own fit error over the 40-100 % operating
    range (about 14 K RMSE when written). If this drops sharply the baseline got stronger; if it
    jumps, check why."""
    g = healthy_grid(Plant(REG))
    err = NominalModel(REG).predict(g.n_rpm, g.m_fuel, g.p_scav, g.t_scav) - g.t_exh
    rmse = float(np.sqrt(np.mean(err**2)))
    assert 8.0 < rmse < 25.0, rmse
