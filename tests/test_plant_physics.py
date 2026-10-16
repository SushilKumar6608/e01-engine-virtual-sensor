"""Plant simulator: convergence, closure, direction of effects (SIMULATION PROJECT).

Design-document quality gates previewed here: 3 (energy and mass closure) and 4 (direction
of effects). Gate: temperature band, scavenge pressure rising with load.
"""
from __future__ import annotations

import numpy as np
import pytest

from vsensor.sim import thermo
from vsensor.sim.fuels import FUEL_SPECIES
from vsensor.sim.plant import Plant, SteadyState

PLANT = Plant()
LOADS = np.linspace(0.30, 1.0, 15)
FUELS = list(FUEL_SPECIES)


def _propeller(fuel, **kw):
    n, mf = PLANT.propeller_law(LOADS, fuel)
    return PLANT.solve(n, mf, fuel=fuel, **kw)


# ---------------------------------------------------------------- convergence -------------
@pytest.mark.parametrize("fuel", FUELS)
def test_converges_over_whole_envelope_including_extremes(fuel):
    n, mf = PLANT.propeller_law(LOADS, fuel)
    for kw in (
        {},
        dict(foul_tc=1.0, foul_cool=1.0),
        dict(t_amb=268.15, p_amb=0.97e5),
        dict(t_amb=313.15, p_amb=1.04e5, foul_tc=0.6),
    ):
        s = PLANT.solve(n, mf, fuel=fuel, **kw)
        assert s.converged, (fuel, kw, s.max_residual)
        assert s.max_residual < 1e-9
        assert np.all(np.isfinite(s.t_exh))


def test_deterministic():
    a, b = _propeller("diesel_like"), _propeller("diesel_like")
    assert np.array_equal(a.t_exh, b.t_exh) and np.array_equal(a.p_scav, b.p_scav)


def test_scalar_and_array_calls_agree():
    n, mf = PLANT.propeller_law(0.62)
    s1 = PLANT.solve(n, mf)
    s2 = PLANT.solve(np.array([n, n]), np.array([mf, mf]))
    assert np.isclose(float(s1.t_exh), s2.t_exh[0], rtol=1e-12)
    assert np.shape(s1.t_exh) == ()


def test_fouling_outside_zero_one_is_refused():
    with pytest.raises(ValueError):
        PLANT.solve(100.0, 0.2, foul_tc=1.2)


# ----------------------------------------------------------- operating point helpers -------
def test_rated_point_gives_rated_power_and_speed():
    n, mf = PLANT.propeller_law(1.0)
    assert np.isclose(n, 123.0)
    assert np.isclose(float(PLANT.brake_power(mf, n)), 7.08e6, rtol=1e-9)


@pytest.mark.parametrize("fuel", FUELS)
def test_fuel_flow_inversion_round_trip(fuel):
    p = np.linspace(0.2, 1.0, 9) * PLANT.p_rated
    n = np.linspace(0.8, 1.0, 9) * PLANT.n_rated
    mf = PLANT.fuel_flow_for_power(p, n, fuel)
    assert np.allclose(PLANT.brake_power(mf, n, fuel), p, rtol=1e-9)


def test_solved_state_reports_the_requested_power():
    s = _propeller("diesel_like")
    assert np.allclose(s.load_fraction, LOADS, rtol=1e-9)


# ------------------------------------------------------------------- closure (gate 3) ------
@pytest.mark.parametrize("fuel", FUELS)
@pytest.mark.parametrize("kw", [{}, dict(foul_tc=0.7, foul_cool=0.5, t_amb=283.15, p_amb=1.01e5)])
def test_whole_engine_energy_balance_closes(fuel, kw):
    """Fuel energy + incoming air = brake power + friction heat + wall heat + cooler heat
    + enthalpy of gas leaving the turbine + turbocharger bearing loss.

    Computed independently from the solver, from its output fields and the thermo table."""
    s = _propeller(fuel, **kw)
    fu = PLANT.fuels[fuel]
    for i in range(len(LOADS)):
        moles = fu.exhaust_moles(s.m_fuel[i], s.m_air[i], PLANT.air)
        h_in = float(thermo.mixture_sensible_enthalpy_rate(PLANT.air.moles(s.m_air[i]), s.t_amb[i]))
        h_out = float(thermo.mixture_sensible_enthalpy_rate(moles, s.t_turb_out[i]))
        p_fric = float(PLANT.friction_power(s.n_rpm[i]))
        lhs = s.q_fuel[i] + h_in
        rhs = s.p_brake[i] + p_fric + s.q_wall[i] + s.q_cooler[i] + h_out + (1 - PLANT.eta_m) * s.p_turbine[i]
        assert abs(lhs - rhs) / s.q_fuel[i] < 2e-3, (i, (lhs - rhs) / s.q_fuel[i])


@pytest.mark.parametrize("fuel", FUELS)
def test_mass_closes(fuel):
    s = _propeller(fuel)
    fu = PLANT.fuels[fuel]
    for i in range(len(LOADS)):
        moles = fu.exhaust_moles(s.m_fuel[i], s.m_air[i], PLANT.air)
        m_out = sum(float(n) * thermo.molar_mass(sp) for sp, n in moles.items())
        assert abs(m_out - (s.m_air[i] + s.m_fuel[i])) < 1e-9 * m_out


@pytest.mark.parametrize("fuel", FUELS)
def test_converged_state_satisfies_its_own_equations(fuel):
    """Recompute each relation from the reported outputs (independent of the iteration)."""
    s = _propeller(fuel, foul_tc=0.4, foul_cool=0.3)
    # engine air flow
    rho = s.p_scav / (PLANT.r_air * s.t_scav)
    pi = s.p_exh / s.p_scav
    m_air = PLANT.lam_coeff * np.sqrt(1 - pi) * rho * PLANT.v_disp * s.n_rpm / 60
    assert np.allclose(m_air, s.m_air, rtol=1e-7)
    # turbine flow law
    k = PLANT.k_turb0 * (1 - PLANT.foul_k * s.foul_tc)
    p_back = PLANT.p_back_ratio * s.p_amb
    assert np.allclose(s.p_exh**2 - p_back**2, ((s.m_air + s.m_fuel) * np.sqrt(s.t_exh) / k) ** 2, rtol=1e-9)
    # compressor: pressure ratio, tip-speed relation and cooler loss
    assert np.allclose((s.p_scav + s.dp_cooler) / s.p_amb, s.pr_comp, rtol=1e-7)
    eta_c = PLANT.eta_c0 * (1 - PLANT.foul_eta_c * s.foul_tc)
    p_comp = s.m_air * PLANT.cp_air * s.t_amb * (s.pr_comp ** ((PLANT.kappa_air - 1) / PLANT.kappa_air) - 1) / eta_c
    assert np.allclose(p_comp, PLANT.eta_m * s.p_turbine, rtol=1e-9)  # turbocharger power balance
    u = s.n_tc * 2 * np.pi / 60 * PLANT.d_comp / 2
    assert np.allclose(PLANT.psi * u**2 * s.m_air, PLANT.eta_m * s.p_turbine, rtol=1e-9)
    # cooler outlet temperature
    eps = PLANT.eps0 * (1 - PLANT.foul_eps * s.foul_cool)
    assert np.allclose(s.t_scav, s.t_comp_out - eps * (s.t_comp_out - (s.t_amb + PLANT.t_cw_offset)), rtol=1e-9)


# ------------------------------------------------------------- plausible band (gate) --
@pytest.mark.parametrize("fuel", FUELS)
def test_scavenge_pressure_rises_with_load_on_propeller_law(fuel):
    s = _propeller(fuel)
    assert np.all(np.diff(s.p_scav) > 0)
    assert np.all(np.diff(s.n_tc) > 0)
    assert np.all(np.diff(s.m_air) > 0)


def test_scavenging_possible_everywhere():
    """Exhaust receiver below scavenge pressure, or air would not flow through the cylinders."""
    for fuel in FUELS:
        s = _propeller(fuel, foul_tc=1.0)
        assert np.all(s.p_exh < s.p_scav)
        assert np.all(s.lambda_total > 1.5)


def test_exhaust_temperature_inside_published_band_from_40_percent_load():
    """600-800 K is the axis range of the exhaust-temperature plot in Alegret et al. 2015 [S01].
    That range is INDICATIVE (read off a plot axis, one dataset, load not stated numerically).
    Below 40 % load there is no source coverage; see the simulator report for 30-40 %."""
    s = _propeller("diesel_like")
    mask = LOADS >= 0.40 - 1e-12
    assert np.all((s.t_exh[mask] >= 600.0) & (s.t_exh[mask] <= 800.0)), s.t_exh[mask]


def test_low_load_values_are_finite_and_documented_as_unverified():
    s = _propeller("diesel_like")
    assert np.all(np.isfinite(s.t_exh)) and np.all(s.t_exh < 1000.0)


# --------------------------------------------------------- direction of effects (gate 4) ---
def _base(fuel="diesel_like", load=0.6):
    n, mf = PLANT.propeller_law(load, fuel)
    return n, mf


@pytest.mark.parametrize("fuel,mf_range", [("diesel_like", (0.12, 0.34)), ("methanol", (0.27, 0.76))])
def test_at_fixed_air_supply_more_fuel_raises_exhaust_temperature(fuel, mf_range):
    """Design gate 4, first half: with the air flow held fixed, burning more fuel makes the exhaust hotter."""
    mf = np.linspace(*mf_range, 20)
    t = PLANT.exhaust_temperature_fixed_air(mf, m_air=11.0, t_scav=310.0, fuel=fuel)
    assert np.all(np.diff(t) > 0)


def test_at_fixed_fuel_more_air_lowers_exhaust_temperature():
    m_air = np.linspace(6.0, 16.0, 20)
    t = PLANT.exhaust_temperature_fixed_air(0.25, m_air, 310.0)
    assert np.all(np.diff(t) < 0)


def test_characterisation_matched_exhaust_temperature_is_nearly_flat_in_fuel_at_fixed_speed():
    """RECORDED MODEL BEHAVIOUR, not a physics requirement and not verified against a source.

    With the turbocharger matched, more fuel brings more boost and more air, so the air-excess
    ratio stays near 2.8-3.0 and the exhaust temperature moves only a few percent while fuel
    doubles. A real engine's curve may be steeper; this is reported as an open limitation.
    """
    mf = np.linspace(0.15, 0.30, 12)
    s = PLANT.solve(110.0, mf)
    assert np.ptp(s.t_exh) < 50.0
    assert np.ptp(s.lambda_total) < 0.3


@pytest.mark.parametrize("fuel", FUELS)
def test_turbocharger_fouling_raises_exhaust_temperature_and_lowers_boost_from_half_load(fuel):
    """Design gate 4, second half. Signature of a fouled turbocharger: hotter exhaust, less boost.

    Not asserted on purpose: exhaust-receiver pressure and turbocharger speed. In the coupled
    steady state the narrower turbine passage is partly offset by the smaller gas flow, so the
    receiver pressure can go either way; I have no source for the expected sign.
    At 30 % load boost rises slightly instead of falling; see the simulator report.
    """
    n, mf = PLANT.propeller_law(np.array([0.5, 0.75, 1.0]), fuel)
    clean = PLANT.solve(n, mf, fuel=fuel)
    foul = PLANT.solve(n, mf, fuel=fuel, foul_tc=1.0)
    assert np.all(foul.t_exh > clean.t_exh + 5.0)
    assert np.all(foul.p_scav < clean.p_scav)
    assert np.all(foul.n_tc != clean.n_tc)  # the speed-to-boost relationship does change


@pytest.mark.parametrize("fuel", FUELS)
def test_cooler_fouling_raises_scavenge_and_exhaust_temperature(fuel):
    n, mf = PLANT.propeller_law(np.array([0.3, 0.6, 1.0]), fuel)
    clean = PLANT.solve(n, mf, fuel=fuel)
    foul = PLANT.solve(n, mf, fuel=fuel, foul_cool=1.0)
    assert np.all(foul.t_scav > clean.t_scav + 5.0)
    assert np.all(foul.t_exh > clean.t_exh)
    assert np.all(foul.dp_cooler > clean.dp_cooler)


def test_fouling_effects_grow_monotonically():
    n, mf = PLANT.propeller_law(0.8)
    t = [float(PLANT.solve(n, mf, foul_tc=f).t_exh) for f in (0.0, 0.25, 0.5, 0.75, 1.0)]
    assert all(b > a for a, b in zip(t, t[1:]))
    t = [float(PLANT.solve(n, mf, foul_cool=f).t_exh) for f in (0.0, 0.25, 0.5, 0.75, 1.0)]
    assert all(b > a for a, b in zip(t, t[1:]))


def test_warmer_ambient_raises_exhaust_temperature():
    n, mf = PLANT.propeller_law(np.array([0.5, 1.0]))
    cold = PLANT.solve(n, mf, t_amb=278.15)
    warm = PLANT.solve(n, mf, t_amb=308.15)
    assert np.all(warm.t_exh > cold.t_exh) and np.all(warm.t_scav > cold.t_scav)


@pytest.mark.parametrize("load", [0.3, 0.6, 1.0])
def test_methanol_exhaust_is_cooler_than_diesel_at_equal_power(load):
    """Direction only; the size is reported in the gate report. Composition argument:
    methanol needs ~2.2x the fuel mass for the same heat, adding hot gas mass and water."""
    d = PLANT.solve(*PLANT.propeller_law(load, "diesel_like"), fuel="diesel_like")
    m = PLANT.solve(*PLANT.propeller_law(load, "methanol"), fuel="methanol")
    assert float(m.t_exh) < float(d.t_exh)
    assert np.isclose(float(m.p_brake), float(d.p_brake), rtol=1e-9)


# ---------------------------------------------------- hidden state stays hidden (gate 7) ----
def test_signals_contain_no_hidden_state_and_hidden_contains_no_signals():
    s = _propeller("diesel_like")
    sig, hid = set(s.signals()), set(s.hidden())
    assert sig == set(SteadyState.OBSERVABLE)
    assert not sig & hid
    assert {"foul_tc", "foul_cool", "p_exh", "m_air", "delivery_ratio", "t_turb_out"} <= hid
    assert "t_turb_out" not in sig  # turbine outlet temperature is excluded on purpose (design 6.3)
