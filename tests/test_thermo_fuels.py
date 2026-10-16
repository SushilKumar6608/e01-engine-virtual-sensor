"""Thermo table, fuels and exhaust bookkeeping (SIMULATION PROJECT)."""
from __future__ import annotations

import numpy as np
import pytest

from vsensor.params import load_registry
from vsensor.sim import thermo
from vsensor.sim.fuels import FUEL_SPECIES, load_air, load_fuel

ct = pytest.importorskip("cantera")

REG = load_registry()
AIR = load_air(REG)
FUELS = {f: load_fuel(f, REG, AIR) for f in FUEL_SPECIES}
_PHASES = {"gri30.yaml": ct.Solution("gri30.yaml"), "nDodecane_Reitz.yaml": ct.Solution("nDodecane_Reitz.yaml")}


@pytest.mark.parametrize("species", thermo.species_names())
def test_numpy_thermo_matches_cantera(species):
    meta = thermo._table()["species"][species]
    th = _PHASES[meta["source_file"]].species(meta["cantera_name"]).thermo
    T = np.linspace(max(meta["t_min"], 250.0), min(meta["t_max"], 2500.0), 80)
    h_ref = np.array([th.h(t) for t in T]) / 1e3
    cp_ref = np.array([th.cp(t) for t in T]) / 1e3
    assert np.allclose(thermo.h_molar(species, T), h_ref, rtol=1e-10, atol=1e-6)
    assert np.allclose(thermo.cp_molar(species, T), cp_ref, rtol=1e-10)


def test_cp_is_derivative_of_h():
    T = np.array([400.0, 700.0, 1100.0])
    for sp in ("CO2", "H2O", "N2", "O2"):
        dh = (thermo.h_molar(sp, T + 0.01) - thermo.h_molar(sp, T - 0.01)) / 0.02
        assert np.allclose(dh, thermo.cp_molar(sp, T), rtol=1e-6)


def test_temperature_solver_inverts_enthalpy():
    moles = FUELS["diesel_like"].exhaust_moles(0.3, 13.0, AIR)
    for T_true in (450.0, 700.0, 950.0):
        H = thermo.mixture_sensible_enthalpy_rate(moles, T_true)
        assert abs(float(thermo.solve_temperature(moles, H, T_guess=600.0)) - T_true) < 1e-6


def test_air_molar_mass_is_sensible():
    assert 0.0289 < AIR.molar_mass < 0.0291
    assert abs(sum(AIR.x.values()) - 1.0) < 1e-12


# ---- fuel properties against independent references (see docs/CANTERA_CHECK.md) ------------
def test_methanol_against_iea_amf_table():
    m = FUELS["methanol"]
    assert abs(m.lhv_liquid / 20.0e6 - 1.0) < 0.01  # S04: 20.0 MJ/kg (rounded)
    assert abs(m.afr_stoich / 6.4 - 1.0) < 0.02  # S04: 6.4
    assert abs(m.molar_mass - 0.032042) < 1e-5


def test_dodecane_against_nist_route():
    d = FUELS["diesel_like"]
    t = thermo.T_REF
    # NIST liquid formation enthalpy -352.1 kJ/mol (Prosen & Rossini 1945) [S03]
    q = -352.1 - (12 * thermo.h_molar("CO2", t) / 1e3 + 13 * thermo.h_molar("H2O", t) / 1e3)
    lhv_nist = q * 1e3 / d.molar_mass
    assert abs(d.lhv_liquid / lhv_nist - 1.0) < 0.005
    assert 44.0e6 <= d.lhv_liquid <= 45.0e6  # inside marine gas oil range [S06]
    assert abs(d.afr_stoich - 15.0) < 0.1


def test_liquid_lhv_is_gas_lhv_minus_latent_heat():
    for f in FUELS.values():
        assert f.lhv_gas > f.lhv_liquid
        assert abs((f.lhv_gas - f.lhv_liquid) * f.molar_mass - f.latent_heat_molar) < 1e-6


# ---- conservation in the exhaust bookkeeping ---------------------------------------------
def _mass(moles):
    return sum(float(n) * thermo.molar_mass(sp) for sp, n in moles.items())


@pytest.mark.parametrize("fuel", list(FUEL_SPECIES))
def test_exhaust_mass_and_atoms_are_conserved(fuel):
    f = FUELS[fuel]
    m_f, m_a = 0.25, 11.0
    out = f.exhaust_moles(m_f, m_a, AIR)
    assert abs(_mass(out) - (m_f + m_a)) / (m_f + m_a) < 1e-9
    n_f = m_f / f.molar_mass
    air = AIR.moles(m_a)
    for atom, n_in_fuel in (("C", f.n_c), ("H", f.n_h)):
        count = lambda sp: thermo.composition(sp).get(atom, 0.0)  # noqa: E731
        n_out = sum(float(n) * count(sp) for sp, n in out.items())
        assert abs(n_out - n_f * n_in_fuel) < 1e-9 * max(1.0, n_out)
    o_in = n_f * f.n_o + 2.0 * float(air["O2"])
    o_out = sum(float(n) * thermo.composition(sp).get("O", 0.0) for sp, n in out.items())
    assert abs(o_in - o_out) < 1e-9 * o_in
    n_in = 2.0 * float(air["N2"])
    n_out = 2.0 * float(out["N2"])
    assert abs(n_in - n_out) < 1e-12 * n_in


@pytest.mark.parametrize("fuel", list(FUEL_SPECIES))
def test_not_enough_air_is_refused(fuel):
    f = FUELS[fuel]
    with pytest.raises(ValueError, match="not enough air"):
        f.exhaust_moles(1.0, 0.9 * f.afr_stoich, AIR)


def test_stoichiometric_air_leaves_no_oxygen():
    for f in FUELS.values():
        out = f.exhaust_moles(1.0, f.afr_stoich, AIR)
        assert abs(float(out["O2"])) < 1e-9 * float(out["N2"])
