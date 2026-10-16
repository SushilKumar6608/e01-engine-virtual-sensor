"""Fuels, air, and exhaust-gas bookkeeping (SIMULATION PROJECT).

Plain-language summary
----------------------
Two fuels are modelled:

* "diesel_like"  - n-dodecane (C12H26) standing in for marine gas oil. It is a
  surrogate: a single clean molecule, not a real blend. Its heating value from
  the thermo data is compared with marine gas oil values in docs/CANTERA_CHECK.md.
* "methanol"     - CH3OH.

For each fuel this module works out, from the chemical formula only:
  * how much air burns it exactly (stoichiometric air-fuel ratio),
  * how much heat it releases (lower heating value, LHV),
  * what the exhaust gas is made of for any amount of extra air.

Why not just look these numbers up? Because the same few lines then work for
both fuels and every operating point, and the check against published tables
(done in scripts/cantera_check.py) tests the whole chain at once.

Lower heating value (LHV) = heat released when the *liquid* fuel at 298 K burns
completely and the products (including water as vapour) return to 298 K.
Here: LHV_liquid = LHV_gas - latent heat of vaporisation, because the thermo
data describe the fuel as a gas but the engine injects a liquid.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from vsensor.params import Registry
from vsensor.sim import thermo

#: fuel id -> (species name in the thermo table, numeric flag used as a model input)
FUEL_SPECIES = {
    "diesel_like": ("C12H26", 0),
    "methanol": ("CH3OH", 1),
}


@dataclass(frozen=True)
class Air:
    """Dry air as a mixture of O2, N2, Ar (mole fractions)."""

    x: dict
    molar_mass: float  # kg/mol

    def moles(self, m_air) -> dict:
        """Mole flows [mol/s] for an air mass flow [kg/s]."""
        n = np.asarray(m_air, dtype=float) / self.molar_mass
        return {sp: xi * n for sp, xi in self.x.items()}


@dataclass(frozen=True)
class Fuel:
    fuel_id: str
    species: str
    code: int
    n_c: float
    n_h: float
    n_o: float
    molar_mass: float  # kg/mol
    latent_heat_molar: float  # J/mol
    lhv_gas: float  # J/kg
    lhv_liquid: float  # J/kg
    o2_per_fuel: float  # mol O2 per mol fuel for complete combustion
    afr_stoich: float  # kg air per kg fuel

    def exhaust_moles(self, m_fuel, m_air, air: Air) -> dict:
        """Mole flows [mol/s] of the exhaust gas, assuming complete combustion.

        The fuel burns with part of the air. All remaining air (including the
        scavenging air that never took part in combustion) stays in the gas.
        Raises if there is not enough air to burn the fuel (rich operation is
        outside this model).
        """
        n_f = np.asarray(m_fuel, dtype=float) / self.molar_mass
        a = air.moles(m_air)
        o2_left = a["O2"] - self.o2_per_fuel * n_f
        # Exactly stoichiometric air is a valid state; tolerate floating-point rounding there.
        if np.any(o2_left < -1e-9 * a["O2"]):
            raise ValueError("not enough air to burn the fuel completely (lambda < 1)")
        o2_left = np.maximum(o2_left, 0.0)
        return {
            "CO2": self.n_c * n_f,
            "H2O": 0.5 * self.n_h * n_f,
            "O2": o2_left,
            "N2": a["N2"],
            "AR": a["AR"],
        }


def load_air(reg: Registry) -> Air:
    x = {"O2": reg["air.x_o2"], "N2": reg["air.x_n2"], "AR": reg["air.x_ar"]}
    total = sum(x.values())
    # O2+N2+Ar sum to ~0.9996: the missing ~0.04 % is trace CO2 etc., which is
    # ignored. The three fractions are renormalised to sum to exactly 1.
    if abs(total - 1.0) > 1e-3:
        raise ValueError(f"air mole fractions sum to {total}, expected about 1")
    x = {k: v / total for k, v in x.items()}
    m = sum(xi * thermo.molar_mass(sp) for sp, xi in x.items())
    return Air(x=x, molar_mass=m)


def load_fuel(fuel_id: str, reg: Registry, air: Air | None = None) -> Fuel:
    species, code = FUEL_SPECIES[fuel_id]
    comp = thermo.composition(species)
    n_c, n_h, n_o = comp.get("C", 0.0), comp.get("H", 0.0), comp.get("O", 0.0)
    M = thermo.molar_mass(species)
    o2 = n_c + n_h / 4.0 - n_o / 2.0

    # Heat released per mol of (gaseous) fuel at 298.15 K, water left as vapour.
    t = thermo.T_REF
    q_gas_molar = (
        thermo.h_molar(species, t)
        + o2 * thermo.h_molar("O2", t)
        - n_c * thermo.h_molar("CO2", t)
        - 0.5 * n_h * thermo.h_molar("H2O", t)
    )
    latent = reg[f"fuel.{fuel_id}.latent_heat"]
    air = air or load_air(reg)
    afr = o2 / air.x["O2"] * air.molar_mass / M
    return Fuel(
        fuel_id=fuel_id,
        species=species,
        code=code,
        n_c=n_c,
        n_h=n_h,
        n_o=n_o,
        molar_mass=M,
        latent_heat_molar=latent,
        lhv_gas=float(q_gas_molar / M),
        lhv_liquid=float((q_gas_molar - latent) / M),
        o2_per_fuel=o2,
        afr_stoich=float(afr),
    )
