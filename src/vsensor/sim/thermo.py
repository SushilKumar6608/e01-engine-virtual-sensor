"""Gas thermodynamics from a frozen NASA-7 table (SIMULATION PROJECT).

Plain-language summary
----------------------
The exhaust temperature comes from an energy balance: heat that is not turned
into work or lost through the walls must leave with the exhaust gas. To turn
"heat leaving" into "temperature", we need to know how much energy a kilogram
of gas holds at each temperature. That is the enthalpy h(T).

Gas tables give h(T) as a polynomial per species ("NASA-7"). The numbers come
from Cantera's data files (see scripts/build_thermo_table.py). Here they are
evaluated with NumPy so that thousands of operating points can be computed at
once, and so the runtime does not depend on Cantera.

Units: SI. Molar quantities are per mol (J/mol), temperatures in K.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np

_TABLE_PATH = Path(__file__).with_name("thermo_nasa7.json")

#: Universal gas constant [J/(mol K)] - the Cantera value (CODATA 2018, exact).
R_UNIVERSAL = 8.31446261815324

#: Reference temperature for sensible enthalpy and for the heating value [K].
T_REF = 298.15


@lru_cache(maxsize=1)
def _table() -> dict:
    return json.loads(_TABLE_PATH.read_text())


def table_metadata() -> dict:
    """Provenance of the thermo table (Cantera version etc.)."""
    t = _table()
    return {"cantera_version": t["cantera_version"], "species": sorted(t["species"])}


def species_names() -> list[str]:
    return sorted(_table()["species"])


def molar_mass(species: str) -> float:
    """Molar mass [kg/mol]."""
    return _table()["species"][species]["molecular_weight_kg_per_kmol"] / 1000.0


def composition(species: str) -> dict[str, float]:
    """Atoms per molecule, e.g. {'C': 1, 'H': 4, 'O': 1}."""
    return dict(_table()["species"][species]["composition"])


def _coeffs(species: str, T: np.ndarray) -> np.ndarray:
    """Pick the low- or high-temperature coefficient set per element of T."""
    sp = _table()["species"][species]
    low = np.asarray(sp["coeffs_low"])
    high = np.asarray(sp["coeffs_high"])
    # Same branch rule as Cantera: low set up to and including T_mid.
    use_high = (np.asarray(T) > sp["t_mid"])[..., None]
    return np.where(use_high, high, low)


def h_molar(species: str, T) -> np.ndarray:
    """Molar enthalpy [J/mol], including the formation enthalpy."""
    T = np.asarray(T, dtype=float)
    a = _coeffs(species, T)
    h_rt = (
        a[..., 0]
        + a[..., 1] * T / 2
        + a[..., 2] * T**2 / 3
        + a[..., 3] * T**3 / 4
        + a[..., 4] * T**4 / 5
        + a[..., 5] / T
    )
    return h_rt * R_UNIVERSAL * T


def cp_molar(species: str, T) -> np.ndarray:
    """Molar heat capacity at constant pressure [J/(mol K)]."""
    T = np.asarray(T, dtype=float)
    a = _coeffs(species, T)
    cp_r = a[..., 0] + a[..., 1] * T + a[..., 2] * T**2 + a[..., 3] * T**3 + a[..., 4] * T**4
    return cp_r * R_UNIVERSAL


def sensible_h_molar(species: str, T) -> np.ndarray:
    """h(T) - h(T_REF) [J/mol]: the energy needed to heat the gas from 298.15 K."""
    return h_molar(species, T) - h_molar(species, T_REF)


# ---------------------------------------------------------------------------
# Mixtures described by mole flows {species: mol/s} (arrays allowed)
# ---------------------------------------------------------------------------
def mixture_sensible_enthalpy_rate(moles: dict, T) -> np.ndarray:
    """Sum_k n_k [h_k(T) - h_k(T_REF)] in W for mole flows n_k in mol/s."""
    return sum(n * sensible_h_molar(sp, T) for sp, n in moles.items())


def mixture_cp_rate(moles: dict, T) -> np.ndarray:
    """Sum_k n_k cp_k(T) in W/K (= m_dot * cp for the mixture)."""
    return sum(n * cp_molar(sp, T) for sp, n in moles.items())


def mixture_gas_constant_rate(moles: dict) -> np.ndarray:
    """Sum_k n_k R in W/K (= m_dot * R_specific)."""
    return sum(n for n in moles.values()) * R_UNIVERSAL


def solve_temperature(moles: dict, enthalpy_rate_above_ref, T_guess=700.0, tol=1e-9, max_iter=60):
    """Find T such that Sum_k n_k [h_k(T)-h_k(T_REF)] = enthalpy_rate_above_ref.

    Newton's method: the derivative of the left side is exactly Sum n_k cp_k(T),
    which is positive, so the iteration is well behaved. Works on arrays.
    """
    target = np.asarray(enthalpy_rate_above_ref, dtype=float)
    T = np.broadcast_to(np.asarray(T_guess, dtype=float), target.shape).copy()
    for _ in range(max_iter):
        f = mixture_sensible_enthalpy_rate(moles, T) - target
        dT = f / mixture_cp_rate(moles, T)
        T = T - dT
        if np.max(np.abs(dT)) < tol:
            break
    else:  # pragma: no cover - would indicate an out-of-range request
        raise RuntimeError("temperature solve did not converge")
    return T
