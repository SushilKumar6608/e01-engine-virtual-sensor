"""Nominal model: what an engineer would have (SIMULATION PROJECT).

Plain-language summary
----------------------
The nominal model estimates the exhaust temperature before the turbine from a
simple heat balance for a *healthy* engine, using only signals an instrument can
deliver: engine speed, fuel flow, scavenge pressure, scavenge temperature, fuel.

  1. Air flow = delivery ratio x air density x displaced volume per second.
     (Speed-density rule: denser air and faster running swallow more air.)
  2. A fixed share of the fuel energy leaves with the exhaust gas.
  3. Exhaust enthalpy = incoming air enthalpy + that share of the fuel energy.
     Solving for temperature gives the estimate.

How it differs from the plant (on purpose - see design doc, section 5.2)
-----------------------------------------------------------------------
* constant delivery ratio, whereas the plant's depends on the exhaust-to-scavenge
  pressure ratio (which the nominal model cannot see: exhaust pressure is not a signal);
* constant exhaust-energy share, whereas the plant's indicated efficiency changes with load;
* it knows nothing about fouling or the sea-water temperature - it only sees their
  effect through the measured scavenge pressure and temperature.

The two constants (delivery ratio, exhaust share) are calibrated once on the
*healthy* plant at ISO ambient with the diesel-like fuel over the 40-100 % operating range
(scripts/calibrate_nominal.py). That is the commissioning data an engineer would
plausibly have, called the *commissioning baseline*. It is still a two-parameter model with
no load-dependent efficiency curve. The stronger baseline M0b (delivery ratio that depends on
the measured boost) and the way M0 and M0b are fitted on the same training episodes are fixed
in docs/MODEL_CHOICES.md before any test result exists.

Leakage rule: this function must never receive hidden plant states. Its argument
list is the contract, and tests/test_nominal.py checks it.
"""
from __future__ import annotations

import numpy as np

from vsensor.params import Registry, load_registry
from vsensor.sim import thermo
from vsensor.sim.fuels import FUEL_SPECIES, Fuel, load_air, load_fuel

CODE_TO_FUEL = {code: fid for fid, (_, code) in FUEL_SPECIES.items()}


class NominalModel:
    def __init__(self, reg: Registry | None = None):
        reg = reg or load_registry()
        self.reg = reg
        self.air = load_air(reg)
        self.fuels: dict[str, Fuel] = {fid: load_fuel(fid, reg, self.air) for fid in FUEL_SPECIES}
        self.v_disp = np.pi / 4.0 * reg["engine.bore"] ** 2 * reg["engine.stroke"] * reg["engine.n_cyl"]
        self.lam = reg["nominal.lambda_del"]
        self.phi = reg["nominal.phi_exhaust"]
        self.r_air = thermo.R_UNIVERSAL / self.air.molar_mass

    def predict(self, n_rpm, m_fuel, p_scav, t_scav, fuel: str = "diesel_like"):
        """Estimated exhaust temperature before the turbine [K]. Inputs in SI units (rpm for speed)."""
        fu = self.fuels[fuel]
        n_rpm, m_fuel, p_scav, t_scav = np.broadcast_arrays(
            *(np.asarray(v, dtype=float) for v in (n_rpm, m_fuel, p_scav, t_scav))
        )
        rho_s = p_scav / (self.r_air * t_scav)
        m_air = self.lam * rho_s * self.v_disp * n_rpm / 60.0
        moles = fu.exhaust_moles(m_fuel, m_air, self.air)
        h_air = thermo.mixture_sensible_enthalpy_rate(self.air.moles(m_air), t_scav)
        q_exhaust = self.phi * m_fuel * fu.lhv_liquid
        return thermo.solve_temperature(moles, h_air + q_exhaust, T_guess=700.0)

    def predict_by_code(self, n_rpm, m_fuel, p_scav, t_scav, fuel_code):
        """Same as :meth:`predict` for arrays where each row has its own fuel flag."""
        n_rpm, m_fuel, p_scav, t_scav, fuel_code = np.broadcast_arrays(
            *(np.asarray(v, dtype=float) for v in (n_rpm, m_fuel, p_scav, t_scav, fuel_code))
        )
        out = np.empty_like(n_rpm)
        for code, fid in CODE_TO_FUEL.items():
            m = fuel_code == code
            if np.any(m):
                out[m] = self.predict(n_rpm[m], m_fuel[m], p_scav[m], t_scav[m], fid)
        return out
