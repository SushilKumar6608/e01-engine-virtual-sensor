"""Plant simulator: the "ground truth" engine (SIMULATION PROJECT).

THIS IS NOT A MODEL OF ANY SPECIFIC COMMERCIAL ENGINE. It is a simplified
steady-state mean-value model with generic parameters (see config/params.yaml
for the origin of every number). Nothing here is validated against a real engine.

What it computes
----------------
Given an operating point (engine speed, fuel flow), ambient conditions, a fuel,
and two hidden health states (turbocharger fouling, air-cooler fouling), it finds
the steady state of the air path and returns the exhaust gas temperature before
the turbine - the quantity the virtual sensor later has to estimate.

The chain, in plain language
----------------------------
1. Fuel burns. Part of the fuel energy becomes piston work (indicated work),
   part is lost through the cylinder walls, the rest leaves as hot exhaust.
2. The engine swallows air. How much depends on the scavenge-air density and on
   how big the pressure drop is from the scavenge receiver to the exhaust
   receiver (a bigger drop pushes more air through).
3. Energy balance: exhaust enthalpy = incoming air enthalpy + the share of fuel
   energy that is left over. Solving that for temperature gives the exhaust
   temperature before the turbine (T_exh).
4. The turbine lets the exhaust gas expand and extracts power. The pressure in the
   exhaust receiver follows from how much gas the turbine lets through
   (Stodola-type flow law).
5. The turbine drives the compressor. All turbine power (minus a small bearing
   loss) goes into compressing the air, which sets the pressure ratio.
6. The compressed air is hot; the air cooler brings it down, with a small
   pressure loss. What comes out is the scavenge pressure and temperature.
7. Steps 2-6 depend on each other (more air -> cooler exhaust -> less turbine
   power -> lower boost -> less air ...). The solver repeats the loop until the
   numbers stop changing (a fixed point = steady state).

Hidden health states (never given to any model):
  foul_tc   0..1  turbocharger fouling: lower compressor and turbine efficiency,
                  narrower turbine passage (higher exhaust-receiver pressure).
  foul_cool 0..1  air-cooler fouling: air leaves the cooler hotter, with a larger
                  pressure loss.

Known simplifications (all documented as assumptions in the registry):
  constant component efficiencies (no compressor/turbine maps), constant-pressure
  turbocharging (no pulse effects), dry air, complete combustion, fuel sensible
  heat neglected, no EGR, no cylinder bypass, no auxiliary blower.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from vsensor.params import Registry, load_registry
from vsensor.sim import thermo
from vsensor.sim.fuels import FUEL_SPECIES, Fuel, load_air, load_fuel

_INSIDE_AIR_MARGIN = 1.02  # lambda_total must be at least this for complete combustion to be defined


@dataclass(frozen=True)
class SteadyState:
    """Result of a steady-state solve. All arrays share one shape. SI units unless noted."""

    # --- observable by instruments (inputs to the sensor model in the data generator) ---
    n_rpm: np.ndarray
    m_fuel: np.ndarray
    p_scav: np.ndarray  # Pa
    t_scav: np.ndarray  # K
    n_tc: np.ndarray  # rpm
    t_amb: np.ndarray  # K
    p_amb: np.ndarray  # Pa
    fuel_code: np.ndarray
    t_exh: np.ndarray  # K  <- TARGET (true value, before any sensor imperfections)
    # --- hidden: models must never see these ---
    foul_tc: np.ndarray
    foul_cool: np.ndarray
    p_exh: np.ndarray  # Pa, exhaust receiver
    m_air: np.ndarray  # kg/s
    delivery_ratio: np.ndarray
    lambda_total: np.ndarray
    p_brake: np.ndarray  # W
    p_indicated: np.ndarray  # W
    eta_brake: np.ndarray
    load_fraction: np.ndarray  # brake power / rated power
    pr_comp: np.ndarray
    t_comp_out: np.ndarray  # K
    p_turbine: np.ndarray  # W
    t_turb_out: np.ndarray  # K
    q_fuel: np.ndarray  # W
    q_wall: np.ndarray  # W
    q_cooler: np.ndarray  # W
    dp_cooler: np.ndarray  # Pa
    iterations: int
    max_residual: float
    converged: bool

    #: names of the instrument-observable fields (everything else is hidden state)
    OBSERVABLE = (
        "n_rpm", "m_fuel", "p_scav", "t_scav", "n_tc", "t_amb", "p_amb", "fuel_code", "t_exh",
    )

    def signals(self) -> dict:
        """Observable signals (noise-free)."""
        return {k: getattr(self, k) for k in self.OBSERVABLE}

    def hidden(self) -> dict:
        """Everything that is not an instrument signal. For tests and analysis only."""
        skip = set(self.OBSERVABLE) | {"iterations", "max_residual", "converged"}
        return {k: v for k, v in self.__dict__.items() if k not in skip}


class Plant:
    """Steady-state plant. Construct once, call :meth:`solve` for any operating points."""

    def __init__(self, reg: Registry | None = None):
        reg = reg or load_registry()
        self.reg = reg
        self.air = load_air(reg)
        self.fuels: dict[str, Fuel] = {fid: load_fuel(fid, reg, self.air) for fid in FUEL_SPECIES}
        r = reg
        # geometry and rating
        self.v_disp = np.pi / 4.0 * r["engine.bore"] ** 2 * r["engine.stroke"] * r["engine.n_cyl"]  # m3
        self.p_rated = r["engine.rated_power"]
        self.n_rated = r["engine.rated_speed"]
        # energy conversion
        self.eta_max = r["engine.eta_ind_max"]
        self.eta_curv = r["engine.eta_ind_curv"]
        self.x_opt = r["engine.eta_ind_x_opt"]
        self.fmep = r["engine.fmep"]
        self.f_wall = r["engine.f_wall"]
        self.p_fric_rated = self.fmep * self.v_disp * self.n_rated / 60.0
        # Efficiency scale per fuel. 1.0 for both in the main data (shared curve, an assumption);
        # the methanol value is only moved in the pre-specified efficiency stress test.
        self.eta_scale = {"diesel_like": 1.0, "methanol": r["engine.eta_scale_methanol"]}
        # reference fuel-energy rate that gives rated brake power at rated speed
        self.q_ref = (self.p_rated + self.p_fric_rated) / float(self.eta_indicated(1.0))
        # air path
        self.lam_coeff = r["scav.lambda_coeff"]
        self.t_cw_offset = r["cooler.t_coolant_offset"]
        self.eps0 = r["cooler.eps"]
        self.k_loss = r["cooler.k_loss"]
        self.foul_eps = r["cooler.foul_eps_loss"]
        self.foul_dp = r["cooler.foul_dp_gain"]
        # turbocharger
        self.eta_c0 = r["tc.eta_c"]
        self.eta_t0 = r["tc.eta_t"]
        self.eta_m = r["tc.eta_mech"]
        self.psi = r["tc.psi"]
        self.d_comp = r["tc.d_comp"]
        self.k_turb0 = r["tc.k_turbine"]
        self.p_back_ratio = r["tc.p_back_ratio"]
        self.cp_air = r["tc.cp_air"]
        self.kappa_air = r["tc.kappa_air"]
        self.foul_eta_c = r["tc.foul_eta_c_loss"]
        self.foul_eta_t = r["tc.foul_eta_t_loss"]
        self.foul_k = r["tc.foul_k_loss"]
        self.r_air = thermo.R_UNIVERSAL / self.air.molar_mass  # J/(kg K)
        # defaults
        self.t_iso = r["env.iso_t_amb"]
        self.p_iso = r["env.iso_p_amb"]

    # ------------------------------------------------------------------ engine
    def eta_indicated(self, x, fuel: str = "diesel_like"):
        """Gross indicated efficiency versus relative fuel-energy rate x = Q_fuel / Q_ref.

        A parabola with its peak at x_opt. Generic shape (assumption), not fitted to data.
        Multiplied by the fuel's efficiency scale (1.0 unless a stress test moves it).
        """
        x = np.asarray(x, dtype=float)
        return self.eta_scale[fuel] * self.eta_max * (1.0 - self.eta_curv * (x - self.x_opt) ** 2)

    def friction_power(self, n_rpm):
        """Friction power [W] = FMEP * displacement * cycles per second (two-stroke: 1 cycle/rev)."""
        return self.fmep * self.v_disp * np.asarray(n_rpm, dtype=float) / 60.0

    def brake_power(self, m_fuel, n_rpm, fuel: str = "diesel_like"):
        q = np.asarray(m_fuel, dtype=float) * self.fuels[fuel].lhv_liquid
        return q * self.eta_indicated(q / self.q_ref, fuel) - self.friction_power(n_rpm)

    def fuel_flow_for_power(self, p_brake, n_rpm, fuel: str = "diesel_like"):
        """Fuel mass flow [kg/s] that gives the requested brake power at the given speed (Newton)."""
        p_brake, n_rpm = np.broadcast_arrays(np.asarray(p_brake, float), np.asarray(n_rpm, float))
        target = p_brake + self.friction_power(n_rpm)  # required indicated work
        scale = self.eta_scale[fuel]
        x = np.maximum(target / self.q_ref / (scale * self.eta_max), 0.05)
        for _ in range(60):
            f = self.q_ref * x * self.eta_indicated(x, fuel) - target
            df = self.q_ref * scale * self.eta_max * (
                1.0 - self.eta_curv * (x - self.x_opt) ** 2 - 2.0 * self.eta_curv * x * (x - self.x_opt)
            )
            dx = f / df
            x = x - dx
            if np.max(np.abs(dx)) < 1e-12:
                break
        return x * self.q_ref / self.fuels[fuel].lhv_liquid

    def propeller_law(self, load, fuel: str = "diesel_like"):
        """Operating point on the propeller law: power ~ speed cubed.

        ``load`` is brake power as a fraction of rated. Returns (n_rpm, m_fuel).
        """
        load = np.asarray(load, dtype=float)
        n = self.n_rated * load ** (1.0 / 3.0)
        return n, self.fuel_flow_for_power(load * self.p_rated, n, fuel)

    # -------------------------------------------------------- energy balance step
    def heat_to_exhaust(self, m_fuel, fuel: str = "diesel_like"):
        """Fuel energy that is neither piston work nor wall loss, i.e. what leaves with the gas [W]."""
        q = np.asarray(m_fuel, dtype=float) * self.fuels[fuel].lhv_liquid
        return q * (1.0 - self.eta_indicated(q / self.q_ref, fuel) - self.f_wall)

    def _exhaust_state(self, fu: Fuel, m_fuel, m_air, t_scav, q_exhaust):
        """Exhaust mole flows and temperature from the energy balance (steps 1-3 of the module text)."""
        moles = fu.exhaust_moles(m_fuel, m_air, self.air)
        h_air_in = m_air * self.cp_air * (t_scav - thermo.T_REF)
        return moles, thermo.solve_temperature(moles, h_air_in + q_exhaust, T_guess=700.0)

    def exhaust_temperature_fixed_air(self, m_fuel, m_air, t_scav, fuel: str = "diesel_like"):
        """Exhaust temperature for a GIVEN air flow and scavenge temperature [K].

        This is the energy-balance step on its own, with the turbocharger loop switched off.
        Design quality gate 4 ("at fixed air supply, exhaust temperature rises with fuel flow")
        is a statement about this function, not about the matched steady state, where the
        turbocharger delivers more air as fuel increases and the temperature barely moves.
        """
        m_fuel, m_air, t_scav = np.broadcast_arrays(*(np.asarray(v, float) for v in (m_fuel, m_air, t_scav)))
        _, t_x = self._exhaust_state(self.fuels[fuel], m_fuel, m_air, t_scav, self.heat_to_exhaust(m_fuel, fuel))
        return t_x

    # ------------------------------------------------------------------ solver
    def solve(
        self,
        n_rpm,
        m_fuel,
        t_amb=None,
        p_amb=None,
        foul_tc=0.0,
        foul_cool=0.0,
        fuel: str = "diesel_like",
        tol: float = 1e-11,
        max_iter: int = 2000,
        damping: float = 0.5,
    ) -> SteadyState:
        fu = self.fuels[fuel]
        t_amb = self.t_iso if t_amb is None else t_amb
        p_amb = self.p_iso if p_amb is None else p_amb
        n_rpm, m_fuel, t_amb, p_amb, f_tc, f_cl = np.broadcast_arrays(
            *(np.asarray(v, dtype=float) for v in (n_rpm, m_fuel, t_amb, p_amb, foul_tc, foul_cool))
        )
        shape = n_rpm.shape
        n_rpm, m_fuel, t_amb, p_amb, f_tc, f_cl = (a.ravel().copy() for a in (n_rpm, m_fuel, t_amb, p_amb, f_tc, f_cl))
        if np.any((f_tc < 0) | (f_tc > 1) | (f_cl < 0) | (f_cl > 1)):
            raise ValueError("fouling states must be in [0, 1]")

        # Quantities that do not depend on the air path ---------------------------------
        q_fuel = m_fuel * fu.lhv_liquid
        x_rel = q_fuel / self.q_ref
        eta_i = self.eta_indicated(x_rel, fuel)
        p_ind = eta_i * q_fuel
        q_wall = self.f_wall * q_fuel
        p_fric = self.friction_power(n_rpm)
        q_exhaust = q_fuel - p_ind - q_wall  # fuel energy that leaves with the gas
        cyc_hz = n_rpm / 60.0
        # Health-dependent component data
        eta_c = self.eta_c0 * (1.0 - self.foul_eta_c * f_tc)
        eta_t = self.eta_t0 * (1.0 - self.foul_eta_t * f_tc)
        k_turb = self.k_turb0 * (1.0 - self.foul_k * f_tc)
        eps = self.eps0 * (1.0 - self.foul_eps * f_cl)
        t_cw = t_amb + self.t_cw_offset
        p_back = self.p_back_ratio * p_amb
        m_air_min = _INSIDE_AIR_MARGIN * fu.afr_stoich * m_fuel

        # Fixed-point iteration on (p_scav, p_exh, t_scav) ----------------------------
        p_s = 3.0 * p_amb
        p_x = 0.75 * p_s
        t_s = t_amb + 25.0
        it, resid = 0, np.inf
        for it in range(1, max_iter + 1):
            rho_s = p_s / (self.r_air * t_s)
            delivery = self.lam_coeff * np.sqrt(np.maximum(1.0 - p_x / p_s, 1e-6))
            m_a = np.maximum(delivery * rho_s * self.v_disp * cyc_hz, m_air_min)
            moles, t_x = self._exhaust_state(fu, m_fuel, m_a, t_s, q_exhaust)
            m_t = m_a + m_fuel
            p_x_new = np.sqrt(p_back**2 + (m_t * np.sqrt(t_x) / k_turb) ** 2)
            cp_x = thermo.mixture_cp_rate(moles, t_x) / m_t
            r_x = thermo.mixture_gas_constant_rate(moles) / m_t
            kappa_x = cp_x / (cp_x - r_x)
            expo = (kappa_x - 1.0) / kappa_x
            p_turb = eta_t * m_t * cp_x * t_x * (1.0 - (p_back / p_x_new) ** expo)
            dh_c = self.eta_m * p_turb / m_a  # actual specific work into the air [J/kg]
            pr_c = (1.0 + eta_c * dh_c / (self.cp_air * t_amb)) ** (self.kappa_air / (self.kappa_air - 1.0))
            t_c2 = t_amb + dh_c / self.cp_air
            t_s_new = t_c2 - eps * (t_c2 - t_cw)
            rho_c2 = pr_c * p_amb / (self.r_air * t_c2)
            dp_cool = self.k_loss * (1.0 + self.foul_dp * f_cl) * m_a**2 / rho_c2
            p_s_new = pr_c * p_amb - dp_cool

            resid = max(
                np.max(np.abs(p_s_new / p_s - 1.0)),
                np.max(np.abs(p_x_new / p_x - 1.0)),
                np.max(np.abs(t_s_new / t_s - 1.0)),
            )
            p_s = p_s + damping * (p_s_new - p_s)
            p_x = p_x + damping * (p_x_new - p_x)
            t_s = t_s + damping * (t_s_new - t_s)
            if resid < tol:
                break
        converged = bool(resid < max(tol * 1e3, 1e-8))

        # One more pass with the converged state so every reported number is consistent.
        rho_s = p_s / (self.r_air * t_s)
        delivery = self.lam_coeff * np.sqrt(np.maximum(1.0 - p_x / p_s, 1e-6))
        m_a_raw = delivery * rho_s * self.v_disp * cyc_hz
        m_a = np.maximum(m_a_raw, m_air_min)
        if np.any(m_a_raw < m_air_min):
            converged = False  # a state with too little air for complete combustion is not valid
        moles, t_x = self._exhaust_state(fu, m_fuel, m_a, t_s, q_exhaust)
        m_t = m_a + m_fuel
        p_x = np.sqrt(p_back**2 + (m_t * np.sqrt(t_x) / k_turb) ** 2)
        cp_x = thermo.mixture_cp_rate(moles, t_x) / m_t
        r_x = thermo.mixture_gas_constant_rate(moles) / m_t
        kappa_x = cp_x / (cp_x - r_x)
        expo = (kappa_x - 1.0) / kappa_x
        p_turb = eta_t * m_t * cp_x * t_x * (1.0 - (p_back / p_x) ** expo)
        dh_c = self.eta_m * p_turb / m_a
        pr_c = (1.0 + eta_c * dh_c / (self.cp_air * t_amb)) ** (self.kappa_air / (self.kappa_air - 1.0))
        t_c2 = t_amb + dh_c / self.cp_air
        t_s_out = t_c2 - eps * (t_c2 - t_cw)
        rho_c2 = pr_c * p_amb / (self.r_air * t_c2)
        dp_cool = self.k_loss * (1.0 + self.foul_dp * f_cl) * m_a**2 / rho_c2
        p_s_out = pr_c * p_amb - dp_cool
        # Turbocharger speed from the Euler turbomachinery relation: specific work = psi * U^2.
        u_tip = np.sqrt(dh_c / self.psi)
        n_tc = u_tip / (self.d_comp / 2.0) * 60.0 / (2.0 * np.pi)
        # Temperature after the turbine from the enthalpy the turbine took out of the gas.
        h_x = thermo.mixture_sensible_enthalpy_rate(moles, t_x)
        t_turb_out = thermo.solve_temperature(moles, h_x - p_turb, T_guess=600.0)
        q_cooler = m_a * self.cp_air * (t_c2 - t_s_out)

        def out(a):
            return np.asarray(a, dtype=float).reshape(shape)

        return SteadyState(
            n_rpm=out(n_rpm), m_fuel=out(m_fuel), p_scav=out(p_s_out), t_scav=out(t_s_out), n_tc=out(n_tc),
            t_amb=out(t_amb), p_amb=out(p_amb), fuel_code=out(np.full_like(n_rpm, fu.code)), t_exh=out(t_x),
            foul_tc=out(f_tc), foul_cool=out(f_cl), p_exh=out(p_x), m_air=out(m_a), delivery_ratio=out(delivery),
            lambda_total=out(m_a / (fu.afr_stoich * m_fuel)), p_brake=out(p_ind - p_fric), p_indicated=out(p_ind),
            eta_brake=out((p_ind - p_fric) / q_fuel), load_fraction=out((p_ind - p_fric) / self.p_rated),
            pr_comp=out(pr_c), t_comp_out=out(t_c2), p_turbine=out(p_turb), t_turb_out=out(t_turb_out),
            q_fuel=out(q_fuel), q_wall=out(q_wall), q_cooler=out(q_cooler), dp_cooler=out(dp_cool),
            iterations=it, max_residual=float(resid), converged=converged,
        )
