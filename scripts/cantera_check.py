"""Cantera mechanism check (SIMULATION PROJECT).

Questions this answers (design document, section 5.3 and the project plan):
  1. Which methanol / n-dodecane mechanisms ship with Cantera?
  2. Do we need kinetics at all, or is thermodynamics enough?
  3. Do the thermo-derived fuel numbers (heating value, stoichiometric air-fuel
     ratio) agree with independent sources?
  4. Do two different Cantera files agree on the product-gas thermodynamics?
  5. Is the frozen NumPy thermo table faithful to Cantera?

Writes docs/CANTERA_CHECK.md. Every number in that file is computed here.
Run:  make cantera-check
"""
from __future__ import annotations

import contextlib
import io
import warnings
from pathlib import Path

import cantera as ct
import numpy as np

from vsensor.params import load_registry
from vsensor.sim import thermo
from vsensor.sim.fuels import load_air, load_fuel

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "CANTERA_CHECK.md"

# ---- independent reference values (with source ids from config/params.yaml) -----
# Methanol (S04, IEA-AMF): LHV 20.0 MJ/kg, stoichiometric A/F 6.4, rounded figures.
METHANOL_LHV_REF = 20.0e6
METHANOL_AFR_REF = 6.4
# n-dodecane liquid formation enthalpy (S03, NIST WebBook): two values from two studies.
DODECANE_DFH_LIQ_KJ = {"Prosen & Rossini 1945": -352.1, "Cox & Pilcher 1970": -349.3}
# Marine gas oil LHV range (S06, Danish EPA 1998): 44-45 MJ/kg.
MGO_LHV_RANGE = (44.0e6, 45.0e6)


def scan_mechanisms() -> list[dict]:
    """Try to load every top-level YAML file Cantera ships; record fuel species."""
    rows = []
    for fname in sorted(f for f in ct.list_data_files() if "/" not in f):
        row = {"file": fname, "species": None, "reactions": None, "methanol": False, "dodecane": False}
        try:
            with warnings.catch_warnings(), contextlib.redirect_stderr(io.StringIO()), contextlib.redirect_stdout(io.StringIO()):
                warnings.simplefilter("ignore")
                g = ct.Solution(fname)
            names = [s.lower() for s in g.species_names]
            row.update(
                species=g.n_species,
                reactions=g.n_reactions,
                methanol="ch3oh" in names,
                dodecane=any(n in ("c12h26", "nc12h26") for n in names),
                n_ch3oh_reactions=sum("CH3OH" in r.equation.upper() for r in g.reactions()),
            )
        except Exception:
            row["note"] = "not a loadable gas phase"
        rows.append(row)
    return rows


def flame_temperature(fname: str, fuel_species: str, o2_per_fuel: float) -> float:
    """Adiabatic flame temperature [K], stoichiometric fuel-air, gas-phase fuel, 298.15 K, 1 atm."""
    g = ct.Solution(fname)
    air_n2 = 0.7808 / 0.2095  # N2 per O2 (same air as the registry)
    g.TPX = 298.15, ct.one_atm, {fuel_species: 1.0, "O2" if "O2" in g.species_names else "o2": o2_per_fuel, ("N2" if "N2" in g.species_names else "n2"): o2_per_fuel * air_n2}
    g.equilibrate("HP")
    return float(g.T)


def main() -> None:
    reg = load_registry()
    air = load_air(reg)
    diesel = load_fuel("diesel_like", reg, air)
    meoh = load_fuel("methanol", reg, air)
    lines: list[str] = []
    ok_all = True

    def check(name: str, passed: bool, detail: str) -> str:
        nonlocal ok_all
        ok_all &= passed
        return f"| {name} | {'PASS' if passed else '**FAIL**'} | {detail} |"

    # ---- 1. mechanisms shipped -------------------------------------------------
    rows = scan_mechanisms()
    relevant = [r for r in rows if r["methanol"] or r["dodecane"]]

    # ---- 3. fuel numbers vs independent sources ---------------------------------
    # n-dodecane: independent route = NIST liquid formation enthalpy + product formation enthalpies
    # taken from the SAME thermo table at 298.15 K (CO2, H2O gas) - so this tests the fuel's
    # formation enthalpy in the Reitz file against NIST, not the products.
    t = thermo.T_REF
    h_co2 = thermo.h_molar("CO2", t) / 1e3
    h_h2o = thermo.h_molar("H2O", t) / 1e3
    o2 = diesel.o2_per_fuel
    lhv_nist = {}
    for who, dfh in DODECANE_DFH_LIQ_KJ.items():
        # Heat released per mol of liquid fuel [kJ/mol] = H(reactants) - H(products), O2 has zero formation enthalpy.
        q = dfh - (diesel.n_c * h_co2 + 0.5 * diesel.n_h * h_h2o)
        lhv_nist[who] = q * 1e3 / diesel.molar_mass  # J/kg
    d_gas_vs_nist = thermo.h_molar("C12H26", t) / 1e3

    # ---- 4. two files agree on product gases? -----------------------------------
    g1, g2 = ct.Solution("gri30.yaml"), ct.Solution("nDodecane_Reitz.yaml")
    # Relative metric only over 500-1000 K (the exhaust range). Near 298 K the quantity
    # h(T)-h(298 K) tends to zero, so a relative difference there is meaningless (a first
    # version of this check used 300-1500 K and "failed" at 300 K for exactly that reason).
    max_rel = 0.0
    worst_sp = ""
    for sp, sp2 in (("CO2", "co2"), ("H2O", "h2o"), ("N2", "n2"), ("O2", "o2")):
        for T in np.linspace(500, 1000, 11):
            a = g1.species(sp).thermo.h(T) - g1.species(sp).thermo.h(298.15)
            b = g2.species(sp2).thermo.h(T) - g2.species(sp2).thermo.h(298.15)
            if abs(a - b) / abs(a) > max_rel:
                max_rel, worst_sp = abs(a - b) / abs(a), sp

    # What matters physically: how many kelvin could this data difference move the exhaust
    # temperature? Evaluate the same representative lean exhaust (diesel-like fuel, excess-air
    # ratio 2.5, 700 K) with each file's product thermo and convert the enthalpy difference to K.
    n_mix = diesel.exhaust_moles(1.0, 2.5 * diesel.afr_stoich, air)  # mol/s for 1 kg/s fuel
    key = {"CO2": "co2", "H2O": "h2o", "N2": "n2", "O2": "o2", "AR": "ar"}
    def sens(gas, sp, T):
        th = gas.species(sp).thermo
        return th.h(T) / 1e3 - th.h(298.15) / 1e3  # J/mol
    H1 = sum(n * sens(g1, sp.upper() if sp != "AR" else "AR", 700.0) for sp, n in n_mix.items())
    H2 = sum(n * sens(g2, key[sp], 700.0) for sp, n in n_mix.items() if sp != "AR")
    H2 += n_mix["AR"] * sens(g1, "AR", 700.0)  # the Reitz file has no argon; same value used
    cp_mix = float(thermo.mixture_cp_rate(n_mix, 700.0))
    dT_files = (H1 - H2) / cp_mix

    # ---- 5. frozen NumPy table faithful to Cantera? ----------------------------
    max_h = max_cp = 0.0
    for name in thermo.species_names():
        meta = thermo._table()["species"][name]
        g = g1 if meta["source_file"] == "gri30.yaml" else g2
        th = g.species(meta["cantera_name"]).thermo
        for T in np.linspace(max(meta["t_min"], 250.0), min(meta["t_max"], 2500.0), 60):
            h_ref, cp_ref = th.h(T) / 1e3, th.cp(T) / 1e3
            max_h = max(max_h, abs(float(thermo.h_molar(name, T)) - h_ref) / max(1.0, abs(h_ref)))
            max_cp = max(max_cp, abs(float(thermo.cp_molar(name, T)) - cp_ref) / cp_ref)

    # ---- 6. flame temperatures (information only) ------------------------------
    tad_meoh = flame_temperature("gri30.yaml", "CH3OH", meoh.o2_per_fuel)
    tad_dod = flame_temperature("nDodecane_Reitz.yaml", "c12h26", diesel.o2_per_fuel)

    # ---- 7. fuel effect on exhaust temperature from composition alone -----------------
    # Same air flow (13.5 kg/s at 317 K), same fuel heat input (14 MW), same share of it leaving
    # with the gas (7 MW). Only the fuel changes. This isolates the exhaust-composition effect;
    # the full plant adds a turbocharger feedback on top (see the simulator report).
    comp = {}
    h_air_c = thermo.mixture_sensible_enthalpy_rate(air.moles(13.5), 317.0)
    for name, F in (("diesel_like", diesel), ("methanol", meoh)):
        mf = 14.0e6 / F.lhv_liquid
        mol = F.exhaust_moles(mf, 13.5, air)
        T = float(thermo.solve_temperature(mol, h_air_c + 7.0e6, T_guess=700.0))
        comp[name] = (mf, T, float(thermo.mixture_cp_rate(mol, T)) / 1e3)
    d_comp_effect = comp["methanol"][1] - comp["diesel_like"][1]

    # ---------------------------------------------------------------- report
    L = lines.append
    L("# Cantera mechanism check")
    L("")
    L("**SIMULATED PROJECT.** This file is generated by `scripts/cantera_check.py`; every number below is computed there. Cantera version: " + ct.__version__ + ".")
    L("")
    L("## 1. Which mechanisms ship with Cantera?")
    L("")
    L("| File | Species | Reactions | Has methanol (CH3OH) | Has n-dodecane (c12h26) | Reactions mentioning CH3OH |")
    L("|---|---|---|---|---|---|")
    for r in relevant:
        L(f"| `{r['file']}` | {r['species']} | {r['reactions']} | {'yes' if r['methanol'] else 'no'} | {'yes' if r['dodecane'] else 'no'} | {r.get('n_ch3oh_reactions', '-')} |")
    L("")
    L(f"Out of {len(rows)} top-level data files tried, only the files above contain either fuel. There is **no dedicated methanol kinetic mechanism** in this Cantera installation. "
      "`gri30.yaml` (GRI-Mech 3.0) is a natural-gas mechanism whose methanol chemistry is limited to the reactions counted above; I did not evaluate its kinetic accuracy for methanol. "
      "`nDodecane_Reitz.yaml` is a kinetic mechanism for n-dodecane.")
    L("")
    L("## 2. Decision: kinetics or thermodynamics?")
    L("")
    L("**Thermodynamics only.** The energy balance needs (a) the heat released per kg of fuel, (b) the air needed to burn it, and (c) the enthalpy of the exhaust gas as a function of temperature. "
      "All three follow from species enthalpies (NASA-7 polynomials) and the chemical formula. No reaction rates are needed because combustion is assumed complete; "
      "how fast it burns does not change how much heat is released. So the question 'is a methanol kinetic mechanism available and suitable?' does not block the project: "
      "the fallback described in the design document (thermo data and tables, no kinetics) is the primary route. "
      "Consequence for the CV and README: the project uses Cantera for *thermodynamic properties*, not for combustion kinetics.")
    L("")
    L("The design document (section 5.3) also lists *adiabatic flame temperature* as an input to the energy balance. It is **not** needed there: "
      "the exhaust temperature before the turbine follows from the energy balance (enthalpy in = enthalpy out + work + losses), not from the peak flame temperature. "
      "Flame temperatures are computed below for information only.")
    L("")
    L("## 3. Checks")
    L("")
    L("| Check | Result | Detail |")
    L("|---|---|---|")
    # methanol LHV
    dm = abs(meoh.lhv_liquid - METHANOL_LHV_REF) / METHANOL_LHV_REF
    L(check("Methanol LHV (liquid) vs IEA-AMF 20.0 MJ/kg [S04]", dm < 0.01, f"thermo-derived {meoh.lhv_liquid/1e6:.3f} MJ/kg (gas-phase {meoh.lhv_gas/1e6:.3f} minus latent heat); difference {100*dm:.2f} % (reference is rounded to 3 digits)"))
    da = abs(meoh.afr_stoich - METHANOL_AFR_REF) / METHANOL_AFR_REF
    L(check("Methanol stoichiometric air-fuel ratio vs IEA-AMF 6.4 [S04]", da < 0.02, f"computed {meoh.afr_stoich:.3f} kg/kg with O2 = {air.x['O2']:.4f} mol fraction; difference {100*da:.2f} %"))
    for who, val in lhv_nist.items():
        dd = abs(diesel.lhv_liquid - val) / val
        L(check(f"n-dodecane LHV (liquid) vs NIST route ({who}) [S03]", dd < 0.01, f"thermo-derived {diesel.lhv_liquid/1e6:.3f} MJ/kg vs {val/1e6:.3f} MJ/kg from NIST liquid formation enthalpy and the same CO2/H2O data; difference {100*dd:.2f} %"))
    inside = MGO_LHV_RANGE[0] * 0.99 <= diesel.lhv_liquid <= MGO_LHV_RANGE[1] * 1.01
    L(check("n-dodecane surrogate LHV inside marine gas oil range 44-45 MJ/kg [S06]", inside, f"{diesel.lhv_liquid/1e6:.3f} MJ/kg. Real gas oil is a blend, so this is a surrogate, not a match to a specific fuel; source is from 1998"))
    L(check("n-dodecane stoichiometric air-fuel ratio", 14.9 < diesel.afr_stoich < 15.1, f"computed {diesel.afr_stoich:.3f} kg/kg (C12H26 + 18.5 O2); no independent source fetched, arithmetic check only"))
    L(check("Product-gas thermo: gri30.yaml vs nDodecane_Reitz.yaml, effect on exhaust temperature", abs(dT_files) < 0.5, f"switching the product data between the two files shifts a representative exhaust temperature (lambda 2.5, 700 K) by {dT_files:+.3f} K. Species-level difference in h(T)-h(298 K) over 500-1000 K is up to {100*max_rel:.2f} % (worst: {worst_sp}), a small heat-capacity difference between database versions. The project uses gri30.yaml"))
    L(check("Frozen NumPy table vs Cantera h(T)", max_h < 1e-8, f"largest relative error {max_h:.1e} over all species, 250-2500 K"))
    L(check("Frozen NumPy table vs Cantera cp(T)", max_cp < 1e-8, f"largest relative error {max_cp:.1e}"))
    L("")
    L(f"Spread between sources for a 'diesel-like' heating value: MAN quotes its fuel-consumption figures for a fuel LHV of 42.7 MJ/kg [S09]; "
      f"the Danish EPA gives 44-45 MJ/kg for marine gas oil [S06]; the n-dodecane surrogate has {diesel.lhv_liquid/1e6:.1f} MJ/kg, i.e. "
      f"{100*(diesel.lhv_liquid/42.7e6-1):+.1f} % against the MAN figure. In the simulator the fuel flow follows from power and efficiency, "
      "so this only changes the fuel mass (about 2 % of the exhaust mass at rated load); the effect on exhaust temperature is negligible. "
      "The surrogate does not represent any specific fuel.")
    L("")
    L("Note on the n-dodecane fuel enthalpy: the `c12h26` formation enthalpy in the Reitz file is "
      f"{d_gas_vs_nist:.1f} kJ/mol (gas) versus NIST's -290.9 or -288.1 kJ/mol (gas, two studies). Latent heat used: {diesel.latent_heat_molar/1e3:.1f} kJ/mol [S03].")
    L("")
    L("## 4. Information only: adiabatic flame temperatures (equilibrium, HP)")
    L("")
    L("Stoichiometric fuel-air at 298.15 K and 1 atm, **gas-phase fuel**, equilibrium with dissociation.")
    L("")
    L("| Fuel | Mechanism file | T_ad [K] |")
    L("|---|---|---|")
    L(f"| methanol | gri30.yaml | {tad_meoh:.0f} |")
    L(f"| n-dodecane | nDodecane_Reitz.yaml | {tad_dod:.0f} |")
    L("")
    L("These were **not compared with literature values** and are not used by the simulators. Real engine flame temperatures differ (high pressure, preheated air, liquid spray).")
    L("")
    L("## 5. What this means for the fuel difference in exhaust temperature")
    L("")
    L("The design document (section 6.2) says fuels differ in flame temperature, so the same load gives a different exhaust temperature. "
      "That is not the mechanism in an energy-balance model: the peak flame temperature does not set the temperature of the mixed gas before the turbine. "
      "The fuel still matters, through two effects that the simulator contains:")
    L("")
    L("1. **Exhaust composition and mass.** Methanol has a much lower heating value, so about "
      f"{comp['methanol'][0]/comp['diesel_like'][0]:.1f} times as much fuel mass is needed for the same heat input, and it burns to more gas (more water vapour). "
      f"With identical air flow, heat input and heat leaving with the gas, the exhaust is {d_comp_effect:+.1f} K different "
      f"({comp['diesel_like'][1]:.0f} K for the diesel-like fuel, {comp['methanol'][1]:.0f} K for methanol; heat-capacity rate "
      f"{comp['diesel_like'][2]:.2f} vs {comp['methanol'][2]:.2f} kW/K).")
    L("2. **Turbocharger feedback.** The extra gas mass passes through the turbine, which raises boost and air flow, which cools the exhaust a little further. "
      "That part appears only in the full plant solve; its size is reported in the simulator report.")
    L("")
    L("An assumption sits underneath: both fuels are given the same indicated efficiency and wall-loss fraction. Real methanol operation may differ; no source for that was found or used.")
    L("")
    L(f"**Overall result: {'all checks passed' if ok_all else 'AT LEAST ONE CHECK FAILED'}.**")
    OUT.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    if not ok_all:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
