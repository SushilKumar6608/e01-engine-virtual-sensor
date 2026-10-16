# Parameter registry

**All values describe a simulated, generic engine. Nothing is validated against a real engine.** Generated from `config/params.yaml` by `scripts/make_params_doc.py`; do not edit by hand.

46 parameters: 15 sourced, 1 derived, 10 calibrated, 20 assumed. Registry hash `7cbe44d3df36cbc2`.

| Status | Meaning |
|---|---|
| sourced | taken from a cited source |
| derived | computed from sourced numbers |
| calibrated | tuned to hit a stated target; a test re-runs the calibration and fails if the stored value is stale |
| assumed | my own engineering assumption, with a written rationale; **not verified** |

## Sources and how each was checked

**S01** - Alegret, Llamas, Vejlgaard-Laursen, Eriksson (2015). Modeling of a Large Marine Two-Stroke Diesel Engine with Cylinder Bypass Valve and EGR System. IFAC MCMC. <https://www.fs.isy.liu.se/Publications/Articles/MCMC_15_GA_XL_MV_LE.pdf>
  - *Verification:* Read via a web-fetch text summary (not the figures themselves). Rated power and speed are stated in the text. The plot axis ranges (scavenge pressure 2-4 bar, exhaust temperature 600-800 K, turbocharger speed 1400-1800 rad/s) were read off one figure (dataset 11) and are INDICATIVE only. The paper does not give bore, stroke, cylinder count, or any rated-point state values.

**S02** - The Motorship (2019), "MAN Energy Solutions to develop second test engine in 2019": new engine has the same dimensions as the existing 4T50ME-X, 500 mm bore, 4 cylinders. <https://motorship.com/man-energy-solutions-to-develop-second-test-engine-in-2019/1156922.article>
  - *Verification:* Read via a web-fetch text summary. Stroke and power are NOT stated there.

**S03** - NIST Chemistry WebBook, n-dodecane (C12H26): enthalpy of vaporisation about 61 kJ/mol (61.4 kJ/mol at 299 K); liquid formation enthalpy -352.1 or -349.3 kJ/mol depending on the cited study. <https://webbook.nist.gov/cgi/cbook.cgi?ID=C112403&Mask=FFF>
  - *Verification:* Read via a web-fetch text summary. My earlier recollection (about 77 kJ/mol) was WRONG; the NIST value is used.

**S04** - IEA Advanced Motor Fuels, methanol fuel properties: LHV 20.0 MJ/kg, stoichiometric air-fuel ratio 6.4, heat of vaporisation 1160-1174 kJ/kg. <https://iea-amf.org/content/fuel_information/methanol/fuel_properties>
  - *Verification:* Read via a web-fetch text summary. Page does not state the temperature of the heat of vaporisation or the unit of the stoichiometric ratio (kg/kg assumed).

**S05** - Cantera 3.2.0 shipped data files gri30.yaml (CO2, H2O, N2, O2, Ar, CH3OH) and nDodecane_Reitz.yaml (c12h26): NASA-7 thermodynamic polynomials. <https://cantera.org>
  - *Verification:* Extracted by scripts/build_thermo_table.py; NumPy evaluation checked against Cantera's own h(T) (tests/test_thermo_fuels.py). Only thermodynamics is used.

**S06** - Danish EPA (1998), marine fuels chapter: marine gas oil lower calorific value "normally from 44000 to 45000 kJ/kg"; marine diesel oil about 44000 kJ/kg. <https://www2.mst.dk/udgiv/Publications/1998/87-7909-173-3/html/kap04_eng.htm>
  - *Verification:* Read via a web-fetch text summary. Old (1998) source; used only to judge the n-dodecane surrogate, not as a model input.

**S07** - Standard dry-air composition (O2 20.95 %, N2 78.08 %, Ar 0.93 % by mole).
  - *Verification:* Textbook value recalled, NOT fetched in this project. Low risk, but unverified.

**S08** - MAN Energy Solutions, "Influence of ambient temperature conditions" (document 5510-0074-03). Table 5: ISO 3046-1:2002 reference = 1000 mbar, 25 C turbocharger air intake, 25 C scavenge-air cooling water. States the scavenge air cooler is designed for a maximum 12 C difference between cooling-water inlet and scavenge air outlet at 100 % SMCR, and that at 15-30 % load scavenge air is only slightly above the cooling-water temperature. Qualitative: exhaust temperature after the turbine drops in winter. <https://www.man-es.com/docs/default-source/marine/tools/5510-0074-03.pdf?sfvrsn=fa9e1cee_7>
  - *Verification:* Read via a web-fetch text summary. No numeric sensitivities of exhaust or scavenge temperature to ambient temperature are given there, so the SIZE of the ambient effect in the simulator is NOT verified; only the direction is qualitatively consistent.

**S09** - MAN B&W G95ME-C9.5 Project Guide (2015 copy): ISO reference conditions 1000 mbar, 25 C air, 25 C scavenge-air coolant; specific fuel consumption stated for a fuel lower calorific value of 42,700 kJ/kg; scavenge air cooler designed for about 12 C difference at specified MCR. <https://www.dieselduck.info/machine/01%20prime%20movers/2015%20MAN%20G95ME-C9%20Project%20Guide.pdf>
  - *Verification:* Read via a web-fetch text summary of an excerpt (third-party copy). The exhaust-gas data chapter was not in the excerpt.

**S10** - MAN B&W G60ME-C9.5 Project Guide data excerpt (Fig. 1.03.01): specific fuel oil consumption, high-load tuning, 165.5 g/kWh at 50 %, 163.0 at 75 %, 167.0 at 100 % load; stroke 2790 mm; mean effective pressure 21.0 bar at L1. <https://s2b4aa72b8b1c7323.jimcontent.com/download/version/1630830691/module/11673801091/name/NTS-MAN%20Burmeister%20&%20Wain%206G60ME-C%20HPSCR%20MK9.5%20Project%20Guide%20Data_compressed.pdf>
  - *Verification:* Read via a web-fetch text summary of an excerpt (third-party copy). This is a DIFFERENT, larger engine (60 cm bore); it is used only for the SHAPE of the efficiency-versus-load curve and as a mean-effective-pressure sanity check, never as engine data for the 4T50ME-X. Whether the percentages refer to the propeller law was not stated in the summary; the propeller law is assumed.

## Sourced parameters

| Id | Value | Unit | What it is | Evidence |
|---|---|---|---|---|
| `air.x_ar` | 0.0093 | - | Mole fraction of Ar in dry air | source S07 |
| `air.x_n2` | 0.7808 | - | Mole fraction of N2 in dry air | source S07 |
| `air.x_o2` | 0.2095 | - | Mole fraction of O2 in dry air | source S07 |
| `calib.sfoc.lhv` | 4.27e+07 | J kg-1 | Fuel lower calorific value on which published MAN fuel consumption figures are based | source S09 |
| `calib.sfoc.load_100` | 167 | g kWh-1 | Published specific fuel consumption at 100 percent load (G60ME-C9.5, high-load tuning) | source S10 |
| `calib.sfoc.load_50` | 165.5 | g kWh-1 | Published specific fuel consumption at 50 percent load (G60ME-C9.5, high-load tuning) | source S10 |
| `calib.sfoc.load_75` | 163 | g kWh-1 | Published specific fuel consumption at 75 percent load (G60ME-C9.5, high-load tuning) | source S10 |
| `calib.target.cooler_dt_rated` | 12 | K | Scavenge air temperature minus cooling-water temperature at rated point, clean cooler | source S08 |
| `engine.bore` | 0.5 | m | Cylinder bore | source S02 |
| `engine.n_cyl` | 4 | - | Number of cylinders | source S02 |
| `engine.rated_power` | 7.08e+06 | W | Rated brake power | source S01 |
| `engine.rated_speed` | 123 | rpm | Rated engine speed | source S01 |
| `env.iso_p_amb` | 100000 | Pa | Reference ambient pressure (1000 mbar, ISO 3046-1) | source S08 |
| `env.iso_t_amb` | 298.15 | K | Reference ambient air temperature (25 C, ISO 3046-1) | source S08 |
| `fuel.diesel_like.latent_heat` | 61400 | J mol-1 | Heat of vaporisation of n-dodecane near 298 K | source S03 |

## Derived parameters

| Id | Value | Unit | What it is | Evidence |
|---|---|---|---|---|
| `fuel.methanol.latent_heat` | 37390 | J mol-1 | Heat of vaporisation of methanol near 298 K | derived from S04 |

## Calibrated parameters

| Id | Value | Unit | What it is | Evidence |
|---|---|---|---|---|
| `cooler.eps` | 0.924086 | - | Clean air-cooler effectiveness | Set by scripts/calibrate_rated.py so that, at the rated clean ISO point, scavenge air leaves the cooler 12 K above the cooling-water temperature (calib.target.cooler_dt_rated, S08/S09). Enforced by tests/test_calibration.py. |
| `cooler.k_loss` | 42.8899 | Pa s2 kg-1 m-3 | Cooler pressure-loss coefficient (dp = k * m_air^2 / density) | Set by scripts/calibrate_rated.py so clean-cooler pressure loss at the rated point equals calib.target.cooler_dp_rated. Enforced by tests/test_calibration.py. |
| `engine.eta_ind_curv` | 0.235576 | - | Curvature of indicated efficiency versus load (parabola) | Fitted with eta_ind_max and eta_ind_x_opt to the published fuel-consumption points (see engine.eta_ind_max). The parabola is only constrained between 50 and 100 percent load; below that it is an extrapolation. Enforced by tests/test_calibration.py. |
| `engine.eta_ind_max` | 0.547449 | - | Peak gross indicated efficiency (work on piston / fuel energy) | With eta_ind_curv and eta_ind_x_opt, fitted so brake efficiency at 50/75/100 percent load on the propeller law equals the published fuel-consumption points (calib.sfoc.*, S10, converted with calib.sfoc.lhv, S09). Exact 3-point fit by scripts/calibrate_efficiency.py. Enforced by tests/test_calibration.py. |
| `engine.eta_ind_x_opt` | 0.607463 | - | Relative fuel-energy rate at which indicated efficiency peaks | Fitted with eta_ind_max and eta_ind_curv to the published fuel-consumption points (see engine.eta_ind_max). Enforced by tests/test_calibration.py. |
| `nominal.lambda_del` | 0.987036 | - | Nominal model air delivery ratio (constant) | One-parameter least-squares fit of exhaust temperature on the healthy ISO diesel-like plant: load 30-100 percent at three speed offsets from the propeller law (scripts/calibrate_nominal.py). Enforced by tests/test_calibration.py. |
| `nominal.phi_exhaust` | 0.422423 | - | Nominal model share of fuel energy that leaves with the exhaust gas | Closed form from the rated-point heat balance (what a shop test would give): 1 - indicated efficiency at rated fuel-energy rate - wall-loss fraction (scripts/calibrate_nominal.py). Not fitted by least squares, because it is not separately identifiable from the delivery ratio. Enforced by tests/test_calibration.py. |
| `scav.lambda_coeff` | 2.26877 | - | Air delivery ratio coefficient (delivery ratio = coeff * sqrt(1 - p_exh/p_scav)) | Set by scripts/calibrate_rated.py so rated clean ISO exhaust temperature hits calib.target.t_exh_rated. Enforced by tests/test_calibration.py. |
| `tc.d_comp` | 0.58283 | m | Compressor wheel diameter (sets turbocharger speed) | Set by scripts/calibrate_rated.py so rated clean turbocharger speed equals calib.target.n_tc_rated. Enforced by tests/test_calibration.py. |
| `tc.k_turbine` | 0.00136701 | kg K0.5 s-1 Pa-1 | Turbine flow coefficient (Stodola-type law, m*sqrt(T) = k*sqrt(p_in^2 - p_out^2)) | Set by scripts/calibrate_rated.py so rated clean scavenge pressure equals calib.target.p_scav_rated. Enforced by tests/test_calibration.py. |

## Assumed parameters

| Id | Value | Unit | What it is | Evidence |
|---|---|---|---|---|
| `calib.target.cooler_dp_rated` | 3000 | Pa | Target clean air-cooler pressure loss at rated point | A few tens of mbar is typical for a large cooler (recalled, not verified). |
| `calib.target.n_tc_rated` | 16200 | rpm | Target turbocharger speed at rated point, clean, ISO | About 1700 rad/s, inside the 1400-1800 rad/s axis range of S01 (indicative). Used only to set the compressor diameter. |
| `calib.target.p_scav_rated` | 360000 | Pa | Target scavenge pressure at rated point, clean, ISO | Inside the 2-4 bar axis range of S01 (indicative) and near the top of it, as expected at full load. Used only to set the turbine flow coefficient. |
| `calib.target.t_exh_rated` | 700 | K | Target exhaust temperature before the turbine at rated point, clean, ISO, diesel-like | Middle of the 600-800 K range plotted in S01 (indicative, read off a plot axis). Used only to set the air-delivery coefficient. |
| `cooler.foul_dp_gain` | 1 | - | Relative increase of cooler pressure loss at full fouling | Fouled coolers restrict flow; magnitude is a placeholder, no cited source. |
| `cooler.foul_eps_loss` | 0.25 | - | Relative loss of cooler effectiveness at full fouling (fouling = 1) | Sets the size of the hidden health state. Chosen large enough to be visible in scavenge temperature; no cited source for real fouling rates. |
| `cooler.t_coolant_offset` | 0 | K | Charge-air coolant temperature minus ambient air temperature | Simplification: coolant (sea water) tracks ambient air, like the ISO reference condition where both are 25 C. Real sea and air temperatures differ. |
| `engine.f_wall` | 0.05 | - | Fraction of fuel energy lost through cylinder walls, jacket and piston cooling | Generic heat-balance share (a few percent). Constant in the plant. Not verified in a cited source. |
| `engine.fmep` | 90000 | Pa | Friction mean effective pressure (constant) | Generic order of magnitude (about 1 bar) for a large slow-speed engine. Friction power then scales with speed. Not verified in a cited source. |
| `engine.stroke` | 2.2 | m | Piston stroke | Not found in any cited source. 2.2 m is my recollection of the usual stroke for 50-cm-bore ME-type engines. It only scales displacement. Consistency check (simulator report): it implies a rated brake mean effective pressure of about 20 bar, against 21.0 bar quoted at L1 for the larger G60ME-C9.5 [S10] - same class, different engine, so this supports the order of magnitude only. |
| `tc.cp_air` | 1005 | J kg-1 K-1 | Air specific heat used for the compressor (constant) | Textbook room-temperature value; actual air cp rises a little with temperature. |
| `tc.eta_c` | 0.84 | - | Compressor isentropic efficiency (clean, constant) | Typical for large turbochargers; constant for simplicity (no compressor map). |
| `tc.eta_mech` | 0.98 | - | Turbocharger mechanical efficiency | Generic bearing-loss figure. |
| `tc.eta_t` | 0.85 | - | Turbine efficiency (clean, constant) | Typical for large turbochargers; constant for simplicity (no turbine map). |
| `tc.foul_eta_c_loss` | 0.05 | - | Relative loss of compressor efficiency at full turbocharger fouling | Size of the hidden health state; placeholder magnitude, no cited source. |
| `tc.foul_eta_t_loss` | 0.07 | - | Relative loss of turbine efficiency at full turbocharger fouling | Size of the hidden health state; placeholder magnitude, no cited source. |
| `tc.foul_k_loss` | 0.1 | - | Relative loss of turbine flow capacity at full turbocharger fouling | Nozzle-ring fouling narrows the gas passage; placeholder magnitude. |
| `tc.kappa_air` | 1.4 | - | Air heat-capacity ratio used for the compressor (constant) | Textbook value for air. |
| `tc.p_back_ratio` | 1.02 | - | Pressure after the turbine divided by ambient pressure | Small exhaust-duct / silencer loss; generic. |
| `tc.psi` | 0.65 | - | Compressor work coefficient (specific work = psi * tip speed^2) | Typical order for backward-swept centrifugal compressors (about 0.6-0.7); sets the link between turbocharger speed and pressure rise. |

