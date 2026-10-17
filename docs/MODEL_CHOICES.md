# Modelling choices, in plain language

**All data in this project is simulated. Nothing is validated against a real engine.**
For each choice: what it is, why, what I rejected, a sentence you can say in an interview, and how well it is backed.
Evidence labels: **sourced** (a cited source), **calibrated** (tuned to a stated target), **assumed** (my engineering judgement, not verified).

## 1. Two simulators: a "plant" and a "nominal" model
* **What:** the plant (`sim/plant.py`) generates the "true" behaviour and the target. The nominal model (`sim/nominal.py`) is what an engineer would have: a simple heat balance for a healthy engine.
* **Why:** if I learn a simulator I wrote myself, the result is circular. The plant has hidden health states and effects the nominal model lacks, so there is a real gap to learn.
* **Interview line:** "The learning problem is the gap between a simple engineering model and a richer simulated plant, not the plant itself."
* **Honest limit:** both are mine, so the gap is by construction. It says nothing about the real-engine gap.

## 2. Exhaust temperature from an energy balance
* **What:** fuel energy minus piston work minus wall loss is what must leave with the exhaust. Adding the incoming air's heat and dividing by how much heat the gas stores per kelvin gives the temperature.
* **Why:** it is the simplest physics that gets the target right and respects conservation of energy.
* **Rejected:** using flame temperature (the design document suggested it). The peak flame temperature does not set the mixed-gas temperature before the turbine. Fuels differ through exhaust composition and mass instead (`docs/CANTERA_CHECK.md`, section 5).
* **Evidence:** the balance closes to within 0.2 % of fuel energy in a test that recomputes it from outputs. Physics: standard.

## 3. Cantera for thermodynamics only
* **What:** Cantera's data files supply the heat content of each gas versus temperature, the formation energies that give the heating values, and the chemistry bookkeeping. No reaction kinetics.
* **Why:** combustion is assumed complete; how fast it burns does not change how much heat appears. Cantera ships a kinetic mechanism for n-dodecane but none dedicated to methanol, which would have blocked a kinetic plan and does not block this one.
* **Interview line:** "I used Cantera for thermodynamic properties, and checked the heating values against independent sources. I did not use it for combustion kinetics."
* **Evidence:** LHV methanol 19.94 vs 20.0 MJ/kg (IEA), n-dodecane 44.10 vs 44.11 MJ/kg (NIST route), all in `CANTERA_CHECK.md`. The thermo data is frozen into a JSON table so the runtime does not need Cantera.

## 4. Fuels: n-dodecane stands in for diesel, plus methanol
* **Why:** a single molecule has a well-defined formula, stoichiometry and heating value. Real marine gas oil is a blend; sources quote 42.7 to 45 MJ/kg, the surrogate has 44.1.
* **Latent heat:** the thermo data describe the fuel as a gas; the engine injects a liquid. I subtract the latent heat (NIST: 61.4 kJ/mol for n-dodecane; IEA: about 1167 kJ/kg for methanol). My first recollection of 77 kJ/mol was wrong; the source check caught it.
* **Assumption:** both fuels share the same efficiency curve. Real methanol operation may differ; no source used.

## 5. Efficiency curve calibrated to published fuel consumption
* **What:** a parabola for gross indicated efficiency versus load plus friction. Three parameters fit exactly three published points (50/75/100 % load: 165.5/163.0/167.0 g/kWh, MAN G60ME-C9.5).
* **Why:** it fixes the shape from data instead of from my intuition.
* **Limits:** a different, larger engine; I assumed the percentages follow the propeller law; below 50 % load it is an extrapolation. **Calibrated, not independent validation.**

## 6. Air flow: delivery ratio that falls as exhaust back-pressure rises
* **What:** air per cycle = coefficient x sqrt(1 - exhaust pressure / scavenge pressure) x air density x swept volume.
* **Why:** a bigger pressure drop from scavenge to exhaust pushes more air through the cylinders (orifice-flow idea). It lets turbocharger fouling reduce air flow, which is physically the point.
* **Honest limit:** the square-root form and its strength are **assumed**. Air flow responds strongly to back-pressure in this model; I have no source for that sensitivity.

## 7. Turbocharger without maps
* **What:** constant compressor, turbine and bearing efficiencies; a Stodola-type law for turbine flow (the reference paper uses a flow law of similar form); turbocharger speed from the Euler relation (specific work = work coefficient x tip speed squared).
* **Why:** maps for a specific turbocharger are not available. Constant efficiencies keep every line explainable.
* **Calibrated:** turbine flow coefficient (to hit 3.6 bar at rated), compressor diameter (16 200 rpm), air coefficient (700 K), cooler loss (3 kPa). **Assumed:** the targets themselves, anchored in the indicative plot ranges of the reference paper.
* **Rejected:** pulse turbocharging, auxiliary blowers, bypass valve and EGR.

## 8. Air cooler
* **What:** effectiveness model with a small pressure loss. Effectiveness is calibrated so scavenge air leaves 12 K above cooling water at rated load, matching MAN's stated design figure (**sourced**).
* **Assumed:** cooling water tracks ambient air temperature (true at the ISO reference only).

## 9. Hidden health states
* **What:** turbocharger fouling and air-cooler fouling, each 0 to 1. The sizes (5 %, 7 %, 10 % efficiency/flow loss; 25 % cooler effectiveness loss) are **assumed** placeholders.
* **Why:** known hidden states are what lets me later test whether intervals are calibrated and whether fault detection works. Models never see them.

## 10. The nominal model's two constants (M0), the stronger baseline (M0b) and the load-only baseline
* **Operating range:** 40-100 % load for training, calibration and the in-distribution test. There is no blower model, so below about 40 % the simulator is not trusted. 30-35 % load data exist only as an *abstention stress test*; no accuracy or coverage is reported there.
* **M0 (reference):** the share of fuel energy leaving with the exhaust comes from the rated heat balance (closed form). The air delivery ratio is a single constant fitted by least squares. Not both by least squares: they are not identifiable (more air with a larger exhaust share gives nearly the same temperature).
* **M0 as "commissioning baseline":** fitted once on the healthy ISO plant grid, 40-100 % load, three speed offsets (39 points): RMSE about 14 K, maximum about 33 K.
* **M0b (headline baseline, form fixed before any test result exists):** same energy balance, but the delivery ratio uses the *measured* boost: lambda = a + b * (p_scav / p_amb), at most 2 parameters, registered as `m0b.*` (status calibrated). M0 and M0b are fitted on the **same training episodes**, never on validation or test data. The healthy-grid fit is kept only under the label "commissioning baseline".
* **Load-only lookup baseline:** exhaust temperature as a function of load fraction only (binned lookup per fuel), fitted on the training episodes. It shows how much of the signal is just "how hard the engine runs".
* **Honest caveat:** M0b may be stronger than a real engineer's model because it is fitted on many operating points of the same simulator. If M2 does not clearly beat M0b, that is reported as is.
* **Abstention pre-registration:** the expected abstain rate is written down per load bin (35 % and 30 % separately) before the stress data are evaluated.
* **Wording rule:** results "rest on the simulator"; never "on verified physics" (README, CV, interviews).

## 10b. Methanol efficiency (shared curve) and the stress test
* **Decision:** in the main data methanol shares the diesel efficiency curve (assumed; a 30-minute source search found nothing usable). The plant has a methanol efficiency scale `engine.eta_scale_methanol`, 1.0 in the main data.
* **Stress test (pre-specified, plant only):** methanol efficiency -3 % and +3 % (splits `stress_methanol_efficiency_minus` / `_plus`).
* **What I measured:** the effect on exhaust temperature reaches 7.6 K at full load, not 4 K or less. It changes sign near 45 % load because a fixed-air effect of about 25-29 K is cancelled by 7-9 % air-flow feedback. That feedback depends on the *unsourced* air-flow sensitivity, so the sign flip is a property of the model, not a finding about real engines.

## 10c. Ensemble (M4)
* Out of the main build. Re-entry rule: decided at the model-selection gate using validation data only. If it comes in: a 5-model LightGBM bag whose spread is conformalised, evaluated on a fresh test set (new `base_seed`). Documented as "scoped, with a re-entry rule".

## 11. Parameter registry as a gate
* Every number lives in `config/params.yaml` with a status. A test fails if a number used in code is missing from the registry, if a registered number is unused, or if a calibrated value is stale.
* 90 parameters: 15 sourced, 1 derived, 10 calibrated, 64 assumed (`docs/PARAMETERS.md`).

## What I could not verify (open)
1. The stroke (assumed 2.2 m; implied mean effective pressure 20 bar, same order as a larger engine's 21 bar).
2. The load dependence of exhaust temperature before the turbine (model: nearly flat from 50 to 100 %; no source found).
3. Sizes of ambient and fouling effects (directions are consistent with MAN's qualitative statement for ambient; fouling signs are model outputs).
4. Everything below about 40 % load (only used as an abstention stress test).
5. S01 was read through a text summary, not figure by figure; its plot ranges are indicative.
