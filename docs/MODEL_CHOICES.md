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

## 10d. Evaluation protocol and baselines (pre-registered, written before any model result)
Written on 2026-10-18, before dataset_v2 was generated and before any baseline was run on validation. No test split has been used for any model result.

**Units.** The dataset stores pressures in bar; `NominalModel.predict` expects Pa. The conversion lives in one adapter function with a test (noise-free plant values give identical output through the adapter and through direct SI input).

**Headline metric.**
* Clean splits: error against the *measured* exhaust temperature (the only thing checkable on a real engine). Model selection and tuning use this error only.
* Error against the *true* temperature is reported beside it, for diagnosis only, never for choosing a model.
* Fault split: the sensor is deliberately wrong, so the measured value is the wrong reference. Models take no exhaust-sensor input, so their predictions cannot change with the fault; the headline there is detection delay (by fault type) and false-alarm rate on the unfaulted episodes, with the true-temperature error beside it. The residual against the measured value is the detector's signal.
* Noise floor row on every split: RMS difference between measured and true. On dataset_v1 validation it was 1.50 K; no input-based model can do better than about that against the measured value (part of it is a per-episode sensor bias).
* Dropouts: rows with a missing measured value leave the measured metric and stay in the true metric; both row counts are printed. In the clean splits the exhaust target is never missing.
* Confidence intervals come from resampling episodes, never rows. Metrics: RMSE, MAE and bias per split, fuel and load bin.
* Prediction intervals are calibrated on the measured target. Coverage of the measured value is the headline; coverage of the true value is shown beside it (expected to be slightly conservative). Coverage is also reported per episode (share of episodes below 80 % of nominal coverage, worst decile), because sensor bias is constant within an episode.
* The harness refuses `test_*` and `stress_*` splits unless unsealed explicitly; each unseal is logged in the repository. Validation only until the model-selection gate.

**Baselines (all fitted on training episodes only).**
* Per-fuel constant: the floor.
* Load-only lookup: load = fuel flow x lower heating value over the rated value for that fuel (not a measured signal). 20 quantile bins per fuel, linear interpolation between bin centres, clamped outside the training range. Fitted on the measured target with missing rows dropped.
* M0: air delivery ratio refitted on the same training episodes as M0b; exhaust share from the rated heat balance. The healthy-grid fit remains only as the labelled commissioning baseline.
* M0b: delivery ratio = a + b * (p_scav / p_amb); exhaust share fixed from the rated heat balance; only a and b are fitted, to the measured exhaust temperature with dropout rows removed, on every 10th second (for speed; autocorrelation does not bias the point estimate, the episode bootstrap handles the intervals). Identifiability is checked with several starting points and an episode-level bootstrap; parameter intervals are reported.
* M0b is cross-fitted for the features stage: 5 episode-grouped folds on train give out-of-fold predictions; validation, calibration and test use the model fitted on all of train.

**Expected ordering, written down before running (validation, RMSE against measured, per fuel):** per-fuel constant worst, then load-only lookup, then M0, then M0b best. If the order differs, the defined checks come first (unit test, fit convergence, residual signs); the result is then reported as it is. Nothing is tuned until the expected order appears.

**Lag is not a fairness problem.** On dataset_v1 validation the true exhaust temperature differs from its lag-free steady-state value by 1.00 K RMS, against baseline errors of about 11 K. The harness also reports the 99th percentile, since ramps are worse than the average.

**Fouling regime (dataset_v2).** The two Beta draws per episode are kept and rescaled into a per-split range, so every other random draw is identical to v1.
* All splits except the high-fouling test: start in [0, 0.45], hard cap 0.50 after drift (turbocharger and air cooler each).
* `test_regime_high_fouling`: 240 episodes (builds with 60, 120 and 180 failed the coverage gate with 4, 1 and 1 of 46 methanol bins under-populated; the last two holes were the slowest, lowest-fuel corner bin, visited by only 2 of 66 methanol episodes against the required 3; episodes were added, thresholds unchanged), two cohorts by episode index, 120 each. `tc_high`: turbocharger starts in [0.55, 0.75], cooler capped. `cooler_high`: cooler starts in [0.55, 0.75], turbocharger capped. Reported separately.
* Expected: M0 degrades most in `tc_high` (negative bias; on dataset_v1 validation its bias went from +4.9 K clean to -10.2 K for turbocharger fouling 0.5 to 0.65) and shows positive bias in `cooler_high` (+5.8 K for cooler fouling 0.5 to 0.65).
* Cold ambient stays, labelled "weak shift, expected to pass", as a negative control.
* The fouling effect size is assumed, not sourced: this split tests extrapolation within the simulator's law, not real-engine behaviour.
* Decision history: made after seeing M0's residuals by fouling on a validation sample (design coverage of the main hidden state); no learned model had been evaluated.

## 10e. Baseline result on validation (simulated data; test and stress splits untouched)
Run once with `scripts/run_baselines_validation.py`; numbers in `reports/baselines/validation_summary.md`.

* RMSE against measured, all validation rows: constant 22.95 K, load lookup 18.56 K, M0 10.47 K, M0b 9.57 K. Noise floor 1.50 K.
* **The pre-registered ordering holds for diesel-like (25.58 > 19.68 > 11.34 > 9.77 K) but not for methanol:** M0 8.51 K [7.79, 9.25] beats M0b 9.18 K [8.46, 9.89] (intervals overlap).
* Defined checks done: unit adapter test passes (bar passed as Pa fails loudly); M0b converged to the same parameters from three starts (a = 0.950, b = 0.029); episode bootstrap gives a in [0.937, 0.967] and b in [0.022, 0.035], so both are identifiable.
* Residual signs: M0b has a bias of about -9.7 K at load above 0.85 for both fuels, M0 about -3.4 K. A delivery ratio that is linear in boost, shared by both fuels, fits the middle of the range and misses the high-load end. This explanation is a hypothesis, not tested.
* Nothing was tuned. M0b stays the headline baseline as pre-registered (best overall); M0 stays as reference and is reported per fuel. The methanol result is reported as it is.

## 10f. Features A-D, M1 and M2 on validation (simulated data; test and stress splits untouched)
Run with `scripts/run_ablation_validation.py`; numbers in `reports/models/ablation_summary.md`. Fitted on every 5th second of train; scalers on train only; M2 settings chosen from a 4-point grid on validation (all four are listed in `reports/models/ablation_log.json`). B-D predict the residual of the cross-fitted M0b estimate; A has no nominal estimate and predicts the measured temperature directly, so A-to-B changes two things at once (physics features and the residual set-up).

* RMSE against measured, all validation rows: M0 10.47 K, M0b 9.57 K; M2 A 4.52, B 3.96, C 3.69, D 3.69; M1 A 5.19, B 3.91, C 3.61, D 3.61. Noise floor 1.50 K. The stage gate "M2 beats M0 on validation" is met.
* **M2 does not beat the ridge model M1** (C: 3.69 against 3.61, intervals overlap). The gradient-boosted model is the planned main model, but on this data the simple model is as good. Reported as it is.
* Raw signals alone (A) already halve the baseline error. The nominal estimate and physics features (B) improve it further (M2 4.52 to 3.96, intervals overlap slightly).
* Dynamic features (C) help on ramps, as the 12 s exhaust lag predicts: on the 10 % of rows with the steepest fuel-flow change, M1 goes from 4.85 K (B) to 3.73 K (C). The 10 s large-change rule of `since_change` fires on only 60 validation rows in 1 episode, so it is not used for reporting; the ramp definition (top 10 % of fuel-flow slope over 30 s, threshold fixed on train) replaced it.
* **Flags (D) change nothing here.** The out-of-range and stuck flags are never set in the clean splits, and the dropout flag affects only the forward-filled turbocharger speed. Their value can only be judged on the fault split (abstain rule, stages 4 and 5); until then D has not earned its place.
* Monotonic constraints (design 8.1) are not applied: the model predicts a residual, where a physical monotonic constraint does not carry over. Open item.
* Permutation importance (M2-C) is led by heat per revolution (4.9 K), engine speed (3.0 K) and the fuel-flow history (1.2 and 0.9 K). That is physically plausible; the nominal estimate matters less (0.6 K). The hidden fouling states are not observable, so the remaining 3.6 K against a 1.5 K noise floor cannot be removed by these features. This explanation is a hypothesis.
* Missing turbocharger-speed values (338 rows in validation) are forward-filled causally; leading gaps use the train median.

## 11. Parameter registry as a gate
* Every number lives in `config/params.yaml` with a status. A test fails if a number used in code is missing from the registry, if a registered number is unused, or if a calibrated value is stale.
* 90 parameters: 15 sourced, 1 derived, 10 calibrated, 64 assumed (`docs/PARAMETERS.md`).

## What I could not verify (open)
1. The stroke (assumed 2.2 m; implied mean effective pressure 20 bar, same order as a larger engine's 21 bar).
2. The load dependence of exhaust temperature before the turbine (model: nearly flat from 50 to 100 %; no source found).
3. Sizes of ambient and fouling effects (directions are consistent with MAN's qualitative statement for ambient; fouling signs are model outputs).
4. Everything below about 40 % load (only used as an abstention stress test).
5. S01 was read through a text summary, not figure by figure; its plot ranges are indicative.
