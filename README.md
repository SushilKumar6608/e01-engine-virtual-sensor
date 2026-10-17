# e01-engine-virtual-sensor

**All data in this repository is simulated. Nothing is validated against a real engine. The simulator is a simplified mean-value model with generic parameters; it is not a model of any specific commercial engine.**

Goal: an uncertainty-aware virtual sensor that estimates the exhaust gas temperature before the turbine of a large two-stroke marine-style engine from cheap signals, with calibrated prediction intervals and abstention outside the training range. Design: see the project design document (v2). Reference structure: Alegret et al. (2015).

## Status

Done so far: plant and nominal simulators, parameter registry, Cantera check, simulated dataset with nine quality gates (`docs/DATA_CARD.md`, `reports/data/data_quality_report.md`). Simulator: `reports/simulator/simulator_report.md`.

**Wording rule:** results here "rest on the simulator". They are never described as resting on verified physics. Operating range of the study: 40-100 % load; 30-35 % load data are an abstention stress test only.

## Layout

* `config/params.yaml`: every parameter, with source or "assumed" rationale. `docs/PARAMETERS.md` is generated from it.
* `src/vsensor/sim/`: `thermo.py`, `fuels.py`, `plant.py` (ground truth), `nominal.py` (engineer's model).
* `src/vsensor/data/`: dataset configuration, episode generator, hashing, the nine quality gates. Config: `config/dataset.yaml`.
* `src/vsensor/calibration.py`: the three calibrations; `tests/test_calibration.py` fails if stored values go stale.
* `scripts/`: `cantera_check.py`, `calibrate_all.py`, `simulator_report.py`, `build_dataset.py`, `make_data_card.py`, `build_thermo_table.py`, `make_params_doc.py`.
* `docs/`: `MODEL_CHOICES.md` (plain-language reasons), `CANTERA_CHECK.md`, `PARAMETERS.md`.

## Run

    pip install -e ".[dev]"
    make test          # 85 unit tests
    make cantera-check
    make report

Tested with Python 3.13; versions are pinned in `pyproject.toml`.

## Limitations

See the last sections of `docs/MODEL_CHOICES.md` and the simulator report. In short: low load (below about 40 %) is outside the study range, part-load exhaust temperature could not be checked against a source, and several magnitudes are assumptions.
