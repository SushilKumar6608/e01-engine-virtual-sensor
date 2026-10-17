PY ?= python3
export PYTHONPATH := src

.PHONY: data data-check data-card test thermo cantera-check calibrate params-doc report

test:            ## run all unit tests
	$(PY) -m pytest

thermo:          ## rebuild frozen thermo table from Cantera
	$(PY) scripts/build_thermo_table.py

cantera-check:   ## Mechanism check -> docs/CANTERA_CHECK.md
	$(PY) scripts/cantera_check.py

calibrate:       ## re-derive all calibrations; add ARGS=--write to store them in config/params.yaml
	$(PY) scripts/calibrate_all.py $(ARGS)

params-doc:      ## regenerate docs/PARAMETERS.md from the registry
	$(PY) scripts/make_params_doc.py

report:          ## Simulator evidence report -> reports/simulator/
	$(PY) scripts/simulator_report.py

data:            ## build the simulated dataset, run the nine gates, release only if all pass
	$(PY) scripts/build_dataset.py

data-check:      ## same, plus a full second rebuild compared by hash
	$(PY) scripts/build_dataset.py --full-check

data-card:       ## regenerate docs/DATA_CARD.md and the data quality report
	$(PY) scripts/make_data_card.py
