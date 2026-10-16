PY ?= python3
export PYTHONPATH := src

.PHONY: test thermo cantera-check calibrate params-doc report

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
