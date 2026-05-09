PY := .venv/bin/python
PIP := .venv/bin/pip

.PHONY: setup install forecast backtest evaluate report clean clean-data clean-report freeze help

help:
	@echo "Targets:"
	@echo "  setup        Create .venv (Python 3.12) and install requirements"
	@echo "  install      Install/refresh requirements into existing .venv"
	@echo "  forecast     Run run_forecast.py -> predictions.csv (May 11 window)"
	@echo "  backtest     Run backtest_may10.py against May 9 actuals"
	@echo "  evaluate     Run evaluate.py against the live API"
	@echo "  report       Compile report/report.tex -> report/report.pdf"
	@echo "  freeze       Write fully-pinned versions to requirements.lock"
	@echo "  clean        Remove caches (__pycache__, .ipynb_checkpoints)"
	@echo "  clean-data   Remove cached data CSVs (forces re-download)"
	@echo "  clean-report Remove LaTeX build artifacts in report/"

setup:
	/opt/homebrew/bin/python3.12 -m venv .venv
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt

install:
	$(PIP) install -r requirements.txt

forecast:
	$(PY) run_forecast.py

backtest:
	$(PY) backtest_may10.py

evaluate:
	$(PY) evaluate.py

report:
	cd report && latexmk -pdf -interaction=nonstopmode -halt-on-error report.tex

freeze:
	$(PIP) freeze > requirements.lock

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type d -name .ipynb_checkpoints -exec rm -rf {} +

clean-data:
	rm -f data/*.csv

clean-report:
	cd report && latexmk -C
