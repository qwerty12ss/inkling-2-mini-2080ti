PYTHON ?= python3

.PHONY: check charts
check:
	$(PYTHON) -m unittest discover -s tests -v
	$(PYTHON) benchmarks/check_data.py
	bash -n scripts/serve.sh
	bash -n scripts/fetch-weights.sh
	bash scripts/serve.sh text-512k --dry-run >/dev/null
	bash scripts/serve.sh vision-448k --dry-run >/dev/null

charts:
	$(PYTHON) benchmarks/plot.py
