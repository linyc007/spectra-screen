.PHONY: benchmarks test lint

# Regenerate every number quoted in the README (Tecator + Indian Pines).
benchmarks:
	python -u benchmarks/run_tecator.py
	python -u benchmarks/run_indian_pines.py

# Optional, needs a working TabPFN install (see HPC note in README):
tabpfn:
	python -u benchmarks/run_tabpfn_tecator.py

test:
	pytest -q

lint:
	ruff check .
