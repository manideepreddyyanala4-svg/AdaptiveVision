.PHONY: install lint format format-check typecheck test run check train-deps

install:
	pip install -e ".[dev]"

train-deps:
	pip install -r training/requirements.txt

lint:
	ruff check .

format:
	ruff format .

format-check:
	ruff format --check .
	black --check src scripts tests

typecheck:
	mypy

test:
	pytest

run:
	python scripts/run_station.py

# Full local gate, mirrors CI.
check: lint format-check typecheck test
