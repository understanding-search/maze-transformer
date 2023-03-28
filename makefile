format:
	python -m pycln --all .
	python -m isort format .
	python -m black .

check-format:
	python -m pycln --check --all .
	python -m isort --check-only .
	python -m black --check .

unit:
	rm -rf .pytest_cache
	poetry run python -m pytest tests/unit

integration:
	rm -rf .pytest_cache
	poetry run python -m pytest -s tests/integration

test: unit integration
