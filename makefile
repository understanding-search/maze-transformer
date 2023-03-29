format:
	poetry run python -m pycln --all .
	poetry run python -m isort format .
	poetry run python -m black .

check-format:
	poetry run python -m pycln --check --all .
	poetry run python -m isort --check-only .
	poetry run python -m black --check .

unit:
	rm -rf .pytest_cache
	poetry run python -m pytest tests/unit

integration:
	rm -rf .pytest_cache
	poetry run python -m pytest -s tests/integration

test: unit integration
