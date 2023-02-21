format:
	python -m pycln --all .
	python -m isort format .
	python -m black .

check-format:
	python -m pycln --check --all .
	python -m isort --check-only .
	python -m black --check .

freeze:
	pip freeze --exclude-editable > requirements.txt

unit:
	python -m pytest tests/unit

integration:
	python -m pytest -s tests/integration

test:
	python -m pytest tests
