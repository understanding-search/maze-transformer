format:
	python -m pycln .
	python -m isort format .
	python -m black .

freeze:
	pip freeze --exclude-editable > requirements.txt

unit:
	python -m pytest tests/unit

integration:
	python -m pytest -s tests/integration

test:
	python -m pytest tests
