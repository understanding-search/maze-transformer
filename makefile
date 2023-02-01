format:
	python -m pycln .
	python -m isort format .
	python -m black .

freeze:
	pip freeze --exclude-editable > requirements.txt

test:
	python -m pytest .
