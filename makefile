CONVERTED_NOTEBOOKS_TEMP_DIR := tests/_temp/notebooks
NOTEBOOKS_DIR := notebooks
ROOT_RELATIVE_TO_NOTEBOOKS := $(shell realpath --relative-to=$(NOTEBOOKS_DIR) .)


.PHONY: default
default: help

.PHONY: lint
lint: clean
	@echo "run linting: mypy"
	python -m mypy --config-file pyproject.toml maze_transformer/
	python -m mypy --config-file pyproject.toml tests/

.PHONY: format
format: clean
	@echo "run formatting: pycln, isort, and black"
	poetry run python -m pycln --all .
	poetry run python -m isort format .
	poetry run python -m black .

.PHONY: check-format
check-format: clean
	@echo "check formatting"
	poetry run python -m pycln --check --all .
	poetry run python -m isort --check-only .
	poetry run python -m black --check .

.PHONY: unit
unit:
	@echo "run unit tests"
	poetry run python -m pytest tests/unit

.PHONY: integration
integration:
	@echo "run integration tests"
	poetry run python -m pytest -s tests/integration

.PHONY: convert_notebooks
convert_notebooks:
	@echo "convert notebooks in $(NOTEBOOKS_DIR) using tests/helpers/convert_ipynb_to_script.py"
	python tests/helpers/convert_ipynb_to_script.py notebooks/ --output_dir $(CONVERTED_NOTEBOOKS_TEMP_DIR) --disable_plots

.PHONY: test_notebooks
test_notebooks: convert_notebooks
	@echo "test notebooks in $(NOTEBOOKS_DIR)"
	@echo "Testing notebooks in $(CONVERTED_NOTEBOOKS_TEMP_DIR)"
	@for file in $(CONVERTED_NOTEBOOKS_TEMP_DIR)/*.py; do \
		echo "  Running $$file"; \
		output_file=$${file%.py}__CI-output.txt; \
		echo "  Output in $$output_file"; \
		(cd $(NOTEBOOKS_DIR) && poetry run python $(ROOT_RELATIVE_TO_NOTEBOOKS)/$$file > $(ROOT_RELATIVE_TO_NOTEBOOKS)/$$output_file 2>&1); \
		if [ $$? -ne 0 ]; then \
			echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"; \
			echo "Error in $$file :"; \
			cat $$output_file; \
			exit 1; \
		fi; \
	done

.PHONY: test
test: clean unit integration test_notebooks
	@echo "run all testts: unit, integration, and notebooks"

.PHONY: clean
clean:
	@echo "cleaning up caches and temp files"
	rm -rf .mypy_cache
	rm -rf .pytest_cache
	rm -rf dist
	rm -rf build
	rm -rf tests/_temp
	python -Bc "import pathlib; [p.unlink() for p in pathlib.Path('.').rglob('*.py[co]')]"
	python -Bc "import pathlib; [p.rmdir() for p in pathlib.Path('.').rglob('__pycache__')]"



# listing targets, from stackoverflow
# https://stackoverflow.com/questions/4219255/how-do-you-get-the-list-of-targets-in-a-makefile
.PHONY: help
help:
	@echo -n "# list make targets"
	@echo ":"
	@cat Makefile | sed -n '/^\.PHONY: / h; /\(^\t@*echo\|^\t:\)/ {H; x; /PHONY/ s/.PHONY: \(.*\)\n.*"\(.*\)"/    make \1\t\2/p; d; x}'| sort -k2,2 |expand -t 30