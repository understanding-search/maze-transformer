CONVERTED_NOTEBOOKS_TEMP_DIR := tests/_temp/notebooks

.PHONY: default
default: help

.PHONY: lint
lint: clean
	python -m mypy --config-file pyproject.toml muutils/
	python -m mypy --config-file pyproject.toml tests/

format: clean
	poetry run python -m pycln --all .
	poetry run python -m isort format .
	poetry run python -m black .

check-format: clean
	poetry run python -m pycln --check --all .
	poetry run python -m isort --check-only .
	poetry run python -m black --check .

unit: clean
	rm -rf .pytest_cache
	poetry run python -m pytest tests/unit

integration: clean
	rm -rf .pytest_cache
	poetry run python -m pytest -s tests/integration

convert_notebooks:
	python tests/helpers/convert_ipynb_to_script.py notebooks/ --output_dir $(CONVERTED_NOTEBOOKS_TEMP_DIR) --disable_plots

test_notebooks: clean convert_notebooks
	@echo "Testing notebooks in $(CONVERTED_NOTEBOOKS_TEMP_DIR)"
	@for file in $(CONVERTED_NOTEBOOKS_TEMP_DIR)/*.py; do \
		echo "  Running $$file"; \
		output_file=$${file%.py}__CI-output.txt; \
		echo "  Output in $$output_file"; \
		poetry run python $$file > $$output_file 2>&1; \
		if [ $$? -ne 0 ]; then \
			echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"; \
			echo "Error in $$file :"; \
			cat $$output_file; \
			exit 1; \
		fi; \
	done

test: unit integration

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
	@cat Makefile | sed -n '/^\.PHONY: / h; /\(^\t@*echo\|^\t:\)/ {H; x; /PHONY/ s/.PHONY: \(.*\)\n.*"\(.*\)"/    make \1\t\2/p; d; x}'| sort -k2,2 |expand -t 25