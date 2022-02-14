NAME := SMAC3
PACKAGE_NAME := smac
DIR := ${CURDIR}
SOURCE_DIR := ${CURDIR}/${PACKAGE_NAME}
DIST := ${DIR}/dist
DOCDIR := ${DIR}/docs
INDEX_HTML := "file://${DOCDIR}/build/html/index.html"
TESTS_DIR := ${DIR}/tests
EXAMPLES_DIR := ${DIR}/examples

PYTHON ?= python
CYTHON ?= cython
PYTEST ?= python -m pytest
CTAGS ?= ctags
PIP ?= python -m pip
MAKE ?= make
BLACK ?= black
ISORT ?= isort
PYDOCSTYLE ?= pydocstyle
MYPY ?= mypy
PRECOMMIT ?= pre-commit
FLAKE8 ?= flake8

.PHONY: install-dev, check-black, check-isort, check-mypy, check-flake8, check, pre-commit, format-black, format-isort, format, tests, docs, clean, build, publish

install-dev:
	$(PIP) install -e ".[tests,examples,docs]"
	pre-commit install

check-black:
	$(BLACK) "${SOURCE_DIR}" --check || :
	$(BLACK) "${EXAMPLES_DIR}" --check || :
	$(BLACK) "${TESTS_DIR}" --check || :

check-isort:
	$(ISORT) "${SOURCE_DIR}" --check || :
	$(ISORT) "${TESTS_DIR}" --check || :

check-pydocstyle:
	$(PYDOCSTYLE) "${SOURCE_DIR}" || :

check-mypy:
	$(MYPY) "${SOURCE_DIR}" || :

check-flake8:
	$(FLAKE8) "${SOURCE_DIR}" || :
	$(FLAKE8) "${TESTS_DIR}" || :

check: check-black check-isort check-mypy check-flake8

pre-commit:
	$(PRECOMMIT) run --all-files

format-black:
	$(BLACK) "${SOURCE_DIR}"
	$(BLACK) "${TESTS_DIR}"
	$(BLACK) "${EXAMPLES_DIR}"

format-isort:
	$(ISORT) "${SOURCE_DIR}"
	$(ISORT) "${TESTS_DIR}"

format: format-black format-isort

tests:
	pytest -v --cov=smac tests --durations=20

docs:
	make -C docs clean
	make -C docs html

clean:
	# remove all files that could have been left by test cases or by manual runs
	# feel free to add more lines
	find . -maxdepth 3 -iname 'smac3-output_*-*-*_*' | tac | while read -r TESTDIR ; do rm -Rf "$${TESTDIR}" ; done
	find . -maxdepth 3 -iname '*.lock' -exec rm {} \;
	rm -Rf run_*
	rm -Rf tests/test_files/scenario_test/tmp_output_*
	rm -Rf tests/test_files/test_*_run1
	rm  -f tests/test_files/test_scenario_options_to_doc.txt
	rm  -f tests/test_files/validation/test_validation_rh.json
	rm -Rf tests/run_*
	rm -Rf tests/test_smbo/run_*
	rm -Rf tests/test_files/test_restore_state/
	rm -Rf tests/test_files/test_restored_state/
	rm -Rf tests/test_files/validation/test/
	rm  -f tests/test_files/validation/validated_runhistory.json*
	rm  -f tests/test_files/validation/validated_runhistory_EPM.json*
	rm -Rf tests/test_files/out_*/

build:
	$(PYTHON) setup.py sdist

# Publish to testpypi
# Will echo the commands to actually publish to be run to publish to actual PyPi
# This is done to prevent accidental publishing but provide the same conveniences
publish: clean build
	$(PIP) install twine
	$(PYTHON) -m twine upload --repository testpypi '${DIST}/*''
	@echo
	@echo "Test with the following:"
	@echo "* Create a new virtual environment to install the uplaoded distribution into"
	@echo "* Run the following:"
	@echo
	@echo "        pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ ${NAME}"
	@echo
	@echo "* Run this to make sure it can import correctly, plus whatever else you'd like to test:"
	@echo
	@echo "        python -c 'import ${PACKAGE_NAME}'"
	@echo
	@echo "Once you have decided it works, publish to actual pypi with"
	@echo
	@echo "    python -m twine upload 'dist/*'"