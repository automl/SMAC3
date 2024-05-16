# These have been configured to only really run short tasks. Longer form tasks
# are usually completed in github actions.

SHELL := /bin/bash

NAME := SMAC3
PACKAGE_NAME := smac
VERSION := 2.1.0

DIR := "${CURDIR}"
SOURCE_DIR := ${PACKAGE_NAME}
DIST := dist
DOCDIR := docs
INDEX_HTML := "file://${DIR}/docs/build/html/index.html"
EXAMPLES_DIR := examples
TESTS_DIR := tests

.PHONY: help install-dev check format pre-commit build tests docs examples clean clean-data clean-docs clean-build publish

help:
	@echo "Makefile ${NAME}"
	@echo "* install-dev      to install all dev requirements and install pre-commit"
	@echo "* check            to check the source code for issues"
	@echo "* format           to format the code with black and isort"
	@echo "* pre-commit       to run the pre-commit check"
	@echo "* build            to build a dist"
	@echo "* tests            to run the tests"
	@echo "* docs             to generate and view the html files, checks links"
	@echo "* examples         to run and generate the examples"
	@echo "* clean            to clean any doc or build files"
	@echo "* publish          to help publish the current branch to pypi"

PYTHON ?= python
PYTEST ?= python -m pytest
PIP ?= python -m pip
MAKE ?= make
BLACK ?= black
ISORT ?= isort
PYDOCSTYLE ?= pydocstyle
MYPY ?= mypy
PRECOMMIT ?= pre-commit
FLAKE8 ?= flake8

install-dev:
	$(PIP) install -e ".[dev]"
	pre-commit install

check-black:
	$(BLACK) ${SOURCE_DIR} --check || :
	$(BLACK) ${EXAMPLES_DIR} --check || :
	$(BLACK) ${TESTS_DIR} --check || :

check-isort:
	$(ISORT) ${SOURCE_DIR} --check || :
	$(ISORT) ${EXAMPLES_DIR} --check || :
	$(ISORT) ${TESTS_DIR} --check || :

check-pydocstyle:
	$(PYDOCSTYLE) ${SOURCE_DIR} || :

check-mypy:
	$(MYPY) ${SOURCE_DIR} || :

check-flake8:
	$(FLAKE8) ${SOURCE_DIR} || :
	$(FLAKE8) ${EXAMPLES_DIR} --check || :
	$(FLAKE8) ${TESTS_DIR} || :

check: check-black check-isort check-mypy check-flake8 check-pydocstyle

pre-commit:
	$(PRECOMMIT) run --all-files

format-black:
	$(BLACK) ${SOURCE_DIR}
	$(BLACK) ${TESTS_DIR}
	$(BLACK) ${EXAMPLES_DIR}

format-isort:
	$(ISORT) ${SOURCE_DIR}
	$(ISORT) ${EXAMPLES_DIR}
	$(ISORT) ${TESTS_DIR}

format: format-black format-isort

tests:
	$(PYTEST) ${TESTS_DIR}

docs:
	$(MAKE) -C ${DOCDIR} docs
	@echo
	@echo "View docs at:"
	@echo ${INDEX_HTML}

examples:
	$(MAKE) -C ${DOCDIR} examples
	@echo
	@echo "View docs at:"
	@echo ${INDEX_HTML}

# Build a distribution in ./dist
build:
	$(PYTHON) setup.py sdist

clean: clean-build clean-docs clean-data

clean-docs:
	$(MAKE) -C ${DOCDIR} clean

clean-build:
	$(PYTHON) setup.py clean
	rm -rf ${DIST}

clean-data:
	# remove all files that could have been left by test cases or by manual runs
	# feel free to add more lines
	find . -maxdepth 3 -iname 'smac3-output_*-*-*_*' | tac | while read -r TESTDIR ; do rm -Rf "$${TESTDIR}" ; done
	find . -maxdepth 3 -iname '*.lock' -exec rm {} \;
	rm -Rf run_*
	rm -Rf test/test_files/scenario_test/tmp_output_*
	rm -Rf test/test_files/test_*_run1
	rm  -f test/test_files/test_scenario_options_to_doc.txt
	rm  -f test/test_files/validation/test_validation_rh.json
	rm -Rf test/run_*
	rm -Rf test/test_smbo/run_*
	rm -Rf test/test_files/test_restore_state/
	rm -Rf test/test_files/test_restored_state/
	rm -Rf test/test_files/validation/test/
	rm  -f test/test_files/validation/validated_runhistory.json*
	rm  -f test/test_files/validation/validated_runhistory_EPM.json*
	rm -Rf test/test_files/out_*/
	rm -Rf smac3-output_*
	rm -Rf .coverage*

# Publish to testpypi
# Will echo the commands to actually publish to be run to publish to actual PyPi
# This is done to prevent accidental publishing but provide the same conveniences
publish: clean build
	read -p "Did you update the version number in Makefile, smac/__init__.py, benchmark/src/wrappers/v20.py, CITATION.cff? \
	Did you add the old version to docs/conf.py? Did you add changes to CHANGELOG.md?"
	
	$(PIP) install twine
	$(PYTHON) -m twine upload --repository testpypi ${DIST}/*
	@echo
	@echo "Test with the following:"
	@echo "* Create a new virtual environment to install the uploaded distribution into"
	@echo "* Run the following:"
	@echo "--- pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ ${PACKAGE_NAME}==${VERSION}"
	@echo
	@echo "* Run this to make sure it can import correctly, plus whatever else you'd like to test:"
	@echo "--- python -c 'import ${PACKAGE_NAME}'"
	@echo
	@echo "Once you have decided it works, publish to actual pypi with"
	@echo "--- python -m twine upload dist/*"
	@echo "After publishing via pypi, please also add a new release on Github and edit the version in the SMAC link \
	on the SMAC Github page."
