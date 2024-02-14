this_makefile := $(lastword $(MAKEFILE_LIST)) # Used to automatically list targets
.DEFAULT_GOAL := list # If someone runs "make", run "make list"

# Source files to format, lint, and type check.
LOCATIONS=src tests

# Unless overridden, build conda environment using the package name.
PACKAGE_NAME = pseudopeople
SAFE_NAME = $(shell python -c "from pkg_resources import safe_name; print(safe_name(\"$(PACKAGE_NAME)\"))")

about_file   = $(shell find src -name __about__.py)
version_line = $(shell grep "__version__ = " ${about_file})
PACKAGE_VERSION = $(shell echo ${version_line} | cut -d "=" -f 2 | xargs)
PACKAGE_VERSION = $(shell echo $(shell pip list | grep -e "$(SAFE_NAME)") | cut -d " " -f2)


# If CONDA_ENV_PATH is set (from a Jenkins build), use the -p flag when making Conda env in
# order to make env at specific path. Otherwise, make a named env at the default path using
# the -n flag.
PYTHON_VERSION ?= 3.11  # TODO: Update when pytype supports >3.10
CONDA_ENV_NAME ?= ${PACKAGE_NAME}_py${PYTHON_VERSION}
CONDA_ENV_CREATION_FLAG = $(if $(CONDA_ENV_PATH),-p ${CONDA_ENV_PATH},-n ${CONDA_ENV_NAME})

# These are the doc and source code files in this repo.
# When one of these files changes, it means that Make targets need to run again.
MAKE_SOURCES := $(shell find . -type d -name "*" ! -path "./.git*" ! -path "./.vscode" ! -path "./output" ! -path "./output/*" ! -path "./archive" ! -path "./dist" ! -path "./output/htmlcov*" ! -path "**/.pytest_cache*" ! -path "**/__pycache__" ! -path "./output/docs_build*" ! -path "./.pytype*" ! -path "." ! -path "./src/${PACKAGE_NAME}/legacy*" ! -path ./.history ! -path "./.history/*" ! -path "./src/${PACKAGE_NAME}.egg-info" ! -path ./.idea ! -path "./.idea/*" )


# Phony targets don't produce artifacts.
.PHONY: .list-targets format test unit help debug

# List of Make targets is generated dynamically. To add description of target, use a # on the target definition.
list help: debug .list-targets

.list-targets: # Print available Make targets
	@echo
	@echo "Make targets:"
	@grep -i "^[a-zA-Z][a-zA-Z0-9_ \.\-]*: .*[#].*" ${this_makefile} | sort | sed 's/:.*#/ : /g' | column -t -s:
	@echo

debug: # Print debug information (environment variables)
	@echo "'make' invoked with these environment variables:"
	@echo "CONDA_ENV_NAME:                   ${CONDA_ENV_NAME}"
	@echo "IHME_PYPI:                        ${IHME_PYPI}"
	@echo "LOCATIONS:                        ${LOCATIONS}"
	@echo "PACKAGE_NAME:                     ${PACKAGE_NAME}"
	@echo "PACKAGE_VERSION:                  ${PACKAGE_VERSION}"
	@echo "PYPI_ARTIFACTORY_CREDENTIALS_USR: ${PYPI_ARTIFACTORY_CREDENTIALS_USR} "
	@echo "Make sources:                     ${MAKE_SOURCES}"

format: setup.py pyproject.toml $(MAKE_SOURCES) # Run the code formatter and import sorter
	-black $(LOCATIONS)
	-isort $(LOCATIONS)
	@echo "Ignore, Created by Makefile, `date`" > $@

test: $(MAKE_SOURCES) # Run full test suite - both integration and unit tests
	pytest --runslow tests/
	@echo "Ignore, Created by Makefile, `date`" > $@

unit: $(MAKE_SOURCES) # Run unit tests only
	pytest --runslow tests/unit
	@echo "Ignore, Created by Makefile, `date`" > $@
