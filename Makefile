PYTHON ?= python3
PIP ?= $(PYTHON) -m pip
FRONTEND_DIR := trader_koo/frontend-v2

.PHONY: backend-install lock-tools lock lock-check frontend-install hooks-install backend-check test frontend-build dependency-status precommit golden guardrails research-db ci

# The hashed locks are resolved for Linux x86_64 and Python 3.11 (Railway and
# CI). Elsewhere (macOS) the .in inputs are installed with the lock versions as
# hash-free constraints, because the +cpu torch wheels exist only for Linux.
backend-install:
	$(PIP) install --upgrade pip
ifeq ($(shell uname -s),Linux)
	$(PIP) install --no-deps --require-hashes -r requirements-dev.txt
	@# Match build.sh: drop GUI opencv-python, reinstall the locked headless wheel.
	$(PIP) uninstall -y opencv-python opencv-python-headless
	awk '/^opencv-python-headless==/ {found = 1} found {print} found && !/\\$$/ {exit}' \
		requirements.txt > opencv-headless.lock.tmp
	$(PIP) install --no-deps --require-hashes -r opencv-headless.lock.tmp
	rm -f opencv-headless.lock.tmp
else
	@tmp=$$(mktemp -d) && \
	sed -e '/--hash=/d' -e 's/ \\$$//' requirements-dev.txt > "$$tmp/constraints.txt" && \
	grep -v -e '^-c ' -e '^-r ' requirements-dev.in > "$$tmp/requirements-dev.in" && \
	$(PIP) install -c "$$tmp/constraints.txt" -r requirements.in -r "$$tmp/requirements-dev.in"
endif
	$(PIP) install --no-deps -e .

# Same pip and pip-tools versions Dependabot runs, so its lock updates match.
LOCK_TOOLS := pip==26.2.1 pip-tools==7.6.1
PIP_COMPILE := $(PYTHON) -m piptools compile --quiet --generate-hashes --strip-extras --allow-unsafe

lock-tools:
	$(PIP) install $(LOCK_TOOLS)

# Regenerate the locks. Run on Linux x86_64 with Python 3.11, for example:
# docker run --rm --platform linux/amd64 -v "$$PWD":/src -w /src python:3.11 \
#   sh -c 'make lock-tools lock PYTHON=python'
lock:
	@test "$$(uname -sm)" = "Linux x86_64" || { echo "make lock must run on Linux x86_64"; exit 2; }
	@$(PYTHON) -c 'import sys; sys.exit(sys.version_info[:2] != (3, 11) and "make lock needs Python 3.11")'
	$(PIP_COMPILE) --output-file=requirements.txt requirements.in
	$(PIP_COMPILE) --output-file=requirements-dev.txt requirements-dev.in

# Fail when a lock no longer matches its .in input. pip-compile keeps existing
# pins, so this only changes when an input or a lock was edited by hand.
lock-check:
	@tmp=$$(mktemp -d) && \
	cp requirements.in requirements-dev.in requirements.txt requirements-dev.txt "$$tmp/" && \
	$(MAKE) --no-print-directory -C "$$tmp" -f "$(CURDIR)/Makefile" lock PYTHON="$(PYTHON)" && \
	diff -u requirements.txt "$$tmp/requirements.txt" && \
	diff -u requirements-dev.txt "$$tmp/requirements-dev.txt" && \
	echo "Dependency locks match their inputs."

frontend-install:
	npm ci --prefix $(FRONTEND_DIR)

hooks-install:
	pre-commit install

backend-check:
	find trader_koo tests \
		-path 'trader_koo/frontend-v2' -prune -o \
		-path 'trader_koo/frontend-v2/*' -prune -o \
		-name '*.py' -print0 | xargs -0 $(PYTHON) -m py_compile

test:
	$(PYTHON) -m pytest tests/ -q

frontend-build:
	npm run build --prefix $(FRONTEND_DIR)

dependency-status:
	$(PYTHON) -m pip check
	$(PYTHON) -m pip list --outdated
	npm outdated --prefix $(FRONTEND_DIR) || true

precommit:
	pre-commit run --all-files

golden:
	PYTHON=$(PYTHON) ops/golden_diff.sh $(or $(BASE),origin/main)

guardrails:
	$(PYTHON) -m ops.guardrails check
	$(PYTHON) -c "from importlinter.cli import lint_imports_command as m; m()" --config .importlinter

research-db:
	@test -n "$(RESEARCH_DB)" -a -n "$(SOURCE_DB)" || { echo "usage: make research-db RESEARCH_DB=<file> SOURCE_DB=<prod export or golden copy> [TICKERS=A,B] [EARNINGS_SOURCE=both|finnhub|yfinance]"; exit 2; }
	$(PYTHON) -m ops.build_research_db --db-path "$(RESEARCH_DB)" --source-db "$(SOURCE_DB)" \
		$(if $(TICKERS),--tickers "$(TICKERS)") $(if $(EARNINGS_SOURCE),--earnings-source "$(EARNINGS_SOURCE)")

ci: backend-check test frontend-build precommit guardrails
