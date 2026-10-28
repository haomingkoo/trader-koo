PYTHON ?= python3
PIP ?= $(PYTHON) -m pip
FRONTEND_DIR := trader_koo/frontend-v2

.PHONY: backend-install lock-tools lock lock-check frontend-install hooks-install backend-check test frontend-build dependency-status precommit golden guardrails research-db ci

# Hashed locks on Linux, hash-free lock constraints elsewhere; see DEPLOYMENT.md, "Python dependency lock".
backend-install:
	$(PIP) install --upgrade pip
ifeq ($(shell uname -s),Linux)
	$(PIP) install --no-deps --require-hashes -r requirements-dev.txt
	bash scripts/use-headless-opencv.sh "$(PYTHON)"
else
	@tmp=$$(mktemp -d) && \
	sed -e '/--hash=/d' -e 's/ \\$$//' requirements-dev.txt > "$$tmp/constraints.txt" && \
	grep -v -e '^-c ' -e '^-r ' requirements-dev.in > "$$tmp/requirements-dev.in" && \
	$(PIP) install -c "$$tmp/constraints.txt" -r requirements.in -r "$$tmp/requirements-dev.in"
endif
	$(PIP) install --no-deps --no-build-isolation -e .

# Same pip and pip-tools versions Dependabot runs; see DEPLOYMENT.md, "Python dependency lock".
LOCK_TOOLS := pip==26.2.1 pip-tools==7.6.1
PIP_COMPILE := $(PYTHON) -m piptools compile --quiet --generate-hashes --strip-extras --allow-unsafe

lock-tools:
	$(PIP) install $(LOCK_TOOLS)

# Regenerate the locks on Linux x86_64; see DEPLOYMENT.md, "Python dependency lock".
lock:
	@test "$$(uname -sm)" = "Linux x86_64" || { echo "make lock must run on Linux x86_64"; exit 2; }
	@$(PYTHON) -c 'import sys; v = sys.version_info[:3]; sys.exit(not (3, 11, 5) <= v < (3, 12) and "make lock needs Python 3.11.5 or later 3.11")'
	$(PIP_COMPILE) --output-file=requirements.txt requirements.in
	$(PIP_COMPILE) --output-file=requirements-dev.txt requirements-dev.in

# Fail when a lock no longer matches its input (pins, not hashes); see DEPLOYMENT.md, "Python dependency lock".
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
