PYTHON ?= python3
PIP ?= $(PYTHON) -m pip
FRONTEND_DIR := trader_koo/frontend-v2

.PHONY: backend-install frontend-install hooks-install backend-check test frontend-build dependency-status precommit golden guardrails research-db ci

backend-install:
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements-dev.txt
	$(PIP) install -e .

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
