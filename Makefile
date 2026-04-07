PYTHON ?= python3
PIP ?= $(PYTHON) -m pip
FRONTEND_DIR := trader_koo/frontend-v2

.PHONY: backend-install frontend-install hooks-install backend-check test frontend-build precommit ci

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

precommit:
	pre-commit run --all-files

ci: backend-check test frontend-build precommit
