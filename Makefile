SHELL := /bin/bash
.DEFAULT_GOAL := help

UV ?= uv
PYTHON ?= $(UV) run python
FLAKE8 ?= $(UV) run flake8
MYPY ?= $(UV) run mypy

# Caminhos que fazem parte da estrutura e NÃO devem ser removidos pelo clean.
# Qualquer outro diretório vazio (inclusive aninhado) é considerado resíduo.
KEEP := ! -path './.git' ! -path './.git/*' ! -path './.github' ! -path './.github/*' \
        ! -path './src' ! -path './tests' ! -path './experiments' \
        ! -path './data' ! -path './data/input' \
        ! -path './llm_sdk' ! -path './GUIAS' ! -path './Material_base'

.PHONY: help install run debug clean lint lint-strict

help:
	@echo "Call Me Maybe — alvos disponíveis:"
	@echo "  install      instala dependências (uv sync)"
	@echo "  run          executa o pipeline (uv run python -m src)"
	@echo "  debug        executa sob o pdb (uv run python -m pdb -m src)"
	@echo "  lint         flake8 + mypy no projeto"
	@echo "  lint-strict  flake8 + mypy --strict"
	@echo "  clean        remove caches, artefatos e diretórios gerados fora da raiz"

install:
	$(UV) sync

run:
	$(PYTHON) -m src

debug:
	$(PYTHON) -m pdb -m src

# mypy falha com exit 2 quando não há nenhum .py (fase inicial do projeto).
lint:
	$(FLAKE8) .
	@if find . -name '*.py' -not -path './.venv/*' -not -path './llm_sdk/*' | grep -q .; then \
		$(MYPY) . ; \
	else \
		echo "mypy: nenhum arquivo .py ainda; nada a checar."; \
	fi

lint-strict:
	$(FLAKE8) .
	@if find . -name '*.py' -not -path './.venv/*' -not -path './llm_sdk/*' | grep -q .; then \
		$(MYPY) . --strict ; \
	else \
		echo "mypy: nenhum arquivo .py ainda; nada a checar."; \
	fi

clean:
	@echo "Removendo caches e artefatos gerados..."
	@find . -type d -name '__pycache__' -prune -exec rm -rf {} +
	@find . -type d \( -name '.pytest_cache' -o -name '.mypy_cache' \
		-o -name '.ruff_cache' -o -name '.hypothesis' \) -prune -exec rm -rf {} +
	@rm -rf data/output build dist .venv venv .coverage .coverage.* htmlcov
	@find . -type d -name '*.egg-info' -prune -exec rm -rf {} +
	@find . -type f \( -name '*.pyc' -o -name '*.pyo' \) -delete
	@# Remove diretórios de topo que não pertencem à estrutura do projeto
	@# (qualquer resíduo criado fora da raiz esperada).
	@find . -mindepth 1 -maxdepth 1 -type d \
		! -name '.git' ! -name '.github' \
		! -name 'src' ! -name 'tests' ! -name 'experiments' \
		! -name 'data' ! -name 'llm_sdk' ! -name 'GUIAS' \
		! -name 'Material_base' ! -name 'docs' \
		-exec rm -rf {} +
	@# Remove qualquer diretório vazio (aninhado) que não pertença à estrutura.
	@find . -mindepth 1 -type d -empty $(KEEP) -delete
	@echo "OK: limpeza concluída."
