SHELL := /bin/bash
.DEFAULT_GOAL := help

# uv: usa o binário do PATH; se ainda não existir, aponta para o destino do
# instalador oficial (~/.local/bin/uv), que é criado pelo alvo `uv`.
UV_INSTALL_URL ?= https://astral.sh/uv/install.sh
ifndef UV
UV := $(shell command -v uv 2>/dev/null || echo $(HOME)/.local/bin/uv)
endif

PYTHON ?= $(UV) run python
FLAKE8 ?= $(UV) run flake8
MYPY ?= $(UV) run mypy

.PHONY: help uv install run debug test test-all clean lint lint-strict

help:
	@echo "Call Me Maybe — alvos disponíveis:"
	@echo "  uv           instala o uv em ~/.local/bin se ainda não estiver disponível"
	@echo "  install      instala o uv (se preciso) e as dependências (uv sync)"
	@echo "  run          executa o pipeline (uv run python -m src)"
	@echo "  debug        executa sob o pdb (uv run python -m pdb -m src)"
	@echo "  test         testes rápidos (pytest -q -m \"not slow\")"
	@echo "  test-all     todos os testes, inclusive os marcados como slow"
	@echo "  lint         flake8 + mypy no projeto"
	@echo "  lint-strict  flake8 + mypy --strict"
	@echo "  clean        remove a saída gerada e caches (mantém .venv e fontes)"

# Instala o uv no diretório do usuário (sem sudo) quando ele não está
# disponível no PATH. Idempotente: se já existe, apenas mostra a versão.
uv:
	@if command -v "$(UV)" >/dev/null 2>&1 || [ -x "$(UV)" ]; then \
		echo "uv já disponível: $$("$(UV)" --version)"; \
	else \
		echo "uv não encontrado; instalando em $(HOME)/.local/bin ..."; \
		if command -v curl >/dev/null 2>&1; then \
			curl -LsSf "$(UV_INSTALL_URL)" | sh; \
		elif command -v wget >/dev/null 2>&1; then \
			wget -qO- "$(UV_INSTALL_URL)" | sh; \
		else \
			echo "erro: instale curl ou wget para baixar o uv" >&2; exit 1; \
		fi; \
		command -v "$(UV)" >/dev/null 2>&1 || [ -x "$(UV)" ] || { \
			echo "erro: falha ao instalar o uv" >&2; exit 1; }; \
		echo "uv instalado: $$("$(UV)" --version)"; \
	fi

install: uv
	$(UV) sync

run:
	$(PYTHON) -m src

debug:
	$(PYTHON) -m pdb -m src

# Suíte rápida: exclui os testes marcados como `slow` (modelo real).
test:
	$(UV) run pytest -q -m "not slow"

# Inclui tudo, inclusive integração com o modelo real.
test-all:
	$(UV) run pytest -q

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
	@echo "Removendo saída gerada e caches..."
	@rm -rf data/output build dist htmlcov .coverage .coverage.*
	@rm -rf .pytest_cache .mypy_cache .ruff_cache .hypothesis
	@find . \( -path './.venv' -o -path './.git' \) -prune -o \
		-type d -name '__pycache__' -exec rm -rf {} +
	@find . \( -path './.venv' -o -path './.git' \) -prune -o \
		-type f \( -name '*.pyc' -o -name '*.pyo' \) -exec rm -f {} +
	@find . \( -path './.venv' -o -path './.git' \) -prune -o \
		-type d -name '*.egg-info' -exec rm -rf {} +
	@echo "OK: limpeza concluída (o .venv foi preservado)."
