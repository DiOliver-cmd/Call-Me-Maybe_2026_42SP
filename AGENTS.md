# Convenções do projeto — Call Me Maybe

Instruções para qualquer sessão de trabalho (humana ou assistida por IA) neste
repositório.

## Idioma

- **Conversa com o autor:** português (pt-BR).
- **Guias, planos e documentação de apoio:** português (pt-BR) — ex.:
  `GUIAS/ESTUDO_MANDATORIO.md`, `GUIAS/TAREFAS_MANDATORIO.md`.
- **Código:** nomes, mensagens de erro e saídas de CLI em inglês; **comentários e
  docstrings em inglês** por padrão.
- **README:** inglês, porque é exigência do subject da 42.

## Processo de execução de task (obrigatório)

Ao iniciar uma task, seguir o ciclo — **salvo pedido explícito do autor para
pular etapa**:

1. O agente fornece o **passo a passo** com comandos prontos e critérios de aceite.
2. O autor **executa** e devolve a saída.
3. O agente **explica** o arquivo criado e como ler o retorno.
4. O agente **registra** a entrada em `GUIAS/DIARIO_DE_EXECUCAO.md`
   (passo a passo, retorno, explicação, decisões).
5. Commit/PR, evidência na issue e atualização do `README.md` no fim da sessão.

## Fluxo de trabalho

- Uma pessoa, **WIP = 1**: só uma issue em `In Progress` por vez.
- Branch por issue (`feat/T4.2-dfa-consume`, `fix/...`, `docs/...`, `chore/...`).
- Commit no padrão Conventional Commits + ID da tarefa:
  `feat(dfa): implement consume(char) (T4.2)`.
- PR pequeno por issue, com `Closes #<n>`; self-review antes do merge.
- Ao fim de **cada sessão**, atualizar o `README.md` com o que evoluiu
  (status do roadmap, comandos, decisões, números medidos).

## Estrutura

```
src/            código da aplicação (entry point: python -m src)
data/input/     functions_definition.json, function_calling_tests.json
tests/          testes unitários e de integração
experiments/    benchmarks, baseline, profiling
llm_sdk/        SDK fornecido — NÃO modificar
GUIAS/          guias internos (ignorado pelo Git)
```

## Comandos

```bash
make install   # uv sync
make run       # uv run python -m src
make lint      # flake8 + mypy
make clean     # remove caches, saída e resíduos fora da estrutura
```

## Regras técnicas

- Python 3.10+, `uv`, `numpy`, `pydantic`, `pytest`; modelo `Qwen/Qwen3-0.6B`.
- Usar apenas a API pública do `llm_sdk`; nunca atributos privados.
- Não commitar `data/output/`, caches, `GUIAS/` nem artefatos de build.
- Sem heurísticas de palavra-chave para escolher função: a decisão é do modelo.
- Fora de escopo: bônus, argumentos aninhados, múltiplos modelos, `encode`/`decode`
  públicos próprios.

## Verificação antes de fechar uma issue

- Critérios de aceite marcados.
- `make lint` e, quando existirem, `pytest -q -m "not slow"`.
- Evidência registrada na issue (comando + saída resumida + commit).
