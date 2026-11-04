*This project has been created as part of the 42 curriculum by dilferre.*

# Call Me Maybe

Function-calling tool with **constrained decoding** on top of `Qwen/Qwen3-0.6B`:
the model proposes the next token, a DFA masks every token that would break the
JSON grammar, and the pipeline writes one function call per user prompt.

Status: repository bootstrap (setup milestone in progress). Full documentation
is delivered in the `docs` milestone.

## Layout

```
src/            application code (entry point: python -m src)
data/input/     functions_definition.json, function_calling_tests.json
tests/          unit and integration tests
experiments/    benchmarks, baselines, profiling notes
llm_sdk/        provided LLM SDK (do not modify)
```

## Quick start

```bash
make install   # uv sync
make run       # uv run python -m src
make lint      # flake8 + mypy
make clean     # remove caches and generated output
```

## Roadmap

| Milestone | Scope | Status |
|-----------|-------|--------|
| M0 — Setup | repository, uv, SDK smoke test, CI, board | in progress |
| M1 — Contracts | Pydantic models, loaders, error policy | planned |
| M2 — Vocabulary | tokenizer, `token_text`, indexes | planned |
| M3 — LLM baseline | client, prompt, unconstrained measurement | planned |
| M4 — Grammar/DFA | states, slots, transitions | planned |
| M5 — Decoder | mask, argmax, generation loop | planned |
| M6 — Pipeline | orchestration, validation, output | planned |
| M7 — CLI/Packaging | `python -m src`, Makefile targets | planned |
| M8 — Hardening | errors, performance, accuracy | planned |
| M9 — Tests | unit, integration, metrics | planned |
| M10 — Docs/Defence | README, architecture, rehearsal | planned |
| M11 — Submission | audit, cleanup, delivery | planned |

> **Session rule:** at the end of every work session, update this README with
> what changed (roadmap status, commands, decisions, measured numbers). The
> README is the living record of the project state.
