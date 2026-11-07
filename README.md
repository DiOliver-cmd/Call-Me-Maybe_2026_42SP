*This project has been created as part of the 42 curriculum by dilferre.*

# Call Me Maybe

Function-calling tool with **constrained decoding** on top of `Qwen/Qwen3-0.6B`:
the model proposes the next token, a DFA masks every token that would break the
JSON grammar, and the pipeline writes one function call per user prompt.

Status: setup milestone complete (M0). The SDK public API is verified and the
model runs locally on CPU; constrained decoding is the next step (M1–M5). Full
documentation is delivered in the `docs` milestone.

Baseline measured on the development machine (CPU, Python 3.13):

- model load: 41.2 s on the first run (includes the Hugging Face download);
- forward pass: 0.209 s median (3 runs, 11-token prompt);
- vocabulary size: 151,936 tokens;
- encode/decode round-trip lossless for ASCII, accented text, JSON and quotes.

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
make uv        # install uv in ~/.local/bin when missing (no sudo)
make install   # uv sync
make run       # uv run python -m src
make test      # pytest -q -m "not slow"
make test-all  # full suite, including slow/model-backed tests
make lint      # flake8 + mypy
make clean     # remove caches and generated output
```

## Roadmap

| Milestone | Scope | Status |
|-----------|-------|--------|
| M0 — Setup | repository, uv, SDK smoke test, CI, board | done |
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
