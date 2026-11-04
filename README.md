*This project has been created as part of the 42 curriculum by DiOliver-cmd.*

# Call Me Maybe

Function-calling tool with **constrained decoding** on top of `Qwen/Qwen3-0.6B`:
the model proposes the next token, a DFA masks every token that would break the
JSON grammar, and the pipeline writes one function call per user prompt.

Status: repository bootstrap (setup milestone). Full documentation is delivered
in the `docs` milestone.

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
