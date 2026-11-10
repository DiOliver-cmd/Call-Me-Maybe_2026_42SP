*This project has been created as part of the 42 curriculum by dilferre.*

# Call Me Maybe

Function-calling tool with **constrained decoding** on top of `Qwen/Qwen3-0.6B`:
the model proposes the next token, a DFA masks every token that would break the
JSON grammar, and the pipeline writes one function call per user prompt.

Status: M0–M5 complete; M6 (pipeline) in progress — `src/vocab.py`
parses `tokenizer.json` (`model.vocab` + `added_tokens`), builds `token_text` via
the SDK's public `decode`, and indexes tokens by first character so a step
narrows the 151,643 allowed tokens before the prefix/DFA check; structural
states fall to a handful, while permissive string states can keep almost all of
them. The
analysis excludes all 26 added control tokens (14 special + 12 markers such as
`<tool_call>` and `<think>`) and the 267 model padding ids from the decoder's
allowed set; byte-level fragments are kept and must be handled at the byte level.
`src/llm_client.py` wraps the SDK behind a small typed interface (`encode`,
`decode`, `logits`, `vocab_path`), normalizing the 2-D encode tensor to a flat
`list[int]` and logits to a 1-D `float64` array, with a lazy real-model load and
an injectable fake. `src/prompt.py` builds a deterministic plain-text prompt
(functions sorted by name, compact JSON, request stripped) without the Qwen chat
template; measured sizes are 383–402 tokens with the full 6-function catalog and
148–167 tokens with a single function. The input catalog, the output call
models, the JSON loaders and the project-wide error policy are defined and
validated with Pydantic (481 tests: 480 passing, one documented xfail, plus a
slow round-trip suite against the real model). The unconstrained baseline
(10 prompts, argmax, 48-token cap) produced
**1/10 valid JSON, 1/10 correct function and 1/10 correct arguments**. The
budget is `MAX_TOKENS = 48`, a 300 s total deadline and a 180 s per-prompt safety
valve; on this CPU the worst case is 36.9 min, so the 5-minute target is
documented as at risk and deferred to T8.4. `src/dfa.py` models the fixed
output-template machine (`START` through `ACCEPT`, 30 states) and its
structural/number/literal/escape/whitespace alphabet; the pure `transition`,
stateful `DFA.consume`, and atomic `DFA.consume_token`/`try_consume_token` are
implemented, including validation for function names, JSON strings/escapes,
numbers and booleans, with the T4.6 case matrix covering structural, slot,
boundary and multi-character-token cases. `src/prompt.py` assembles the
fixed-key output template and splits it into tokenized fixed segments plus
model-filled slots for direct injection into the DFA. `src/decoder.py` adds the
slot-scoped T5.1 mask: `allowed_token_ids` returns `()` outside model-filled
slots so fixed text is always injected, rejects multi-character tokens that
cross into a fixed closing quote/comma/brace, and never mutates the input
machine. `slot_can_stop` marks complete name/number/boolean values for T5.3 EOS
handling. T5.2 adds `masked_logits`, which sanitizes every non-finite score to
`-inf` and keeps only allowed positions, and `select_next_token`, which raises
`DecodingError` when the mask has no finite score and otherwise returns the
lowest-id argmax. T5.3 adds `decode_call`, which alternates tokenized fixed
injection with model-filled name/value slots, never calls logits for fixed text,
resolves `<|im_end|>`/`<|endoftext|>` when EOS is not supplied, counts every
logits call including EOS against `MAX_TOKENS`, and returns a validated
`FunctionCall` only after `ACCEPT`. T5.4 hardens the deterministic fake matrix
with a reusable `ScriptedLLM`, high-logit distractors, boolean/empty/multi-string
slots, escaped Unicode, explicit EOS overrides and exact `MAX_TOKENS` boundaries.
T6.1 adds the pipeline orchestrator: `run` loads the catalog and prompts in
order, reuses one tokenizer index, preserves each original prompt, and aborts
with a prompt-positioned `DecodingError` instead of returning partial results.
`build_decoder_index` caches decoded token text under `.cache/token_text.json`
and rebuilds after a stale/partial cache. T6.2 validates every decoded call
against the catalog as the final safety net: the name must be declared,
parameters must match exactly (all declared parameters are required, and extra
keys are rejected), and values must have the declared types; `bool` never
satisfies `number` or `integer`. Validation failures retain the prompt position
and text through the pipeline's `DecodingError`. T6.3 adds `write_output`, which
validates the full batch before touching the filesystem, creates parent
directories, and publishes indented UTF-8 JSON atomically through a
same-directory temporary file, `flush`/`fsync`, and `os.replace`; failures remove
the temporary file and leave no partial output. The T6.4 end-to-end run is next.
The decoded-cache benchmark (5-run median after one warm-up, measured
on the development machine; wall-clock varies about ±20%) measured
NAME_VALUE 2 candidates/101 ms, PARAM_VALUE_START 302/512 ms,
PARAM_VALUE_STRING 147k/2971 ms, PARAM_VALUE_NUMBER 13/654 ms,
PARAM_VALUE_BOOLEAN 2/90 ms, PARAM_VALUE_DONE 0/0 ms and ACCEPT 0/0 ms per step.
The string-value state still validates ~150k token texts across 17,773/17,805
buckets, so its fast path and caching work remain deferred to T8.3. Full
documentation is delivered in the `docs` milestone.

Baseline measured on the development machine (CPU, Python 3.13):

- model load: 41.2 s on the first run (includes the Hugging Face download);
- forward pass: 0.209 s median (3 runs, 11-token prompt);
- vocabulary size: 151,669 entries in `tokenizer.json` (14 special tokens); the
  model logits/embedding expose 151,936 positions;
- decoding all 151,669 ids with the SDK's public `decode`: 0.483 s total;
- `encode`/`decode` round-trip lossless for ASCII, accented text, JSON and quotes;
- per-id `decode([id])` is lossy for byte-level fragments (e.g. ids 94–98 decode
  to U+FFFD); multi-byte tokens must be handled at the byte level (T2.2);
- token index built in 0.057 s; first-char lookup narrows 151,643 allowed tokens
  to 60 candidates for `{` and 424 for `"` (up to ~940× faster than a full scan);
- SDK `encode()` returns a 2-D `(1, n)` tensor, so the decoder must flatten it
  before passing ids to `decode` (T2.4); round-trip is lossless for `{"a": 1}`,
  `"hello"` and `"olá"`;
- unconstrained baseline (10 prompts, argmax, 48-token cap): 1/10 valid JSON,
  1/10 correct function, 1/10 correct arguments; 1301.6 s total (~130 s/prompt);
- cost per generated token is ~2.5 s with the ~390-token 6-function prompt
  (CPU, no KV cache in the public SDK), so the 5-minute target is at risk and
  must be validated in T8.4.

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
| M1 — Contracts | Pydantic models, loaders, error policy | done |
| M2 — Vocabulary | tokenizer, `token_text`, indexes | done |
| M3 — LLM baseline | client, prompt, unconstrained measurement | done |
| M4 — Grammar/DFA | states, slots, transitions | done |
| M5 — Decoder | mask, argmax, generation loop | done |
| M6 — Pipeline | orchestration, validation, output | in progress |
| M7 — CLI/Packaging | `python -m src`, Makefile targets | planned |
| M8 — Hardening | errors, performance, accuracy | planned |
| M9 — Tests | unit, integration, metrics | planned |
| M10 — Docs/Defence | README, architecture, rehearsal | planned |
| M11 — Submission | audit, cleanup, delivery | planned |

> **Session rule:** at the end of every work session, update this README with
> what changed (roadmap status, commands, decisions, measured numbers). The
> README is the living record of the project state.
