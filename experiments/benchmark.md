# Token and time budget (T3.4, updated by T8.4)

Deterministic summary of the measurements behind `src/config.py`. Every number
below comes from a real run; nothing here is generated at document-build time.

> **T8.4 update:** the compact prompt described in
> [T8.4 — measured result](#t84--measured-result) brings the default 17-prompt
> run to **192.2 s**, inside the 300 s target. The T3.4 analysis below is kept
> as the historical baseline that motivated the change.

## Measurements

| Measurement | Value | Source |
| --- | --- | --- |
| Prompt size, full 6-function catalog | 390 tokens | T3.2 / T3.3 diagnostic |
| Greedy per-step, ~390-token prompt | **2.494 s/step** (10-sample average; min 2.387 s, max 2.814 s) | T3.3 diagnostic |
| Overall per-token, including context growth | **2.712 s/token** | 1301.576 s / 480 tokens = 2.7116 s |
| 10 prompts x 48 tokens, total | **1301.58 s** (21.7 min) | `experiments/baseline_report.md` |
| 10 prompts x 48 tokens, average | **130.16 s/prompt** | `experiments/baseline_report.md` |
| Baseline outcomes | 1/10 valid JSON, 1/10 function correct, 1/10 arguments correct | `experiments/baseline_report.md` |
| Stop reasons | `max_tokens` 10/10, `eos` 0/10, `decode_error` 0/10 | `experiments/baseline_outputs.json` |

The first `client.encode` call took 106.357 s because it constructs the real
backend (model initialization); the next `encode` took 0.0014 s. Case 0 in the
baseline (209.01 s) includes that one-time cost; the other nine cases averaged
about 121.7 s each.

## Worst case

`estimate_seconds(n_prompts, max_tokens) = n_prompts * max_tokens * 2.712`.
This is a lower bound in practice: the forward pass grows with context length.

| Prompts | Tokens/prompt | Estimated seconds | Minutes | Within 300 s? |
| --- | --- | --- | --- | --- |
| 1 | 24 | 65.1 | 1.1 | yes |
| 1 | 48 | 130.2 | 2.2 | yes |
| 10 | 24 | 650.9 | 10.8 | no |
| 10 | 48 | 1301.8 | 21.7 | no |
| 17 | 24 | 1106.5 | 18.4 | no |
| 17 | 48 | **2213.0** | **36.9** | no |

Conclusion: the 5-minute target is **unreachable on this CPU** with the
6-function prompt. Even the most optimistic full-catalog case (17 prompts x 24
tokens) is about 3.7x over budget, and the configured 17 x 48 case is about
7.4x over. The subject's 300 s target is kept as the reference in
`TOTAL_TIMEOUT_SECONDS`; meeting it is deferred to T8.4.

## Why `MAX_TOKENS = 48`

A compliant answer (`{"name": ..., "arguments": {...}}`) costs about 20 tokens.
The T3.3 baseline used 48 tokens and only rambling output was cut off: the one
fully correct case completed inside 48 tokens. `48` therefore keeps roughly
2.4x headroom over a compliant answer without paying for runaway generations.
Lowering it much further would start truncating legitimate calls; raising it
only makes the time overrun worse.

`PROMPT_TIMEOUT_SECONDS = 180` is a safety valve for one prompt: the baseline
average was 130.16 s, so 180 s leaves margin for the normal case. The slowest
observed case was 209.01 s, but that value includes the one-time model
initialization and is not representative of a warm run.

## Abort behavior

`ensure_within_deadline(elapsed, limit)` raises `BudgetExceededError` when the
elapsed time is strictly greater than the limit. `BudgetExceededError` subclasses
`DecodingError`, which subclasses `ProjectError`, so the existing CLI policy maps
it to exit code **1** and prints `error: time budget exceeded: ...`.

No partial output is written: the pipeline builds every call first and writes
the output file only after all calls succeed (T6.3). An abort therefore leaves
no truncated `data/output/` file behind.

## Mitigations considered in T8.4

- **Shorter prompt**: the 6-function catalog serialization is ~390 tokens, and
  every step re-reads it. A compact schema cuts both prompt size and per-step
  cost — **implemented and validated in T8.4**.
- **Fewer tokens**: 48 is already near the compliant-answer floor (~20), so the
  remaining gain is limited to rambling suppression.
- **Faster hardware**: GPU or a faster CPU would attack the per-step cost
  directly; this is the most direct lever.
- **Batching**: processing several prompts per forward pass could amortize the
  per-step cost if the SDK supports it.

---

# T8.4 — measured result

Command: `uv run python -m experiments.time_budget` (real model, real catalog,
all 17 fixture prompts, `NUMBER_EOS_BONUS` on, `MAX_TOKENS = 48`).
Gitignored JSON: `experiments/time_budget_report.json`.

Machine: development CPU, 20 cores, Python 3.13, Qwen/Qwen3-0.6B, no KV cache
in the public SDK. The model was already in the Hugging Face cache.

## Before and after

| Measurement | Verbose prompt (T3.2/T8.3) | Compact prompt (T8.4) | Change |
| --- | --- | --- | --- |
| Prompt tokens, full catalog | 383–402 | **133–151** | −63% |
| Model steps per prompt | 12–14 | 13.82 mean | ~equal |
| Seconds per model step | 1.818 | **0.729** | −60% |
| Decoding total (17 prompts) | 702 s wall (T8.3) | **183.5 s** | −74% |
| Wall clock, 17 prompts | 702 s (11m42s) | **192.2 s** (3m12s) | −73% |

The two "before" rows come from different sources: 1.818 s/step is the T8.4
probe that ran the verbose prompt on this machine for two prompts (20.25 s and
26.08 s over 10 and 14 steps), while 702 s is the full T8.3 default run.

## Full compact run

| Phase | Seconds |
| --- | ---: |
| Model load (cached weights, no download) | 8.245 |
| Decoder index build (token-text cache present) | 0.506 |
| Decoding, 17 prompts / 235 steps | 183.477 |
| **Wall clock** | **192.227** |

- Production CLI run (`uv run python -m src`, default flags): **180.24 s**,
  exit code 0, 17 calls written.

- Per prompt: **10.792 s mean**, 7.116 s median.
- Per step: **0.729 s mean** at ~140 prompt tokens.
- Budget: `estimate_seconds(17)` = 8.751 + 17 × 13.82 × 0.729 = **180.0 s**;
  `exceeds_time_target(17)` is now `False`.
- A first run on a fresh machine adds the model download (~40 s, measured
  during the verbose probe: 46.5 s to the first `encode`), so the worst case is
  about **232 s**, still under 300 s.

Production command, same machine, immediately after: `uv run python -m src`
(the subject's exact invocation, default flags) finished in **180.24 s** with
exit code 0 and wrote all 17 calls to
`data/output/function_calls.json` (`MAXRSS` 4.9 GB). The 12 s
difference against the harness is the instrumentation overhead of
`experiments/time_budget.py`.

## Per-prompt detail

| # | Steps | Seconds | s/step | Prompt tokens | Function |
| ---: | ---: | ---: | ---: | ---: | --- |
| 0 | 8 | 6.74 | 0.84 | 140 | `fn_add_numbers` |
| 1 | 12 | 10.44 | 0.87 | 144 | `fn_add_numbers` |
| 2 | 9 | 8.16 | 0.91 | 134 | `fn_greet` |
| 3 | 8 | 5.50 | 0.69 | 133 | `fn_greet` |
| 4 | 7 | 4.93 | 0.70 | 135 | `fn_reverse_string` |
| 5 | 7 | 4.75 | 0.68 | 135 | `fn_reverse_string` |
| 6 | 8 | 6.59 | 0.82 | 139 | `fn_get_square_root` |
| 7 | 9 | 7.12 | 0.79 | 139 | `fn_get_square_root` |
| 8 | 37 | 32.20 | 0.87 | 151 | `fn_substitute_string_with_regex` |
| 9 | 22 | 15.87 | 0.72 | 142 | `fn_substitute_string_with_regex` |
| 10 | 27 | 21.79 | 0.81 | 151 | `fn_substitute_string_with_regex` |
| 11 | 6 | 4.00 | 0.67 | 134 | `fn_set_alarm` |
| 12 | 6 | 3.92 | 0.65 | 135 | `fn_set_alarm` |
| 13 | 10 | 7.38 | 0.74 | 141 | `fn_add_numbers` |
| 14 | 16 | 10.88 | 0.68 | 143 | `fn_reverse_string` |
| 15 | 34 | 26.82 | 0.79 | 147 | `fn_substitute_string_with_regex` |
| 16 | 9 | 6.38 | 0.71 | 141 | `fn_add_numbers` |

The cost follows the number of generated tokens, not the number of prompts:
the three regex-substitution calls take 37, 22 and 27 steps because the model
fills three string slots character by character, and together they are 41% of
the decoding time.

## What changed in the code

- `src/prompt.py`: `build_prompt` now emits one signature line per function
  (`fn_add_numbers(a:number, b:number): Add two numbers together and return
  their sum.`) instead of the verbose JSON catalog, the "Response format" block
  and the prose instructions. The names, parameter names, types and
  descriptions are unchanged, so the model sees the same catalog content.
  `build_verbose_prompt` keeps the old format for A/B measurement and for
  `experiments/baseline.py`, which must stay reproducible.
- `src/decoder.py`: `decode_call` accepts an optional `prompt_builder` so the
  benchmark measures both formats through the same decoder.
- `src/config.py`: `MEASURED_SECONDS_PER_TOKEN` (verbose baseline) is replaced
  by `MEASURED_SECONDS_PER_STEP`, `MEASURED_STEPS_PER_PROMPT` and
  `MEASURED_SETUP_SECONDS`; `estimate_seconds` now projects from measurements
  instead of assuming every prompt burns `MAX_TOKENS`.
- `experiments/time_budget.py`: the measurement harness (`--limit`,
  `--prompt-style`, per-prompt rows, JSON report).
- `experiments/prompt_report.py`: reports both prompt formats.

## Follow-up

The compact prompt changes what the model reads, so the choice accuracy must be
re-measured: prompt 16 ("Do the useful thing with 42 and 'banana'.") now
resolves to `fn_add_numbers` instead of `fn_reverse_string`. That is tracked as
T8.5, together with the pre-existing string-slot contamination (`"world}}"`,
invented regex patterns).

