# Token and time budget (T3.4)

Deterministic summary of the measurements behind `src/config.py`. Every number
below comes from the T3.3 baseline (`experiments/baseline_report.md` and the
gitignored `experiments/baseline_outputs.json`); nothing here is generated at
document-build time.

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

## Mitigations to validate in T8.4

These are candidates for T8.4; none is implemented now.

- **Shorter prompt**: the 6-function catalog serialization is ~390 tokens, and
  every step re-reads it. A compact schema could cut both prompt size and
  per-step cost.
- **Fewer tokens**: 48 is already near the compliant-answer floor (~20), so the
  remaining gain is limited to rambling suppression.
- **Faster hardware**: GPU or a faster CPU would attack the 2.494 s/step
  directly; this is the most direct lever.
- **Batching**: processing several prompts per forward pass could amortize the
  per-step cost if the SDK supports it.

The 300 s requirement is tracked as at risk until T8.4 produces measurements.
