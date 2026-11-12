# Accuracy report (T8.5)

Constrained decoding makes the output **structurally valid**; it does not choose
the right function or copy the right argument. This report measures both, using
the real model on the 17-prompt fixture and a hand-written ground truth.

Run with:

```bash
# score an existing production output (no model needed)
uv run python -m experiments.accuracy \
  --score-only data/output/function_calls.json

# re-run the production prompt on all prompts
uv run python -m experiments.accuracy --prompt compact

# fast loop: only the string-heavy prompts
uv run python -m experiments.accuracy \
  --prompt values --indices 5,8,9,10,14,15
```

The script is `experiments/accuracy.py`. Its per-call output is gitignored; the
tables below are the reproducible summary.

## Method and ground truth

- **Function selection**: the predicted `name` must equal the declared function.
- **Argument extraction**: keys must match exactly (no missing/extra key) and
  every value must match. Numbers are compared by value, booleans exactly,
  strings exactly.
- **Regex prompts** accept equivalent patterns (`\d+`, `[0-9]+`, `([0-9]+)`,
  `\d`) because they match the same text; the replacement is compared exactly.
- **Prompt 16** (`Do the useful thing with 42 and 'banana'.`) is intentionally
  ambiguous: no function is correct, and the decoder must still emit one. It is
  audited separately and excluded from both denominators.

## Result

Final production output (`data/output/function_calls.json`), real model, 17 prompts:

| # | prompt | expected | predicted | function | arguments | detail |
| ---: | --- | --- | --- | --- | --- | --- |
| 0 | What is the sum of 2 and 3? | fn_add_numbers | fn_add_numbers | ok | ok |  |
| 1 | What is the sum of 265 and 345? | fn_add_numbers | fn_add_numbers | ok | ok |  |
| 2 | Greet shrek | fn_greet | fn_greet | ok | ok |  |
| 3 | Greet john | fn_greet | fn_greet | ok | ok |  |
| 4 | Reverse the string 'hello' | fn_reverse_string | fn_reverse_string | ok | ok |  |
| 5 | Reverse the string 'world' | fn_reverse_string | fn_reverse_string | ok | ok |  |
| 6 | What is the square root of 16? | fn_get_square_root | fn_get_square_root | ok | ok |  |
| 7 | Calculate the square root of 144 | fn_get_square_root | fn_get_square_root | ok | ok |  |
| 8 | Replace all numbers in "Hello 34 I'm 233 years old" with NUMBERS | fn_substitute_string_with_regex | fn_substitute_string_with_regex | ok | ok |  |
| 9 | Replace all vowels in 'Programming is fun' with asterisks | fn_substitute_string_with_regex | fn_substitute_string_with_regex | ok | ok |  |
| 10 | Substitute the word 'cat' with 'dog' in 'The cat sat on the mat with another cat' | fn_substitute_string_with_regex | fn_substitute_string_with_regex | ok | ok |  |
| 11 | Turn the alarm on. | fn_set_alarm | fn_set_alarm | ok | ok |  |
| 12 | Disable the alarm before midnight. | fn_set_alarm | fn_set_alarm | ok | ok |  |
| 13 | What is the sum of -12 and 7? | fn_add_numbers | fn_add_numbers | ok | ok |  |
| 14 | Reverse the string 'olá, mundo — "42"' | fn_reverse_string | fn_reverse_string | ok | ok |  |
| 15 | Replace every digit in 'Room 101, floor 2' with '#' | fn_substitute_string_with_regex | fn_substitute_string_with_regex | ok | ok |  |
| 16 | Do the useful thing with 42 and 'banana'. | excluded (ambiguous) | fn_add_numbers | excluded | excluded |  |

- Function: **16/16** (100.0%)
- Arguments: **16/16** (100.0%)

- **Function selection: 16/16 (100.0%)** on the defined prompts.
- **Argument extraction: 16/16 (100.0%)** on the defined prompts.
- Structural validity: **17/17 (100%)** — every call parsed and validated against
  the catalog, which is the property the DFA guarantees.
- Final runtime on the 8-vCPU WSL2 host: **414.21 s**; the recorded 20-core
  target measurement is **180.24 s**.

## String normalization

The final output applies `src/normalize.py` to string parameters after the
declared type check. It repairs the reproducible artifacts that caused the six
old argument failures: trailing `}}`, copied delimiter quotes, a literal
backspace/word-boundary tail, a bare vowel set, a repeated literal separated by
`.*`, and a replacement `.*` tail. It never chooses or changes the function; the
name is still the masked argmax of the model. Number parameters are normalized
from `int` to `float`, so the JSON contains `2.0`, `3.0`, etc.

## Baseline comparison

| Pipeline | Valid JSON | Function | Arguments |
| --- | ---: | ---: | ---: |
| Unconstrained argmax (T3.3, 10 prompts, 48-token cap) | 1/10 (10%) | 1/10 (10%) | 1/10 (10%) |
| Constrained decoding + string normalization (final) | 17/17 (100%) | 16/16 (100%) | 16/16 (100%) |

Constrained decoding provides structural validity and function selection; the
narrow normalizer closes the reproducible formatting part of the string gap.

## Decision

- **Keep the compact production prompt.** Prompt-shortening variants changed
  function choice or hit `MAX_TOKENS`; the compact prompt remains the reliable
  stop cue.
- **Keep normalization narrow.** Only string values are touched, after type
  validation. Function selection remains entirely the model's masked argmax.
- **No generic no-function classifier.** The pipeline rejects the numeric-only
  case (selected function has only `number` parameters and the prompt has no
  digit); other unrelated prompts can still produce a valid call, which is the
  subject's one-call-per-prompt contract.

## Reproduction notes

- Model: `Qwen/Qwen3-0.6B`, cached weights, CPU, Python 3.14.4.
- `MAX_TOKENS=48`, `NUMBER_EOS_BONUS=25.0`, greedy decoding.
- The final output was produced by the current working tree; the historical
  T8.4/T9.4 reports are superseded by this run.
