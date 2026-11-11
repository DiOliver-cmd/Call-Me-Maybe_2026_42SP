# Metrics report (T9.4)

Source: `saved output data/output/function_calling_results.json`.

| Metric | Value |
| --- | ---: |
| JSON valid | 17/17 |
| Schema valid | 17/17 |
| Function accuracy | 16/16 (100.0%) |
| Argument accuracy | 10/16 (62.5%) |
| Runtime total | 180.24 s |

Runtime is the T8.4 CLI measurement (180.24 s); use `--run-real` or `--time-report` to measure a fresh run.

## Per-prompt

| # | prompt | expected | predicted | function | arguments | detail |
| ---: | --- | --- | --- | --- | --- | --- |
| 0 | What is the sum of 2 and 3? | fn_add_numbers | fn_add_numbers | ok | ok |  |
| 1 | What is the sum of 265 and 345? | fn_add_numbers | fn_add_numbers | ok | ok |  |
| 2 | Greet shrek | fn_greet | fn_greet | ok | ok |  |
| 3 | Greet john | fn_greet | fn_greet | ok | ok |  |
| 4 | Reverse the string 'hello' | fn_reverse_string | fn_reverse_string | ok | ok |  |
| 5 | Reverse the string 'world' | fn_reverse_string | fn_reverse_string | ok | FAIL | s='world}}' != 'world' |
| 6 | What is the square root of 16? | fn_get_square_root | fn_get_square_root | ok | ok |  |
| 7 | Calculate the square root of 144 | fn_get_square_root | fn_get_square_root | ok | ok |  |
| 8 | Replace all numbers in "Hello 34 I'm 233 years old" with NUMBERS | fn_substitute_string_with_regex | fn_substitute_string_with_regex | ok | FAIL | source_string='Hello 34 I\'m 233 years old", ' != "Hello 34 I'm 233 years old" |
| 9 | Replace all vowels in 'Programming is fun' with asterisks | fn_substitute_string_with_regex | fn_substitute_string_with_regex | ok | FAIL | replacement='*.*' != '*' |
| 10 | Substitute the word 'cat' with 'dog' in 'The cat sat on the mat with another cat' | fn_substitute_string_with_regex | fn_substitute_string_with_regex | ok | FAIL | source_string="The cat sat on the mat with another cat\n\nReplace all occurrences matching the regex 'cat' with 'dog' in the string." != 'The cat sat on the mat with another cat' |
| 11 | Turn the alarm on. | fn_set_alarm | fn_set_alarm | ok | ok |  |
| 12 | Disable the alarm before midnight. | fn_set_alarm | fn_set_alarm | ok | ok |  |
| 13 | What is the sum of -12 and 7? | fn_add_numbers | fn_add_numbers | ok | ok |  |
| 14 | Reverse the string 'olá, mundo — "42"' | fn_reverse_string | fn_reverse_string | ok | FAIL | s='olá, mundo — "42"}}' != 'olá, mundo — "42"' |
| 15 | Replace every digit in 'Room 101, floor 2' with '#' | fn_substitute_string_with_regex | fn_substitute_string_with_regex | ok | FAIL | regex='#\\d+' != '\\d+' |
| 16 | Do the useful thing with 42 and 'banana'. | excluded (ambiguous) | fn_reverse_string | excluded | excluded |  |

## Reproduce

```bash
# Score the saved production output without loading the model.
uv run python -m experiments.metrics \
  --output data/output/function_calling_results.json \
  --runtime-seconds 180.24 \
  --report experiments/report.md

# Re-run the real pipeline and measure the runtime.
uv run python -m experiments.metrics --run-real \
  --output data/output/function_calling_results.json \
  --report experiments/report.md

# Deterministic fake batch (no model).
uv run python -m experiments.metrics --fake \
  --report /tmp/fake_report.md
```

The ground truth and the prompt-variant study are in
`experiments/accuracy.py` / `experiments/accuracy.md`; the
performance budget is in `experiments/benchmark.md`.
