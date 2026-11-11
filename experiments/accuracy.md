# Accuracy report (T8.5)

Constrained decoding makes the output **structurally valid**; it does not choose
the right function or copy the right argument. This report measures both, using
the real model on the 17-prompt fixture and a hand-written ground truth.

Run with:

```bash
# score an existing production output (no model needed)
uv run python -m experiments.accuracy \
  --score-only data/output/function_calling_results.json

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

Production prompt (`build_prompt`, compact), real model, 17 prompts:

| # | prompt | expected | predicted | function | arguments | detail |
| ---: | --- | --- | --- | --- | --- | --- |
| 0 | What is the sum of 2 and 3? | `fn_add_numbers` | `fn_add_numbers` | ok | ok | |
| 1 | What is the sum of 265 and 345? | `fn_add_numbers` | `fn_add_numbers` | ok | ok | |
| 2 | Greet shrek | `fn_greet` | `fn_greet` | ok | ok | |
| 3 | Greet john | `fn_greet` | `fn_greet` | ok | ok | |
| 4 | Reverse the string 'hello' | `fn_reverse_string` | `fn_reverse_string` | ok | ok | |
| 5 | Reverse the string 'world' | `fn_reverse_string` | `fn_reverse_string` | ok | **FAIL** | `s='world}}' != 'world'` |
| 6 | What is the square root of 16? | `fn_get_square_root` | `fn_get_square_root` | ok | ok | |
| 7 | Calculate the square root of 144 | `fn_get_square_root` | `fn_get_square_root` | ok | ok | |
| 8 | Replace all numbers in "Hello 34 I'm 233 years old" with NUMBERS | `fn_substitute_string_with_regex` | `fn_substitute_string_with_regex` | ok | **FAIL** | `source_string='Hello 34 I\'m 233 years old", '` |
| 9 | Replace all vowels in 'Programming is fun' with asterisks | `fn_substitute_string_with_regex` | `fn_substitute_string_with_regex` | ok | **FAIL** | `replacement='*.*' != '*'` |
| 10 | Substitute the word 'cat' with 'dog' in 'The cat sat on the mat with another cat' | `fn_substitute_string_with_regex` | `fn_substitute_string_with_regex` | ok | **FAIL** | source includes a copied explanation |
| 11 | Turn the alarm on. | `fn_set_alarm` | `fn_set_alarm` | ok | ok | |
| 12 | Disable the alarm before midnight. | `fn_set_alarm` | `fn_set_alarm` | ok | ok | |
| 13 | What is the sum of -12 and 7? | `fn_add_numbers` | `fn_add_numbers` | ok | ok | |
| 14 | Reverse the string 'olá, mundo — "42"' | `fn_reverse_string` | `fn_reverse_string` | ok | **FAIL** | `s='olá, mundo — "42"}}'` |
| 15 | Replace every digit in 'Room 101, floor 2' with '#' | `fn_substitute_string_with_regex` | `fn_substitute_string_with_regex` | ok | **FAIL** | `regex='#\d+'`, `replacement='#}}'` |
| 16 | Do the useful thing with 42 and 'banana'. | excluded | `fn_reverse_string` | excluded | excluded | no correct answer |

- **Function selection: 16/16 (100.0%)** on the defined prompts.
- **Argument extraction: 10/16 (62.5%)** on the defined prompts.
- Structural validity: **17/17 (100%)** — every call parsed and validated against
  the catalog, which is the property the DFA guarantees.

## Baseline comparison

| Pipeline | Valid JSON | Function | Arguments |
| --- | ---: | ---: | ---: |
| Unconstrained argmax (T3.3, 10 prompts, 48-token cap) | 1/10 (10%) | 1/10 (10%) | 1/10 (10%) |
| Constrained decoding (this report, 16 defined prompts) | 17/17 (100%) | 16/16 (100%) | 10/16 (62.5%) |

Constrained decoding buys structural validity and function selection; argument
extraction in long string slots remains the weak point.

## Prompt variants tested

The six failures are all string slots. Three instruction variants tried to make
the model emit values without re-writing the JSON:

| Variant | Wording | Result on #8 |
| --- | --- | --- |
| `values` | "The output JSON structure is already written. Fill only…" | `MAX_TOKENS` 48 |
| `minimal` | "The JSON is already written… no quotes, braces, keys…" | `MAX_TOKENS` 48 |
| `roles` | above + parameter roles + "copy the text, then stop" | `MAX_TOKENS` 48 |

Root cause: the compact prompt's `Return one JSON object` gives the 0.6B model a
stop cue. When the instruction says "output only values", the model keeps
generating (it copies the request or explains it) until the token budget ends.
The compact prompt is therefore kept: it is the only variant that stops
reliably, and its residual errors are short `}}`/prose tails rather than
unbounded generations.

## Decision

- **Keep the compact production prompt.** Prompt wording alone does not move
  argument accuracy on this 0.6B model; the tested alternatives regress to
  `MAX_TOKENS`.
- **Justified deviation from the ≥90% argument target.** Selection is 100% and
  structural validity is 100%, but exact string copying is 62.5%. The failures
  are concentrated in long values (#8, #10) and in the regex/replacement split
  (#9, #15). Closing that gap needs a larger model or a decoder that extracts a
  span instead of generating characters — both outside the mandatory scope.
- **No heuristic was added.** Function choice remains the model's argmax over
  the masked logits; no keyword rule, no post-processing of values.

## Reproduction notes

- Model: `Qwen/Qwen3-0.6B`, cached weights, CPU, Python 3.13.
- `MAX_TOKENS=48`, `NUMBER_EOS_BONUS=25.0`, greedy decoding.
- The three variant runs aborted at #8; that is the recorded result, not a
  measurement artefact.
- The full production run that produced the table above is the T8.4 run
  (`data/output/function_calling_results.json`, commit `453b8a6`).
