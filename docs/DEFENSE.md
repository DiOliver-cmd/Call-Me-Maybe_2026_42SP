# Defense rehearsal kit

This is the practice script for the oral defense and the live modification. It
is deliberately short: read it once, then rehearse without opening the code.

Companion documents:

- `README.md` — result, commands and public design summary.
- `docs/ARCHITECTURE.md` — module map, flow, decisions and limitations.

---

## 10-minute script

| Time | Say | Open if asked |
| --- | --- | --- |
| 0:00–1:00 | **Problem and result.** The program reads a catalog and N requests and writes one JSON call per request. It runs 17 prompts in about 180 s, with 17/17 valid calls, 16/16 correct function names (100%) and 10/16 exact arguments (62.5%). | `experiments/report.md` |
| 1:00–3:00 | **Pipeline.** CLI → loaders → catalog/prompts → compact prompt → fixed prefix injection → DFA → masked argmax → `FunctionCall` → catalog validation → atomic JSON write. The decoder index is built once. | `src/pipeline.py:run`, `src/prompt.py:build_prompt` |
| 3:00–5:30 | **Constrained decoding.** `tokenizer.json` → `token_text`; first-character index; `allowed_token_ids` returns only tokens that can advance the current DFA state; `masked_logits` sets the rest to `-inf`; `select_next_token` takes the highest finite logit. Fixed text is injected, not generated. | `src/decoder.py:decode_call`, `src/dfa.py:DFA.consume` |
| 5:30–7:30 | **Why it is valid by construction.** The model never emits braces, keys or commas; the DFA only permits the declared function prefix and the declared scalar type; Pydantic validates the final call against the catalog; the writer publishes only a complete, validated batch. | `src/schemas.py:validate_call`, `src/pipeline.py:write_output` |
| 7:30–9:00 | **Decisions.** Template-specific DFA instead of general JSON; greedy argmax; no heuristics and no retries; compact prompt to meet 300 s; EOS bonus for numbers; mask cache. | `docs/ARCHITECTURE.md` decision table |
| 9:00–10:00 | **Limits.** Arguments are 62.5%; long strings and regex/replacement are the weak spots. No no-op option, no nested arguments, no KV cache/batching, number precision can lose `.0`. Validity is guaranteed; semantics is measured. | `docs/ARCHITECTURE.md` limitations |

The one-sentence thesis: **the model chooses the function and the values, the
grammar guarantees the shape, and the measurements tell the truth about
semantics.**

---

## Likely questions and short answers

### 1. Why `-inf` instead of just asking for JSON in the prompt?

A prompt is a soft bias; a 0.6B model can still emit prose, a wrong key or a
trailing comma. Masking with `-inf` makes every disallowed token's softmax
probability exactly zero, so `argmax` cannot select it. The prompt explains the
task; the mask enforces the contract. They are complementary, not alternatives.

### 2. How do you guarantee 100% valid output?

Three layers: fixed text is injected by the decoder, never generated; the DFA
only allows tokens that keep the current state valid; `FunctionCall` and
`catalog.validate_call` re-check name, keys and strict types before the atomic
write. Validity is a property of the construction, not of the model's mood.

### 3. If the mask restricts the tokens, is the model still choosing the function?

Yes. In the name slot the mask is the set of tokens that are prefixes of
declared names; the model's logits decide which prefix wins and when to stop.
There is no `if "sum" in prompt` or keyword table. The same mechanism chooses
values inside each slot. The grammar constrains the space; the model picks the
point.

### 4. How do multi-character tokens and prefixes work?

`token_text` maps each token id to its decoded string. `allowed_token_ids`
checks the whole token string against the DFA from the current state, not just
its first character (the first-character index only narrows the candidate set).
`DFA.try_consume_token` advances on a copy, so a token that would cross a
boundary invalidly is rejected without corrupting the state. A token can cover
several characters and still be accepted if the entire token remains valid.

### 5. What happens if every token is masked?

`masked_logits` returns an all-`-inf` array, but `select_next_token` checks for
a finite score and raises `DecodingError` with the allowed-id count. It never
falls back to index 0 or to an arbitrary token. The pipeline aborts with a
clear error and writes no partial output.

### 6. How do you add a function or a type?

A function is data: add it to `data/input/functions_definition.json`; the prompt,
name mask and validation are driven by the catalog. A new scalar type is code:
add it to `KNOWN_PARAMETER_TYPES` in `schemas.py`, add its DFA states/transitions,
extend `_SLOT_TARGETS` in `decoder.py`, and extend the prompt signature. The
live-modification map in `docs/ARCHITECTURE.md` lists the exact files and tests.

### 7. How did you measure accuracy and time?

`experiments/accuracy.py` runs the defined fixtures and compares the call to the
expected one; `experiments/metrics.py` computes validity, function accuracy,
argument accuracy and runtime. `experiments/time_budget.py` measures the real
pipeline. The reported run is 17/17 valid, 16/16 function, 10/16 arguments,
180.24 s, against a 300 s target. The slow integration suite replays three real
prompts end to end.

### 8. Does the DFA validate key order?

The DFA validates the template structure; the key order is fixed by injection,
so the model cannot produce a reordered object. The strict test
`test_matrix_wrong_key_order_is_rejected` is an `xfail` that records this
assumption; if injection changes, the test fails. This is a deliberate
separation: the grammar owns the shape, the writer owns the order.

### 9. Why no retries or heuristic fallback?

The subject requires the model to make the decision. A retry loop would hide
decoding failures, add latency and distort the accuracy measurement. A keyword
fallback would replace the model with a heuristic. The project reports a wrong
call instead of repairing it silently.

### 10. Why the `NUMBER_EOS_BONUS = 25.0`?

A complete number can always continue with `.` or another digit, so an unbiased
EOS is rarely the argmax. The bonus is calibrated to make EOS win once the
number is syntactically complete. The cost is that `2.0` can stop as `2`, which
is still a valid number but loses precision. It is documented, not hidden.

### 11. Why cache allowed tokens?

The allowed-token scan depends on the DFA state plus the buffer that can change
the outcome. Caching by `(state, discriminator)` makes repeated states cheap
while remaining transparent: a hit returns exactly the uncached tuple. The cache
is process-local; the token-text cache is persisted.

### 12. Why a first-character index / trie?

Scanning the whole vocabulary for every slot would be wasteful. The first
character narrows the candidates, and prefix matching over function names is the
trie idea: a token is allowed only if it is a prefix of a declared name. The DFA
does the same job for values and structural states.

### 13. What if the request is ambiguous or unrelated?

The grammar forces exactly one declared function per prompt, because that is the
subject contract. An unrelated request still produces a schema-valid call; it
may be semantically wrong. There is no "no function" option, and that is a
documented limitation rather than a hidden heuristic.

### 14. Where would you change the output path or the model?

The output path is a CLI flag (`--output`); no code change. The model is isolated
in `src/llm_client.py`, behind the public SDK surface; no `torch` or
`transformers` import leaks into `src/`.

---

## Live drills

Target: any one drill in under 10 minutes, with the tests still green after
rollback. Practice on a branch or a temporary copy; never commit the drill.

### Drill A — add a function with an `integer` parameter (data only)

```bash
cp data/input/functions_definition.json /tmp/functions_def_drill.json
uv run python - <<'PY'
import json
from pathlib import Path
p = Path('/tmp/functions_def_drill.json')
data = json.loads(p.read_text())
data.append({
    "name": "fn_is_even",
    "description": "Return whether an integer is even.",
    "parameters": {"n": {"type": "integer"}},
    "returns": {"type": "boolean"},
})
p.write_text(json.dumps(data, indent=2) + "\n")
PY
uv run python - <<'PY'
from src.data_loader import load_functions
from src.prompt import build_prompt
from src.schemas import FunctionCall
cat = load_functions('/tmp/functions_def_drill.json')
assert len(cat.functions) == 7
assert 'fn_is_even(n:integer)' in build_prompt(cat, 'Is 4 even?')
call = FunctionCall.model_validate({"name": "fn_is_even", "parameters": {"n": 4}})
cat.validate_call(call)
print('accepted:', call.model_dump())
PY
```

What to say: the catalog is data; the prompt and name mask pick up the new
function without touching the decoder. `integer` is already a supported scalar
type; the strict validator rejects `true` for an integer.

### Drill B — add a parameter to an existing function (data only)

Add `"scale": {"type": "number"}` to `fn_add_numbers` in a temporary copy, then:

```bash
uv run python - <<'PY'
from src.data_loader import load_functions
from src.prompt import build_prompt
cat = load_functions('/tmp/functions_def_drill.json')
prompt = build_prompt(cat, 'Add 1 and 2, then scale by 10.')
line = next(l for l in prompt.splitlines() if l.startswith('fn_add_numbers'))
print(line)
assert 'scale:number' in line
PY
```

What to say: parameter order in the serialized call follows the declaration
order, and the DFA/template already covers the extra slot because it is driven
by the catalog.

### Drill C — change the output path (no code)

```bash
uv run python -m src --output /tmp/cmm-drill.json
```

What to say: the CLI owns the path; the writer validates the whole batch before
`os.replace`, so a failure never leaves a partial file.

### Drill D — add a validation (small code change)

Example: reject an empty function description in `schemas.py`, add one test,
run:

```bash
uv run pytest -q tests/test_schemas.py tests/test_validation.py
```

What to say: validation belongs at the boundary; the decoder should not carry
business rules that Pydantic can enforce once.

---

## Dry-run evidence (2026-11-11)

Drill A was executed against `/tmp/functions_def_drill.json`; the repository was
not modified.

```text
functions after edit: 7
catalog functions: 7
prompt line: fn_is_even(n:integer): Return whether an integer is even.
accepted: {'prompt': 'Is 4 even?', 'name': 'fn_is_even', 'parameters': {'n': 4}}
bool rejected: SchemaError function 'fn_is_even' parameter 'n' must be an integer (int, not bool), got bool
```

This confirms the data-driven path: a new function plus an `integer` parameter
needs no decoder change, and the strict type check rejects a boolean.

---

## Pre-defense checklist

- [ ] I can explain each module in `docs/ARCHITECTURE.md` without opening it.
- [ ] I can name the four constrained-decoding steps and say why `-inf`.
- [ ] I can explain fixed injection vs. model generation in one sentence.
- [ ] I can show 17/17 valid, 16/16 function, 10/16 arguments and 180.24 s.
- [ ] I can run Drill A in under 10 minutes and explain why no code changed.
- [ ] I can answer all 14 questions above in under 30 seconds each.
- [ ] I can point to the exact file for a new type, a new function, a new
      validation and a new output path.
- [ ] I know the honest limitation: validity is guaranteed, argument semantics
      is measured at 62.5%.

## Common traps

- Saying "the prompt guarantees JSON". The prompt asks; the mask enforces.
- Saying "the DFA validates the key order". Injection fixes the order; the DFA
  validates the structure.
- Claiming 100% argument accuracy. It is 10/16 (62.5%); function accuracy is
  16/16.
- Forgetting that a wrong call aborts the batch and leaves no partial output.
- Calling the name mask a heuristic. It is a grammar constraint; the logits
  still choose among valid prefixes.
