# Test fixtures (T1.4)

These files are **hand-authored for this project** to exercise
`src/data_loader.py` and, later, the fake pipeline. They are not taken from the
subject. Files marked as invalid are intentionally malformed or schema-invalid
and must not be used as real inputs.

## Valid fixtures

| File | Purpose | Expected loader outcome |
|------|---------|-------------------------|
| `functions_valid_edge.json` | Function catalog covering `string`, `number`, `boolean`, `integer` and a 3-parameter function. | `load_functions` succeeds (5 functions). |
| `prompts_valid_edge.json` | Prompts covering a negative number, escaped quotes/Unicode, a boolean request and an ambiguous prompt. | `load_prompts` succeeds (4 prompts, order preserved). |

## Invalid function fixtures

| File | Purpose | Expected loader outcome |
|------|---------|-------------------------|
| `functions_empty.json` | Empty root list. | `InputError`: "at least one function". |
| `functions_unknown_type.json` | Parameter type `array`, unsupported by the grammar. | `SchemaError`: "unsupported parameter type". |
| `functions_duplicate_name.json` | Two functions share one name. | `SchemaError`: "duplicate". |
| `functions_missing_field.json` | Function without the required `returns` field. | `SchemaError`: "invalid function catalog". |
| `functions_broken.json` | Truncated JSON array. | `InputError`: "invalid JSON" and "line". |
| `functions_root_object.json` | Root is `{}` instead of a list. | `SchemaError`: "invalid function catalog". |

## Invalid prompt fixtures

| File | Purpose | Expected loader outcome |
|------|---------|-------------------------|
| `prompts_empty.json` | Empty root list. | `InputError`: "at least one prompt". |
| `prompts_broken.json` | Truncated JSON array. | `InputError`: "invalid JSON". |
| `prompts_not_list.json` | Root is `{}` instead of a list. | `SchemaError`: "expected a JSON list". |
| `prompts_missing_key.json` | Second entry lacks `prompt`. | `SchemaError`: "index 1" and "missing". |
| `prompts_blank.json` | Prompt is whitespace only. | `InputError`: "blank". |
| `prompts_not_object.json` | Entry is a bare string. | `SchemaError`: "index 0" and "must be an object". |
