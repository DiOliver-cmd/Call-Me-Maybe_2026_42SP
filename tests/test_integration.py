"""Slow end-to-end tests with the real model (T9.2).

These tests exercise the production path (``main`` -> ``pipeline.run`` ->
decoder -> catalog validation -> writer) with the real SDK. They are marked
``slow`` so ``pytest -m "not slow"`` never imports torch or the model weights.

Run them explicitly with::

    uv run pytest -q -m slow tests/test_integration.py
    make test-all
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from src.__main__ import main
from src.data_loader import load_functions, load_prompts
from src.schemas import FunctionCallBatch

pytestmark = pytest.mark.slow

FUNCTIONS = Path("data/input/functions_definition.json")
PROMPTS = Path("data/input/function_calling_tests.json")

# Indices in function_calling_tests.json: sum, greeting and reverse.
CASES = [
    (0, "fn_add_numbers"),
    (2, "fn_greet"),
    (4, "fn_reverse_string"),
]


def test_real_model_three_prompts_end_to_end(tmp_path: Path) -> None:
    """Decode three representative prompts and validate the written JSON."""
    prompts = load_prompts(PROMPTS)
    selected = [prompts[index] for index, _ in CASES]

    input_path = tmp_path / "prompts.json"
    input_path.write_text(
        json.dumps([{"prompt": prompt} for prompt in selected]),
        encoding="utf-8",
    )
    output_path = tmp_path / "function_calling_results.json"

    started = time.perf_counter()
    code = main(
        [
            "--functions_definition",
            str(FUNCTIONS),
            "--input",
            str(input_path),
            "--output",
            str(output_path),
        ]
    )
    elapsed = time.perf_counter() - started

    assert code == 0
    assert output_path.exists()

    raw = json.loads(output_path.read_text(encoding="utf-8"))
    batch = FunctionCallBatch.model_validate(raw)
    catalog = load_functions(FUNCTIONS)

    assert len(batch) == len(selected)
    assert [call.prompt for call in batch] == selected
    assert [call.name for call in batch] == [name for _, name in CASES]

    for call in batch:
        # Final safety net: names, required keys and strict scalar types.
        catalog.validate_call(call)

    sum_call, greet_call, reverse_call = batch.calls
    assert type(sum_call.parameters["a"]) in (int, float)
    assert type(sum_call.parameters["b"]) in (int, float)
    assert isinstance(greet_call.parameters["name"], str)
    assert isinstance(reverse_call.parameters["s"], str)

    # The subject budget is five minutes for the full fixture; this is a much
    # smaller run, so a generous bound still catches a pathological regression.
    assert elapsed < 300, f"real-model run took {elapsed:.1f}s"
