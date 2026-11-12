"""Offline end-to-end coverage for the T6.4 fake experiment."""

from __future__ import annotations

import json
from pathlib import Path

from experiments.e2e import FAKE_PROMPT, run_fake


def test_run_fake_writes_expected_sum_call(tmp_path: Path) -> None:
    output = tmp_path / "nested" / "e2e_fake.json"

    calls = run_fake(output)

    assert len(calls) == 1
    call = calls[0]
    assert call.prompt == FAKE_PROMPT
    assert call.name == "fn_add_numbers"
    assert call.parameters == {"a": 2.0, "b": 3.0}
    assert type(call.parameters["a"]) is float
    assert type(call.parameters["b"]) is float

    assert output.exists()
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload == [
        {
            "prompt": FAKE_PROMPT,
            "name": "fn_add_numbers",
            "parameters": {"a": 2.0, "b": 3.0},
        }
    ]
