"""Smoke test for the provided LLM SDK (T0.3).

Exercises only the public API: model loading, encode/decode round-trip,
one forward pass and latency measurement.

Run with: uv run python experiments/smoke.py
"""

from __future__ import annotations

import json
import platform
import statistics
import sys
import time
from pathlib import Path

from llm_sdk import Small_LLM_Model

REPORT = Path("experiments/smoke_report.json")
SAMPLES = ["hello", "olá", '{"a": 1}', 'name="shrek"']


def timed(label: str, fn):
    """Run fn once and return (result, elapsed_seconds)."""
    start = time.perf_counter()
    result = fn()
    elapsed = time.perf_counter() - start
    print(f"{label}: {elapsed:.3f}s")
    return result, elapsed


def main() -> int:
    report: dict[str, object] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
    }

    model, load_s = timed("load model", Small_LLM_Model)
    report["load_s"] = round(load_s, 3)

    for getter in ("get_path_to_vocab_file", "get_path_to_tokenizer_file"):
        try:
            path = getattr(model, getter)()
            print(f"{getter}: {path}")
            report[getter] = path
        except Exception as exc:  # record and continue: informs T2.1
            print(f"{getter}: FAILED -> {type(exc).__name__}: {exc}")
            report[getter] = f"ERROR: {type(exc).__name__}"

    # encode -> decode round-trip
    roundtrip: dict[str, bool] = {}
    for text in SAMPLES:
        ids = model.encode(text)[0].tolist()
        back = model.decode(ids)
        roundtrip[text] = back == text
        print(f"roundtrip {text!r}: ids={ids} -> {back!r} equal={back == text}")
    report["roundtrip"] = roundtrip

    # logits shape + latency for the next-token distribution
    ids = model.encode("What is the sum of 2 and 3?")[0].tolist()
    times = []
    logits = []
    for _ in range(3):
        logits, elapsed = timed("forward", lambda: model.get_logits_from_input_ids(ids))
        times.append(elapsed)
    report["prompt_tokens"] = len(ids)
    report["logits_len"] = len(logits)
    report["forward_times_s"] = [round(t, 3) for t in times]
    report["forward_median_s"] = round(statistics.median(times), 3)
    report["logits_are_finite"] = all(x == x and abs(x) != float("inf") for x in logits)

    REPORT.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
