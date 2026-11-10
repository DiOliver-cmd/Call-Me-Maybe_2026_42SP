"""Measure the real decoding budget end to end (T8.4).

The subject requires the whole run over the prompt fixture to finish in under
five minutes. This script reproduces the production path (``src.pipeline``)
with the real model and records where the time goes:

* model load (first ``encode`` triggers SDK construction);
* decoder index build (token-text cache hit or rebuild);
* one row per prompt: wall time, model steps (``logits`` calls) and prompt size.

Run with::

    uv run python -m experiments.time_budget
    uv run python -m experiments.time_budget --limit 3

The JSON report is gitignored; the summary table is copied into
``experiments/benchmark.md``.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from src.config import MAX_TOKENS, TOTAL_TIMEOUT_SECONDS
from src.data_loader import load_functions, load_prompts
from src.decoder import NUMBER_EOS_BONUS, decode_call
from src.llm_client import LLMClient
from src.pipeline import build_decoder_index
from src.prompt import build_prompt, build_verbose_prompt
from src.schemas import FunctionCatalog

__all__ = ["CountingClient", "measure", "main"]

FUNCTIONS = Path("data/input/functions_definition.json")
PROMPTS = Path("data/input/function_calling_tests.json")
REPORT = Path("experiments/time_budget_report.json")


class CountingClient(LLMClient):
    """An :class:`LLMClient` that counts and times its ``logits`` calls.

    Everything else is inherited unchanged, so the decoder and the index builder
    see the real client. Only the model forward is instrumented, because that is
    the cost that grows with prompt length.
    """

    def __init__(self) -> None:
        super().__init__()
        self.logits_calls = 0
        self.logits_seconds = 0.0

    def logits(self, ids: list[int]) -> Any:
        start = time.perf_counter()
        result = super().logits(ids)
        self.logits_seconds += time.perf_counter() - start
        self.logits_calls += 1
        return result


PROMPT_BUILDERS: dict[str, Callable[[FunctionCatalog, str], str]] = {
    "compact": build_prompt,
    "verbose": build_verbose_prompt,
}


def measure(limit: int | None = None, style: str = "compact") -> dict[str, Any]:
    """Run the real pipeline and return the timing report."""
    builder = PROMPT_BUILDERS[style]
    client = CountingClient()

    load_start = time.perf_counter()
    client.encode("warmup")
    load_s = time.perf_counter() - load_start

    index_start = time.perf_counter()
    index = build_decoder_index(client)
    index_s = time.perf_counter() - index_start

    catalog = load_functions(FUNCTIONS)
    prompts = load_prompts(PROMPTS)
    selected = prompts[:limit] if limit else prompts

    rows: list[dict[str, Any]] = []
    run_start = time.perf_counter()
    for position, prompt in enumerate(selected):
        calls_before = client.logits_calls
        seconds_before = client.logits_seconds
        start = time.perf_counter()

        call = decode_call(
            client,
            catalog,
            prompt,
            index=index,
            number_eos_bonus=NUMBER_EOS_BONUS,
            max_tokens=MAX_TOKENS,
            prompt_builder=builder,
        )
        catalog.validate_call(call)

        elapsed = time.perf_counter() - start
        steps = client.logits_calls - calls_before
        forward_s = client.logits_seconds - seconds_before
        prompt_tokens = len(client.encode(builder(catalog, prompt)))

        rows.append(
            {
                "index": position,
                "prompt": prompt,
                "prompt_tokens": prompt_tokens,
                "steps": steps,
                "seconds": round(elapsed, 3),
                "forward_seconds": round(forward_s, 3),
                "seconds_per_step": round(elapsed / steps, 3) if steps else None,
                "name": call.name,
                "parameters": call.model_dump(mode="json")["parameters"],
            }
        )
        print(
            f"[{position:02d}] {elapsed:7.2f}s  steps={steps:2d}  "
            f"prompt={prompt_tokens:3d} tok  {call.name}  {prompt!r}"
        )
    run_s = time.perf_counter() - run_start

    total_steps = sum(row["steps"] for row in rows)
    times = [row["seconds"] for row in rows]
    steps_per_prompt = total_steps / len(rows) if rows else 0.0

    report: dict[str, Any] = {
        "prompt_style": style,
        "max_tokens": MAX_TOKENS,
        "total_timeout_seconds": TOTAL_TIMEOUT_SECONDS,
        "prompts_in_fixture": len(prompts),
        "prompts_measured": len(rows),
        "load_s": round(load_s, 3),
        "index_build_s": round(index_s, 3),
        "decode_total_s": round(run_s, 3),
        "wall_total_s": round(load_s + index_s + run_s, 3),
        "total_steps": total_steps,
        "steps_per_prompt": round(steps_per_prompt, 2),
        "seconds_per_prompt_mean": round(statistics.mean(times), 3),
        "seconds_per_prompt_median": round(statistics.median(times), 3),
        "seconds_per_step_mean": round(
            sum(row["forward_seconds"] for row in rows) / total_steps, 3
        )
        if total_steps
        else None,
        "rows": rows,
    }

    # Project the full fixture when only a prefix was measured.
    if len(rows) < len(prompts) and rows:
        remaining = len(prompts) - len(rows)
        report["projected_decode_s"] = round(
            run_s + remaining * statistics.mean(times), 3
        )
        report["projected_wall_s"] = round(
            load_s + index_s + report["projected_decode_s"], 3
        )
    return report


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point: measure, write the JSON report and print a summary."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="measure only the first N prompts (default: all)",
    )
    parser.add_argument(
        "--prompt-style",
        choices=sorted(PROMPT_BUILDERS),
        default="compact",
        help="prompt format to measure (default: %(default)s)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPORT,
        help="JSON report path (default: %(default)s)",
    )
    args = parser.parse_args(argv)

    report = measure(args.limit, args.prompt_style)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print()
    print(f"style:       {report['prompt_style']}")
    print(f"load:        {report['load_s']} s")
    print(f"index:       {report['index_build_s']} s")
    print(f"decode:      {report['decode_total_s']} s")
    print(f"wall total:  {report['wall_total_s']} s")
    print(
        f"steps:       {report['total_steps']} total, "
        f"{report['steps_per_prompt']} per prompt"
    )
    print(f"per prompt:  {report['seconds_per_prompt_mean']} s mean, "
          f"{report['seconds_per_prompt_median']} s median")
    if "projected_wall_s" in report:
        print(f"projected:   {report['projected_wall_s']} s for the fixture")
    print(f"report:      {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
