"""End-to-end pipeline smoke test with a deterministic fake or the real model.

The fake mode is fully offline: it loads the real catalog, builds a local
character-token index, and scripts the decoder to call ``fn_add_numbers(2, 3)``.
The optional real mode exercises the same ``run`` and ``write_output`` path with
``LLMClient``; the orchestrator runs it separately because loading Qwen is slow.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.data_loader import load_functions  # noqa: E402
from src.decoder import decode_call  # noqa: E402
from src.errors import ProjectError  # noqa: E402
from src.llm_client import LLMClient  # noqa: E402
from src.pipeline import build_decoder_index, run, write_output  # noqa: E402
from src.schemas import FunctionCall, FunctionCatalog  # noqa: E402
from src.vocab import TokenizerIndex  # noqa: E402

__all__ = [
    "DEFAULT_FAKE_OUTPUT",
    "DEFAULT_REAL_OUTPUT",
    "FAKE_PROMPT",
    "REAL_PROMPT",
    "main",
    "run_fake",
    "run_real",
]

REPO_ROOT = _PROJECT_ROOT
FUNCTIONS_PATH = REPO_ROOT / "data" / "input" / "functions_definition.json"
DEFAULT_FAKE_OUTPUT = Path("data/output/e2e_fake.json")
DEFAULT_REAL_OUTPUT = Path("data/output/e2e_real.json")
FAKE_PROMPT = "What is the sum of 2 and 3?"
REAL_PROMPT = FAKE_PROMPT

_EOS_ID = 1000
_EOS_TEXT = "<|im_end|>"
_FAKE_CHARS = 'abcdefghijklmnopqrstuvwxyz_0123456789.-\\" '


def _tokens(text: str) -> list[int]:
    """Return one character token id per character."""
    return [ord(char) for char in text]


def _fake_index(catalog: FunctionCatalog) -> TokenizerIndex:
    """Build a char-token index covering the real catalog names and numbers."""
    chars = set(_FAKE_CHARS)
    for function in catalog:
        chars.update(function.name)
    texts = {ord(char): char for char in sorted(chars)}
    texts[_EOS_ID] = _EOS_TEXT
    allowed_ids = tuple(
        token_id for token_id in sorted(texts) if token_id != _EOS_ID
    )

    buckets: dict[str, list[int]] = {}
    for token_id in allowed_ids:
        text = texts[token_id]
        if text:
            buckets.setdefault(text[0], []).append(token_id)
    return TokenizerIndex(
        texts=texts,
        allowed_ids=allowed_ids,
        by_first_char={
            char: tuple(token_ids) for char, token_ids in buckets.items()
        },
    )


class _ScriptedBackend:
    """Minimal local backend that plays a fixed next-token script."""

    def __init__(self, script: list[int], *, size: int = 2048) -> None:
        self.script = list(script)
        self.size = size
        self.calls = 0
        self.decode_calls: list[list[int]] = []

    def encode(self, text: str) -> list[list[int]]:
        """Return one character-id row, like the SDK's 2-D tensor."""
        return [[ord(char) for char in text]]

    def decode(self, ids: list[int]) -> str:
        """Record and join character ids."""
        self.decode_calls.append(list(ids))
        return "".join(chr(token_id) for token_id in ids)

    def get_logits_from_input_ids(self, input_ids: list[int]) -> list[float]:
        """Return logits favouring the next scripted token."""
        self.calls += 1
        if not self.script:
            raise RuntimeError("script exhausted before decoding finished")
        token_id = self.script.pop(0)
        logits = [-1.0] * self.size
        logits[token_id] = 10.0
        return logits

    def get_path_to_vocab_file(self) -> str:
        """Return an unused path; fake mode never loads real assets."""
        return "unused"

    def get_path_to_tokenizer_file(self) -> str:
        """Return an unused path; fake mode injects its own index."""
        return "unused"


def _sum_script() -> list[int]:
    """Script ``fn_add_numbers`` with number arguments 2 and 3."""
    return (
        _tokens("fn_add_numbers")
        + [_EOS_ID]
        + _tokens("2")
        + [_EOS_ID]
        + _tokens("3")
        + [_EOS_ID]
    )


def _expected_sum_call(prompt: str) -> FunctionCall:
    return FunctionCall(
        prompt=prompt,
        name="fn_add_numbers",
        parameters={"a": 2, "b": 3},
    )


def _check_sum_call(call: FunctionCall, prompt: str) -> None:
    expected = _expected_sum_call(prompt)
    if call != expected:
        raise RuntimeError(
            f"unexpected call: expected {expected.model_dump(mode='json')}, "
            f"got {call.model_dump(mode='json')}"
        )
    for name, value in call.parameters.items():
        if type(value) is not int:
            raise RuntimeError(
                f"parameter {name!r} must be int, got {type(value).__name__}"
            )


def run_fake(output: Path = DEFAULT_FAKE_OUTPUT) -> list[FunctionCall]:
    """Run the offline end-to-end path and write ``output``."""
    catalog = load_functions(FUNCTIONS_PATH)
    model = LLMClient(backend=_ScriptedBackend(_sum_script()))
    call = decode_call(
        model,
        catalog,
        FAKE_PROMPT,
        index=_fake_index(catalog),
    )
    catalog.validate_call(call)
    _check_sum_call(call, FAKE_PROMPT)
    write_output(output, [call])
    return [call]


def run_real(
    prompt: str = REAL_PROMPT,
    output: Path = DEFAULT_REAL_OUTPUT,
) -> list[FunctionCall]:
    """Run one prompt through the real model and write ``output``."""
    model = LLMClient()
    index = build_decoder_index(model)
    with tempfile.TemporaryDirectory() as tmp_dir:
        input_path = Path(tmp_dir) / "e2e_input.json"
        input_path.write_text(
            json.dumps([{"prompt": prompt}], ensure_ascii=False),
            encoding="utf-8",
        )
        started = time.perf_counter()
        calls = run(
            FUNCTIONS_PATH,
            input_path,
            model=model,
            index=index,
        )
        elapsed_s = time.perf_counter() - started

    if len(calls) != 1:
        raise RuntimeError(f"expected 1 call, got {len(calls)}")
    catalog = load_functions(FUNCTIONS_PATH)
    catalog.validate_call(calls[0])
    if calls[0].prompt != prompt:
        raise RuntimeError(
            f"prompt changed: expected {prompt!r}, got {calls[0].prompt!r}"
        )
    write_output(output, calls)

    print(f"elapsed_s={elapsed_s:.6f}")
    print(json.dumps(calls[0].model_dump(mode="json"), ensure_ascii=False))
    print(
        "types="
        + json.dumps(
            {
                name: type(value).__name__
                for name, value in calls[0].parameters.items()
            },
            sort_keys=True,
        )
    )
    return calls


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--real",
        action="store_true",
        help="run one prompt with the real model instead of the fake",
    )
    parser.add_argument(
        "--prompt",
        default=REAL_PROMPT,
        help="prompt used by --real (default: the sum prompt)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="output path; defaults to data/output/e2e_{fake,real}.json",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point; return a process exit code."""
    args = _parse_args(argv)
    output = args.output
    try:
        if args.real:
            real_output = output or DEFAULT_REAL_OUTPUT
            run_real(args.prompt, real_output)
            print(f"output={real_output}")
            return 0

        fake_output = output or DEFAULT_FAKE_OUTPUT
        started = time.perf_counter()
        calls = run_fake(fake_output)
        elapsed_s = time.perf_counter() - started
        print(f"output={fake_output}")
        print(json.dumps(calls[0].model_dump(mode="json"), ensure_ascii=False))
        print(f"elapsed_s={elapsed_s:.6f}")
        return 0
    except (ProjectError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
