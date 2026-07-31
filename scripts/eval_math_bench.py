#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, median
from typing import Any

import httpx
from datasets import concatenate_datasets, load_dataset
from math_verify import parse, verify
from transformers import AutoTokenizer, PreTrainedTokenizerFast

DEFAULT_BASE_URL = "http://127.0.0.1:8000/v1"
DEFAULT_MODEL = "dapo-candidate"
DEFAULT_TOKENIZER = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
DEFAULT_TOKENIZER_REVISION = "ad9f0ae0864d7fbcd1cd905e3c6c5b069cc8b562"
PROMPT_PREFIX = "Solve the following math problem. Explain your reasoning and put the final answer in \\boxed{}.\n\n"


@dataclass(frozen=True)
class Problem:
    bench: str
    idx: int
    prompt: str
    answer: str


@dataclass(frozen=True)
class Result:
    bench: str
    idx: int
    answer: str
    prediction: str
    correct: bool
    closed_reasoning: bool
    finish_reason: str | None
    token_limit_hit: bool
    thinking_tokens: int
    completion_tokens: int | None
    response: str


def extract_last_boxed(text: str) -> str:
    start = text.rfind("\\boxed{")
    if start == -1:
        return ""
    i = start + len("\\boxed{")
    depth = 1
    while i < len(text) and depth:
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
        i += 1
    if depth:
        return ""
    return text[start + len("\\boxed{") : i - 1].strip()


def response_after_reasoning(text: str) -> str:
    if "<think>" in text and "</think>" not in text:
        return ""
    return text.split("</think>")[-1]


def score_answer(gold: str, response: str, timeout_seconds: int) -> bool:
    prediction = extract_last_boxed(response_after_reasoning(response))
    if not prediction:
        return False
    try:
        return bool(
            verify(
                parse(f"\\boxed{{{gold}}}", parsing_timeout=timeout_seconds),
                parse(f"\\boxed{{{prediction}}}", parsing_timeout=timeout_seconds),
                timeout_seconds=timeout_seconds,
            )
        )
    except Exception:
        return False


def load_tokenizer(name: str, revision: str) -> Any:
    if name == DEFAULT_TOKENIZER:
        return PreTrainedTokenizerFast.from_pretrained(name, revision=revision)
    return AutoTokenizer.from_pretrained(name, revision=revision)


def thinking_text(response: str, token_limit_hit: bool) -> tuple[str, bool]:
    if "</think>" in response:
        before_close = response.split("</think>", 1)[0]
        return before_close.removeprefix("<think>"), True
    if "<think>" in response:
        return response.split("<think>", 1)[1], False
    if token_limit_hit:
        return response, False
    return "", True


def load_problems(bench: str) -> list[Problem]:
    if bench == "aime24":
        rows = load_dataset(
            "HuggingFaceH4/aime_2024",
            split="train",
            revision="2fe88a2f1091d5048c0f36abc874fb997b3dd99a",
        )
        return [
            Problem(bench, i, PROMPT_PREFIX + row["problem"], str(int(row["answer"])))
            for i, row in enumerate(rows)
        ]
    if bench == "aime25":
        rows = concatenate_datasets(
            [
                load_dataset(
                    "opencompass/AIME2025",
                    subset,
                    split="test",
                    revision="a6ad95f611d72cf628a80b58bd0432ef6638f958",
                )
                for subset in ["AIME2025-I", "AIME2025-II"]
            ]
        )
        return [
            Problem(
                bench,
                i,
                PROMPT_PREFIX + row["question"],
                "".join(c for c in row["answer"] if c.isdigit() or c == "."),
            )
            for i, row in enumerate(rows)
        ]
    if bench == "math500":
        rows = load_dataset("HuggingFaceH4/MATH-500", split="test")
        return [
            Problem(bench, i, PROMPT_PREFIX + row["problem"], str(row["answer"]))
            for i, row in enumerate(rows)
        ]
    if bench == "gsm8k":
        rows = load_dataset("openai/gsm8k", "main", split="test")
        return [
            Problem(bench, i, PROMPT_PREFIX + row["question"], row["answer"].split("####", 1)[1].strip())
            for i, row in enumerate(rows)
        ]
    raise ValueError(f"unsupported benchmark: {bench}")


async def complete(
    client: httpx.AsyncClient,
    problem: Problem,
    args: argparse.Namespace,
    tokenizer: Any,
) -> Result:
    response = await client.post(
        "/chat/completions",
        json={
            "model": args.model,
            "messages": [{"role": "user", "content": problem.prompt}],
            "temperature": args.temperature,
            "top_p": args.top_p,
            "max_tokens": args.max_tokens,
        },
    )
    response.raise_for_status()
    payload = response.json()
    choice = payload["choices"][0]
    message = choice["message"]
    text = message.get("content") or ""
    if reasoning := message.get("reasoning_content"):
        text = f"<think>{reasoning}</think>\n{text}"
    prediction = extract_last_boxed(response_after_reasoning(text))
    usage = payload.get("usage") or {}
    completion_tokens = usage.get("completion_tokens")
    finish_reason = choice.get("finish_reason")
    token_limit_hit = finish_reason == "length" or completion_tokens == args.max_tokens
    thought, closed = thinking_text(text, token_limit_hit)
    thinking_tokens = len(tokenizer.encode(thought, add_special_tokens=False))
    if completion_tokens is not None:
        thinking_tokens = min(thinking_tokens, completion_tokens)
    return Result(
        bench=problem.bench,
        idx=problem.idx,
        answer=problem.answer,
        prediction=prediction,
        correct=score_answer(problem.answer, text, args.verify_timeout),
        closed_reasoning=closed,
        finish_reason=finish_reason,
        token_limit_hit=token_limit_hit,
        thinking_tokens=thinking_tokens,
        completion_tokens=completion_tokens,
        response=text,
    )


async def run(args: argparse.Namespace) -> list[Result]:
    problems: list[Problem] = []
    for bench in args.bench:
        rows = load_problems(bench)
        problems.extend(rows[: args.limit] if args.limit else rows)

    tokenizer = load_tokenizer(args.tokenizer, args.tokenizer_revision)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    limits = httpx.Limits(max_connections=args.concurrency)
    async with httpx.AsyncClient(base_url=args.base_url, timeout=None, limits=limits) as client:
        semaphore = asyncio.Semaphore(args.concurrency)

        async def guarded(problem: Problem) -> Result:
            async with semaphore:
                return await complete(client, problem, args, tokenizer)

        tasks = [asyncio.create_task(guarded(problem)) for problem in problems]
        results: list[Result] = []
        with args.output.open("w") as f:
            for i, task in enumerate(asyncio.as_completed(tasks), start=1):
                result = await task
                results.append(result)
                f.write(json.dumps(result.__dict__, ensure_ascii=False) + "\n")
                f.flush()
                print(
                    f"[{i}/{len(tasks)}] {result.bench}#{result.idx} "
                    f"correct={int(result.correct)} thinking={result.thinking_tokens} "
                    f"finish={result.finish_reason} limit={int(result.token_limit_hit)}",
                    flush=True,
                )
        return results


def summarize(results: Sequence[Result]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for bench in sorted({result.bench for result in results}):
        rows = [result for result in results if result.bench == bench]
        correct_rows = [result for result in rows if result.correct]
        thinking = [result.thinking_tokens for result in rows]
        correct_thinking = [result.thinking_tokens for result in correct_rows]
        completion_tokens = [result.completion_tokens for result in rows if result.completion_tokens is not None]
        finish_reasons = sorted({str(result.finish_reason) for result in rows})
        summary[bench] = {
            "n": len(rows),
            "correct": sum(result.correct for result in rows),
            "accuracy": sum(result.correct for result in rows) / len(rows),
            "closed_reasoning_rate": sum(result.closed_reasoning for result in rows) / len(rows),
            "token_limit_hits": sum(result.token_limit_hit for result in rows),
            "finish_reasons": finish_reasons,
            "thinking_tokens_mean": mean(thinking),
            "thinking_tokens_median": median(thinking),
            "correct_thinking_tokens_mean": mean(correct_thinking) if correct_thinking else None,
            "correct_thinking_tokens_median": median(correct_thinking) if correct_thinking else None,
            "completion_tokens_mean": mean(completion_tokens) if completion_tokens else None,
            "completion_tokens_median": median(completion_tokens) if completion_tokens else None,
        }
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a served model on small math benchmarks.")
    parser.add_argument("--bench", action="append", choices=["aime24", "aime25", "math500", "gsm8k"], required=True)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--tokenizer", default=DEFAULT_TOKENIZER)
    parser.add_argument("--tokenizer-revision", default=DEFAULT_TOKENIZER_REVISION)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=0, help="Per-benchmark limit; 0 means full benchmark.")
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-tokens", type=int, default=16384)
    parser.add_argument("--verify-timeout", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results = asyncio.run(run(args))
    print(json.dumps(summarize(results), indent=2))


if __name__ == "__main__":
    main()
