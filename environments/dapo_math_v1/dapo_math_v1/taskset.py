from __future__ import annotations

import hashlib
import re
import unicodedata
from functools import cache
from typing import Literal

import verifiers.v1 as vf
from pydantic import Field
from transformers import PreTrainedTokenizerFast
from verifiers.v1.types import AssistantMessage

DATASET_NAME = "open-r1/DAPO-Math-17k-Processed"
DATASET_REVISION = "31dd309567e3da778038cc87d868b6097a3ccf68"
MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
MODEL_REVISION = "ad9f0ae0864d7fbcd1cd905e3c6c5b069cc8b562"
FINAL_ANSWER = re.compile(r"\s*Answer:\s*\\boxed\{([+-]?\d+)\}\s*")
BOXED_INTEGER = re.compile(r"\\boxed\{([+-]?\d+)\}")
INTEGER = re.compile(r"[+-]?\d+")
PROMPT_TEMPLATE = (
    "Solve this math problem step by step:\n\n{question}\n\n"
    "After your reasoning, output exactly one final line in this format: "
    "Answer: \\boxed{{<integer>}}"
)


def extract_final_integer(response: str) -> int | None:
    match = FINAL_ANSWER.fullmatch(response)
    return int(match.group(1)) if match is not None else None


def extract_last_boxed_integer(response: str) -> int | None:
    matches = BOXED_INTEGER.findall(response)
    return int(matches[-1]) if matches else None


def partition_bucket(question: str, modulus: int) -> int:
    normalized = " ".join(unicodedata.normalize("NFKC", question).casefold().split())
    return int.from_bytes(hashlib.sha256(normalized.encode()).digest()[:8], "big") % modulus


@cache
def _tokenizer(name: str, revision: str):
    return PreTrainedTokenizerFast.from_pretrained(name, revision=revision)


def thinking_token_count(trace: vf.Trace, tokenizer) -> int:
    node = next(
        (node for node in reversed(trace.nodes) if node.sampled and isinstance(node.message, AssistantMessage)),
        None,
    )
    if node is None:
        return 0
    completion_ids = [token_id for token_id, sampled in zip(node.token_ids, node.mask, strict=True) if sampled]
    marker_ids = tokenizer.encode("</think>", add_special_tokens=False)
    marker_length = len(marker_ids)
    for index in range(len(completion_ids) - marker_length + 1):
        if completion_ids[index : index + marker_length] == marker_ids:
            return index
    return len(completion_ids)


class DAPOMathData(vf.TaskData):
    question: str
    answer: int


class DAPOMathConfig(vf.TasksetConfig):
    partition: Literal["train", "eval"] = "train"
    holdout_modulus: int = Field(default=16, ge=2)


class DAPOMathTask(vf.Task[DAPOMathData]):
    @vf.stop
    async def single_turn(self, trace: vf.Trace) -> bool:
        return trace.num_turns >= 1

    @vf.reward(weight=1.0)
    async def correct(self, trace: vf.Trace) -> float:
        messages = trace.assistant_messages
        if trace.is_truncated or not messages or messages[-1].reasoning_content is None:
            return 0.0
        prediction = extract_last_boxed_integer(trace.last_reply)
        return float(prediction is not None and prediction == self.data.answer)

    @vf.metric
    async def format_valid(self, trace: vf.Trace) -> float:
        messages = trace.assistant_messages
        return float(
            not trace.is_truncated
            and bool(messages)
            and messages[-1].reasoning_content is not None
            and extract_final_integer(trace.last_reply) is not None
        )

    @vf.metric
    async def thinking_tokens(self, trace: vf.Trace) -> float:
        tokenizer = _tokenizer(MODEL_NAME, MODEL_REVISION)
        return float(thinking_token_count(trace, tokenizer))


class DAPOMathTaskset(vf.Taskset[DAPOMathTask, DAPOMathConfig]):
    def load(self) -> list[DAPOMathTask]:
        from datasets import load_dataset

        rows = load_dataset(
            DATASET_NAME,
            "en",
            split="train",
            revision=DATASET_REVISION,
        )
        tasks: list[DAPOMathTask] = []
        for index, row in enumerate(rows):
            question = str(row["prompt"])
            solution = str(row["solution"])
            if INTEGER.fullmatch(solution) is None:
                raise ValueError(f"non-integer DAPO gold at row {index}: {solution!r}")
            is_eval = partition_bucket(question, self.config.holdout_modulus) == 0
            if is_eval != (self.config.partition == "eval"):
                continue
            tasks.append(
                DAPOMathTask(
                    DAPOMathData(
                        idx=index,
                        name=str(row["extra_info"]["index"]),
                        prompt=PROMPT_TEMPLATE.format(question=question),
                        question=question,
                        answer=int(solution),
                    ),
                    self.config.task,
                )
            )
        return tasks
