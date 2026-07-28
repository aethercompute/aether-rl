import random
import re
import string
from typing import Literal

import verifiers.v1 as vf

SYSTEM_PROMPT = "Reverse the provided text. Reply with only the reversed text and no explanation."
PLAIN_RESPONSE = re.compile(r"[a-z]+")


def parse_response(response: str | None) -> str:
    value = (response or "").strip()
    if "\n" in value:
        return ""
    for quote in ('"', "'", "`"):
        if len(value) >= 2 and value.startswith(quote) and value.endswith(quote):
            value = value[1:-1].strip()
            break
    return value.lower() if PLAIN_RESPONSE.fullmatch(value.lower()) else ""


class ReverseTextProofData(vf.TaskData):
    answer: str


class ReverseTextProofTask(vf.Task[ReverseTextProofData]):
    @vf.stop
    async def single_turn(self, trace: vf.Trace) -> bool:
        return trace.num_turns >= 1

    @vf.reward(weight=1.0)
    async def character_accuracy(self, trace: vf.Trace) -> float:
        prediction = parse_response(trace.last_reply)
        if not prediction:
            return 0.0
        correct = sum(left == right for left, right in zip(prediction, self.data.answer, strict=False))
        return correct / max(len(prediction), len(self.data.answer))

    @vf.metric
    async def exact_match(self, trace: vf.Trace) -> float:
        return float(parse_response(trace.last_reply) == self.data.answer)

    @vf.metric
    async def exact_format(self, trace: vf.Trace) -> float:
        response = (trace.last_reply or "").strip()
        return float(bool(PLAIN_RESPONSE.fullmatch(response)))

    @vf.metric
    async def length_accuracy(self, trace: vf.Trace) -> float:
        return float(len(parse_response(trace.last_reply)) == len(self.data.answer))


class ReverseTextProofConfig(vf.TasksetConfig):
    split: Literal["train", "eval"] = "train"


class ReverseTextProofTaskset(vf.Taskset[ReverseTextProofTask, ReverseTextProofConfig]):
    def load(self) -> list[ReverseTextProofTask]:
        train_inputs = _generate_inputs(seed=100, count=512)
        inputs = (
            train_inputs
            if self.config.split == "train"
            else _generate_inputs(seed=200, count=128, excluded=set(train_inputs))
        )
        return [
            ReverseTextProofTask(
                ReverseTextProofData(
                    idx=index,
                    prompt=f"Reverse this text: {text}",
                    system_prompt=SYSTEM_PROMPT,
                    answer=text[::-1],
                ),
                self.config.task,
            )
            for index, text in enumerate(inputs)
        ]


def _generate_inputs(*, seed: int, count: int, excluded: set[str] | None = None) -> list[str]:
    rng = random.Random(seed)
    used = set() if excluded is None else set(excluded)
    generated = []
    while len(generated) < count:
        value = "".join(rng.choices(string.ascii_lowercase, k=rng.randint(3, 5)))
        if value in used:
            continue
        used.add(value)
        generated.append(value)
    return generated
