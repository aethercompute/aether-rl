import hashlib
import random
import re
from typing import Literal

import verifiers.v1 as vf

CODEWORDS = ("dax", "wug", "zorp", "kiv")
MARKERS = ("A", "B", "C", "D")
CODEWORD_PATTERN = re.compile(r"\b(dax|wug|zorp|kiv)\b", re.IGNORECASE)
SYSTEM_PROMPT = (
    "You are learning a fixed hidden mapping from marker letters to codewords using reward feedback. "
    "Reply with exactly one allowed codeword and no other text."
)


class CodewordData(vf.TaskData):
    answer: str


class CodewordTask(vf.Task[CodewordData]):
    @vf.stop
    async def single_turn(self, trace: vf.Trace) -> bool:
        return trace.num_turns >= 1

    @vf.reward(weight=1.0)
    async def accuracy(self, trace: vf.Trace) -> float:
        response = (trace.last_reply or "").strip().lower()
        matches = CODEWORD_PATTERN.findall(response)
        return float(len(matches) == 1 and matches[0].lower() == self.data.answer)

    @vf.metric
    async def exact_format(self, trace: vf.Trace) -> float:
        return float((trace.last_reply or "").strip().lower() in CODEWORDS)


class CodewordConfig(vf.TasksetConfig):
    split: Literal["train", "eval"] = "train"


class CodewordTaskset(vf.Taskset[CodewordTask, CodewordConfig]):
    def load(self) -> list[CodewordTask]:
        size = 256 if self.config.split == "train" else 64
        tasks = []
        for index in range(size):
            identity = hashlib.sha256(f"codeword-v1:{self.config.split}:{index}".encode()).hexdigest()
            choices = list(CODEWORDS)
            random.Random(int(identity, 16)).shuffle(choices)
            tasks.append(
                CodewordTask(
                    CodewordData(
                        idx=index,
                        prompt=(
                            f"Marker: {MARKERS[index % len(MARKERS)]}\n"
                            f"Case: {identity[:12]}\n"
                            f"Allowed codewords: {', '.join(choices)}"
                        ),
                        system_prompt=SYSTEM_PROMPT,
                        answer=CODEWORDS[index % len(CODEWORDS)],
                    ),
                    self.config.task,
                )
            )
        return tasks
