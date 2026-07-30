"""DeepScaleR math tasks with the original training prompt and grader format."""

from typing import Literal

import verifiers.v1 as vf

DEEPSCALER_DATASET_NAME = "agentica-org/DeepScaleR-Preview-Dataset"
DEEPSCALER_DATASET_REVISION = "b6ae8c60f5c1f2b594e2140b91c49c9ad0949e29"
AIME24_DATASET_NAME = "HuggingFaceH4/aime_2024"
AIME24_DATASET_REVISION = "2fe88a2f1091d5048c0f36abc874fb997b3dd99a"
INSTRUCTION = "Let's think step by step and output the final answer within \\boxed{}."
ANSWER_CORRECTIONS = {
    3153: r"\left\{3,-\frac{3}{2}+\frac{3i\sqrt{3}}{2},-\frac{3}{2}-\frac{3i\sqrt{3}}{2}\right\}",
    4949: r"\frac{2 k n}{n+k-1}-2 \frac{k!n!}{(k+n-1)!}",
}


class DeepScaleRData(vf.TaskData):
    answer: str


class DeepScaleRTaskConfig(vf.TaskConfig):
    math_verify_timeout: int = 5


def has_deepscaler_format(response: str) -> bool:
    if response.count("<think>") != 1 or response.count("</think>") != 1:
        return False
    think_start = response.find("<think>")
    think_end = response.rfind("</think>")
    if think_end < think_start:
        return False
    final_answer = response[think_end + len("</think>") :]
    return bool(vf.extract_boxed_answer(final_answer, strict=True).strip())


class DeepScaleRTask(vf.Task[DeepScaleRData, vf.State, DeepScaleRTaskConfig]):
    @vf.metric
    async def exact_format(self, trace: vf.Trace) -> float:
        return float(has_deepscaler_format(trace.last_reply))

    @vf.reward(weight=1.0)
    async def correct(self, trace: vf.Trace) -> float:
        if not has_deepscaler_format(trace.last_reply):
            return 0.0
        return vf.verify_boxed_math_answer(
            trace.last_reply,
            self.data.answer,
            timeout_seconds=self.config.math_verify_timeout,
        )


class DeepScaleRConfig(vf.TasksetConfig):
    dataset: Literal["train", "aime24"] = "train"
    task: DeepScaleRTaskConfig = DeepScaleRTaskConfig()


class DeepScaleRTaskset(vf.Taskset[DeepScaleRTask, DeepScaleRConfig]):
    def load(self) -> list[DeepScaleRTask]:
        from datasets import load_dataset

        if self.config.dataset == "aime24":
            rows = load_dataset(
                AIME24_DATASET_NAME,
                split="train",
                revision=AIME24_DATASET_REVISION,
            )
            examples = ((index, str(row["problem"]), str(int(row["answer"]))) for index, row in enumerate(rows))
        else:
            rows = load_dataset(
                DEEPSCALER_DATASET_NAME,
                split="train",
                revision=DEEPSCALER_DATASET_REVISION,
            )
            examples = (
                (
                    index,
                    str(row["problem"]),
                    ANSWER_CORRECTIONS.get(index, str(row["answer"]).strip()),
                )
                for index, row in enumerate(rows)
            )

        return [
            DeepScaleRTask(
                DeepScaleRData(
                    idx=index,
                    prompt=problem + " " + INSTRUCTION,
                    answer=answer,
                ),
                self.config.task,
            )
            for index, problem, answer in examples
            if answer
        ]
