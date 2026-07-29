from collections.abc import Iterator

import verifiers.v1 as vf

PROMPT = "Write a super short story about a boy's morning routine."
REWARD_CAP = 50


def up_count(response: str | None) -> int:
    return (response or "").count("up")


class UpStoryTask(vf.Task[vf.TaskData]):
    @vf.stop
    async def single_turn(self, trace: vf.Trace) -> bool:
        return trace.num_turns >= 1

    @vf.reward(weight=1.0)
    async def up_reward(self, trace: vf.Trace) -> float:
        return min(up_count(trace.last_reply), REWARD_CAP) / REWARD_CAP

    @vf.metric
    async def ups(self, trace: vf.Trace) -> float:
        return float(up_count(trace.last_reply))

    @vf.metric
    async def reward_cap_reached(self, trace: vf.Trace) -> float:
        return float(up_count(trace.last_reply) >= REWARD_CAP)


class UpStoryTaskset(vf.Taskset[UpStoryTask, vf.TasksetConfig]):
    INFINITE = True

    def load(self) -> Iterator[UpStoryTask]:
        index = 0
        while True:
            yield UpStoryTask(vf.TaskData(idx=index, prompt=PROMPT), self.config.task)
            index += 1
