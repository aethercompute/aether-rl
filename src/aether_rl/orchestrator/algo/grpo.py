from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

from aether_rl.configs.algorithm import GRPOAlgoConfig
from aether_rl.orchestrator.algo.base import Algorithm

if TYPE_CHECKING:
    from aether_rl.orchestrator.types import Rollout
    from aether_rl.utils.client import InferencePool


class GRPOAlgorithm(Algorithm):
    """Group Relative Policy Optimization: sample a group of rollouts from the
    policy per example; credit = reward minus the group mean (optionally
    length-shaped); action tokens feed the ``rl`` loss."""

    def __init__(self, config: GRPOAlgoConfig, policy_pool: InferencePool):
        super().__init__(config, policy_pool)
        self.length_penalty = config.length_penalty

    async def score_group(self, group: list[Rollout]) -> None:
        rewards = torch.tensor([rollout.reward for rollout in group], dtype=torch.float32)
        length_penalty = self.length_penalty
        if length_penalty is None:
            advantages = rewards - rewards.mean()
        elif length_penalty.type == "linear":
            output = torch.tensor([rollout.num_output_tokens for rollout in group], dtype=rewards.dtype)
            total = torch.tensor([rollout.num_total_tokens for rollout in group], dtype=rewards.dtype)
            turns = torch.tensor([rollout.num_turns for rollout in group], dtype=rewards.dtype)
            input = total - output
            penalty_frac = (
                length_penalty.num_output_tokens_weight * (output / output.max().clamp(min=1))
                + length_penalty.num_input_tokens_weight * (input / input.max().clamp(min=1))
                + length_penalty.num_turns_weight * (turns / turns.max().clamp(min=1))
            )
            penalty = rewards.mean() * penalty_frac
            shaped_rewards = rewards - penalty
            advantages = shaped_rewards - shaped_rewards.mean()
        else:
            reward_values = rewards.tolist()
            if any(reward not in (0.0, 1.0) for reward in reward_values):
                raise ValueError("shortest-correct thinking penalty requires binary aggregate rewards")
            shaped_rewards = rewards.clone()
            correct = [index for index, reward in enumerate(reward_values) if reward == 1.0]
            if correct:
                lengths: dict[int, float] = {}
                metric = length_penalty.thinking_length_metric
                for index in correct:
                    value = group[index].metrics.get(metric)
                    if value is None or not math.isfinite(value) or value < 0:
                        raise ValueError(f"correct rollout has invalid {metric!r} metric: {value!r}")
                    lengths[index] = value
                shortest = min(lengths.values())
                for index, length in lengths.items():
                    if length > shortest:
                        shaped_rewards[index] -= length_penalty.penalty
            advantages = shaped_rewards - shaped_rewards.mean()
        for rollout, advantage in zip(group, advantages.tolist(), strict=True):
            rollout.assign_advantages(advantage)
