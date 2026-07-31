import pytest

from aether_rl.configs.trainer import DefaultLossConfig
from aether_rl.trainer.rl.loss import rl_normalization_scale


def test_dr_grpo_normalization_uses_fixed_per_response_denominator():
    config = DefaultLossConfig(
        rl_normalization="dr_grpo",
        dr_grpo_max_completion_tokens=16_384,
    )
    assert rl_normalization_scale(config, active_tokens=12_000, sequences=16, cp_size=2) == 16 * 16_384


def test_active_token_normalization_preserves_existing_behavior():
    assert rl_normalization_scale(DefaultLossConfig(), active_tokens=12_000, sequences=16, cp_size=2) == 12_000


def test_dr_grpo_normalization_requires_fixed_completion_length():
    with pytest.raises(ValueError, match="requires dr_grpo_max_completion_tokens"):
        DefaultLossConfig(rl_normalization="dr_grpo")
