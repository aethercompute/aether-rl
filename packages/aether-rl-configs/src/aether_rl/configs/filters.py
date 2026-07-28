from typing import Annotated, Literal, TypeAlias

from pydantic import Field

from aether_rl.utils.config import BaseConfig


class GibberishFilterConfig(BaseConfig):
    type: Literal["gibberish"] = "gibberish"
    token_id_threshold: int = Field(100_000, ge=0)
    logprob_offset: float = Field(5.0, ge=0)
    enforce: bool = False


class RepetitionFilterConfig(BaseConfig):
    type: Literal["repetition"] = "repetition"
    window: int = Field(3_000, ge=1)
    prob_threshold: float = Field(0.99, gt=0, le=1)
    enforce: bool = False


class ZeroAdvantageFilterConfig(BaseConfig):
    type: Literal["zero_advantage"] = "zero_advantage"
    enforce: bool = True


FilterConfig: TypeAlias = Annotated[
    GibberishFilterConfig | RepetitionFilterConfig | ZeroAdvantageFilterConfig,
    Field(discriminator="type"),
]
