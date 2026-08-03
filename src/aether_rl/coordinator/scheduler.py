from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Protocol, cast

from aether_rl.protocol import AssignmentLease, LeaseRequest

from .database import CoordinatorRepository


class SerializedCall(Protocol):
    def __call__(self, function: Callable[..., object], /, *args: object, **kwargs: object) -> Awaitable[object]: ...


class CoordinatorScheduler:
    durable_mutations = True

    def __init__(
        self,
        repository: CoordinatorRepository,
        serialized_call: SerializedCall | object,
        *,
        lease_duration_seconds: float,
        loaded_preference_seconds: float,
        max_policy_lag: int,
    ):
        if lease_duration_seconds <= 0:
            raise ValueError("lease duration must be positive")
        if loaded_preference_seconds < 0:
            raise ValueError("loaded preference must be non-negative")
        if max_policy_lag < 0:
            raise ValueError("maximum policy lag must be non-negative")
        self.repository = repository
        call = getattr(serialized_call, "call", serialized_call)
        if not callable(call):
            raise TypeError("serialized_call must be callable or provide a call method")
        self.serialized_call = call
        self.lease_duration_seconds = lease_duration_seconds
        self.loaded_preference_seconds = loaded_preference_seconds
        self.max_policy_lag = max_policy_lag
        self.repository.configure_scheduler(
            max_policy_lag=max_policy_lag,
            loaded_policy_preference_seconds=loaded_preference_seconds,
        )

    async def try_lease(self, request: LeaseRequest) -> AssignmentLease | None:
        result = await self.serialized_call(
            self.repository.lease_or_create_next_compatible,
            request,
            lease_duration_seconds=self.lease_duration_seconds,
        )
        return cast(AssignmentLease | None, result)
