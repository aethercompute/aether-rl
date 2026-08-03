from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass, field

import httpx
from openai import AsyncOpenAI
from verifiers.v1.clients.train import TrainClient

from aether_rl.protocol import InferenceReply, InferenceRequest


class InferenceLeaseClosed(RuntimeError):
    pass


@dataclass
class _LeaseExchange:
    worker_id: str
    worker_session_id: str
    requests: asyncio.Queue[InferenceRequest] = field(default_factory=asyncio.Queue)
    replies: dict[str, InferenceReply] = field(default_factory=dict)
    waiters: dict[str, asyncio.Future[InferenceReply]] = field(default_factory=dict)
    active_request: InferenceRequest | None = None
    closed: bool = False


class InferenceBroker:
    def __init__(self, *, body_limit_bytes: int):
        self.body_limit_bytes = body_limit_bytes
        self._leases: dict[str, _LeaseExchange] = {}

    def register(self, lease_id: str, worker_id: str, worker_session_id: str) -> None:
        if lease_id in self._leases:
            raise RuntimeError("inference lease is already registered")
        self._leases[lease_id] = _LeaseExchange(worker_id, worker_session_id)

    def close(self, lease_id: str) -> None:
        exchange = self._leases.pop(lease_id, None)
        if exchange is None or exchange.closed:
            return
        exchange.closed = True
        for waiter in exchange.waiters.values():
            if not waiter.done():
                waiter.set_exception(InferenceLeaseClosed("inference lease is closed"))

    async def request(
        self, lease_id: str, method: str, path: str, headers: dict[str, str], body: bytes
    ) -> InferenceReply:
        if len(body) > self.body_limit_bytes:
            raise ValueError("inference request body is too large")
        exchange = self._leases.get(lease_id)
        if exchange is None or exchange.closed:
            raise InferenceLeaseClosed("inference lease is closed")
        request_id = f"inference-{uuid.uuid4().hex}"
        request = InferenceRequest(
            request_id=request_id,
            method=method,
            path=path,
            headers=headers,
            body=body,
        )
        waiter = asyncio.get_running_loop().create_future()
        exchange.waiters[request_id] = waiter
        await exchange.requests.put(request)
        try:
            return await waiter
        finally:
            exchange.waiters.pop(request_id, None)

    async def exchange(
        self,
        lease_id: str,
        worker_id: str,
        worker_session_id: str,
        reply: InferenceReply | None,
        wait_seconds: float,
    ) -> InferenceRequest | None | bool:
        exchange = self._leases.get(lease_id)
        if exchange is None:
            return False
        if (exchange.worker_id, exchange.worker_session_id) != (worker_id, worker_session_id):
            raise PermissionError("inference lease belongs to a different worker session")
        if reply is not None:
            if len(reply.body) > self.body_limit_bytes:
                raise ValueError("inference reply body is too large")
            previous = exchange.replies.get(reply.request_id)
            if previous is not None and previous != reply:
                raise ValueError("inference request already has a different reply")
            exchange.replies[reply.request_id] = reply
            waiter = exchange.waiters.get(reply.request_id)
            if waiter is not None and not waiter.done():
                waiter.set_result(reply)
            if exchange.active_request is not None and exchange.active_request.request_id == reply.request_id:
                exchange.active_request = None
        if exchange.closed:
            return False
        if exchange.active_request is not None:
            return exchange.active_request
        try:
            exchange.active_request = exchange.requests.get_nowait()
            return exchange.active_request
        except asyncio.QueueEmpty:
            pass
        try:
            exchange.active_request = await asyncio.wait_for(exchange.requests.get(), timeout=wait_seconds)
            return exchange.active_request
        except TimeoutError:
            return None


class BrokerTransport(httpx.AsyncBaseTransport):
    _ROUTES = {("GET", "/v1/models"), ("POST", "/inference/v1/generate")}

    def __init__(self, broker: InferenceBroker, lease_id: str):
        self.broker = broker
        self.lease_id = lease_id

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        route = (request.method, request.url.path)
        if route not in self._ROUTES:
            raise httpx.UnsupportedProtocol(f"inference route is not allowed: {request.method} {request.url.path}")
        body = await request.aread()
        reply = await self.broker.request(
            self.lease_id,
            request.method,
            request.url.path,
            {key: value for key, value in request.headers.items() if key.lower() not in {"authorization", "host"}},
            body,
        )
        return httpx.Response(reply.status_code, headers=reply.headers, content=reply.body, request=request)


def create_train_client(
    broker: InferenceBroker,
    lease_id: str,
    *,
    renderer_model_name: str,
    renderer_model_revision: str,
) -> TrainClient:
    http_client = httpx.AsyncClient(transport=BrokerTransport(broker, lease_id), timeout=None)
    openai = AsyncOpenAI(
        base_url="http://inference.invalid/v1",
        api_key="EMPTY",
        http_client=http_client,
        max_retries=0,
    )
    return TrainClient(
        openai,
        renderer_model_name=renderer_model_name,
        renderer_model_revision=renderer_model_revision,
    )
