from __future__ import annotations

import json
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Literal

import httpx
import zstandard
from pydantic import ValidationError

from aether_rl.protocol import (
    PROTOCOL_VERSION,
    AssignmentLease,
    AssignmentLeaseBatch,
    FailureEnvelope,
    LeaseRenewRequest,
    LeaseRenewResponse,
    LeaseRequest,
    PolicyLocations,
    PolicyManifest,
    ResultEnvelope,
    SubmissionResponse,
    WorkerHeartbeat,
    WorkerHeartbeatResponse,
    WorkerRegistration,
    WorkerRegistrationResponse,
    canonical_json_bytes,
)

from .spool import SpoolEntry


@dataclass(frozen=True)
class CoordinatorAPIError(RuntimeError):
    status_code: int
    code: str
    message: str
    retry_after: float | None = None

    @property
    def retryable(self) -> bool:
        return self.status_code in {408, 425, 429} or self.status_code >= 500

    def __str__(self) -> str:
        return f"coordinator request failed ({self.status_code} {self.code}): {self.message}"


class CoordinatorProtocolError(RuntimeError):
    pass


class CoordinatorClient:
    def __init__(
        self,
        coordinator_url: str,
        token: str,
        *,
        timeout_seconds: float,
        client: httpx.AsyncClient | None = None,
        artifact_client: httpx.AsyncClient | None = None,
        result_compression: Literal["identity", "zstd"] = "identity",
    ):
        if not token:
            raise ValueError("coordinator token must not be empty")
        self._owns_client = client is None
        self.client = client or httpx.AsyncClient(
            base_url=coordinator_url.rstrip("/"),
            timeout=httpx.Timeout(timeout_seconds),
            limits=httpx.Limits(max_connections=16, max_keepalive_connections=8),
        )
        self._owns_artifact_client = artifact_client is None and client is None
        self.artifact_client = artifact_client or client
        if self.artifact_client is None:
            self.artifact_client = httpx.AsyncClient(
                timeout=httpx.Timeout(timeout_seconds),
                limits=httpx.Limits(max_connections=8, max_keepalive_connections=4),
                follow_redirects=False,
            )
        self.headers = {
            "Authorization": f"Bearer {token}",
            "Aether-Protocol-Version": str(PROTOCOL_VERSION),
        }
        self.result_compression = result_compression
        self._compressor = zstandard.ZstdCompressor(level=3) if result_compression == "zstd" else None

    async def register(self, registration: WorkerRegistration) -> WorkerRegistrationResponse:
        response = await self._control("POST", "/api/v1/workers/register", registration)
        return self._model(response, WorkerRegistrationResponse)

    async def heartbeat(self, heartbeat: WorkerHeartbeat) -> WorkerHeartbeatResponse:
        response = await self._control("POST", "/api/v1/workers/heartbeat", heartbeat)
        return self._model(response, WorkerHeartbeatResponse)

    async def lease(self, request: LeaseRequest) -> AssignmentLease | None:
        response = await self._control("POST", "/api/v1/assignments/lease", request, expected=(200, 204))
        return None if response.status_code == 204 else self._model(response, AssignmentLease)

    async def lease_group(self, request: LeaseRequest) -> tuple[AssignmentLease, ...]:
        response = await self._control("POST", "/api/v1/assignments/lease-group", request, expected=(200, 204))
        if response.status_code == 204:
            return ()
        return self._model(response, AssignmentLeaseBatch).leases

    async def renew(self, request: LeaseRenewRequest) -> LeaseRenewResponse:
        response = await self._control("POST", f"/api/v1/assignments/{request.assignment_id}/renew", request)
        return self._model(response, LeaseRenewResponse)

    async def submit(self, entry: SpoolEntry) -> SubmissionResponse:
        path = f"/api/v1/assignments/{entry.envelope.assignment_id}/"
        if isinstance(entry.envelope, ResultEnvelope):
            body = entry.body if self._compressor is None else self._compressor.compress(entry.body)
            headers = {"Content-Type": "application/msgpack"}
            if self._compressor is not None:
                headers["Content-Encoding"] = "zstd"
            try:
                response = await self._request("PUT", path + "result", content=body, headers=headers)
            except CoordinatorAPIError as error:
                if self._compressor is None or error.status_code != 415:
                    raise
                response = await self._request(
                    "PUT",
                    path + "result",
                    content=entry.body,
                    headers={"Content-Type": "application/msgpack"},
                )
        elif isinstance(entry.envelope, FailureEnvelope):
            response = await self._request(
                "POST",
                path + "failure",
                content=entry.body,
                headers={"Content-Type": "application/json"},
            )
        else:
            raise TypeError("unsupported spooled terminal envelope")
        return self._model(response, SubmissionResponse)

    async def get_policy_manifest(self, policy_id: str) -> httpx.Response:
        response = await self.client.get(
            f"/api/v1/policies/{policy_id}/manifest",
            headers=self.headers | {"Accept-Encoding": "identity"},
        )
        if response.status_code != 200:
            raise self._api_error(response)
        return response

    async def get_current_policy(self) -> PolicyManifest:
        response = await self.client.get("/api/v1/policies/current", headers=self.headers)
        if response.status_code != 200:
            raise self._api_error(response)
        return self._model(response, PolicyManifest)

    async def get_policy_locations(self, policy_id: str) -> PolicyLocations:
        response = await self.client.get(f"/api/v1/policies/{policy_id}/locations", headers=self.headers)
        if response.status_code != 200:
            raise self._api_error(response)
        return self._model(response, PolicyLocations)

    @asynccontextmanager
    async def stream_policy_file(self, policy_id: str, name: str, *, offset: int = 0):
        headers = self.headers | {"Accept-Encoding": "identity"}
        if offset:
            headers["Range"] = f"bytes={offset}-"
        request = self.client.build_request(
            "GET",
            f"/api/v1/policies/{policy_id}/files/{name}",
            headers=headers,
        )
        response = await self.client.send(request, stream=True)
        expected = 206 if offset else 200
        if response.status_code != expected:
            await response.aread()
            error = self._api_error(response)
            await response.aclose()
            raise error
        try:
            yield response
        finally:
            await response.aclose()

    async def close(self) -> None:
        if self._owns_client:
            await self.client.aclose()
        if self._owns_artifact_client:
            await self.artifact_client.aclose()

    async def _control(self, method: str, path: str, model, *, expected: tuple[int, ...] = (200,)) -> httpx.Response:
        return await self._request(
            method,
            path,
            content=canonical_json_bytes(model),
            headers={"Content-Type": "application/json"},
            expected=expected,
        )

    async def _request(
        self,
        method: str,
        path: str,
        *,
        content: bytes,
        headers: dict[str, str],
        expected: tuple[int, ...] = (200,),
    ) -> httpx.Response:
        response = await self.client.request(method, path, content=content, headers=self.headers | headers)
        if response.status_code in expected:
            return response
        raise self._api_error(response)

    @staticmethod
    def _api_error(response: httpx.Response) -> CoordinatorAPIError:
        try:
            payload = response.json()
            error = payload["error"]
            code = str(error["code"])
            message = str(error["message"])
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            code = "invalid_error_response"
            message = "coordinator returned an invalid error response"
        retry_after = _retry_after(response.headers.get("retry-after"))
        return CoordinatorAPIError(response.status_code, code, message, retry_after)

    @staticmethod
    def _model(response: httpx.Response, model_type):
        try:
            return model_type.model_validate_json(response.content)
        except ValidationError as error:
            raise CoordinatorProtocolError("coordinator returned a malformed success response") from error


def _retry_after(value: str | None) -> float | None:
    if value is None:
        return None
    try:
        seconds = float(value)
    except ValueError:
        return None
    return max(0.0, seconds)
