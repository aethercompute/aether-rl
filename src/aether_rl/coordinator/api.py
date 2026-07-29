from __future__ import annotations

import asyncio
import hashlib
import hmac
import inspect
import json
import logging
import os
import sqlite3
import uuid
from collections.abc import Awaitable, Callable
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from typing import Protocol

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response, StreamingResponse
from pydantic import ValidationError
from starlette.background import BackgroundTask
from starlette.exceptions import HTTPException as StarletteHTTPException

from aether_rl.protocol import (
    PROTOCOL_VERSION,
    AssignmentLease,
    FailureEnvelope,
    LeaseRenewal,
    LeaseRenewRequest,
    LeaseRenewResponse,
    LeaseRequest,
    ResultEnvelope,
    SubmissionResponse,
    WorkerHeartbeat,
    WorkerHeartbeatResponse,
    WorkerRegistration,
    WorkerRegistrationResponse,
    canonical_json_bytes,
    decode_result_envelope,
    sha256_digest,
)

from .database import (
    ArtifactCorruptionError,
    CapacityError,
    ConflictError,
    CoordinatorRepository,
    IncompatibleWorkerError,
    InvalidStateError,
    LeaseRequestDisposition,
    NotFoundError,
)
from .scheduler import CoordinatorScheduler
from .spool import ImmutableArtifactConflictError

logger = logging.getLogger(__name__)


class LeaseProvider(Protocol):
    durable_mutations: bool

    async def try_lease(self, request: LeaseRequest) -> AssignmentLease | None: ...


class NoLeaseProvider:
    durable_mutations = False

    async def try_lease(self, request: LeaseRequest) -> AssignmentLease | None:
        return None


class CoordinatorService:
    def __init__(self, repository: CoordinatorRepository):
        self.repository = repository
        self._executor: ThreadPoolExecutor | None = None
        self._lock = asyncio.Lock()

    async def call(self, function: Callable[..., object], /, *args: object, **kwargs: object) -> object:
        async with self._lock:
            loop = asyncio.get_running_loop()
            if self._executor is None:
                self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="coordinator-db")
            return await loop.run_in_executor(self._executor, lambda: function(*args, **kwargs))

    def close(self) -> None:
        if self._executor is not None:
            self._executor.shutdown(wait=True)
            self._executor = None


class APIError(Exception):
    def __init__(self, status_code: int, code: str, message: str, headers: dict[str, str] | None = None):
        self.status_code = status_code
        self.code = code
        self.message = message
        self.headers = headers


def _error(status_code: int, code: str, message: str, headers: dict[str, str] | None = None) -> JSONResponse:
    return JSONResponse(status_code=status_code, content={"error": {"code": code, "message": message}}, headers=headers)


def _lease_response(lease: AssignmentLease | None) -> JSONResponse:
    if lease is None:
        raise InvalidStateError("leased request is missing its durable lease")
    return JSONResponse(content=lease.model_dump(mode="json"), headers={"Cache-Control": "no-store"})


def _no_work_response() -> Response:
    return Response(status_code=204, headers={"Retry-After": "1", "Cache-Control": "no-store"})


async def _complete_no_work(
    service: CoordinatorService, repository: CoordinatorRepository, request_id: str
) -> Response:
    disposition = await service.call(repository.mark_lease_request_no_work, request_id)
    if not isinstance(disposition, LeaseRequestDisposition):
        raise TypeError("repository returned an invalid lease request disposition")
    return _lease_response(disposition.lease) if disposition.state == "leased" else _no_work_response()


def create_coordinator_app(
    repository: CoordinatorRepository,
    *,
    token: str | None = None,
    lease_provider: LeaseProvider | None = None,
    trainer_ready: Callable[[], bool | Awaitable[bool]] = lambda: False,
    control_body_limit_bytes: int = 1024 * 1024,
    result_body_limit_bytes: int = 64 * 1024 * 1024,
    lease_duration_seconds: float = 30.0,
    loaded_policy_preference_seconds: float = 5.0,
    max_policy_lag: int = 0,
    max_lease_wait_seconds: float = 30.0,
    durable_provider_timeout_seconds: float = 30.0,
    lease_poll_interval_seconds: float = 0.1,
    stale_after_seconds: float = 60.0,
    lease_reaper_interval_seconds: float = 1.0,
    policy_verification_interval_seconds: float = 30.0,
    startup: Callable[[], Awaitable[None]] | None = None,
    shutdown: Callable[[], Awaitable[None]] | None = None,
    gate_leases_on_trainer: bool = False,
) -> FastAPI:
    auth_token = token if token is not None else os.environ.get("AETHER_COORDINATOR_TOKEN")
    if not auth_token:
        raise ValueError("a non-empty coordinator token is required")
    try:
        auth_token_bytes = auth_token.encode("ascii")
    except UnicodeEncodeError as error:
        raise ValueError("coordinator token must contain only ASCII characters") from error
    if control_body_limit_bytes < 1 or result_body_limit_bytes < 1:
        raise ValueError("body limits must be positive")
    if (
        lease_duration_seconds <= 0
        or max_lease_wait_seconds < 0
        or durable_provider_timeout_seconds <= 0
        or lease_poll_interval_seconds <= 0
        or lease_reaper_interval_seconds <= 0
        or policy_verification_interval_seconds <= 0
        or loaded_policy_preference_seconds < 0
        or max_policy_lag < 0
    ):
        raise ValueError("lease timing values are invalid")

    service = CoordinatorService(repository)
    provider = lease_provider or CoordinatorScheduler(
        repository,
        service.call,
        lease_duration_seconds=lease_duration_seconds,
        loaded_preference_seconds=loaded_policy_preference_seconds,
        max_policy_lag=max_policy_lag,
    )
    provider_tasks: set[asyncio.Task] = set()

    @asynccontextmanager
    async def lifespan(application: FastAPI):
        async def verify_active_policy() -> None:
            try:
                manifest = await service.call(repository.active_policy)
                if manifest.adapter is not None:
                    for artifact in manifest.adapter.files:
                        file, _, digest = await service.call(
                            repository.open_policy_file, manifest.policy_id, artifact.name
                        )
                        try:
                            await asyncio.to_thread(_verify_open_file, file, digest)
                        finally:
                            file.close()
            except Exception:
                application.state.policy_integrity_ok = False
                logger.exception("Active coordinator policy verification failed")
            else:
                application.state.policy_integrity_ok = True

        async def reap_expired_leases() -> None:
            while True:
                await asyncio.sleep(lease_reaper_interval_seconds)
                try:
                    await service.call(repository.expire_leases)
                except Exception:
                    logger.exception("Coordinator lease reaper failed")

        async def verify_policies() -> None:
            while True:
                await asyncio.sleep(policy_verification_interval_seconds)
                await verify_active_policy()

        if startup is not None:
            await startup()
        reaper = asyncio.create_task(reap_expired_leases())
        verifier = asyncio.create_task(verify_policies())
        await verify_active_policy()
        try:
            yield
        finally:
            reaper.cancel()
            verifier.cancel()
            if getattr(provider, "durable_mutations", False):
                await asyncio.gather(*provider_tasks, return_exceptions=True)
            else:
                for task in provider_tasks:
                    task.cancel()
            await asyncio.gather(reaper, verifier, return_exceptions=True)
            if shutdown is not None:
                await shutdown()
            service.close()

    app = FastAPI(lifespan=lifespan)
    app.state.coordinator_service = service
    app.state.policy_integrity_ok = True

    async def lease_gate_open() -> bool:
        if not gate_leases_on_trainer:
            return True
        callback_result = trainer_ready()
        is_trainer_ready = await callback_result if inspect.isawaitable(callback_result) else callback_result
        return is_trainer_ready and app.state.policy_integrity_ok

    @app.middleware("http")
    async def authenticate(request: Request, call_next: Callable[[Request], Awaitable[Response]]) -> Response:
        if request.url.path.startswith("/api/v1/"):
            supplied = request.headers.get("authorization", "")
            scheme, separator, credential = supplied.partition(" ")
            valid_scheme = bool(separator) and scheme.lower() == "bearer"
            try:
                credential_bytes = credential.encode("ascii")
            except UnicodeEncodeError:
                credential_bytes = b""
            valid_credential = hmac.compare_digest(credential_bytes, auth_token_bytes)
            if not (valid_scheme and valid_credential):
                return _error(
                    401,
                    "unauthorized",
                    "authentication required",
                    {"WWW-Authenticate": "Bearer"},
                )
            if request.headers.get("aether-protocol-version") != str(PROTOCOL_VERSION):
                return _error(400, "unsupported_protocol_version", "Aether-Protocol-Version must be 1")
        return await call_next(request)

    @app.exception_handler(APIError)
    async def handle_api_error(request: Request, error: APIError) -> JSONResponse:
        return _error(error.status_code, error.code, error.message, error.headers)

    @app.exception_handler(RequestValidationError)
    async def handle_request_validation(request: Request, error: RequestValidationError) -> JSONResponse:
        return _error(422, "malformed_request", "request validation failed")

    @app.exception_handler(StarletteHTTPException)
    async def handle_http_error(request: Request, error: StarletteHTTPException) -> JSONResponse:
        code = "not_found" if error.status_code == 404 else "http_error"
        return _error(error.status_code, code, "resource not found" if error.status_code == 404 else str(error.detail))

    @app.exception_handler(NotFoundError)
    async def handle_missing(request: Request, error: NotFoundError) -> JSONResponse:
        return _error(404, "not_found", "resource not found")

    @app.exception_handler(ConflictError)
    @app.exception_handler(IncompatibleWorkerError)
    @app.exception_handler(InvalidStateError)
    async def handle_conflict(request: Request, error: Exception) -> JSONResponse:
        return _error(409, "conflict", str(error))

    @app.exception_handler(CapacityError)
    async def handle_capacity(request: Request, error: CapacityError) -> JSONResponse:
        return _error(429, "capacity_exceeded", str(error), {"Retry-After": "1"})

    @app.exception_handler(sqlite3.OperationalError)
    async def handle_sqlite(request: Request, error: sqlite3.OperationalError) -> JSONResponse:
        if "locked" in str(error).lower() or "busy" in str(error).lower():
            return _error(503, "database_busy", "coordinator database is busy", {"Retry-After": "1"})
        return _error(500, "internal_error", "coordinator database failure")

    @app.exception_handler(ArtifactCorruptionError)
    @app.exception_handler(ImmutableArtifactConflictError)
    async def handle_corruption(request: Request, error: Exception) -> JSONResponse:
        return _error(500, "artifact_corruption", "coordinator artifact verification failed")

    @app.exception_handler(Exception)
    async def handle_internal_error(request: Request, error: Exception) -> JSONResponse:
        return _error(500, "internal_error", "internal coordinator error")

    async def parse_control_body(
        request: Request,
        model: type[WorkerRegistration]
        | type[WorkerHeartbeat]
        | type[LeaseRequest]
        | type[LeaseRenewRequest]
        | type[FailureEnvelope],
    ):
        _validate_json_body(request, control_body_limit_bytes)
        data = bytearray()
        async for chunk in request.stream():
            if len(data) + len(chunk) > control_body_limit_bytes:
                raise APIError(413, "payload_too_large", "request body is too large")
            data.extend(chunk)
        try:
            return _parse_model_json(model, data)
        except APIError:
            raise
        except ValidationError as error:
            if any(item["loc"] and item["loc"][-1] == "protocol_version" for item in error.errors()):
                raise APIError(400, "protocol_version", "body protocol_version must be 1") from error
            raise APIError(422, "malformed_request", "request body is malformed") from error

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/ready")
    async def ready() -> JSONResponse:
        try:
            await service.call(repository.verify_ready)
            callback_result = trainer_ready()
            is_ready = await callback_result if inspect.isawaitable(callback_result) else callback_result
        except Exception:
            is_ready = False
        is_ready = bool(is_ready and app.state.policy_integrity_ok)
        status_code = 200 if is_ready else 503
        return JSONResponse(status_code=status_code, content={"status": "ready" if is_ready else "not_ready"})

    @app.post("/api/v1/workers/register")
    async def register(request: Request) -> JSONResponse:
        registration = await parse_control_body(request, WorkerRegistration)
        record = await service.call(repository.register_worker, registration)
        response = WorkerRegistrationResponse(
            worker_id=record.worker_id,
            worker_session_id=record.worker_session_id,
            created=record.created,
            server_time=repository.clock(),
        )
        return JSONResponse(status_code=200, content=response.model_dump(mode="json"))

    @app.post("/api/v1/workers/heartbeat")
    async def heartbeat(request: Request) -> WorkerHeartbeatResponse:
        heartbeat_request = await parse_control_body(request, WorkerHeartbeat)
        renewals, stop_ids = await service.call(
            repository.record_heartbeat, heartbeat_request, duration_seconds=lease_duration_seconds
        )
        return WorkerHeartbeatResponse(server_time=repository.clock(), renewals=renewals, stop_lease_ids=stop_ids)

    @app.post("/api/v1/assignments/lease")
    async def lease(request: Request) -> Response:
        if not await lease_gate_open():
            raise APIError(503, "trainer_unavailable", "trainer or coordinator processing is unavailable")
        lease_request = await parse_control_body(request, LeaseRequest)
        disposition = await service.call(repository.validate_lease_request, lease_request)
        if not isinstance(disposition, LeaseRequestDisposition):
            raise TypeError("repository returned an invalid lease request disposition")
        if disposition.state == "leased":
            return _lease_response(disposition.lease)
        if disposition.state == "no_work":
            return _no_work_response()
        wait_seconds = min(lease_request.wait_seconds, max_lease_wait_seconds)
        deadline = asyncio.get_running_loop().time() + wait_seconds
        initial_attempt = True
        while True:
            if not await lease_gate_open():
                raise APIError(503, "trainer_unavailable", "trainer or coordinator processing is unavailable")
            remaining = deadline - asyncio.get_running_loop().time()
            if not initial_attempt and remaining <= 0:
                return await _complete_no_work(service, repository, lease_request.request_id)
            provider_timeout = max(0.001, remaining)
            provider_task = asyncio.create_task(provider.try_lease(lease_request))
            provider_tasks.add(provider_task)
            provider_task.add_done_callback(provider_tasks.discard)
            if getattr(provider, "durable_mutations", False):
                durable_timeout = min(
                    durable_provider_timeout_seconds,
                    max(lease_poll_interval_seconds, remaining),
                )
                done, _ = await asyncio.wait({provider_task}, timeout=durable_timeout)
                if not done:
                    provider_task.add_done_callback(_consume_task_result)
                    raise APIError(
                        503,
                        "lease_pending",
                        "durable lease operation is still pending; retry with the same request ID",
                        {"Retry-After": "1"},
                    )
                offered = provider_task.result()
            else:
                done, _ = await asyncio.wait({provider_task}, timeout=provider_timeout)
                if not done:
                    provider_task.cancel()
                    provider_task.add_done_callback(_consume_task_result)
                    return await _complete_no_work(service, repository, lease_request.request_id)
                offered = provider_task.result()
            if offered is not None:
                if not await lease_gate_open():
                    await service.call(repository.release_unoffered_lease, offered.lease_id)
                    raise APIError(503, "trainer_unavailable", "trainer or coordinator processing is unavailable")
                disposition = await service.call(repository.associate_offered_lease, lease_request, offered)
                if not isinstance(disposition, LeaseRequestDisposition):
                    raise TypeError("repository returned an invalid lease request disposition")
                return _lease_response(disposition.lease)
            remaining = deadline - asyncio.get_running_loop().time()
            if remaining <= 0:
                return await _complete_no_work(service, repository, lease_request.request_id)
            await asyncio.sleep(min(lease_poll_interval_seconds, remaining))
            initial_attempt = False

    @app.post("/api/v1/assignments/{assignment_id}/renew")
    async def renew(assignment_id: str, request: Request) -> LeaseRenewResponse:
        renewal_request = await parse_control_body(request, LeaseRenewRequest)
        if renewal_request.assignment_id != assignment_id:
            raise APIError(409, "identity_mismatch", "body assignment does not match the path")
        renewed = await service.call(
            repository.renew_lease,
            renewal_request.lease_id,
            worker_id=renewal_request.worker_id,
            worker_session_id=renewal_request.worker_session_id,
            duration_seconds=lease_duration_seconds,
            expected_assignment_id=assignment_id,
            sent_at=renewal_request.sent_at,
            acknowledge_cancellation=True,
        )
        if isinstance(renewed, str):
            return LeaseRenewResponse(server_time=repository.clock(), action="stop", reason=renewed)
        if not isinstance(renewed, AssignmentLease):
            raise TypeError("repository returned an invalid lease renewal")
        return LeaseRenewResponse(
            server_time=repository.clock(),
            action="renewed",
            renewal=LeaseRenewal(assignment_id=assignment_id, lease_id=renewed.lease_id, expires_at=renewed.expires_at),
        )

    @app.put("/api/v1/assignments/{assignment_id}/result")
    async def result(assignment_id: str, request: Request) -> JSONResponse:
        result_content_type = _validate_result_body(request, result_body_limit_bytes)
        assignment_limit = await service.call(repository.assignment_result_size_limit, assignment_id)
        effective_limit = min(result_body_limit_bytes, assignment_limit)
        incoming = repository.spool.incoming_dir
        temporary_path = incoming / f"upload-{uuid.uuid4().hex}.tmp"
        descriptor = os.open(temporary_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
        size = 0
        try:
            with os.fdopen(descriptor, "wb") as file:
                async for chunk in request.stream():
                    size += len(chunk)
                    if size > effective_limit:
                        raise APIError(413, "payload_too_large", "result body is too large")
                    await asyncio.to_thread(file.write, chunk)
                await asyncio.to_thread(_flush_file, file)
            try:
                envelope_bytes = await asyncio.to_thread(temporary_path.read_bytes)
                envelope = (
                    await asyncio.to_thread(decode_result_envelope, envelope_bytes)
                    if result_content_type == "application/msgpack"
                    else await asyncio.to_thread(_parse_model_json, ResultEnvelope, envelope_bytes)
                )
            except APIError:
                raise
            except ValidationError as error:
                if any(item["loc"] and item["loc"][-1] == "protocol_version" for item in error.errors()):
                    raise APIError(400, "protocol_version", "body protocol_version must be 1") from error
                raise APIError(422, "malformed_request", "result body is malformed") from error
            except (ValueError, json.JSONDecodeError) as error:
                raise APIError(422, "malformed_request", "result body is malformed") from error
            if envelope.assignment_id != assignment_id:
                raise APIError(409, "identity_mismatch", "body assignment does not match the path")
            accepted = await service.call(repository.accept_result, envelope)
            response = SubmissionResponse(**accepted.__dict__)
            return JSONResponse(status_code=200, content=response.model_dump(mode="json"))
        finally:
            try:
                await asyncio.to_thread(_remove_temporary_upload, temporary_path, incoming)
            except OSError:
                logger.exception("Failed to remove coordinator result upload temporary file")

    @app.post("/api/v1/assignments/{assignment_id}/failure")
    async def failure(assignment_id: str, request: Request) -> JSONResponse:
        envelope = await parse_control_body(request, FailureEnvelope)
        if envelope.assignment_id != assignment_id:
            raise APIError(409, "identity_mismatch", "body assignment does not match the path")
        accepted = await service.call(repository.accept_failure, envelope)
        response = SubmissionResponse(**accepted.__dict__)
        return JSONResponse(status_code=200, content=response.model_dump(mode="json"))

    @app.get("/api/v1/policies/current")
    async def current_policy() -> JSONResponse:
        manifest = await service.call(repository.active_policy)
        return JSONResponse(content=manifest.model_dump(mode="json"), headers={"Cache-Control": "no-store"})

    @app.get("/api/v1/policies/{policy_id}/manifest")
    async def policy_manifest(policy_id: str, request: Request) -> Response:
        manifest = await service.call(repository.get_policy, policy_id)
        etag = _quoted_etag(sha256_digest(canonical_json_bytes(manifest)))
        headers = {"ETag": etag, "Cache-Control": "private, max-age=31536000, immutable"}
        if _etag_matches(request.headers.get("if-none-match"), etag):
            return Response(status_code=304, headers=headers)
        return Response(content=canonical_json_bytes(manifest), media_type="application/json", headers=headers)

    @app.get("/api/v1/policies/{policy_id}/files/{name}")
    async def policy_file(policy_id: str, name: str, request: Request) -> Response:
        file, size, digest = await service.call(repository.open_policy_file, policy_id, name)
        try:
            await asyncio.to_thread(_verify_open_file, file, digest)
        except BaseException:
            file.close()
            raise
        etag = _quoted_etag(digest)
        range_header = request.headers.get("range")
        try:
            range_start, range_end = _parse_range(range_header, size)
        except BaseException:
            file.close()
            raise
        content_length = range_end - range_start + 1
        headers = {
            "ETag": etag,
            "Cache-Control": "private, max-age=31536000, immutable",
            "Accept-Ranges": "bytes",
            "Content-Length": str(content_length),
        }
        if _etag_matches(request.headers.get("if-none-match"), etag):
            file.close()
            return Response(status_code=304, headers=headers)
        media_type = "application/json" if name == "adapter_config.json" else "application/octet-stream"
        status_code = 200
        if range_header is not None:
            status_code = 206
            headers["Content-Range"] = f"bytes {range_start}-{range_end}/{size}"
        file.seek(range_start)
        return StreamingResponse(
            _file_chunks(file, content_length),
            status_code=status_code,
            media_type=media_type,
            headers=headers,
            background=BackgroundTask(file.close),
        )

    @app.get("/api/v1/status")
    async def status() -> dict[str, object]:
        await service.call(repository.expire_leases)
        snapshot = await service.call(repository.status_snapshot, stale_after_seconds=stale_after_seconds)
        callback_result = trainer_ready()
        is_trainer_ready = await callback_result if inspect.isawaitable(callback_result) else callback_result
        return {"protocol_version": PROTOCOL_VERSION, **snapshot, "trainer_ready": bool(is_trainer_ready)}

    return app


def _validate_json_body(request: Request, limit: int) -> None:
    if request.headers.get("content-type", "").split(";", 1)[0].strip().lower() != "application/json":
        raise APIError(415, "unsupported_media_type", "Content-Type must be application/json")
    if request.headers.get("content-encoding", "identity").lower() != "identity":
        raise APIError(415, "unsupported_encoding", "Content-Encoding must be identity")
    content_length = request.headers.get("content-length")
    if content_length is not None:
        try:
            declared_size = int(content_length)
        except ValueError as error:
            raise APIError(400, "invalid_content_length", "Content-Length must be an integer") from error
        if declared_size < 0:
            raise APIError(400, "invalid_content_length", "Content-Length must be non-negative")
        if declared_size > limit:
            raise APIError(413, "payload_too_large", "request body is too large")


def _validate_result_body(request: Request, limit: int) -> str:
    content_type = request.headers.get("content-type", "").split(";", 1)[0].strip().lower()
    if content_type not in {"application/json", "application/msgpack"}:
        raise APIError(415, "unsupported_media_type", "Content-Type must be application/json or application/msgpack")
    if request.headers.get("content-encoding", "identity").lower() != "identity":
        raise APIError(415, "unsupported_encoding", "Content-Encoding must be identity")
    content_length = request.headers.get("content-length")
    if content_length is not None:
        try:
            declared_size = int(content_length)
        except ValueError as error:
            raise APIError(400, "invalid_content_length", "Content-Length must be an integer") from error
        if declared_size < 0:
            raise APIError(400, "invalid_content_length", "Content-Length must be non-negative")
        if declared_size > limit:
            raise APIError(413, "payload_too_large", "request body is too large")
    return content_type


def _quoted_etag(digest: str) -> str:
    return f'"{digest}"'


def _parse_model_json(model, data: bytes | bytearray):
    try:
        raw = json.loads(data)
    except (TypeError, ValueError) as error:
        raise APIError(422, "malformed_request", "request body is malformed") from error
    if not isinstance(raw, dict) or "protocol_version" not in raw:
        raise APIError(400, "unsupported_protocol_version", "body protocol_version is required")
    try:
        return model.model_validate(raw)
    except ValidationError as error:
        if any(item["loc"] and item["loc"][-1] == "protocol_version" for item in error.errors()):
            raise APIError(400, "unsupported_protocol_version", "body protocol_version must be 1") from error
        raise


def _flush_file(file) -> None:
    file.flush()
    os.fsync(file.fileno())


def _remove_temporary_upload(path, directory) -> None:
    path.unlink(missing_ok=True)
    descriptor = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _etag_matches(header: str | None, current: str) -> bool:
    if header is None:
        return False
    current_opaque = current.removeprefix("W/")
    return any(
        candidate.strip() == "*" or candidate.strip().removeprefix("W/") == current_opaque
        for candidate in header.split(",")
    )


def _file_chunks(file, remaining: int):
    while remaining > 0 and (chunk := file.read(min(1024 * 1024, remaining))):
        remaining -= len(chunk)
        yield chunk


def _verify_open_file(file, expected_digest: str) -> None:
    digest = hashlib.sha256()
    while chunk := file.read(1024 * 1024):
        digest.update(chunk)
    if f"sha256:{digest.hexdigest()}" != expected_digest:
        raise ArtifactCorruptionError("published policy file digest does not match its manifest")
    file.seek(0)


def _consume_task_result(task: asyncio.Task) -> None:
    if task.cancelled():
        return
    try:
        task.exception()
    except asyncio.CancelledError:
        pass


def _parse_range(header: str | None, size: int) -> tuple[int, int]:
    error_headers = {"Content-Range": f"bytes */{size}"}
    if header is None:
        return 0, size - 1
    if not header.startswith("bytes=") or "," in header:
        raise APIError(416, "invalid_range", "only one byte range is supported", error_headers)
    start_text, separator, end_text = header.removeprefix("bytes=").partition("-")
    if not separator:
        raise APIError(416, "invalid_range", "byte range is malformed", error_headers)
    try:
        if start_text:
            start = int(start_text)
            end = int(end_text) if end_text else size - 1
        else:
            suffix = int(end_text)
            if suffix <= 0:
                raise ValueError
            start = max(0, size - suffix)
            end = size - 1
    except ValueError as error:
        raise APIError(416, "invalid_range", "byte range is malformed", error_headers) from error
    if start < 0 or end < start or start >= size:
        raise APIError(416, "invalid_range", "byte range is unsatisfiable", error_headers)
    return start, min(end, size - 1)
