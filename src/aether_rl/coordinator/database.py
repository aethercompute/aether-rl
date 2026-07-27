from __future__ import annotations

import fcntl
import json
import os
import secrets
import sqlite3
import stat
import time
import uuid
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, BinaryIO, Literal

from aether_rl.protocol import (
    AssignmentLease,
    EnvironmentIdentity,
    FailureEnvelope,
    LeaseRenewal,
    LeaseRequest,
    PolicyManifest,
    ResultEnvelope,
    RolloutAssignment,
    WorkerCapabilities,
    WorkerHeartbeat,
    WorkerRegistration,
    canonical_json_bytes,
    decode_result_envelope,
    policy_manifest_digest,
    result_envelope_bytes,
    sha256_digest,
)
from aether_rl.trainer.policy import verify_lora_policy

from .migrations import MIGRATIONS, SCHEMA_VERSION
from .spool import AtomicSpool

if TYPE_CHECKING:
    from .environments import EnvironmentSourceSpec


class CoordinatorError(Exception):
    pass


class SchemaVersionError(CoordinatorError):
    pass


class CoordinatorLockError(CoordinatorError):
    pass


class ConflictError(CoordinatorError):
    pass


class InvalidStateError(CoordinatorError):
    pass


class IncompatibleWorkerError(CoordinatorError):
    pass


class CapacityError(CoordinatorError):
    pass


class ArtifactCorruptionError(CoordinatorError):
    pass


class NotFoundError(CoordinatorError):
    pass


@dataclass(frozen=True)
class RegistrationRecord:
    worker_id: str
    worker_session_id: str
    created: bool


@dataclass(frozen=True)
class AcceptanceRecord:
    assignment_id: str
    envelope_digest: str
    duplicate: bool
    terminal: bool


@dataclass(frozen=True)
class PendingResult:
    assignment_id: str
    envelope_digest: str
    path: Path


@dataclass(frozen=True)
class CreatedGroup:
    group_id: str
    creation_key: str
    sequence: int
    source_id: str
    source_cursor: int
    assignments: tuple[RolloutAssignment, ...]


@dataclass(frozen=True)
class LeaseRequestDisposition:
    state: Literal["pending", "no_work", "leased"]
    lease: AssignmentLease | None = None


@dataclass(frozen=True)
class GroupOutcome:
    assignment: RolloutAssignment
    outcome: str
    envelope_digest: str | None
    result_path: Path | None


@dataclass(frozen=True)
class ReadyGroup:
    group_id: str
    source_id: str | None
    kind: Literal["train", "eval"]
    sequence: int
    outcomes: tuple[GroupOutcome, ...]


@dataclass(frozen=True)
class PendingProcessedRollout:
    group_id: str
    ordinal: int
    token_count: int
    artifact_path: Path
    artifact_digest: str
    size_bytes: int


@dataclass(frozen=True)
class ProcessedGroupRecord:
    group_id: str
    artifact_digest: str
    artifact_path: Path
    size_bytes: int


@dataclass(frozen=True)
class TrainingBatchRecord:
    step: int
    artifact_digest: str
    artifact_path: Path
    size_bytes: int
    sample_count: int


class CoordinatorRepository:
    def __init__(
        self,
        database_path: Path,
        run_root: Path,
        *,
        clock: Callable[[], float] = time.time,
        busy_timeout_ms: int = 5_000,
        retry_base_seconds: float = 1.0,
        retry_max_seconds: float = 60.0,
    ):
        if busy_timeout_ms < 0:
            raise ValueError("busy_timeout_ms must be non-negative")
        if retry_base_seconds <= 0 or retry_max_seconds <= 0:
            raise ValueError("retry delays must be positive")
        self.database_path = Path(database_path)
        self.run_root = Path(run_root).resolve()
        self.clock = clock
        self.retry_base_seconds = retry_base_seconds
        self.retry_max_seconds = retry_max_seconds
        self._verified_policy_digests: set[str] = set()
        self.run_root.mkdir(parents=True, exist_ok=True)
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        control_dir = self.run_root / "control"
        control_dir.mkdir(exist_ok=True)
        lock_descriptor = os.open(control_dir / "coordinator.lock", os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
        self._lock_file = os.fdopen(lock_descriptor, "a+")
        try:
            fcntl.flock(self._lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            self._lock_file.close()
            raise CoordinatorLockError(f"another coordinator owns the run lock: {self.run_root}") from error
        try:
            self.connection = sqlite3.connect(self.database_path, isolation_level=None, check_same_thread=False)
            self.connection.row_factory = sqlite3.Row
            self.connection.execute("PRAGMA foreign_keys = ON")
            self.connection.execute("PRAGMA journal_mode = WAL")
            self.connection.execute("PRAGMA synchronous = FULL")
            self.connection.execute(f"PRAGMA busy_timeout = {busy_timeout_ms:d}")
            self.spool = AtomicSpool(self.run_root)
            self._migrate()
            if self.connection.execute("SELECT 1 FROM runs WHERE singleton = 1").fetchone() is not None:
                self.recover()
        except BaseException:
            if hasattr(self, "connection"):
                self.connection.close()
            fcntl.flock(self._lock_file.fileno(), fcntl.LOCK_UN)
            self._lock_file.close()
            raise

    def __enter__(self) -> CoordinatorRepository:
        return self

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        self.close()

    def close(self) -> None:
        if self._lock_file.closed:
            return
        self.connection.close()
        fcntl.flock(self._lock_file.fileno(), fcntl.LOCK_UN)
        self._lock_file.close()

    @contextmanager
    def _transaction(self) -> Iterator[None]:
        self.connection.execute("BEGIN IMMEDIATE")
        try:
            yield
        except BaseException:
            self.connection.rollback()
            raise
        else:
            self.connection.commit()

    def _migrate(self) -> None:
        with self._transaction():
            self.connection.execute(
                "CREATE TABLE IF NOT EXISTS schema_migrations "
                "(version INTEGER PRIMARY KEY CHECK (version > 0), applied_at REAL NOT NULL)"
            )
            row = self.connection.execute("SELECT MAX(version) AS version FROM schema_migrations").fetchone()
            current = row["version"] or 0
            if current > SCHEMA_VERSION:
                raise SchemaVersionError(
                    f"database schema version {current} is newer than supported version {SCHEMA_VERSION}"
                )
        for version in range(current + 1, SCHEMA_VERSION + 1):
            with self._transaction():
                for statement in MIGRATIONS[version]:
                    self.connection.execute(statement)
                self.connection.execute(
                    "INSERT INTO schema_migrations(version, applied_at) VALUES (?, ?)",
                    (version, self.clock()),
                )

    def initialize_run(self, base_policy: PolicyManifest) -> None:
        if base_policy.policy_version != 0:
            raise ValueError("a run must be initialized from policy version 0")
        manifest_json = canonical_json_bytes(base_policy)
        manifest_digest = policy_manifest_digest(base_policy)
        with self._transaction():
            existing = self.connection.execute("SELECT * FROM runs WHERE singleton = 1").fetchone()
            if existing is not None:
                policy = self.connection.execute(
                    "SELECT manifest_digest FROM policies WHERE policy_id = ?", (existing["base_policy_id"],)
                ).fetchone()
                if existing["run_id"] == base_policy.run_id and policy["manifest_digest"] == manifest_digest:
                    return
                raise ConflictError("coordinator database is already initialized for a different run")
            self.connection.execute(
                "INSERT INTO policies VALUES (?, ?, ?, ?, NULL, ?)",
                (
                    base_policy.policy_id,
                    0,
                    manifest_digest,
                    manifest_json,
                    base_policy.created_at,
                ),
            )
            self.connection.execute(
                "INSERT INTO runs VALUES (1, ?, ?, ?, ?, ?)",
                (
                    base_policy.run_id,
                    canonical_json_bytes(base_policy.base_model),
                    base_policy.policy_id,
                    base_policy.policy_id,
                    base_policy.created_at,
                ),
            )

    def record_policy(self, manifest: PolicyManifest, artifact_path: Path) -> None:
        verified, relative_path, manifest_json, digest = self._prepare_trained_policy_record(manifest, artifact_path)
        with self._transaction():
            self._record_policy_in_transaction(verified, relative_path, manifest_json, digest)

    def record_and_activate_policy(self, manifest: PolicyManifest, artifact_path: Path) -> PolicyManifest:
        verified, relative_path, manifest_json, digest = self._prepare_trained_policy_record(manifest, artifact_path)
        with self._transaction():
            self._record_policy_in_transaction(verified, relative_path, manifest_json, digest)
            active = self.connection.execute(
                "SELECT p.policy_version FROM runs r JOIN policies p ON p.policy_id = r.active_policy_id "
                "WHERE r.singleton = 1"
            ).fetchone()
            if active is None:
                raise InvalidStateError("run has not been initialized")
            if verified.policy_version < active["policy_version"]:
                raise InvalidStateError("policy activation must be monotonic")
            self.connection.execute("UPDATE runs SET active_policy_id = ? WHERE singleton = 1", (verified.policy_id,))
            self._reconcile_stale_work(self.clock())
        return verified

    def _prepare_trained_policy_record(
        self, manifest: PolicyManifest, artifact_path: Path
    ) -> tuple[PolicyManifest, str, bytes, str]:
        if manifest.policy_version == 0:
            raise ValueError("record_policy only records trained policies")
        verified = verify_lora_policy(Path(artifact_path), expected=manifest)
        relative_path = self._relative_path(Path(artifact_path))
        manifest_json = canonical_json_bytes(verified)
        digest = policy_manifest_digest(verified)
        self._verified_policy_digests.add(digest)
        return verified, relative_path, manifest_json, digest

    def _record_policy_in_transaction(
        self, verified: PolicyManifest, relative_path: str, manifest_json: bytes, digest: str
    ) -> None:
        run = self._run()
        if verified.run_id != run["run_id"] or canonical_json_bytes(verified.base_model) != run["base_model_json"]:
            raise ConflictError("policy does not belong to this run and base model")
        existing = self.connection.execute(
            "SELECT manifest_digest, artifact_path FROM policies WHERE policy_id = ? OR policy_version = ?",
            (verified.policy_id, verified.policy_version),
        ).fetchone()
        if existing is not None:
            if existing["manifest_digest"] == digest and existing["artifact_path"] == relative_path:
                return
            raise ConflictError("policy version or ID already has different immutable contents")
        self.connection.execute(
            "INSERT INTO policies VALUES (?, ?, ?, ?, ?, ?)",
            (
                verified.policy_id,
                verified.policy_version,
                digest,
                manifest_json,
                relative_path,
                verified.created_at,
            ),
        )

    def activate_policy(self, policy_id: str) -> PolicyManifest:
        with self._transaction():
            target = self.connection.execute("SELECT * FROM policies WHERE policy_id = ?", (policy_id,)).fetchone()
            if target is None:
                raise InvalidStateError("cannot activate an unpublished policy")
            self._verify_policy_row(target)
            active = self.connection.execute(
                "SELECT p.policy_version FROM runs r JOIN policies p ON p.policy_id = r.active_policy_id "
                "WHERE r.singleton = 1"
            ).fetchone()
            if active is None:
                raise InvalidStateError("run has not been initialized")
            if target["policy_version"] < active["policy_version"]:
                raise InvalidStateError("policy activation must be monotonic")
            self.connection.execute("UPDATE runs SET active_policy_id = ? WHERE singleton = 1", (policy_id,))
            self._reconcile_stale_work(self.clock())
        return PolicyManifest.model_validate_json(target["manifest_json"])

    def configure_scheduler(self, *, max_policy_lag: int, loaded_policy_preference_seconds: float) -> None:
        if max_policy_lag < 0 or loaded_policy_preference_seconds < 0:
            raise ValueError("scheduler preference and policy lag must be non-negative")
        with self._transaction():
            self.connection.execute(
                "UPDATE scheduler_state SET max_policy_lag = ?, loaded_policy_preference_seconds = ? "
                "WHERE singleton = 1",
                (max_policy_lag, loaded_policy_preference_seconds),
            )
            self._reconcile_stale_work(self.clock())

    def active_policy(self) -> PolicyManifest:
        row = self.connection.execute(
            "SELECT p.* FROM runs r JOIN policies p ON p.policy_id = r.active_policy_id WHERE r.singleton = 1"
        ).fetchone()
        if row is None:
            raise InvalidStateError("run has not been initialized")
        self._verify_policy_row(row)
        return PolicyManifest.model_validate_json(row["manifest_json"])

    def register_worker(self, registration: WorkerRegistration) -> RegistrationRecord:
        capabilities_json = canonical_json_bytes(registration.capabilities)
        received_at = self.clock()
        with self._transaction():
            run = self._run()
            if canonical_json_bytes(registration.capabilities.base_model) != run["base_model_json"]:
                raise IncompatibleWorkerError("worker base model does not match the run")
            existing = self.connection.execute(
                "SELECT worker_id, capabilities_json FROM worker_sessions WHERE worker_session_id = ?",
                (registration.worker_session_id,),
            ).fetchone()
            if existing is not None:
                if (
                    existing["worker_id"] != registration.worker_id
                    or existing["capabilities_json"] != capabilities_json
                ):
                    raise ConflictError("worker session identity or capabilities conflict")
                return RegistrationRecord(registration.worker_id, registration.worker_session_id, False)
            self.connection.execute(
                "INSERT OR IGNORE INTO workers VALUES (?, ?)",
                (registration.worker_id, received_at),
            )
            self.connection.execute(
                "INSERT INTO worker_sessions "
                "(worker_session_id, worker_id, capabilities_json, registered_at, last_seen_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (
                    registration.worker_session_id,
                    registration.worker_id,
                    capabilities_json,
                    received_at,
                    received_at,
                ),
            )
        return RegistrationRecord(registration.worker_id, registration.worker_session_id, True)

    def record_heartbeat(
        self, heartbeat: WorkerHeartbeat, *, duration_seconds: float
    ) -> tuple[tuple[LeaseRenewal, ...], tuple[str, ...]]:
        if duration_seconds <= 0:
            raise ValueError("lease duration must be positive")
        now = self.clock()
        renewals: list[LeaseRenewal] = []
        stop_ids: list[str] = []
        with self._transaction():
            self._reconcile_stale_work(now)
            session = self.connection.execute(
                "SELECT worker_id, last_heartbeat_sent_at FROM worker_sessions WHERE worker_session_id = ?",
                (heartbeat.worker_session_id,),
            ).fetchone()
            if session is None or session["worker_id"] != heartbeat.worker_id:
                raise IncompatibleWorkerError("worker session is not registered for this worker")
            if session["last_heartbeat_sent_at"] is not None and heartbeat.sent_at <= session["last_heartbeat_sent_at"]:
                raise ConflictError("heartbeat sent_at must increase within a worker session")
            self._validate_policy_ids(heartbeat.loaded_policy_ids)

            leases = (
                {
                    row["lease_id"]: row
                    for row in self.connection.execute(
                        "SELECT l.*, a.state AS assignment_state, a.deadline_at FROM lease_attempts l "
                        "JOIN assignments a USING (assignment_id) WHERE l.lease_id IN "
                        f"({','.join('?' for _ in heartbeat.active_lease_ids)})",
                        heartbeat.active_lease_ids,
                    ).fetchall()
                }
                if heartbeat.active_lease_ids
                else {}
            )
            for lease in leases.values():
                if (
                    lease["worker_id"] != heartbeat.worker_id
                    or lease["worker_session_id"] != heartbeat.worker_session_id
                ):
                    raise ConflictError("heartbeat includes a lease owned by another worker session")

            cancelled_lease_ids = (
                {
                    row["lease_id"]
                    for row in self.connection.execute(
                        "SELECT lease_id FROM lease_cancellations WHERE lease_id IN "
                        f"({','.join('?' for _ in heartbeat.active_lease_ids)})",
                        heartbeat.active_lease_ids,
                    ).fetchall()
                }
                if heartbeat.active_lease_ids
                else set()
            )

            for lease_id in heartbeat.active_lease_ids:
                lease = leases.get(lease_id)
                if lease_id in cancelled_lease_ids:
                    self.connection.execute(
                        "UPDATE lease_cancellations SET delivered_at = COALESCE(delivered_at, ?) WHERE lease_id = ?",
                        (now, lease_id),
                    )
                if (
                    lease is None
                    or lease_id in cancelled_lease_ids
                    or lease["state"] != "active"
                    or lease["assignment_state"] != "leased"
                    or lease["expires_at"] <= now
                    or (lease["deadline_at"] is not None and lease["deadline_at"] <= now)
                ):
                    stop_ids.append(lease_id)
                    continue
                expires_at = max(lease["expires_at"], now + duration_seconds)
                if lease["deadline_at"] is not None:
                    expires_at = min(expires_at, lease["deadline_at"])
                self.connection.execute(
                    "UPDATE lease_attempts SET expires_at = ? WHERE lease_id = ?", (expires_at, lease_id)
                )
                renewals.append(
                    LeaseRenewal(assignment_id=lease["assignment_id"], lease_id=lease_id, expires_at=expires_at)
                )
            self.connection.execute(
                "UPDATE worker_sessions SET last_seen_at = ?, last_heartbeat_sent_at = ? WHERE worker_session_id = ?",
                (now, heartbeat.sent_at, heartbeat.worker_session_id),
            )
        return tuple(renewals), tuple(stop_ids)

    def validate_lease_request(self, request: LeaseRequest) -> LeaseRequestDisposition:
        request_digest = sha256_digest(canonical_json_bytes(request))
        with self._transaction():
            existing = self.connection.execute(
                "SELECT * FROM lease_requests WHERE request_id = ?", (request.request_id,)
            ).fetchone()
            if existing is not None:
                if existing["request_digest"] != request_digest:
                    raise ConflictError("lease request ID already has different contents")
                return self._lease_request_disposition(existing)
            session = self.connection.execute(
                "SELECT * FROM worker_sessions WHERE worker_session_id = ?", (request.worker_session_id,)
            ).fetchone()
            if session is None or session["worker_id"] != request.worker_id:
                raise IncompatibleWorkerError("worker session is not registered for this worker")
            if (
                session["last_lease_request_sent_at"] is not None
                and request.sent_at <= session["last_lease_request_sent_at"]
            ):
                raise ConflictError("lease request sent_at must increase within a worker session")
            capabilities = WorkerCapabilities.model_validate_json(session["capabilities_json"])
            if not set(request.environments).issubset(capabilities.environments):
                raise IncompatibleWorkerError("lease request contains an unsupported environment")
            self._validate_policy_ids(request.loaded_policy_ids)
            now = self.clock()
            active_count = self.connection.execute(
                "SELECT COUNT(*) AS count FROM lease_attempts l JOIN assignments a USING (assignment_id) "
                "LEFT JOIN lease_cancellations c USING (lease_id) "
                "WHERE l.worker_id = ? AND l.worker_session_id = ? AND l.state = 'active' AND l.expires_at > ? "
                "AND (a.deadline_at IS NULL OR a.deadline_at > ?) "
                "AND (c.lease_id IS NULL OR c.delivered_at IS NULL)",
                (request.worker_id, request.worker_session_id, now, now),
            ).fetchone()["count"]
            if request.available_slots > capabilities.max_concurrent_assignments - active_count:
                raise CapacityError("requested slots exceed worker session capacity")
            self.connection.execute(
                "INSERT INTO lease_requests "
                "(request_id, worker_session_id, request_digest, state, lease_id, created_at, completed_at) "
                "VALUES (?, ?, ?, 'pending', NULL, ?, NULL)",
                (request.request_id, request.worker_session_id, request_digest, now),
            )
            self.connection.execute(
                "UPDATE worker_sessions SET last_lease_request_sent_at = ? WHERE worker_session_id = ?",
                (request.sent_at, request.worker_session_id),
            )
        return LeaseRequestDisposition("pending")

    def mark_lease_request_no_work(self, request_id: str) -> LeaseRequestDisposition:
        with self._transaction():
            row = self.connection.execute("SELECT * FROM lease_requests WHERE request_id = ?", (request_id,)).fetchone()
            if row is None:
                raise InvalidStateError("lease request has not been validated")
            if row["state"] == "pending":
                self.connection.execute(
                    "UPDATE lease_requests SET state = 'no_work', completed_at = ? "
                    "WHERE request_id = ? AND state = 'pending'",
                    (self.clock(), request_id),
                )
                row = self.connection.execute(
                    "SELECT * FROM lease_requests WHERE request_id = ?", (request_id,)
                ).fetchone()
            return self._lease_request_disposition(row)

    def associate_offered_lease(self, request: LeaseRequest, offered: AssignmentLease) -> LeaseRequestDisposition:
        with self._transaction():
            request_row = self.connection.execute(
                "SELECT * FROM lease_requests WHERE request_id = ?", (request.request_id,)
            ).fetchone()
            if request_row is None:
                raise InvalidStateError("lease request has not been validated")
            disposition = self._lease_request_disposition(request_row)
            if disposition.state != "pending":
                if disposition.lease == offered:
                    return disposition
                raise ConflictError("lease request already has a different disposition")
            self._validate_offered_lease(request, offered)
            associated = self.connection.execute(
                "SELECT request_id FROM lease_requests WHERE lease_id = ? AND request_id != ?",
                (offered.lease_id, request.request_id),
            ).fetchone()
            if associated is not None:
                raise ConflictError("lease is already associated with a different request")
            self.connection.execute(
                "UPDATE lease_requests SET state = 'leased', lease_id = ?, completed_at = ? "
                "WHERE request_id = ? AND state = 'pending'",
                (offered.lease_id, self.clock(), request.request_id),
            )
        return LeaseRequestDisposition("leased", offered)

    def validate_offered_lease(self, request: LeaseRequest, offered: AssignmentLease) -> None:
        self._validate_offered_lease(request, offered)

    def _validate_offered_lease(self, request: LeaseRequest, offered: AssignmentLease) -> None:
        if offered.worker_id != request.worker_id or offered.worker_session_id != request.worker_session_id:
            raise ConflictError("offered lease does not belong to the requesting worker session")
        if offered.assignment.environment not in request.environments:
            raise ConflictError("offered lease environment was not advertised by the worker")
        lease = self.connection.execute(
            "SELECT * FROM lease_attempts WHERE lease_id = ?", (offered.lease_id,)
        ).fetchone()
        assignment = self._assignment(offered.assignment.assignment_id)
        now = self.clock()
        if (
            lease is None
            or lease["assignment_id"] != offered.assignment.assignment_id
            or lease["attempt"] != offered.attempt
            or lease["worker_id"] != offered.worker_id
            or lease["worker_session_id"] != offered.worker_session_id
            or lease["state"] != "active"
            or lease["expires_at"] <= now
            or lease["issued_at"] != offered.issued_at
            or lease["expires_at"] != offered.expires_at
            or assignment["state"] != "leased"
            or (assignment["deadline_at"] is not None and assignment["deadline_at"] <= now)
            or assignment["current_lease_id"] != offered.lease_id
            or canonical_json_bytes(offered.assignment) != assignment["assignment_json"]
        ):
            raise ConflictError("offered lease does not match durable coordinator state")

    def register_scheduler_source(self, spec: EnvironmentSourceSpec) -> None:
        from .environments import EnvironmentSourceSpec

        if not isinstance(spec, EnvironmentSourceSpec):
            raise TypeError("spec must be an EnvironmentSourceSpec")
        immutable = (
            spec.kind,
            spec.environment.id,
            spec.environment.revision,
            spec.weight,
            canonical_json_bytes(list(spec.tasks)),
            canonical_json_bytes(spec.sampling),
            spec.group_size,
            spec.max_attempts,
            spec.result_size_limit_bytes,
            spec.assignment_timeout_seconds,
            int(spec.enabled),
        )
        with self._transaction():
            existing = self.connection.execute(
                "SELECT * FROM scheduler_sources WHERE source_id = ?", (spec.source_id,)
            ).fetchone()
            if existing is not None:
                persisted = (
                    existing["kind"],
                    existing["environment_id"],
                    existing["environment_revision"],
                    existing["weight"],
                    existing["tasks_json"],
                    existing["sampling_json"],
                    existing["group_size"],
                    existing["max_attempts"],
                    existing["result_size_limit_bytes"],
                    existing["assignment_timeout_seconds"],
                    existing["enabled"],
                )
                if persisted == immutable:
                    return
                raise ConflictError("scheduler source already has different immutable contents or settings")
            self.connection.execute(
                "INSERT INTO scheduler_sources "
                "(source_id, kind, environment_id, environment_revision, weight, virtual_finish, cursor, "
                "tasks_json, sampling_json, group_size, max_attempts, result_size_limit_bytes, "
                "assignment_timeout_seconds, enabled) "
                "VALUES (?, ?, ?, ?, ?, COALESCE((SELECT MIN(virtual_finish) FROM scheduler_sources "
                "WHERE enabled = 1), 0), 0, ?, ?, ?, ?, ?, ?, ?)",
                (spec.source_id, *immutable),
            )

    def create_next_group(
        self,
        kind: str | None,
        id_factory: Callable[[], str] = lambda: uuid.uuid4().hex,
        *,
        environments: Sequence[EnvironmentIdentity] | None = None,
    ) -> CreatedGroup | None:
        if kind not in {None, "train", "eval"}:
            raise ValueError("group kind must be train, eval, or None")
        now = self.clock()
        with self._transaction():
            environment_filter = ""
            environment_values: tuple[object, ...] = ()
            if environments is not None:
                identities = tuple(environments)
                if not identities:
                    return None
                environment_filter = (
                    " AND ("
                    + " OR ".join("(environment_id = ? AND environment_revision = ?)" for _ in identities)
                    + ")"
                )
                environment_values = tuple(
                    value for environment in identities for value in (environment.id, environment.revision)
                )
            kind_filter = "" if kind is None else " AND kind = ?"
            kind_values: tuple[object, ...] = () if kind is None else (kind,)
            source = self.connection.execute(
                "SELECT * FROM scheduler_sources WHERE enabled = 1 "
                f"{kind_filter}{environment_filter} ORDER BY virtual_finish, source_id LIMIT 1",
                (*kind_values, *environment_values),
            ).fetchone()
            if source is None:
                return None
            policy_row = self.connection.execute(
                "SELECT p.* FROM runs r JOIN policies p ON p.policy_id = r.active_policy_id WHERE r.singleton = 1"
            ).fetchone()
            if policy_row is None:
                raise InvalidStateError("run has not been initialized")
            self._verify_policy_row(policy_row)
            policy = PolicyManifest.model_validate_json(policy_row["manifest_json"])
            tasks = json.loads(source["tasks_json"])
            cursor = source["cursor"]
            task = tasks[cursor % len(tasks)]
            creation_key = f"source:{source['source_id']}:occurrence:{cursor}"
            existing = self.connection.execute(
                "SELECT group_id, sequence FROM rollout_groups WHERE creation_key = ?", (creation_key,)
            ).fetchone()
            if existing is not None:
                rows = self.connection.execute(
                    "SELECT assignment_json FROM assignments WHERE group_id = ? ORDER BY group_index",
                    (existing["group_id"],),
                ).fetchall()
                if len(rows) != source["group_size"]:
                    raise ConflictError("existing scheduler group is incomplete")
                self.connection.execute(
                    "UPDATE scheduler_sources SET cursor = cursor + 1, "
                    "virtual_finish = virtual_finish + group_size / weight WHERE source_id = ? AND cursor = ?",
                    (source["source_id"], cursor),
                )
                return CreatedGroup(
                    existing["group_id"],
                    creation_key,
                    existing["sequence"],
                    source["source_id"],
                    cursor,
                    tuple(RolloutAssignment.model_validate_json(row["assignment_json"]) for row in rows),
                )
            sequence = self._allocate_group_sequence()
            group_id = id_factory()
            deadline = now + source["assignment_timeout_seconds"] if source["assignment_timeout_seconds"] else None
            environment = {"id": source["environment_id"], "revision": source["environment_revision"]}
            sampling = json.loads(source["sampling_json"])
            assignments = tuple(
                RolloutAssignment(
                    assignment_id=id_factory(),
                    group_id=group_id,
                    group_index=index,
                    group_size=source["group_size"],
                    kind=source["kind"],
                    environment=environment,
                    task_data=task,
                    sampling=sampling,
                    policy=policy,
                    created_at=now,
                    deadline_at=deadline,
                    result_size_limit_bytes=source["result_size_limit_bytes"],
                )
                for index in range(source["group_size"])
            )
            self.connection.execute(
                "INSERT INTO rollout_groups "
                "(group_id, policy_id, kind, environment_id, environment_revision, task_json, sampling_json, "
                "group_size, state, created_at, creation_key, sequence, source_id) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'collecting', ?, ?, ?, ?)",
                (
                    group_id,
                    policy.policy_id,
                    source["kind"],
                    source["environment_id"],
                    source["environment_revision"],
                    canonical_json_bytes(task),
                    source["sampling_json"],
                    source["group_size"],
                    now,
                    creation_key,
                    sequence,
                    source["source_id"],
                ),
            )
            for assignment in assignments:
                self._insert_assignment(assignment, source["max_attempts"])
            self.connection.execute(
                "UPDATE scheduler_sources SET cursor = cursor + 1, "
                "virtual_finish = virtual_finish + group_size / weight "
                "WHERE source_id = ?",
                (source["source_id"],),
            )
        return CreatedGroup(group_id, creation_key, sequence, source["source_id"], cursor, assignments)

    def create_group(self, assignments: Sequence[RolloutAssignment], *, max_attempts: int) -> None:
        if max_attempts < 1:
            raise ValueError("max_attempts must be positive")
        if not assignments:
            raise ValueError("a rollout group must contain assignments")
        first = assignments[0]
        expected_indices = list(range(first.group_size))
        if len(assignments) != first.group_size or sorted(item.group_index for item in assignments) != expected_indices:
            raise ValueError("assignments must contain each group index exactly once")
        common = (
            first.group_id,
            first.group_size,
            first.kind,
            canonical_json_bytes(first.environment),
            canonical_json_bytes(first.task_data),
            canonical_json_bytes(first.sampling),
            first.policy.policy_id,
            policy_manifest_digest(first.policy),
            first.created_at,
            first.deadline_at,
            first.result_size_limit_bytes,
        )
        if any(
            (
                item.group_id,
                item.group_size,
                item.kind,
                canonical_json_bytes(item.environment),
                canonical_json_bytes(item.task_data),
                canonical_json_bytes(item.sampling),
                item.policy.policy_id,
                policy_manifest_digest(item.policy),
                item.created_at,
                item.deadline_at,
                item.result_size_limit_bytes,
            )
            != common
            for item in assignments
        ):
            raise ValueError("all assignments in a group must have identical group inputs")
        if len({item.assignment_id for item in assignments}) != len(assignments):
            raise ValueError("assignment IDs must be unique")
        with self._transaction():
            policy = self.connection.execute(
                "SELECT manifest_digest FROM policies WHERE policy_id = ?", (first.policy.policy_id,)
            ).fetchone()
            if policy is None or policy["manifest_digest"] != policy_manifest_digest(first.policy):
                raise ConflictError("assignment policy is not the exact published policy")
            sequence = self._allocate_group_sequence()
            creation_key = f"manual:{first.group_id}"
            self.connection.execute(
                "INSERT INTO rollout_groups "
                "(group_id, policy_id, kind, environment_id, environment_revision, task_json, sampling_json, "
                "group_size, state, created_at, creation_key, sequence) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'collecting', ?, ?, ?)",
                (
                    first.group_id,
                    first.policy.policy_id,
                    first.kind,
                    first.environment.id,
                    first.environment.revision,
                    canonical_json_bytes(first.task_data),
                    canonical_json_bytes(first.sampling),
                    first.group_size,
                    first.created_at,
                    creation_key,
                    sequence,
                ),
            )
            for assignment in sorted(assignments, key=lambda item: item.group_index):
                self._insert_assignment(assignment, max_attempts)

    def lease_next_compatible(
        self,
        request: LeaseRequest,
        *,
        lease_duration_seconds: float,
        lease_id_factory: Callable[[], str] = lambda: f"lease-{secrets.token_hex(32)}",
    ) -> AssignmentLease | None:
        if lease_duration_seconds <= 0:
            raise ValueError("lease duration must be positive")
        now = self.clock()
        with self._transaction():
            request_row = self.connection.execute(
                "SELECT * FROM lease_requests WHERE request_id = ?", (request.request_id,)
            ).fetchone()
            if request_row is None:
                raise InvalidStateError("lease request has not been validated")
            disposition = self._lease_request_disposition(request_row)
            if disposition.state == "leased":
                return disposition.lease
            if disposition.state == "no_work":
                return None
            self._expire_leases(now)
            session = self.connection.execute(
                "SELECT * FROM worker_sessions WHERE worker_session_id = ?", (request.worker_session_id,)
            ).fetchone()
            if session is None or session["worker_id"] != request.worker_id:
                raise IncompatibleWorkerError("worker session is not registered for this worker")
            capabilities = WorkerCapabilities.model_validate_json(session["capabilities_json"])
            requested = set(request.environments)
            if not requested.issubset(capabilities.environments):
                raise IncompatibleWorkerError("lease request contains an unsupported environment")
            self._validate_policy_ids(request.loaded_policy_ids)
            settings = self._scheduler_settings()
            if settings["max_policy_lag"] is None or settings["loaded_policy_preference_seconds"] is None:
                raise InvalidStateError("scheduler settings have not been configured")
            self._cancel_stale_train_groups(settings["max_policy_lag"], now)
            active_count = self.connection.execute(
                "SELECT COUNT(*) AS count FROM lease_attempts l JOIN assignments a USING (assignment_id) "
                "LEFT JOIN lease_cancellations c USING (lease_id) "
                "WHERE l.worker_id = ? AND l.worker_session_id = ? AND l.state = 'active' AND l.expires_at > ? "
                "AND (a.deadline_at IS NULL OR a.deadline_at > ?) "
                "AND (c.lease_id IS NULL OR c.delivered_at IS NULL)",
                (request.worker_id, request.worker_session_id, now, now),
            ).fetchone()["count"]
            remaining_capacity = capabilities.max_concurrent_assignments - active_count
            if remaining_capacity < 1:
                return None
            environment_clauses = " OR ".join(
                "(g.environment_id = ? AND g.environment_revision = ?)" for _ in request.environments
            )
            environment_values = tuple(
                value for environment in request.environments for value in (environment.id, environment.revision)
            )
            candidates = self.connection.execute(
                "SELECT a.*, g.sequence, g.environment_id, g.environment_revision "
                "FROM assignments a JOIN rollout_groups g USING (group_id) "
                "LEFT JOIN assignment_cancellations c USING (assignment_id) "
                "WHERE a.state IN ('pending', 'retry_wait') AND a.available_at <= ? "
                "AND (a.deadline_at IS NULL OR a.deadline_at > ?) AND c.assignment_id IS NULL "
                f"AND ({environment_clauses}) "
                "ORDER BY g.sequence, a.group_index, a.assignment_id",
                (now, now, *environment_values),
            ).fetchall()
            if not candidates:
                return None
            oldest = candidates[0]
            chosen = oldest
            if now - oldest["available_at"] < settings["loaded_policy_preference_seconds"]:
                loaded = set(request.loaded_policy_ids)
                chosen = next((candidate for candidate in candidates if candidate["policy_id"] in loaded), oldest)
            deadline = chosen["deadline_at"]
            expires_at = (
                min(now + lease_duration_seconds, deadline) if deadline is not None else now + lease_duration_seconds
            )
            attempt = chosen["attempt_count"] + 1
            lease_id = lease_id_factory()
            self.connection.execute(
                "INSERT INTO lease_attempts "
                "(lease_id, assignment_id, attempt, worker_id, worker_session_id, issued_at, expires_at, state) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, 'active')",
                (
                    lease_id,
                    chosen["assignment_id"],
                    attempt,
                    request.worker_id,
                    request.worker_session_id,
                    now,
                    expires_at,
                ),
            )
            self.connection.execute(
                "UPDATE assignments SET state = 'leased', attempt_count = ?, current_lease_id = ? "
                "WHERE assignment_id = ?",
                (attempt, lease_id, chosen["assignment_id"]),
            )
            self.connection.execute(
                "UPDATE lease_requests SET state = 'leased', lease_id = ?, completed_at = ? "
                "WHERE request_id = ? AND state = 'pending'",
                (lease_id, now, request.request_id),
            )
        return AssignmentLease(
            lease_id=lease_id,
            attempt=attempt,
            worker_id=request.worker_id,
            worker_session_id=request.worker_session_id,
            issued_at=now,
            expires_at=expires_at,
            assignment=RolloutAssignment.model_validate_json(chosen["assignment_json"]),
        )

    def lease_or_create_next_compatible(
        self, request: LeaseRequest, *, lease_duration_seconds: float
    ) -> AssignmentLease | None:
        lease = self.lease_next_compatible(request, lease_duration_seconds=lease_duration_seconds)
        if lease is not None or not self._lease_request_can_generate(request):
            return lease
        created = self.create_next_group(None, environments=request.environments)
        if created is None:
            return None
        return self.lease_next_compatible(request, lease_duration_seconds=lease_duration_seconds)

    def create_lease(
        self,
        assignment_id: str,
        *,
        worker_id: str,
        worker_session_id: str,
        lease_id: str,
        duration_seconds: float,
    ) -> AssignmentLease:
        if duration_seconds <= 0:
            raise ValueError("lease duration must be positive")
        self.expire_leases()
        now = self.clock()
        deadline_passed = False
        with self._transaction():
            assignment = self._assignment(assignment_id)
            if (
                assignment["state"] == "failed"
                and assignment["deadline_at"] is not None
                and assignment["deadline_at"] <= now
            ):
                raise InvalidStateError("assignment deadline has passed")
            if assignment["state"] not in {"pending", "retry_wait"} or assignment["available_at"] > now:
                raise InvalidStateError("assignment is not due for leasing")
            session = self._compatible_session(worker_id, worker_session_id, assignment)
            if assignment["attempt_count"] >= assignment["max_attempts"]:
                raise InvalidStateError("assignment has exhausted its attempts")
            deadline = assignment["deadline_at"]
            if deadline is not None and now >= deadline:
                self._terminalize_without_envelope(assignment, "deadline_exceeded", now)
                deadline_passed = True
            else:
                active_count = self.connection.execute(
                    "SELECT COUNT(*) AS count FROM lease_attempts WHERE worker_id = ? AND worker_session_id = ? "
                    "AND state = 'active'",
                    (worker_id, worker_session_id),
                ).fetchone()["count"]
                capabilities = WorkerCapabilities.model_validate_json(session["capabilities_json"])
                if active_count >= capabilities.max_concurrent_assignments:
                    raise CapacityError("worker session has no free assignment capacity")
                expires_at = min(now + duration_seconds, deadline) if deadline is not None else now + duration_seconds
                attempt = assignment["attempt_count"] + 1
                self.connection.execute(
                    "INSERT INTO lease_attempts "
                    "(lease_id, assignment_id, attempt, worker_id, worker_session_id, issued_at, expires_at, state) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, 'active')",
                    (lease_id, assignment_id, attempt, worker_id, worker_session_id, now, expires_at),
                )
                self.connection.execute(
                    "UPDATE assignments SET state = 'leased', attempt_count = ?, current_lease_id = ? "
                    "WHERE assignment_id = ?",
                    (attempt, lease_id, assignment_id),
                )
        if deadline_passed:
            raise InvalidStateError("assignment deadline has passed")
        model = RolloutAssignment.model_validate_json(assignment["assignment_json"])
        return AssignmentLease(
            lease_id=lease_id,
            attempt=attempt,
            worker_id=worker_id,
            worker_session_id=worker_session_id,
            issued_at=now,
            expires_at=expires_at,
            assignment=model,
        )

    def renew_lease(
        self,
        lease_id: str,
        *,
        worker_id: str,
        worker_session_id: str,
        duration_seconds: float,
        expected_assignment_id: str | None = None,
        sent_at: float | None = None,
        acknowledge_cancellation: bool = False,
    ) -> AssignmentLease | str:
        if duration_seconds <= 0:
            raise ValueError("lease duration must be positive")
        now = self.clock()
        with self._transaction():
            self._reconcile_stale_work(now)
            lease = self.connection.execute("SELECT * FROM lease_attempts WHERE lease_id = ?", (lease_id,)).fetchone()
            if (
                lease is not None
                and expected_assignment_id is not None
                and lease["assignment_id"] != expected_assignment_id
            ):
                raise ConflictError("lease does not belong to the assignment path")
            if lease is None or lease["state"] != "active" or lease["expires_at"] <= now:
                raise InvalidStateError("lease is not active and unexpired")
            if lease["worker_id"] != worker_id or lease["worker_session_id"] != worker_session_id:
                raise ConflictError("lease worker session does not match")
            cancellation = self.connection.execute(
                "SELECT reason FROM lease_cancellations WHERE lease_id = ?", (lease_id,)
            ).fetchone()
            if cancellation is not None:
                if not acknowledge_cancellation:
                    raise InvalidStateError("lease has been cancelled")
                self.connection.execute(
                    "UPDATE lease_cancellations SET delivered_at = COALESCE(delivered_at, ?) WHERE lease_id = ?",
                    (now, lease_id),
                )
                return cancellation["reason"]
            if (
                sent_at is not None
                and lease["last_renew_sent_at"] is not None
                and sent_at <= lease["last_renew_sent_at"]
            ):
                raise ConflictError("renewal sent_at must increase for a lease")
            assignment = self._assignment(lease["assignment_id"])
            deadline = assignment["deadline_at"]
            if deadline is not None and now >= deadline:
                raise InvalidStateError("assignment deadline has passed")
            requested_expiry = max(lease["expires_at"], now + duration_seconds)
            expires_at = min(requested_expiry, deadline) if deadline is not None else requested_expiry
            self.connection.execute(
                "UPDATE lease_attempts SET expires_at = ?, last_renew_sent_at = COALESCE(?, last_renew_sent_at) "
                "WHERE lease_id = ?",
                (expires_at, sent_at, lease_id),
            )
        model = RolloutAssignment.model_validate_json(assignment["assignment_json"])
        return AssignmentLease(
            lease_id=lease_id,
            attempt=lease["attempt"],
            worker_id=worker_id,
            worker_session_id=worker_session_id,
            issued_at=lease["issued_at"],
            expires_at=expires_at,
            assignment=model,
        )

    def get_policy(self, policy_id: str) -> PolicyManifest:
        row = self.connection.execute("SELECT * FROM policies WHERE policy_id = ?", (policy_id,)).fetchone()
        if row is None:
            raise NotFoundError("policy not found")
        self._verify_policy_row(row)
        return PolicyManifest.model_validate_json(row["manifest_json"])

    def resolve_policy_file(self, policy_id: str, name: str) -> tuple[Path, int, str]:
        row = self.connection.execute("SELECT * FROM policies WHERE policy_id = ?", (policy_id,)).fetchone()
        if row is None:
            raise NotFoundError("policy not found")
        self._verify_policy_row(row)
        manifest = PolicyManifest.model_validate_json(row["manifest_json"])
        listed = (
            next((item for item in manifest.adapter.files if item.name == name), None) if manifest.adapter else None
        )
        if listed is None:
            raise NotFoundError("policy file not found")
        raw_path = self.run_root / row["artifact_path"] / name
        if raw_path.is_symlink():
            raise ArtifactCorruptionError("published policy file is a symlink")
        try:
            path = raw_path.resolve()
            path.relative_to(self.run_root)
            if not path.is_file() or path.stat().st_size != listed.size_bytes:
                raise ValueError("policy file size does not match manifest")
            if self.spool.file_digest(path) != listed.digest:
                raise ValueError("policy file digest does not match manifest")
        except (OSError, ValueError) as error:
            raise ArtifactCorruptionError("published policy file is corrupt") from error
        return path, listed.size_bytes, listed.digest

    def open_policy_file(self, policy_id: str, name: str) -> tuple[BinaryIO, int, str]:
        row = self.connection.execute("SELECT * FROM policies WHERE policy_id = ?", (policy_id,)).fetchone()
        if row is None:
            raise NotFoundError("policy not found")
        self._verify_policy_row(row)
        manifest = PolicyManifest.model_validate_json(row["manifest_json"])
        listed = (
            next((item for item in manifest.adapter.files if item.name == name), None) if manifest.adapter else None
        )
        if listed is None:
            raise NotFoundError("policy file not found")
        raw_path = self.run_root / row["artifact_path"] / name
        try:
            descriptor = os.open(raw_path, os.O_RDONLY | os.O_NOFOLLOW)
        except OSError as error:
            raise ArtifactCorruptionError("published policy file cannot be opened safely") from error
        file: BinaryIO | None = None
        try:
            file_stat = os.fstat(descriptor)
            if not stat.S_ISREG(file_stat.st_mode) or file_stat.st_size != listed.size_bytes:
                raise ValueError("policy file size does not match manifest")
            file = os.fdopen(descriptor, "rb")
            descriptor = -1
            return file, listed.size_bytes, listed.digest
        except (OSError, ValueError) as error:
            if descriptor >= 0:
                os.close(descriptor)
            if file is not None:
                file.close()
            raise ArtifactCorruptionError("published policy file is corrupt") from error

    def assignment_result_size_limit(self, assignment_id: str) -> int:
        assignment = RolloutAssignment.model_validate_json(self._assignment(assignment_id)["assignment_json"])
        return assignment.result_size_limit_bytes

    def verify_ready(self) -> None:
        run = self._run()
        row = self.connection.execute(
            "SELECT * FROM policies WHERE policy_id = ?", (run["active_policy_id"],)
        ).fetchone()
        if row is None:
            raise ArtifactCorruptionError("active policy is missing")
        self._verify_policy_row(row)

    def status_snapshot(self, *, stale_after_seconds: float) -> dict[str, object]:
        if stale_after_seconds < 0:
            raise ValueError("stale_after_seconds must be non-negative")
        now = self.clock()
        with self._transaction():
            run = self._run()
            active_policy_version = self.connection.execute(
                "SELECT policy_version FROM policies WHERE policy_id = ?", (run["active_policy_id"],)
            ).fetchone()["policy_version"]
            workers = self.connection.execute("SELECT COUNT(*) AS count FROM workers").fetchone()["count"]
            sessions = self.connection.execute("SELECT COUNT(*) AS count FROM worker_sessions").fetchone()["count"]
            stale_sessions = self.connection.execute(
                "SELECT COUNT(*) AS count FROM worker_sessions WHERE last_seen_at < ?", (now - stale_after_seconds,)
            ).fetchone()["count"]
            active_leases = self.connection.execute(
                "SELECT COUNT(*) AS count FROM lease_attempts WHERE state = 'active' AND expires_at > ?", (now,)
            ).fetchone()["count"]
            counts: dict[str, dict[str, int]] = {}
            for table, name, state_column in (
                ("assignments", "assignments", "state"),
                ("rollout_groups", "groups", "state"),
                ("accepted_results", "results", "processing_state"),
            ):
                counts[name] = {
                    row["state"]: row["count"]
                    for row in self.connection.execute(
                        f"SELECT {state_column} AS state, COUNT(*) AS count FROM {table} GROUP BY {state_column}"
                    ).fetchall()
                }
        return {
            "run_id": run["run_id"],
            "active_policy_id": run["active_policy_id"],
            "active_policy_version": active_policy_version,
            "workers": workers,
            "worker_sessions": sessions,
            "stale_worker_sessions": stale_sessions,
            "active_leases": active_leases,
            **counts,
            "stale_cutoff": now - stale_after_seconds,
            "server_time": now,
        }

    def expire_leases(self) -> int:
        now = self.clock()
        with self._transaction():
            self._reconcile_stale_work(now)
            return self._expire_leases(now)

    def accept_result(self, envelope: ResultEnvelope) -> AcceptanceRecord:
        existing_artifact = self.connection.execute(
            "SELECT artifact_path FROM accepted_results WHERE assignment_id = ?", (envelope.assignment_id,)
        ).fetchone()
        legacy_json = existing_artifact is not None and existing_artifact["artifact_path"].endswith(".json")
        envelope_bytes = canonical_json_bytes(envelope) if legacy_json else result_envelope_bytes(envelope)
        envelope_digest = sha256_digest(envelope_bytes)
        duplicate = self._existing_terminal(envelope.assignment_id, envelope.lease_id, envelope_digest)
        if duplicate is not None:
            self.spool.publish_result(envelope_digest, envelope_bytes, suffix=".json" if legacy_json else ".msgpack")
            return duplicate
        assignment, _ = self._validate_active_envelope(envelope)
        if envelope.requested_policy_id != assignment["policy_id"]:
            raise ConflictError("result policy ID does not match the assignment")
        if envelope.requested_policy_digest != assignment["policy_manifest_digest"]:
            raise ConflictError("result policy manifest digest does not match the assignment")
        artifact_path = self.spool.publish_result(
            envelope_digest, envelope_bytes, suffix=".json" if legacy_json else ".msgpack"
        )
        now = self.clock()
        with self._transaction():
            duplicate = self._existing_terminal(envelope.assignment_id, envelope.lease_id, envelope_digest)
            if duplicate is not None:
                return duplicate
            assignment, _ = self._validate_active_envelope(envelope, now=now)
            if envelope.requested_policy_id != assignment["policy_id"]:
                raise ConflictError("result policy ID does not match the assignment")
            if envelope.requested_policy_digest != assignment["policy_manifest_digest"]:
                raise ConflictError("result policy manifest digest does not match the assignment")
            self.connection.execute(
                "INSERT INTO accepted_results VALUES (?, ?, ?, ?, ?, ?, 'pending')",
                (envelope.assignment_id, envelope.lease_id, envelope_digest, artifact_path, len(envelope_bytes), now),
            )
            self.connection.execute(
                "INSERT INTO assignment_outcomes VALUES (?, 'result', ?, ?, ?)",
                (envelope.assignment_id, envelope.lease_id, envelope_digest, now),
            )
            self.connection.execute(
                "UPDATE lease_attempts SET state = 'succeeded' WHERE lease_id = ?", (envelope.lease_id,)
            )
            self.connection.execute(
                "UPDATE assignments SET state = 'succeeded', current_lease_id = NULL WHERE assignment_id = ?",
                (envelope.assignment_id,),
            )
            self.connection.execute(
                "UPDATE assignment_cancellations SET terminal = 1 WHERE assignment_id = ?",
                (envelope.assignment_id,),
            )
            self._recompute_group(assignment["group_id"])
        return AcceptanceRecord(envelope.assignment_id, envelope_digest, False, True)

    def accept_failure(self, envelope: FailureEnvelope) -> AcceptanceRecord:
        envelope_bytes = canonical_json_bytes(envelope)
        envelope_digest = sha256_digest(envelope_bytes)
        duplicate = self._existing_failure(envelope, envelope_digest)
        if duplicate is not None:
            return duplicate
        now = self.clock()
        with self._transaction():
            duplicate = self._existing_failure(envelope, envelope_digest)
            if duplicate is not None:
                return duplicate
            assignment, _ = self._validate_active_envelope(envelope, now=now)
            self.connection.execute(
                "UPDATE lease_attempts SET state = 'failed' WHERE lease_id = ?", (envelope.lease_id,)
            )
            cancellation = self.connection.execute(
                "SELECT 1 FROM lease_cancellations WHERE lease_id = ?", (envelope.lease_id,)
            ).fetchone()
            if cancellation is not None:
                self.connection.execute(
                    "UPDATE assignments SET state = 'failed', current_lease_id = NULL WHERE assignment_id = ?",
                    (envelope.assignment_id,),
                )
                self.connection.execute(
                    "UPDATE assignment_cancellations SET terminal = 1 WHERE assignment_id = ?",
                    (envelope.assignment_id,),
                )
                self.connection.execute(
                    "INSERT INTO assignment_outcomes "
                    "(assignment_id, outcome, lease_id, envelope_digest, completed_at) "
                    "VALUES (?, 'failure', ?, ?, ?)",
                    (envelope.assignment_id, envelope.lease_id, envelope_digest, now),
                )
                terminal = True
            else:
                terminal = self._retry_or_terminal(
                    assignment, now, envelope=envelope, digest=envelope_digest, lease_id=envelope.lease_id
                )
            self.connection.execute(
                "INSERT INTO failures VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    envelope.lease_id,
                    envelope.assignment_id,
                    envelope_digest,
                    envelope_bytes,
                    now,
                    envelope.retryable,
                    terminal,
                ),
            )
            self._recompute_group(assignment["group_id"])
        return AcceptanceRecord(envelope.assignment_id, envelope_digest, False, terminal)

    def claim_pending_results(self, limit: int) -> list[PendingResult]:
        if limit < 1:
            raise ValueError("claim limit must be positive")
        with self._transaction():
            rows = self.connection.execute(
                "SELECT assignment_id, envelope_digest, artifact_path FROM accepted_results "
                "WHERE processing_state = 'pending' ORDER BY accepted_at, assignment_id LIMIT ?",
                (limit,),
            ).fetchall()
            self.connection.executemany(
                "UPDATE accepted_results SET processing_state = 'processing' WHERE assignment_id = ?",
                ((row["assignment_id"],) for row in rows),
            )
        return [
            PendingResult(row["assignment_id"], row["envelope_digest"], self.run_root / row["artifact_path"])
            for row in rows
        ]

    def mark_result_processed(self, assignment_id: str) -> None:
        with self._transaction():
            cursor = self.connection.execute(
                "UPDATE accepted_results SET processing_state = 'processed' "
                "WHERE assignment_id = ? AND processing_state = 'processing'",
                (assignment_id,),
            )
            if cursor.rowcount != 1:
                raise InvalidStateError("result is not claimed for processing")

    def ready_groups(self, limit: int = 1) -> tuple[ReadyGroup, ...]:
        if limit < 1:
            raise ValueError("limit must be positive")
        groups = self.connection.execute(
            "SELECT g.* FROM rollout_groups g LEFT JOIN processed_groups p USING (group_id) "
            "WHERE g.state = 'ready' AND p.group_id IS NULL ORDER BY g.sequence, g.group_id LIMIT ?",
            (limit,),
        ).fetchall()
        ready: list[ReadyGroup] = []
        for group in groups:
            rows = self.connection.execute(
                "SELECT a.assignment_json, o.outcome, r.envelope_digest, r.artifact_path "
                "FROM assignments a JOIN assignment_outcomes o USING (assignment_id) "
                "LEFT JOIN accepted_results r USING (assignment_id) "
                "WHERE a.group_id = ? ORDER BY a.group_index, a.assignment_id",
                (group["group_id"],),
            ).fetchall()
            if len(rows) != group["group_size"]:
                raise InvalidStateError("ready group does not have one terminal outcome per assignment")
            outcomes = tuple(
                GroupOutcome(
                    assignment=RolloutAssignment.model_validate_json(row["assignment_json"]),
                    outcome=row["outcome"],
                    envelope_digest=row["envelope_digest"],
                    result_path=None if row["artifact_path"] is None else self.run_root / row["artifact_path"],
                )
                for row in rows
            )
            ready.append(
                ReadyGroup(
                    group_id=group["group_id"],
                    source_id=group["source_id"],
                    kind=group["kind"],
                    sequence=group["sequence"],
                    outcomes=outcomes,
                )
            )
        return tuple(ready)

    def record_processed_group(
        self,
        group_id: str,
        *,
        input_digest: str,
        artifact_digest: str,
        artifact_path: Path,
        size_bytes: int,
        token_counts: Sequence[int],
    ) -> None:
        relative_path = artifact_path.relative_to(self.run_root)
        now = self.clock()
        with self._transaction():
            existing = self.connection.execute(
                "SELECT * FROM processed_groups WHERE group_id = ?", (group_id,)
            ).fetchone()
            expected = (
                input_digest,
                artifact_digest,
                str(relative_path),
                size_bytes,
                len(token_counts),
            )
            if existing is not None:
                persisted = (
                    existing["input_digest"],
                    existing["artifact_digest"],
                    existing["artifact_path"],
                    existing["size_bytes"],
                    existing["rollout_count"],
                )
                if persisted != expected:
                    raise ConflictError("processed group already has different output")
                return
            group = self.connection.execute(
                "SELECT state FROM rollout_groups WHERE group_id = ?", (group_id,)
            ).fetchone()
            if group is None or group["state"] != "ready":
                raise InvalidStateError("group is not ready for processing")
            self.connection.execute(
                "INSERT INTO processed_groups "
                "(group_id, input_digest, artifact_digest, artifact_path, size_bytes, rollout_count, processed_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (group_id, input_digest, artifact_digest, str(relative_path), size_bytes, len(token_counts), now),
            )
            self.connection.executemany(
                "INSERT INTO processed_rollouts(group_id, ordinal, token_count) VALUES (?, ?, ?)",
                ((group_id, ordinal, token_count) for ordinal, token_count in enumerate(token_counts)),
            )
            self.connection.execute(
                "UPDATE accepted_results SET processing_state = 'processed' WHERE assignment_id IN "
                "(SELECT assignment_id FROM assignments WHERE group_id = ?)",
                (group_id,),
            )

    def pending_processed_rollouts(self) -> tuple[PendingProcessedRollout, ...]:
        rows = self.connection.execute(
            "SELECT r.group_id, r.ordinal, r.token_count, p.artifact_path, p.artifact_digest, p.size_bytes "
            "FROM processed_rollouts r JOIN processed_groups p USING (group_id) "
            "JOIN rollout_groups g USING (group_id) WHERE r.batch_step IS NULL AND r.discarded = 0 "
            "ORDER BY g.sequence, r.ordinal, r.group_id"
        ).fetchall()
        return tuple(
            PendingProcessedRollout(
                group_id=row["group_id"],
                ordinal=row["ordinal"],
                token_count=row["token_count"],
                artifact_path=self.run_root / row["artifact_path"],
                artifact_digest=row["artifact_digest"],
                size_bytes=row["size_bytes"],
            )
            for row in rows
        )

    def processed_groups(self) -> tuple[ProcessedGroupRecord, ...]:
        rows = self.connection.execute("SELECT * FROM processed_groups ORDER BY processed_at, group_id").fetchall()
        return tuple(
            ProcessedGroupRecord(
                group_id=row["group_id"],
                artifact_digest=row["artifact_digest"],
                artifact_path=self.run_root / row["artifact_path"],
                size_bytes=row["size_bytes"],
            )
            for row in rows
        )

    def next_training_batch_step(self) -> int:
        return self.connection.execute("SELECT COALESCE(MAX(step), 0) + 1 FROM training_batches").fetchone()[0]

    def discard_processed_rollouts(self, members: Sequence[tuple[str, int]]) -> None:
        if not members:
            raise ValueError("discarded rollout membership must not be empty")
        with self._transaction():
            updated = 0
            for group_id, ordinal in members:
                cursor = self.connection.execute(
                    "UPDATE processed_rollouts SET discarded = 1 "
                    "WHERE group_id = ? AND ordinal = ? AND batch_step IS NULL AND discarded = 0",
                    (group_id, ordinal),
                )
                updated += cursor.rowcount
            if updated != len(members):
                raise ConflictError("processed rollout is no longer available for discard")

    def record_training_batch(
        self,
        *,
        step: int,
        artifact_digest: str,
        artifact_path: Path,
        size_bytes: int,
        sample_count: int,
        members: Sequence[tuple[str, int]],
    ) -> None:
        if not members or sample_count < 1:
            raise ValueError("training batch must contain members and samples")
        relative_path = artifact_path.relative_to(self.run_root)
        with self._transaction():
            existing = self.connection.execute("SELECT * FROM training_batches WHERE step = ?", (step,)).fetchone()
            expected = (artifact_digest, str(relative_path), size_bytes, sample_count)
            if existing is not None:
                persisted = (
                    existing["artifact_digest"],
                    existing["artifact_path"],
                    existing["size_bytes"],
                    existing["sample_count"],
                )
                if persisted != expected:
                    raise ConflictError("training batch step already has different output")
                return
            if step != self.next_training_batch_step():
                raise InvalidStateError("training batch step is not next in sequence")
            placeholders = ", ".join("(?, ?)" for _ in members)
            values = tuple(value for member in members for value in member)
            available = self.connection.execute(
                f"SELECT COUNT(*) FROM processed_rollouts WHERE batch_step IS NULL AND discarded = 0 "
                f"AND (group_id, ordinal) IN "
                f"({placeholders})",
                values,
            ).fetchone()[0]
            if available != len(members):
                raise ConflictError("training batch members are no longer available")
            self.connection.execute(
                "INSERT INTO training_batches "
                "(step, artifact_digest, artifact_path, size_bytes, sample_count, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (step, artifact_digest, str(relative_path), size_bytes, sample_count, self.clock()),
            )
            self.connection.executemany(
                "UPDATE processed_rollouts SET batch_step = ?, batch_ordinal = ? "
                "WHERE group_id = ? AND ordinal = ? AND batch_step IS NULL",
                ((step, batch_ordinal, *member) for batch_ordinal, member in enumerate(members)),
            )

    def training_batches(self) -> tuple[TrainingBatchRecord, ...]:
        rows = self.connection.execute("SELECT * FROM training_batches ORDER BY step").fetchall()
        return tuple(
            TrainingBatchRecord(
                step=row["step"],
                artifact_digest=row["artifact_digest"],
                artifact_path=self.run_root / row["artifact_path"],
                size_bytes=row["size_bytes"],
                sample_count=row["sample_count"],
            )
            for row in rows
        )

    def recover(self) -> None:
        removed_incoming = False
        for path in self.spool.incoming_dir.iterdir():
            if path.is_file() or path.is_symlink():
                path.unlink(missing_ok=True)
                removed_incoming = True
        if removed_incoming:
            self.spool._fsync_directory(self.spool.incoming_dir)
        rows = self.connection.execute(
            "SELECT assignment_id, lease_id, artifact_path, envelope_digest, size_bytes FROM accepted_results"
        ).fetchall()
        referenced: set[Path] = set()
        for row in rows:
            try:
                path = self.spool.resolve_result(row["artifact_path"])
            except (OSError, ValueError) as error:
                raise ArtifactCorruptionError("referenced result artifact path is invalid") from error
            referenced.add(path.resolve())
            if path.is_symlink() or not path.is_file():
                raise ArtifactCorruptionError(f"referenced result artifact is missing: {path}")
            if path.stat().st_size != row["size_bytes"] or self.spool.file_digest(path) != row["envelope_digest"]:
                raise ArtifactCorruptionError(f"referenced result artifact is corrupt: {path}")
            try:
                data = path.read_bytes()
                envelope = (
                    ResultEnvelope.model_validate_json(data) if path.suffix == ".json" else decode_result_envelope(data)
                )
            except ValueError as error:
                raise ArtifactCorruptionError(f"referenced result envelope is invalid: {path}") from error
            if envelope.assignment_id != row["assignment_id"] or envelope.lease_id != row["lease_id"]:
                raise ArtifactCorruptionError(f"referenced result identity does not match its database row: {path}")
        policies = self.connection.execute(
            "SELECT * FROM policies WHERE policy_version > 0 ORDER BY policy_version"
        ).fetchall()
        for policy in policies:
            self._verify_policy_row(policy)
        removed_result = False
        for path in self.spool.results_dir.iterdir():
            if path.is_file() and path.resolve() not in referenced:
                path.unlink()
                removed_result = True
        if removed_result:
            self.spool._fsync_directory(self.spool.results_dir)
        with self._transaction():
            self.connection.execute(
                "UPDATE accepted_results SET processing_state = 'pending' WHERE processing_state = 'processing'"
            )
        self.expire_leases()
        with self._transaction():
            self._recompute_groups()

    def assignment_state(self, assignment_id: str) -> str:
        return self._assignment(assignment_id)["state"]

    def group_state(self, group_id: str) -> str:
        row = self.connection.execute("SELECT state FROM rollout_groups WHERE group_id = ?", (group_id,)).fetchone()
        if row is None:
            raise NotFoundError("rollout group not found")
        return row["state"]

    def cancellation_state(self, assignment_id: str) -> tuple[str, bool] | None:
        row = self.connection.execute(
            "SELECT reason, terminal FROM assignment_cancellations WHERE assignment_id = ?", (assignment_id,)
        ).fetchone()
        return None if row is None else (row["reason"], bool(row["terminal"]))

    def _run(self) -> sqlite3.Row:
        row = self.connection.execute("SELECT * FROM runs WHERE singleton = 1").fetchone()
        if row is None:
            raise InvalidStateError("run has not been initialized")
        return row

    def _scheduler_settings(self) -> sqlite3.Row:
        row = self.connection.execute("SELECT * FROM scheduler_state WHERE singleton = 1").fetchone()
        if row is None:
            raise InvalidStateError("scheduler state is missing")
        return row

    def _reconcile_stale_work(self, now: float) -> None:
        max_policy_lag = self._scheduler_settings()["max_policy_lag"]
        if max_policy_lag is not None:
            self._cancel_stale_train_groups(max_policy_lag, now)

    def _lease_request_disposition(self, row: sqlite3.Row) -> LeaseRequestDisposition:
        lease = self._assignment_lease(row["lease_id"]) if row["state"] == "leased" else None
        return LeaseRequestDisposition(row["state"], lease)

    def _assignment_lease(self, lease_id: str) -> AssignmentLease:
        row = self.connection.execute(
            "SELECT l.*, a.assignment_json FROM lease_attempts l JOIN assignments a USING (assignment_id) "
            "WHERE l.lease_id = ?",
            (lease_id,),
        ).fetchone()
        if row is None:
            raise InvalidStateError("persisted lease request references a missing lease")
        return AssignmentLease(
            lease_id=row["lease_id"],
            attempt=row["attempt"],
            worker_id=row["worker_id"],
            worker_session_id=row["worker_session_id"],
            issued_at=row["issued_at"],
            expires_at=row["expires_at"],
            assignment=RolloutAssignment.model_validate_json(row["assignment_json"]),
        )

    def _lease_request_can_generate(self, request: LeaseRequest) -> bool:
        now = self.clock()
        with self._transaction():
            row = self.connection.execute(
                "SELECT state FROM lease_requests WHERE request_id = ?", (request.request_id,)
            ).fetchone()
            if row is None or row["state"] != "pending":
                return False
            session = self.connection.execute(
                "SELECT capabilities_json FROM worker_sessions WHERE worker_session_id = ? AND worker_id = ?",
                (request.worker_session_id, request.worker_id),
            ).fetchone()
            if session is None:
                return False
            capabilities = WorkerCapabilities.model_validate_json(session["capabilities_json"])
            active_count = self.connection.execute(
                "SELECT COUNT(*) AS count FROM lease_attempts l JOIN assignments a USING (assignment_id) "
                "LEFT JOIN lease_cancellations c USING (lease_id) "
                "WHERE l.worker_id = ? AND l.worker_session_id = ? AND l.state = 'active' AND l.expires_at > ? "
                "AND (a.deadline_at IS NULL OR a.deadline_at > ?) "
                "AND (c.lease_id IS NULL OR c.delivered_at IS NULL)",
                (request.worker_id, request.worker_session_id, now, now),
            ).fetchone()["count"]
            if active_count >= capabilities.max_concurrent_assignments:
                return False
            environment_clauses = " OR ".join(
                "(g.environment_id = ? AND g.environment_revision = ?)" for _ in request.environments
            )
            environment_values = tuple(
                value for environment in request.environments for value in (environment.id, environment.revision)
            )
            pending = self.connection.execute(
                "SELECT 1 FROM assignments a JOIN rollout_groups g USING (group_id) "
                "LEFT JOIN assignment_cancellations c USING (assignment_id) "
                "WHERE a.state IN ('pending', 'retry_wait') AND a.available_at <= ? "
                "AND (a.deadline_at IS NULL OR a.deadline_at > ?) AND c.assignment_id IS NULL "
                f"AND ({environment_clauses}) LIMIT 1",
                (now, now, *environment_values),
            ).fetchone()
            return pending is None

    def _assignment(self, assignment_id: str) -> sqlite3.Row:
        row = self.connection.execute("SELECT * FROM assignments WHERE assignment_id = ?", (assignment_id,)).fetchone()
        if row is None:
            raise NotFoundError("assignment not found")
        return row

    def _compatible_session(self, worker_id: str, worker_session_id: str, assignment: sqlite3.Row) -> sqlite3.Row:
        session = self.connection.execute(
            "SELECT * FROM worker_sessions WHERE worker_session_id = ?", (worker_session_id,)
        ).fetchone()
        if session is None or session["worker_id"] != worker_id:
            raise IncompatibleWorkerError("worker session is not registered for this worker")
        capabilities = WorkerCapabilities.model_validate_json(session["capabilities_json"])
        model = RolloutAssignment.model_validate_json(assignment["assignment_json"])
        if model.environment not in capabilities.environments:
            raise IncompatibleWorkerError("worker does not support the assignment environment")
        return session

    def _validate_policy_ids(self, policy_ids: Sequence[str]) -> None:
        if not policy_ids:
            return
        rows = self.connection.execute(
            f"SELECT policy_id FROM policies WHERE policy_id IN ({','.join('?' for _ in policy_ids)})", policy_ids
        ).fetchall()
        if {row["policy_id"] for row in rows} != set(policy_ids):
            raise IncompatibleWorkerError("worker reported an unknown policy ID")

    def _validate_active_envelope(
        self,
        envelope: ResultEnvelope | FailureEnvelope,
        *,
        now: float | None = None,
    ) -> tuple[sqlite3.Row, sqlite3.Row]:
        received_at = self.clock() if now is None else now
        assignment = self._assignment(envelope.assignment_id)
        lease = self.connection.execute(
            "SELECT * FROM lease_attempts WHERE lease_id = ?", (envelope.lease_id,)
        ).fetchone()
        if (
            lease is None
            or lease["assignment_id"] != envelope.assignment_id
            or lease["attempt"] != envelope.attempt
            or lease["worker_id"] != envelope.worker_id
            or lease["worker_session_id"] != envelope.worker_session_id
            or lease["state"] != "active"
            or assignment["state"] != "leased"
            or assignment["current_lease_id"] != envelope.lease_id
        ):
            raise ConflictError("envelope does not match the current active lease")
        if received_at >= lease["expires_at"]:
            raise InvalidStateError("lease expired before submission was received")
        if assignment["deadline_at"] is not None and received_at >= assignment["deadline_at"]:
            raise InvalidStateError("assignment deadline passed before submission was received")
        return assignment, lease

    def _existing_terminal(self, assignment_id: str, lease_id: str, digest: str) -> AcceptanceRecord | None:
        result = self.connection.execute(
            "SELECT lease_id, envelope_digest FROM accepted_results WHERE assignment_id = ?", (assignment_id,)
        ).fetchone()
        if result is None:
            outcome = self.connection.execute(
                "SELECT outcome FROM assignment_outcomes WHERE assignment_id = ?", (assignment_id,)
            ).fetchone()
            if outcome is not None:
                raise ConflictError("assignment already has a terminal outcome")
            return None
        if result["lease_id"] != lease_id or result["envelope_digest"] != digest:
            raise ConflictError("assignment already has a different accepted result")
        return AcceptanceRecord(assignment_id, digest, True, True)

    def _existing_failure(self, envelope: FailureEnvelope, digest: str) -> AcceptanceRecord | None:
        existing = self.connection.execute(
            "SELECT assignment_id, envelope_digest, terminal FROM failures WHERE lease_id = ?",
            (envelope.lease_id,),
        ).fetchone()
        if existing is None:
            return None
        if existing["assignment_id"] != envelope.assignment_id or existing["envelope_digest"] != digest:
            raise ConflictError("lease already has a different failure envelope")
        return AcceptanceRecord(envelope.assignment_id, digest, True, bool(existing["terminal"]))

    def _retry_or_terminal(
        self,
        assignment: sqlite3.Row,
        now: float,
        *,
        envelope: FailureEnvelope | None = None,
        digest: str | None = None,
        lease_id: str | None = None,
    ) -> bool:
        retryable = envelope is None or envelope.retryable
        delay = min(self.retry_max_seconds, self.retry_base_seconds * 2 ** (assignment["attempt_count"] - 1))
        deadline = assignment["deadline_at"]
        can_retry = (
            retryable
            and assignment["attempt_count"] < assignment["max_attempts"]
            and (deadline is None or now + delay < deadline)
        )
        if can_retry:
            self.connection.execute(
                "UPDATE assignments SET state = 'retry_wait', available_at = ?, current_lease_id = NULL "
                "WHERE assignment_id = ?",
                (now + delay, assignment["assignment_id"]),
            )
            return False
        if assignment["attempt_count"] >= assignment["max_attempts"]:
            outcome = "attempts_exhausted"
        elif envelope is not None and not envelope.retryable:
            outcome = "failure"
        elif deadline is not None and now + delay >= deadline:
            outcome = "deadline_exceeded"
        else:
            outcome = "attempts_exhausted"
        self.connection.execute(
            "UPDATE assignments SET state = 'failed', current_lease_id = NULL WHERE assignment_id = ?",
            (assignment["assignment_id"],),
        )
        self.connection.execute(
            "INSERT INTO assignment_outcomes VALUES (?, ?, ?, ?, ?)",
            (
                assignment["assignment_id"],
                outcome,
                envelope.lease_id if envelope else lease_id,
                digest,
                now,
            ),
        )
        return True

    def _expire_leases(self, now: float) -> int:
        leases = self.connection.execute(
            "SELECT a.*, l.lease_id, l.expires_at, c.lease_id IS NOT NULL AS cancelled "
            "FROM lease_attempts l JOIN assignments a USING (assignment_id) "
            "LEFT JOIN lease_cancellations c USING (lease_id) "
            "WHERE l.state = 'active' AND (l.expires_at <= ? OR (a.deadline_at IS NOT NULL AND a.deadline_at <= ?)) "
            "ORDER BY l.expires_at, l.assignment_id",
            (now, now),
        ).fetchall()
        for lease in leases:
            self.connection.execute(
                "UPDATE lease_attempts SET state = 'expired' WHERE lease_id = ?", (lease["lease_id"],)
            )
            if lease["cancelled"]:
                self.connection.execute(
                    "UPDATE assignments SET state = 'failed', current_lease_id = NULL WHERE assignment_id = ?",
                    (lease["assignment_id"],),
                )
                self.connection.execute(
                    "UPDATE assignment_cancellations SET terminal = 1 WHERE assignment_id = ?",
                    (lease["assignment_id"],),
                )
                self.connection.execute(
                    "INSERT OR IGNORE INTO assignment_outcomes "
                    "(assignment_id, outcome, lease_id, envelope_digest, completed_at) "
                    "VALUES (?, 'failure', ?, NULL, ?)",
                    (lease["assignment_id"], lease["lease_id"], now),
                )
            else:
                self._retry_or_terminal(lease, now, lease_id=lease["lease_id"])
        overdue = self.connection.execute(
            "SELECT * FROM assignments WHERE state IN ('pending', 'retry_wait') "
            "AND deadline_at IS NOT NULL AND deadline_at <= ? ORDER BY deadline_at, assignment_id",
            (now,),
        ).fetchall()
        for assignment in overdue:
            self._terminalize_without_envelope(assignment, "deadline_exceeded", now)
        self._recompute_groups()
        return len(leases)

    def _cancel_stale_train_groups(self, max_policy_lag: int, now: float) -> None:
        active_version = self.connection.execute(
            "SELECT p.policy_version FROM runs r JOIN policies p ON p.policy_id = r.active_policy_id "
            "WHERE r.singleton = 1"
        ).fetchone()["policy_version"]
        groups = self.connection.execute(
            "SELECT g.group_id FROM rollout_groups g JOIN policies p USING (policy_id) "
            "WHERE g.kind = 'train' AND ? - p.policy_version > ? ORDER BY g.sequence, g.group_id",
            (active_version, max_policy_lag),
        ).fetchall()
        for group in groups:
            assignments = self.connection.execute(
                "SELECT * FROM assignments WHERE group_id = ? AND state IN ('pending', 'retry_wait', 'leased') "
                "ORDER BY group_index, assignment_id",
                (group["group_id"],),
            ).fetchall()
            for assignment in assignments:
                terminal = assignment["state"] != "leased"
                self.connection.execute(
                    "INSERT OR IGNORE INTO assignment_cancellations "
                    "(assignment_id, reason, requested_at, terminal) VALUES (?, 'policy_stale', ?, ?)",
                    (assignment["assignment_id"], now, int(terminal)),
                )
                if terminal:
                    self.connection.execute(
                        "UPDATE assignments SET state = 'failed', current_lease_id = NULL WHERE assignment_id = ?",
                        (assignment["assignment_id"],),
                    )
                    self.connection.execute(
                        "UPDATE assignment_cancellations SET terminal = 1 WHERE assignment_id = ?",
                        (assignment["assignment_id"],),
                    )
                    self.connection.execute(
                        "INSERT OR IGNORE INTO assignment_outcomes "
                        "(assignment_id, outcome, lease_id, envelope_digest, completed_at) "
                        "VALUES (?, 'failure', NULL, NULL, ?)",
                        (assignment["assignment_id"], now),
                    )
                elif assignment["current_lease_id"] is not None:
                    self.connection.execute(
                        "INSERT OR IGNORE INTO lease_cancellations "
                        "(lease_id, assignment_id, reason, requested_at, delivered_at) "
                        "VALUES (?, ?, 'policy_stale', ?, NULL)",
                        (assignment["current_lease_id"], assignment["assignment_id"], now),
                    )
            self._recompute_group(group["group_id"])

    def _allocate_group_sequence(self) -> int:
        row = self.connection.execute("SELECT next_group_sequence FROM scheduler_state WHERE singleton = 1").fetchone()
        sequence = row["next_group_sequence"]
        self.connection.execute(
            "UPDATE scheduler_state SET next_group_sequence = ? WHERE singleton = 1", (sequence + 1,)
        )
        return sequence

    def _insert_assignment(self, assignment: RolloutAssignment, max_attempts: int) -> None:
        self.connection.execute(
            "INSERT INTO assignments "
            "(assignment_id, group_id, group_index, assignment_json, policy_id, policy_manifest_digest, state, "
            "max_attempts, attempt_count, available_at, deadline_at, current_lease_id) "
            "VALUES (?, ?, ?, ?, ?, ?, 'pending', ?, 0, ?, ?, NULL)",
            (
                assignment.assignment_id,
                assignment.group_id,
                assignment.group_index,
                canonical_json_bytes(assignment),
                assignment.policy.policy_id,
                policy_manifest_digest(assignment.policy),
                max_attempts,
                assignment.created_at,
                assignment.deadline_at,
            ),
        )

    def _terminalize_without_envelope(self, assignment: sqlite3.Row, outcome: str, now: float) -> None:
        self.connection.execute(
            "UPDATE assignments SET state = 'failed', current_lease_id = NULL WHERE assignment_id = ?",
            (assignment["assignment_id"],),
        )
        self.connection.execute(
            "INSERT INTO assignment_outcomes VALUES (?, ?, NULL, NULL, ?)",
            (assignment["assignment_id"], outcome, now),
        )
        self._recompute_group(assignment["group_id"])

    def _is_terminal(self, assignment_id: str) -> bool:
        return self._assignment(assignment_id)["state"] in {"succeeded", "failed"}

    def _recompute_group(self, group_id: str) -> None:
        count = self.connection.execute(
            "SELECT COUNT(*) AS count FROM assignments WHERE group_id = ? AND state NOT IN ('succeeded', 'failed')",
            (group_id,),
        ).fetchone()["count"]
        state = "ready" if count == 0 else "collecting"
        self.connection.execute("UPDATE rollout_groups SET state = ? WHERE group_id = ?", (state, group_id))

    def _recompute_groups(self) -> None:
        group_ids = self.connection.execute("SELECT group_id FROM rollout_groups ORDER BY group_id").fetchall()
        for row in group_ids:
            self._recompute_group(row["group_id"])

    def _relative_path(self, path: Path) -> str:
        try:
            return path.resolve().relative_to(self.run_root).as_posix()
        except ValueError as error:
            raise ValueError("artifact path must be inside the run root") from error

    def _verify_policy_row(self, row: sqlite3.Row) -> None:
        try:
            manifest = PolicyManifest.model_validate_json(row["manifest_json"])
            if (
                manifest.policy_id != row["policy_id"]
                or manifest.policy_version != row["policy_version"]
                or policy_manifest_digest(manifest) != row["manifest_digest"]
            ):
                raise ValueError("policy manifest does not match its database identity")
            if row["policy_version"] == 0:
                return
            raw_path = self.run_root / row["artifact_path"]
            if raw_path.is_symlink():
                raise ValueError("policy artifact directory must not be a symlink")
            artifact_path = raw_path.resolve()
            artifact_path.relative_to(self.run_root)
            if row["manifest_digest"] in self._verified_policy_digests:
                if manifest.adapter is None:
                    raise ValueError("trained policy is missing an adapter manifest")
                for file in manifest.adapter.files:
                    file_path = artifact_path / file.name
                    if file_path.is_symlink() or not file_path.is_file() or file_path.stat().st_size != file.size_bytes:
                        raise ValueError("policy artifact file no longer matches its manifest")
                return
            verify_lora_policy(artifact_path, expected=manifest)
            self._verified_policy_digests.add(row["manifest_digest"])
        except (OSError, ValueError) as error:
            raise ArtifactCorruptionError(f"published policy artifact is corrupt: {row['policy_id']}") from error


CoordinatorState = CoordinatorRepository
