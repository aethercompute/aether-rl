from __future__ import annotations

import fcntl
import os
import sqlite3
import time
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from aether_rl.protocol import (
    AssignmentLease,
    FailureEnvelope,
    PolicyManifest,
    ResultEnvelope,
    RolloutAssignment,
    WorkerCapabilities,
    WorkerRegistration,
    canonical_json_bytes,
    policy_manifest_digest,
    sha256_digest,
)
from aether_rl.trainer.policy import verify_lora_policy

from .migrations import MIGRATIONS, SCHEMA_VERSION
from .spool import AtomicSpool


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
            self.connection = sqlite3.connect(self.database_path, isolation_level=None)
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
        if manifest.policy_version == 0:
            raise ValueError("record_policy only records trained policies")
        verified = verify_lora_policy(Path(artifact_path), expected=manifest)
        relative_path = self._relative_path(Path(artifact_path))
        manifest_json = canonical_json_bytes(verified)
        digest = policy_manifest_digest(verified)
        with self._transaction():
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
        return PolicyManifest.model_validate_json(target["manifest_json"])

    def active_policy(self) -> PolicyManifest:
        row = self.connection.execute(
            "SELECT p.manifest_json FROM runs r JOIN policies p ON p.policy_id = r.active_policy_id "
            "WHERE r.singleton = 1"
        ).fetchone()
        if row is None:
            raise InvalidStateError("run has not been initialized")
        return PolicyManifest.model_validate_json(row["manifest_json"])

    def register_worker(self, registration: WorkerRegistration) -> RegistrationRecord:
        capabilities_json = canonical_json_bytes(registration.capabilities)
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
                (registration.worker_id, registration.registered_at),
            )
            self.connection.execute(
                "INSERT INTO worker_sessions VALUES (?, ?, ?, ?, ?)",
                (
                    registration.worker_session_id,
                    registration.worker_id,
                    capabilities_json,
                    registration.registered_at,
                    registration.registered_at,
                ),
            )
        return RegistrationRecord(registration.worker_id, registration.worker_session_id, True)

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
            self.connection.execute(
                "INSERT INTO rollout_groups VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'collecting', ?)",
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
                ),
            )
            for assignment in sorted(assignments, key=lambda item: item.group_index):
                self.connection.execute(
                    "INSERT INTO assignments VALUES (?, ?, ?, ?, ?, ?, 'pending', ?, 0, ?, ?, NULL)",
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
                    "INSERT INTO lease_attempts VALUES (?, ?, ?, ?, ?, ?, ?, 'active')",
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
    ) -> AssignmentLease:
        if duration_seconds <= 0:
            raise ValueError("lease duration must be positive")
        now = self.clock()
        with self._transaction():
            lease = self.connection.execute("SELECT * FROM lease_attempts WHERE lease_id = ?", (lease_id,)).fetchone()
            if lease is None or lease["state"] != "active" or lease["expires_at"] <= now:
                raise InvalidStateError("lease is not active and unexpired")
            if lease["worker_id"] != worker_id or lease["worker_session_id"] != worker_session_id:
                raise ConflictError("lease worker session does not match")
            assignment = self._assignment(lease["assignment_id"])
            deadline = assignment["deadline_at"]
            if deadline is not None and now >= deadline:
                raise InvalidStateError("assignment deadline has passed")
            requested_expiry = max(lease["expires_at"], now + duration_seconds)
            expires_at = min(requested_expiry, deadline) if deadline is not None else requested_expiry
            self.connection.execute(
                "UPDATE lease_attempts SET expires_at = ? WHERE lease_id = ?", (expires_at, lease_id)
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

    def expire_leases(self) -> int:
        now = self.clock()
        with self._transaction():
            leases = self.connection.execute(
                "SELECT a.*, l.lease_id, l.expires_at FROM lease_attempts l "
                "JOIN assignments a USING (assignment_id) "
                "WHERE l.state = 'active' AND (l.expires_at <= ? OR (a.deadline_at IS NOT NULL AND a.deadline_at <= ?)) "
                "ORDER BY l.expires_at, l.assignment_id",
                (now, now),
            ).fetchall()
            for lease in leases:
                self.connection.execute(
                    "UPDATE lease_attempts SET state = 'expired' WHERE lease_id = ?", (lease["lease_id"],)
                )
                self._retry_or_terminal(lease, now)
            overdue = self.connection.execute(
                "SELECT * FROM assignments WHERE state IN ('pending', 'retry_wait') "
                "AND deadline_at IS NOT NULL AND deadline_at <= ? ORDER BY deadline_at, assignment_id",
                (now,),
            ).fetchall()
            for assignment in overdue:
                self._terminalize_without_envelope(assignment, "deadline_exceeded", now)
            self._recompute_groups()
        return len(leases)

    def accept_result(self, envelope: ResultEnvelope) -> AcceptanceRecord:
        envelope_bytes = canonical_json_bytes(envelope)
        envelope_digest = sha256_digest(envelope_bytes)
        duplicate = self._existing_terminal(envelope.assignment_id, envelope.lease_id, envelope_digest)
        if duplicate is not None:
            self.spool.publish_result(envelope_digest, envelope_bytes)
            return duplicate
        assignment, _ = self._validate_active_envelope(envelope)
        if envelope.requested_policy_id != assignment["policy_id"]:
            raise ConflictError("result policy ID does not match the assignment")
        if envelope.requested_policy_digest != assignment["policy_manifest_digest"]:
            raise ConflictError("result policy manifest digest does not match the assignment")
        artifact_path = self.spool.publish_result(envelope_digest, envelope_bytes)
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
            terminal = self._retry_or_terminal(assignment, now, envelope=envelope, digest=envelope_digest)
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
                envelope = ResultEnvelope.model_validate_json(path.read_bytes())
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
            raise KeyError(group_id)
        return row["state"]

    def _run(self) -> sqlite3.Row:
        row = self.connection.execute("SELECT * FROM runs WHERE singleton = 1").fetchone()
        if row is None:
            raise InvalidStateError("run has not been initialized")
        return row

    def _assignment(self, assignment_id: str) -> sqlite3.Row:
        row = self.connection.execute("SELECT * FROM assignments WHERE assignment_id = ?", (assignment_id,)).fetchone()
        if row is None:
            raise KeyError(assignment_id)
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
                envelope.lease_id if envelope else assignment["lease_id"],
                digest,
                now,
            ),
        )
        return True

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
            verify_lora_policy(artifact_path, expected=manifest)
        except (OSError, ValueError) as error:
            raise ArtifactCorruptionError(f"published policy artifact is corrupt: {row['policy_id']}") from error


CoordinatorState = CoordinatorRepository
