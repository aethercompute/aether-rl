from __future__ import annotations

SCHEMA_VERSION = 3

MIGRATIONS: dict[int, tuple[str, ...]] = {
    1: (
        """
        CREATE TABLE runs (
            singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
            run_id TEXT NOT NULL UNIQUE,
            base_model_json BLOB NOT NULL,
            base_policy_id TEXT NOT NULL REFERENCES policies(policy_id),
            active_policy_id TEXT NOT NULL REFERENCES policies(policy_id),
            created_at REAL NOT NULL
        )
        """,
        """
        CREATE TABLE policies (
            policy_id TEXT PRIMARY KEY,
            policy_version INTEGER NOT NULL UNIQUE CHECK (policy_version >= 0),
            manifest_digest TEXT NOT NULL UNIQUE,
            manifest_json BLOB NOT NULL,
            artifact_path TEXT,
            created_at REAL NOT NULL,
            UNIQUE (policy_id, manifest_digest),
            CHECK ((policy_version = 0 AND artifact_path IS NULL) OR
                   (policy_version > 0 AND artifact_path IS NOT NULL))
        )
        """,
        """
        CREATE TABLE workers (
            worker_id TEXT PRIMARY KEY,
            first_registered_at REAL NOT NULL
        )
        """,
        """
        CREATE TABLE worker_sessions (
            worker_session_id TEXT PRIMARY KEY,
            worker_id TEXT NOT NULL REFERENCES workers(worker_id),
            capabilities_json BLOB NOT NULL,
            registered_at REAL NOT NULL,
            last_seen_at REAL NOT NULL,
            UNIQUE (worker_id, worker_session_id)
        )
        """,
        """
        CREATE TABLE rollout_groups (
            group_id TEXT PRIMARY KEY,
            policy_id TEXT NOT NULL REFERENCES policies(policy_id),
            kind TEXT NOT NULL CHECK (kind IN ('train', 'eval')),
            environment_id TEXT NOT NULL,
            environment_revision TEXT NOT NULL,
            task_json BLOB NOT NULL,
            sampling_json BLOB NOT NULL,
            group_size INTEGER NOT NULL CHECK (group_size > 0),
            state TEXT NOT NULL CHECK (state IN ('collecting', 'ready')),
            created_at REAL NOT NULL,
            UNIQUE (group_id, policy_id)
        )
        """,
        """
        CREATE TABLE assignments (
            assignment_id TEXT PRIMARY KEY,
            group_id TEXT NOT NULL REFERENCES rollout_groups(group_id),
            group_index INTEGER NOT NULL CHECK (group_index >= 0),
            assignment_json BLOB NOT NULL,
            policy_id TEXT NOT NULL REFERENCES policies(policy_id),
            policy_manifest_digest TEXT NOT NULL,
            state TEXT NOT NULL CHECK (state IN
                ('pending', 'retry_wait', 'leased', 'succeeded', 'failed')),
            max_attempts INTEGER NOT NULL CHECK (max_attempts > 0),
            attempt_count INTEGER NOT NULL DEFAULT 0 CHECK (attempt_count >= 0),
            available_at REAL NOT NULL,
            deadline_at REAL,
            current_lease_id TEXT,
            UNIQUE (group_id, group_index),
            FOREIGN KEY (assignment_id, current_lease_id)
                REFERENCES lease_attempts(assignment_id, lease_id),
            FOREIGN KEY (group_id, policy_id)
                REFERENCES rollout_groups(group_id, policy_id),
            FOREIGN KEY (policy_id, policy_manifest_digest)
                REFERENCES policies(policy_id, manifest_digest)
        )
        """,
        """
        CREATE TABLE lease_attempts (
            lease_id TEXT PRIMARY KEY,
            assignment_id TEXT NOT NULL REFERENCES assignments(assignment_id),
            attempt INTEGER NOT NULL CHECK (attempt > 0),
            worker_id TEXT NOT NULL REFERENCES workers(worker_id),
            worker_session_id TEXT NOT NULL REFERENCES worker_sessions(worker_session_id),
            issued_at REAL NOT NULL,
            expires_at REAL NOT NULL,
            state TEXT NOT NULL CHECK (state IN
                ('active', 'succeeded', 'failed', 'expired')),
            UNIQUE (assignment_id, attempt),
            UNIQUE (assignment_id, lease_id),
            FOREIGN KEY (worker_id, worker_session_id)
                REFERENCES worker_sessions(worker_id, worker_session_id),
            CHECK (expires_at > issued_at)
        )
        """,
        """
        CREATE TABLE accepted_results (
            assignment_id TEXT PRIMARY KEY REFERENCES assignments(assignment_id),
            lease_id TEXT NOT NULL UNIQUE REFERENCES lease_attempts(lease_id),
            envelope_digest TEXT NOT NULL UNIQUE,
            artifact_path TEXT NOT NULL UNIQUE,
            size_bytes INTEGER NOT NULL CHECK (size_bytes > 0),
            accepted_at REAL NOT NULL,
            processing_state TEXT NOT NULL CHECK
                (processing_state IN ('pending', 'processing', 'processed')),
            FOREIGN KEY (assignment_id, lease_id)
                REFERENCES lease_attempts(assignment_id, lease_id)
        )
        """,
        """
        CREATE TABLE failures (
            lease_id TEXT PRIMARY KEY REFERENCES lease_attempts(lease_id),
            assignment_id TEXT NOT NULL REFERENCES assignments(assignment_id),
            envelope_digest TEXT NOT NULL UNIQUE,
            envelope_json BLOB NOT NULL,
            accepted_at REAL NOT NULL,
            retryable INTEGER NOT NULL CHECK (retryable IN (0, 1)),
            terminal INTEGER NOT NULL CHECK (terminal IN (0, 1)),
            FOREIGN KEY (assignment_id, lease_id)
                REFERENCES lease_attempts(assignment_id, lease_id)
        )
        """,
        """
        CREATE TABLE assignment_outcomes (
            assignment_id TEXT PRIMARY KEY REFERENCES assignments(assignment_id),
            outcome TEXT NOT NULL CHECK (outcome IN ('result', 'failure', 'attempts_exhausted', 'deadline_exceeded')),
            lease_id TEXT REFERENCES lease_attempts(lease_id),
            envelope_digest TEXT,
            completed_at REAL NOT NULL,
            FOREIGN KEY (assignment_id, lease_id)
                REFERENCES lease_attempts(assignment_id, lease_id)
        )
        """,
        "CREATE INDEX assignments_due_idx ON assignments(state, available_at, deadline_at)",
        "CREATE INDEX leases_active_idx ON lease_attempts(state, expires_at, assignment_id)",
        "CREATE INDEX results_processing_idx ON accepted_results(processing_state, accepted_at)",
    ),
    2: (
        "ALTER TABLE worker_sessions ADD COLUMN last_heartbeat_sent_at REAL",
        "ALTER TABLE lease_attempts ADD COLUMN last_renew_sent_at REAL",
    ),
    3: ("ALTER TABLE worker_sessions ADD COLUMN last_lease_request_sent_at REAL",),
}
