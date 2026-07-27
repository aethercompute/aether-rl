from __future__ import annotations

SCHEMA_VERSION = 5

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
    4: (
        "ALTER TABLE rollout_groups ADD COLUMN creation_key TEXT",
        "ALTER TABLE rollout_groups ADD COLUMN sequence INTEGER",
        "UPDATE rollout_groups SET creation_key = 'manual:' || group_id",
        """
        UPDATE rollout_groups AS target
        SET sequence = (
            SELECT COUNT(*)
            FROM rollout_groups AS candidate
            WHERE candidate.created_at < target.created_at
               OR (candidate.created_at = target.created_at AND candidate.group_id <= target.group_id)
        )
        """,
        "CREATE UNIQUE INDEX rollout_groups_creation_key_idx ON rollout_groups(creation_key) WHERE creation_key IS NOT NULL",
        "CREATE UNIQUE INDEX rollout_groups_sequence_idx ON rollout_groups(sequence) WHERE sequence IS NOT NULL",
        "CREATE INDEX rollout_groups_policy_sequence_idx ON rollout_groups(kind, policy_id, sequence)",
        """
        CREATE TABLE scheduler_state (
            singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
            next_group_sequence INTEGER NOT NULL CHECK (next_group_sequence > 0),
            max_policy_lag INTEGER CHECK (max_policy_lag IS NULL OR max_policy_lag >= 0),
            loaded_policy_preference_seconds REAL
                CHECK (loaded_policy_preference_seconds IS NULL OR loaded_policy_preference_seconds >= 0)
        )
        """,
        """
        INSERT INTO scheduler_state(singleton, next_group_sequence, max_policy_lag, loaded_policy_preference_seconds)
        SELECT 1, COALESCE(MAX(sequence), 0) + 1, NULL, NULL FROM rollout_groups
        """,
        """
        CREATE TABLE scheduler_sources (
            source_id TEXT PRIMARY KEY,
            kind TEXT NOT NULL CHECK (kind IN ('train', 'eval')),
            environment_id TEXT NOT NULL,
            environment_revision TEXT NOT NULL,
            weight REAL NOT NULL CHECK (weight > 0),
            virtual_finish REAL NOT NULL DEFAULT 0,
            cursor INTEGER NOT NULL DEFAULT 0 CHECK (cursor >= 0),
            tasks_json BLOB NOT NULL,
            sampling_json BLOB NOT NULL,
            group_size INTEGER NOT NULL CHECK (group_size > 0),
            max_attempts INTEGER NOT NULL CHECK (max_attempts > 0),
            result_size_limit_bytes INTEGER NOT NULL CHECK (result_size_limit_bytes > 0),
            assignment_timeout_seconds REAL,
            enabled INTEGER NOT NULL CHECK (enabled IN (0, 1)),
            CHECK (assignment_timeout_seconds IS NULL OR assignment_timeout_seconds > 0)
        )
        """,
        "CREATE INDEX scheduler_sources_pick_idx ON scheduler_sources(kind, enabled, virtual_finish, source_id)",
        """
        CREATE TABLE assignment_cancellations (
            assignment_id TEXT PRIMARY KEY REFERENCES assignments(assignment_id),
            reason TEXT NOT NULL CHECK (reason IN ('policy_stale', 'cancelled')),
            requested_at REAL NOT NULL,
            terminal INTEGER NOT NULL CHECK (terminal IN (0, 1))
        )
        """,
        """
        CREATE TABLE lease_cancellations (
            lease_id TEXT PRIMARY KEY,
            assignment_id TEXT NOT NULL,
            reason TEXT NOT NULL CHECK (reason IN ('policy_stale', 'cancelled')),
            requested_at REAL NOT NULL,
            delivered_at REAL,
            FOREIGN KEY (assignment_id, lease_id)
                REFERENCES lease_attempts(assignment_id, lease_id)
        )
        """,
        """
        CREATE TABLE lease_requests (
            request_id TEXT PRIMARY KEY,
            worker_session_id TEXT NOT NULL REFERENCES worker_sessions(worker_session_id),
            request_digest TEXT NOT NULL,
            state TEXT NOT NULL CHECK (state IN ('pending', 'no_work', 'leased')),
            lease_id TEXT REFERENCES lease_attempts(lease_id),
            created_at REAL NOT NULL,
            completed_at REAL,
            CHECK (length(request_digest) = 71 AND substr(request_digest, 1, 7) = 'sha256:'),
            CHECK (completed_at IS NULL OR completed_at >= created_at),
            CHECK ((state = 'pending' AND lease_id IS NULL AND completed_at IS NULL) OR
                   (state = 'no_work' AND lease_id IS NULL AND completed_at IS NOT NULL) OR
                   (state = 'leased' AND lease_id IS NOT NULL AND completed_at IS NOT NULL))
        )
        """,
        "CREATE INDEX lease_requests_session_idx ON lease_requests(worker_session_id, created_at)",
        "CREATE UNIQUE INDEX lease_requests_lease_idx ON lease_requests(lease_id) WHERE lease_id IS NOT NULL",
        "CREATE INDEX assignments_schedulable_idx ON assignments(state, available_at, group_id, group_index, assignment_id)",
    ),
    5: (
        "ALTER TABLE rollout_groups ADD COLUMN source_id TEXT REFERENCES scheduler_sources(source_id)",
        """
        UPDATE rollout_groups AS g
        SET source_id = (
            SELECT s.source_id FROM scheduler_sources AS s
            WHERE substr(g.creation_key, 1, length('source:' || s.source_id || ':occurrence:')) =
                  'source:' || s.source_id || ':occurrence:'
            ORDER BY length(s.source_id) DESC
            LIMIT 1
        )
        WHERE g.creation_key LIKE 'source:%'
        """,
        "CREATE INDEX rollout_groups_processing_idx ON rollout_groups(state, sequence, group_id)",
        """
        CREATE TABLE training_batches (
            step INTEGER PRIMARY KEY CHECK (step > 0),
            artifact_digest TEXT NOT NULL UNIQUE,
            artifact_path TEXT NOT NULL UNIQUE,
            size_bytes INTEGER NOT NULL CHECK (size_bytes > 0),
            sample_count INTEGER NOT NULL CHECK (sample_count > 0),
            created_at REAL NOT NULL,
            CHECK (length(artifact_digest) = 71 AND substr(artifact_digest, 1, 7) = 'sha256:')
        )
        """,
        """
        CREATE TABLE processed_groups (
            group_id TEXT PRIMARY KEY REFERENCES rollout_groups(group_id),
            input_digest TEXT NOT NULL,
            artifact_digest TEXT NOT NULL UNIQUE,
            artifact_path TEXT NOT NULL UNIQUE,
            size_bytes INTEGER NOT NULL CHECK (size_bytes > 0),
            rollout_count INTEGER NOT NULL CHECK (rollout_count >= 0),
            processed_at REAL NOT NULL,
            CHECK (length(input_digest) = 71 AND substr(input_digest, 1, 7) = 'sha256:'),
            CHECK (length(artifact_digest) = 71 AND substr(artifact_digest, 1, 7) = 'sha256:')
        )
        """,
        """
        CREATE TABLE processed_rollouts (
            group_id TEXT NOT NULL REFERENCES processed_groups(group_id),
            ordinal INTEGER NOT NULL CHECK (ordinal >= 0),
            token_count INTEGER NOT NULL CHECK (token_count >= 0),
            batch_step INTEGER REFERENCES training_batches(step),
            batch_ordinal INTEGER,
            discarded INTEGER NOT NULL DEFAULT 0 CHECK (discarded IN (0, 1)),
            PRIMARY KEY (group_id, ordinal),
            UNIQUE (batch_step, batch_ordinal),
            CHECK ((batch_step IS NULL AND batch_ordinal IS NULL) OR
                   (batch_step IS NOT NULL AND batch_ordinal IS NOT NULL AND batch_ordinal >= 0)),
            CHECK (discarded = 0 OR batch_step IS NULL)
        )
        """,
        "CREATE INDEX processed_rollouts_pending_idx ON processed_rollouts(batch_step, group_id, ordinal)",
    ),
}
