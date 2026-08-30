ALTER TABLE attempt_evidence RENAME TO attempt_evidence_v1;

CREATE TABLE attempt_evidence (
    attempt_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL,
    issue_number INTEGER NOT NULL CHECK (issue_number > 0),
    attempt_purpose TEXT NOT NULL CHECK (attempt_purpose IN ('benchmark', 'preflight')),
    dispatch_order INTEGER NOT NULL CHECK (dispatch_order >= 0),
    provider_outcome TEXT CHECK (
        provider_outcome IS NULL OR provider_outcome IN (
            'success', 'rate_limit', 'timeout', 'server_error', 'network_error',
            'authentication', 'invalid_request', 'protocol_error', 'unknown'
        )
    ),
    scored_outcome TEXT CHECK (
        scored_outcome IS NULL OR scored_outcome IN (
            'correct', 'incorrect_label', 'invalid_output', 'request_error'
        )
    ),
    parsed_label TEXT CHECK (
        parsed_label IS NULL OR parsed_label IN (
            'bug', 'enhancement', 'question', 'documentation', 'security', 'other'
        )
    ),
    usable INTEGER NOT NULL CHECK (usable IN (0, 1)),
    evidence_json TEXT NOT NULL,
    FOREIGN KEY (run_id) REFERENCES run_manifests(run_id),
    UNIQUE (run_id, issue_number, attempt_purpose)
);

INSERT INTO attempt_evidence (
    attempt_id,
    run_id,
    issue_number,
    attempt_purpose,
    dispatch_order,
    provider_outcome,
    scored_outcome,
    parsed_label,
    usable,
    evidence_json
)
SELECT
    attempt_id,
    run_id,
    issue_number,
    attempt_purpose,
    dispatch_order,
    NULL,
    scored_outcome,
    parsed_label,
    usable,
    evidence_json
FROM attempt_evidence_v1;

DROP TABLE attempt_evidence_v1;

CREATE INDEX attempt_evidence_run_order_idx
ON attempt_evidence (run_id, dispatch_order);
