CREATE TABLE run_manifests (
    run_id TEXT PRIMARY KEY,
    model_id TEXT NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('running', 'complete', 'incomplete')),
    corpus_version TEXT NOT NULL,
    corpus_sha256 TEXT NOT NULL,
    prompt_version TEXT NOT NULL,
    parser_version TEXT NOT NULL,
    generation_configuration_sha256 TEXT NOT NULL,
    expected_count INTEGER NOT NULL CHECK (expected_count > 0),
    manifest_json TEXT NOT NULL
);

CREATE TABLE attempt_evidence (
    attempt_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL,
    issue_number INTEGER NOT NULL CHECK (issue_number > 0),
    attempt_purpose TEXT NOT NULL CHECK (attempt_purpose IN ('benchmark', 'preflight')),
    dispatch_order INTEGER NOT NULL CHECK (dispatch_order >= 0),
    scored_outcome TEXT NOT NULL CHECK (
        scored_outcome IN ('correct', 'incorrect_label', 'invalid_output', 'request_error')
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

CREATE INDEX attempt_evidence_run_order_idx
ON attempt_evidence (run_id, dispatch_order);

