from __future__ import annotations

import argparse
import asyncio
import json
import os
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

from inferencebench.config import Settings
from inferencebench.evaluation.bootstrap import (
    build_shared_bootstrap_manifest,
    generate_shared_resamples,
    write_shared_bootstrap_plan,
)
from inferencebench.workflows.corpus import (
    load_active_corpus_summary,
    refresh_doctl_corpus,
)
from inferencebench.workflows.candidates import load_candidate_catalog
from inferencebench.workflows.candidate_baseline import (
    CandidateBootstrapAnalysis,
    CandidateEvidenceTable,
    CandidateBaselineProgress,
    baseline_bootstrap_rows,
    build_candidate_bootstrap_analysis,
    build_candidate_evidence_table,
    create_candidate_baseline_authorization,
    execute_candidate_baseline,
    prepare_candidate_baseline,
    write_json_artifact,
)
from inferencebench.workflows.candidate_dispositions import (
    derive_candidate_screening_dispositions,
)
from inferencebench.workflows.saved_comparison import load_saved_comparison
from inferencebench.workflows.human_review import (
    load_completed_human_review,
    prepare_active_review_queue,
)
from inferencebench.artifacts import CorpusArtifacts
from inferencebench.ground_truth.annotations import HumanReviewArtifacts, model_visible_input
from inferencebench.ground_truth.strata import (
    build_mapping_audit_report,
    load_mapping_manifest,
)
from inferencebench.ground_truth.evaluation import (
    build_evaluation_corpus,
    load_diagnostic_decisions,
    write_evaluation_corpus,
)
from inferencebench.ground_truth.drafts import (
    HumanReviewDraftError,
    append_entry,
    create_draft,
    finalize_draft,
    load_draft,
    new_entry,
    next_issue,
    record_second_pass,
    save_draft,
    write_completed_artifact,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="InferenceBench evidence utilities")
    subcommands = parser.add_subparsers(dest="command", required=True)
    subcommands.add_parser(
        "init-db",
        help="initialize SQLite and seed the immutable Saved Comparison fixture",
    )
    subcommands.add_parser(
        "show-comparison",
        help="print the rebuilt fixture comparison as JSON",
    )
    snapshot_parser = subcommands.add_parser(
        "snapshot-corpus",
        help="fetch and freeze a new immutable digitalocean/doctl Corpus",
    )
    snapshot_parser.add_argument(
        "--version",
        required=True,
        help="new immutable version name, for example doctl-2026-08-30",
    )
    subcommands.add_parser(
        "show-corpus",
        help="print the active frozen Corpus identity without calling GitHub",
    )
    subcommands.add_parser(
        "show-candidates",
        help="print the frozen Eligible Candidate Pool and Pricing Snapshot",
    )
    review_queue_parser = subcommands.add_parser(
        "review-queue",
        help="print title/body-only rows for prediction-blind human review",
    )
    review_queue_parser.add_argument("--seed", type=int, required=True)
    review_queue_parser.add_argument("--limit", type=int, default=100)
    human_review_parser = subcommands.add_parser(
        "show-human-review",
        help="validate and summarize a completed Human-Reviewed Ground Truth version",
    )
    human_review_parser.add_argument("--version", required=True)
    mapping_audit_parser = subcommands.add_parser(
        "show-mapping-audit",
        help="validate frozen closed-label rules and report five-for-five audit support",
    )
    mapping_audit_parser.add_argument("--mapping", required=True)
    mapping_audit_parser.add_argument("--human-review-version", required=True)
    finalize_evaluation_parser = subcommands.add_parser(
        "finalize-evaluation-corpus",
        help="freeze accepted random and diagnostic Human-Reviewed Ground Truth",
    )
    finalize_evaluation_parser.add_argument("--version", required=True)
    finalize_evaluation_parser.add_argument("--human-review-version", required=True)
    finalize_evaluation_parser.add_argument("--diagnostic-decisions", required=True)
    initialize_draft_parser = subcommands.add_parser(
        "init-human-review-draft",
        help="create a local prediction-blind human-review draft",
    )
    initialize_draft_parser.add_argument("--draft", required=True)
    initialize_draft_parser.add_argument("--random-order-seed", type=int, required=True)
    initialize_draft_parser.add_argument("--partition-seed", type=int, required=True)
    initialize_draft_parser.add_argument("--quality-control-seed", type=int, required=True)
    next_review_parser = subcommands.add_parser(
        "next-human-review", help="show the next title/body-only issue in a draft"
    )
    next_review_parser.add_argument("--draft", required=True)
    record_review_parser = subcommands.add_parser(
        "record-human-review", help="append one reviewed outcome to a local draft"
    )
    record_review_parser.add_argument("--draft", required=True)
    record_review_parser.add_argument("--status", choices=("accepted", "unresolved", "excluded"), required=True)
    record_review_parser.add_argument("--initial-label", choices=("bug", "enhancement", "question", "documentation", "security", "other"))
    record_review_parser.add_argument("--final-label", choices=("bug", "enhancement", "question", "documentation", "security", "other"))
    record_review_parser.add_argument("--confidence", choices=("high", "medium", "low"))
    record_review_parser.add_argument("--passes", type=int, default=1)
    record_review_parser.add_argument("--second-pass", action="store_true")
    record_review_parser.add_argument("--input-sufficiency", choices=("sufficient", "insufficient"), required=True)
    record_review_parser.add_argument("--exclusion-reason")
    record_review_parser.add_argument("--notes")
    finalize_review_parser = subcommands.add_parser(
        "finalize-human-review", help="freeze a completed 100-annotation Ground Truth artifact"
    )
    finalize_review_parser.add_argument("--draft", required=True)
    finalize_review_parser.add_argument("--version", required=True)
    second_pass_parser = subcommands.add_parser(
        "record-human-review-second-pass",
        help="record a solo second-pass judgment for an accepted annotation",
    )
    second_pass_parser.add_argument("--draft", required=True)
    second_pass_parser.add_argument("--issue-number", type=int, required=True)
    second_pass_parser.add_argument("--final-label", choices=("bug", "enhancement", "question", "documentation", "security", "other"), required=True)
    second_pass_parser.add_argument("--confidence", choices=("high", "medium", "low"), required=True)
    second_pass_parser.add_argument("--notes")
    candidate_plan_parser = subcommands.add_parser(
        "show-candidate-baseline-plan",
        help="print the locally resolved paid Candidate Baseline Plan without inference",
    )
    candidate_plan_parser.add_argument(
        "--version", default="candidate-baseline-v1"
    )
    candidate_run_parser = subcommands.add_parser(
        "run-candidate-baseline",
        help=(
            "run the explicitly authorized 25-model paid screen, then calculate "
            "the Shared Bootstrap Plan locally"
        ),
    )
    candidate_run_parser.add_argument(
        "--version", default="candidate-baseline-v1"
    )
    candidate_run_parser.add_argument("--output-directory", required=True)
    candidate_run_parser.add_argument("--confirm-plan-sha256", required=True)
    candidate_run_parser.add_argument("--authorized-by", required=True)
    disposition_parser = subcommands.add_parser(
        "derive-candidate-screening-dispositions",
        help="derive local conservative dispositions from completed screening artifacts",
    )
    disposition_parser.add_argument("--input-directory", required=True)
    disposition_parser.add_argument("--output", required=True)
    return parser


def main() -> None:
    arguments = build_parser().parse_args()
    settings = Settings.from_environment()
    if arguments.command == "snapshot-corpus":
        created = refresh_doctl_corpus(
            settings,
            corpus_version=arguments.version,
            token=os.environ.get("GITHUB_TOKEN"),
        )
        manifest = created.manifest
        print(f"Frozen Corpus written to {created.directory}")
        print(
            f"{manifest.issue_count} issues retained; "
            f"{manifest.excluded_pull_request_count} pull requests excluded; "
            f"SHA-256 {manifest.artifact_sha256}"
        )
        return
    if arguments.command == "show-corpus":
        print(load_active_corpus_summary(settings).model_dump_json(indent=2))
        return
    if arguments.command == "show-candidates":
        print(load_candidate_catalog(settings).model_dump_json(indent=2))
        return
    if arguments.command == "show-candidate-baseline-plan":
        print(
            prepare_candidate_baseline(
                settings, arguments.version
            ).model_dump_json(indent=2)
        )
        return
    if arguments.command == "run-candidate-baseline":
        _run_candidate_baseline_command(settings, arguments)
        return
    if arguments.command == "derive-candidate-screening-dispositions":
        input_directory = Path(arguments.input_directory).expanduser()
        output_path = Path(arguments.output).expanduser()
        if output_path.exists():
            raise SystemExit(
                f"Disposition artifact is immutable and already exists: {output_path}"
            )
        try:
            evidence = CandidateEvidenceTable.model_validate_json(
                (input_directory / "candidate-evidence.json").read_text(
                    encoding="utf-8"
                )
            )
            bootstrap = CandidateBootstrapAnalysis.model_validate_json(
                (input_directory / "bootstrap-analysis.json").read_text(
                    encoding="utf-8"
                )
            )
        except FileNotFoundError as error:
            raise SystemExit(
                "Candidate screening is not complete yet; expected "
                "candidate-evidence.json and bootstrap-analysis.json."
            ) from error
        dispositions = derive_candidate_screening_dispositions(evidence, bootstrap)
        write_json_artifact(output_path, dispositions)
        print(
            f"Candidate Screening Dispositions written to {output_path}: "
            f"{sum(row.disposition == 'retained' for row in dispositions.dispositions)} retained, "
            f"{sum(row.disposition == 'screened_out' for row in dispositions.dispositions)} screened out, "
            f"{sum(row.disposition == 'not_comparable' for row in dispositions.dispositions)} not comparable."
        )
        return
    if arguments.command == "review-queue":
        if arguments.limit <= 0:
            raise SystemExit("--limit must be positive")
        queue = prepare_active_review_queue(settings, arguments.seed)
        rows = tuple(
            {
                "random_order_position": item.random_order_position,
                "issue_number": item.issue_number,
                "input": {"title": item.title, "body": item.body},
            }
            for item in queue[: arguments.limit]
        )
        print(json.dumps({"random_order_seed": arguments.seed, "items": rows}, indent=2))
        return
    if arguments.command == "show-human-review":
        print(load_completed_human_review(settings, arguments.version).model_dump_json(indent=2))
        return
    if arguments.command == "show-mapping-audit":
        corpus_manifest, issues = CorpusArtifacts(settings.corpus_root).load_active()
        _, reviews = HumanReviewArtifacts(
            settings.project_root / "artifacts" / "ground_truth"
        ).load_version(arguments.human_review_version, corpus_manifest, issues)
        mapping_path = settings.project_root / "artifacts" / "ground_truth" / "mappings" / arguments.mapping
        report = build_mapping_audit_report(
            corpus_manifest,
            issues,
            load_mapping_manifest(mapping_path),
            reviews,
        )
        print(report.model_dump_json(indent=2))
        return
    if arguments.command == "finalize-evaluation-corpus":
        corpus_manifest, issues = CorpusArtifacts(settings.corpus_root).load_active()
        _, random_reviews = HumanReviewArtifacts(
            settings.project_root / "artifacts" / "ground_truth"
        ).load_version(arguments.human_review_version, corpus_manifest, issues)
        decisions = load_diagnostic_decisions(
            settings.project_root / "artifacts" / "ground_truth" / "drafts" / arguments.diagnostic_decisions
        )
        manifest, annotations = build_evaluation_corpus(
            corpus_manifest=corpus_manifest,
            issues=issues,
            random_reviews=random_reviews,
            diagnostics=decisions,
            ground_truth_version=arguments.version,
        )
        directory = write_evaluation_corpus(
            settings.project_root / "artifacts" / "ground_truth", manifest, annotations
        )
        print(f"Evaluation Corpus written to {directory} ({manifest.annotation_count} annotations)")
        return
    drafts_root = settings.project_root / "artifacts" / "ground_truth" / "drafts"
    draft_path = drafts_root / f"{arguments.draft}.json" if hasattr(arguments, "draft") else None
    if arguments.command == "init-human-review-draft":
        if Path(arguments.draft).name != arguments.draft or draft_path is None:
            raise SystemExit("--draft must be a local filename")
        if draft_path.exists():
            raise SystemExit(f"Draft already exists: {draft_path}")
        corpus_manifest, _ = CorpusArtifacts(settings.corpus_root).load_active()
        draft = create_draft(
            corpus_manifest=corpus_manifest,
            rubric_path=settings.project_root / "docs" / "label-rubric-v1.md",
            random_order_seed=arguments.random_order_seed,
            partition_seed=arguments.partition_seed,
            quality_control_seed=arguments.quality_control_seed,
        )
        save_draft(draft_path, draft)
        print(f"Human review draft created at {draft_path}")
        return
    if arguments.command in {"next-human-review", "record-human-review", "record-human-review-second-pass", "finalize-human-review"}:
        assert draft_path is not None
        draft = load_draft(draft_path)
        corpus_manifest, issues = CorpusArtifacts(settings.corpus_root).load_active()
        if arguments.command == "next-human-review":
            position, issue = next_issue(draft, issues)
            print(json.dumps({"random_order_position": position, "issue_number": issue.issue_number, "input": model_visible_input(issue)}, indent=2))
            return
        if arguments.command == "record-human-review":
            _, issue = next_issue(draft, issues)
            entry = new_entry(
                issue_number=issue.issue_number,
                initial_label=arguments.initial_label,
                final_label=arguments.final_label,
                confidence=arguments.confidence,
                review_status=arguments.status,
                review_pass_count=arguments.passes,
                requires_second_pass=arguments.second_pass,
                input_sufficiency=arguments.input_sufficiency,
                exclusion_reason=arguments.exclusion_reason,
                review_notes=arguments.notes,
            )
            save_draft(draft_path, append_entry(draft, issues, entry))
            print(f"Recorded issue #{issue.issue_number}")
            return
        if arguments.command == "record-human-review-second-pass":
            save_draft(
                draft_path,
                record_second_pass(
                    draft,
                    issue_number=arguments.issue_number,
                    final_label=arguments.final_label,
                    confidence=arguments.confidence,
                    review_notes=arguments.notes,
                ),
            )
            print(f"Recorded second pass for issue #{arguments.issue_number}")
            return
        manifest, annotations = finalize_draft(
            draft,
            corpus_manifest=corpus_manifest,
            issues=issues,
            ground_truth_version=arguments.version,
        )
        directory = write_completed_artifact(
            settings.project_root / "artifacts" / "ground_truth", manifest, annotations
        )
        print(f"Completed Human-Reviewed Ground Truth written to {directory}")
        return

    comparison = load_saved_comparison(settings)
    if arguments.command == "init-db":
        print(f"Saved Comparison evidence ready at {settings.database_path}")
        print(
            f"{comparison.model_a.model_id}: {comparison.model_a.accuracy:.0%}; "
            f"{comparison.model_b.model_id}: {comparison.model_b.accuracy:.0%}"
        )
        return
    print(comparison.model_dump_json(indent=2))


def _run_candidate_baseline_command(
    settings: Settings, arguments: argparse.Namespace
) -> None:
    plan = prepare_candidate_baseline(settings, arguments.version)
    if not plan.can_authorize:
        details = "\n".join(f"- {blocker}" for blocker in plan.blockers)
        raise SystemExit(
            "Candidate Baseline is blocked; no provider requests were made.\n"
            f"Plan SHA-256: {plan.content_sha256}\n{details}"
        )
    api_key = os.environ.get("DO_INFERENCE_API_KEY")
    if not api_key:
        raise SystemExit(
            "DO_INFERENCE_API_KEY is required; no provider requests were made."
        )
    authorization = create_candidate_baseline_authorization(
        plan,
        confirmed_plan_sha256=arguments.confirm_plan_sha256,
        authorized_by=arguments.authorized_by,
        authorized_at=datetime.now(UTC),
    )
    output_directory = Path(arguments.output_directory).expanduser()
    if output_directory.exists():
        raise SystemExit(
            f"Output directory already exists; baseline artifacts are immutable: "
            f"{output_directory}"
        )
    output_directory.mkdir(parents=True)
    write_json_artifact(output_directory / "baseline-plan.json", plan)
    write_json_artifact(output_directory / "authorization.json", authorization)

    run_settings = replace(
        settings, database_path=output_directory / "attempt-evidence.sqlite3"
    )

    def report(progress: CandidateBaselineProgress) -> None:
        print(
            f"[{progress.candidate_position}/25] {progress.model_id}: "
            f"{progress.persisted_count}/{progress.expected_count} "
            f"({progress.status.value})",
            flush=True,
        )

    execution = asyncio.run(
        execute_candidate_baseline(
            run_settings,
            plan,
            authorization,
            api_key=api_key,
            progress_callback=report,
        )
    )
    write_json_artifact(output_directory / "baseline-execution.json", execution)
    evidence = build_candidate_evidence_table(run_settings, plan, execution)
    write_json_artifact(output_directory / "candidate-evidence.json", evidence)

    rows = baseline_bootstrap_rows(run_settings)
    resamples = generate_shared_resamples(
        rows,
        seed=plan.shared_bootstrap_seed,
        resample_count=plan.shared_bootstrap_resample_count,
    )
    bootstrap_manifest = build_shared_bootstrap_manifest(
        plan_version=f"{plan.baseline_version}-shared-bootstrap-v1",
        baseline_plan_sha256=plan.content_sha256,
        ground_truth_version=plan.ground_truth_version,
        ground_truth_sha256=plan.ground_truth_sha256,
        rows=rows,
        resamples=resamples,
        seed=plan.shared_bootstrap_seed,
    )
    write_shared_bootstrap_plan(
        output_directory / "shared-bootstrap", bootstrap_manifest, resamples
    )
    analysis = build_candidate_bootstrap_analysis(
        run_settings,
        plan,
        execution,
        bootstrap_manifest,
        resamples,
    )
    write_json_artifact(output_directory / "bootstrap-analysis.json", analysis)
    complete = sum(row.status == "complete" for row in execution.runs)
    print(
        f"Candidate Baseline evidence written to {output_directory}. "
        f"Complete candidates: {complete}/25; bootstrap provider requests: 0."
    )


if __name__ == "__main__":
    main()
