# InferenceBench

[Open the running application ](https://explicitly-constant-ind-sonic.trycloudflare.com)

InferenceBench is a reproducible evaluation harness for choosing a DigitalOcean Serverless Inference model for high-volume GitHub issue classification. It uses `digitalocean/doctl` as a representative repository-level workload and keeps the evidence needed to explain the resulting recommendation.

## Recommendation

The recommended comparison pair is:

- **`mistral-3-14B`** as the quality-and-latency choice.
- **`deepseek-v4-flash-0731`** as the lower-cost alternative.

They are meaningfully different operational choices, rather than close variants. Both independently classified the complete 536-issue Corpus with the same frozen contract, timeout, and concurrency 4. The Primary Scored Holdout was 80 issues; the remaining 416 unscored issues are used for agreement and disagreement analysis.

| Complete-run evidence at concurrency 4 | `mistral-3-14B` | `deepseek-v4-flash-0731` |
|---|---:|---:|
| Primary Holdout accuracy | 91.25% (73/80) | 87.50% (70/80) |
| Primary Holdout macro-F1 | 0.767 | 0.748 |
| Cost per correct classification | $0.0001877 | $0.0000516 |
| Total 536-issue run cost | $0.0901 | $0.0261 |
| p95 request latency | 1.47 s | 1.55 s |
| Sustained throughput | 4.48 req/s | 3.80 req/s |
| Wall-clock time | 119.6 s | 140.9 s |
| Invalid outputs / request errors | 0 / 0 | 1 / 0 |

In short: choose Mistral when the measured quality, latency, and output-adherence advantage justify roughly 3.6x the per-correct cost. Choose DeepSeek when cost is the stronger constraint and its lower Primary Holdout quality, slower observed tail latency, and one invalid output are acceptable. The models agree on 366 of 416 unscored issues (88.0% strict agreement); the remaining cases are a useful review queue, not a measure of correctness.

The broader candidate screen evaluated all 25 active candidates on the same deterministic, class-stratified 40-issue subset. It identified this meaningful quality/cost pair; the complete-Corpus comparison above is the evidence used for the repository-level recommendation.

## Customer question

Which DigitalOcean-hosted open-weight model should classify issues into this customer taxonomy?

`bug`, `enhancement`, `question`, `documentation`, `security`, `other`

Every issue receives exactly one label. `other` is reserved for items that genuinely do not fit the first five categories, including duplicates, spam, off-topic reports, and irreducibly ambiguous cases.

## What was evaluated

The frozen `doctl-2026-08-30` Corpus contains 536 GitHub issues from `digitalocean/doctl`, drawn from both open and closed issues. Pull requests are excluded. The Corpus Snapshot Manifest records the GitHub query, pagination, retrieval interval, ordering rule, counts, and SHA-256 content hash so the same source population can be reused.

The Evaluation Corpus contains 120 accepted Ground Truth Labels; the remaining 416 issues form the Unscored Corpus. Ground truth is recorded with its provenance, rubric version, confidence, and review status. Historical maintainer labels are useful evidence, but are not automatically treated as truth: closed-issue mappings are accepted only after the documented human review and audit rules are met.

The broader candidate screen evaluated the 25 callable DigitalOcean-hosted generative models from the frozen discovery snapshot under one Shared Inference Contract. One discovered model, `arcee-trinity-large-thinking`, is retained in catalog evidence but excluded from execution because the provider reported that it was unavailable for the subscription tier used for the evaluation.

## Evaluation method

The evaluation controls the inputs that should be comparable across models:

- Every model sees the same canonical issue title and body, the same six-label definitions, and the same output contract.
- Each issue is sent as its own inference request. Issues are never batched into one prompt.
- Generation settings, parser, timeout, and concurrency are persisted with each run.
- Automatic retries are disabled for benchmark attempts. Invalid output and request errors remain visible evaluation outcomes rather than disappearing from quality metrics.
- The candidate screen uses the same deterministic, class-stratified 40-issue subset for every active candidate. A fixed set of 10,000 shared bootstrap resamples is applied to all candidates for paired uncertainty analysis; it does not make additional provider calls.
- The final comparison runs each selected model independently over the entire Corpus at the displayed, configurable concurrency. This avoids mixing the two models in one hidden shared request pool.

Candidate elimination is conservative. A candidate is screened out only when one same comparator is no worse across the required quality, reliability, output-adherence, cost, and per-class safeguards and has supporting paired-bootstrap evidence. Incomplete token or price evidence makes a candidate not comparable on cost rather than artificially cheap.

## What the application shows

### Scored View

For Ground Truth-labelled issues, the application shows:

- accuracy and supported-class macro-F1 for each model;
- precision, recall, and F1 for every customer label with support made explicit;
- side-by-side confusion matrices, including a visible `No Valid Prediction` outcome for invalid output or request failure;
- filters and drill-downs for model disagreements, with the Ground Truth Label, issue input, parsed prediction, raw output, and attempt evidence visible.

### Unscored View

For issues without Ground Truth Labels, the application shows:

- each model's suggested label and raw output;
- strict agreement across all expected issues, plus both-valid agreement with its conditional denominator;
- each model's label distribution;
- filters for label disagreements, one-sided failures, and joint failures.

Agreement is not presented as accuracy. It is a review signal for deciding which unlabelled cases deserve attention.

### Operational View

Each run reports the operational evidence required to make a production decision:

- request-level token counts and published per-token prices used to calculate cost per call and total cost;
- cost per correct classification, when the denominator and cost evidence are defined;
- p50 and p95 usable request latency, alongside the recorded concurrency;
- queue wait, full run wall-clock time, sustained requests per second, and sustained usable classifications per second;
- failure rate over every expected request, broken down by invalid output, timeout, rate limit, and other typed request errors.

Raw Run Manifests and Attempt Evidence are persisted before summaries. Aggregate tables and charts are derived from that immutable evidence, so the headline numbers can be recomputed and inspected.

## Evidence and reproducibility

The submitted artifacts include:

- the frozen Corpus and Corpus Snapshot Manifest;
- the labelled Evaluation Corpus and Ground Truth provenance;
- the model discovery snapshot and Pricing Snapshot;
- the versioned Shared Inference Contract;
- raw SQLite Attempt Evidence and Run Manifests;
- candidate-screening, bootstrap, and model-disposition artifacts;
- the application source and tests.

Useful result locations are:

```text
artifacts/corpus/
artifacts/ground_truth/
artifacts/model_catalog/
artifacts/pricing/
artifacts/prompts/
artifacts/results/
```

To run locally:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
python -m streamlit run app.py
```

To build and run the same application in a container:

```bash
docker build -t inferencebench .
docker run --rm -p 8501:8501 inferencebench
```

The default review opens the persisted Selection Decision Record and its two completed model runs without making a provider request. A fresh or empty evidence database falls back to a credential-free fixture so the interface remains inspectable. Live inference is initiated only from **Run Comparison View**.

Set the following before starting a live comparison:

```bash
export DO_INFERENCE_API_KEY='your DigitalOcean Serverless Inference API key'
```

The key is read from the environment and is not written to run configuration, SQLite evidence, exported artifacts, URLs, or rendered error messages. Runtime paths and the initial concurrency can be configured with `INFERENCEBENCH_DB_PATH`, `INFERENCEBENCH_CORPUS_ROOT`, `INFERENCEBENCH_GROUND_TRUTH_ROOT`, `INFERENCEBENCH_EVALUATION_VERSION`, `INFERENCEBENCH_MODEL_CATALOG_PATH`, `INFERENCEBENCH_PRICING_PATH`, `INFERENCEBENCH_CONTRACT_PATH`, `INFERENCEBENCH_DEFAULT_CONCURRENCY`, and `INFERENCEBENCH_SHARED_TIMEOUT_SECONDS`.

Run the checks with:

```bash
python -m pytest
```

## Production path

`doctl` is evidence for one representative workload, not proof that either model is best for every repository or product line. The proposed rollout path is:

1. Validate the selected model against labelled samples from additional repositories and check the per-class risks that matter to the customer.
2. Shadow the current classifier on real traffic without changing routing; review disagreements, failures, latency, and cost.
3. Roll out gradually with customer-defined quality, reliability, latency, and cost gates.
4. Retain a fallback or human-review route for low-confidence, high-risk, disagreement-prone, or policy-sensitive cases.
5. Monitor label distribution shifts, sampled correctness, typed failures, cost, and latency; revisit the model choice when those signals change.

InferenceBench does not implement that production system. It produces the traceable evaluation evidence needed to choose a candidate and define those next validation gates.
