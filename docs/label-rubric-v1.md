# InferenceBench Label Rubric v1

Status: accepted for planning

This rubric converts each eligible `doctl` issue into exactly one label from the customer taxonomy required by the assignment:

```text
bug | enhancement | question | documentation | security | other
```

It applies to ground-truth annotation and to the shared model prompt. Changing these rules requires a new rubric version and invalidates direct comparison with results scored under an older version.

## Decision classification

- `REQUIRED` — Assign exactly one of the six customer labels to every inference request.
- `REQUIRED` — Use `other` for issues that genuinely do not fit the first five, including spam, duplicates, off-topic content, and ambiguity.
- `DESIGN-CHOICE` — Resolve overlapping categories with the precedence order below.
- `REJECTED` — Preserve multiple labels because the repository historically used them.
- `REJECTED` — Let each annotator or model invent its own interpretation of the label boundaries.

## Precedence

Apply the first matching rule:

1. **Other** — The issue is a duplicate, spam, off-topic, genuinely ambiguous, or otherwise does not fit the first five categories.
2. **Security** — The primary concern is a vulnerability, credential exposure, unauthorized access, unsafe permission boundary, or other security impact.
3. **Documentation** — The required correction is limited to documentation, examples, help text, or explanatory material; product behavior does not need to change.
4. **Bug** — Existing product behavior is broken or contradicts a documented or reasonably established expectation.
5. **Enhancement** — The issue asks for a new capability or an intentional change to behavior that is not already promised.
6. **Question** — The issue primarily asks for explanation or usage support and does not request a product or documentation change.

`Other` has first precedence only when the issue genuinely falls outside the substantive five classes. It must not become a shortcut for a difficult but classifiable issue.

## Boundary examples

| Scenario | Label | Reason |
|---|---|---|
| A defect exposes credentials in logs | `security` | Security impact takes precedence over the fact that the behavior is also a bug. |
| A command works, but its documented flag is wrong | `documentation` | Only documentation needs correction. |
| A documented command crashes with valid input | `bug` | Existing promised behavior is broken. |
| A user asks for a new output format | `enhancement` | The requested behavior does not already exist. |
| A user asks how authentication works | `question` | The request is for information rather than a change. |
| An issue is explicitly a duplicate or cannot be assigned a substantive class | `other` | The assignment explicitly places these cases in `other`. |

## Annotation rule

The first annotation pass uses exactly the same issue information provided to every model:

```text
issue title
issue body
```

Model predictions, maintainer labels, issue state, closure metadata, comments, and linked work are hidden during this pass.

After committing the first-pass label and confidence, the annotator may inspect maintainer labels or closure context as corroborating evidence. This evidence is recorded as provenance and may trigger review, but it must not silently override the rubric.

If post-submission context changes the answer because the title and body did not contain enough information, record `input_sufficiency = insufficient`. Such an issue cannot become an accepted item in the Random Human-Reviewed Sample. It may be retained in diagnostic analysis or remain in the Unscored Corpus rather than becoming an unfair model error.

A genuinely ambiguous issue can correctly receive `other`. An annotation that remains unresolved because the available evidence or rubric is inadequate is different: it stays outside the accepted Evaluation Corpus until reviewed or the rubric is clarified.

## Review fields

Each Ground Truth Label records:

```text
issue_id
label
rubric_version: v1
ground_truth_source
sampling_stratum
confidence
review_status
review_notes
input_sufficiency
maintainer_labels_at_snapshot
```
