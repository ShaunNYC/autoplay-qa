# Help game QA learn from experience

We want game testing to become **more accurate with every test**: better at
choosing experiments, recognizing real defects, and explaining what developers
can reproduce. A community collection of carefully documented interactions can
support that journey—and, over time, open models specialized for game testing.

Useful data is more than a bug screenshot. It connects a decision to an observed
outcome, explains why that outcome matters, and lets someone else check it.
Normal behavior, disproved hypotheses, and corrected AI judgments are valuable
examples too.

## Start with one useful case

Open a **Test data contribution** issue with a short description and the game's
sharing constraints. Propose a small, distinct batch before uploading large
archives. Maintainers can agree on a reviewed pull request or an external
artifact whose access and redistribution permissions are clear. An issue is a
proposal; it does not automatically enter a dataset or a training collection.

Keep native recordings in their original format. Add a small metadata file and
human-readable notes around them; see [evidence formats](evidence-formats.md)
for compatible readers and replay paths. Do not convert a dashboard screenshot
into a claimed native trace.

## What makes a strong example

- **Reproducible context:** game and build identifier, adapter/protocol version,
  platform, initial state or permitted save fixture, seed when applicable, and
  exact setup and replay instructions. Describe unavailable prerequisites.
- **Ordered evidence:** observations before and after actions, executed controls,
  execution outcomes, timestamps or step identifiers, and references from every
  claim to its evidence. Label the actor as human, script, real model, or stub;
  identify the model/configuration when relevant.
- **A reason to expect something different:** cite visible help, documented
  behavior, a game developer's specification, or a clearly described comparison.
  Keep an uncertain interpretation labeled uncertain.
- **Honest experiment history:** preserve hypotheses and predictions recorded
  before their actions. Label explanations written after an outcome as
  retrospective. Record the intervention, normal control, expected difference,
  actual result, and whether the hypothesis was retained, revised, or rejected.
- **Verification:** provide independent fresh-instance reproductions, including
  attempts that did not reproduce. When available, include the same case on a
  repaired build and a normal control that still passes.

You can contribute an incomplete or unexplained observation. Mark missing
fields as unknown and explain what would resolve them. Preserve original model
claims alongside later corrections, including false alarms. Do not quietly
replace a wrong judgment with a corrected one.

## A small contribution package

This is a proposed contribution convention, not a validator schema:

```text
case-001/
  README.md          # setup, expected behavior, reproduction steps
  metadata.json      # provenance, permissions, review state
  trace.jsonl        # original recording format, sanitized for sharing
  checks/            # optional replay results, controls, fixed-build results
```

```json
{
  "case_id": "case-001",
  "game": "your-game",
  "build": "commit-or-build-id",
  "adapter": "adapter-name@version",
  "initial_state": "README.md#setup",
  "seed": null,
  "actor": {"kind": "human", "model": null},
  "recording_format": "describe-your-native-format",
  "hypothesis_timing": "retrospective",
  "evidence_stage": "pending",
  "verdict": "uncertain",
  "reviewer": null,
  "bug_family": "unassigned",
  "artifact_sha256": {"trace.jsonl": "REPLACE_WITH_DIGEST"},
  "data_license": "proposed-license-or-permission-reference",
  "training_opt_in": false
}
```

Use `null` or an explicit explanation for unknown values. Add observed actions,
requests, cost, and elapsed time where available; do not invent missing usage.
Data files may have different permissions from repository code.

## Review makes the collection useful

| Evidence stage | Who assigns it | Meaning |
| --- | --- | --- |
| Pending | Contributor on submission | Provenance and claims await review; uncertainty is welcome. |
| Reproduced | Independent reviewer | A named reviewer repeated the recorded outcome in a fresh instance and linked the result. |
| Reviewed | Maintainer after evidence review | Provenance, expected behavior, reproduction and verdict have been checked; review and disagreements remain attached. |

A repeated outcome is not automatically a defect. Track the verdict separately:
supported defect, expected behavior, or uncertain. A normal control can be
reviewed too. Neither the acting model nor the original reporter independently
confirms its own claim. Add corrections and new review revisions without
erasing earlier evidence.

Prefer diverse, small batches over duplicate volume. Group near-duplicates and
related cases by game, build lineage, and defect family. Keep related cases
together when creating train/development/test splits, and reserve held-out
games or builds for transfer evaluation. Evaluation-only answers and defect
labels belong outside the testing agent's inputs.

## Share deliberately; opt in to training separately

Share material you have permission to distribute. Identify ownership and
licenses for recordings, screenshots, saves, and third-party assets; flag
unclear permissions for review. Remove credentials, personal information,
private chats, local paths, and confidential game content. Include a redaction
note so reviewers understand any evidence gaps. Never attach a secret to prove
that it was removed.

**Contributing evidence does not automatically enroll it in project training.**
Default `training_opt_in` to `false`. A future training collection should require
a separate, explicit opt-in describing the intended reuse, applicable data
license, and attribution. Maintainers must check permissions before inclusion;
an opt-in checkbox alone does not settle third-party rights. Keep data release
versions, origins, and inclusion decisions traceable.

## What we want to measure together

More data should earn its place through better results: supported defect
precision, discovery coverage on independent cases, fresh-instance reproduction,
false alarms on normal behavior, fewer redundant actions, and usable regression
cases. Compare systems under the same action, time, and request budgets. Measure
improvement on held-out evidence, not on examples a model has already seen.

That is the shared goal of self-evolving game QA: community experience helps
agents test more intelligently, and every improvement remains checkable.
