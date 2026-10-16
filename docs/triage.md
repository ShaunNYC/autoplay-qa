# Candidate review with Nemotron Ultra

`triage` prepares evidence from a supported session recording and asks Ultra for
a tentative `bug`, `not_bug`, or `inconclusive` assessment. Bug assessments add
severity, player impact, cited steps, and limitations. A model assessment does
not replace native reproduction or developer judgment.

## Prepare evidence offline

```sh
python3 -m autoplay_qa triage runs/my-session.jsonl --dry-run
```

Dry-run writes evidence packets and pending reports without loading a provider
credential or sending inference. If a recording contains no reportable
candidates, the command skips inference. That describes the recording, not the
absence of defects in the whole game.

Use supported original session recordings. Sanitized viewer JSON under
`web/public/demo/` is not an input to this CLI. See
[evidence formats](evidence-formats.md).

## Optional live review

After configuring your own account and authorizing the evidence transmission:

```sh
python3 -m autoplay_qa triage runs/my-session.jsonl \
  --max-requests 4 --max-seconds 180 --timeout 30 \
  --max-tokens 8192 --max-usd 0.50
```

This is a real provider call when candidates are present. The default reviewer
is `nvidia/Nemotron-3-Ultra-550b-a55b`; the default exploration model is unchanged.
Each candidate has at most one review request. Review does not automatically
retry failed inference. Usage and configured-price reservations are recorded.
The cost cap is an estimate, not a provider balance or guaranteed bill.

The command writes Markdown, structured JSON, and an audit JSONL. Unreviewed
candidates remain pending. `--out`, `--json-out`, and `--log` select destinations;
use fresh paths for new work.

## Evidence and reproduction

The reviewer receives public controls, before/after observations, feedback,
candidate facts, and any explicitly supplied verification result. Hidden
evaluation labels and developer rules stay outside the request. Recorded model
notes remain claims.

`--verify-repro` re-executes each issue's original prefix once before review. It
requires the compatible native runtime and does not minimize the sequence.
Provide `--game` or `--unity-player` when applicable. A recurrence receipt
establishes observed behavior within that environment; it does not establish
the intended design by itself.
