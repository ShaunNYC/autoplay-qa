# Adaptive exploration modules

The default exploration protocol is `lean-v1`. Adaptive versions are opt-in
research interfaces that preserve the same public observation/action boundary.
The adapter supplies native controls; the model chooses the next one.

| Protocol | Main mechanism |
| --- | --- |
| `adaptive-v1` | Bounded current context with relevant observed transitions |
| `adaptive-v2` | Prior-run experiment counts, currently unattempted controls, and immediate prediction checks |
| `adaptive-v3` | Tentative investigations, selected conditions, forecasts, and counterexample citations |
| `adaptive-v4` | Bounded metadata feedback, investigation-definition tracking, and request deadline admission |

These mechanisms expose information and record decisions. Their presence does
not establish a learning or discovery improvement.

## Scope and context

Adaptive inputs include the current public observation and available controls,
bounded recent transitions, optional relevant prior facts, and tentative notes.
The default input ceiling is 24,000 UTF-8 JSON bytes including the response
schema. This is a byte limit, not a token claim. Optional material can be omitted
with explicit receipts; required current feedback is not silently discarded.

An agenda describes executed attempts. It does not supply a solution, unobserved
passability, or a route. A prediction mismatch means the forecast differed from
the result; it is not automatically a defect.

## Experience API

`ExperienceStore` holds public transition facts under a project/build scope.
Use a separate namespace for a different build, scene, or control contract.
`--memory-db`, `--project-id`, and `--memory-read-only` expose this behavior in
the adaptive CLI. Selecting an adaptive mode still performs real inference
when used through `explore`; see [getting started](getting-started.md).

The [experience guide](experience.md) describes persistence, citations, and
frozen snapshots. Read-only evaluation keeps later cases from receiving the
current case's observations as additional training history.

## Questions for evaluation

Compare cold and frozen-history conditions under equal action, request, time,
and model limits. Separate actual delivery of experience, model acknowledgment,
test selection, and independently supported findings. Preserve original claims
and missing observations. Implementation tests with fake responses establish
the plumbing, not model intelligence.

The core includes focused-fact and comparison interfaces. Experimental budget
and analysis modules retain specific research-protocol constants for
compatibility; they are not an active authorization or a generic campaign
launcher. Private study orchestration is not part of this public release.
New model use needs the contributor's own explicit scope, budget, and data
permission.
