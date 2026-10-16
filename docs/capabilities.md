# Capability status

AutoPlay QA combines usable testing infrastructure with experimental agent
mechanisms. This page distinguishes what the software provides from outcomes
that still need research.

| Area | Available implementation | Scope |
| --- | --- | --- |
| Reference testing | Grid gameplay, scripted fixtures, reports, reproduction tools | Controlled examples for exercising the pipeline |
| Model exploration | Nemotron action selection through a public observation/control contract | A bounded sequence of decisions; no guarantee of coverage or discovery |
| Candidate review | Ultra verdicts, severity, player impact, citations, and limitations | Tentative judgments for developer inspection |
| Runtime connection | Local protocol, Unity package, native Mars and browser adapters | Requires the appropriate runtime/build; Mars source is separately supplied |
| Evidence | Recordings, format-specific replay, native reproduction tools | Verifies recorded behavior within the tested environment |
| Public viewer | Saved observation playback and sanitized evidence downloads | Drawn state, not native gameplay video; viewer JSON is not session JSONL |
| Bounded operation | Action/request/time limits and supported cost-ledger workflows | Cost caps are estimates; direct `explore` has no dollar-cap flag |
| Observed experience | Scoped transition storage, retrieval, and frozen snapshots | External memory; no model-weight training |
| Investigation | Agendas, predictions, tentative questions, counterexamples, and prospective browser plans | Experimental research interfaces |
| Regression export | Generic case format and a local execution tool | Requires appropriately reviewed evidence and a compatible runtime |

The codebase supports studying better experiment selection and experience use.
It does not establish reliable learning gains, autonomous discovery of unknown
defects, or superiority over conventional automation. Self-evolution and
learned-mechanics transfer across games are future community goals.

Scripted coverage, model exploration, human findings, and synthetic tests serve
different purposes. Their source labels should remain visible in any report or
demonstration. A model suspicion, observed recurrence, and independently
confirmed defect are separate claims.
