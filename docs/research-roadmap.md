# Community research roadmap

Our vision is a self-evolving testing partner that learns from interaction,
designs useful experiments, and produces regression evidence developers can
trust. AutoPlay QA contributes a shared runtime and evidence foundation for
working toward that vision. We want more testing to lead to better judgment:
more confirmed findings, fewer false alarms, less uninformative repetition,
and stronger regression coverage.

## Directions explored in the codebase

| Direction | Existing foundation | Community question |
| --- | --- | --- |
| Goal-directed play and lean exploration | Public observation/action loops and tentative notes | Which decisions are most useful under a small action budget? |
| Hybrid automation | Scripted coverage followed by model-selected controls | Where can adaptive testing add evidence beyond known checks? |
| Runtime integration | Grid, local protocol, Unity, and Mars adapters | How can another game expose dependable public feedback and controls? |
| Evidence and review | Recordings, native reproduction tools, reports, and Ultra assessments | How can developers inspect uncertainty and supporting evidence efficiently? |
| Bounded context | Current state, recent facts, selected history, and omission receipts | What information improves a specific next decision? |
| Observed experience | Scoped transition storage and frozen snapshots | When is prior evidence applicable and useful? |
| Experiment agendas | Observed attempts, unchanged results, and prediction checks | How can an agent avoid uninformative repetition? |
| Sustained investigation | Tentative questions, conditions, revisions, and counterexamples | How can an agent maintain and revise a useful inquiry? |
| Focused fact cards | Compact current-state and other-state facts | How do relevance and presentation affect actual decisions? |
| Controlled comparisons | Cold/frozen-history conditions and different experience amounts | What changes can be attributed to experience rather than configuration? |
| Browser semantic testing | Visible controls and predictions recorded before intervention | Can an agent design a discriminating experiment from public clues? |
| Regression artifacts | Case export, assertions, normal controls, and a local runner | How can independently confirmed findings become useful checks? |

These are implementation and research surfaces. The benefit of a strategy is a
separate question from whether its storage, control, or recording mechanism
works.

## Contributions with practical value

Add adapters and clean example applications. Improve evidence navigation and
format-specific reproduction. Develop agents that state clear expectations,
choose useful controls, and respond to contradictory observations. Contribute
independently understood evaluation cases with private labels separated from
public model inputs.

Compare methods under explicit conditions and record actual controls, requests,
usage, elapsed time, and supported findings. A framework that preserves the
evidence makes advances in models and methods easier for others to evaluate.

## Community data and game-specialized models

We want developers and players to co-create an open collection of testing
experience: mechanics exercised, experiments attempted, normal behavior,
confirmed problems, disproved suspicions, and fixes checked. The
[contribution guide](contributing-data.md) starts with small reproducible
bundles and a review trail so others can understand each example's context.

The path we propose is:

1. Collect permissioned cases with visible feedback and reproducible steps.
2. Review outcomes, link corrections, and group duplicate bug families.
3. Publish versioned evaluation collections with held-out games and builds.
4. Curate a separately permissioned training collection for open models
   specialized in game interaction, experiment design, and bug assessment.
5. Compare new models on held-out cases, measuring confirmed findings, false
   alarms, reproducibility, and testing cost before proposing the next revision.

Contributors can help by providing cases, independently checking a replay,
adding adapters, improving dataset tooling, or studying model training methods.
The aim is a shared improvement loop in which developer feedback makes future
testing more useful across the game community.

## Longer-term goals

Cross-build and cross-game learned-mechanics transfer, model-weight adaptation,
community experience sharing, and automatic policy improvement remain future
directions. They need applicability checks, data permissions, and independent
evaluation. A shared adapter interface does not itself transfer learned rules.

This roadmap invites contributions; it does not schedule a paid campaign or
authorize use of anyone else's account or game data.
