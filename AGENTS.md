# AutoPlay QA agent guidance

Build a developer tool that connects model-selected tests to real runtime
feedback and reproducible evidence. Self-evolving testing agents are a research
vision; describe implemented mechanisms and measured capabilities separately.

## Implementation boundaries

- Preserve `GameAdapter.observe()` / `act()`, default protocols, and existing
  recording contracts. Use opt-in versions for incompatible research changes.
- Give the actor public feedback and exposed controls, not hidden solutions,
  developer rules, evaluation labels, or preselected bug routes.
- Treat model notes and verdicts as claims. Keep observed transitions, software
  facts, native reproduction, and independent defect judgments separate.
- Keep the origin of scripted, stub, model-selected, and human-generated
  evidence explicit. Never present a fake-model check as model performance.
- Ordinary session JSONL, benchmark outputs, semantic traces, and sanitized web
  JSON have different entry points. Follow `docs/evidence-formats.md`.

## Authorized execution

- Use fake or stub HTTP for implementation and tests. Do not call an inference
  provider without the current operator's explicit bounded authorization for
  execution, cost, and data transmission.
- Never reuse another operator's credentials, allowances, study identifiers,
  ledgers, deployment identity, or historical execution permission.
- Preserve local original evidence and unknown-usage reservations. Do not retry
  unknown mutations or provider requests merely to obtain a better result.
- Do not publish, deploy, or upload private artifacts without authorization.
  Keep provider keys, model thinking, private source, and personal paths out of
  public files and outputs.

## Verification

Run focused tests for the change and the relevant broader suite. Python tests
are under `tests/`; web checks are described in `web/README.md`. Use the proper
format-specific entry point for replay and reproduction. Report unavailable
runtime dependencies rather than substituting synthetic output for a native
check. Do not install or launch Unity/Mars merely to edit documentation.

Keep changes scoped and reviewable. Preserve third-party notices. Do not change
model defaults or expose owner-only actions to public visitors as incidental
parts of another task.
