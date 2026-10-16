# Prospective semantic testing

The semantic module is a research interface for testing visible browser
workflows. It asks an actor to state an observable expectation before changing
the interface, then compare what happened with that expectation.

## Evidence order

Each investigation follows this sequence:

1. A public clue and its evidence reference.
2. A falsifiable hypothesis.
3. Intervention and control operations.
4. The predicted difference between their outcomes.
5. Actual recorded observations.
6. A decision to retain, revise, or reject the hypothesis.

The plan is persisted before its relevant controls execute. Later revisions do
not overwrite earlier predictions. A candidate declared through `suspected_bug`
can enter the evidence pipeline without a software detector firing first.

## Browser boundary

The browser adapter exposes ordinary visible click, input, select, and wait
controls. It provides no bug-specific route or reproduction macro. The Mars
integration covers functions, parameters, running, and save/restore through a
separately supplied compatible application. See
[runtime prerequisites](integrations.md#mars-browser-editor).

Fresh browser contexts, source/build identity, observed initial state, and
browser configuration belong to the trace. A fixed clock or random seed helps
comparison but can affect behavior; a defect judgment needs to account for the
test environment.

## Independent evidence and regression

`autoplay_qa.semantic_analysis` separates candidate review, raw trajectory
comparison, native reproduction, and regression evidence. The generic case
format stores ordinary controls and observable assertions such as `equals`,
`not_equals`, `contains`, and `exists`. Its normal-control variant checks
ordinary behavior alongside the target case.

`tools/run_semantic_regression.py` executes a compatible case against a prepared
local browser build without inference. It needs a real case definition and its
matching runtime inputs. The export machinery is an implementation capability;
it is not evidence that the actor has independently discovered a new defect.

## Research interfaces

The source includes core interfaces for bounded requests, provenance, analysis,
and local regression execution. Experimental analysis and budget modules retain
specific research-protocol constants for compatibility. They do not make those
protocols generic or convey an active authorization. The public release omits
private study orchestration, ledgers, and operator archives.

Use the core interfaces and fake-response tests to understand the protocol.
Any future study needs its own explicit question, source/build preparation,
baseline, resource limits, data permission, and independent review. The current
research interface does not establish autonomous unknown-defect discovery or
an advantage over scripted coverage.
