# Observed experience

`autoplay_qa.experience.ExperienceStore` retains actual acknowledged public
transitions for adaptive experiments. It does not train model weights, choose
actions, or declare bugs.

## Facts and citations

A fact includes the observed before/after state, executed native control,
visible feedback, and original run/step citation. Repeated outcomes can remain
distinct. An unchanged state is an observation, not automatically an error.
Detailed recordings remain the source for full execution evidence.

Experience is scoped by project, build/game identity, initial scene, control
catalog, coordinate frame, and state identity rules. A similar coordinate in
another scene is not automatically the same condition. Build changes require
explicit versioning and revalidation of any reused evidence.

## Storage and evaluation

The store uses SQLite and bounded retrieval. Snapshots use the SQLite backup
mechanism. Read-only snapshots support evaluation without adding the evaluation
session's outcomes to later sessions' experience.

The prior-run experiment index excludes the current run and records observed
attempts and outcome counts. Omitted or unavailable counts are explicit; they
must not be interpreted as untested controls or zero outcomes.

Optional investigation snapshots retain tentative questions, condition choices,
revisions, and counterexample citations. They remain separate from transition
facts. Reviewer feedback has its own history and is not automatically treated
as model training or ground truth.

## Useful distinctions

Storing a fact, retrieving it, receiving a model acknowledgment, and changing a
later decision are different events. A benefit claim needs an appropriate
comparison using actual testing outcomes. The store makes those comparisons
possible; it does not answer them by itself.

Keep experience databases local unless their contents have been reviewed for
sharing. Public observations can still contain project-specific information.
Cross-user sharing, automatic fact migration, cross-game learned mechanics,
and model-weight training are future work.
