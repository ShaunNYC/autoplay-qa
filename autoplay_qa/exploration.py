"""Session-scoped mechanics learning; the model chooses every experiment/action."""

from collections import Counter
import copy
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
from math import isfinite
from statistics import median
import time
from typing import Callable

from .game import ACTIONS, DIRECTIONS, GameAdapter, evaluate_transition
from .nebius import InvalidDecision, NebiusClient, NebiusError
from .session import SessionLog, redact_secrets


CONTROLS = (*ACTIONS, "reset", "finish")
HYPOTHESIS_STATUSES = ("untested", "supported", "contradicted", "uncertain")
NONCOMPARISON_LABELS = {"untested", "supported", "uncertain", "verified"}
PROMPT_VERSION = "measured-scope-v27"
EXPLORATION_FOCUSES = {
    "mechanics": "Learn unfamiliar controls and object interactions through your own experiments.",
    "qa": (
        "Test this game as a QA playtester: learn how it works, probe edges, "
        "and flag behavior that seems inconsistent or wrong."
    ),
    "reset-retention": (
        "Investigate two independent questions with distinct experiment IDs: "
        "Does a movement effect you choose repeat after restart? "
        "Does an interaction effect you choose repeat after restart? "
        "Choose effects, comparable conditions, controls, and reset timing yourself. "
        "Choose an intentional boundary/refusal probe. "
        "Cite actual original/repeated action steps in each hypothesis; refusals count as effects. "
        "Finish after both comparisons are assessed, or explain remaining uncertainty within the budget."
    ),
    "interaction-retention": (
        "Does an interaction effect you choose repeat from the same visible starting conditions in a later episode? "
        "Choose the effect, conditions, experiments, controls, and reset timing yourself. "
        "Refused interactions are useful evidence. Explain what remains untested when finishing. "
        "This is a narrow interaction diagnostic; the broader movement-and-interaction milestone remains separate."
    ),
}
RETENTION_FOCUSES = {"reset-retention", "interaction-retention"}


def _string_schema(limit: int) -> dict:
    return {"type": "string", "minLength": 1, "maxLength": limit}


EXPLORATION_SCHEMA = {
    "type": "object",
    "properties": {
        "action": {"type": "string", "enum": list(CONTROLS)},
        "reason": _string_schema(240),
        "plan": _string_schema(360),
        "experiment_id": {
            **_string_schema(32),
            "description": "Identifies ONE narrow experiment question. Reuse this ID for retests of that question; different questions need distinct IDs.",
        },
        "experiment_goal": {
            **_string_schema(240),
            "description": "ONE narrow falsifiable question about a chosen effect under stated conditions, separate from the broader session question; do not combine its independent questions.",
        },
        "prediction": {
            **_string_schema(240),
            "description": "Only the ONE next control's immediate observable outcome; exclude results of later controls and unexecuted future trials.",
        },
        "assessment": {
            "anyOf": [
                {"type": "null"},
                {
                    "type": "object",
                    "properties": {"step": {"type": "integer", "minimum": 1}, "outcome": _string_schema(360)},
                    "required": ["step", "outcome"],
                    "additionalProperties": False,
                },
            ],
        },
        "hypotheses": {
            "type": "array", "maxItems": 8,
            "items": {
                "type": "object",
                "properties": {
                    "id": _string_schema(32),
                    "claim": {
                        **_string_schema(240),
                        "description": "A tentative claim under stated conditions. Supporting/refuting evidence comes from actual recorded actions bearing on this claim; untested proposals may have empty evidence.",
                    },
                    "status": {"type": "string", "enum": list(HYPOTHESIS_STATUSES)},
                    "evidence_steps": {"type": "array", "maxItems": 8, "items": {"type": "integer", "minimum": 1}},
                },
                "required": ["id", "claim", "status", "evidence_steps"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["action", "reason", "plan", "experiment_id", "experiment_goal", "prediction", "assessment", "hypotheses"],
    "additionalProperties": False,
}

# Emit reflection before planning to separate the completed outcome from the
# prediction for the next action. This affects formatting, not action selection.
_REFLECTION_ORDER = ("assessment", "hypotheses", "experiment_id", "experiment_goal", "plan", "action", "prediction", "reason")
EXPLORATION_SCHEMA["properties"] = {name: EXPLORATION_SCHEMA["properties"][name] for name in _REFLECTION_ORDER}
EXPLORATION_SCHEMA["required"] = list(_REFLECTION_ORDER)

EXPLORATION_PROMPT = """You are AutoPlay QA's mechanics learner. Answer session_question
through YOUR experiments. Actual states and recorded feedback override tentative
notes/plans; predictions are never observations.
Only action executes; plans/predictions are unexecuted intent. Writing reset in a
plan does not reset anything. Episode 0 without a recorded reset means no reset occurred.

Each tests one narrow falsifiable effect under stated conditions, with its own
experiment ID. Keep it active until answered or explicitly inconclusive; reuse
its ID for retests. Deliberate in a few internal sentences.

FIRST assess completed_outcome_to_assess against its earlier prediction.
assessment=null only when pending_assessment_step=null. When comparison_contract
is present, copy its latest_control_receipt's step, control, episode_before,
episode_after into assessment; outcome describes actual feedback versus prediction.
Choose ONE allowed control and predict only its immediate result.
x increases east; y increases south.

After first game trial, record an uncertain hypothesis: observed effect, control,
evidence step, comparison=null. Assessment alone isn't learning. Additional untested
proposals may coexist. Eight IDs; claims <=240 characters.
Preserve unchanged claim wording and add trial references. Mark refuted claims
contradicted or refine them. Untested=proposal; uncertain=inconclusive;
supported=fits; contradicted=refuted. Labels never independently verify truth.

With comparison_contract, each note declares one game control, unchanged for
that ID. Every evidence step must actually execute that control; cite control
trials, not reset setup. comparison=null before an eligible pair exists. Across-reset
comparisons cite original and repeated trials with matching actual starting
conditions in different episodes. Copy comparison fields from an eligible pair;
result describes recorded effects, not claim truth. For
reset-retention, supported/contradicted require that completed, assessed pair.
Different conditions do not refute conditioned repeatability.

reset costs one action, restores seeded start, retains memory/global steps.
finish costs no game action. Only reset/finish after terminal states.
Finish when answered or inconclusive; assess the final outcome.

Return ONLY JSON matching response_schema, ordered: assessment, hypotheses,
experiment_id, experiment_goal, plan, action, prediction, reason.
"""

REFLECTION_PROMPT = """You are AutoPlay QA's mechanics learner. Answer session_question
through experiments YOU choose. All allowed controls remain available.
Recorded states and feedback override notes. Predictions and plans never execute.
reset is setup, never a movement or interaction trial.

FIRST assess reflection_contract.pending_outcome. Copy its actual position_before,
observed_outcome, observed_events, control and episodes. Compare the earlier
expected_outcome with those five observed facets: matched or mismatched.
expected_outcome is the authoritative forecast; prediction only explains it.
Do not compare your new prediction or an imagined later action. Explain the
recorded result versus the earlier prediction in outcome. This prose remains
model text; typed comparisons do not verify hypotheses.

Choose one narrow falsifiable experiment with one experiment_control and a
stable experiment_id. Keep that target when reusing the ID. Different questions
need different IDs. action_role=trial executes that target; setup may use any
control to prepare conditions, including navigation or reset; finish ends the
session. Labels describe your intent, not proven research quality.

Choose ONE action and predict only its immediate expected_outcome: position,
inventory, key, gate_open, status. A wrong forecast is useful evidence.
For finish, expected_outcome=null: no game action runs. x increases east;
y increases south. Only reset/finish after terminal states.

After the first game trial, record an uncertain hypothesis citing its actual
control and step. Untested proposals may accompany it. Preserve unchanged claim
wording and earlier evidence when retesting. Supported/contradicted retention
notes need both assessed, same-condition trials from different episodes, with
the same experiment ID/target and role trial. Copy one eligible comparison;
otherwise keep uncertain and comparison=null. Refusals count as observed effects.
Eight hypothesis IDs, claims <=240 characters; never cite future steps or reset
setup as control trials. Freeform claims remain tentative.

Use recorded progress to identify unanswered questions. Finish when answered or
inconclusive, assess the last outcome, and state remaining uncertainty. Completing
one target does not answer another. Return ONLY JSON matching the supplied
response contract and schema. Never describe a planned control as completed.
"""

GROUNDED_PROMPT = """You are AutoPlay QA's mechanics learner. Investigate session_question
through experiments YOU choose. Actual states and feedback override tentative
claims. All allowed controls remain available; no route is supplied.

FIRST assess reflection_contract.pending_outcome, or null if none exists.
expected_outcome is the SOLE original forecast. Compare it with observed_outcome:
copy prediction_verdict and mismatched_facets, then explain those actual differences
and feedback in outcome. Rendered prediction text comes from that same forecast,
not another model prediction. Hypothesis claims and outcome prose remain tentative.

Choose one narrow question, experiment_control and stable experiment_id. Trial
executes that target; setup may navigate or reset. Each independent question
needs its own ID. Predict ONLY your next action's five visible state facets in
expected_outcome. Wrong forecasts are useful evidence. finish has null forecast.
x increases east; y increases south. Only reset/finish after terminal states.

Your plan is an executable commitment of 1-4 controls YOU select. Only its first
control executes now and must equal action. research_context.pending_plan contains
the actual remaining tail. Continue that exact tail with the same experiment ID,
or revise it explicitly using recorded evidence and a reason. If no tail remains,
start a new plan. Future controls are intentions, not observations or predictions;
they never execute automatically. Finish may appear only last. Avoid spending the
budget on identical trials after your planned repetitions: choose your next
operation for the unanswered question, or finish inconclusively. Counts and
revisions expose repetition; they do not prove progress.

After the first game trial, keep an uncertain observed-effect note with actual
control and completed evidence. Additional untested proposals may coexist.
Retain unchanged claims/evidence across resets. Supported/contradicted retention
notes require both assessed same-condition trials in different episodes, with
the same ID/target and trial role; copy an eligible pair. Otherwise keep uncertain
and comparison=null. Refusals are useful effects. Eight note IDs, claims<=240
characters. Never cite predictions, future actions or reset setup as game trials.

Answer both independent session questions when requested. Completing one does
not complete the other. Assess the last outcome and state gaps when finishing.
Return ONLY JSON matching response_contract and the supplied schema.
"""

STATE_PROMPT = """You are AutoPlay QA's mechanics learner. Answer session_question through
experiments YOU choose. Actual states and feedback override tentative claims.
All allowed controls remain available; no route is supplied.

FIRST assess reflection_contract.pending_outcome, or null if none exists.
before_outcome and observed_outcome are ACTUAL states. actual_changed_facets
compares those two. mismatched_facets instead compares your earlier predicted
AFTER-state with observed_outcome. A forecast mismatch does not establish object
appearance, collection, or persistence. Describe changes only from actual before
and after values. Copy the original prediction_verdict and mismatched_facets, then
explain actual feedback and those differences. Outcome prose remains tentative.

key is the visible WORLD object position from observation.objects.key; inventory
lists held items. expected_outcome is your SOLE next-action forecast, not a past
state. Predict only one action's immediate five visible facets; wrong forecasts
are useful evidence. Readable prediction is rendered from those same values.
finish has null forecast. x increases east; y increases south.

Choose your plan of 1-4 controls. Only its first control runs and must equal action.
research_context.pending_plan is the actual remaining tail. Continue that exact
tail with the same experiment ID, or revise it explicitly using recorded evidence
and a reason. No tail means you may start a new plan. Future controls never run
automatically. Finish only last; only reset/finish after terminal states. Repeated
new plans or revisions do not by themselves answer a question.

Choose one narrow experiment_control and stable experiment_id. Different targets
or questions need different IDs. Trial measures the target; setup prepares other
conditions. A blocked/refused result can be a completed trial; zero change does
not make it setup. Keep the ID/target when retesting its effect. Choose operations
for unanswered questions instead of spending all the budget on identical trials.

After the first game trial, record an uncertain observed-effect note. Choose its
ACTUAL control and completed evidence before writing the claim; an interaction
is not a movement trial. Additional untested proposals may coexist. Retain unchanged
claims and evidence across resets. Supported/contradicted retention requires both
assessed same-condition trials in different episodes, same ID/target, and trial
role. Copy an eligible pair or keep uncertain with comparison=null. Eight note
IDs, claims<=240 characters. Never cite predictions, future actions or reset setup
as game trials. Claims remain tentative.

Answer both independent session questions when requested; one does not answer
the other. Assess the last outcome and state gaps when finishing. Return only
JSON matching response_contract and the supplied schema.
"""

MEASURED_PROMPT = """You are AutoPlay QA's mechanics learner. Answer session_question with
experiments YOU choose. All allowed controls remain available. Actual states and
feedback are evidence; predictions and plans are never observations.

FIRST assess the latest completed outcome. Copy its actual_changed_facets,
prediction_verdict and mismatched_facets. Actual changes compare before_outcome
with observed_outcome. Forecast differences compare the original expected_outcome
with observed_outcome. Explain these separately; a mistaken forecast does not
show a pickup, movement, or object appearance. key means WORLD object coordinates;
inventory means held items. Outcome prose remains unverified.

Choose which ONE narrow question the next action measures. Keep its experiment
ID, exact original goal and target when revisiting it. A different question needs
a different ID. trial measures experiment_control; setup prepares conditions.
A refused or blocked action can be a trial. Zero change does not make it setup.
Use experiment_progress to see tested questions, assessed pairs and missing notes.
Movement and interaction are independent questions when both are requested.

Choose a plan of 1-4 controls; only its head executes and must equal action.
continue keeps the exact pending tail AND question. retarget keeps that same tail
but explicitly selects another question/ID with a reason and completed evidence.
revise freely replaces the tail with a reason and evidence. new starts when no
tail remains. Keeping queued controls need not keep the old question. Future
controls never run automatically. Finish only last; only reset/finish after won.

After each experiment's first assessed trial, record an uncertain note citing
that experiment's actual trial/control IN THE SAME RESPONSE as its assessment.
experiment_progress counts only previously accepted assessments. Its zero count
does not defer this note: follow required_trial_notes_after_valid_assessment in
response_contract or the top-level request. When first_observed_update_required is true, include an observed
evidence-bearing note now; untested proposals or an empty list cannot satisfy it.
A note from another question is insufficient.
Each evidence-bearing note includes effect: choose condition_step from its cited
assessed game actions and choose one facet, then copy that action's ACTUAL before
and after values. This is a sampled measurement, not your forecast or a universal
rule. No evidence means effect=null. Keep unchanged notes across resets and cite
both trials for a comparison. Supported/contradicted retention needs an eligible
assessed same-condition pair from different episodes, same ID/target, trial role.
Until such a pair exists for the note's actual control, comparison must be JSON
null, never an empty object, a string, or a proposed future comparison. A single
completed trial cannot supply a cross-episode comparison.
Claims remain tentative; eight IDs, claims<=240 characters. Never cite future
controls or reset setup as game trials.

Predict only this next action's five immediate public facets in expected_outcome.
Wrong forecasts are useful; readable prediction is software-derived. Finish has
null forecast. x increases east; y increases south. Finish when answered or
inconclusive, assess the final outcome and state gaps. Return only the JSON
decision matching response_contract and the supplied schema.
"""


def _text(value, name: str, limit: int) -> str:
    if not isinstance(value, str):
        kind = {type(None): "null", bool: "boolean", int: "integer", float: "number",
                list: "array", dict: "object"}.get(type(value), "non-string")
        raise InvalidDecision(f"{name} must be a string; received type {kind}.")
    trimmed = value.strip()
    count = len(trimmed)
    if count == 0:
        raise InvalidDecision(f"{name} is empty after trimming whitespace; expected 1-{limit} characters.")
    if count > limit:
        raise InvalidDecision(f"{name} has {count} trimmed characters; limit is {limit}.")
    return trimmed


def _declared_comparison_pairs(pairs: list[dict], records: dict[int, dict]) -> list[dict]:
    """Keep actual eligible pairs declared as trials of one experiment target."""
    return [pair for pair in pairs if all(
        records[step]["decision"]["action_role"] == "trial"
        and records[step]["decision"]["experiment_control"] == pair["control"]
        for step in (pair["original_step"], pair["repeated_step"])
    ) and records[pair["original_step"]]["decision"]["experiment_id"]
    == records[pair["repeated_step"]]["decision"]["experiment_id"]]


@dataclass(frozen=True)
class Hypothesis:
    id: str
    claim: str
    status: str
    evidence_steps: tuple[int, ...]


@dataclass(frozen=True)
class Assessment:
    step: int
    outcome: str


@dataclass(frozen=True)
class BoundHypothesis(Hypothesis):
    control: str
    comparison: dict | None


@dataclass(frozen=True)
class ReceiptAssessment(Assessment):
    control: str
    episode_before: int
    episode_after: int


@dataclass(frozen=True)
class ReflectedAssessment(ReceiptAssessment):
    position_before: list[int]
    observed_outcome: dict
    observed_events: list[str]
    prediction_verdict: str


@dataclass(frozen=True)
class GroundedAssessment(Assessment):
    prediction_verdict: str
    mismatched_facets: list[str]


@dataclass(frozen=True)
class MeasuredAssessment(GroundedAssessment):
    actual_changed_facets: list[str]


@dataclass(frozen=True)
class MeasuredHypothesis(BoundHypothesis):
    effect: dict | None


@dataclass(frozen=True)
class ExplorationDecision:
    action: str
    reason: str
    plan: str
    experiment_id: str
    experiment_goal: str
    prediction: str
    assessment: Assessment | None
    hypotheses: tuple[Hypothesis, ...]

    @classmethod
    def parse(cls, content: str, allowed: list[str], completed_steps: set[int], pending_step: int | None, *, comparison_records: dict | None = None, assessed_steps: set[int] | None = None, require_comparison: bool = False, previous_hypotheses: dict | None = None, reflection_records: dict | None = None, experiment_bindings: dict | None = None, reflection_version: int = 1) -> "ExplorationDecision":
        try:
            value = json.loads(content)
        except (json.JSONDecodeError, TypeError):
            raise InvalidDecision("Response must contain only a valid exploration JSON object.") from None
        reflected = reflection_records is not None
        grounded = reflected and reflection_version in {2, 3, 4}
        measured = reflected and reflection_version == 4
        if type(reflection_version) is not int or reflection_version not in {1, 2, 3, 4}:
            raise InvalidDecision("Unsupported reflection contract version.")
        required = set(EXPLORATION_SCHEMA["required"])
        if reflected:
            required.update(("expected_outcome", "experiment_control", "action_role"))
        if grounded:
            required.remove("prediction")
        if not isinstance(value, dict) or set(value) != required:
            raise InvalidDecision("Expected exactly the exploration decision fields from the schema.")
        if not isinstance(value["action"], str) or value["action"] not in allowed:
            raise InvalidDecision("Action is not allowed in the current state; after terminal states use reset or finish.")
        string_fields = (("reason", 240), ("experiment_id", 32), ("experiment_goal", 240)) if grounded else (("reason", 240), ("plan", 360), ("experiment_id", 32), ("experiment_goal", 240), ("prediction", 240))
        strings = {name: _text(value[name], name, limit) for name, limit in string_fields}
        if grounded:
            if measured:
                from .research2 import validate_plan, validate_goal

                strings["experiment_goal"] = validate_goal(strings["experiment_id"], strings["experiment_goal"], reflection_records)
            else:
                from .research import validate_plan

            strings["plan"] = validate_plan(value["plan"], action=value["action"], experiment_id=strings["experiment_id"], records=reflection_records, allowed=allowed)
        assessment = value["assessment"]
        strict = comparison_records is not None
        candidate_assessed = set(assessed_steps or ())
        intent = None
        if reflected:
            if not strict:
                raise InvalidDecision("Structured reflection requires the comparison contract.")
            if measured:
                from .measurements import validate_intent, validate_assessment as validate_reflected_assessment
            elif grounded:
                from .forecast import validate_intent, validate_assessment as validate_reflected_assessment
            else:
                from .reflection import validate_intent, validate_assessment as validate_reflected_assessment

            intent = validate_intent(value, experiment_bindings or {})
            assessment = validate_reflected_assessment(assessment, reflection_records, pending_step)
        if strict:
            from .comparisons import validate_assessment, validate_hypothesis

            if not reflected:
                assessment = validate_assessment(assessment, comparison_records, pending_step)
        if pending_step is None:
            if assessment is not None:
                raise InvalidDecision("No completed outcome awaits assessment; assessment must be null.")
        else:
            expected_assessment = {"step", "outcome", "control", "episode_before", "episode_after"} if strict else {"step", "outcome"}
            if reflected:
                expected_assessment = {"step", "outcome", "prediction_verdict", "mismatched_facets"} if grounded else expected_assessment | {"position_before", "observed_outcome", "observed_events", "prediction_verdict"}
                if measured:
                    expected_assessment.add("actual_changed_facets")
            if not isinstance(assessment, dict) or set(assessment) != expected_assessment:
                raise InvalidDecision("Assess the latest completed outcome with step and outcome fields.")
            if type(assessment["step"]) is not int or assessment["step"] != pending_step:
                raise InvalidDecision(f"assessment.step must be pending_assessment_step={pending_step}, not a tick or future step.")
            outcome = _text(assessment["outcome"], "assessment.outcome", 360)
            normalized = outcome.lower().strip(" .!")
            schema_fragments = ("hypotheses:", 'hypotheses":', "experiment_id:", 'experiment_id":')
            if normalized in NONCOMPARISON_LABELS or normalized.startswith(("pending", "awaiting execution", "not yet executed", "unassessed")) or any(fragment in normalized for fragment in schema_fragments):
                raise InvalidDecision("assessment.outcome must describe the observed result and compare the prediction, not a hypothesis-status label, pending placeholder, or schema fragment.")
            if strict:
                if measured:
                    assessment = MeasuredAssessment(pending_step, outcome, assessment["prediction_verdict"], assessment["mismatched_facets"], assessment["actual_changed_facets"])
                elif grounded:
                    assessment = GroundedAssessment(pending_step, outcome, assessment["prediction_verdict"], assessment["mismatched_facets"])
                elif reflected:
                    assessment = ReflectedAssessment(pending_step, outcome, assessment["control"], assessment["episode_before"], assessment["episode_after"], assessment["position_before"], assessment["observed_outcome"], assessment["observed_events"], assessment["prediction_verdict"])
                else:
                    assessment = ReceiptAssessment(pending_step, outcome, assessment["control"], assessment["episode_before"], assessment["episode_after"])
                candidate_assessed.add(pending_step)
            else:
                assessment = Assessment(pending_step, outcome)
        updates = value["hypotheses"]
        if not isinstance(updates, list) or len(updates) > 8:
            raise InvalidDecision("hypotheses must be a list of at most eight updates.")
        hypotheses = []
        seen_ids = set()
        for item in updates:
            expected_hypothesis = {"id", "claim", "status", "evidence_steps", "control", "comparison"} if strict else {"id", "claim", "status", "evidence_steps"}
            if measured:
                expected_hypothesis.add("effect")
            if not isinstance(item, dict) or set(item) != expected_hypothesis:
                if measured:
                    raise InvalidDecision("Each measured hypothesis needs id, claim, status, evidence_steps, control, comparison, and effect.")
                if strict:
                    raise InvalidDecision("Each hypothesis needs exactly id, claim, status, evidence_steps, control, and comparison.")
                raise InvalidDecision("Each hypothesis needs exactly id, claim, status, and evidence_steps.")
            identifier = _text(item["id"], "hypothesis.id", 32)
            if identifier in seen_ids:
                raise InvalidDecision("Duplicate hypothesis IDs are not allowed.")
            seen_ids.add(identifier)
            claim = _text(item["claim"], "hypothesis.claim", 240)
            if not isinstance(item["status"], str) or item["status"] not in HYPOTHESIS_STATUSES:
                raise InvalidDecision("Hypothesis status must be untested, supported, contradicted, or uncertain.")
            evidence = item["evidence_steps"]
            if not isinstance(evidence, list) or len(evidence) > 8 or any(type(step) is not int or step not in completed_steps for step in evidence):
                raise InvalidDecision("Evidence must reference at most eight existing completed global steps, never current/future steps.")
            if len(set(evidence)) != len(evidence):
                raise InvalidDecision("Duplicate evidence step references are not allowed.")
            if item["status"] in {"supported", "contradicted"} and not evidence:
                raise InvalidDecision("Supported or contradicted hypotheses require recorded evidence.")
            if strict:
                previous = (previous_hypotheses or {}).get(identifier)
                core = {**item, "id": identifier, "claim": claim}
                if measured:
                    core.pop("effect")
                    previous = {key: value for key, value in previous.items() if key != "effect"} if previous is not None else None
                try:
                    normalized = validate_hypothesis(core, comparison_records, candidate_assessed, require_comparison=require_comparison, previous=previous)
                except InvalidDecision as exc:
                    shape_error = "Comparison needs exactly original/repeated steps, original/repeated episodes, and result."
                    if not measured or str(exc) != shape_error:
                        raise
                    from .comparisons import comparison_context
                    from .comparison_diagnostics import comparison_shape_diagnostic

                    facts = comparison_context(comparison_records, candidate_assessed, pending_step)
                    pairs = _declared_comparison_pairs(facts["eligible_pairs"], reflection_records)
                    cited_steps = set(core["evidence_steps"])
                    pair_count = 0
                    for pair in pairs:
                        if pair["control"] != core["control"] or not {pair["original_step"], pair["repeated_step"]} <= cited_steps:
                            continue
                        comparison = {name: value for name, value in pair.items() if name != "control"}
                        # Check every cited starting condition as well as the
                        # pair endpoints. This candidate is diagnostic only;
                        # it never becomes a model decision or memory update.
                        try:
                            validate_hypothesis({**core, "comparison": comparison}, comparison_records, candidate_assessed, require_comparison=require_comparison, previous=previous)
                        except InvalidDecision:
                            continue
                        pair_count += 1
                    raise InvalidDecision(comparison_shape_diagnostic(core["comparison"], eligible_pair_count=pair_count)) from None
                if measured:
                    from .measurements import validate_effect

                    effect = validate_effect(item["effect"], normalized, comparison_records, candidate_assessed)
                    hypotheses.append(MeasuredHypothesis(identifier, claim, normalized["status"], tuple(normalized["evidence_steps"]), normalized["control"], normalized["comparison"], effect))
                else:
                    hypotheses.append(BoundHypothesis(identifier, claim, normalized["status"], tuple(normalized["evidence_steps"]), normalized["control"], normalized["comparison"]))
            else:
                hypotheses.append(Hypothesis(identifier, claim, item["status"], tuple(evidence)))
        if reflected:
            if measured:
                from .measurements import validate_trial_comparisons
                from .research2 import unnoted_trial_steps

                candidate_notes = {**(previous_hypotheses or {}), **{item.id: asdict(item) for item in hypotheses}}
                missing = unnoted_trial_steps(reflection_records, candidate_assessed, candidate_notes)
                if missing:
                    diagnostics = [{"control": item.control, "status": item.status, "evidence_steps": list(item.evidence_steps)} for item in hypotheses]
                    raise InvalidDecision(f"Each assessed experiment needs an evidence-linked note of its own actual trial/control in this same response; missing trial steps {missing}. Validated proposed notes (proposals alone do not satisfy learning): {json.dumps(diagnostics, separators=(',', ':'))}.")
            else:
                from .reflection import validate_trial_comparisons

            validate_trial_comparisons([json.loads(json.dumps(asdict(item))) for item in hypotheses], reflection_records)
            decision_class = GroundedDecision if grounded else ReflectiveDecision
            return decision_class(value["action"], **strings, assessment=assessment, hypotheses=tuple(hypotheses), **intent)
        return cls(value["action"], **strings, assessment=assessment, hypotheses=tuple(hypotheses))


@dataclass(frozen=True)
class ReflectiveDecision(ExplorationDecision):
    expected_outcome: dict | None
    experiment_control: str
    action_role: str


@dataclass(frozen=True)
class GroundedDecision:
    """Model-written fields only; readable prediction has explicit derivation."""
    action: str
    reason: str
    plan: dict
    experiment_id: str
    experiment_goal: str
    assessment: GroundedAssessment | None
    hypotheses: tuple[Hypothesis, ...]
    expected_outcome: dict | None
    experiment_control: str
    action_role: str

    @property
    def prediction(self) -> str:
        from .forecast import render_prediction

        return render_prediction(self.action, self.expected_outcome)


@dataclass(frozen=True)
class ExplorationLimits:
    max_actions: int = 24
    max_requests: int = 28
    max_seconds: float = 300
    history_size: int = 6

    def __post_init__(self):
        if min(self.max_actions, self.max_requests, self.max_seconds, self.history_size) <= 0:
            raise ValueError("All exploration limits must be positive.")


def turn_schema(allowed: list[str], completed_steps: set[int], pending_step: int | None, require_learning: bool = False, *, comparison_contract: dict | None = None, require_comparison: bool = False, comparison_records: dict | None = None, reflection_contract: dict | None = None, research_context: dict | None = None, assessed_steps: set[int] | None = None, required_learning_steps: list[int] | None = None) -> dict:
    """Constrain response bookkeeping to real history; never supply mechanics."""
    schema = copy.deepcopy(EXPLORATION_SCHEMA)
    properties = schema["properties"]
    properties["action"]["enum"] = allowed
    if pending_step is None:
        properties["assessment"] = {"type": "null"}
    else:
        assessment = properties["assessment"]["anyOf"][1]
        assessment["properties"]["step"]["enum"] = [pending_step]
        properties["assessment"] = assessment
    items = properties["hypotheses"]["items"]["properties"]
    if completed_steps:
        items["evidence_steps"]["items"]["enum"] = sorted(completed_steps)
    else:
        items["evidence_steps"]["maxItems"] = 0
        items["status"]["enum"] = ["untested", "uncertain"]
    if require_learning:
        properties["hypotheses"]["minItems"] = 1
        properties["hypotheses"]["description"] = (
            "At least one update must be supported, contradicted, or uncertain and cite a completed step. "
            "Other updates may be untested proposals with empty evidence."
        )
    if comparison_contract is not None:
        receipt = comparison_contract["latest_control_receipt"]
        if pending_step is not None:
            assessment = properties["assessment"]
            for name in ("control", "episode_before", "episode_after"):
                assessment["properties"][name] = {"type": "string" if name == "control" else "integer", "enum": [receipt[name]]}
                assessment["required"].append(name)
        hypothesis_schema = properties["hypotheses"]["items"]
        items["control"] = {"type": "string", "enum": list(ACTIONS), "description": "One recorded game control; retain it for this hypothesis ID."}
        pairs = comparison_contract["eligible_pairs"]
        if reflection_contract is not None:
            pairs = _declared_comparison_pairs(pairs, comparison_records)
        if pairs:
            fields = ("original_step", "repeated_step", "original_episode", "repeated_episode", "result")
            pair_schema = {
                "type": "object",
                "properties": {name: {"type": "string" if name == "result" else "integer", "enum": sorted({pair[name] for pair in pairs})} for name in fields},
                "required": list(fields), "additionalProperties": False,
            }
            items["comparison"] = {"anyOf": [{"type": "null"}, pair_schema], "description": "Copy the five fields from one eligible pair matching this note's control and evidence."}
        else:
            items["comparison"] = {"type": "null"}
            if require_comparison:
                items["status"]["enum"] = ["untested", "uncertain"]
        hypothesis_schema["required"].extend(("control", "comparison"))
        # Resets remain assessable but are not evidence for a game control.
        trial_steps = comparison_contract["completed_trial_steps"]
        if trial_steps:
            items["evidence_steps"]["items"]["enum"] = trial_steps
        else:
            items["evidence_steps"]["maxItems"] = 0
        variants = []
        for control in ACTIONS:
            constraints = {"control": {"enum": [control]}}
            if comparison_records is not None:
                refs = sorted(step for step, record in comparison_records.items() if record["type"] == "step" and record["decision"]["action"] == control)
                constraints["evidence_steps"] = {"items": {"enum": refs}} if refs else {"maxItems": 0}
            # A control without a pair can still be tested and learned tentatively.
            variants.append({"properties": {
                **constraints, "comparison": {"type": "null"},
                "status": {"enum": ["untested", "uncertain"] if require_comparison else items["status"]["enum"]},
            }})
            control_pairs = [pair for pair in pairs if pair["control"] == control]
            if control_pairs:
                exact_pairs = [{
                    "type": "object",
                    "properties": {name: {"type": "string" if name == "result" else "integer", "enum": [pair[name]]} for name in fields},
                    "required": list(fields), "additionalProperties": False,
                } for pair in control_pairs]
                variants.append({"properties": {**constraints, "comparison": {"anyOf": exact_pairs}}})
        # Spell out each alternative instead of depending on a decoder to
        # intersect it with the surrounding object's required fields.
        # Local validation remains authoritative even with constrained output.
        complete_variants = []
        for variant in variants:
            branch_properties = copy.deepcopy(items)
            for name, constraint in variant["properties"].items():
                original = branch_properties[name]
                branch_properties[name] = copy.deepcopy(constraint) if name == "comparison" else {**original, **constraint}
                if "items" in original and "items" in constraint:
                    branch_properties[name]["items"] = {**original["items"], **constraint["items"]}
            complete_variants.append({
                "type": "object", "properties": branch_properties,
                "required": list(hypothesis_schema["required"]), "additionalProperties": False,
            })
        hypothesis_schema["anyOf"] = complete_variants
        if require_learning:
            # At least one learned note, without turning every accompanying
            # untested proposal into a fabricated observed result.
            properties["hypotheses"]["contains"] = {
                "type": "object",
                "properties": {
                    "status": {"type": "string", "enum": ["uncertain", "supported", "contradicted"]},
                    "evidence_steps": {"type": "array", "minItems": 1},
                },
                "required": ["status", "evidence_steps"],
            }
    if reflection_contract is not None:
        from .reflection import prediction_schema

        grounded = reflection_contract.get("contract_version") in {2, 3, 4}
        state_grounded = reflection_contract.get("contract_version") in {3, 4}
        measured = reflection_contract.get("contract_version") == 4

        if require_learning:
            first = copy.deepcopy(properties["hypotheses"]["items"])
            first["properties"]["status"]["enum"] = [status for status in first["properties"]["status"]["enum"] if status != "untested"]
            first["properties"]["evidence_steps"]["minItems"] = 1
            description = "A tentative effect actually observed under the cited trial's starting conditions, including refused feedback; do not substitute an unexecuted future condition or a prediction."
            first["properties"]["claim"]["description"] = description
            alternatives = []
            for branch in first["anyOf"]:
                branch_properties = branch["properties"]
                if branch_properties["evidence_steps"].get("maxItems") == 0:
                    continue
                branch_properties["status"]["enum"] = [status for status in branch_properties["status"]["enum"] if status != "untested"]
                branch_properties["evidence_steps"]["minItems"] = 1
                branch_properties["claim"]["description"] = description
                alternatives.append(branch)
            first["anyOf"] = alternatives
            properties["hypotheses"]["prefixItems"] = [first]
            properties["hypotheses"]["description"] = "First update records an observed, evidence-linked effect. Later updates may remain untested proposals with empty evidence."
        properties["expected_outcome"] = {"anyOf": [{"type": "null"}, prediction_schema()], "description": "Predict the sole next control's immediate public after-state; null only for finish. Wrong forecasts are allowed."}
        properties["experiment_control"] = {"type": "string", "enum": list(ACTIONS), "description": "The one game control this narrow question tests; stable for its experiment ID. Setup controls may differ."}
        properties["action_role"] = {"type": "string", "enum": ["trial", "setup", "finish"], "description": "Trial executes the experiment target. Setup may navigate or reset. Finish executes no game action."}
        schema["required"].extend(("expected_outcome", "experiment_control", "action_role"))
        if grounded:
            if measured:
                from .measurements import assessment_schema, effect_schema
                from .research2 import plan_schema, experiment_goals
            else:
                from .forecast import assessment_schema
                from .research import plan_schema

            if research_context is None:
                raise ValueError("Single-forecast schema requires recorded plan context.")
            properties.pop("prediction")
            properties["assessment"] = assessment_schema(comparison_records, pending_step)
            properties["plan"] = plan_schema(allowed, research_context)
        elif pending_step is not None:
            receipt = reflection_contract["pending_outcome"]
            assessment = properties["assessment"]
            observed_schema = prediction_schema()
            for name, value in receipt["observed_outcome"].items():
                observed_schema["properties"][name]["enum"] = [value]
            assessment["properties"].update({
                "position_before": {"type": "array", "items": {"type": "integer"}, "minItems": 2, "maxItems": 2, "enum": [receipt["position_before"]]},
                "observed_outcome": observed_schema,
                "observed_events": {"type": "array", "items": {"type": "string"}, "enum": [receipt["observed_events"]]},
                "prediction_verdict": {"type": "string", "enum": [receipt["prediction_verdict"]], "description": "Recorded comparison of the original forecast's five facets with the actual after-state; freeform prose remains unverified."},
            })
            assessment["required"].extend(("position_before", "observed_outcome", "observed_events", "prediction_verdict"))
            order = ("step", "control", "episode_before", "episode_after", "position_before", "observed_outcome", "observed_events", "prediction_verdict", "outcome")
            assessment["properties"] = {name: assessment["properties"][name] for name in order}
            assessment["required"] = list(order)
        if state_grounded:
            hypothesis_order = ("control", "evidence_steps", "effect", "comparison", "status", "id", "claim") if measured else ("control", "evidence_steps", "comparison", "status", "id", "claim")
            notes = [properties["hypotheses"]["items"], *properties["hypotheses"].get("prefixItems", [])]
            for note_schema in notes:
                for variant in (note_schema, *note_schema.get("anyOf", [])):
                    if measured:
                        targets = variant["properties"]["control"]["enum"]
                        variant["properties"]["effect"] = effect_schema(comparison_records, set(assessed_steps or ()), pending_step, control=targets[0] if len(targets) == 1 else None)
                    variant["properties"] = {name: variant["properties"][name] for name in hypothesis_order}
                    variant["required"] = list(hypothesis_order)
        if measured and required_learning_steps:
            properties["hypotheses"]["minItems"] = max(1, properties["hypotheses"].get("minItems", 0))
            requirements = []
            for first_step in required_learning_steps:
                decision = comparison_records[first_step]["decision"]
                refs = [step for step, record in comparison_records.items() if record["type"] == "step" and record["decision"]["experiment_id"] == decision["experiment_id"] and record["decision"]["action_role"] == "trial"]
                requirements.append({"contains": {"type": "object", "properties": {
                    "control": {"type": "string", "enum": [decision["experiment_control"]]},
                    "status": {"type": "string", "enum": ["uncertain", "supported", "contradicted"]},
                    "evidence_steps": {"type": "array", "contains": {"type": "integer", "enum": refs}},
                    "effect": {"type": "object"},
                }, "required": ["control", "status", "evidence_steps", "effect"]}})
            properties["hypotheses"]["allOf"] = requirements
        order = ("assessment", "hypotheses", "experiment_control", "experiment_id", "experiment_goal", "action_role", "plan", "action", "expected_outcome", "reason") if measured else ("assessment", "hypotheses", "plan", "action", "experiment_control", "experiment_id", "action_role", "experiment_goal", "expected_outcome", "reason") if state_grounded else ("assessment", "hypotheses", "experiment_id", "experiment_control", "experiment_goal", "plan", "action_role", "action", "expected_outcome", "reason") if grounded else ("assessment", "hypotheses", "experiment_id", "experiment_control", "experiment_goal", "plan", "action_role", "action", "expected_outcome", "prediction", "reason")
        schema["properties"] = {name: properties[name] for name in order}
        schema["required"] = list(order)
        # Bind declared trial intent to its selected target in the schema as
        # well as the parser. Setup keeps every allowed non-finish control;
        # these associations choose neither an experiment nor a next control.
        alternatives = []
        for control in ACTIONS:
            if control not in allowed:
                continue
            branch = copy.deepcopy(schema["properties"])
            branch["action"]["enum"] = [control]
            branch["experiment_control"]["enum"] = [control]
            branch["expected_outcome"] = prediction_schema()
            branch["action_role"]["enum"] = ["trial"]
            if grounded:
                branch["plan"] = plan_schema(allowed, research_context, action=control)
            alternatives.append({"type": "object", "properties": branch, "required": list(order), "additionalProperties": False})
        game_controls = [control for control in allowed if control != "finish"]
        if game_controls and not grounded:
            branch = copy.deepcopy(schema["properties"])
            branch["action"]["enum"] = game_controls
            branch["expected_outcome"] = prediction_schema()
            branch["action_role"]["enum"] = ["setup"]
            alternatives.append({"type": "object", "properties": branch, "required": list(order), "additionalProperties": False})
        elif grounded:
            for control in game_controls:
                branch = copy.deepcopy(schema["properties"])
                branch["action"]["enum"] = [control]
                branch["expected_outcome"] = prediction_schema()
                branch["action_role"]["enum"] = ["setup"]
                branch["plan"] = plan_schema(allowed, research_context, action=control)
                alternatives.append({"type": "object", "properties": branch, "required": list(order), "additionalProperties": False})
        if "finish" in allowed:
            branch = copy.deepcopy(schema["properties"])
            branch["action"]["enum"] = ["finish"]
            branch["expected_outcome"] = {"type": "null"}
            branch["action_role"]["enum"] = ["finish"]
            if grounded:
                branch["plan"] = plan_schema(allowed, research_context, action="finish")
            alternatives.append({"type": "object", "properties": branch, "required": list(order), "additionalProperties": False})
        if state_grounded:
            # Reflect the parser's existing plan/experiment association in
            # complete schema branches. A voluntary revision can still choose
            # any current control/target; no experiment or route is supplied.
            bindings = {record["decision"]["experiment_id"]: record["decision"]["experiment_control"] for record in (comparison_records or {}).values()}
            known_ids = sorted(bindings)
            goals = experiment_goals(comparison_records) if measured else {}
            pending_plan = research_context["pending_plan"]
            scoped = []
            for alternative in alternatives:
                for chosen_plan in alternative["properties"]["plan"]["anyOf"]:
                    branch = copy.deepcopy(alternative)
                    branch["properties"]["plan"] = copy.deepcopy(chosen_plan)
                    targets = branch["properties"]["experiment_control"]["enum"]
                    mode = chosen_plan["properties"]["mode"]["enum"][0]
                    if mode == "continue":
                        identifier = pending_plan["experiment_id"]
                        target = bindings[identifier]
                        if target not in targets:
                            continue
                        branch["properties"]["experiment_id"]["enum"] = [identifier]
                        branch["properties"]["experiment_control"]["enum"] = [target]
                        if measured:
                            branch["properties"]["experiment_goal"]["enum"] = [goals[identifier]]
                        scoped.append(branch)
                    elif not known_ids:
                        scoped.append(branch)
                    else:
                        fresh = copy.deepcopy(branch)
                        fresh["properties"]["experiment_id"]["not"] = {"enum": known_ids}
                        scoped.append(fresh)
                        for target in targets:
                            matching_ids = sorted(identifier for identifier, control in bindings.items() if control == target)
                            if measured and mode == "retarget":
                                matching_ids = [identifier for identifier in matching_ids if identifier != pending_plan["experiment_id"]]
                            if matching_ids:
                                for identifiers in ([identifier] for identifier in matching_ids) if measured else [matching_ids]:
                                    existing = copy.deepcopy(branch)
                                    existing["properties"]["experiment_id"]["enum"] = identifiers
                                    existing["properties"]["experiment_control"]["enum"] = [target]
                                    if measured:
                                        existing["properties"]["experiment_goal"]["enum"] = [goals[identifiers[0]]]
                                    scoped.append(existing)
            alternatives = scoped
        schema["anyOf"] = alternatives
    if reflection_contract is not None and reflection_contract.get("contract_version") == 4:
        from .schema_refs import factor_schema

        return factor_schema(schema)
    return schema


def exploration_view(observation: dict) -> dict:
    """Whitelisted learner view; host snapshots retain their original replay shape."""
    keys = (
        "game", "tick", "status", "map", "grid", "coordinates", "legend",
        "player", "objects", "entities", "events",
    )
    view = copy.deepcopy({key: observation[key] for key in keys if key in observation})
    if view.get("tick") == 0:
        view["events"] = ["Environment started."]
    if isinstance(view.get("legend"), dict) and "D" in view["legend"]:
        view["legend"]["D"] = "closed gate"
    player = view.get("player")
    position = player.get("position") if isinstance(player, dict) else None
    rows = view.get("map", view.get("grid", []))
    neighbors = {}
    if (
        isinstance(position, (list, tuple))
        and len(position) == 2
        and all(type(value) is int for value in position)
    ):
        x, y = position
        for action, (dx, dy) in DIRECTIONS.items():
            nx, ny = x + dx, y + dy
            tile = rows[ny][nx] if (
                isinstance(rows, list)
                and 0 <= ny < len(rows)
                and isinstance(rows[ny], (str, list))
                and 0 <= nx < len(rows[ny])
            ) else None
            neighbors[action] = {"destination": {"x": nx, "y": ny}, "tile": tile}
    view["neighbors"] = neighbors
    actions = observation.get("actions", observation.get("controls", []))
    actions = list(actions) if isinstance(actions, (list, tuple)) else []
    capabilities = observation.get("_capabilities")
    can_reset = not isinstance(capabilities, dict) or capabilities.get("reset") is not False
    meta_controls = (["reset"] if can_reset else []) + ["finish"]
    view["controls"] = meta_controls if observation.get("status") == "won" else [
        *actions, *meta_controls,
    ]
    return view


def state_fingerprint(observation: dict) -> str:
    keys = ("game", "status", "map", "grid", "player", "objects", "entities")
    return json.dumps(
        {key: observation[key] for key in keys if key in observation},
        sort_keys=True,
        separators=(",", ":"),
    )


def outcome_view(observation: dict) -> dict:
    """Past outcomes need state deltas and feedback, without repeated full maps."""
    view = exploration_view(observation)
    return {key: view[key] for key in ("tick", "status", "player", "objects", "events") if key in view}


class Coverage:
    def __init__(self, observation: dict):
        self.visits = Counter({tuple(observation["player"]["position"]): 1})
        self.states = {state_fingerprint(observation)}
        self.attempts = {}
        self.action_types = set()
        self.control_counts = Counter()
        self.blocked_movements = 0
        self.unchanged_interactions = 0
        self.reset_count = 0
        self.trials = {}

    def record(self, before: dict, action: str, after: dict, *, step: int | None = None, episode: int = 0):
        fingerprint = state_fingerprint(before)
        self.attempts.setdefault(fingerprint, Counter())[action] += 1
        self.states.add(state_fingerprint(after))
        self.visits[tuple(after["player"]["position"])] += 1
        if action in ACTIONS:
            self.action_types.add(action)
            self.control_counts[action] += 1
            if step is not None:
                self.trials.setdefault((fingerprint, action), {}).setdefault(episode, []).append(step)
        if action == "reset":
            self.reset_count += 1
        if action in DIRECTIONS and before["player"]["position"] == after["player"]["position"]:
            self.blocked_movements += 1
        if action == "interact" and fingerprint == state_fingerprint(after):
            self.unchanged_interactions += 1

    def context(self, observation: dict) -> dict:
        return {
            "visited_positions": [{"position": list(position), "observations": count} for position, count in sorted(self.visits.items())],
            "distinct_observed_states": len(self.states),
            "prior_attempts_in_current_state": dict(self.attempts.get(state_fingerprint(observation), {})),
            "game_control_use_counts": {action: self.control_counts[action] for action in ACTIONS},
            "reset_use_count": self.reset_count,
            "cross_episode_retests": [
                {"action": action, "position": json.loads(fingerprint)["player"]["position"],
                 "trials": [{"episode": episode, "steps": steps} for episode, steps in sorted(episodes.items())]}
                for (fingerprint, action), episodes in sorted(self.trials.items()) if len(episodes) > 1
            ],
        }

    def summary(self) -> dict:
        return {
            "visited_positions": len(self.visits), "distinct_observed_states": len(self.states),
            "distinct_state_action_pairs": sum(len(attempts) for attempts in self.attempts.values()),
            "game_action_types": sorted(self.action_types), "blocked_movements": self.blocked_movements,
            "unchanged_interactions": self.unchanged_interactions,
        }


def observed_changes(before: dict, after: dict) -> dict:
    """Only actual state differences; no engine-rule conclusions."""
    old_position, position = before["player"]["position"], after["player"]["position"]
    return {
        "position_delta": {"dx": position[0] - old_position[0], "dy": position[1] - old_position[1]},
        "tick_delta": after["tick"] - before["tick"],
        "inventory_before": before["player"]["inventory"], "inventory_after": after["player"]["inventory"],
        "changed_objects": {key: {"before": before["objects"].get(key), "after": after["objects"].get(key)} for key in sorted(set(before["objects"]) | set(after["objects"])) if before["objects"].get(key) != after["objects"].get(key)},
    }


def recent_outcome_context(history: list[dict]) -> list[dict]:
    """Keep six outcomes' evidence without repeating unchanged object snapshots."""
    outcomes = []
    for index, item in enumerate(history):
        entry = {key: item[key] for key in ("step", "episode_before", "episode_after", "experiment_id", "action", "prediction")}
        if index == len(history) - 1:
            entry["details_in"] = "completed_outcome_to_assess"
        else:
            before, after = item["before"], item["after"]
            entry.update(
                position_before=before["player"]["position"], position_after=after["player"]["position"],
                status_before=before["status"], status_after=after["status"],
                observed_changes=observed_changes(before, after), observed_events=after["events"],
            )
        outcomes.append(entry)
    return outcomes


def trial_note_requirements(records: dict[int, dict], assessed_steps: set[int], notes, pending_step: int | None) -> dict:
    """Conditional note obligations after this response's valid assessment.

    Actual progress is never advanced here. The referenced trials already ran;
    this describes validation of a proposed reflection, not a next action.
    """
    from .research2 import unnoted_trial_steps

    candidate = assessed_steps | ({pending_step} if pending_step is not None else set())
    missing = unnoted_trial_steps(records, candidate, notes)
    requirements = []
    for step in missing:
        decision = records[step]["decision"]
        refs = [number for number, record in records.items() if number in candidate and record["type"] == "step" and record["decision"]["experiment_id"] == decision["experiment_id"] and record["decision"]["action_role"] == "trial"]
        requirements.append({"experiment_id": decision["experiment_id"], "control": decision["experiment_control"], "evidence_trial_steps": sorted(refs)})
    return {"pending_assessment_step": pending_step, "condition": "If this response validly assesses the pending outcome, include a non-untested evidence-bearing note of each listed experiment in this SAME response. Counts below do not claim an assessment has already been accepted. Choose the claim and measured facet yourself.", "experiments": requirements}


def reflection_response_contract(schema: dict, *, required_trial_notes: dict | None = None) -> dict:
    """Explain output bookkeeping without repeating the decoder's full schema."""
    if "$defs" in schema:
        from .schema_refs import expand_schema

        schema = expand_schema(schema)
    properties = schema["properties"]
    assessment = properties["assessment"]
    grounded = "prediction" not in properties
    measured = "effect" in properties["hypotheses"]["items"]["properties"]
    contract = {
        "required_fields": list(schema["required"]),
        "string_limits": {name: properties[name]["maxLength"] for name in ("experiment_id", "experiment_goal", "plan", "prediction", "reason") if name in properties and "maxLength" in properties[name]},
        "action": properties["action"]["enum"],
        "experiment_control": "One game control, stable for this experiment ID; it does not choose the next action.",
        "action_role": "trial executes experiment_control; setup prepares conditions using any allowed game control or reset; finish only ends the session.",
        "expected_outcome": {
            "fields": ["position", "inventory", "key", "gate_open", "status"],
            "types": "position=[x,y] integers; inventory<=8 held-item strings<=32 chars; key=visible WORLD object position [x,y] or null; gate_open=boolean; status=playing or won." if grounded else "position=[x,y] integers; inventory<=8 literal item strings<=32 chars; key=[x,y] or null; gate_open=boolean; status=playing or won.",
            "rule": "The sole model-written forecast: only this action's immediate result. May be wrong. Required except null for finish. Readable prediction is software-rendered from these values." if grounded else "Forecast only the selected action's immediate result. May be wrong. Required except null for finish; prediction text only explains this authoritative forecast.",
        },
        "assessment": None if assessment.get("type") == "null" else {
            "required_fields": list(assessment["required"]),
            "rule": "Copy step, original prediction_verdict and mismatched_facets from reflection_contract.pending_outcome; outcome<=360 chars explains actual feedback and those differences." if grounded else "Copy actual fields from reflection_contract.pending_outcome; outcome<=360 chars explains observed feedback versus its original expected_outcome.",
        },
        "hypotheses": {
            "required_fields": ["id", "claim", "status", "evidence_steps", "control", "comparison"],
            "limits": "Eight updates and eight distinct session IDs; ID<=32 chars, claim<=240 chars, at most eight unique evidence steps.",
            "statuses": list(HYPOTHESIS_STATUSES),
            "first_observed_update_required": properties["hypotheses"].get("minItems", 0) > 0,
            "rule": "If required, first update records an uncertain observed effect using its actual completed control/step. Proposals may follow. Preserve existing learned notes and stable controls. Claims remain unverified.",
            "comparison": "null until an eligible completed assessed pair exists; copy its original_step, repeated_step, original_episode, repeated_episode, result. Both actual trials must share experiment ID/target, same starting conditions and different episodes; include both in evidence. Retention support/contradiction requires a pair.",
            "evidence": "Cite existing completed game trials of this note's control, never reset, future controls or predictions.",
        },
    }
    if grounded:
        contract["plan"] = {
            "required_fields": ["mode", "controls", "revision_reason", "evidence_steps"],
            "rule": "Choose 1-4 controls; first must equal action. Only it runs. new needs no pending tail, empty reason/evidence; continue keeps exact pending tail/same experiment and empty reason/evidence; revise replaces freely with nonempty reason<=240 chars and 1-8 unique completed evidence steps. Finish only last. Resets may justify a plan revision but never game-mechanics hypotheses.",
            "modes": ["new", "continue", "revise"],
        }
    if measured:
        contract["experiment_goal"] = "One narrow question, exact original wording for an existing ID; choose another ID for a different question."
        contract["hypotheses"]["required_fields"].append("effect")
        contract["hypotheses"]["effect"] = {
            "fields": ["condition_step", "facet", "before", "after"],
            "facets": ["position", "inventory", "key", "gate_open", "status"],
            "rule": "Null only with empty evidence. Otherwise choose one cited assessed game step and one facet; before/after must copy its actual recorded values, never forecast values. This sampled measurement does not verify claim text.",
        }
        contract["hypotheses"]["rule"] = "Every experiment with an assessed trial needs an evidence-bearing note of its own target and declared trial. Other experiments' notes or setup alone are insufficient. Proposals may coexist; claims remain unverified."
        if required_trial_notes is not None:
            contract["hypotheses"]["required_trial_notes_after_valid_assessment"] = required_trial_notes
        if contract["assessment"] is not None:
            contract["assessment"]["rule"] = "Copy step, actual_changed_facets, original prediction_verdict and mismatched_facets. Explain actual before-to-after changes separately from forecast errors; prose remains unverified."
        contract["plan"]["modes"].append("retarget")
        contract["plan"]["rule"] += " retarget keeps the exact pending controls but chooses a different experiment ID, with nonempty reason and completed evidence."
    return contract


def exploration_messages(observation: dict, episode: int, active: dict | None, hypotheses: dict[str, Hypothesis], history: list[dict], coverage: Coverage, pending_step: int | None, remaining: dict, feedback: str = "", response_schema: dict | None = None, *, focus: str = "mechanics", question_progress: dict | None = None, comparison_contract: dict | None = None, reflection_contract: dict | None = None, response_format: str = "json_object", research_context: dict | None = None, experiment_progress: dict | None = None, required_trial_notes: dict | None = None) -> list[dict]:
    changes = observed_changes(history[-1]["before"], history[-1]["after"]) if history else None
    context = {
        "completed_outcome_to_assess": {
            "step": history[-1]["step"], "action_that_already_ran": history[-1]["action"],
            "episode_before": history[-1]["episode_before"], "episode_after": history[-1]["episode_after"],
            "prediction_made_before_it": history[-1]["prediction"],
            "before_state": history[-1]["before"], "after_state": history[-1]["after"],
            "observed_changes": changes,
        } if history else None,
        "session_question": EXPLORATION_FOCUSES[focus],
        "observation": exploration_view(observation), "episode": episode,
        "active_experiment": active, "hypotheses": [asdict(item) for item in hypotheses.values()],
        "recent_outcomes": recent_outcome_context(history), "pending_assessment_step": pending_step,
        "coverage": coverage.context(observation), "remaining_budget": remaining,
    }
    if question_progress is not None:
        context["question_progress"] = question_progress
    if comparison_contract is not None:
        context["comparison_contract"] = comparison_contract
    if reflection_contract is not None:
        context["reflection_contract"] = reflection_contract
        grounded = reflection_contract.get("contract_version") in {2, 3, 4}
        completed = context["completed_outcome_to_assess"]
        if completed is not None:
            prior_text = completed.pop("prediction_made_before_it")
            completed["expected_outcome"] = reflection_contract["pending_outcome"]["expected_outcome"]
            if grounded:
                completed["prediction_source"] = reflection_contract["pending_outcome"]["prediction_source"]
            else:
                completed["prediction_explanation"] = prior_text
        for outcome, past in zip(context["recent_outcomes"], history):
            prior_text = outcome.pop("prediction")
            if grounded:
                outcome.update(expected_outcome=past["expected_outcome"], prediction_source=past["prediction_source"])
            else:
                outcome["prediction_explanation"] = prior_text
        if grounded and active is not None:
            context["active_experiment"] = {key: value for key, value in active.items() if key != "plan"}
    if research_context is not None:
        context["research_context"] = research_context
    if experiment_progress is not None:
        context["experiment_progress"] = experiment_progress
    if feedback:
        context["validation_feedback"] = feedback
    if response_schema is not None:
        # The formal schema is always recorded and sent to the client. Reflected
        # schema mode needs readable bookkeeping here, not a duplicate decoder
        # schema. Plain/object modes and legacy prompts retain the full schema.
        if reflection_contract is not None and response_format == "json_schema":
            context["response_contract"] = reflection_response_contract(response_schema, required_trial_notes=required_trial_notes)
        else:
            context["response_schema"] = response_schema
            if required_trial_notes is not None:
                context["required_trial_notes_after_valid_assessment"] = required_trial_notes
    return [
        {"role": "system", "content": MEASURED_PROMPT if reflection_contract is not None and reflection_contract.get("contract_version") == 4 else STATE_PROMPT if reflection_contract is not None and reflection_contract.get("contract_version") == 3 else GROUNDED_PROMPT if research_context is not None else REFLECTION_PROMPT if reflection_contract is not None else EXPLORATION_PROMPT},
        {"role": "user", "content": json.dumps(context, separators=(",", ":"))},
    ]


def run_exploration(
    game: GameAdapter,
    client: NebiusClient,
    log: SessionLog,
    reset_factory: Callable[[], GameAdapter],
    limits: ExplorationLimits = ExplorationLimits(),
    on_action: Callable[[dict], None] | None = None,
    evaluator: Callable[[dict, str, dict], dict] = evaluate_transition,
    *,
    focus: str = "mechanics",
    comparison_contract: bool = False,
    reflection_contract: bool | int = False,
) -> dict:
    if focus not in EXPLORATION_FOCUSES:
        raise ValueError("Unknown exploration focus.")
    if reflection_contract and not comparison_contract:
        raise ValueError("Structured reflection requires the comparison contract.")
    if type(reflection_contract) not in {bool, int} or reflection_contract not in {False, True, 2, 3, 4}:
        raise ValueError("Unsupported reflection contract version.")
    reflection_version = reflection_contract if type(reflection_contract) is int and reflection_contract >= 2 else 1
    if reflection_contract and reflection_version >= 2:
        if reflection_version == 4:
            from .research2 import plan_context, scope_progress, unnoted_trial_steps
            from .measurements import evidence_measurements
        else:
            from .research import plan_context
    started = time.monotonic()
    deadline = started + limits.max_seconds
    initial_observation = game.observe()
    coverage = Coverage(initial_observation)
    steps = requests = rejected = invalid_streak = resets = episode = 0
    responses = 0
    usage_complete = True
    hypotheses: dict[str, Hypothesis] = {}
    active = None
    history = []
    latencies = []
    completed_steps = set()
    pending_step = None
    experiments = {}
    comparison_records = {}
    assessed_steps = set()
    receipts = []
    experiment_bindings = {}
    feedback = ""
    totals = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    status = "action_limit"
    error = None
    try:
        while True:
            if steps >= limits.max_actions:
                status = "action_limit"
                break
            if requests >= limits.max_requests:
                status = "request_limit"
                break
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                status = "time_limit"
                break
            before = game.observe()
            allowed = exploration_view(before)["controls"]
            requires_learning = steps > resets and not any(item.evidence_steps and item.status != "untested" for item in hypotheses.values())
            facts = None
            if comparison_contract:
                from .comparisons import comparison_context, comparison_receipts

                facts = comparison_context(comparison_records, assessed_steps, pending_step)
            reflection = None
            research = None
            experiment_progress = None
            required_notes = None
            required_trial_notes = None
            if reflection_contract:
                if reflection_version >= 2:
                    if reflection_version == 4:
                        from .measurements import REFLECTION_CONTRACT_VERSION, reflection_receipt, experiment_context, trial_progress, validate_trial_comparisons, prediction_record, comparison_receipts
                    elif reflection_version == 3:
                        from .grounding import REFLECTION_CONTRACT_VERSION, reflection_receipt, experiment_context, trial_progress, validate_trial_comparisons, prediction_record
                    else:
                        from .forecast import REFLECTION_CONTRACT_VERSION, reflection_receipt, experiment_context, trial_progress, validate_trial_comparisons, prediction_record
                    if reflection_version != 4:
                        from .research import plan_context

                    research = plan_context(comparison_records, before)
                    if reflection_version == 4:
                        notes = {key: asdict(item) for key, item in hypotheses.items()}
                        experiment_progress = scope_progress(comparison_records, assessed_steps, notes)
                        candidate_assessed = assessed_steps | ({pending_step} if pending_step is not None else set())
                        required_notes = unnoted_trial_steps(comparison_records, candidate_assessed, notes)
                        required_trial_notes = trial_note_requirements(comparison_records, assessed_steps, notes, pending_step)
                        requires_learning = requires_learning or bool(required_notes)
                else:
                    from .reflection import REFLECTION_CONTRACT_VERSION, reflection_receipt, experiment_context, trial_progress, validate_trial_comparisons

                reflection = {"contract_version": REFLECTION_CONTRACT_VERSION,
                              "pending_outcome": reflection_receipt(comparison_records, pending_step),
                              "experiments": experiment_context(experiment_bindings, comparison_records, assessed_steps)}
            schema = turn_schema(allowed, completed_steps, pending_step, requires_learning, comparison_contract=facts, require_comparison=focus in RETENTION_FOCUSES, comparison_records=comparison_records if comparison_contract else None, reflection_contract=reflection, research_context=research, assessed_steps=assessed_steps, required_learning_steps=required_notes)
            typical_latency = median(latencies) if latencies else None
            progress = None
            if focus in RETENTION_FOCUSES:
                # Read only recorded state/control/evidence facts. Never expose
                # engine contracts or choose the next trial for the learner.
                from .review import retention_progress

                progress = retention_progress(log.path)
                if reflection_contract:
                    progress["declared_trial_progress"] = trial_progress(experiment_bindings, comparison_records, assessed_steps)
            messages = exploration_messages(
                before, episode, active, hypotheses, history, coverage, pending_step,
                {"actions": limits.max_actions - steps, "requests": limits.max_requests - requests, "seconds": round(remaining, 1),
                 "recent_request_median_seconds": round(typical_latency, 3) if typical_latency is not None else None,
                 "approximate_remaining_decisions": min(limits.max_requests - requests, limits.max_actions - steps, int(remaining / max(typical_latency, 0.001))) if typical_latency is not None else None},
                feedback, response_schema=schema, focus=focus, question_progress=progress, comparison_contract=facts, reflection_contract=reflection,
                response_format=client.settings.response_format,
                research_context=research,
                experiment_progress=experiment_progress,
                required_trial_notes=required_trial_notes,
            )
            if reflection_version == 4:
                # Schema preparation can be substantial. Start inference only
                # with a fresh remaining allowance, never the earlier estimate.
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    status = "time_limit"
                    break
            requests += 1
            request_fields = {"request_number": requests, "episode": episode, "observation_tick": before["tick"], "messages": messages, "response_schema": schema, "temperature": client.temperature, "top_p": client.top_p}
            if reflection_version == 4:
                # Persist the actual attempted request even on API failure or
                # interruption. Capturing dispatch time before the call keeps
                # the logical request row first without costly logging before
                # inference consumes its freshly clipped deadline.
                dispatch_fields = {"elapsed_ms": round((time.monotonic() - log.started) * 1000), "timestamp": datetime.now(timezone.utc).isoformat()}
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    requests -= 1
                    status = "time_limit"
                    break
                try:
                    completion = client.complete(messages, timeout=min(client.timeout, remaining), response_schema=schema)
                finally:
                    log.write("request", **request_fields, **dispatch_fields)
            else:
                log.write("request", **request_fields)
                completion = client.complete(messages, timeout=min(client.timeout, remaining), response_schema=schema)
            responses += 1
            latency_ms = completion.metadata.get("latency_ms")
            if type(latency_ms) in {int, float} and isfinite(latency_ms) and latency_ms > 0:
                latencies = [*latencies, latency_ms / 1000][-6:]
            usage_complete = usage_complete and all(type(completion.metadata["usage"].get(name)) is int for name in totals)
            for name in totals:
                totals[name] += completion.metadata["usage"].get(name, 0)
            decision = None
            candidate_memory = None
            invalid_error = None
            try:
                if completion.metadata["finish_reason"] != "stop":
                    raise InvalidDecision("Completion was truncated or interrupted; no action or memory update was applied.")
                decision = ExplorationDecision.parse(
                    completion.content, allowed, completed_steps, pending_step,
                    comparison_records=comparison_records if comparison_contract else None,
                    assessed_steps=assessed_steps, require_comparison=focus in RETENTION_FOCUSES,
                    previous_hypotheses={key: asdict(item) for key, item in hypotheses.items()},
                    reflection_records=comparison_records if reflection_contract else None,
                    experiment_bindings=experiment_bindings,
                    reflection_version=reflection_version,
                )
                candidate_memory = {**hypotheses, **{item.id: item for item in decision.hypotheses}}
                if len(candidate_memory) > 8:
                    raise InvalidDecision("Session memory permits eight distinct hypothesis IDs; revise existing IDs rather than adding a ninth.")
                if reflection_contract:
                    validate_trial_comparisons([json.loads(json.dumps(asdict(item))) for item in candidate_memory.values()], comparison_records)
                if steps > resets and not any(item.evidence_steps and item.status != "untested" for item in candidate_memory.values()):
                    raise InvalidDecision("Record a first tentative learned hypothesis using a completed evidence step after testing a control. Assessment alone is not persistent learning; uncertain is allowed.")
            except InvalidDecision as exc:
                invalid_error = str(exc)
            safe_answer = json.dumps(redact_secrets(asdict(decision), log.secrets), separators=(",", ":")) if invalid_error is None else None
            response_layout = {}
            if invalid_error is None and reflection_version in {3, 4}:
                raw_value = json.loads(completion.content)
                response_layout = {"response_field_order": list(raw_value), "hypothesis_field_orders": [list(note) for note in raw_value["hypotheses"]]}
            log.write("completion", request_number=requests, answer=safe_answer, answer_characters=len(completion.content), **completion.metadata, **response_layout)
            if time.monotonic() >= deadline:
                status = "time_limit"
                break
            if invalid_error is not None:
                rejected += 1
                invalid_streak += 1
                feedback = invalid_error
                if history:
                    last = history[-1]
                    feedback += " The last executed outcome (not your next proposal) is: " + json.dumps({
                        "step": last["step"], "action": last["action"],
                        "position_before": last["before"]["player"]["position"],
                        "position_after": last["after"]["player"]["position"],
                        "events": last["after"]["events"],
                    }, separators=(",", ":"))
                log.write("rejected_decision", request_number=requests, error=feedback)
                if invalid_streak >= 2:
                    status, error = "invalid_decision", feedback
                    break
                continue
            invalid_streak = 0
            feedback = ""
            hypotheses = candidate_memory
            if comparison_contract and decision.assessment is not None:
                assessed_steps.add(decision.assessment.step)
            active = {"id": decision.experiment_id, "goal": decision.experiment_goal, "plan": decision.plan}
            if reflection_contract:
                experiment_bindings[decision.experiment_id] = decision.experiment_control
                active.update(control=decision.experiment_control, action_role=decision.action_role)
            memory = [asdict(item) for item in hypotheses.values()]
            memory_metadata = {}
            if comparison_contract:
                memory = [{**item, "evidence_steps": list(item["evidence_steps"])} for item in memory]
                receipts = comparison_receipts(memory, comparison_records)
                memory_metadata["comparison_receipts"] = receipts
            if reflection_contract:
                memory_metadata.update(experiment_bindings=dict(experiment_bindings), reflection_receipt=reflection["pending_outcome"])
                if reflection_version == 4:
                    memory_metadata["evidence_measurements"] = evidence_measurements(memory, comparison_records)
            log.write("memory", request_number=requests, episode=episode, active_experiment=active, assessment=asdict(decision.assessment) if decision.assessment is not None else None, hypotheses=memory, **memory_metadata)
            pending_step = None
            if decision.action == "finish":
                provenance = prediction_record(decision.action, decision.expected_outcome) if reflection_contract and reflection_version >= 2 else {}
                log.write("finish", request_number=requests, episode=episode, decision=asdict(decision), observation=before, **provenance)
                status = "finished"
                break
            if time.monotonic() >= deadline:
                # Persisting a valid reflection can consume the remaining time.
                # Keep that learning, but do not start a new game action.
                status = "time_limit"
                break
            if decision.action == "reset":
                reset_game = reset_factory()
                after = reset_game.observe()
                game = reset_game
                old_episode = episode
                episode += 1
                resets += 1
                record = {"type": "reset", "episode_before": old_episode, "episode_after": episode, "before": before, "after": after}
            else:
                after = game.act(decision.action)
                evaluation = evaluator(before, decision.action, after)
                record = {"type": "step", "episode": episode, "before": before, "after": after, "evaluation": evaluation}
            steps += 1
            completed_steps.add(steps)
            pending_step = steps
            record.update(step=steps, request_number=requests, decision=asdict(decision))
            if reflection_contract and reflection_version >= 2:
                record.update(prediction_record(decision.action, decision.expected_outcome))
            log.write(record["type"], **{key: value for key, value in record.items() if key != "type"})
            if comparison_contract:
                comparison_records[steps] = record
            coverage.record(before, decision.action, after, step=steps, episode=episode - 1 if decision.action == "reset" else episode)
            experiment = experiments.setdefault(decision.experiment_id, {"id": decision.experiment_id, "goal": decision.experiment_goal, "steps": [], "action_types": []})
            experiment["goal"] = decision.experiment_goal
            experiment["steps"].append(steps)
            if decision.action not in experiment["action_types"]:
                experiment["action_types"].append(decision.action)
            if reflection_contract:
                experiment["control"] = decision.experiment_control
                experiment.setdefault("trial_steps", [])
                experiment.setdefault("setup_steps", [])
                experiment["trial_steps" if decision.action_role == "trial" else "setup_steps"].append(steps)
            history.append({
                "step": steps, "episode_before": episode - 1 if decision.action == "reset" else episode, "episode_after": episode,
                "experiment_id": decision.experiment_id, "action": decision.action, "prediction": decision.prediction,
                "before": outcome_view(before), "after": outcome_view(after),
            })
            if reflection_contract and reflection_version >= 2:
                history[-1].update(expected_outcome=decision.expected_outcome, prediction_source=record["prediction_source"])
            history = history[-limits.history_size:]
            if on_action:
                on_action(record)
            if record.get("evaluation", {}).get("contract_violations"):
                status = "contract_failure"
                break
    except NebiusError as exc:
        status = "time_limit" if time.monotonic() >= deadline else "inference_error"
        error = str(exc)
    except KeyboardInterrupt:
        status = "interrupted"
    except (ValueError, KeyError, RuntimeError) as exc:
        status, error = "environment_error", f"{type(exc).__name__}: {exc}"
    observable_checks = {
        "multiple_experiments": len(experiments) >= 2,
        "movement_actions": bool(coverage.action_types.intersection(DIRECTIONS)),
        "interaction_actions": "interact" in coverage.action_types,
        "blocked_or_unchanged_interaction": coverage.blocked_movements + coverage.unchanged_interactions > 0,
        "evidence_linked_hypotheses": any(item.evidence_steps and item.status != "untested" for item in hypotheses.values()),
    }
    summary = {
        "status": status, "partial": status != "finished", "steps": steps, "focus": focus,
        "game_actions": steps - resets, "resets": resets, "episodes": episode + 1,
        "inference_requests": requests, "rejected_decisions": rejected, "usage": totals,
        "completed_inference_responses": responses, "usage_complete": usage_complete and responses == requests,
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "final_observation": game.observe(), "active_experiment": active,
        "experiments": list(experiments.values()), "coverage": coverage.summary(),
        "hypotheses": [asdict(item) for item in hypotheses.values()],
        "unassessed_final_step": pending_step,
        "unassessed_final_outcome": {
            "step": pending_step, "action": history[-1]["action"],
            "prediction": history[-1]["prediction"],
            "observed_events": history[-1]["after"]["events"],
        } if pending_step is not None else None,
        "pilot_observations": observable_checks,
        "ready_for_live_review": all(observable_checks.values()),
        "review_note": "Intentional probing and hypothesis accuracy still require inspection of the recorded decisions and outcomes.",
        "learning_claims": "Tentative model hypotheses linked to recorded observations; not independently verified rules or bugs.",
    }
    if error:
        summary["error"] = error
    if comparison_contract:
        summary["comparison_receipts"] = receipts
    if reflection_contract:
        summary["experiment_bindings"] = dict(experiment_bindings)
        summary["reflection_receipts"] = [{**reflection_receipt(comparison_records, step), "assessed": step in assessed_steps} for step in sorted(comparison_records)]
        if reflection_version >= 2:
            summary["research_context"] = plan_context(comparison_records, summary["final_observation"], finished=status == "finished")
            if reflection_version == 4:
                summary["hypotheses"] = json.loads(json.dumps(summary["hypotheses"]))
                summary["evidence_measurements"] = evidence_measurements(summary["hypotheses"], comparison_records)
                summary["experiment_progress"] = scope_progress(comparison_records, assessed_steps, summary["hypotheses"])
            if summary["unassessed_final_outcome"] is not None:
                summary["unassessed_final_outcome"].update(expected_outcome=history[-1]["expected_outcome"], prediction_source=history[-1]["prediction_source"])
    log.write("summary", **summary)
    return summary
