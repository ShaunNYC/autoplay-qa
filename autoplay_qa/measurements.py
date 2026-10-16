"""Model-selected, evidence-grounded measurements with tentative prose claims.

Reflection contract 4 binds a selected effect to one assessed actual game trial.
Other cited trials may differ; their actual measurements remain inspectable.
Forecast errors, observed state changes, and hypothesis truth stay distinct.
"""

from copy import deepcopy
import json

from .comparisons import comparison_receipts as _comparison_receipts
from .grounding import (
    ASSESSMENT_FIELDS as _FORECAST_ASSESSMENT_FIELDS, DECISION_FIELDS, FACET_ORDER,
    PREDICTION_SOURCE, assessment_schema as _forecast_assessment_schema,
    experiment_context as _experiment_context, mismatched_facets, outcome_projection,
    prediction_record, prediction_schema, reflection_receipt, render_prediction,
    trial_progress, validate_assessment as _forecast_assessment,
    validate_intent, validate_trial_comparisons as _trial_comparisons,
)
from .game import ACTIONS
from .nebius import InvalidDecision


REFLECTION_CONTRACT_VERSION = 4
ASSESSMENT_FIELDS = _FORECAST_ASSESSMENT_FIELDS | {"actual_changed_facets"}
EFFECT_FIELDS = {"condition_step", "facet", "before", "after"}


def _json(value) -> str:
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError):
        raise InvalidDecision("Recorded measurement facts must contain finite JSON values.") from None


def _game_record(records: dict[int, dict], step: int, control: str | None = None) -> dict:
    if not isinstance(records, dict) or type(step) is not int or step < 1 or step not in records:
        raise InvalidDecision("A measurement must cite an existing completed game step.")
    record = records[step]
    if not isinstance(record, dict) or record.get("type") != "step":
        raise InvalidDecision("A measurement witness must be a game trial, excluding reset and finish.")
    decision = record.get("decision")
    if not isinstance(decision, dict) or not isinstance(decision.get("action"), str) or decision["action"] not in ACTIONS:
        raise InvalidDecision("A measurement witness needs an actual recorded game control.")
    if control is not None and decision["action"] != control:
        raise InvalidDecision("The measurement witness must execute the hypothesis control.")
    # Existing receipt checks bind record steps and actual episode metadata.
    reflection_receipt(records, step)
    return record


def _check_effect(value, note: dict, records: dict[int, dict], assessed_steps: set[int] | None) -> dict | None:
    if not isinstance(note, dict) or not isinstance(note.get("control"), str) or note["control"] not in ACTIONS:
        raise InvalidDecision("A measured hypothesis needs a declared game control.")
    refs = note.get("evidence_steps")
    if not isinstance(refs, list) or len(refs) > 8 or any(type(step) is not int for step in refs) or len(set(refs)) != len(refs):
        raise InvalidDecision("Measured evidence must contain at most eight unique completed integer game steps.")
    if not refs:
        if value is not None:
            raise InvalidDecision("A hypothesis without recorded evidence must have effect=null.")
        return None
    if not isinstance(value, dict) or set(value) != EFFECT_FIELDS:
        raise InvalidDecision("An evidence-linked effect needs exactly condition_step, facet, before, and after.")
    step, facet = value["condition_step"], value["facet"]
    if type(step) is not int or step not in refs:
        raise InvalidDecision("effect.condition_step must be one of the hypothesis's actual evidence steps.")
    if not isinstance(facet, str) or facet not in FACET_ORDER:
        raise InvalidDecision("effect.facet must be position, inventory, key, gate_open, or status.")
    for ref in refs:
        _game_record(records, ref, note["control"])
    if assessed_steps is not None and step not in {item for item in assessed_steps if type(item) is int}:
        raise InvalidDecision("The selected effect witness must have an accepted assessment, including a current valid assessment.")
    record = _game_record(records, step, note["control"])
    before, after = outcome_projection(record["before"])[facet], outcome_projection(record["after"])[facet]
    if _json(value["before"]) != _json(before) or _json(value["after"]) != _json(after):
        raise InvalidDecision("The effect's before/after values must match its actual witness state, never a forecast value.")
    return {"condition_step": step, "facet": facet, "before": deepcopy(before), "after": deepcopy(after)}


def validate_effect(value, note: dict, records: dict[int, dict], assessed_steps: set[int]) -> dict | None:
    """Validate only the selected assessed witness, not other trials' outcomes."""
    if not isinstance(assessed_steps, (set, frozenset)):
        raise InvalidDecision("Accepted assessment steps are required for effect validation.")
    return _check_effect(value, note, records, assessed_steps)


def effect_schema(records: dict[int, dict], assessed_steps: set[int], pending_step: int | None = None, *, control: str | None = None) -> dict:
    """Offer factual past witness shapes while leaving their selection to the model."""
    if not isinstance(records, dict) or any(type(step) is not int or step < 1 for step in records):
        raise InvalidDecision("Measurement records must be keyed by positive integer steps.")
    if not isinstance(assessed_steps, (set, frozenset)):
        raise InvalidDecision("Effect schema needs accepted assessment steps.")
    if control is not None and (not isinstance(control, str) or control not in ACTIONS):
        raise InvalidDecision("Effect schema control must be a game control.")
    eligible = {step for step in assessed_steps if type(step) is int and step in records}
    if pending_step is not None:
        if type(pending_step) is not int or not records or pending_step != max(records):
            raise InvalidDecision("Pending measurement assessment must identify the latest completed action.")
        eligible.add(pending_step)
    variants = []
    facets = prediction_schema()["properties"]
    for step in sorted(eligible):
        record = records[step]
        if not isinstance(record, dict):
            raise InvalidDecision("Recorded measurement action metadata is invalid.")
        if record.get("type") != "step" or control is not None and record.get("decision", {}).get("action") != control:
            continue
        record = _game_record(records, step, control)
        before, after = outcome_projection(record["before"]), outcome_projection(record["after"])
        for facet in FACET_ORDER:
            before_schema, after_schema = deepcopy(facets[facet]), deepcopy(facets[facet])
            before_schema["enum"], after_schema["enum"] = [before[facet]], [after[facet]]
            variants.append({
                "type": "object", "properties": {
                    "condition_step": {"type": "integer", "enum": [step]},
                    "facet": {"type": "string", "enum": [facet]},
                    "before": before_schema, "after": after_schema,
                }, "required": ["condition_step", "facet", "before", "after"], "additionalProperties": False,
            })
    return {"anyOf": [{"type": "null"}, *variants]} if variants else {"type": "null"}


def validate_assessment(value, records: dict[int, dict], pending_step: int | None) -> dict | None:
    """Separate actual changes from the original forecast's mismatched facets."""
    if pending_step is None:
        return _forecast_assessment(value, records, None)
    if not isinstance(value, dict) or set(value) != ASSESSMENT_FIELDS:
        raise InvalidDecision("Measured assessment needs the four forecast fields plus actual_changed_facets.")
    original = _forecast_assessment({key: value[key] for key in _FORECAST_ASSESSMENT_FIELDS}, records, pending_step)
    actual = reflection_receipt(records, pending_step)
    if not isinstance(value["actual_changed_facets"], list) or _json(value["actual_changed_facets"]) != _json(actual["actual_changed_facets"]):
        raise InvalidDecision("assessment.actual_changed_facets must describe actual before/after changes, not forecast mismatches.")
    return {**original, "actual_changed_facets": deepcopy(actual["actual_changed_facets"])}


def assessment_schema(records: dict[int, dict], pending_step: int | None) -> dict:
    schema = _forecast_assessment_schema(records, pending_step)
    if pending_step is None:
        return schema
    schema["properties"]["actual_changed_facets"] = {
        "type": "array", "maxItems": len(FACET_ORDER),
        "items": {"type": "string", "enum": list(FACET_ORDER)},
        "enum": [reflection_receipt(records, pending_step)["actual_changed_facets"]],
        "description": "Exact actual before/after changed facets, independent of forecast mismatches.",
    }
    schema["required"].append("actual_changed_facets")
    return schema


def _core_notes(notes) -> list[dict]:
    items = notes.values() if isinstance(notes, dict) else notes
    result = []
    for note in items:
        if not isinstance(note, dict):
            raise InvalidDecision("Measured notes must be hypothesis objects.")
        result.append({key: value for key, value in note.items() if key != "effect"})
    return result


def comparison_receipts(notes, records: dict[int, dict]) -> list[dict]:
    """Preserve comparison-1 facts after removing the measured-effect field."""
    return _comparison_receipts(_core_notes(notes), records)


def validate_trial_comparisons(notes, records: dict[int, dict]) -> None:
    return _trial_comparisons(_core_notes(notes), records)


def evidence_measurements(notes, records: dict[int, dict]) -> list[dict]:
    """Derive actual values for every cited row; never create learned claims."""
    items = notes.values() if isinstance(notes, dict) else notes
    receipts = []
    for note in items:
        # The caller already validates accepted assessment membership. As with
        # comparison receipts, this rechecks actual structural facts without
        # inventing which steps were assessed.
        effect = _check_effect(note.get("effect"), note, records, None)
        if effect is None:
            continue
        rows = []
        for step in note["evidence_steps"]:
            record = _game_record(records, step, note["control"])
            receipt = reflection_receipt(records, step)
            before, after = receipt["before_outcome"][effect["facet"]], receipt["observed_outcome"][effect["facet"]]
            rows.append({
                "step": step, "episode": receipt["episode_before"],
                "action_role": record["decision"]["action_role"],
                "before": deepcopy(before), "after": deepcopy(after),
                "actual_changed": _json(before) != _json(after),
            })
        receipts.append({
            "hypothesis_id": note["id"], "control": note["control"],
            "facet": effect["facet"], "condition_step": effect["condition_step"],
            "measurements": rows, "source": "actual_recorded_state", "claims_verified": False,
        })
    return receipts


def experiment_context(bindings: dict[str, str], records: dict[int, dict], assessed_steps: set[int]) -> dict:
    """Add first executed model goals; accepted-but-unexecuted IDs have no goal."""
    from .research2 import experiment_goals

    context = _experiment_context(bindings, records, assessed_steps)
    goals = experiment_goals(records)
    for experiment in context["experiments"]:
        experiment["goal"] = goals.get(experiment["id"])
    return context
