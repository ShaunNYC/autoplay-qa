"""Typed forecasts and actual reflection facts, without mechanics or a policy.

This additive contract requires comparison contract 1 in its caller. Forecasts
describe five public state facets and may be wrong. Only after execution do we
compare them with the recorded outcome. Prose and hypothesis truth are unverified.
"""

from copy import deepcopy
import json

from .comparisons import comparison_receipts, validate_assessment as validate_control_assessment
from .game import ACTIONS
from .nebius import InvalidDecision


REFLECTION_CONTRACT_VERSION = 1
_FACETS = {"position", "inventory", "key", "gate_open", "status"}
_BASE_ASSESSMENT = {"step", "outcome", "control", "episode_before", "episode_after"}
_FACT_ASSESSMENT = {"position_before", "observed_outcome", "observed_events", "prediction_verdict"}
_INTENT_FIELDS = ("expected_outcome", "experiment_control", "action_role")


def _text(value, field: str, limit: int) -> str:
    if not isinstance(value, str) or not value.strip() or len(value.strip()) > limit:
        raise InvalidDecision(f"{field} must contain 1-{limit} trimmed characters.")
    return value.strip()


def _position(value, field: str) -> list[int]:
    if not isinstance(value, list) or len(value) != 2 or any(type(item) is not int for item in value):
        raise InvalidDecision(f"{field} must be an [x,y] pair of integers.")
    return list(value)


def _outcome(value, field: str) -> dict:
    if not isinstance(value, dict) or set(value) != _FACETS:
        raise InvalidDecision(f"{field} needs exactly position, inventory, key, gate_open, and status.")
    inventory = value["inventory"]
    if not isinstance(inventory, list) or len(inventory) > 8:
        raise InvalidDecision(f"{field}.inventory must be a list of at most eight bounded strings.")
    for item in inventory:
        if not isinstance(item, str) or not item.strip() or len(item) > 32:
            raise InvalidDecision(f"{field}.inventory items must be nonempty strings of at most 32 characters.")
    # Inventory names are literal observable values. Trimming a forecast could
    # silently turn a different predicted item into an observed one.
    inventory = list(inventory)
    if type(value["gate_open"]) is not bool:
        raise InvalidDecision(f"{field}.gate_open must be a boolean.")
    if not isinstance(value["status"], str) or value["status"] not in {"playing", "won"}:
        raise InvalidDecision(f"{field}.status must be playing or won.")
    return {
        "position": _position(value["position"], f"{field}.position"),
        "inventory": inventory,
        "key": None if value["key"] is None else _position(value["key"], f"{field}.key"),
        "gate_open": value["gate_open"], "status": value["status"],
    }


def _same_json(value, expected) -> bool:
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) == json.dumps(expected, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError):
        raise InvalidDecision("Reflection facts must contain finite JSON values.") from None


def outcome_projection(observation: dict) -> dict:
    """Copy only the five visible facets, never developer or passability hints."""
    try:
        value = {
            "position": observation["player"]["position"],
            "inventory": observation["player"]["inventory"],
            "key": observation["objects"]["key"],
            "gate_open": observation["objects"]["gate"]["open"],
            "status": observation["status"],
        }
    except (KeyError, TypeError):
        raise InvalidDecision("Recorded public state is incomplete for reflection.") from None
    return _outcome(value, "observed_outcome")


def prediction_schema() -> dict:
    """Describe forecast types only; enumerate no correct future outcomes."""
    coordinate = {"type": "array", "minItems": 2, "maxItems": 2, "items": {"type": "integer"}}
    return {
        "type": "object",
        "properties": {
            "position": deepcopy(coordinate),
            "inventory": {"type": "array", "maxItems": 8, "items": {"type": "string", "minLength": 1, "maxLength": 32}},
            "key": {"anyOf": [{"type": "null"}, deepcopy(coordinate)]},
            "gate_open": {"type": "boolean"},
            "status": {"type": "string", "enum": ["playing", "won"]},
        },
        "required": ["position", "inventory", "key", "gate_open", "status"],
        "additionalProperties": False,
    }


def validate_intent(value: dict, bindings: dict[str, str]) -> dict:
    """Check declared question/role and forecast shape, without changing memory."""
    if not isinstance(value, dict) or any(field not in value for field in (*_INTENT_FIELDS, "action", "experiment_id")):
        raise InvalidDecision("Reflection decisions need expected_outcome, experiment_control, and action_role.")
    identifier = _text(value["experiment_id"], "experiment_id", 32)
    control, action, role = value["experiment_control"], value["action"], value["action_role"]
    if not isinstance(control, str) or control not in ACTIONS:
        raise InvalidDecision("experiment_control must declare one game control, excluding reset and finish.")
    if not isinstance(bindings, dict) or any(not isinstance(key, str) or not isinstance(item, str) or item not in ACTIONS for key, item in bindings.items()):
        raise InvalidDecision("Recorded experiment bindings are invalid.")
    if identifier in bindings and bindings[identifier] != control:
        raise InvalidDecision("A reused experiment ID must retain its declared experiment_control.")
    if not isinstance(action, str) or action not in {*ACTIONS, "reset", "finish"}:
        raise InvalidDecision("Reflection action must be a game control, reset, or finish.")
    if not isinstance(role, str) or role not in {"trial", "setup", "finish"}:
        raise InvalidDecision("action_role must be trial, setup, or finish.")
    if action == "finish":
        if role != "finish" or value["expected_outcome"] is not None:
            raise InvalidDecision("finish requires action_role=finish and expected_outcome=null.")
        expected = None
    else:
        if role == "finish" or action == "reset" and role != "setup":
            raise InvalidDecision("A reset must be setup; only finish can have action_role=finish.")
        if role == "trial" and action != control:
            raise InvalidDecision("A trial must execute its experiment_control; preparation controls can be setup.")
        expected = _outcome(value["expected_outcome"], "expected_outcome")
    return {"expected_outcome": expected, "experiment_control": control, "action_role": role}


def _latest_record(records: dict[int, dict], pending_step: int) -> tuple[dict, dict]:
    if not isinstance(records, dict) or any(type(step) is not int or step < 1 for step in records):
        raise InvalidDecision("Reflection records must be keyed by positive integer action steps.")
    if type(pending_step) is not int or not records or pending_step != max(records):
        raise InvalidDecision("Reflection pending step must identify the latest completed action.")
    record = records[pending_step]
    if not isinstance(record, dict) or not isinstance(record.get("decision"), dict):
        raise InvalidDecision("Recorded reflection action metadata is invalid.")
    before = record.get("episode_before", record.get("episode"))
    after = record.get("episode_after", record.get("episode"))
    core = validate_control_assessment({
        "step": pending_step, "outcome": "Recorded reflection receipt.",
        "control": record["decision"].get("action"), "episode_before": before, "episode_after": after,
    }, records, pending_step)
    return record, core


def reflection_receipt(records: dict[int, dict], pending_step: int | None) -> dict | None:
    """Return actual past facts and compare the original recorded forecast."""
    if pending_step is None:
        return None
    if not isinstance(records, dict) or any(type(step) is not int or step < 1 for step in records):
        raise InvalidDecision("Reflection records must be keyed by positive integer action steps.")
    if type(pending_step) is not int or pending_step not in records:
        raise InvalidDecision("Reflection receipt must identify an existing completed action.")
    prefix = {step: record for step, record in records.items() if step <= pending_step}
    record, core = _latest_record(prefix, pending_step)
    intent = validate_intent(record["decision"], {})
    expected = intent["expected_outcome"]
    if expected is None:
        raise InvalidDecision("A completed game/reset action must have a recorded forecast.")
    before, after = record.get("before"), record.get("after")
    observed = outcome_projection(after)
    position_before = outcome_projection(before)["position"]
    events = after.get("events")
    if not isinstance(events, list) or any(not isinstance(item, str) for item in events):
        raise InvalidDecision("Recorded reflection feedback must be an array of strings.")
    events = ["Environment started."] if after.get("tick") == 0 else deepcopy(events)
    return {
        **{field: core[field] for field in ("step", "control", "episode_before", "episode_after")},
        "position_before": position_before, "expected_outcome": expected,
        "observed_outcome": observed, "observed_events": events,
        "prediction_verdict": "matched" if _same_json(expected, observed) else "mismatched",
    }


def validate_assessment(value, records: dict[int, dict], pending_step: int | None) -> dict | None:
    """Bind typed reflection to actual feedback; outcome prose stays unverified."""
    if pending_step is None:
        return validate_control_assessment(value, records, None)
    if not isinstance(value, dict) or set(value) != _BASE_ASSESSMENT | _FACT_ASSESSMENT:
        raise InvalidDecision("Reflection assessment needs the five control fields plus position_before, observed_outcome, observed_events, and prediction_verdict.")
    core = validate_control_assessment({field: value[field] for field in _BASE_ASSESSMENT}, records, pending_step)
    actual = reflection_receipt(records, pending_step)
    # Do not normalize factual echoes: their exact JSON types/values must match
    # execution, including boolean versus integer and nontrimmed event strings.
    for field in ("position_before", "observed_outcome", "observed_events", "prediction_verdict"):
        if not _same_json(value[field], actual[field]):
            raise InvalidDecision(f"assessment.{field} disagrees with the actual recorded reflection receipt.")
    return {**core, **{field: deepcopy(actual[field]) for field in ("position_before", "observed_outcome", "observed_events", "prediction_verdict")}}


def validate_trial_comparisons(notes, records: dict[int, dict]) -> None:
    """Require core pair trials to belong to one declared experiment target."""
    items = list(notes.values()) if isinstance(notes, dict) else list(notes)
    # Ordinary comparison validation binds controls, episodes, conditions,
    # feedback and references. These additional checks concern trial intent.
    for receipt in comparison_receipts(items, records):
        pair, control = receipt["comparison"], receipt["control"]
        identifiers = []
        for step in (pair["original_step"], pair["repeated_step"]):
            decision = records[step]["decision"]
            intent = validate_intent(decision, {})
            if intent["experiment_control"] != control or intent["action_role"] != "trial":
                raise InvalidDecision("Comparison pair steps must be declared trials of the hypothesis control.")
            identifiers.append(_text(decision["experiment_id"], "experiment_id", 32))
        if identifiers[0] != identifiers[1]:
            raise InvalidDecision("Comparison pair trials must belong to the same actual experiment ID.")


def experiment_context(bindings: dict[str, str], records: dict[int, dict], assessed_steps: set[int]) -> dict:
    """Expose declared targets and completed role counts, never a next control."""
    if not isinstance(bindings, dict) or any(not isinstance(key, str) or not isinstance(control, str) or control not in ACTIONS for key, control in bindings.items()):
        raise InvalidDecision("Recorded experiment bindings are invalid.")
    if not isinstance(records, dict) or any(type(step) is not int or step < 1 for step in records):
        raise InvalidDecision("Reflection records must be keyed by positive integer action steps.")
    assessed = {step for step in assessed_steps if type(step) is int}
    experiments = {identifier: {"id": identifier, "control": control, "trial_steps": [], "setup_steps": [], "assessed_steps": []} for identifier, control in bindings.items()}
    for step in sorted(records):
        record = records[step]
        if not isinstance(record, dict) or not isinstance(record.get("decision"), dict):
            raise InvalidDecision("Recorded reflection action metadata is invalid.")
        decision = record["decision"]
        intent = validate_intent(decision, bindings)
        identifier = _text(decision["experiment_id"], "experiment_id", 32)
        if identifier not in experiments:
            raise InvalidDecision("A completed action must have an accepted experiment binding.")
        if intent["action_role"] == "finish":
            raise InvalidDecision("Finish has no completed game/reset step.")
        # Bind the catalog's controls/episodes to actual step/reset metadata too.
        _latest_record({key: value for key, value in records.items() if key <= step}, step)
        experiments[identifier][f"{intent['action_role']}_steps"].append(step)
        if step in assessed:
            experiments[identifier]["assessed_steps"].append(step)
    return {"contract_version": REFLECTION_CONTRACT_VERSION, "experiments": list(experiments.values())}


def trial_progress(bindings: dict[str, str], records: dict[int, dict], assessed_steps: set[int]) -> dict:
    """Count completed declared trials, without prescribing controls or claims.

    Same-start repeats count later trial rows once each, including repeats in
    one episode. Assessed cross-episode pairs count comparable pairs whose two
    actual trials both have accepted assessments. Neither count proves learning.
    """
    context = experiment_context(bindings, records, assessed_steps)
    assessed = {step for step in assessed_steps if type(step) is int}
    condition_fields = ("game", "status", "map", "player", "objects")
    experiments = []
    for declared in context["experiments"]:
        trials = []
        for step in declared["trial_steps"]:
            record = records[step]
            before = record.get("before")
            if not isinstance(before, dict) or any(field not in before for field in condition_fields):
                raise InvalidDecision("Recorded trial starting state is incomplete for progress counts.")
            conditions = {field: before[field] for field in condition_fields}
            # Validate finite JSON even if this is the experiment's first trial.
            _same_json(conditions, conditions)
            trials.append({"step": step, "episode": record.get("episode_before", record.get("episode")), "conditions": conditions})
        repeats = pairs = 0
        for index, repeated in enumerate(trials):
            comparable = [original for original in trials[:index] if _same_json(original["conditions"], repeated["conditions"])]
            repeats += bool(comparable)
            pairs += sum(
                original["episode"] < repeated["episode"]
                and original["step"] in assessed and repeated["step"] in assessed
                for original in comparable
            )
        experiments.append({
            "id": declared["id"], "control": declared["control"],
            "trial_count": len(trials), "assessed_trial_count": sum(trial["step"] in assessed for trial in trials),
            "trial_episodes": sorted({trial["episode"] for trial in trials}),
            "same_start_repeated_trial_count": repeats, "assessed_cross_episode_pair_count": pairs,
        })
    return {
        "scope": "Completed trial-role actions grouped by declared experiment ID and control; starting conditions exclude ticks and feedback.",
        "claims_verified": False, "experiments": experiments,
    }
