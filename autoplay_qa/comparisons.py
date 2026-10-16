"""Pure checks of recorded comparison facts, never mechanics or action policy.

Records are actual completed step/reset rows keyed by their global action number.
The caller owns replay and atomic memory updates. These checks bind model fields
to those receipts; they do not establish hypothesis truth or intentionality.
"""

from copy import deepcopy
from itertools import combinations
import json

from .game import ACTIONS
from .nebius import InvalidDecision


COMPARISON_CONTRACT_VERSION = 1
MAX_CONTEXT_TRIALS = 24
_STATUSES = {"untested", "supported", "contradicted", "uncertain"}
_CONDITION_FIELDS = ("game", "status", "map", "player", "objects")
_ASSESSMENT_FIELDS = {"step", "outcome", "control", "episode_before", "episode_after"}
_HYPOTHESIS_FIELDS = {"id", "claim", "status", "evidence_steps", "control", "comparison"}
_COMPARISON_FIELDS = {"original_step", "repeated_step", "original_episode", "repeated_episode", "result"}


def _text(value, field: str, limit: int) -> str:
    if not isinstance(value, str):
        raise InvalidDecision(f"{field} must be a string.")
    value = value.strip()
    if not value:
        raise InvalidDecision(f"{field} must contain nonempty trimmed text.")
    if len(value) > limit:
        raise InvalidDecision(f"{field} exceeds its {limit}-character limit.")
    return value


def _record(records: dict[int, dict], step: int) -> dict:
    """Check receipt metadata without consulting game rules or an engine."""
    if type(step) is not int or step < 1 or step not in records:
        raise InvalidDecision("Reference must identify an existing completed action step.")
    record = records[step]
    if not isinstance(record, dict) or type(record.get("step")) is not int or record["step"] != step:
        raise InvalidDecision("Recorded action metadata is invalid.")
    action = record.get("decision", {}).get("action") if isinstance(record.get("decision"), dict) else None
    kind = record.get("type")
    if kind == "step":
        before = record.get("episode_before", record.get("episode"))
        after = record.get("episode_after", record.get("episode"))
        if not isinstance(action, str) or action not in ACTIONS or before != after:
            raise InvalidDecision("Recorded game-control metadata is invalid.")
        if "episode" in record and (type(record["episode"]) is not int or record["episode"] != before):
            raise InvalidDecision("Recorded game-control episode metadata is inconsistent.")
    elif kind == "reset":
        before, after = record.get("episode_before"), record.get("episode_after")
        if action != "reset":
            raise InvalidDecision("Recorded reset-control metadata is invalid.")
    else:
        raise InvalidDecision("Reference must identify an actual step or reset receipt.")
    if type(before) is not int or type(after) is not int or before < 0 or after < 0:
        raise InvalidDecision("Recorded action episodes must be nonnegative integers.")
    if kind == "reset" and after != before + 1:
        raise InvalidDecision("Recorded reset must advance exactly one episode.")
    return {"record": record, "step": step, "control": action, "episode_before": before, "episode_after": after, "record_type": kind}


def _ordered_steps(records: dict[int, dict]) -> list[int]:
    if not isinstance(records, dict) or any(type(step) is not int or step < 1 for step in records):
        raise InvalidDecision("Completed records must be keyed by positive integer action steps.")
    return sorted(records)


def _conditions(observation: dict) -> dict:
    if not isinstance(observation, dict) or any(field not in observation for field in _CONDITION_FIELDS):
        raise InvalidDecision("Recorded trial state is incomplete for comparison.")
    return {field: observation[field] for field in _CONDITION_FIELDS}


def _outcome(observation: dict) -> dict:
    result = _conditions(observation)
    if "events" not in observation:
        raise InvalidDecision("Recorded trial feedback is missing for comparison.")
    return {**result, "events": observation["events"]}


def _same_json(first: dict, second: dict) -> bool:
    """Exact JSON facts distinguish booleans from numbers and ignore key order."""
    try:
        return json.dumps(first, sort_keys=True, separators=(",", ":"), allow_nan=False) == json.dumps(second, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError):
        raise InvalidDecision("Recorded comparison states must contain finite JSON values.") from None


def _visible_state(observation: dict, *, feedback: bool) -> dict:
    """Public observation fields; never copy arbitrary nested hint metadata."""
    _conditions(observation)
    player, objects = observation["player"], observation["objects"]
    if not isinstance(player, dict) or not isinstance(objects, dict):
        raise InvalidDecision("Recorded public trial state is invalid.")
    visible = {field: deepcopy(observation[field]) for field in ("game", "status", "map")}
    visible["player"] = {field: deepcopy(player[field]) for field in ("position", "inventory") if field in player}
    visible["objects"] = {field: deepcopy(objects[field]) for field in ("key", "exit") if field in objects}
    gate = objects.get("gate")
    if isinstance(gate, dict):
        visible["objects"]["gate"] = {field: deepcopy(gate[field]) for field in ("position", "open") if field in gate}
    if feedback:
        if "events" not in observation:
            raise InvalidDecision("Recorded trial feedback is missing for comparison.")
        visible["events"] = ["Environment started."] if observation.get("tick") == 0 else deepcopy(observation["events"])
    return visible


def _receipt(fact: dict) -> dict:
    record = fact["record"]
    return {
        **{field: fact[field] for field in ("step", "control", "episode_before", "episode_after", "record_type")},
        "before": _visible_state(record.get("before"), feedback=True),
        "after": _visible_state(record.get("after"), feedback=True),
    }


def validate_assessment(value, records: dict[int, dict], pending_step: int | None) -> dict | None:
    """Bind an assessment to the latest actual completed control and episodes."""
    if pending_step is None:
        if value is not None:
            raise InvalidDecision("Assessment must be null when no completed action awaits assessment.")
        return None
    steps = _ordered_steps(records)
    if type(pending_step) is not int or not steps or pending_step != steps[-1]:
        raise InvalidDecision("Pending assessment must identify the latest completed action.")
    fact = _record(records, pending_step)
    if not isinstance(value, dict) or set(value) != _ASSESSMENT_FIELDS:
        raise InvalidDecision("Assessment needs exactly step, outcome, control, episode_before, and episode_after.")
    if type(value["step"]) is not int or value["step"] != pending_step:
        raise InvalidDecision("Assessment step must match the latest completed action.")
    if not isinstance(value["control"], str) or value["control"] != fact["control"]:
        raise InvalidDecision("Assessment control must match the actual completed control.")
    for field in ("episode_before", "episode_after"):
        if type(value[field]) is not int or value[field] != fact[field]:
            raise InvalidDecision("Assessment episodes must match the actual completed control receipt.")
    return {"step": pending_step, "outcome": _text(value["outcome"], "assessment.outcome", 360), "control": fact["control"], "episode_before": fact["episode_before"], "episode_after": fact["episode_after"]}


def _validate_comparison(value, control: str, refs: list[int], records: dict[int, dict], assessed_steps: set[int] | None) -> dict:
    if not isinstance(value, dict) or set(value) != _COMPARISON_FIELDS:
        raise InvalidDecision("Comparison needs exactly original/repeated steps, original/repeated episodes, and result.")
    for field in ("original_step", "repeated_step", "original_episode", "repeated_episode"):
        if type(value[field]) is not int:
            raise InvalidDecision("Comparison steps and episodes must be integers.")
    original_step, repeated_step = value["original_step"], value["repeated_step"]
    if original_step >= repeated_step or not {original_step, repeated_step} <= set(refs):
        raise InvalidDecision("Comparison must cite ordered original and repeated completed trials in evidence_steps.")
    original, repeated = _record(records, original_step), _record(records, repeated_step)
    if any(fact["record_type"] != "step" or fact["control"] != control for fact in (original, repeated)):
        raise InvalidDecision("Comparison trials must be actual game steps of the hypothesis control.")
    if (
        value["original_episode"] != original["episode_before"]
        or value["repeated_episode"] != repeated["episode_before"]
        or original["episode_before"] >= repeated["episode_before"]
    ):
        raise InvalidDecision("Comparison must match actual trial episodes and span a later episode.")
    baseline = _conditions(original["record"].get("before"))
    if any(not _same_json(_conditions(_record(records, step)["record"].get("before")), baseline) for step in refs):
        raise InvalidDecision("All comparison evidence trials must have identical recorded starting conditions.")
    if assessed_steps is not None:
        assessed = {step for step in assessed_steps if type(step) is int}
        if not {original_step, repeated_step} <= assessed:
            raise InvalidDecision("Both comparison trials must have accepted assessments, including any current valid assessment.")
    if not isinstance(value["result"], str) or value["result"] not in {"same", "different"}:
        raise InvalidDecision("Comparison result must be same or different.")
    actual = "same" if _same_json(_outcome(original["record"].get("after")), _outcome(repeated["record"].get("after"))) else "different"
    if value["result"] != actual:
        raise InvalidDecision("Comparison result disagrees with recorded after-state and feedback, ignoring ticks.")
    return {field: value[field] for field in ("original_step", "repeated_step", "original_episode", "repeated_episode", "result")}


def _validate_hypothesis(note, records: dict[int, dict], assessed_steps: set[int] | None, *, require_comparison: bool, previous=None) -> dict:
    if not isinstance(note, dict) or set(note) != _HYPOTHESIS_FIELDS:
        raise InvalidDecision("Hypothesis needs exactly id, claim, status, evidence_steps, control, and comparison.")
    identifier = _text(note["id"], "hypothesis.id", 32)
    claim = _text(note["claim"], "hypothesis.claim", 240)
    status, control = note["status"], note["control"]
    if not isinstance(status, str) or status not in _STATUSES:
        raise InvalidDecision("Hypothesis status must be untested, supported, contradicted, or uncertain.")
    if not isinstance(control, str) or control not in ACTIONS:
        raise InvalidDecision("Hypothesis control must be one actual game control, excluding reset and finish.")
    if previous is not None:
        if not isinstance(previous, dict) or previous.get("id") != identifier:
            raise InvalidDecision("Previous hypothesis metadata must match the updated ID.")
        if previous.get("control") != control:
            raise InvalidDecision("An existing hypothesis ID must retain its declared control.")
    refs = note["evidence_steps"]
    if not isinstance(refs, list) or len(refs) > 8 or any(type(step) is not int for step in refs):
        raise InvalidDecision("Hypothesis evidence must be a list of at most eight integer completed game steps.")
    if len(set(refs)) != len(refs):
        raise InvalidDecision("Duplicate hypothesis evidence step references are not allowed.")
    for step in refs:
        fact = _record(records, step)
        if fact["record_type"] != "step" or fact["control"] != control:
            raise InvalidDecision("Every evidence reference must be a completed game trial of the declared control.")
    if status in {"supported", "contradicted"} and not refs:
        raise InvalidDecision("Supported or contradicted hypotheses require actual recorded evidence.")
    comparison = note["comparison"]
    if require_comparison and status in {"supported", "contradicted"} and comparison is None:
        raise InvalidDecision("Supported or contradicted reset-retention notes require an assessed cross-episode comparison.")
    if comparison is not None:
        comparison = _validate_comparison(comparison, control, refs, records, assessed_steps)
    return {"id": identifier, "claim": claim, "status": status, "evidence_steps": list(refs), "control": control, "comparison": comparison}


def validate_hypothesis(note, records: dict[int, dict], assessed_steps: set[int], *, require_comparison: bool, previous=None) -> dict:
    """Validate citations and comparison facts without interpreting a claim."""
    if assessed_steps is None:
        raise InvalidDecision("Accepted assessment steps are required for hypothesis validation.")
    return _validate_hypothesis(note, records, assessed_steps, require_comparison=require_comparison, previous=previous)


def comparison_context(records: dict[int, dict], assessed_steps: set[int], pending_step: int | None) -> dict:
    """Expose completed trial facts and comparable pairs, never a next action."""
    steps = _ordered_steps(records)
    facts = [_record(records, step) for step in steps]
    all_trials = [fact for fact in facts if fact["record_type"] == "step"]
    trials = all_trials[-MAX_CONTEXT_TRIALS:]
    assessed = {step for step in assessed_steps if type(step) is int}
    if pending_step is not None and (type(pending_step) is not int or not steps or pending_step != steps[-1]):
        raise InvalidDecision("Pending assessment must identify the latest completed action.")
    eligible = []
    for original, repeated in combinations(all_trials, 2):
        if (
            original["control"] != repeated["control"]
            or original["episode_before"] >= repeated["episode_before"]
            or original["step"] not in assessed
            or (repeated["step"] not in assessed and repeated["step"] != pending_step)
            or not _same_json(_conditions(original["record"].get("before")), _conditions(repeated["record"].get("before")))
        ):
            continue
        result = "same" if _same_json(_outcome(original["record"].get("after")), _outcome(repeated["record"].get("after"))) else "different"
        eligible.append({"original_step": original["step"], "repeated_step": repeated["step"], "original_episode": original["episode_before"], "repeated_episode": repeated["episode_before"], "result": result, "control": original["control"]})
    return {
        "contract_version": COMPARISON_CONTRACT_VERSION,
        "completed_trial_steps": [fact["step"] for fact in all_trials],
        "completed_trials": [_receipt(fact) for fact in trials],
        "eligible_pairs": eligible,
        "latest_control_receipt": _receipt(facts[-1]) if facts else None,
    }


def comparison_receipts(notes, records: dict[int, dict]) -> list[dict]:
    """Record actual comparison receipts; verified=False applies to claims."""
    result = []
    items = notes.values() if isinstance(notes, dict) else notes
    for note in items:
        if note.get("comparison") is None:
            continue
        # The controller already checks accepted assessments. Recheck structural
        # citation facts here without inventing assessment or hypothesis truth.
        checked = _validate_hypothesis(note, records, None, require_comparison=False)
        pair = checked["comparison"]
        original = records[pair["original_step"]]
        repeated = records[pair["repeated_step"]]
        result.append({
            "contract_version": COMPARISON_CONTRACT_VERSION,
            "hypothesis_id": checked["id"], "control": checked["control"],
            "comparison": deepcopy(pair),
            "starting_conditions": _visible_state(original["before"], feedback=False),
            "observed_outcomes": {
                "original": _visible_state(original["after"], feedback=True),
                "repeated": _visible_state(repeated["after"], feedback=True),
            },
            "verified": False,
        })
    return result
