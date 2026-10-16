"""Model-selected control commitments and recorded progress, never an action policy.

The model writes every control in a short plan and chooses every next action.
Continuing consumes its own recorded plan; revising remains available after
feedback. No plan tail executes automatically, and no route, quota, or reset is
provided here. Claims, revision reasons, and research quality remain unverified.
"""

from copy import deepcopy
import json

from .game import ACTIONS
from .nebius import InvalidDecision


RESEARCH_CONTRACT_VERSION = 1
PLAN_CONTROLS = (*ACTIONS, "reset", "finish")
MAX_PLAN_CONTROLS = 4
_PLAN_FIELDS = {"mode", "controls", "revision_reason", "evidence_steps"}
_CONDITION_FIELDS = ("game", "status", "map", "player", "objects")
_SCOPE = (
    "Pending controls are model-written intent, never automatically executed. "
    "Revisions and state/control attempts count completed game/reset actions only; "
    "starting conditions exclude ticks and feedback. Research quality is unverified."
)


def _text(value, name: str, maximum: int, *, empty: bool = False) -> str:
    if not isinstance(value, str):
        raise InvalidDecision(f"{name} must be a string.")
    value = value.strip()
    if len(value) > maximum or not value and not empty:
        raise InvalidDecision(f"{name} must contain {'0' if empty else '1'}-{maximum} trimmed characters.")
    return value


def _keys(records: dict[int, dict]) -> list[int]:
    if not isinstance(records, dict) or any(type(step) is not int or step < 1 for step in records):
        raise InvalidDecision("Research records must be keyed by positive integer completed steps.")
    return sorted(records)


def _allowed(allowed) -> list[str]:
    if not isinstance(allowed, (list, tuple)) or not allowed:
        raise InvalidDecision("Research controls need a nonempty current allowed-control list.")
    if any(not isinstance(control, str) or control not in PLAN_CONTROLS for control in allowed):
        raise InvalidDecision("Research allowed controls contain an unknown control.")
    if len(set(allowed)) != len(allowed):
        raise InvalidDecision("Research allowed controls must be unique.")
    return list(allowed)


def _normalize(value, *, action: str, experiment_id: str, completed: set[int], pending: dict | None, allowed: list[str]) -> dict:
    if not isinstance(value, dict) or set(value) != _PLAN_FIELDS:
        raise InvalidDecision("plan needs exactly mode, controls, revision_reason, and evidence_steps.")
    identifier = _text(experiment_id, "experiment_id", 32)
    if not isinstance(action, str) or action not in allowed:
        raise InvalidDecision("The plan's selected action must be allowed in the current state.")
    mode = value["mode"]
    if not isinstance(mode, str) or mode not in {"new", "continue", "revise"}:
        raise InvalidDecision("plan.mode must be new, continue, or revise.")
    controls = value["controls"]
    if not isinstance(controls, list) or not 1 <= len(controls) <= MAX_PLAN_CONTROLS:
        raise InvalidDecision("plan.controls must contain one to four model-selected controls.")
    if any(not isinstance(control, str) or control not in PLAN_CONTROLS for control in controls):
        raise InvalidDecision("Each planned control must be a game control, reset, or finish.")
    if controls[0] != action:
        raise InvalidDecision("The first planned control must equal the actual selected action.")
    if "finish" in controls[:-1]:
        raise InvalidDecision("No controls may follow finish in a plan.")
    reason = _text(value["revision_reason"], "plan.revision_reason", 240, empty=True)
    evidence = value["evidence_steps"]
    if not isinstance(evidence, list) or len(evidence) > 8 or any(type(step) is not int or step not in completed for step in evidence):
        raise InvalidDecision("Plan evidence must cite at most eight existing completed action steps, never future steps.")
    if len(set(evidence)) != len(evidence):
        raise InvalidDecision("Plan evidence steps must be unique.")
    if mode == "new":
        if pending is not None:
            raise InvalidDecision("A recorded plan tail remains; continue it or explicitly revise it using actual evidence.")
        if reason or evidence:
            raise InvalidDecision("A new plan requires empty revision_reason and evidence_steps.")
    elif mode == "continue":
        if pending is None:
            raise InvalidDecision("No recorded plan tail remains to continue; choose a new plan.")
        if identifier != pending["experiment_id"] or controls != pending["controls"]:
            raise InvalidDecision("Continuing must retain the exact recorded remaining controls and experiment ID.")
        if reason or evidence:
            raise InvalidDecision("A continued plan requires empty revision_reason and evidence_steps.")
    elif not reason or not evidence:
        raise InvalidDecision("A revised plan requires a nonempty revision_reason and actual completed evidence steps.")
    return {"mode": mode, "controls": list(controls), "revision_reason": reason, "evidence_steps": list(evidence)}


def _history(records: dict[int, dict]) -> tuple[dict | None, int]:
    """Reconstruct only accepted plans attached to actual completed actions."""
    pending = None
    revisions = 0
    completed = set()
    for step in _keys(records):
        record = records[step]
        if not isinstance(record, dict) or type(record.get("step")) is not int or record["step"] != step:
            raise InvalidDecision("Recorded research action metadata is invalid.")
        decision = record.get("decision")
        if not isinstance(decision, dict):
            raise InvalidDecision("Recorded research actions need their actual accepted decisions.")
        action, kind = decision.get("action"), record.get("type")
        if kind == "step":
            before = record.get("episode_before", record.get("episode"))
            after = record.get("episode_after", record.get("episode"))
            if not isinstance(action, str) or action not in ACTIONS or before != after:
                raise InvalidDecision("Recorded research game-control metadata is invalid.")
            if "episode" in record and (type(record["episode"]) is not int or record["episode"] != before):
                raise InvalidDecision("Recorded research game-control episode is inconsistent.")
        elif kind == "reset":
            before, after = record.get("episode_before"), record.get("episode_after")
            if action != "reset":
                raise InvalidDecision("Recorded research reset metadata is invalid.")
        else:
            raise InvalidDecision("Research history contains only actual completed game/reset steps.")
        if type(before) is not int or type(after) is not int or before < 0 or after < 0 or kind == "reset" and after != before + 1:
            raise InvalidDecision("Recorded research episode transitions are invalid.")
        identifier = _text(decision.get("experiment_id"), "experiment_id", 32)
        plan = _normalize(decision.get("plan"), action=action, experiment_id=identifier, completed=completed, pending=pending, allowed=list(PLAN_CONTROLS))
        revisions += plan["mode"] == "revise"
        tail = plan["controls"][1:]
        pending = {"source_step": step, "experiment_id": identifier, "controls": tail} if tail else None
        completed.add(step)
    return pending, revisions


def validate_plan(value, *, action: str, experiment_id: str, records: dict[int, dict], allowed: list[str]) -> dict:
    """Check one voluntary plan update without changing history or choosing an action."""
    pending, _ = _history(records)
    return _normalize(value, action=action, experiment_id=experiment_id, completed=set(records), pending=pending, allowed=_allowed(allowed))


def _conditions(observation: dict) -> str:
    if not isinstance(observation, dict) or any(field not in observation for field in _CONDITION_FIELDS):
        raise InvalidDecision("Recorded public state is incomplete for research attempt counts.")
    try:
        return json.dumps({field: observation[field] for field in _CONDITION_FIELDS}, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError):
        raise InvalidDecision("Research starting conditions must contain finite JSON facts.") from None


def plan_context(records: dict[int, dict], observation: dict, *, finished: bool = False) -> dict:
    """Expose a real retained tail and current-state attempt counts, never a next choice."""
    if type(finished) is not bool:
        raise InvalidDecision("Research session_finished must be a boolean.")
    pending, revisions = _history(records)
    current = _conditions(observation)
    counts = {control: 0 for control in (*ACTIONS, "reset")}
    for step in _keys(records):
        record = records[step]
        if _conditions(record.get("before")) == current:
            counts[record["decision"]["action"]] += 1
    return {
        "contract_version": RESEARCH_CONTRACT_VERSION, "session_finished": finished,
        "pending_plan": None if finished else deepcopy(pending),
        "completed_steps": _keys(records), "plan_revisions": revisions,
        "current_state_action_attempts": counts, "scope": _SCOPE, "claims_verified": False,
    }


def research_context(records: dict[int, dict], observation: dict, *, finished: bool = False) -> dict:
    """Alias used by recording/request fields; facts are identical to plan_context."""
    return plan_context(records, observation, finished=finished)


def _control_schema(heads: list[str]) -> dict:
    # Complete finite-length alternatives keep finish last without prescribing
    # any control sequence. Subsequent controls are entirely model-selected.
    variants = []
    for length in range(1, MAX_PLAN_CONTROLS + 1):
        first = heads if length == 1 else [control for control in heads if control != "finish"]
        if not first:
            continue
        prefix = [{"type": "string", "enum": first}]
        for index in range(1, length):
            prefix.append({"type": "string", "enum": list(PLAN_CONTROLS if index == length - 1 else PLAN_CONTROLS[:-1])})
        variants.append({"type": "array", "minItems": length, "maxItems": length,
                         "items": {"type": "string", "enum": list(PLAN_CONTROLS)}, "prefixItems": prefix})
    return {"type": "array", "minItems": 1, "maxItems": MAX_PLAN_CONTROLS,
            "items": {"type": "string", "enum": list(PLAN_CONTROLS)}, "anyOf": variants}


def plan_schema(allowed: list[str], context: dict, *, action: str | None = None) -> dict:
    """Shape voluntary updates; continuation follows only a model's prior commitment."""
    allowed = _allowed(allowed)
    if not isinstance(context, dict) or type(context.get("contract_version")) is not int or context["contract_version"] != RESEARCH_CONTRACT_VERSION:
        raise InvalidDecision("Plan schema requires reconstructed research contract 1 facts.")
    if context.get("session_finished") is not False:
        raise InvalidDecision("No plan may be proposed after a recorded session finish.")
    completed = context.get("completed_steps")
    if not isinstance(completed, list) or any(type(step) is not int or step < 1 for step in completed) or completed != sorted(set(completed)):
        raise InvalidDecision("Plan schema requires unique actual completed step references.")
    if action is not None and (not isinstance(action, str) or action not in allowed):
        raise InvalidDecision("Plan schema action must be allowed now.")
    heads = [action] if action is not None else allowed
    pending = context.get("pending_plan")
    if pending is not None:
        if (
            not isinstance(pending, dict) or set(pending) != {"source_step", "experiment_id", "controls"}
            or type(pending["source_step"]) is not int or not completed or pending["source_step"] != completed[-1]
            or not isinstance(pending["controls"], list) or not 1 <= len(pending["controls"]) < MAX_PLAN_CONTROLS
            or any(not isinstance(control, str) or control not in PLAN_CONTROLS for control in pending["controls"])
            or "finish" in pending["controls"][:-1]
        ):
            raise InvalidDecision("Plan schema pending controls are invalid.")
        _text(pending["experiment_id"], "pending_plan.experiment_id", 32)
    base = {
        "type": "object", "properties": {
            "mode": {"type": "string", "enum": ["new", "continue", "revise"]},
            "controls": _control_schema(heads),
            "revision_reason": {"type": "string", "maxLength": 240},
            "evidence_steps": {"type": "array", "maxItems": 8, "uniqueItems": True, "items": {"type": "integer"}},
        }, "required": ["mode", "controls", "revision_reason", "evidence_steps"], "additionalProperties": False,
    }
    if completed:
        base["properties"]["evidence_steps"]["items"]["enum"] = completed
    else:
        base["properties"]["evidence_steps"]["maxItems"] = 0
    alternatives = []
    if pending is None:
        new = deepcopy(base)
        new["properties"]["mode"]["enum"] = ["new"]
        new["properties"]["revision_reason"]["enum"] = [""]
        new["properties"]["evidence_steps"]["maxItems"] = 0
        alternatives.append(new)
    elif pending["controls"][0] in heads:
        continuation = deepcopy(base)
        continuation["properties"]["mode"]["enum"] = ["continue"]
        continuation["properties"]["controls"]["enum"] = [deepcopy(pending["controls"])]
        continuation["properties"]["revision_reason"]["enum"] = [""]
        continuation["properties"]["evidence_steps"]["maxItems"] = 0
        alternatives.append(continuation)
    if completed:
        revision = deepcopy(base)
        revision["properties"]["mode"]["enum"] = ["revise"]
        revision["properties"]["revision_reason"]["minLength"] = 1
        revision["properties"]["evidence_steps"]["minItems"] = 1
        alternatives.append(revision)
    base["anyOf"] = alternatives
    return base
