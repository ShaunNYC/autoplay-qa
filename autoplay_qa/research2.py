"""Explicit question switches and experiment-local learning bookkeeping.

Every control, question, role and explanation remains model-selected. Retarget
preserves a model's remaining controls while explicitly changing the question
they serve. These helpers count executed facts; they neither run a plan nor
verify the meaning or truth of a learned claim.
"""

from copy import deepcopy
from dataclasses import asdict, is_dataclass
from itertools import combinations

from .comparisons import _conditions as _comparison_conditions, _outcome, _record, _same_json
from .game import ACTIONS
from .nebius import InvalidDecision
from .research import MAX_PLAN_CONTROLS, PLAN_CONTROLS, _allowed, _conditions, _control_schema, _keys, _text


RESEARCH_CONTRACT_VERSION = 2
_PLAN_FIELDS = {"mode", "controls", "revision_reason", "evidence_steps"}
_PLAN_MODES = ("new", "continue", "revise", "retarget")
_COMPARISON_FIELDS = ("original_step", "repeated_step", "original_episode", "repeated_episode", "result")
_STATUSES = {"untested", "uncertain", "supported", "contradicted"}
_FAMILIES = ("movement", "interaction", "wait")
_SCOPE = (
    "Pending controls are model-written intent, never automatically executed. "
    "Revisions, scope switches and state/control attempts count completed game/reset actions only; "
    "starting conditions exclude ticks and feedback. Research quality is unverified."
)
_PROGRESS_SCOPE = (
    "Executed actions grouped by declared stable question and target. Eligible pairs use assessed "
    "trial-role actions with identical visible starting conditions in different episodes. Note counts "
    "describe citations, not truthful claims, intentionality or completed research."
)


def _normalize(value, *, action: str, experiment_id: str, completed: set[int], pending: dict | None, allowed: list[str]) -> dict:
    if not isinstance(value, dict) or set(value) != _PLAN_FIELDS:
        raise InvalidDecision("plan needs exactly mode, controls, revision_reason, and evidence_steps.")
    identifier = _text(experiment_id, "experiment_id", 32)
    if not isinstance(action, str) or action not in allowed:
        raise InvalidDecision("The plan's selected action must be allowed in the current state.")
    mode = value["mode"]
    if not isinstance(mode, str) or mode not in _PLAN_MODES:
        raise InvalidDecision("plan.mode must be new, continue, revise, or retarget.")
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
            raise InvalidDecision("A recorded plan tail remains; continue, retarget, or explicitly revise it using actual evidence.")
        if reason or evidence:
            raise InvalidDecision("A new plan requires empty revision_reason and evidence_steps.")
    elif mode == "continue":
        if pending is None:
            raise InvalidDecision("No recorded plan tail remains to continue; choose a new plan.")
        if identifier != pending["experiment_id"] or controls != pending["controls"]:
            raise InvalidDecision("Continuing must retain the exact recorded remaining controls and experiment ID.")
        if reason or evidence:
            raise InvalidDecision("A continued plan requires empty revision_reason and evidence_steps.")
    elif mode == "retarget":
        if pending is None:
            raise InvalidDecision("Retarget requires an actual recorded remaining plan tail.")
        if controls != pending["controls"]:
            raise InvalidDecision("Retarget must preserve the exact recorded remaining controls.")
        if identifier == pending["experiment_id"]:
            raise InvalidDecision("Retarget must select a different experiment ID.")
        if not reason or not evidence:
            raise InvalidDecision("Retarget requires a nonempty revision_reason and actual completed evidence steps.")
    elif not reason or not evidence:
        raise InvalidDecision("A revised plan requires a nonempty revision_reason and actual completed evidence steps.")
    return {"mode": mode, "controls": list(controls), "revision_reason": reason, "evidence_steps": list(evidence)}


def _history(records: dict[int, dict]) -> tuple[dict | None, int, int]:
    pending = None
    revisions = switches = 0
    completed = set()
    for step in _keys(records):
        fact = _record(records, step)
        decision = fact["record"]["decision"]
        identifier = _text(decision.get("experiment_id"), "experiment_id", 32)
        plan = _normalize(decision.get("plan"), action=fact["control"], experiment_id=identifier,
                          completed=completed, pending=pending, allowed=list(PLAN_CONTROLS))
        revisions += plan["mode"] == "revise"
        switches += plan["mode"] == "retarget"
        tail = plan["controls"][1:]
        pending = {"source_step": step, "experiment_id": identifier, "controls": tail} if tail else None
        completed.add(step)
    return pending, revisions, switches


def validate_plan(value, *, action: str, experiment_id: str, records: dict[int, dict], allowed: list[str]) -> dict:
    """Validate a proposal atomically; only a later executed row consumes it."""
    pending, _, _ = _history(records)
    return _normalize(value, action=action, experiment_id=experiment_id, completed=set(records),
                      pending=pending, allowed=_allowed(allowed))


def plan_context(records: dict[int, dict], observation: dict, *, finished: bool = False) -> dict:
    if type(finished) is not bool:
        raise InvalidDecision("Research session_finished must be a boolean.")
    pending, revisions, switches = _history(records)
    current = _conditions(observation)
    counts = {control: 0 for control in (*ACTIONS, "reset")}
    for step in _keys(records):
        record = records[step]
        if _conditions(record.get("before")) == current:
            counts[record["decision"]["action"]] += 1
    return {
        "contract_version": RESEARCH_CONTRACT_VERSION, "session_finished": finished,
        "pending_plan": None if finished else deepcopy(pending),
        "completed_steps": _keys(records), "plan_revisions": revisions, "scope_switches": switches,
        "current_state_action_attempts": counts, "scope": _SCOPE, "claims_verified": False,
    }


def research_context(records: dict[int, dict], observation: dict, *, finished: bool = False) -> dict:
    return plan_context(records, observation, finished=finished)


def plan_schema(allowed: list[str], context: dict, *, action: str | None = None) -> dict:
    """Complete choices encode past commitments without supplying a next action."""
    allowed = _allowed(allowed)
    if not isinstance(context, dict) or type(context.get("contract_version")) is not int or context["contract_version"] != RESEARCH_CONTRACT_VERSION:
        raise InvalidDecision("Plan schema requires reconstructed research contract 2 facts.")
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
            "mode": {"type": "string", "enum": list(_PLAN_MODES)},
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
        if pending is not None and pending["controls"][0] in heads:
            switch = deepcopy(revision)
            switch["properties"]["mode"]["enum"] = ["retarget"]
            switch["properties"]["controls"]["enum"] = [deepcopy(pending["controls"])]
            alternatives.append(switch)
    base["anyOf"] = alternatives
    return base


def experiment_goals(records: dict[int, dict]) -> dict[str, str]:
    """Return each question's first executed goal; reject silent goal changes."""
    goals = {}
    for step in _keys(records):
        decision = _record(records, step)["record"]["decision"]
        identifier = _text(decision.get("experiment_id"), "experiment_id", 32)
        goal = _text(decision.get("experiment_goal"), "experiment_goal", 240)
        if identifier in goals and goal != goals[identifier]:
            raise InvalidDecision("A reused experiment ID must retain its original experiment_goal; choose another ID for another question.")
        goals.setdefault(identifier, goal)
    return goals


def validate_goal(identifier: str, goal: str, records: dict[int, dict]) -> str:
    """Check a chosen goal without committing an unexecuted scope."""
    identifier = _text(identifier, "experiment_id", 32)
    goal = _text(goal, "experiment_goal", 240)
    original = experiment_goals(records).get(identifier)
    if original is not None and original != goal:
        raise InvalidDecision("A reused experiment ID must retain its original experiment_goal; choose another ID for another question.")
    return goal


def _declared(records: dict[int, dict]) -> dict[str, dict]:
    goals = experiment_goals(records)
    declared = {}
    for step in _keys(records):
        fact = _record(records, step)
        decision = fact["record"]["decision"]
        identifier = _text(decision.get("experiment_id"), "experiment_id", 32)
        control, role = decision.get("experiment_control"), decision.get("action_role")
        if not isinstance(control, str) or control not in ACTIONS:
            raise InvalidDecision("Recorded experiment target must be one game control.")
        if not isinstance(role, str) or role not in {"trial", "setup"}:
            raise InvalidDecision("Completed game/reset rows must declare trial or setup roles.")
        if fact["record_type"] == "reset" and role != "setup" or role == "trial" and fact["control"] != control:
            raise InvalidDecision("Recorded trial/setup role disagrees with the actual control.")
        if identifier in declared and declared[identifier]["control"] != control:
            raise InvalidDecision("A reused experiment ID must retain its declared experiment_control.")
        entry = declared.setdefault(identifier, {"id": identifier, "goal": goals[identifier], "control": control,
                                                "trial_steps": [], "setup_steps": []})
        entry[f"{role}_steps"].append(step)
    return declared


def _assessed(records: dict[int, dict], assessed_steps) -> set[int]:
    if not isinstance(assessed_steps, (set, frozenset, list, tuple)):
        raise InvalidDecision("Assessed steps must identify actual completed rows.")
    values = list(assessed_steps)
    if any(type(step) is not int or step not in records for step in values) or len(set(values)) != len(values):
        raise InvalidDecision("Assessed steps must be unique existing completed step numbers.")
    return set(values)


def _notes(notes, records: dict[int, dict]) -> list[dict]:
    if isinstance(notes, dict):
        values = list(notes.values())
    elif isinstance(notes, (list, tuple)):
        values = list(notes)
    else:
        raise InvalidDecision("Learning notes must be a list or an ID-keyed mapping.")
    if len(values) > 8:
        raise InvalidDecision("Session learning permits at most eight note IDs.")
    normalized = []
    seen = set()
    for value in values:
        if is_dataclass(value) and not isinstance(value, type):
            value = asdict(value)
        if not isinstance(value, dict):
            raise InvalidDecision("Each learning note must contain its actual control and citations.")
        identifier = _text(value.get("id"), "hypothesis.id", 32)
        if identifier in seen:
            raise InvalidDecision("Duplicate hypothesis IDs are not allowed.")
        seen.add(identifier)
        control, status, refs = value.get("control"), value.get("status"), value.get("evidence_steps")
        if not isinstance(control, str) or control not in ACTIONS or not isinstance(status, str) or status not in _STATUSES:
            raise InvalidDecision("Learning notes need a declared game control and valid tentative status.")
        if isinstance(refs, tuple):
            refs = list(refs)
        if not isinstance(refs, list) or len(refs) > 8 or any(type(step) is not int or step not in records for step in refs) or len(set(refs)) != len(refs):
            raise InvalidDecision("Note evidence must contain unique actual completed step references.")
        normalized.append({"id": identifier, "control": control, "status": status,
                           "evidence_steps": list(refs), "comparison": deepcopy(value.get("comparison"))})
    return normalized


def _family(control: str) -> str:
    return "interaction" if control == "interact" else "wait" if control == "wait" else "movement"


def scope_progress(records: dict[int, dict], assessed_steps, notes) -> dict:
    """Show actual question-local trial, pair and citation counts, never a policy."""
    declared = _declared(records)
    assessed = _assessed(records, assessed_steps)
    updates = _notes(notes, records)
    experiments = []
    family_entries = {family: [] for family in _FAMILIES}
    for entry in declared.values():
        trials, setup = entry["trial_steps"], entry["setup_steps"]
        assessed_trials = [step for step in trials if step in assessed]
        pairs = []
        for original, repeated in combinations(assessed_trials, 2):
            first, second = _record(records, original), _record(records, repeated)
            if first["episode_before"] >= second["episode_before"] or not _same_json(
                _comparison_conditions(first["record"].get("before")),
                _comparison_conditions(second["record"].get("before")),
            ):
                continue
            result = "same" if _same_json(_outcome(first["record"].get("after")), _outcome(second["record"].get("after"))) else "different"
            pairs.append({"original_step": original, "repeated_step": repeated,
                          "original_episode": first["episode_before"], "repeated_episode": second["episode_before"],
                          "result": result, "control": entry["control"]})
        cited = [note for note in updates if note["status"] != "untested" and note["control"] == entry["control"]
                 and set(note["evidence_steps"]).intersection(assessed_trials)]
        paired = [note for note in cited if any(
            _same_json(note["comparison"], {field: pair[field] for field in _COMPARISON_FIELDS})
            and {pair["original_step"], pair["repeated_step"]}.issubset(note["evidence_steps"])
            for pair in pairs
        )]
        note_ids = sorted(note["id"] for note in cited)
        comparison_ids = sorted(note["id"] for note in paired)
        progress = {
            "id": entry["id"], "goal": entry["goal"], "control": entry["control"],
            "trial_steps": list(trials), "setup_steps": list(setup), "assessed_trial_steps": assessed_trials,
            "trial_count": len(trials), "setup_count": len(setup), "assessed_trial_count": len(assessed_trials),
            "eligible_pairs": pairs, "eligible_pair_count": len(pairs),
            "note_ids": note_ids, "note_count": len(note_ids),
            "comparison_note_ids": comparison_ids, "comparison_note_count": len(comparison_ids),
            "unnoted_first_trial_step": assessed_trials[0] if assessed_trials and not note_ids else None,
        }
        experiments.append(progress)
        family_entries[_family(entry["control"])].append(progress)
    families = {}
    for family, entries in family_entries.items():
        families[family] = {
            "experiment_count": len(entries),
            "trial_count": sum(entry["trial_count"] for entry in entries),
            "setup_count": sum(entry["setup_count"] for entry in entries),
            "assessed_trial_count": sum(entry["assessed_trial_count"] for entry in entries),
            "eligible_pair_count": sum(entry["eligible_pair_count"] for entry in entries),
            "note_count": len({identifier for entry in entries for identifier in entry["note_ids"]}),
            "comparison_note_count": len({identifier for entry in entries for identifier in entry["comparison_note_ids"]}),
        }
    return {"contract_version": RESEARCH_CONTRACT_VERSION, "scope": _PROGRESS_SCOPE, "claims_verified": False,
            "experiments": experiments, "families": families}


def unnoted_trial_steps(records: dict[int, dict], assessed_steps, notes) -> list[int]:
    """List experiments' first assessed trials lacking their own cited note."""
    progress = scope_progress(records, assessed_steps, notes)
    return sorted(entry["unnoted_first_trial_step"] for entry in progress["experiments"]
                  if entry["unnoted_first_trial_step"] is not None)
