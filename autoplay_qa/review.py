"""Read-only progress checks from recordings, without another inference call.

These checks describe recorded execution and bookkeeping. They cannot establish
hypothesis accuracy, deliberate probing, or that memory caused a model's action.
"""

from collections import Counter
from copy import deepcopy
import json
import math
from pathlib import Path
from statistics import median

from . import session
from .exploration import NONCOMPARISON_LABELS, Coverage, outcome_view, state_fingerprint
from .game import DIRECTIONS


def _hypotheses(record: dict) -> dict:
    items = record.get("hypotheses", [])
    if not isinstance(items, list):
        return {}
    return {
        item["id"]: item for item in items
        if isinstance(item, dict) and isinstance(item.get("id"), str)
    }


def _evidence(note: dict, completed: set[int]) -> set[int]:
    refs = note.get("evidence_steps", [])
    if not isinstance(refs, list) or any(type(ref) is not int or ref not in completed for ref in refs):
        return set()
    return set(refs)


def _reset_learning(records: list[dict]) -> tuple[dict, dict]:
    """Require retained content and evidence for both trials, not just matching IDs."""
    notes = {}
    notes_completed = set()
    actions = {}
    requests = {}
    assessments = set()
    resets = []
    current_reset = None
    for record in records:
        kind = record.get("type")
        if kind == "memory":
            notes = _hypotheses(record)
            notes_completed = set(actions)
            assessment = record.get("assessment")
            step = assessment.get("step") if isinstance(assessment, dict) else None
            if type(step) is not int or step not in actions or not isinstance(assessment.get("outcome"), str) or not assessment["outcome"].strip():
                continue
            if assessment["outcome"].strip().lower().strip(" .!") in NONCOMPARISON_LABELS:
                continue
            assessments.add(step)
            if current_reset is None or step <= current_reset["reset_step"] or actions[step]["kind"] != "step":
                continue
            current_reset["post_reset_assessed_steps"].add(step)
            trial = actions[step]
            supplied = requests.get(trial["request_number"], {})
            for old_step, previous in actions.items():
                if old_step not in assessments or previous["kind"] != "step" or previous["episode"] >= trial["episode"] or previous["attempt"] != trial["attempt"]:
                    continue
                for identifier, original in current_reset["notes"].items():
                    update, context_note = notes.get(identifier, {}), supplied.get(identifier, {})
                    if (
                        identifier in current_reset["retained_hypothesis_ids"]
                        and original.get("status") in {"supported", "contradicted", "uncertain"}
                        and old_step in current_reset["evidence_before_reset"].get(identifier, set())
                        and context_note.get("claim") == original.get("claim")
                        and old_step in _evidence(context_note, {number for number in actions if number < step})
                        and update.get("claim") == original.get("claim")
                        and update.get("status") in {"supported", "contradicted", "uncertain"}
                        and {old_step, step} <= _evidence(update, set(actions))
                    ):
                        current_reset["retest_pairs"].append({
                            "before_step": old_step, "after_step": step,
                            "action": trial["action"], "hypothesis_id": identifier,
                        })
        elif kind == "request":
            # Replay already checked the schema-2 request and its observation.
            context = json.loads(record["messages"][1]["content"])
            supplied = _hypotheses(context)
            requests[record["request_number"]] = supplied
            if current_reset is not None:
                retained = {
                    identifier for identifier, original in current_reset["notes"].items()
                    if supplied.get(identifier) == original
                }
                if retained:
                    current_reset["requests_with_retained_memory"] += 1
                    current_reset["retained_hypothesis_ids"].update(retained)
        elif kind in {"step", "reset"}:
            step = record["step"]
            action = record["decision"]["action"]
            actions[step] = {
                "kind": kind, "episode": record.get("episode", record.get("episode_before", 0)),
                "request_number": record.get("request_number"), "action": action,
                "attempt": (state_fingerprint(record["before"]), action),
            }
            if kind == "reset":
                current_reset = {
                    "reset_step": step, "episode_after": record["episode_after"],
                    "notes": notes.copy(), "requests_with_retained_memory": 0,
                    "evidence_before_reset": {identifier: _evidence(note, notes_completed) for identifier, note in notes.items()},
                    "retained_hypothesis_ids": set(), "post_reset_assessed_steps": set(),
                    "retest_pairs": [],
                }
                resets.append(current_reset)
    completed = set(actions)
    learning = {
        "available": True, "retained_hypotheses": len(notes),
        "evidence_linked_hypotheses": sum(
            bool(_evidence(note, notes_completed)) and note.get("status") in {"supported", "contradicted", "uncertain"}
            for note in notes.values()
        ),
        "assessed_steps": sorted(assessments),
        "unassessed_final_step": max(completed) if completed and max(completed) not in assessments else None,
    }
    details = [{
        "reset_step": reset["reset_step"], "episode_after": reset["episode_after"],
        "hypotheses_before_reset": len(reset["notes"]),
        "requests_with_retained_memory": reset["requests_with_retained_memory"],
        "retained_hypothesis_ids": sorted(reset["retained_hypothesis_ids"]),
        "post_reset_assessed_steps": sorted(reset["post_reset_assessed_steps"]),
        "retest_pairs": reset["retest_pairs"],
    } for reset in resets]
    reset_learning = {
        "available": True,
        "memory_supplied_after_reset": any(reset["requests_with_retained_memory"] for reset in resets),
        "retained_hypothesis_ids": sorted(set().union(*(reset["retained_hypothesis_ids"] for reset in resets))),
        "post_reset_assessed_steps": sorted(set().union(*(reset["post_reset_assessed_steps"] for reset in resets))),
        "retest_pairs": [pair for reset in resets for pair in reset["retest_pairs"]],
        "behavioral_retest_observed": any(reset["retest_pairs"] for reset in resets),
        "resets": details,
    }
    return learning, reset_learning


def _has_comparison_contract(records: list[dict]) -> bool:
    header = records[0]
    if "comparison_contract" not in header:
        return False
    if type(header["comparison_contract"]) is not int or header["comparison_contract"] != 1 or header.get("schema_version") != 2:
        raise ValueError("Review supports comparison contract version 1 on schema-2 exploration only.")
    return True


def _has_reflection_contract(records: list[dict]) -> bool:
    header = records[0]
    if "reflection_contract" not in header:
        return False
    if type(header["reflection_contract"]) is not int or header["reflection_contract"] not in {1, 2, 3, 4} or not _has_comparison_contract(records):
        raise ValueError("Review supports reflection contract versions 1, 2, 3 or 4 with comparison contract 1 only.")
    return True


def _bound_reset_learning(records: list[dict]) -> tuple[dict, dict]:
    """Keep legacy retention guards, then require explicit bindings in new logs."""
    learning, retention = _reset_learning(records)
    if not _has_comparison_contract(records):
        return learning, retention
    declared = set()
    for record in records:
        if record.get("type") != "memory":
            continue
        assessment = record.get("assessment")
        assessed_step = assessment.get("step") if isinstance(assessment, dict) else None
        for identifier, note in _hypotheses(record).items():
            pair = note.get("comparison")
            if isinstance(pair, dict) and pair.get("repeated_step") == assessed_step:
                declared.add((identifier, pair.get("original_step"), pair.get("repeated_step"), note.get("control")))
    for reset in retention["resets"]:
        reset["retest_pairs"] = [
            pair for pair in reset["retest_pairs"]
            if (pair["hypothesis_id"], pair["before_step"], pair["after_step"], pair["action"]) in declared
        ]
    retention["retest_pairs"] = [pair for reset in retention["resets"] for pair in reset["retest_pairs"]]
    retention["behavioral_retest_observed"] = bool(retention["retest_pairs"])
    return learning, retention


def _lean_learning(records: list[dict]) -> tuple[dict, dict, dict]:
    """Describe host-computed lean evidence without requiring model assessments."""
    from .lean import software_facts

    actions = [record for record in records if record.get("type") in {"step", "reset"}]
    completed = {record["step"] for record in actions}
    memories = [record for record in records if record.get("type") == "memory"]
    latest = memories[-1] if memories else {}
    notes = _hypotheses(latest)
    completed_at_memory = {
        record["step"] for record in actions
        if record.get("request_number", 0) < latest.get("request_number", 0)
    }
    linked = sum(
        bool(_evidence(note, completed_at_memory))
        for note in notes.values()
    )
    facts = software_facts(actions, history_size=None)
    pairs = facts["cross_episode_same_start_pairs"]

    requests_after_reset = []
    retained_ids = set()
    current_notes = {}
    current_reset = None
    reset_details = []
    for record in records:
        kind = record.get("type")
        if kind == "memory":
            current_notes = _hypotheses(record)
        elif kind == "reset":
            current_reset = {
                "reset_step": record["step"],
                "episode_after": record["episode_after"],
                "notes_before_reset": deepcopy(current_notes),
                "requests_with_retained_memory": 0,
                "retained_hypothesis_ids": set(),
            }
            reset_details.append(current_reset)
        elif kind == "request" and current_reset is not None and record.get("episode") == current_reset["episode_after"]:
            context = json.loads(record["messages"][1]["content"])
            supplied = {
                item["id"]: item for item in context.get("notes", [])
                if isinstance(item, dict) and isinstance(item.get("id"), str)
            }
            retained = {
                identifier for identifier, note in current_reset["notes_before_reset"].items()
                if supplied.get(identifier) == note
            }
            if retained:
                requests_after_reset.append(record["request_number"])
                retained_ids.update(retained)
                current_reset["requests_with_retained_memory"] += 1
                current_reset["retained_hypothesis_ids"].update(retained)

    retest_pairs = [
        {
            "before_step": pair["original_step"],
            "after_step": pair["repeated_step"],
            "action": pair["action"],
            "result": pair["result"],
            "source": "software_same_start_comparison",
        }
        for pair in pairs
    ]
    learning = {
        "available": True,
        "retained_hypotheses": len(notes),
        "evidence_linked_hypotheses": linked,
        "assessed_steps": sorted(completed),
        "unassessed_final_step": None,
        "assessment_source": "software_before_after_facts",
    }
    reset_learning = {
        "available": True,
        "memory_supplied_after_reset": bool(requests_after_reset),
        "retained_hypothesis_ids": sorted(retained_ids),
        "post_reset_assessed_steps": sorted({pair["repeated_step"] for pair in pairs}),
        "retest_pairs": retest_pairs,
        "behavioral_retest_observed": bool(retest_pairs),
        "resets": [
            {
                "reset_step": reset["reset_step"],
                "episode_after": reset["episode_after"],
                "hypotheses_before_reset": len(reset["notes_before_reset"]),
                "requests_with_retained_memory": reset["requests_with_retained_memory"],
                "retained_hypothesis_ids": sorted(reset["retained_hypothesis_ids"]),
                "software_comparisons_after_reset": sum(
                    pair["repeated_episode"] >= reset["episode_after"] for pair in pairs
                ),
            }
            for reset in reset_details
        ],
        "source": "software_same_start_comparisons",
    }
    return learning, reset_learning, {
        **facts,
        "facts_validated_by_replay": True,
        "claims_verified": False,
        "note": "Before/after facets and same-start pairs are software-derived. Model notes remain tentative.",
    }


def _comparison_facts(records: list[dict]) -> dict:
    """Facts about typed bindings; natural-language learning remains unverified."""
    if records[0].get("reflection_contract") == 4:
        from .measurements import comparison_receipts
    else:
        from .comparisons import comparison_receipts

    actions = {record["step"]: record for record in records if record.get("type") in {"step", "reset"}}
    memories = [record for record in records if record.get("type") == "memory"]
    notes = _hypotheses(memories[-1]) if memories else {}
    receipts = comparison_receipts(notes, actions)
    controls = {receipt["control"] for receipt in receipts}
    return {
        "version": 1, "bindings_validated_against_execution": True,
        "validated_comparisons": [{
            "hypothesis_id": receipt["hypothesis_id"], "control": receipt["control"],
            **receipt["comparison"],
        } for receipt in receipts],
        "control_families": {"movement": bool(controls.intersection(DIRECTIONS)), "interaction": "interact" in controls},
        "learning_claims_verified": False,
    }


def _reflection_facts(records: list[dict]) -> dict:
    """Typed forecasts and actual outcomes; free prose remains model text."""
    version = records[0]["reflection_contract"]
    if version == 4:
        from .measurements import experiment_context, reflection_receipt
    elif version == 3:
        from .grounding import experiment_context, reflection_receipt
    elif version == 2:
        from .forecast import experiment_context, reflection_receipt
    else:
        from .reflection import experiment_context, reflection_receipt

    actions = {record["step"]: record for record in records if record.get("type") in {"step", "reset"}}
    memories = [record for record in records if record.get("type") == "memory"]
    bindings = memories[-1]["experiment_bindings"] if memories else {}
    assessed = {
        record["assessment"]["step"] for record in memories
        if isinstance(record.get("assessment"), dict)
    }
    receipts = [
        {**reflection_receipt(actions, step), "assessed": step in assessed}
        for step in sorted(actions)
    ]
    result = {
        "version": version, "bindings_validated_against_execution": True,
        "experiment_bindings": bindings,
        "experiments": experiment_context(bindings, actions, assessed),
        "recorded_reflections": receipts,
        "assessed_steps": sorted(assessed),
        "prediction_matches": sum(receipt["prediction_verdict"] == "matched" for receipt in receipts),
        "prediction_mismatches": sum(receipt["prediction_verdict"] == "mismatched" for receipt in receipts),
        "freeform_assessments_verified": False, "learning_claims_verified": False,
        "note": "Typed forecasts are model predictions. Receipts validate their comparison with recorded outcomes; freeform assessments and claims require review.",
    }
    if version in {2, 3, 4}:
        result["note"] = "Expected-outcome values are model forecasts. Readable predictions are software renderings with recorded provenance. Receipts validate execution and differing forecast facets; freeform assessments and claims require review."
    if version in {3, 4}:
        result["note"] = "Expected-outcome values are model forecasts. Readable predictions are software renderings. Before outcomes and actual changed facets describe replayed execution; forecast mismatches remain separate. Freeform assessments and claims require review."
    return result


def _research_facts(records: list[dict]) -> dict:
    """Report executed planning facts without presenting revisions as truth."""
    if records[0].get("reflection_contract") == 4:
        from .research2 import RESEARCH_CONTRACT_VERSION, plan_context
    else:
        from .research import RESEARCH_CONTRACT_VERSION, plan_context

    actions = {record["step"]: record for record in records if record.get("type") in {"step", "reset"}}
    finished = any(record.get("type") == "finish" for record in records)
    observation = records[0]["initial_observation"]
    for record in records:
        if record.get("type") in {"step", "reset"}:
            observation = record["after"]
        elif record.get("type") == "finish":
            observation = record["observation"]
    return {
        "version": RESEARCH_CONTRACT_VERSION,
        "execution_bound_controls_validated": True,
        "planned_controls_are_unexecuted_intent": True,
        "progress": plan_context(actions, observation, finished=finished),
        "revision_reasons_verified": False,
        "note": "Plans are model-selected intentions. Replay checks the sole executed control and whether continuation or an explicit revision matches prior planning; it verifies no rationale, mechanics, or memory causality.",
    }


def retention_progress(path: Path) -> dict:
    """Describe recorded trial bookkeeping without replay or an action suggestion.

    The controller flushes its recording before constructing a new request. This
    view exposes the same retention matches as review, plus conditions for real
    trials cited by current learning. It adds no inference, stored hypotheses,
    engine contracts, or claims that a hypothesis has been verified.
    """
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if (
        not records or records[0].get("type") != "session"
        or records[0].get("schema_version") != 2
        or records[0].get("task") != "explore"
    ):
        raise ValueError("Retention progress requires a schema-2 exploration recording.")
    _, reset_learning = _bound_reset_learning(records)
    pairs = reset_learning["retest_pairs"]
    repeated_actions = {pair["action"] for pair in pairs}
    trials = {}
    notes = {}
    notes_completed = set()
    experiment_ids = set()
    for record in records:
        kind = record.get("type")
        if kind in {"step", "reset"}:
            trials[record["step"]] = record
            identifier = record.get("decision", {}).get("experiment_id")
            if isinstance(identifier, str) and identifier:
                experiment_ids.add(identifier)
        elif kind == "memory":
            notes = _hypotheses(record)
            notes_completed = set(trials)
    conditions = []
    for identifier, note in sorted(notes.items()):
        if note.get("status") not in {"supported", "contradicted", "uncertain"}:
            continue
        for step in sorted(_evidence(note, notes_completed)):
            trial = trials[step]
            before = trial["before"]
            objects = before["objects"]
            conditions.append({
                "hypothesis_id": identifier, "step": step,
                "episode": trial.get("episode", trial.get("episode_before", 0)),
                "action": trial["decision"]["action"],
                "player_position": before["player"]["position"],
                "inventory": before["player"]["inventory"],
                "objects": {
                    "key": objects["key"],
                    "gate": {key: objects["gate"][key] for key in ("position", "open")},
                    "exit": objects["exit"],
                },
            })
    return {
        "families_observed": {
            "movement": bool(repeated_actions.intersection(DIRECTIONS)),
            "interaction": "interact" in repeated_actions,
        },
        "evidence_pairs": pairs,
        "baseline_conditions": conditions,
        "experiment_ids_seen": sorted(experiment_ids),
    }


def _evidence_state(observation: dict) -> dict:
    """Show learner-visible facts, including the neutral episode start message."""
    visible = outcome_view(observation)
    return {key: visible[key] for key in ("status", "player", "objects", "events") if key in visible}


def _recorded_hypothesis_evidence(records: list[dict], recording_complete: bool) -> dict:
    """Quote latest accepted memory beside its actual cited execution receipts."""
    result = {
        "available": records[0]["schema_version"] == 2,
        "recording_complete": recording_complete,
        "claims_verified": False,
        "assessments_verified": False,
        "state_projection": "Learner-visible outcome fields from replayed host observations.",
        "notes": [],
    }
    if not result["available"]:
        result["reason"] = "Schema 1 recordings have no exploration hypothesis memory."
        return result

    actions = {}
    assessments = {}
    latest_memory = {}
    completed_at_memory = set()
    for record in records:
        kind = record.get("type")
        if kind in {"step", "reset"}:
            actions[record["step"]] = record
        if kind == "memory":
            latest_memory = record
            completed_at_memory = set(actions)
            assessment = record.get("assessment")
        elif kind in {"step", "reset", "finish"}:
            assessment = record.get("decision", {}).get("assessment")
        else:
            continue
        if (
            isinstance(assessment, dict)
            and type(assessment.get("step")) is int
            and assessment["step"] in actions
            and isinstance(assessment.get("outcome"), str)
        ):
            assessments[assessment["step"]] = deepcopy(assessment)

    for note in _hypotheses(latest_memory).values():
        refs = deepcopy(note.get("evidence_steps", []))
        receipts = []
        for step in refs if isinstance(refs, list) else []:
            record = actions.get(step) if type(step) is int else None
            if record is None:
                receipts.append({"step": step, "available": False, "reason": "No recorded action for this reference."})
                continue
            if step not in completed_at_memory:
                receipts.append({"step": step, "available": False, "reason": "Referenced action was not completed when this memory was recorded."})
                continue
            before_episode = record.get("episode_before", record.get("episode"))
            after_episode = record.get("episode_after", record.get("episode"))
            receipts.append({
                "step": step, "available": True, "record_type": record["type"],
                "control": record["decision"]["action"],
                "episode_before": before_episode, "episode_after": after_episode,
                "before": _evidence_state(record["before"]), "after": _evidence_state(record["after"]),
                "model_prediction": record["decision"].get("prediction"),
                "model_assessment": deepcopy(assessments.get(step)),
            })
            if records[0].get("prompt_version") == "lean-v1":
                receipts[-1].pop("model_prediction")
                receipts[-1].pop("model_assessment")
                fact = deepcopy(record.get("software_fact"))
                receipts[-1].update(
                    model_expected_position=deepcopy(record["decision"].get("expected_position")),
                    expected_position_matched=fact.get("expected_position_matched") if isinstance(fact, dict) else None,
                    actual_changed_facets=fact.get("actual_changed_facets") if isinstance(fact, dict) else None,
                    software_fact=fact,
                )
            if records[0].get("reflection_contract") in {2, 3, 4}:
                receipts[-1].pop("model_prediction")
                receipts[-1].update(
                    rendered_prediction=record["prediction"],
                    prediction_source=record["prediction_source"],
                    model_plan=deepcopy(record["decision"]["plan"]),
                )
                if records[0]["reflection_contract"] in {3, 4}:
                    from .grounding import reflection_receipt

                    grounded = reflection_receipt(actions, step)
                    receipts[-1].update(
                        actual_before_outcome=deepcopy(grounded["before_outcome"]),
                        actual_changed_facets=deepcopy(grounded["actual_changed_facets"]),
                    )
            if _has_reflection_contract(records):
                receipts[-1].update(
                    experiment_id=record["decision"]["experiment_id"],
                    experiment_control=record["decision"]["experiment_control"],
                    action_role=record["decision"]["action_role"],
                    model_expected_outcome=deepcopy(record["decision"]["expected_outcome"]),
                    typed_reflection_validated=step in assessments,
                )
        entry = {
            "id": note["id"], "claim": deepcopy(note.get("claim")),
            "model_status": deepcopy(note.get("status")), "tentative": True, "verified": False,
            "evidence_steps": refs, "cited_steps": receipts,
        }
        if "control" in note:
            entry["declared_control"] = deepcopy(note["control"])
        if "comparison" in note:
            entry["declared_comparison"] = deepcopy(note["comparison"])
        if records[0].get("reflection_contract") == 4:
            from .measurements import evidence_measurements

            entry["model_effect"] = deepcopy(note["effect"])
            entry["evidence_measurements"] = evidence_measurements([note], actions)
        result["notes"].append(entry)
    result["note"] = "Claims, predictions, and assessments are model text. Replayed receipts establish execution, not truth, intent, or causality."
    if records[0].get("prompt_version") == "lean-v1":
        result["software_facts_validated"] = True
        result["note"] = "Expected positions and notes are model text. Replayed software facts establish execution and state changes, not claim truth, intent, universality, or memory causality."
    if _has_reflection_contract(records):
        result["typed_reflection_validated"] = True
        result["note"] = "Freeform claims, predictions, and assessments are model text. Typed reflected fields match recorded outcomes and forecasts; they establish no hypothesis truth, intent, or memory causality."
        if records[0]["reflection_contract"] in {2, 3, 4}:
            result["note"] = "Expected-outcome values are model forecasts; readable predictions are software renderings with explicit provenance. Replayed receipts validate actual execution and differing forecast facets. Freeform assessments and claims establish no verified truth, intent, or memory causality."
        if records[0]["reflection_contract"] == 4:
            result["effect_bindings_validated"] = True
            result["note"] = "Effect fields are model-selected sampled witnesses bound to cited actual controls and states. Evidence measurements come from replayed states; they verify no universal mechanics, freeform claim, intent, or memory causality."
    return result


def review_session(path: Path, *, include_evidence: bool = False, game_factory=None) -> dict:
    """Replay first; report facts only. P95 uses the nearest-rank percentile.

    Missing legacy metadata is unavailable rather than zero measured latency.
    Request timings cover completed responses with recorded finite latencies.
    Default output omits raw prompts, answers, claims, and error text. Optional
    evidence quotes model notes beside replayed receipts without verifying them.
    """
    replayed = session.replay(path, game_factory=game_factory)
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    from .adaptive_protocol import ADAPTIVE_VERSIONS
    if records[0].get("prompt_version") in ADAPTIVE_VERSIONS:
        from .adaptive_replay import adaptive_review
        return adaptive_review(records, replayed, include_evidence=include_evidence)
    header = records[0]
    summary = records[-1] if records[-1].get("type") == "summary" else None
    schema = header["schema_version"]
    completions = [record for record in records if record.get("type") == "completion"]
    latencies = sorted(
        record["latency_ms"] for record in completions
        if type(record.get("latency_ms")) in {int, float} and math.isfinite(record["latency_ms"]) and record["latency_ms"] >= 0
    )
    latency = {
        "count": len(latencies),
        "median": round(median(latencies), 3) if latencies else None,
        "p95": latencies[math.ceil(0.95 * len(latencies)) - 1] if latencies else None,
        "max": latencies[-1] if latencies else None,
    }
    requests = sum(record.get("type") == "request" for record in records)
    diagnostics = {}
    for key in ("reasoning_tokens", "raw_content_characters", "answer_characters"):
        values = [record[key] for record in completions if type(record.get(key)) is int and record[key] >= 0]
        diagnostics[key] = {"recorded_responses": len(values), "total": sum(values) if values else None}
    output_tokens = [
        record.get("usage", {}).get("completion_tokens") for record in completions
        if type(record.get("usage", {}).get("completion_tokens")) is int and record["usage"]["completion_tokens"] >= 0
    ]
    diagnostics["output_tokens"] = {"recorded_responses": len(output_tokens), "total": sum(output_tokens) if output_tokens else None}
    decisions = [len(record["answer"]) for record in completions if isinstance(record.get("answer"), str)]
    diagnostics["recorded_decision_characters"] = {"recorded_responses": len(decisions), "total": sum(decisions) if decisions else None}
    coverage = Coverage(header["initial_observation"])
    counts = Counter()
    experiments = {}
    tested_experiments = set()
    trial_controls = set()
    resets = 0
    for record in records:
        if record.get("type") not in {"step", "reset"}:
            continue
        action = record["decision"]["action"]
        counts[action] += 1
        resets += record["type"] == "reset"
        coverage.record(record["before"], action, record["after"])
        identifier = record["decision"].get("experiment_id")
        if schema == 2 and identifier is not None:
            experiments.setdefault(identifier, Counter())[action] += 1
            if record["decision"].get("action_role") == "trial":
                tested_experiments.add(identifier)
                trial_controls.add(action)
    lean_mode = header.get("prompt_version") == "lean-v1"
    lean_facts = None
    if lean_mode:
        learning, reset_learning, lean_facts = _lean_learning(records)
    else:
        learning, reset_learning = _bound_reset_learning(records) if schema == 2 else (
            {"available": False}, {"available": False},
        )
    observed = coverage.summary()
    movement = bool(set(counts).intersection(DIRECTIONS))
    interaction = bool(counts["interact"])
    reflection_contract = _has_reflection_contract(records)
    mechanics_observed = bool(
        (True if lean_mode else (len(tested_experiments) if reflection_contract else len(experiments)) >= 2)
        and (bool(trial_controls.intersection(DIRECTIONS)) if reflection_contract else movement)
        and ("interact" in trial_controls if reflection_contract else interaction)
        and observed["blocked_movements"] + observed["unchanged_interactions"] > 0
        and learning.get("evidence_linked_hypotheses", 0) > 0
    )
    repeated_actions = {pair["action"] for pair in reset_learning.get("retest_pairs", [])}
    retested = {"movement": bool(repeated_actions.intersection(DIRECTIONS)), "interaction": "interact" in repeated_actions}
    reset_observed = bool(reset_learning.get("memory_supplied_after_reset") and repeated_actions)
    focus = header.get("focus") if schema == 2 else None
    focused_observed = (
        mechanics_observed if focus == "mechanics"
        else reset_observed and all(retested.values()) if focus == "reset-retention"
        else reset_observed and retested["interaction"] if focus == "interaction-retention"
        else None
    )
    status = summary.get("status") if summary else None
    error_status = status in {
        "invalid_decision", "model_output_limit", "inference_error", "provider_unavailable",
        "contract_failure", "environment_error", "interrupted",
    }
    advance_candidate = bool(
        mechanics_observed and reset_observed and focused_observed is not False
        and all(retested.values())
        and replayed["recording_complete"]
        and learning.get("unassessed_final_step") is None
        and not error_status
    )
    result = {
        "schema_version": schema, "mode": header["mode"],
        "task": header.get("task", "playthrough"),
        "focus": focus, "prompt_version": header.get("prompt_version"),
        "recording_complete": replayed["recording_complete"],
        "status": status,
        "partial": summary.get("partial", summary.get("status") not in {"finished", "won"}) if summary else True,
        "execution_replayed": True,
        "requests": {
            "attempted": requests, "completed": len(completions),
            "without_completion": max(0, requests - len(completions)),
            "rejected_decisions": sum(record.get("type") == "rejected_decision" for record in records),
            "truncated_completions": sum(record.get("finish_reason") == "length" for record in completions),
            "response_ids_recorded": sum(isinstance(record.get("completion_id"), str) and bool(record["completion_id"]) for record in completions),
            "latency_ms": latency, "latency_unavailable": len(completions) - len(latencies),
            "diagnostics": diagnostics,
        },
        "coverage": {
            **observed, "game_actions": sum(counts.values()) - resets,
            "resets": resets, "episodes": resets + 1, "action_counts": dict(sorted(counts.items())),
            "experiment_count": len(experiments),
            "experiments": [{"id": identifier, "action_counts": dict(sorted(actions.items()))} for identifier, actions in experiments.items()],
            "movement": movement, "interaction": interaction,
        },
        "learning": learning, "reset_learning": reset_learning,
        "goal_checks": {
            "mechanics_coverage_observed": mechanics_observed,
            "reset_retention_observed": reset_observed,
            "reset_retested_control_families": retested,
            "focused_question_observed": focused_observed,
            "next_stage_candidate": "controlled_defect_reproduction" if advance_candidate else "exploration_validation",
        },
        "review_needed": schema == 2, "learning_claims_verified": False, "bugs_verified": False,
        "review_note": "Recorded bookkeeping cannot establish distinct mechanics from experiment IDs, hypothesis accuracy, deliberate probing, or that retained memory caused a repeated action. Stage candidates require manual review.",
    }
    if include_evidence:
        result["hypothesis_evidence"] = _recorded_hypothesis_evidence(records, replayed["recording_complete"])
    if lean_facts is not None:
        result["lean_facts"] = lean_facts
    if _has_comparison_contract(records):
        result["comparison_contract"] = _comparison_facts(records)
    if _has_reflection_contract(records):
        result["reflection_contract"] = _reflection_facts(records)
        result["coverage"]["tested_experiment_count"] = len(tested_experiments)
        if records[0]["reflection_contract"] in {2, 3, 4}:
            result["research_contract"] = _research_facts(records)
        if records[0]["reflection_contract"] == 4:
            from .research2 import scope_progress

            ledger = {record["step"]: record for record in records if record.get("type") in {"step", "reset"}}
            memories = [record for record in records if record.get("type") == "memory"]
            notes = _hypotheses(memories[-1]) if memories else {}
            assessed = {record["assessment"]["step"] for record in memories if isinstance(record.get("assessment"), dict)}
            result["experiment_progress"] = scope_progress(ledger, assessed, notes)
        if focus == "interaction-retention":
            result["goal_checks"]["focus_note"] = "This focused question checks interaction retention only. Advancing still requires both movement and interaction comparisons plus manual review."
    return result
