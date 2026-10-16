"""Offline issue reports, deterministic reproduction, and defect evaluation.

The reporting layer intentionally treats model notes as claims and software
anomalies as observations.  It reads only a completed (or partially flushed)
JSONL recording and creates fresh local game instances; it never performs an
inference or network request.
"""

from __future__ import annotations

from collections import deque
from copy import deepcopy
import heapq
import itertools
import math
import json
from pathlib import Path
from typing import Callable, Iterable, Sequence

from .continuous import (
    ContinuousSessionDetector,
    FLOAT_TOLERANCE,
    continuous_anomalies,
    continuous_observation_equal,
    is_continuous_visible,
)
from .collision_geometry import passed_solid_geometry_supported
from .game import DIRECTIONS, KeyGateGame


ISSUE_SOURCES = ("suspected_bug", "software_anomaly")
ANOMALY_TYPES = (
    "prediction_surprise",
    "cross_episode_nondeterminism",
    "reset_inconsistency",
    "map_object_inconsistency",
    "label_contradiction",
    "passed_through_solid",
    "blocked_no_visible_obstacle",
    "fell_through_visible_floor",
    "sank_into_visible_floor",
    "player_stuck_in_geometry",
)

ISSUE_ANOMALY_TYPES = {
    "cross_episode_nondeterminism",
    "reset_inconsistency",
    "map_object_inconsistency",
    "label_contradiction",
    "passed_through_solid",
    "blocked_no_visible_obstacle",
    "fell_through_visible_floor",
    "sank_into_visible_floor",
    "player_stuck_in_geometry",
}

CONTINUOUS_ANOMALY_TYPES = {
    "passed_through_solid",
    "blocked_no_visible_obstacle",
    "fell_through_visible_floor",
    "sank_into_visible_floor",
    "player_stuck_in_geometry",
}

_CONTINUOUS_EXACT_MINIMIZATION_LIMIT = 64
_CONTINUOUS_FALLBACK_MINIMIZATION_ATTEMPTS = 12
_CONTINUOUS_TARGET_SEARCH_DEPTH = 16
_CONTINUOUS_TARGET_SEARCH_STATES = 512

LABEL_CONTRADICTION_ASPECTS = {
    "entered_wall_labeled_tile",
    "blocked_on_floor_labeled_tile",
    "blocked_on_open_gate_labeled_tile",
    "blocked_on_key_labeled_tile",
    "blocked_on_exit_labeled_tile",
    "key_use_feedback_inventory",
    "key_collection_feedback_state",
    "gate_open_feedback_state",
    "exit_status_unchanged",
}


def load_recording(path: str | Path) -> list[dict]:
    """Load a JSONL recording with useful line-oriented errors."""
    source = Path(path)
    records: list[dict] = []
    with source.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON in recording at line {line_number}: {exc.msg}."
                ) from None
            if not isinstance(record, dict):
                raise ValueError(
                    f"Recording line {line_number} is not a JSON object."
                )
            records.append(record)
    if not records or records[0].get("type") != "session":
        raise ValueError("Recording has no leading session record.")
    return records


def _coerce_records(recording: str | Path | Sequence[dict]) -> list[dict]:
    if isinstance(recording, (str, Path)):
        return load_recording(recording)
    records = list(recording)
    if not records or not isinstance(records[0], dict) or records[0].get("type") != "session":
        raise ValueError("Recording has no leading session record.")
    if any(not isinstance(record, dict) for record in records):
        raise ValueError("Recording contains a non-object record.")
    return records


def _header(records: Sequence[dict]) -> dict:
    return records[0]


def _ground_truth(records: Sequence[dict]) -> dict | None:
    evaluation = _header(records).get("evaluation")
    injected = evaluation.get("injected_defect") if isinstance(evaluation, dict) else None
    if injected is None:
        return None
    if isinstance(injected, str):
        return {"name": injected, "description": "", "details": {}}
    if not isinstance(injected, dict) or not isinstance(injected.get("name"), str):
        return None
    return injected


def _normalize_type(value: object) -> str:
    if not isinstance(value, str):
        return "unknown_anomaly"
    normalized = value.strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "prediction_mismatch": "prediction_surprise",
        "position_prediction_surprise": "prediction_surprise",
        "same_start_different_outcome": "cross_episode_nondeterminism",
        "cross_episode_non_determinism": "cross_episode_nondeterminism",
        "reset_state_inconsistency": "reset_inconsistency",
        "map_inconsistency": "map_object_inconsistency",
        "object_inconsistency": "map_object_inconsistency",
        "passed_through_visible_solid": "passed_through_solid",
        "passed_through_wall": "passed_through_solid",
        "passed_through_solid_wall": "passed_through_solid",
        "blocked_no_obstacle": "blocked_no_visible_obstacle",
        "blocked_with_no_visible_obstacle": "blocked_no_visible_obstacle",
        "movement_blocked_no_visible_obstacle": "blocked_no_visible_obstacle",
        "fell_through_floor": "fell_through_visible_floor",
        "player_fell_through_visible_floor_tile": "fell_through_visible_floor",
        "sank_into_floor": "sank_into_visible_floor",
        "player_sank_into_visible_floor_tile": "sank_into_visible_floor",
        "stuck_in_geometry": "player_stuck_in_geometry",
    }
    return aliases.get(normalized, normalized or "unknown_anomaly")


def _step_records(records: Sequence[dict]) -> list[dict]:
    rows = [
        record for record in records
        if record.get("type") in {"step", "reset", "sweep_step", "sweep_reset"}
    ]
    return sorted(rows, key=lambda row: row.get("step", 0))


def _step_index(records: Sequence[dict]) -> dict[int, dict]:
    return {
        record["step"]: record
        for record in _step_records(records)
        if type(record.get("step")) is int
    }


def _public_facets(observation: object) -> dict:
    if not isinstance(observation, dict):
        return {}
    player = observation.get("player")
    objects = observation.get("objects")
    player = player if isinstance(player, dict) else {}
    objects = objects if isinstance(objects, dict) else {}
    gate = objects.get("gate")
    gate = gate if isinstance(gate, dict) else {}
    if "key" in objects and "open" in gate:
        return {
            "position": deepcopy(player.get("position")),
            "inventory": deepcopy(player.get("inventory", [])),
            "key": deepcopy(objects.get("key")),
            "gate_open": gate.get("open"),
            "status": observation.get("status"),
        }
    entities = observation.get("entities")
    entities = entities if isinstance(entities, list) else []
    return {
        "position": deepcopy(player.get("position")),
        "inventory": deepcopy(player.get("inventory", [])),
        "entities": deepcopy(entities),
        "status": observation.get("status"),
    }


def _feedback(observation: object) -> list:
    if not isinstance(observation, dict):
        return []
    events = observation.get("events")
    return deepcopy(events) if isinstance(events, list) else []


def _compact_transition(record: dict) -> dict:
    decision = record.get("decision")
    decision = decision if isinstance(decision, dict) else {}
    fact = record.get("software_fact")
    fact = fact if isinstance(fact, dict) else {}
    return {
        "step": record.get("step"),
        "action": decision.get("action"),
        "before": _public_facets(record.get("before")),
        "after": _public_facets(record.get("after")),
        "feedback": _feedback(record.get("after")),
        "expected_position": deepcopy(
            fact.get("expected_position", decision.get("expected_position"))
        ),
    }


def _note_candidates(records: Sequence[dict]) -> list[dict]:
    """Return the latest complete memory snapshot, with a decision fallback."""
    latest: dict[str, dict] = {}
    fallback: dict[str, dict] = {}
    saw_memory = False
    for record in records:
        candidates: object = None
        if record.get("type") == "memory" and isinstance(record.get("hypotheses"), list):
            saw_memory = True
            candidates = record["hypotheses"]
        elif record.get("type") == "summary" and isinstance(record.get("hypotheses"), list):
            saw_memory = True
            candidates = record["hypotheses"]
        elif not saw_memory and record.get("type") in {"step", "reset", "finish"}:
            decision = record.get("decision")
            if isinstance(decision, dict):
                updates = decision.get("notes")
                if isinstance(updates, list):
                    for note in updates:
                        identifier = note.get("id") if isinstance(note, dict) else None
                        if isinstance(identifier, str) and identifier:
                            fallback[identifier] = deepcopy(note)
            continue
        if not isinstance(candidates, list):
            continue
        snapshot: dict[str, dict] = {}
        for note in candidates:
            if not isinstance(note, dict):
                continue
            identifier = note.get("id")
            if isinstance(identifier, str) and identifier:
                snapshot[identifier] = deepcopy(note)
        latest = snapshot
    return list((latest if saw_memory else fallback).values())


def _evidence_steps(value: object, valid_steps: set[int]) -> list[int]:
    if not isinstance(value, dict):
        return []
    found: list[int] = []

    def add(candidate: object) -> None:
        if type(candidate) is int and candidate in valid_steps and candidate not in found:
            found.append(candidate)

    values = value.get("evidence_steps")
    if isinstance(values, list):
        for candidate in values:
            add(candidate)
    for key in (
        "step",
        "original_step",
        "repeated_step",
        "reset_step",
        "evidence_step",
        "first_step",
        "second_step",
    ):
        add(value.get(key))
    evidence = value.get("evidence")
    if isinstance(evidence, dict):
        for key, candidate in evidence.items():
            if "step" in str(key):
                if isinstance(candidate, list):
                    for item in candidate:
                        add(item)
                else:
                    add(candidate)
    elif isinstance(evidence, list):
        for item in evidence:
            if isinstance(item, dict):
                for key, candidate in item.items():
                    if "step" in str(key):
                        add(candidate)
            else:
                add(item)
    return sorted(found)


def _anomaly_lists(record: dict) -> Iterable[object]:
    for key in ("anomalies", "software_anomalies", "anomaly_facts"):
        if key in record:
            yield record[key]
    fact = record.get("software_fact")
    if isinstance(fact, dict):
        for key in ("anomalies", "software_anomalies", "anomaly_facts"):
            if key in fact:
                yield fact[key]
    facts = record.get("software_facts")
    if isinstance(facts, dict):
        for key in ("anomalies", "software_anomalies", "anomaly_facts"):
            if key in facts:
                yield facts[key]


def _stored_anomalies(records: Sequence[dict]) -> list[dict]:
    valid_steps = set(_step_index(records))
    anomalies: list[dict] = []
    for record in records:
        for raw_list in _anomaly_lists(record):
            if isinstance(raw_list, dict):
                raw_list = [raw_list]
            if not isinstance(raw_list, list):
                continue
            for raw in raw_list:
                if isinstance(raw, str):
                    anomaly = {"type": raw}
                elif isinstance(raw, dict):
                    anomaly = deepcopy(raw)
                else:
                    continue
                anomaly["type"] = _normalize_type(
                    anomaly.get("type", anomaly.get("kind", anomaly.get("code")))
                )
                if "step" not in anomaly and type(record.get("step")) is int:
                    anomaly["step"] = record["step"]
                anomaly["evidence_steps"] = _evidence_steps(anomaly, valid_steps)
                anomalies.append(anomaly)
    return anomalies


def _observation_without_tick(observation: object) -> object:
    if not isinstance(observation, dict):
        return observation
    value = deepcopy(observation)
    value.pop("tick", None)
    return value


def _observations_equal(
    left: object,
    right: object,
    *,
    ignore_tick: bool = False,
) -> bool:
    """Compare continuous observations tolerantly and legacy state exactly."""
    if is_continuous_visible(left) and is_continuous_visible(right):
        return continuous_observation_equal(
            left,
            right,
            tolerance=FLOAT_TOLERANCE,
            ignore_tick=ignore_tick,
        )
    if ignore_tick:
        left = _observation_without_tick(left)
        right = _observation_without_tick(right)
    return left == right


def _outcomes_equal(left: dict, right: dict) -> bool:
    """Compare two executed outcomes using their advertised world semantics."""
    left_after = left.get("after")
    right_after = right.get("after")
    return (
        _observations_equal(left_after, right_after, ignore_tick=True)
        and _feedback(left_after) == _feedback(right_after)
    )


def _tile(observation: object, position: tuple[int, int] | list[int]) -> str | None:
    if not isinstance(observation, dict) or len(position) != 2:
        return None
    rows = observation.get("map", observation.get("grid"))
    x, y = position
    if (
        type(x) is not int
        or type(y) is not int
        or not isinstance(rows, list)
        or not 0 <= y < len(rows)
        or not isinstance(rows[y], str)
        or not 0 <= x < len(rows[y])
    ):
        return None
    return rows[y][x]


def _legend_meaning(observation: object, glyph: str | None) -> str:
    if not isinstance(observation, dict) or not isinstance(glyph, str):
        return ""
    legend = observation.get("legend")
    meaning = legend.get(glyph) if isinstance(legend, dict) else None
    return meaning.lower() if isinstance(meaning, str) else ""


def _event_text(observation: object) -> str:
    return " ".join(
        event.lower() for event in _feedback(observation) if isinstance(event, str)
    )


def _generic_entity(observation: object, kind: str) -> dict:
    if not isinstance(observation, dict):
        return {}
    entities = observation.get("entities")
    if not isinstance(entities, list):
        return {}
    for entity in entities:
        if not isinstance(entity, dict):
            continue
        if any(
            isinstance(value, str) and value.lower() == kind
            for value in (entity.get("id"), entity.get("kind"))
        ):
            return entity
    return {}


def _entity_state(entity: dict) -> dict:
    state = entity.get("state") if isinstance(entity, dict) else None
    return state if isinstance(state, dict) else {}


def _inventory_tokens(inventory: object) -> set[str]:
    if not isinstance(inventory, list):
        return set()
    return {
        str(item.get("kind", item.get("id", ""))).lower()
        if isinstance(item, dict) else str(item).lower()
        for item in inventory
    }


def _grid_legend_enabled(observation: object) -> bool:
    """Honor remote capabilities while retaining historical recordings."""
    if not isinstance(observation, dict):
        return False
    capabilities = observation.get("_capabilities")
    if not isinstance(capabilities, dict):
        return True
    return capabilities.get("grid") is True and capabilities.get("legend") is True


def _movement_target(record: dict) -> tuple[list[int], str | None, str] | None:
    decision = record.get("decision")
    decision = decision if isinstance(decision, dict) else {}
    action = decision.get("action")
    before = record.get("before")
    if (
        action not in DIRECTIONS
        or not isinstance(before, dict)
        or not _grid_legend_enabled(before)
    ):
        return None
    before_position = _public_facets(before).get("position")
    if (
        not isinstance(before_position, list)
        or len(before_position) != 2
        or not all(type(value) is int for value in before_position)
    ):
        return None
    dx, dy = DIRECTIONS[action]
    target = [before_position[0] + dx, before_position[1] + dy]
    glyph = _tile(before, target)
    return target, glyph, _legend_meaning(before, glyph)


def _label_contradiction_aspects(record: dict) -> list[str]:
    """Return contradictions grounded only in shown labels, state, and feedback."""
    before = record.get("before")
    after = record.get("after")
    if not isinstance(before, dict) or not isinstance(after, dict):
        return []
    before_facets = _public_facets(before)
    after_facets = _public_facets(after)
    before_position = before_facets.get("position")
    after_position = after_facets.get("position")
    aspects: list[str] = []
    movement = _movement_target(record)
    if movement is not None and isinstance(before_position, list):
        target, glyph, meaning = movement
        moved = after_position == target and target != before_position
        blocked = (
            after_position == before_position
            and "blocked" in _event_text(after)
        )
        is_wall = glyph == "#" or "wall" in meaning
        if moved and is_wall:
            aspects.append("entered_wall_labeled_tile")

        passable_aspect: str | None = None
        if glyph == "." or "floor" in meaning:
            passable_aspect = "blocked_on_floor_labeled_tile"
        elif glyph == "/" or "open gate" in meaning:
            gate = before.get("objects", {}).get("gate") if isinstance(before.get("objects"), dict) else None
            if not isinstance(gate, dict) or gate.get("position") != target or gate.get("open") is True:
                passable_aspect = "blocked_on_open_gate_labeled_tile"
        elif glyph == "K" or meaning == "key" or meaning.endswith(" key"):
            passable_aspect = "blocked_on_key_labeled_tile"
        elif glyph == "E" or "exit" in meaning:
            passable_aspect = "blocked_on_exit_labeled_tile"
        if blocked and passable_aspect is not None:
            aspects.append(passable_aspect)

        if (
            moved
            and (glyph == "E" or "exit" in meaning)
            and before_facets.get("status") == after_facets.get("status")
        ):
            aspects.append("exit_status_unchanged")

    feedback = _event_text(after)
    inventory = after_facets.get("inventory")
    inventory = inventory if isinstance(inventory, list) else []
    inventory_tokens = _inventory_tokens(inventory)
    objects = after.get("objects")
    objects = objects if isinstance(objects, dict) else {}
    if (
        any(phrase in feedback for phrase in ("used the key", "consumed the key", "key was used"))
        and "key" in inventory_tokens
    ):
        aspects.append("key_use_feedback_inventory")
    if any(phrase in feedback for phrase in ("collected the key", "picked up the key")):
        key_entity = _generic_entity(after, "key")
        key_state = _entity_state(key_entity)
        key_available = (
            key_entity.get("position") is not None
            and key_state.get("available", True) is not False
        ) if key_entity else objects.get("key") is not None
        if "key" not in inventory_tokens or key_available:
            aspects.append("key_collection_feedback_state")
    gate = objects.get("gate")
    gate = gate if isinstance(gate, dict) else {}
    generic_gate = _generic_entity(after, "gate")
    gate_open = _entity_state(generic_gate).get("open") if generic_gate else gate.get("open")
    if (
        any(phrase in feedback for phrase in ("unlocked the gate", "opened the gate"))
        and gate_open is not True
    ):
        aspects.append("gate_open_feedback_state")
    return aspects


def _map_inconsistency_aspects(record: dict) -> list[str]:
    decision = record.get("decision")
    decision = decision if isinstance(decision, dict) else {}
    before = record.get("before")
    after = record.get("after")
    if not isinstance(before, dict) or not isinstance(after, dict):
        return []
    after_facets = _public_facets(after)
    after_position = after_facets.get("position")
    aspects: list[str] = []

    rows = after.get("map", after.get("grid"))
    has_grid_legend = (
        _grid_legend_enabled(after)
        and bool(rows)
        and isinstance(rows, list)
        and isinstance(after.get("legend"), dict)
    )
    if has_grid_legend and isinstance(after_position, list):
        player_glyph = _tile(after, after_position)
        if "player" not in _legend_meaning(after, player_glyph):
            aspects.append("player_position_glyph")

    objects = after.get("objects")
    objects = objects if isinstance(objects, dict) else {}
    exit_position = objects.get("exit")

    gate = objects.get("gate")
    gate = gate if isinstance(gate, dict) else {}
    object_specs = [
        ("key", objects.get("key"), "key"),
        ("exit", exit_position, "exit"),
        ("gate", gate.get("position"), "open gate" if gate.get("open") is True else "gate"),
    ]
    for object_name, position, expected_meaning in object_specs:
        if (
            not has_grid_legend
            or
            not isinstance(position, list)
            or len(position) != 2
            or not all(type(value) is int for value in position)
            or position == after_position
        ):
            continue
        meaning = _legend_meaning(after, _tile(after, position))
        matches = expected_meaning in meaning
        if object_name == "gate" and expected_meaning == "gate":
            matches = "gate" in meaning and "open" not in meaning
        if not matches:
            aspects.append(f"{object_name}_object_glyph")

    inventory = after.get("player", {}).get("inventory", []) if isinstance(after.get("player"), dict) else []
    generic_key = _generic_entity(after, "key")
    if generic_key:
        key_state = _entity_state(generic_key)
        key_in_world = (
            generic_key.get("position") is not None
            and key_state.get("present", True) is not False
            and key_state.get("available", True) is not False
            and key_state.get("collected", False) is not True
        )
    else:
        key_in_world = objects.get("key") is not None
    if "key" in _inventory_tokens(inventory) and key_in_world:
        aspects.append("key_listed_in_world_and_inventory")
    return aspects


def _map_inconsistency_reasons(record: dict) -> list[str]:
    descriptions = {
        "player_position_glyph": "the player's position lacked the player glyph",
        "key_object_glyph": "the key object and map glyph disagreed",
        "exit_object_glyph": "the exit object and map glyph disagreed",
        "gate_object_glyph": "the gate object and map glyph disagreed",
        "key_listed_in_world_and_inventory": "the key appears both in inventory and on the map",
    }
    return [descriptions.get(aspect, aspect) for aspect in _map_inconsistency_aspects(record)]


def _label_contradiction_reason(aspect: str) -> str:
    return {
        "entered_wall_labeled_tile": "movement entered a tile labeled as a wall",
        "blocked_on_floor_labeled_tile": "movement into a tile labeled as floor was blocked",
        "blocked_on_open_gate_labeled_tile": "movement into a visibly open gate was blocked",
        "blocked_on_key_labeled_tile": "movement into a tile labeled as a key was blocked",
        "blocked_on_exit_labeled_tile": "movement into a tile labeled as an exit was blocked",
        "key_use_feedback_inventory": "feedback said the key was used but it remained in inventory",
        "key_collection_feedback_state": "key collection feedback disagreed with inventory or object state",
        "gate_open_feedback_state": "gate-opening feedback disagreed with the visible gate state",
        "exit_status_unchanged": "status did not change after entering a tile labeled as the exit",
    }.get(aspect, aspect.replace("_", " "))


def _reporting_anomalies(records: Sequence[dict]) -> list[dict]:
    """Derive report-only contradictions from data visible in each transition."""
    derived: list[dict] = []
    steps = _step_records(records)
    initial = _header(records).get("initial_observation")
    continuous_detector = ContinuousSessionDetector()
    for record in steps:
        step = record.get("step")
        decision = record.get("decision")
        action = decision.get("action") if isinstance(decision, dict) else None
        derived.extend(continuous_detector.anomalies(record))
        for aspect in _label_contradiction_aspects(record):
            derived.append({
                "type": "map_object_inconsistency",
                "category": "label_contradiction",
                "step": step,
                "evidence_steps": [step],
                "action": action,
                "aspect": aspect,
                "details": [_label_contradiction_reason(aspect)],
            })
        for aspect in _map_inconsistency_aspects(record):
            derived.append({
                "type": "map_object_inconsistency",
                "step": step,
                "evidence_steps": [step],
                "action": action,
                "aspect": aspect,
                "details": _map_inconsistency_reasons(record),
            })

        if (
            record.get("type") in {"reset", "sweep_reset"}
            and isinstance(initial, dict)
            and not _observations_equal(
                record.get("after"), initial, ignore_tick=True
            )
        ):
            derived.append({
                "type": "reset_inconsistency",
                "step": step,
                "evidence_steps": [step],
                "action": "reset",
                "expected": _public_facets(initial),
                "actual": _public_facets(record.get("after")),
            })

    trials = [
        record for record in steps
        if record.get("type") in {"step", "sweep_step"}
    ]
    for repeated_index, repeated in enumerate(trials):
        repeated_decision = repeated.get("decision")
        repeated_action = (
            repeated_decision.get("action")
            if isinstance(repeated_decision, dict) else None
        )
        repeated_episode = repeated.get(
            "episode", repeated.get("episode_after", 0)
        )
        repeated_outcome = {
            "facets": _public_facets(repeated.get("after")),
            "feedback": _feedback(repeated.get("after")),
        }
        for original in trials[:repeated_index]:
            original_decision = original.get("decision")
            original_action = (
                original_decision.get("action")
                if isinstance(original_decision, dict) else None
            )
            original_episode = original.get(
                "episode", original.get("episode_after", 0)
            )
            if original_episode == repeated_episode or original_action != repeated_action:
                continue
            if not _visible_starts_equal(original, repeated):
                continue
            original_outcome = {
                "facets": _public_facets(original.get("after")),
                "feedback": _feedback(original.get("after")),
            }
            if _outcomes_equal(original, repeated):
                continue
            derived.append({
                "type": "cross_episode_nondeterminism",
                "step": repeated.get("step"),
                "original_step": original.get("step"),
                "repeated_step": repeated.get("step"),
                "evidence_steps": [original.get("step"), repeated.get("step")],
                "action": repeated_action,
                "expected": deepcopy(original_outcome),
                "actual": deepcopy(repeated_outcome),
            })
    return derived


def _derived_anomalies(records: Sequence[dict]) -> list[dict]:
    """Compatibility fallback for lean recordings predating explicit anomaly lists."""
    derived: list[dict] = []
    steps = _step_records(records)
    initial = _header(records).get("initial_observation")
    for record in steps:
        step = record.get("step")
        fact = record.get("software_fact")
        fact = fact if isinstance(fact, dict) else {}
        expected = fact.get("expected_position")
        if expected is None:
            decision = record.get("decision")
            expected = decision.get("expected_position") if isinstance(decision, dict) else None
        matched = fact.get("expected_position_matched")
        actual_position = _public_facets(record.get("after")).get("position")
        if expected is not None and (matched is False or actual_position != expected):
            derived.append({
                "type": "prediction_surprise",
                "step": step,
                "evidence_steps": [step],
                "expected_position": deepcopy(expected),
                "actual_position": deepcopy(actual_position),
            })
        if record.get("type") in {"reset", "sweep_reset"} and initial is not None:
            if not _observations_equal(
                record.get("after"), initial, ignore_tick=True
            ):
                derived.append({
                    "type": "reset_inconsistency",
                    "step": step,
                    "evidence_steps": [step],
                    "expected": _public_facets(initial),
                    "actual": _public_facets(record.get("after")),
                })
        aspects = _map_inconsistency_aspects(record)
        for aspect in aspects:
            derived.append({
                "type": "map_object_inconsistency",
                "step": step,
                "evidence_steps": [step],
                "aspect": aspect,
                "details": _map_inconsistency_reasons(record),
            })

    # Aggregate comparison facts may occur in every memory row; collect and
    # deduplicate below.  Also derive pairs for compact synthetic recordings.
    pairs: list[dict] = []
    for record in records:
        containers = [record]
        if isinstance(record.get("software_facts"), dict):
            containers.append(record["software_facts"])
        for container in containers:
            value = container.get("cross_episode_same_start_pairs")
            if isinstance(value, list):
                pairs.extend(item for item in value if isinstance(item, dict))
    for pair in pairs:
        if pair.get("result") == "different":
            evidence = [
                step for step in (pair.get("original_step"), pair.get("repeated_step"))
                if type(step) is int
            ]
            derived.append({
                "type": "cross_episode_nondeterminism",
                **deepcopy(pair),
                "evidence_steps": evidence,
            })
    return derived


def _anomaly_key(anomaly: dict) -> tuple:
    kind = _normalize_type(anomaly.get("type"))
    evidence = tuple(anomaly.get("evidence_steps", []))
    if evidence:
        discriminator = anomaly.get("aspect") if kind == "map_object_inconsistency" else None
        return kind, evidence, discriminator
    return kind, json.dumps(anomaly, sort_keys=True, separators=(",", ":"), default=str)


def _canonical_anomaly(anomaly: dict) -> dict:
    value = deepcopy(anomaly)
    kind = _normalize_type(value.get("type"))
    if kind == "label_contradiction":
        kind = "map_object_inconsistency"
        value["category"] = "label_contradiction"
    elif (
        kind == "map_object_inconsistency"
        and value.get("aspect") in LABEL_CONTRADICTION_ASPECTS
    ):
        value["category"] = "label_contradiction"
    value["type"] = kind
    return value


def _anomalies(records: Sequence[dict]) -> list[dict]:
    unique: dict[tuple, dict] = {}
    # Prefer the richer explicitly recorded value over a compatibility-derived
    # value having the same type and evidence steps.
    for anomaly in _reporting_anomalies(records):
        anomaly = _canonical_anomaly(anomaly)
        unique.setdefault(_anomaly_key(anomaly), anomaly)
    header = _header(records)
    qa_mode = header.get("focus") == "qa" or any(
        record.get("qa_mode") is True for record in records
    )
    if qa_mode:
        for anomaly in _derived_anomalies(records):
            anomaly = _canonical_anomaly(anomaly)
            unique.setdefault(_anomaly_key(anomaly), anomaly)
    for anomaly in _stored_anomalies(records):
        anomaly = _canonical_anomaly(anomaly)
        key = _anomaly_key(anomaly)
        if key in unique:
            merged = {**unique[key], **anomaly}
            merged["evidence_steps"] = anomaly.get(
                "evidence_steps", unique[key].get("evidence_steps", [])
            )
            unique[key] = merged
        else:
            unique[key] = anomaly
    return list(unique.values())


def _anomaly_expected_actual(
    anomaly: dict,
    evidence: list[dict],
    initial: object,
) -> tuple[object, object]:
    expected = anomaly.get("expected", anomaly.get("prior_outcome"))
    actual = anomaly.get("actual", anomaly.get("observed"))
    kind = _normalize_type(anomaly.get("type"))
    if kind == "prediction_surprise" and evidence:
        expected = anomaly.get("expected", anomaly.get(
            "expected_position", evidence[-1].get("expected_position")
        ))
        actual = anomaly.get("actual", {
            "facets": evidence[-1].get("after"),
            "feedback": evidence[-1].get("feedback"),
        })
    elif kind == "cross_episode_nondeterminism" and len(evidence) >= 2:
        expected = anomaly.get("expected", {
            "prior_facets": evidence[0].get("after"),
            "prior_feedback": evidence[0].get("feedback"),
        })
        actual = anomaly.get("actual", {
            "repeated_facets": evidence[-1].get("after"),
            "repeated_feedback": evidence[-1].get("feedback"),
        })
    elif kind == "reset_inconsistency" and evidence:
        expected = anomaly.get("expected", _public_facets(initial))
        actual = anomaly.get("actual", {
            "facets": evidence[-1].get("after"),
            "feedback": evidence[-1].get("feedback"),
        })
    elif kind in {"map_object_inconsistency", "label_contradiction"} and evidence:
        expected = anomaly.get(
            "expected",
            "Visible map glyphs, object state, inventory, movement, and status agree.",
        )
        actual = anomaly.get("actual", {
            "details": deepcopy(anomaly.get("details", [])),
            "facets": evidence[-1].get("after"),
            "feedback": evidence[-1].get("feedback"),
        })
    elif kind in CONTINUOUS_ANOMALY_TYPES and evidence:
        expected = anomaly.get(
            "expected",
            "Movement agrees with the bounds and state of visible solid entities.",
        )
        actual = anomaly.get("actual", {
            "facets": evidence[-1].get("after"),
            "feedback": evidence[-1].get("feedback"),
        })
    if expected is None:
        expected = "The prior observation or immediate model prediction recurs."
    if actual is None:
        actual = {"evidence": evidence}
    return deepcopy(expected), deepcopy(actual)


def _title_for_anomaly(kind: str, anomaly: object = None) -> str:
    """Describe a software anomaly using only its visible-evidence detail kind."""
    detail_titles = {
        "key_use_feedback_inventory": (
            "Key remained in inventory after feedback said it was used"
        ),
        "exit_status_unchanged": "Entering the exit tile did not end the game",
        "blocked_on_floor_labeled_tile": "Movement into a floor tile was blocked",
        "entered_wall_labeled_tile": "Movement into a wall tile succeeded",
        "blocked_on_open_gate_labeled_tile": (
            "Movement into an open gate tile was blocked"
        ),
        "blocked_on_key_labeled_tile": "Movement into a key tile was blocked",
        "blocked_on_exit_labeled_tile": "Movement into the exit tile was blocked",
        "key_collection_feedback_state": (
            "Key collection feedback did not match the visible state"
        ),
        "gate_open_feedback_state": (
            "Gate-opening feedback did not match the visible state"
        ),
        "player_position_glyph": "Player position did not match the visible map",
        "key_object_glyph": "Key position did not match the visible map",
        "exit_object_glyph": "Exit position did not match the visible map",
        "gate_object_glyph": "Gate position did not match the visible map",
        "key_listed_in_world_and_inventory": (
            "Key appeared both in inventory and in the world"
        ),
    }
    if isinstance(anomaly, dict):
        details: list[object] = [anomaly.get("aspect")]
        aspects = anomaly.get("aspects")
        if isinstance(aspects, list):
            details.extend(aspects)
        for detail in details:
            if isinstance(detail, str) and detail in detail_titles:
                return detail_titles[detail]
    return {
        "prediction_surprise": "Observed position differed from the immediate prediction",
        "cross_episode_nondeterminism": "Equivalent start and action produced a different outcome",
        "reset_inconsistency": "Reset did not restore the initial state",
        "map_object_inconsistency": "Visible map and game state were inconsistent",
        "label_contradiction": "Observed outcome contradicted a visible label or feedback",
        "passed_through_solid": "Player passed through a solid wall",
        "blocked_no_visible_obstacle": "Movement blocked with no visible obstacle",
        "fell_through_visible_floor": "Player fell through a visible floor tile",
        "sank_into_visible_floor": "Player sank into a visible floor tile",
        "player_stuck_in_geometry": "Player stuck in geometry",
    }.get(kind, kind.replace("_", " ").strip().capitalize() or "Software anomaly")


def _note_expected_actual(evidence: list[dict], claim: str) -> tuple[object, object]:
    """Prefer the model forecast or a comparable prior outcome over prose."""
    expected: object = {"model_flag": claim}
    if len(evidence) >= 2:
        original, repeated = evidence[-2], evidence[-1]
        if (
            original.get("action") == repeated.get("action")
            and original.get("before") == repeated.get("before")
        ):
            expected = {
                "prior_step": original.get("step"),
                "prior_facets": deepcopy(original.get("after")),
                "prior_feedback": deepcopy(original.get("feedback")),
            }
    if isinstance(expected, dict) and "prior_step" not in expected:
        forecast = next(
            (
                item.get("expected_position")
                for item in reversed(evidence)
                if item.get("expected_position") is not None
            ),
            None,
        )
        if forecast is not None:
            expected = {
                "model_expected_position": deepcopy(forecast),
                "model_flag": claim,
            }
    latest = evidence[-1] if evidence else {}
    actual = {
        "observed_facets": deepcopy(latest.get("after", {})),
        "feedback": deepcopy(latest.get("feedback", [])),
        "evidence": deepcopy(evidence),
    }
    return expected, actual


def _claim_signature(claim: object) -> str | None:
    if not isinstance(claim, str):
        return None
    text = claim.lower()
    if "gate" in text and "key" in text and any(
        word in text for word in ("consume", "consumed", "kept", "remain", "retain", "inventory")
    ):
        return "gate-keeps-key"
    if "reset" in text and any(
        word in text for word in ("key", "inventory", "restore", "retain", "state", "leak")
    ):
        return "reset-leak"
    if "exit" in text and any(
        word in text for word in ("win", "won", "victory", "second", "did not", "does not", "not ")
    ):
        return "exit-no-win"
    if "wall" in text and any(
        word in text for word in ("clip", "walk", "move", "enter", "pass")
    ):
        return "wall-clip"
    if "block" in text and any(
        word in text for word in ("floor", "walkable", "open tile", "empty tile")
    ):
        return "phantom-block"
    return None


def _transition_signature(transition: dict, initial: object = None) -> str | None:
    action = transition.get("action")
    before = transition.get("_before_observation")
    after = transition.get("_after_observation")
    if action == "reset" and initial is not None and after is not None:
        if not _observations_equal(after, initial, ignore_tick=True):
            return "respawn-leak" if is_continuous_visible(after) else "reset-leak"
    if isinstance(before, dict) and isinstance(after, dict):
        synthetic = {
            "type": "step",
            "step": transition.get("source_step", transition.get("step", 0)),
            "decision": {"action": action},
            "before": before,
            "after": after,
        }
        continuous_signatures = {
            "passed_through_solid": "wall-tunnel",
            "blocked_no_visible_obstacle": "door-ghost-collider",
            "fell_through_visible_floor": "floor-gap",
            "sank_into_visible_floor": "floor-gap",
            "player_stuck_in_geometry": "floor-gap",
        }
        for anomaly in continuous_anomalies(synthetic):
            if (_normalize_type(anomaly.get("type")) == "passed_through_solid"
                    and not passed_solid_geometry_supported(synthetic, anomaly)):
                continue
            signature = continuous_signatures.get(_normalize_type(anomaly.get("type")))
            if signature is not None:
                return signature
        aspects = set(_label_contradiction_aspects(synthetic))
        if "key_use_feedback_inventory" in aspects:
            return "key-not-consumed" if is_continuous_visible(after) else "gate-keeps-key"
        if "entered_wall_labeled_tile" in aspects:
            return "wall-clip"
        if "blocked_on_floor_labeled_tile" in aspects:
            return "phantom-block"
        if "exit_status_unchanged" in aspects:
            return "exit-no-win"
    return None


def _observed_signature(issue: dict, records: Sequence[dict]) -> str | None:
    index = _step_index(records)
    initial = _header(records).get("initial_observation")
    for step in issue.get("evidence_steps", []):
        record = index.get(step)
        if record is None:
            continue
        compact = _compact_transition(record)
        compact["_before_observation"] = record.get("before")
        compact["_after_observation"] = record.get("after")
        signature = _transition_signature(compact, initial)
        if signature is not None:
            return signature
    kind = _normalize_type(issue.get("type"))
    if kind == "reset_inconsistency":
        return (
            "respawn-leak"
            if is_continuous_visible(initial)
            else "reset-leak"
        )
    return None


def _observation_start_value(observation: object) -> object:
    if not isinstance(observation, dict):
        return None
    return {
        key: deepcopy(observation[key])
        for key in ("game", "status", "map", "grid", "player", "objects", "entities")
        if key in observation
    }


def _observation_start_signature(observation: object) -> str:
    value = _observation_start_value(observation)
    if value is None:
        return "unknown"
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def _visible_start_signature(record: dict | None) -> str:
    before = record.get("before") if isinstance(record, dict) else None
    return _observation_start_signature(before)


def _visible_starts_equal(left: dict, right: dict) -> bool:
    left_before = left.get("before") if isinstance(left, dict) else None
    right_before = right.get("before") if isinstance(right, dict) else None
    if is_continuous_visible(left_before) and is_continuous_visible(right_before):
        return continuous_observation_equal(
            _observation_start_value(left_before),
            _observation_start_value(right_before),
            tolerance=FLOAT_TOLERANCE,
        )
    return _visible_start_signature(left) == _visible_start_signature(right)


def _issue_location(
    steps: list[int],
    index: dict[int, dict],
    *,
    prefer_last: bool = False,
    explicit_action: object = None,
) -> tuple[str | None, str]:
    anchor = steps[-1] if prefer_last and steps else (steps[0] if steps else None)
    record = index.get(anchor) if anchor is not None else None
    decision = record.get("decision") if isinstance(record, dict) else None
    action = explicit_action if isinstance(explicit_action, str) else (
        decision.get("action") if isinstance(decision, dict) else None
    )
    return action, _visible_start_signature(record)


def _issue_aspects(anomaly: dict) -> list[str]:
    values: list[str] = []
    aspect = anomaly.get("aspect")
    if isinstance(aspect, str) and aspect:
        values.append(aspect)
    aspects = anomaly.get("aspects")
    if isinstance(aspects, list):
        for item in aspects:
            if isinstance(item, str) and item and item not in values:
                values.append(item)
    return values


def _merge_issue(existing: dict, incoming: dict) -> None:
    existing["evidence_steps"] = sorted(set(
        existing.get("evidence_steps", []) + incoming.get("evidence_steps", [])
    ))
    for key in (
        "aspects", "observed_signatures", "note_ids", "claims", "phase_sources",
    ):
        merged = list(existing.get(key, []))
        for value in incoming.get(key, []):
            if value not in merged:
                merged.append(deepcopy(value))
        if merged:
            existing[key] = merged
    if existing.get("observed_signature") is None and incoming.get("observed_signature") is not None:
        existing["observed_signature"] = incoming["observed_signature"]
    anomaly = existing.get("anomaly")
    if isinstance(anomaly, dict):
        anomaly["evidence_steps"] = list(existing["evidence_steps"])
        if existing.get("aspects"):
            anomaly["aspects"] = list(existing["aspects"])


def _deduplicate_issues(candidates: list[dict]) -> list[dict]:
    unique: dict[tuple[str, str | None, str], dict] = {}
    for candidate in candidates:
        if _normalize_type(candidate.get("type")) in CONTINUOUS_ANOMALY_TYPES:
            # A sweep can encounter the same missing collider from many floor
            # cells.  These are repeated sightings of one visible-evidence
            # defect class, not separate user-facing bugs.
            key = (
                candidate["type"],
                None,
                "continuous",
            )
        else:
            key = (
                candidate["type"],
                candidate.get("action"),
                candidate.get("start_signature", "unknown"),
            )
        if key in unique:
            _merge_issue(unique[key], candidate)
        else:
            unique[key] = candidate
    issues = list(unique.values())
    for number, issue in enumerate(issues, 1):
        issue["number"] = number
    return issues


def _outcome_for_record(record: dict) -> dict:
    return {
        "facets": _public_facets(record.get("after")),
        "feedback": _feedback(record.get("after")),
    }


def _phase_sources(steps: list[int], index: dict[int, dict]) -> list[str]:
    values: list[str] = []
    for step in steps:
        record = index.get(step)
        if not isinstance(record, dict):
            continue
        source = record.get("source")
        if not isinstance(source, str) or not source:
            source = "sweep" if record.get("type") in {"sweep_step", "sweep_reset"} else "llm"
        if source not in values:
            values.append(source)
    return values


def _anomaly_has_issue_evidence(
    kind: str,
    anomaly: dict,
    steps: list[int],
    index: dict[int, dict],
    initial: object,
    continuous_evidence: dict[str, set[int]] | None = None,
) -> bool:
    if kind == "reset_inconsistency":
        return isinstance(initial, dict) and any(
            index[step].get("type") in {"reset", "sweep_reset"}
            and not _observations_equal(
                index[step].get("after"), initial, ignore_tick=True
            )
            for step in steps
        )
    if kind == "cross_episode_nondeterminism":
        evidence = [index[step] for step in steps]
        return any(
            left.get("type") in {"step", "sweep_step"}
            and right.get("type") in {"step", "sweep_step"}
            and left.get("episode", left.get("episode_after", 0))
            != right.get("episode", right.get("episode_after", 0))
            and isinstance(left.get("decision"), dict)
            and isinstance(right.get("decision"), dict)
            and left["decision"].get("action") == right["decision"].get("action")
            and _visible_starts_equal(left, right)
            and not _outcomes_equal(left, right)
            for left_index, left in enumerate(evidence)
            for right in evidence[left_index + 1:]
        )
    if kind in CONTINUOUS_ANOMALY_TYPES:
        if continuous_evidence is not None:
            matching = set(steps) & continuous_evidence.get(kind, set())
            return bool(matching) and (
                kind != "passed_through_solid" or any(
                    passed_solid_geometry_supported(index[step], anomaly)
                    for step in matching
                )
            )
        ordered = sorted(index.values(), key=lambda row: row.get("step", 0))
        wanted = set(steps)
        detector = ContinuousSessionDetector()
        for record in ordered:
            observed = {
                _normalize_type(value.get("type"))
                for value in detector.anomalies(record)
                if isinstance(value, dict)
                and (_normalize_type(value.get("type")) != "passed_through_solid"
                     or passed_solid_geometry_supported(record, value))
            }
            if (record.get("step") in wanted and kind in observed
                    and (kind != "passed_through_solid"
                         or passed_solid_geometry_supported(record, anomaly))):
                return True
        return False
    if kind in {"map_object_inconsistency", "label_contradiction"}:
        stated = set(_issue_aspects(anomaly))
        for step in steps:
            record = index[step]
            observed = set(_map_inconsistency_aspects(record)) | set(
                _label_contradiction_aspects(record)
            )
            if observed and (not stated or observed & stated):
                return True
        return False
    return kind in ISSUE_ANOMALY_TYPES and bool(steps)


def _extract_triage(records: Sequence[dict]) -> tuple[list[dict], list[dict]]:
    index = _step_index(records)
    valid_steps = set(index)
    continuous_evidence: dict[str, set[int]] = {
        kind: set() for kind in CONTINUOUS_ANOMALY_TYPES
    }
    continuous_detector = ContinuousSessionDetector()
    for record in sorted(index.values(), key=lambda row: row.get("step", 0)):
        for anomaly in continuous_detector.anomalies(record):
            kind = _normalize_type(anomaly.get("type"))
            if (kind in continuous_evidence and type(record.get("step")) is int
                    and (kind != "passed_through_solid"
                         or passed_solid_geometry_supported(record, anomaly))):
                continuous_evidence[kind].add(record["step"])
    issue_candidates: list[dict] = []
    observations: list[dict] = []

    for note in _note_candidates(records):
        if note.get("kind", "mechanic") != "suspected_bug":
            continue
        steps = _evidence_steps(note, valid_steps)
        claim = note.get("claim") if isinstance(note.get("claim"), str) else "Suspected bug"
        title = claim.strip().rstrip(".")[:120] or "Suspected bug"
        if not steps:
            observations.append({
                "source": "suspected_bug",
                "type": "suspected_bug",
                "note_id": note.get("id"),
                "title": title,
                "claim": claim,
                "evidence_steps": [],
                "reason": "The note cites no valid executed step.",
            })
            continue
        evidence = [_compact_transition(index[step]) for step in steps]
        expected, actual = _note_expected_actual(evidence, claim)
        action, start_signature = _issue_location(
            steps, index, prefer_last=True
        )
        issue = {
            "source": "suspected_bug",
            "type": "suspected_bug",
            "note_id": note.get("id"),
            "note_ids": [note.get("id")] if isinstance(note.get("id"), str) else [],
            "title": title,
            "claim": claim,
            "claims": [claim],
            "evidence_steps": steps,
            "expected": expected,
            "actual": actual,
            "action": action,
            "start_signature": start_signature,
            "phase_sources": _phase_sources(steps, index),
        }
        issue["observed_signature"] = _observed_signature(issue, records)
        issue["observed_signatures"] = (
            [issue["observed_signature"]] if issue["observed_signature"] else []
        )
        issue_candidates.append(issue)

    initial = _header(records).get("initial_observation")
    for anomaly in _anomalies(records):
        kind = _normalize_type(anomaly.get("type"))
        steps = _evidence_steps(anomaly, valid_steps)
        evidence = [_compact_transition(index[step]) for step in steps]
        expected, actual = _anomaly_expected_actual(anomaly, evidence, initial)
        minimum_evidence = 2 if kind == "cross_episode_nondeterminism" else 1
        enough_evidence = len(steps) >= minimum_evidence
        validated = enough_evidence and _anomaly_has_issue_evidence(
            kind,
            anomaly,
            steps,
            index,
            initial,
            continuous_evidence,
        )
        if kind not in ISSUE_ANOMALY_TYPES or not validated:
            reason = (
                "A prediction mismatch is model uncertainty, not game evidence."
                if kind == "prediction_surprise"
                else "Expanded-box overlap lacks rounded capsule/box penetration evidence from the recorded segment."
                if kind == "passed_through_solid" and not validated
                else "The anomaly is not an issue type with validated executed evidence."
            )
            observations.append({
                "source": "software_anomaly",
                "type": kind,
                "title": _title_for_anomaly(kind, anomaly),
                "evidence_steps": steps,
                "expected": expected,
                "actual": actual,
                "anomaly": deepcopy(anomaly),
                "reason": reason,
            })
            continue
        action, start_signature = _issue_location(
            steps,
            index,
            prefer_last=False,
            explicit_action=anomaly.get("action"),
        )
        if kind == "reset_inconsistency":
            # Every reset promises the same session-defined episode start, so
            # pre-reset player positions must not fragment one reset defect.
            start_signature = _observation_start_signature(initial)
        aspects = _issue_aspects(anomaly)
        issue = {
            "source": "software_anomaly",
            "type": kind,
            "category": anomaly.get("category", kind),
            "title": _title_for_anomaly(kind, anomaly),
            "evidence_steps": steps,
            "expected": expected,
            "actual": actual,
            "anomaly": deepcopy(anomaly),
            "action": action,
            "start_signature": start_signature,
            "aspects": aspects,
            "phase_sources": _phase_sources(steps, index),
        }
        issue["observed_signature"] = _observed_signature(issue, records)
        issue["observed_signatures"] = (
            [issue["observed_signature"]] if issue["observed_signature"] else []
        )
        issue_candidates.append(issue)

    for number, observation in enumerate(observations, 1):
        observation["number"] = number
    return _deduplicate_issues(issue_candidates), observations


def extract_issues(recording: str | Path | Sequence[dict]) -> list[dict]:
    """Return only reportable issues, excluding non-counting observations."""
    records = _coerce_records(recording)
    issues, _ = _extract_triage(records)
    return issues


def extract_observations(recording: str | Path | Sequence[dict]) -> list[dict]:
    """Return tentative signals that must not count as issues or false positives."""
    records = _coerce_records(recording)
    _, observations = _extract_triage(records)
    return observations


def _actions_through(records: Sequence[dict], final_step: int) -> list[dict]:
    controls = []
    for record in _step_records(records):
        step = record.get("step")
        if type(step) is not int or step > final_step:
            continue
        decision = record.get("decision")
        action = decision.get("action") if isinstance(decision, dict) else None
        if not isinstance(action, str):
            continue
        controls.append({"source_step": step, "action": action})
    return controls


def _recorded_shortest_reproduction_candidates(
    records: Sequence[dict],
    issue: dict,
) -> list[list[dict]]:
    """Find shortest evidence paths in the session's observed state graph.

    This is a safe first reduction for long continuous sweeps: every edge is
    an action the session actually executed from the same visible state.  Each
    returned path keeps an original evidence action and is still verified in a
    fresh environment before use.
    """
    evidence_steps = {
        step for step in issue.get("evidence_steps", []) if type(step) is int
    }
    if not evidence_steps:
        return []
    steps = [record for record in _step_records(records)]
    by_step = {
        record.get("step"): record
        for record in steps
        if type(record.get("step")) is int
    }
    header = _header(records)
    initial = header.get("initial_observation")
    if not isinstance(initial, dict) and steps:
        initial = steps[0].get("before")
    if not isinstance(initial, dict):
        return []
    start_signature = _observation_start_signature(initial)
    candidates: list[list[dict]] = []
    seen_candidates: set[tuple[tuple[int, str], ...]] = set()

    for evidence_step in sorted(evidence_steps):
        target = by_step.get(evidence_step)
        decision = target.get("decision") if isinstance(target, dict) else None
        action = decision.get("action") if isinstance(decision, dict) else None
        before = target.get("before") if isinstance(target, dict) else None
        if not isinstance(action, str) or not isinstance(before, dict):
            continue
        goal_signature = _observation_start_signature(before)
        graph: dict[str, list[tuple[str, dict]]] = {}
        for record in steps:
            step = record.get("step")
            if type(step) is not int or step >= evidence_step:
                continue
            record_decision = record.get("decision")
            record_action = (
                record_decision.get("action")
                if isinstance(record_decision, dict) else None
            )
            record_before = record.get("before")
            record_after = record.get("after")
            if (
                not isinstance(record_action, str)
                or not isinstance(record_before, dict)
                or not isinstance(record_after, dict)
            ):
                continue
            left = _observation_start_signature(record_before)
            right = _observation_start_signature(record_after)
            graph.setdefault(left, []).append((right, {
                "source_step": step,
                "action": record_action,
            }))

        pending = deque([start_signature])
        previous: dict[str, tuple[str, dict] | None] = {
            start_signature: None
        }
        while pending and goal_signature not in previous:
            current = pending.popleft()
            for neighbor, control in graph.get(current, []):
                if neighbor in previous:
                    continue
                previous[neighbor] = current, control
                pending.append(neighbor)
        if goal_signature not in previous:
            continue

        path: list[dict] = []
        cursor = goal_signature
        while cursor != start_signature:
            link = previous.get(cursor)
            if link is None:
                path = []
                break
            parent, control = link
            path.append(control)
            cursor = parent
        path.reverse()
        path.append({"source_step": evidence_step, "action": action})
        signature = tuple(
            (control["source_step"], control["action"])
            for control in path
        )
        if signature not in seen_candidates:
            seen_candidates.add(signature)
            candidates.append(path)
    candidates.sort(key=lambda value: (len(value), [
        item["source_step"] for item in value
    ]))
    return candidates


def _issue_target_position(issue: dict) -> tuple[float, float] | None:
    """Return the visible horizontal location associated with an issue."""
    anomaly = issue.get("anomaly")
    anomaly = anomaly if isinstance(anomaly, dict) else {}
    actual = anomaly.get("actual")
    actual = actual if isinstance(actual, dict) else issue.get("actual")
    actual = actual if isinstance(actual, dict) else {}
    position = actual.get("after_position")
    if not (
        isinstance(position, (list, tuple))
        and len(position) >= 3
        and all(
            not isinstance(value, bool)
            and isinstance(value, (int, float))
            and math.isfinite(float(value))
            for value in position[:3]
        )
    ):
        expected = anomaly.get("expected")
        expected = expected if isinstance(expected, dict) else {}
        floor = expected.get("floor_entity")
        floor = floor if isinstance(floor, dict) else {}
        position = floor.get("position")
    if not (
        isinstance(position, (list, tuple))
        and len(position) >= 3
        and all(
            not isinstance(value, bool)
            and isinstance(value, (int, float))
            and math.isfinite(float(value))
            for value in position[:3]
        )
    ):
        return None
    return float(position[0]), float(position[2])


def _recorded_subsequence_controls(
    recorded_controls: Sequence[dict],
    actions: Sequence[str],
) -> list[dict] | None:
    """Attach ordered recording provenance to a synthesized short path."""
    result: list[dict] = []
    cursor = 0
    for action in actions:
        while (
            cursor < len(recorded_controls)
            and recorded_controls[cursor].get("action") != action
        ):
            cursor += 1
        if cursor >= len(recorded_controls):
            return None
        result.append(deepcopy(recorded_controls[cursor]))
        cursor += 1
    return result


def _visible_target_reproduction_candidate(
    records: Sequence[dict],
    issue: dict,
    recorded_controls: Sequence[dict],
    *,
    seed: int,
    defect: str | None,
    game_factory: Callable[[int, str | None], object] | None,
) -> tuple[list[dict] | None, int]:
    """Search public controls for a short path back to a visible sink.

    Long physics sweeps often reach a floor defect only after scanning many
    unrelated cells, so deletion-only minimization can get trapped in a long
    locally minimal walk.  This bounded best-first search uses only advertised
    controls, visible state, and the issue's observed location.  Every returned
    action is also mapped to an ordered occurrence in the recording and is
    verified again by the caller in a fresh game.
    """
    if (
        game_factory is None
        or _normalize_type(issue.get("type")) != "sank_into_visible_floor"
    ):
        return None, 0
    target = _issue_target_position(issue)
    if target is None:
        return None, 0

    preferred = ("north", "south", "east", "west", "interact")
    recorded_actions = {
        control.get("action")
        for control in recorded_controls
        if isinstance(control.get("action"), str)
        and control.get("action") != "reset"
    }
    actions = [action for action in preferred if action in recorded_actions]
    actions.extend(sorted(recorded_actions - set(actions)))
    if not actions:
        return None, 0

    game = None
    attempts = 0
    try:
        game = game_factory(seed, defect)
        reset = getattr(game, "reset", None)
        observe = getattr(game, "observe", None)
        act = getattr(game, "act", None)
        if not callable(reset) or not callable(observe) or not callable(act):
            return None, attempts
        initial = deepcopy(observe())

        def heuristic(observation: object) -> float:
            if not isinstance(observation, dict):
                return 0.0
            player = observation.get("player")
            player = player if isinstance(player, dict) else {}
            position = player.get("position")
            if not (
                isinstance(position, (list, tuple))
                and len(position) >= 3
                and all(
                    not isinstance(value, bool)
                    and isinstance(value, (int, float))
                    and math.isfinite(float(value))
                    for value in (position[0], position[2])
                )
            ):
                return 0.0
            return math.hypot(
                float(position[0]) - target[0],
                float(position[2]) - target[1],
            )

        counter = itertools.count()
        frontier: list[tuple[float, int, int, tuple[str, ...], object]] = []
        heapq.heappush(
            frontier,
            (heuristic(initial), 0, next(counter), (), initial),
        )
        best_cost = {_observation_start_signature(initial): 0}

        while frontier and len(best_cost) <= _CONTINUOUS_TARGET_SEARCH_STATES:
            _, cost, _, path, terminal = heapq.heappop(frontier)
            terminal_signature = _observation_start_signature(terminal)
            if (
                cost != best_cost.get(terminal_signature)
                or cost >= _CONTINUOUS_TARGET_SEARCH_DEPTH
            ):
                continue
            for action in actions:
                candidate_actions = path + (action,)
                attempts += 1
                try:
                    returned = reset()
                    after = deepcopy(
                        returned if isinstance(returned, dict) else observe()
                    )
                    detector = ContinuousSessionDetector()
                    reproduced = False
                    for step, candidate_action in enumerate(candidate_actions, 1):
                        before = after
                        returned = act(candidate_action)
                        after = deepcopy(
                            returned if isinstance(returned, dict) else observe()
                        )
                        synthetic = {
                            "type": "step",
                            "step": step,
                            "decision": {"action": candidate_action},
                            "before": before,
                            "after": after,
                        }
                        if any(
                            _normalize_type(anomaly.get("type"))
                            == "sank_into_visible_floor"
                            for anomaly in detector.anomalies(synthetic)
                        ):
                            reproduced = True
                            break
                except (KeyError, TypeError, ValueError):
                    continue

                if reproduced:
                    controls = _recorded_subsequence_controls(
                        recorded_controls, candidate_actions
                    )
                    if controls is not None:
                        return controls, attempts
                    continue

                signature = _observation_start_signature(after)
                next_cost = cost + 1
                if next_cost >= best_cost.get(signature, math.inf):
                    continue
                best_cost[signature] = next_cost
                heapq.heappush(
                    frontier,
                    (
                        next_cost + heuristic(after),
                        next_cost,
                        next(counter),
                        candidate_actions,
                        after,
                    ),
                )
    except (KeyError, TypeError, ValueError):
        return None, attempts
    finally:
        close = getattr(game, "close", None)
        if callable(close):
            try:
                close()
            except (OSError, ValueError):
                pass
    return None, attempts


def _new_game(seed: int, defect: str | None):
    # Phase-2 games accept defect as a keyword.  The fallback keeps clean legacy
    # recordings usable with a pre-Phase-2 adapter implementation.
    if defect is None:
        try:
            return KeyGateGame(seed, defect=None)
        except TypeError:
            return KeyGateGame(seed)
    return KeyGateGame(seed, defect=defect)


def _execute(
    seed: int,
    defect: str | None,
    controls: Sequence[dict],
    game_factory: Callable[[int, str | None], object] | None,
) -> tuple[object, list[dict], str | None]:
    game = None
    try:
        game = game_factory(seed, defect) if game_factory is not None else _new_game(seed, defect)
        initial = deepcopy(game.observe())
        trace: list[dict] = []
        episode = 0
        for control in controls:
            before = deepcopy(game.observe())
            action = control["action"]
            episode_before = episode
            if action == "reset":
                reset = getattr(game, "reset", None)
                if not callable(reset):
                    return initial, trace, "Game adapter does not support reset()."
                returned = reset()
                after = deepcopy(returned if isinstance(returned, dict) else game.observe())
                episode += 1
            else:
                after = deepcopy(game.act(action))
            trace.append({
                "source_step": control["source_step"],
                "action": action,
                "before": _public_facets(before),
                "after": _public_facets(after),
                "feedback": _feedback(after),
                "episode_before": episode_before,
                "episode_after": episode,
                "_before_observation": before,
                "_after_observation": after,
            })
        return initial, trace, None
    except (KeyError, TypeError, ValueError) as exc:
        return {}, [], str(exc)
    finally:
        close = getattr(game, "close", None)
        if callable(close):
            try:
                close()
            except (OSError, ValueError):
                # Reproduction already has its result; a best-effort protocol
                # close must not replace it with a teardown error.
                pass


def _same_outcome(left: dict, right: dict) -> bool:
    left_raw = left.get("_after_observation")
    right_raw = right.get("_after_observation")
    if is_continuous_visible(left_raw) and is_continuous_visible(right_raw):
        return (
            continuous_observation_equal(
                left_raw,
                right_raw,
                tolerance=FLOAT_TOLERANCE,
                ignore_tick=True,
            )
            and left.get("feedback") == right.get("feedback")
        )
    return left.get("after") == right.get("after") and left.get("feedback") == right.get("feedback")


def _trace_reproduces(issue: dict, records: Sequence[dict], initial: object, trace: list[dict]) -> bool:
    by_step = {item["source_step"]: item for item in trace}
    evidence_steps = issue.get("evidence_steps", [])
    recorded_index = _step_index(records)
    kind = _normalize_type(issue.get("type"))
    if kind == "prediction_surprise" and evidence_steps:
        expected = issue.get("expected")
        if isinstance(expected, dict):
            expected = expected.get("position")
        return expected is not None and any(
            by_step[step].get("after", {}).get("position") != expected
            for step in evidence_steps if step in by_step
        )
    if kind == "cross_episode_nondeterminism" and len(evidence_steps) >= 2:
        candidates = [by_step[step] for step in evidence_steps if step in by_step]
        return any(
            left.get("episode_after") != right.get("episode_after")
            and left.get("action") == right.get("action")
            and (
                continuous_observation_equal(
                    _observation_start_value(left.get("_before_observation")),
                    _observation_start_value(right.get("_before_observation")),
                    tolerance=FLOAT_TOLERANCE,
                )
                if is_continuous_visible(left.get("_before_observation"))
                and is_continuous_visible(right.get("_before_observation"))
                else left.get("before") == right.get("before")
            )
            and not _same_outcome(left, right)
            for left_index, left in enumerate(candidates)
            for right in candidates[left_index + 1:]
        )
    if kind == "reset_inconsistency" and evidence_steps:
        return any(
            by_step[step].get("action") == "reset"
            and not _observations_equal(
                by_step[step].get("_after_observation"),
                initial,
                ignore_tick=True,
            )
            for step in evidence_steps if step in by_step
        )
    if kind in CONTINUOUS_ANOMALY_TYPES and evidence_steps:
        detector = ContinuousSessionDetector()
        for target in trace:
            synthetic = {
                "type": "step",
                "step": target.get("source_step", 0),
                "decision": {"action": target.get("action")},
                "before": target.get("_before_observation"),
                "after": target.get("_after_observation"),
            }
            anomalies = detector.anomalies(synthetic)
            # Continuous issues are deduplicated as defect classes across the
            # session.  A minimized fresh run reproduces the issue when the
            # same visible-evidence predicate recurs, even if deleting prefix
            # controls makes it recur on a different original source step.
            if any(
                _normalize_type(anomaly.get("type")) == kind
                and (kind != "passed_through_solid"
                     or passed_solid_geometry_supported(synthetic, anomaly))
                for anomaly in anomalies
            ):
                return True
        return False
    if kind == "map_object_inconsistency" and evidence_steps:
        expected_aspects = set(issue.get("aspects", []))
        for step in evidence_steps:
            target = by_step.get(step)
            if target is None:
                continue
            synthetic = {
                "decision": {"action": target["action"]},
                "before": target["_before_observation"],
                "after": target["_after_observation"],
            }
            aspects = set(_map_inconsistency_aspects(synthetic)) | set(
                _label_contradiction_aspects(synthetic)
            )
            if aspects and (not expected_aspects or aspects & expected_aspects):
                return True
        return False
    if kind == "label_contradiction" and evidence_steps:
        expected_aspects = set(issue.get("aspects", []))
        for step in evidence_steps:
            target = by_step.get(step)
            if target is None:
                continue
            synthetic = {
                "decision": {"action": target["action"]},
                "before": target["_before_observation"],
                "after": target["_after_observation"],
            }
            aspects = set(_label_contradiction_aspects(synthetic))
            if aspects and (not expected_aspects or aspects & expected_aspects):
                return True
        return False

    # A known behavior signature can make a model-authored bug note precise,
    # but it must not bypass an anomaly's stricter relational predicate (for
    # example, cross-episode nondeterminism requires two distinct episodes).
    signature = issue.get("observed_signature")
    if issue.get("source") == "suspected_bug" and signature:
        return any(
            _transition_signature(by_step[step], initial) == signature
            for step in evidence_steps
            if step in by_step
        )

    # A free-form suspected note without a recognized defect signature recurs
    # when at least one cited executed outcome recurs.  This keeps merged
    # sightings reproducible when minimization retains just one occurrence.
    if issue.get("source") == "suspected_bug" and evidence_steps:
        for step in evidence_steps:
            replayed = by_step.get(step)
            recorded = recorded_index.get(step)
            if replayed is None or recorded is None:
                continue
            compact = _compact_transition(recorded)
            replayed_raw = replayed.get("_after_observation")
            recorded_raw = recorded.get("after")
            if (
                is_continuous_visible(replayed_raw)
                and is_continuous_visible(recorded_raw)
                and _observations_equal(
                    replayed_raw, recorded_raw, ignore_tick=True
                )
                and replayed.get("feedback") == compact.get("feedback")
            ):
                return True
            if (
                replayed.get("after") == compact.get("after")
                and replayed.get("feedback") == compact.get("feedback")
            ):
                return True
        return False
    return False


def _reproduction_plan(
    records: Sequence[dict],
    issue: dict,
    *,
    minimize: bool,
    game_factory: Callable[[int, str | None], object] | None,
) -> dict:
    evidence_steps = issue.get("evidence_steps", [])
    if not evidence_steps:
        return {
            "controls": [],
            "original_count": 0,
            "reproduced": False,
            "error": "Issue has no executable evidence steps.",
            "minimization_complete": False,
            "minimization_strategy": "unavailable",
            "verification_attempts": 0,
        }
    header = _header(records)
    seed = header.get("seed", 0)
    if type(seed) is not int:
        seed = 0
    ground_truth = _ground_truth(records)
    defect = ground_truth.get("name") if ground_truth else None
    kind = _normalize_type(issue.get("type"))
    # Continuous issues are merged across repeated encounters.  Reproduce the
    # first observed occurrence instead of replaying through the final repeat.
    evidence_anchor = (
        min(evidence_steps)
        if kind in CONTINUOUS_ANOMALY_TYPES
        else max(evidence_steps)
    )
    controls = _actions_through(records, evidence_anchor)
    original_count = len(controls)
    verification_attempts = 0

    def verifies(candidate: Sequence[dict]) -> tuple[bool, str | None]:
        nonlocal verification_attempts
        verification_attempts += 1
        initial, trace, error = _execute(seed, defect, candidate, game_factory)
        if error is not None:
            return False, error
        return _trace_reproduces(issue, records, initial, trace), None

    reproduced, error = verifies(controls)
    minimization_complete = not minimize
    minimization_strategy = "not_requested" if not minimize else "drop_chunks"
    if reproduced and minimize:
        initial_observation = _header(records).get("initial_observation")
        long_continuous = (
            original_count > _CONTINUOUS_EXACT_MINIMIZATION_LIMIT
            and is_continuous_visible(initial_observation)
        )
        if long_continuous:
            # A bounded public-control search can escape long, locally minimal
            # sweep walks around a visible sink.  It uses no developer rules or
            # supplied route, and the result is still confirmed in a fresh
            # player before it replaces the recording prefix.
            graph_candidate_verified = False
            candidate, search_attempts = _visible_target_reproduction_candidate(
                records,
                issue,
                controls,
                seed=seed,
                defect=defect,
                game_factory=game_factory,
            )
            verification_attempts += search_attempts
            if candidate is not None:
                candidate_reproduced, candidate_error = verifies(candidate)
                if candidate_reproduced:
                    controls = candidate
                    error = candidate_error
                    graph_candidate_verified = True
                    minimization_strategy = "visible_target_search"

            # For other continuous issues, or if bounded search cannot find a
            # path, build a shortest route from transitions already observed in
            # this session and confirm it against a fresh player.
            if not graph_candidate_verified:
                minimization_strategy = "recorded_state_graph"
                for candidate in _recorded_shortest_reproduction_candidates(
                    records, issue
                )[:4]:
                    candidate_reproduced, candidate_error = verifies(candidate)
                    if candidate_reproduced:
                        controls = candidate
                        error = candidate_error
                        graph_candidate_verified = True
                        break

            # A verified search or recorded-state path is already compact, so
            # a complete drop-one pass now needs at most that path's length in
            # fresh launches, not hundreds.
            # If an adapter's floating state prevented graph matching, retain
            # an explicit fallback bound and report incomplete minimization.
            attempts = 0
            index = 0
            while (
                index < len(controls)
                and (
                    graph_candidate_verified
                    or attempts < _CONTINUOUS_FALLBACK_MINIMIZATION_ATTEMPTS
                )
            ):
                candidate = controls[:index] + controls[index + 1:]
                candidate_reproduced, _ = verifies(candidate)
                attempts += 1
                if candidate_reproduced:
                    controls = candidate
                else:
                    index += 1
            minimization_complete = index >= len(controls)
        else:
            # Remove large contiguous chunks first.  The final pass makes the
            # result 1-minimal for ordinary-sized recordings.
            partitions = 2
            while len(controls) >= 2:
                chunk_size = math.ceil(len(controls) / partitions)
                reduced = False
                for start in range(0, len(controls), chunk_size):
                    candidate = controls[:start] + controls[start + chunk_size:]
                    candidate_reproduced, _ = verifies(candidate)
                    if candidate_reproduced:
                        controls = candidate
                        partitions = max(2, partitions - 1)
                        reduced = True
                        break
                if reduced:
                    continue
                if partitions >= len(controls):
                    break
                partitions = min(len(controls), partitions * 2)

            index = 0
            while index < len(controls):
                candidate = controls[:index] + controls[index + 1:]
                candidate_reproduced, _ = verifies(candidate)
                if candidate_reproduced:
                    controls = candidate
                else:
                    index += 1
            minimization_complete = True
        reproduced, error = verifies(controls)
    return {
        "controls": controls,
        "original_count": original_count,
        "reproduced": reproduced,
        "error": error,
        "minimization_complete": minimization_complete,
        "minimization_strategy": minimization_strategy,
        "verification_attempts": verification_attempts,
    }


def reproduce_issue(
    recording: str | Path | Sequence[dict],
    issue_number: int,
    *,
    minimize: bool = True,
    game_factory: Callable[[int, str | None], object] | None = None,
) -> dict:
    """Replay and minimize one issue; ``exit_code`` is 0 only on recurrence."""
    records = _coerce_records(recording)
    issues = extract_issues(records)
    if type(issue_number) is not int or not 1 <= issue_number <= len(issues):
        raise ValueError(f"Issue number must be between 1 and {len(issues)}.")
    issue = issues[issue_number - 1]
    plan = _reproduction_plan(
        records, issue, minimize=minimize, game_factory=game_factory
    )
    result = {
        "issue": issue_number,
        "source": issue["source"],
        "phase_sources": list(issue.get("phase_sources", [])),
        "type": issue["type"],
        "title": issue["title"],
        "evidence_steps": list(issue["evidence_steps"]),
        "actions": [control["action"] for control in plan["controls"]],
        "source_steps": [control["source_step"] for control in plan["controls"]],
        "original_action_count": plan["original_count"],
        "minimized_action_count": len(plan["controls"]),
        "minimization_complete": plan["minimization_complete"],
        "minimization_strategy": plan["minimization_strategy"],
        "verification_attempts": plan["verification_attempts"],
        "reproduced": plan["reproduced"],
        "exit_code": 0 if plan["reproduced"] else 1,
    }
    if plan["error"] is not None:
        result["error"] = plan["error"]
    return result


def _markdown_value(value: object) -> str:
    if isinstance(value, str):
        return value
    return "```json\n" + json.dumps(value, indent=2, sort_keys=True) + "\n```"


def generate_report(
    recording: str | Path | Sequence[dict],
    *,
    with_ground_truth: bool = False,
    minimize: bool = True,
    game_factory: Callable[[int, str | None], object] | None = None,
    reproduction_results: Sequence[dict] | None = None,
) -> str:
    """Generate one Markdown document containing every extracted issue."""
    records = _coerce_records(recording)
    header = _header(records)
    issues, observations = _extract_triage(records)
    game_id = header.get("game_version")
    if not isinstance(game_id, str):
        initial = header.get("initial_observation")
        game_id = initial.get("game", "unknown") if isinstance(initial, dict) else "unknown"
    seed = header.get("seed", "unknown")
    lines = [
        "# AutoPlay QA bug report",
        "",
        f"Environment: game `{game_id}`; seed `{seed}`.",
    ]
    if with_ground_truth:
        truth = _ground_truth(records)
        if truth is None:
            lines.extend(["", "Ground truth: clean control (no injected defect)."]) 
        else:
            description = truth.get("description")
            detail = f" — {description}" if isinstance(description, str) and description else ""
            lines.extend(["", f"Ground truth: `{truth['name']}`{detail}"])
    if not issues:
        lines.extend(["", "No reportable issues were recorded."])

    supplied_reproductions = {
        result.get("issue"): result
        for result in (reproduction_results or ())
        if isinstance(result, dict) and type(result.get("issue")) is int
    }

    for issue in issues:
        supplied = supplied_reproductions.get(issue["number"])
        if supplied is None:
            plan = _reproduction_plan(
                records, issue, minimize=minimize, game_factory=game_factory
            )
            actions = [control["action"] for control in plan["controls"]]
        else:
            actions = [
                action for action in supplied.get("actions", [])
                if isinstance(action, str)
            ]
            plan = {
                "minimization_complete": supplied.get(
                    "minimization_complete", not minimize
                )
            }
        reproduction_heading = (
            "Minimal reproduction"
            if plan.get("minimization_complete") is True
            else "Reduced reproduction"
        )
        lines.extend([
            "",
            f"## Issue {issue['number']}: {issue['title']}",
            "",
            f"Source: `{issue['source']}` / `{issue['type']}`.",
            "",
            "Phase source: " + ", ".join(
                f"`{source}`" for source in issue.get("phase_sources", [])
            ) + ".",
        ])
        if issue.get("category") == "label_contradiction":
            details = [
                _label_contradiction_reason(aspect)
                for aspect in issue.get("aspects", [])
                if aspect in LABEL_CONTRADICTION_ASPECTS
            ]
            lines.extend([
                "",
                "Classification: visible label/feedback contradiction.",
            ])
            if details:
                lines.extend(["", "Observed contradiction: " + "; ".join(details) + "."])
        lines.extend([
            "",
            f"### {reproduction_heading}",
            "",
            "Start a fresh environment and perform:",
            "",
        ])
        if actions:
            lines.extend(
                f"{index}. `{action}`" for index, action in enumerate(actions, 1)
            )
        else:
            lines.append("No executable action sequence was recorded.")
        if plan.get("minimization_complete") is False:
            lines.extend([
                "",
                "This long continuous trace was reduced with a bounded "
                "fresh-environment verification budget; the sequence is "
                "reproducing but may admit further reduction.",
            ])
        lines.extend([
            "",
            "### Expected",
            "",
            _markdown_value(issue["expected"]),
            "",
            "### Actual",
            "",
            _markdown_value(issue["actual"]),
            "",
            "### Evidence",
            "",
            "Recording steps: " + (
                ", ".join(str(step) for step in issue["evidence_steps"])
                if issue["evidence_steps"] else "none"
            ) + ".",
        ])
    if observations:
        lines.extend([
            "",
            "## Appendix: Observations (not issues)",
            "",
            "These signals are retained for review but are not counted as issues or false positives.",
            "",
        ])
        for observation in observations:
            steps = observation.get("evidence_steps", [])
            evidence = ", ".join(str(step) for step in steps) if steps else "none"
            lines.append(
                f"- Observation {observation['number']}: {observation['title']} "
                f"(`{observation['source']}` / `{observation['type']}`; steps: {evidence}). "
                f"{observation.get('reason', '')}".rstrip()
            )
    lines.append("")
    return "\n".join(lines)


def write_report(
    recording: str | Path | Sequence[dict],
    out: str | Path | None = None,
    *,
    with_ground_truth: bool = False,
    minimize: bool = True,
    game_factory: Callable[[int, str | None], object] | None = None,
) -> Path:
    """Write a Markdown report and return its resolved output path."""
    if out is None:
        if not isinstance(recording, (str, Path)):
            raise ValueError("An output path is required for in-memory recordings.")
        source = Path(recording)
        out = source.with_suffix(".report.md")
    destination = Path(out)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        generate_report(
            recording,
            with_ground_truth=with_ground_truth,
            minimize=minimize,
            game_factory=game_factory,
        ),
        encoding="utf-8",
    )
    return destination.resolve()


def _normalize_defect_name(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    return value.strip().lower().replace("_", "-")


def _issue_matches_defect(issue: dict, defect: str) -> bool:
    """Match only executed evidence shapes plausibly caused by each defect."""
    signatures = {
        _normalize_defect_name(value)
        for value in (
            issue.get("observed_signature"),
            *issue.get("observed_signatures", []),
        )
        if _normalize_defect_name(value) is not None
    }
    if issue.get("source") == "suspected_bug":
        equivalent_signatures = {
            "floor-gap": {"floor-gap"},
            "door-ghost-collider": {"door-ghost-collider"},
            "key-not-consumed": {"key-not-consumed", "gate-keeps-key"},
            "respawn-leak": {"respawn-leak", "reset-leak"},
            "wall-tunnel": {"wall-tunnel", "wall-clip"},
        }.get(defect, {defect})
        return bool(signatures & equivalent_signatures)

    kind = _normalize_type(issue.get("type"))
    aspects = set(issue.get("aspects", []))
    if defect in {"reset-leak", "respawn-leak"}:
        return kind == "reset_inconsistency"
    continuous_matches = {
        "floor-gap": {
            "fell_through_visible_floor",
            "sank_into_visible_floor",
            "player_stuck_in_geometry",
        },
        "door-ghost-collider": {"blocked_no_visible_obstacle"},
        "wall-tunnel": {"passed_through_solid"},
    }
    if defect in continuous_matches:
        return kind in continuous_matches[defect]
    if kind != "map_object_inconsistency":
        return False
    return {
        "wall-clip": "entered_wall_labeled_tile" in aspects,
        "phantom-block": "blocked_on_floor_labeled_tile" in aspects,
        "gate-keeps-key": "key_use_feedback_inventory" in aspects,
        "key-not-consumed": "key_use_feedback_inventory" in aspects,
        "exit-no-win": "exit_status_unchanged" in aspects,
    }.get(defect, False)


def evaluate_recording(
    recording: str | Path | Sequence[dict],
    *,
    verify_reproduction: bool = True,
    game_factory: Callable[[int, str | None], object] | None = None,
) -> dict:
    """Compare extracted issue evidence with hidden injected-defect metadata."""
    records = _coerce_records(recording)
    truth = _ground_truth(records)
    truth_name = _normalize_defect_name(truth.get("name")) if truth else None
    issues, observations = _extract_triage(records)
    matched = [
        issue for issue in issues
        if truth_name is not None
        and _issue_matches_defect(issue, truth_name)
    ]
    matched_numbers = {issue["number"] for issue in matched}
    reproduction: list[dict] = []
    if verify_reproduction:
        for issue in issues:
            reproduction.append(reproduce_issue(
                records, issue["number"], game_factory=game_factory
            ))
    reproduced_by_number = {
        result["issue"]: result["reproduced"] for result in reproduction
    }
    flagged = [issue for issue in issues if issue["source"] == "suspected_bug"]
    false_positives = [
        issue for issue in issues if issue["number"] not in matched_numbers
    ]
    hit = bool(matched) if truth_name is not None else None
    return {
        "recording": str(recording) if isinstance(recording, (str, Path)) else "<memory>",
        "game": _header(records).get("game_version", "unknown"),
        "seed": _header(records).get("seed"),
        "injected_defect": truth_name,
        "detection": "hit" if hit else ("miss" if hit is False else "clean"),
        "hit": hit,
        "issue_count": len(issues),
        "observation_count": len(observations),
        "flagged_issue_count": len(flagged),
        "matched_issue_numbers": sorted(matched_numbers),
        "false_positive_count": len(false_positives),
        "false_positive_issue_numbers": [issue["number"] for issue in false_positives],
        "repro_verified": bool(matched_numbers) and all(
            reproduced_by_number.get(number, False) for number in matched_numbers
        ),
        "issues": [
            {
                "number": issue["number"],
                "source": issue["source"],
                "type": issue["type"],
                "matched": issue["number"] in matched_numbers,
                "reproduced": reproduced_by_number.get(issue["number"]),
                "phase_sources": list(issue.get("phase_sources", [])),
            }
            for issue in issues
        ],
        "observations": [
            {
                "number": observation["number"],
                "source": observation["source"],
                "type": observation["type"],
                "evidence_steps": list(observation.get("evidence_steps", [])),
            }
            for observation in observations
        ],
    }


def evaluate_recordings(
    recordings: Iterable[str | Path | Sequence[dict]],
    *,
    verify_reproduction: bool = True,
    game_factory: Callable[[int, str | None], object] | None = None,
) -> dict:
    """Evaluate multiple recordings and include small aggregate totals."""
    results = [
        evaluate_recording(
            recording,
            verify_reproduction=verify_reproduction,
            game_factory=game_factory,
        )
        for recording in recordings
    ]
    return {
        "recordings": results,
        "totals": {
            "recordings": len(results),
            "defect_recordings": sum(result["injected_defect"] is not None for result in results),
            "hits": sum(result["detection"] == "hit" for result in results),
            "misses": sum(result["detection"] == "miss" for result in results),
            "false_positives": sum(result["false_positive_count"] for result in results),
            "repro_verified": sum(result["repro_verified"] for result in results),
        },
    }


# Concise aliases for CLI wiring and external callers.
report_recording = generate_report
reproduce = reproduce_issue
evaluate = evaluate_recordings
