"""Lean exploration: models choose actions and tentative notes; software owns facts."""

from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import copy
import json
from math import dist, isfinite
from statistics import median
import time
from typing import Callable

from .continuous import (
    continuous_anomalies,
    continuous_observation_equal,
    is_continuous_visible,
)
from .exploration import EXPLORATION_FOCUSES, ExplorationLimits, Coverage, exploration_view
from .game import ACTIONS, DIRECTIONS, GameAdapter, evaluate_transition
from .nebius import InvalidDecision, NebiusClient, NebiusError, TransientNebiusError
from .session import SessionLog, redact_secrets


LEAN_PROMPT_VERSION = "lean-v1"
LEAN_CONTROLS = (*ACTIONS, "reset", "finish")
LEAN_NOTE_STATUSES = ("uncertain", "supported", "contradicted")
LEAN_NOTE_KINDS = ("mechanic", "suspected_bug")
LEAN_NOTE_LIMIT = 6
LEAN_EVIDENCE_LIMIT = 6
LEAN_TEXT_LIMIT = 200
LEAN_TRUNCATION_DIAGNOSTIC_CHARS = 300
# Kept for old recording replay; new request contexts use the lean-only cap below.
LEAN_HISTORY_SIZE = 6
LEAN_CONTEXT_HISTORY_SIZE = 8
LEAN_CROSS_EPISODE_PAIR_LIMIT = 12
LEAN_ANOMALY_LIMIT = 8
LEAN_COVERAGE_NEAREST_LIMIT = 8
LEAN_CONTINUOUS_HISTORY_SIZE = 4
LEAN_CONTINUOUS_PROMPT_MAX_BYTES = 12_000
LEAN_CONTINUOUS_ENTITY_LIMIT = 96
DEFAULT_DIRECT_FALLBACK_MAX_TOKENS = 1536
DEFAULT_TRANSIENT_RETRY_BACKOFF_SECONDS = 0.5
LEAN_SLOW_THINKING_TIMEOUT_FRACTION = 0.6
LEAN_TIME_PRESSURE_MULTIPLIER = 2.0
LEAN_REASON_FALLBACK = "No valid reason supplied."
LEAN_QA_CHECKLIST = """Probe boundaries: move into walls, edges, and objects. Try each control in different situations.
Prefer unattempted controls at visited positions before finishing.
Revisit objects after their state changes.
Use reset to check the start is restored and earlier results repeat.
If something you observed contradicts what the visible map, labels, or an earlier identical attempt showed, record it as a suspected_bug note.
Note anything inconsistent."""
LEAN_CONTINUOUS_QA_CHECKLIST = """Probe visible solid bounds, floor edges, and dynamic objects. Try each direction from sampled floor cells.
Prefer untried directions at visited floor cells before finishing.
Revisit nearby objects after their visible state changes.
Use reset to check the start is restored and earlier results repeat.
If movement or feedback contradicts visible entities or an earlier identical attempt, record a suspected_bug note.
Note anything inconsistent."""
LEAN_CONTINUOUS_PROMPT_SUFFIX = """
This is a continuous world. Positions are [x,y,z], horizontal movement is in x/z,
and static_solids is a compact visible-bounds table. Dynamic entities and player
state are shown separately. Use null for expected_position because the response
contract's legacy two-integer prediction field does not represent continuous 3D
coordinates. Coverage uses visited floor-tile cells and tried directions.
"""


def lean_schema(allowed: list[str] | tuple[str, ...] = LEAN_CONTROLS) -> dict:
    """Return the small, phase-independent model response schema."""
    return {
        "type": "object",
        "properties": {
            "action": {"type": "string", "enum": list(allowed)},
            "reason": {"type": "string", "minLength": 1, "maxLength": LEAN_TEXT_LIMIT},
            "expected_position": {
                "type": ["array", "null"],
                "items": {"type": "integer"},
                "minItems": 2,
                "maxItems": 2,
            },
            "notes": {
                "type": "array",
                "maxItems": LEAN_NOTE_LIMIT,
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "string", "minLength": 1, "maxLength": 32},
                        "claim": {"type": "string", "minLength": 1, "maxLength": LEAN_TEXT_LIMIT},
                        "status": {"type": "string", "enum": list(LEAN_NOTE_STATUSES)},
                        "kind": {"type": "string", "enum": list(LEAN_NOTE_KINDS)},
                        "evidence_steps": {
                            "type": "array",
                            "maxItems": LEAN_EVIDENCE_LIMIT,
                            "items": {"type": "integer", "minimum": 1},
                        },
                    },
                    "required": ["id", "claim", "status", "evidence_steps"],
                    "additionalProperties": False,
                },
            },
        },
        "required": ["action", "reason", "expected_position", "notes"],
        "additionalProperties": False,
    }


LEAN_SCHEMA = lean_schema()

LEAN_PROMPT = """You are AutoPlay QA's mechanics explorer. Choose one experiment and one
next action from the allowed controls. Learn only from the visible observation and
the host-computed factual log. The host executes only action; reason and notes are
tentative model text. expected_position predicts the immediate [x,y] after the
chosen action, or null when you do not predict it. Notes are optional updates to
session memory. Reuse an existing id to update its note; invent a new id only when
note_slots_remaining is positive. A note's optional kind is mechanic by default;
use suspected_bug only for behavior you think may be wrong. Keep claims narrow
and observable. Use supported only when cited feedback backs the claim,
contradicted only when it refutes the claim, and uncertain otherwise. Cite only
completed factual-log step numbers. Use reset when you choose to compare a result
in a later episode, and finish when further testing is not useful. Return only the
four JSON fields in response_contract. Decide quickly; one short paragraph of
thinking is enough. Keep it brief.
"""


@dataclass(frozen=True)
class LeanNote:
    id: str
    claim: str
    status: str
    evidence_steps: tuple[int, ...]
    kind: str = "mechanic"


def note_dict(note: LeanNote) -> dict:
    return {**asdict(note), "evidence_steps": list(note.evidence_steps)}


def _warning(code: str, message: str) -> dict:
    return {"code": code, "message": message}


def _unique_warnings(warnings: list[dict]) -> list[dict]:
    """Preserve warning order while collapsing exact repeats in one response."""
    unique = []
    seen = set()
    for warning in warnings:
        identity = (warning.get("code"), warning.get("message"))
        if identity in seen:
            continue
        seen.add(identity)
        unique.append(warning)
    return unique


@dataclass(frozen=True)
class LeanDecision:
    action: str
    reason: str
    expected_position: list[int] | None
    notes: tuple[LeanNote, ...]

    @classmethod
    def parse(
        cls,
        content: str,
        allowed: list[str],
        completed_steps: set[int],
        *,
        evidence_limit: int | None = LEAN_EVIDENCE_LIMIT,
        text_limit: int = LEAN_TEXT_LIMIT,
    ) -> tuple["LeanDecision", list[dict]]:
        """Parse leniently: only the absence of a legal action rejects content."""
        if evidence_limit is not None and (type(evidence_limit) is not int or evidence_limit <= 0):
            raise ValueError("Evidence limit must be a positive integer or None.")
        if type(text_limit) is not int or text_limit <= 0:
            raise ValueError("Text limit must be a positive integer.")
        try:
            value = json.loads(content)
        except (json.JSONDecodeError, TypeError):
            raise InvalidDecision("Response did not provide a usable legal action in a JSON object.") from None
        if not isinstance(value, dict):
            raise InvalidDecision("Response did not provide a usable legal action in a JSON object.")
        action = value.get("action")
        if not isinstance(action, str) or action not in allowed:
            raise InvalidDecision("Response action is missing or is not currently legal.")

        warnings: list[dict] = []
        unknown_count = len(set(value) - {"action", "reason", "expected_position", "notes"})
        if unknown_count:
            warnings.append(_warning("ignored_top_level_fields", f"Ignored {unknown_count} unknown top-level field(s)."))

        reason_value = value.get("reason")
        if not isinstance(reason_value, str):
            reason = LEAN_REASON_FALLBACK
            warnings.append(_warning("invalid_reason", "Replaced a non-string or missing reason with a short fallback."))
        else:
            reason = reason_value.strip()
            if not reason:
                reason = LEAN_REASON_FALLBACK
                warnings.append(_warning("invalid_reason", "Replaced an empty reason with a short fallback."))
            elif len(reason) > text_limit:
                reason = reason[:text_limit]
                warnings.append(_warning(
                    "trimmed_reason",
                    f"Trimmed reason to {text_limit} characters.",
                ))

        expected = value.get("expected_position")
        valid_position = (
            isinstance(expected, list)
            and len(expected) == 2
            and all(type(coordinate) is int for coordinate in expected)
        )
        if expected is None:
            expected_position = None
        elif valid_position:
            expected_position = list(expected)
        else:
            expected_position = None
            warnings.append(_warning("invalid_expected_position", "Replaced invalid expected_position with null."))

        raw_notes = value.get("notes")
        if not isinstance(raw_notes, list):
            raw_notes = []
            warnings.append(_warning("invalid_notes", "Dropped notes because notes was not an array."))
        if len(raw_notes) > LEAN_NOTE_LIMIT:
            warnings.append(_warning("too_many_notes", f"Dropped {len(raw_notes) - LEAN_NOTE_LIMIT} note(s) beyond the six-note response limit."))

        notes: list[LeanNote] = []
        seen_ids: set[str] = set()
        for note_index, item in enumerate(raw_notes[:LEAN_NOTE_LIMIT]):
            label = f"note[{note_index}]"
            if not isinstance(item, dict):
                warnings.append(_warning("invalid_note", f"Dropped {label}: note was not an object."))
                continue
            identifier = item.get("id")
            claim = item.get("claim")
            status = item.get("status")
            kind = item.get("kind", "mechanic")
            if not isinstance(identifier, str) or not 1 <= len(identifier.strip()) <= 32:
                warnings.append(_warning("invalid_note_id", f"Dropped {label}: id was not a 1-32 character string."))
                continue
            identifier = identifier.strip()
            if identifier in seen_ids:
                warnings.append(_warning("duplicate_note_id", f"Dropped {label}: its id duplicated an earlier update."))
                continue
            if not isinstance(claim, str) or not 1 <= len(claim.strip()) <= text_limit:
                warnings.append(_warning(
                    "invalid_note_claim",
                    f"Dropped {label}: claim was not a 1-{text_limit} character string.",
                ))
                continue
            if status not in LEAN_NOTE_STATUSES:
                warnings.append(_warning("invalid_note_status", f"Dropped {label}: status was not an allowed tentative status."))
                continue
            if kind not in LEAN_NOTE_KINDS:
                warnings.append(_warning("invalid_note_kind", f"Dropped {label}: kind was not mechanic or suspected_bug."))
                continue

            raw_evidence = item.get("evidence_steps")
            evidence: list[int] = []
            if not isinstance(raw_evidence, list):
                warnings.append(_warning("invalid_evidence_list", f"Dropped {label}'s evidence because it was not an array."))
            else:
                invalid_count = duplicate_count = excess_count = 0
                for step in raw_evidence:
                    if type(step) is not int or step not in completed_steps:
                        invalid_count += 1
                        continue
                    if step in evidence:
                        duplicate_count += 1
                        continue
                    if evidence_limit is not None and len(evidence) >= evidence_limit:
                        excess_count += 1
                        continue
                    evidence.append(step)
                if invalid_count or duplicate_count or excess_count:
                    details = []
                    if invalid_count:
                        details.append(f"{invalid_count} invalid")
                    if duplicate_count:
                        details.append(f"{duplicate_count} duplicate")
                    if excess_count:
                        details.append(f"{excess_count} beyond the {evidence_limit}-reference limit")
                    warnings.append(_warning(
                        "normalized_evidence_steps",
                        f"Normalized {label}'s evidence_steps; dropped " + ", ".join(details) + ".",
                    ))
            seen_ids.add(identifier)
            notes.append(LeanNote(identifier, claim.strip(), status, tuple(evidence), kind))

        return cls(action, reason, expected_position, tuple(notes)), _unique_warnings(warnings)


def decision_dict(decision: LeanDecision) -> dict:
    value = {
        "action": decision.action,
        "reason": decision.reason,
        "expected_position": copy.deepcopy(decision.expected_position),
        "notes": [note_dict(note) for note in decision.notes],
    }
    # Only versioned parsers produce this canonical optional metadata.  Legacy
    # decisions retain their original four-field recording contract.
    value.update(copy.deepcopy(getattr(decision, "adaptive_fields", {})))
    return value


def public_facets(observation: dict) -> dict:
    """Project only factual public state used by lean comparisons."""
    player = observation.get("player")
    player = player if isinstance(player, dict) else {}
    objects = observation.get("objects")
    objects = objects if isinstance(objects, dict) else {}
    gate = objects.get("gate")
    if "key" in objects and isinstance(gate, dict) and "open" in gate:
        # Preserve the historical KeyGate fact shape byte-for-byte.
        return {
            "position": copy.deepcopy(player.get("position")),
            "inventory": copy.deepcopy(player.get("inventory", [])),
            "key": copy.deepcopy(objects["key"]),
            "gate_open": gate["open"],
            "status": observation.get("status"),
        }
    entities = observation.get("entities")
    entities = entities if isinstance(entities, list) else []
    return {
        "position": copy.deepcopy(player.get("position")),
        "inventory": copy.deepcopy(player.get("inventory", [])),
        "entities": copy.deepcopy(entities),
        "status": observation.get("status"),
    }


def actual_changed_facets(before: dict, after: dict) -> list[str]:
    old, new = public_facets(before), public_facets(after)
    return [name for name in old if old[name] != new[name]]


def _visible_events(observation: dict) -> list[str]:
    if observation.get("tick") == 0:
        return ["Environment started."]
    return copy.deepcopy(observation.get("events", []))


def _observed_outcome(observation: dict) -> dict:
    return {
        "facets": public_facets(observation),
        "feedback": _visible_events(observation),
    }


def _glyph_at(observation: dict, position: list[int] | tuple[int, int]) -> str | None:
    if len(position) != 2 or not all(type(value) is int for value in position):
        return None
    x, y = position
    rows = observation.get("map", observation.get("grid", []))
    if not (0 <= y < len(rows) and isinstance(rows[y], str) and 0 <= x < len(rows[y])):
        return None
    return rows[y][x]


def _legend_meaning(observation: dict, glyph: str | None) -> str:
    if glyph is None:
        return ""
    meaning = observation.get("legend", {}).get(glyph, "")
    return meaning.lower() if isinstance(meaning, str) else ""


def _event_text(observation: dict) -> str:
    events = observation.get("events")
    if not isinstance(events, list):
        return ""
    return " ".join(item for item in events if isinstance(item, str)).lower()


def _grid_legend_enabled(observation: dict) -> bool:
    """Honor an explicit remote capability advertisement when one exists."""
    capabilities = observation.get("_capabilities")
    if not isinstance(capabilities, dict):
        return True  # Historical/built-in observations predate the handshake.
    return capabilities.get("grid") is True and capabilities.get("legend") is True


def _generic_entity(observation: dict, kind: str) -> dict:
    entities = observation.get("entities")
    if not isinstance(entities, list):
        return {}
    for entity in entities:
        if not isinstance(entity, dict):
            continue
        identifier = entity.get("id")
        entity_kind = entity.get("kind")
        if any(
            isinstance(value, str) and value.lower() == kind
            for value in (identifier, entity_kind)
        ):
            return entity
    return {}


def _entity_state(entity: dict) -> dict:
    state = entity.get("state") if isinstance(entity, dict) else None
    return state if isinstance(state, dict) else {}


def _map_object_anomalies(record: dict) -> list[dict]:
    """Describe visible map disagreements without consulting game rules or layout."""
    before, after = record["before"], record["after"]
    step = record["step"]
    action = record["decision"]["action"]
    base = {"type": "map_object_inconsistency", "step": step, "evidence_steps": [step], "action": action}
    anomalies: list[dict] = []

    rows = after.get("map", after.get("grid"))
    has_grid_legend = (
        _grid_legend_enabled(before)
        and _grid_legend_enabled(after)
        and bool(rows)
        and isinstance(rows, list)
        and isinstance(after.get("legend"), dict)
    )

    if has_grid_legend and action in DIRECTIONS:
        old_position = before["player"]["position"]
        dx, dy = DIRECTIONS[action]
        target = [old_position[0] + dx, old_position[1] + dy]
        glyph = _glyph_at(before, target)
        meaning = _legend_meaning(before, glyph)
        actual_position = after["player"]["position"]
        if "wall" in meaning and actual_position == target:
            anomalies.append({
                **base,
                "aspect": "entered_wall_labeled_tile",
                "expected": {"target": target, "glyph": glyph, "legend_meaning": meaning},
                "actual": {"position": copy.deepcopy(actual_position), "feedback": _visible_events(after)},
            })
        elif "floor" in meaning and actual_position == old_position:
            anomalies.append({
                **base,
                "aspect": "blocked_on_floor_labeled_tile",
                "expected": {"target": target, "glyph": glyph, "legend_meaning": meaning},
                "actual": {"position": copy.deepcopy(actual_position), "feedback": _visible_events(after)},
            })

    player = after.get("player")
    player = player if isinstance(player, dict) else {}
    position = player.get("position")
    if has_grid_legend and isinstance(position, (list, tuple)):
        player_glyph = _glyph_at(after, position)
        if "player" not in _legend_meaning(after, player_glyph):
            anomalies.append({
                **base,
                "aspect": "player_position_glyph",
                "expected": {"position": copy.deepcopy(position), "legend_meaning": "player"},
                "actual": {"glyph": player_glyph, "legend_meaning": _legend_meaning(after, player_glyph)},
            })

    objects = after.get("objects", {})
    object_specs = [
        ("key", objects.get("key"), "key"),
        ("exit", objects.get("exit"), "exit"),
    ]
    gate = objects.get("gate", {})
    gate_position = gate.get("position") if isinstance(gate, dict) else None
    gate_meaning = "open gate" if isinstance(gate, dict) and gate.get("open") is True else "gate"
    object_specs.append(("gate", gate_position, gate_meaning))
    for object_name, object_position, expected_meaning in object_specs:
        if not (
            has_grid_legend
            and
            isinstance(object_position, list)
            and len(object_position) == 2
            and all(type(value) is int for value in object_position)
        ):
            continue
        if object_position == position:
            continue  # The player glyph visibly overlays an object at its position.
        glyph = _glyph_at(after, object_position)
        meaning = _legend_meaning(after, glyph)
        matches = expected_meaning in meaning
        if object_name == "gate" and expected_meaning == "gate":
            matches = "gate" in meaning and "open" not in meaning
        if not matches:
            anomalies.append({
                **base,
                "aspect": f"{object_name}_object_glyph",
                "expected": {"position": copy.deepcopy(object_position), "legend_meaning": expected_meaning},
                "actual": {"glyph": glyph, "legend_meaning": meaning},
            })

    key_position = objects.get("key")
    inventory = after.get("player", {}).get("inventory", [])
    generic_key = _generic_entity(after, "key")
    generic_key_state = _entity_state(generic_key)
    key_visible_in_world = (
        generic_key.get("position") is not None
        and generic_key_state.get("present", True) is not False
        and generic_key_state.get("available", True) is not False
        and generic_key_state.get("collected") is not True
    ) if generic_key else key_position is not None
    if key_visible_in_world and isinstance(inventory, list) and "key" in inventory:
        anomalies.append({
            **base,
            "aspect": "key_listed_in_world_and_inventory",
            "expected": {"key_object_or_inventory": "one visible location"},
            "actual": {"key_object": copy.deepcopy(key_position), "inventory": copy.deepcopy(inventory)},
        })

    # These contradictions need only public feedback, inventory, and entity
    # state, so they remain available when a game advertises no grid/legend.
    feedback = _event_text(after)
    inventory_tokens = {
        str(item.get("kind", item.get("id", ""))).lower()
        if isinstance(item, dict) else str(item).lower()
        for item in inventory
    } if isinstance(inventory, list) else set()
    if (
        any(phrase in feedback for phrase in ("used the key", "consumed the key", "key was used"))
        and "key" in inventory_tokens
    ):
        anomalies.append({
            **base,
            "aspect": "key_use_feedback_inventory",
            "expected": {"inventory_excludes": "key"},
            "actual": {"inventory": copy.deepcopy(inventory), "feedback": _visible_events(after)},
        })
    if any(phrase in feedback for phrase in ("collected the key", "picked up the key")):
        key_state = _entity_state(generic_key)
        still_available = (
            generic_key.get("position") is not None
            and key_state.get("available", True) is not False
        ) if generic_key else objects.get("key") is not None
        if "key" not in inventory_tokens or still_available:
            anomalies.append({
                **base,
                "aspect": "key_collection_feedback_state",
                "expected": {"inventory_contains": "key", "world_key_available": False},
                "actual": {"inventory": copy.deepcopy(inventory), "feedback": _visible_events(after)},
            })
    generic_gate = _generic_entity(after, "gate")
    generic_gate_open = _entity_state(generic_gate).get("open")
    legacy_gate_open = gate.get("open") if isinstance(gate, dict) else None
    if (
        any(phrase in feedback for phrase in ("unlocked the gate", "opened the gate"))
        and (generic_gate_open if generic_gate else legacy_gate_open) is not True
    ):
        anomalies.append({
            **base,
            "aspect": "gate_open_feedback_state",
            "expected": {"gate_open": True},
            "actual": {"gate_open": generic_gate_open if generic_gate else legacy_gate_open, "feedback": _visible_events(after)},
        })
    return anomalies


def _reset_anomaly(record: dict, prior_records: list[dict] | tuple[dict, ...]) -> dict | None:
    if record.get("type") not in {"reset", "sweep_reset"}:
        return None
    initial = prior_records[0]["before"] if prior_records else record["before"]
    after = record["after"]
    keys = sorted((set(initial) | set(after)) - {"tick"})
    if is_continuous_visible(initial) and is_continuous_visible(after):
        changed = [
            key for key in keys
            if not continuous_observation_equal(
                initial.get(key), after.get(key), tolerance=0.05,
            )
        ]
    else:
        changed = [key for key in keys if initial.get(key) != after.get(key)]
    if not changed:
        return None
    step = record["step"]
    return {
        "type": "reset_inconsistency",
        "step": step,
        "evidence_steps": [step],
        "action": "reset",
        "different_fields_ignoring_tick": changed,
        "expected": _observed_outcome(initial),
        "actual": _observed_outcome(after),
    }


def _cross_episode_anomalies(record: dict, prior_records: list[dict] | tuple[dict, ...]) -> list[dict]:
    if record.get("type") != "step":
        return []
    repeated_episode = record.get("episode", record.get("episode_after", 0))
    anomalies = []
    for original in prior_records:
        if original.get("type") != "step":
            continue
        original_episode = original.get("episode", original.get("episode_after", 0))
        if original_episode == repeated_episode:
            continue
        if original["decision"]["action"] != record["decision"]["action"]:
            continue
        if not _same_start(original, record):
            continue
        if _same_outcome(original, record):
            continue
        anomalies.append({
            "type": "cross_episode_nondeterminism",
            "step": record["step"],
            "evidence_steps": [original["step"], record["step"]],
            "action": record["decision"]["action"],
            "prior": {"step": original["step"], "episode": original_episode, **_observed_outcome(original["after"])},
            "expected": {"same_start_as_step": original["step"], **_observed_outcome(original["after"])},
            "actual": {"step": record["step"], "episode": repeated_episode, **_observed_outcome(record["after"])},
        })
    return anomalies


def _step_anomalies(record: dict, prior_records: list[dict] | tuple[dict, ...]) -> list[dict]:
    decision = record["decision"]
    actual_position = record["after"].get("player", {}).get("position")
    expected_position = decision.get("expected_position")
    anomalies = []
    if expected_position is not None and actual_position != expected_position:
        anomalies.append({
            "type": "prediction_surprise",
            "step": record["step"],
            "evidence_steps": [record["step"]],
            "action": decision["action"],
            "expected": {"position": copy.deepcopy(expected_position)},
            "actual": {"position": copy.deepcopy(actual_position), **_observed_outcome(record["after"])},
        })
    anomalies.extend(_map_object_anomalies(record))
    anomalies.extend(continuous_anomalies(record, prior_records, tolerance=0.05))
    reset = _reset_anomaly(record, prior_records)
    if reset is not None:
        anomalies.append(reset)
    anomalies.extend(_cross_episode_anomalies(record, prior_records))
    return anomalies


def step_fact(
    record: dict,
    prior_records: list[dict] | tuple[dict, ...] = (),
    *,
    include_non_qa_reset_anomalies: bool = True,
) -> dict:
    decision = record["decision"]
    expected_position = copy.deepcopy(decision.get("expected_position"))
    actual_position = record["after"].get("player", {}).get("position")
    fact = {
        "step": record["step"],
        "action": decision["action"],
        "episode_before": record.get("episode_before", record.get("episode", 0)),
        "episode_after": record.get("episode_after", record.get("episode", 0)),
        "before": public_facets(record["before"]),
        "after": public_facets(record["after"]),
        "feedback": _visible_events(record["after"]),
        "actual_changed_facets": actual_changed_facets(record["before"], record["after"]),
        "expected_position": expected_position,
        "expected_position_matched": actual_position == expected_position if expected_position is not None else None,
    }
    if record.get("qa_mode") is True:
        fact["anomalies"] = _step_anomalies(record, prior_records)
    elif include_non_qa_reset_anomalies and record.get("type") == "reset":
        reset = _reset_anomaly(record, prior_records)
        if reset is not None:
            fact["anomalies"] = [reset]
    return fact


def _start_signature(record: dict) -> str:
    before = record["before"]
    value = {
        key: before[key]
        for key in ("game", "status", "map", "grid", "player", "objects", "entities")
        if key in before
    }
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _outcome_signature(record: dict) -> str:
    value = {"facets": public_facets(record["after"]), "feedback": _visible_events(record["after"])}
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _same_start(left: dict, right: dict) -> bool:
    left_before = left["before"]
    right_before = right["before"]
    if is_continuous_visible(left_before) and is_continuous_visible(right_before):
        keys = ("game", "status", "map", "grid", "player", "objects", "entities")
        left_value = {key: left_before[key] for key in keys if key in left_before}
        right_value = {key: right_before[key] for key in keys if key in right_before}
        return continuous_observation_equal(left_value, right_value, tolerance=0.05)
    return _start_signature(left) == _start_signature(right)


def _same_outcome(left: dict, right: dict) -> bool:
    left_after = left["after"]
    right_after = right["after"]
    if is_continuous_visible(left_after) and is_continuous_visible(right_after):
        left_value = {
            "facets": public_facets(left_after),
            "feedback": _visible_events(left_after),
        }
        right_value = {
            "facets": public_facets(right_after),
            "feedback": _visible_events(right_after),
        }
        return continuous_observation_equal(left_value, right_value, tolerance=0.05)
    return _outcome_signature(left) == _outcome_signature(right)


def cross_episode_pairs(records: list[dict]) -> list[dict]:
    """Compute comparable game-control pairs without model bookkeeping."""
    trials = [record for record in records if record.get("type") == "step"]
    pairs = []
    for repeated_index, repeated in enumerate(trials):
        repeated_episode = repeated.get("episode", repeated.get("episode_after", 0))
        for original in trials[:repeated_index]:
            original_episode = original.get("episode", original.get("episode_after", 0))
            if original_episode == repeated_episode:
                continue
            if original["decision"]["action"] != repeated["decision"]["action"]:
                continue
            if not _same_start(original, repeated):
                continue
            pairs.append({
                "original_step": original["step"],
                "repeated_step": repeated["step"],
                "action": repeated["decision"]["action"],
                "original_episode": original_episode,
                "repeated_episode": repeated_episode,
                "result": "same" if _same_outcome(original, repeated) else "different",
            })
    return pairs


def retention_facts(pairs: list[dict]) -> dict:
    counts: dict[str, Counter] = {}
    for pair in pairs:
        counts.setdefault(pair["action"], Counter())[pair["result"]] += 1
    return {
        "pair_count": len(pairs),
        "same_outcome_pairs": sum(pair["result"] == "same" for pair in pairs),
        "different_outcome_pairs": sum(pair["result"] == "different" for pair in pairs),
        "by_action": [
            {
                "action": action,
                "pair_count": sum(result_counts.values()),
                "same_outcome_pairs": result_counts["same"],
                "different_outcome_pairs": result_counts["different"],
            }
            for action, result_counts in sorted(counts.items())
        ],
    }


def software_facts(
    records: list[dict],
    history_size: int | None = LEAN_HISTORY_SIZE,
    *,
    include_non_qa_reset_anomalies: bool = True,
) -> dict:
    pairs = cross_episode_pairs(records)
    facts = [
        step_fact(
            record,
            records[:index],
            include_non_qa_reset_anomalies=include_non_qa_reset_anomalies,
        )
        for index, record in enumerate(records)
    ]
    if history_size is not None:
        facts = facts[-history_size:]
    result = {
        "factual_log": facts,
        "cross_episode_same_start_pairs": pairs,
        "retention_facts": retention_facts(pairs),
    }
    if any("anomalies" in fact for fact in facts):
        result["anomalies"] = [
            copy.deepcopy(anomaly)
            for fact in facts
            for anomaly in fact.get("anomalies", [])
        ]
    return result


def _older_factual_summary(facts: list[dict]) -> dict:
    counts = Counter()
    for fact in facts:
        before_status = fact["before"]["status"]
        after_status = fact["after"]["status"]
        if before_status != after_status:
            outcome = f"status:{after_status}"
        else:
            outcome = "changed" if fact["actual_changed_facets"] else "unchanged"
        counts[(fact["action"], outcome)] += 1
    return {
        "step_count": len(facts),
        "first_step": facts[0]["step"] if facts else None,
        "last_step": facts[-1]["step"] if facts else None,
        "counts_by_action_outcome": [
            [action, outcome, count]
            for (action, outcome), count in sorted(counts.items())
        ],
    }


def _anomaly_identity(anomaly: dict) -> str:
    volatile = {"step", "episode", "evidence_steps", "same_start_as_step"}

    def stable(value):
        if isinstance(value, dict):
            return {
                key: stable(item)
                for key, item in value.items()
                if key not in volatile
            }
        if isinstance(value, list):
            return [stable(item) for item in value]
        return value

    return json.dumps(stable(anomaly), sort_keys=True, separators=(",", ":"))


def _compact_anomalies(anomalies: list[dict]) -> tuple[list[dict], dict]:
    unique: dict[str, dict] = {}
    for anomaly in anomalies:
        identity = _anomaly_identity(anomaly)
        # Keep the newest evidence for a repeated anomaly and move it to the end.
        unique.pop(identity, None)
        unique[identity] = copy.deepcopy(anomaly)
    values = list(unique.values())
    included = values[-LEAN_ANOMALY_LIMIT:]
    return included, {
        "observed": len(anomalies),
        "unique": len(values),
        "included": len(included),
        "omitted": max(0, len(values) - len(included)),
    }


def compact_software_facts(
    records: list[dict],
    history_size: int = LEAN_CONTEXT_HISTORY_SIZE,
    *,
    include_non_qa_reset_anomalies: bool = True,
) -> dict:
    """Project bounded software facts for a lean request without changing evidence logs."""
    if type(history_size) is not int or history_size <= 0:
        raise ValueError("Lean context history size must be a positive integer.")
    full = software_facts(
        records,
        history_size=None,
        include_non_qa_reset_anomalies=include_non_qa_reset_anomalies,
    )
    all_facts = full["factual_log"]
    recent = copy.deepcopy(all_facts[-history_size:])
    for fact in recent:
        fact.pop("anomalies", None)
    pair_fields = [
        "original_step", "repeated_step", "action",
        "original_episode", "repeated_episode", "result",
    ]
    compact_pairs = [
        [pair[field] for field in pair_fields]
        for pair in full["cross_episode_same_start_pairs"][-LEAN_CROSS_EPISODE_PAIR_LIMIT:]
    ]
    result = {
        "context_compaction": {
            "version": 1,
            "factual_log_limit": history_size,
            "cross_episode_pair_limit": LEAN_CROSS_EPISODE_PAIR_LIMIT,
            "anomaly_limit": LEAN_ANOMALY_LIMIT,
        },
        "factual_log": recent,
        "earlier_factual_summary": _older_factual_summary(all_facts[:-history_size]),
        "cross_episode_pair_fields": pair_fields,
        "cross_episode_same_start_pairs": compact_pairs,
        "retention_facts": full["retention_facts"],
    }
    if "anomalies" in full:
        anomalies, summary = _compact_anomalies(full["anomalies"])
        result["anomalies"] = anomalies
        result["anomaly_summary"] = summary
    return result


def _continuous_visible(observation: object) -> bool:
    if not isinstance(observation, dict):
        return False
    capabilities = observation.get("_capabilities")
    return (
        isinstance(capabilities, dict)
        and capabilities.get("continuous") is True
        and capabilities.get("solids_visible") is True
    )


def _round_continuous_number(value: object) -> object:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return copy.deepcopy(value)
    if isinstance(value, int):
        return value
    if not isfinite(float(value)):
        return copy.deepcopy(value)
    rounded = round(float(value), 2)
    return 0.0 if rounded == 0.0 else rounded


def _rounded_continuous_value(value: object) -> object:
    if isinstance(value, dict):
        return {
            key: _rounded_continuous_value(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_rounded_continuous_value(item) for item in value]
    return _round_continuous_number(value)


def _is_static_solid_entity(entity: object) -> bool:
    if not isinstance(entity, dict):
        return False
    position = entity.get("position")
    size = entity.get("size")
    state = entity.get("state")
    kind = entity.get("kind")
    return (
        isinstance(kind, str)
        and any(
            token in kind.lower()
            for token in ("floor", "ground", "platform", "tile", "wall", "terrain")
        )
        and isinstance(position, (list, tuple))
        and isinstance(size, (list, tuple))
        and isinstance(state, dict)
        and state.get("solid") is True
        and set(state).issubset({"solid"})
    )


def continuous_exploration_view(observation: dict) -> dict:
    """Build the bounded learner view for continuous visible-solid games."""
    entities = observation.get("entities")
    entities = entities if isinstance(entities, list) else []
    static = [entity for entity in entities if _is_static_solid_entity(entity)]
    dynamic = [
        copy.deepcopy(entity)
        for entity in entities
        if not _is_static_solid_entity(entity)
    ]
    static.sort(key=lambda entity: (
        str(entity.get("kind", "")), str(entity.get("id", "")),
    ))
    static_rows = [
        [
            entity.get("kind"),
            _rounded_continuous_value(entity.get("position")),
            _rounded_continuous_value(entity.get("size")),
        ]
        for entity in static[:LEAN_CONTINUOUS_ENTITY_LIMIT]
    ]
    view = {
        key: copy.deepcopy(observation[key])
        for key in ("game", "tick", "status", "player", "events")
        if key in observation
    }
    if view.get("tick") == 0:
        view["events"] = ["Environment started."]
    view["static_solids"] = {
        "fields": ["kind", "position", "size"],
        "count": len(static),
        "included": len(static_rows),
        "items": static_rows,
    }
    view["dynamic_entities"] = dynamic
    view["coordinates"] = {
        "position": "[x,y,z]",
        "horizontal_axes": ["x", "z"],
    }
    actions = observation.get("actions", observation.get("controls", []))
    actions = list(actions) if isinstance(actions, (list, tuple)) else []
    capabilities = observation.get("_capabilities")
    can_reset = not isinstance(capabilities, dict) or capabilities.get("reset") is not False
    meta_controls = (["reset"] if can_reset else []) + ["finish"]
    view["controls"] = meta_controls if observation.get("status") == "won" else [
        *actions, *meta_controls,
    ]
    return view


def _compact_continuous_payload(value: object) -> object:
    """Remove repeated static geometry from prompt-local historical facts."""
    if isinstance(value, dict):
        compact = {}
        for key, item in value.items():
            if key == "entities" and isinstance(item, list):
                compact["dynamic_entities"] = [
                    copy.deepcopy(entity)
                    for entity in item
                    if not _is_static_solid_entity(entity)
                ]
            else:
                compact[key] = _compact_continuous_payload(item)
        return compact
    if isinstance(value, list):
        return [_compact_continuous_payload(item) for item in value]
    if isinstance(value, tuple):
        return [_compact_continuous_payload(item) for item in value]
    return _round_continuous_number(value)


def _continuous_floor_cells(observation: dict) -> list[tuple[tuple[float, float], tuple[float, float]]]:
    entities = observation.get("entities")
    if not isinstance(entities, list):
        return []
    result = []
    for entity in entities:
        if not _is_static_solid_entity(entity):
            continue
        kind = str(entity.get("kind", "")).lower()
        if not any(token in kind for token in ("floor", "ground", "platform", "tile")):
            continue
        position = entity.get("position")
        size = entity.get("size")
        if not (
            isinstance(position, (list, tuple))
            and isinstance(size, (list, tuple))
            and len(position) == len(size) == 3
            and all(
                isinstance(value, (int, float)) and not isinstance(value, bool)
                and isfinite(float(value))
                for value in (*position, *size)
            )
        ):
            continue
        result.append((
            (float(position[0]), float(position[2])),
            (float(size[0]) / 2.0, float(size[2]) / 2.0),
        ))
    return sorted(result)


def _continuous_tile_cell(
    observation: dict,
    floors: list[tuple[tuple[float, float], tuple[float, float]]],
) -> tuple[float, float] | None:
    player = observation.get("player")
    position = player.get("position") if isinstance(player, dict) else None
    if not (
        isinstance(position, (list, tuple))
        and len(position) == 3
        and all(
            isinstance(value, (int, float)) and not isinstance(value, bool)
            and isfinite(float(value))
            for value in position
        )
    ):
        return None
    point = float(position[0]), float(position[2])
    matches = [
        center for center, half in floors
        if abs(point[0] - center[0]) <= half[0] + 0.12
        and abs(point[1] - center[1]) <= half[1] + 0.12
    ]
    if matches:
        return min(matches, key=lambda center: (
            (point[0] - center[0]) ** 2 + (point[1] - center[1]) ** 2,
            center,
        ))
    return round(point[0], 1), round(point[1], 1)


def continuous_position_control_coverage(
    observation: dict,
    records: list[dict] | tuple[dict, ...],
) -> dict:
    """Summarize continuous coverage as visible floor cells and directions."""
    floors = _continuous_floor_cells(observation)
    directions = [
        action for action in DIRECTIONS
        if action in observation.get("actions", observation.get("controls", ACTIONS))
    ]
    attempted: dict[tuple[float, float], set[str]] = {}
    visited: set[tuple[float, float]] = set()
    current = _continuous_tile_cell(observation, floors)
    if current is not None:
        visited.add(current)
    for record in records:
        before = record.get("before")
        after = record.get("after")
        if not isinstance(before, dict) or not isinstance(after, dict):
            continue
        before_cell = _continuous_tile_cell(before, floors)
        after_cell = _continuous_tile_cell(after, floors)
        if before_cell is not None:
            visited.add(before_cell)
        if after_cell is not None:
            visited.add(after_cell)
        action = record.get("decision", {}).get("action")
        if before_cell is not None and action in directions:
            attempted.setdefault(before_cell, set()).add(action)

    def distance(cell: tuple[float, float]) -> float:
        return dist(cell, current) if current is not None else 0.0

    ordered = sorted(visited, key=lambda cell: (distance(cell), cell[1], cell[0]))
    rows = []
    gaps = []
    pair_count = 0
    for cell in ordered[:LEAN_CONTINUOUS_ENTITY_LIMIT]:
        tried = [action for action in directions if action in attempted.get(cell, set())]
        untried = [action for action in directions if action not in attempted.get(cell, set())]
        pair_count += len(tried)
        rows.append({
            "cell": [round(cell[0], 2), round(cell[1], 2)],
            "tried_directions": tried,
        })
        if untried and len(gaps) < LEAN_COVERAGE_NEAREST_LIMIT:
            gaps.append({
                "cell": [round(cell[0], 2), round(cell[1], 2)],
                "distance": round(distance(cell), 2),
                "untried_directions": untried,
            })
    return {
        "coordinate_space": "continuous_xz_floor_cells",
        "directions": directions,
        "visited_tile_cells": rows,
        "tile_direction_pair_counts": {
            "attempted": pair_count,
            "unattempted": len(visited) * len(directions) - pair_count,
        },
        "nearest_visited_tile_cells_with_untried_directions": gaps,
    }


def _bounded_continuous_messages(context: dict) -> list[dict]:
    """Keep one continuous request below the fixed prompt byte ceiling."""
    messages = [
        {"role": "system", "content": LEAN_PROMPT + LEAN_CONTINUOUS_PROMPT_SUFFIX},
        {"role": "user", "content": json.dumps(context, separators=(",", ":"))},
    ]

    def encoded_size() -> int:
        messages[1]["content"] = json.dumps(context, separators=(",", ":"))
        return len(json.dumps(messages, separators=(",", ":")).encode("utf-8"))

    omitted_facts = 0
    omitted_sweep_cells = 0
    omitted_coverage_cells = 0
    omitted_static_solids = 0
    while encoded_size() >= LEAN_CONTINUOUS_PROMPT_MAX_BYTES:
        factual_log = context.get("factual_log")
        if isinstance(factual_log, list) and len(factual_log) > 1:
            factual_log.pop(0)
            omitted_facts += 1
            continue
        sweep = context.get("sweep_coverage")
        tried = sweep.get("tried_directions") if isinstance(sweep, dict) else None
        if isinstance(tried, list) and len(tried) > 8:
            tried.pop()
            omitted_sweep_cells += 1
            continue
        coverage = context.get("exploration_coverage")
        visited = coverage.get("visited_tile_cells") if isinstance(coverage, dict) else None
        if isinstance(visited, list) and len(visited) > LEAN_COVERAGE_NEAREST_LIMIT:
            visited.pop()
            omitted_coverage_cells += 1
            continue
        anomalies = context.get("anomalies")
        if isinstance(anomalies, list) and len(anomalies) > 1:
            anomalies.pop(0)
            continue
        observation = context.get("observation")
        static = observation.get("static_solids") if isinstance(observation, dict) else None
        items = static.get("items") if isinstance(static, dict) else None
        if isinstance(items, list) and len(items) > 8:
            items.pop()
            static["included"] = len(items)
            omitted_static_solids += 1
            continue
        break
    if any((omitted_facts, omitted_sweep_cells, omitted_coverage_cells, omitted_static_solids)):
        context["continuous_prompt_compaction"] = {
            "omitted_factual_steps": omitted_facts,
            "omitted_sweep_cells": omitted_sweep_cells,
            "omitted_coverage_cells": omitted_coverage_cells,
            "omitted_static_solids": omitted_static_solids,
        }
        encoded_size()
    return messages


def position_control_coverage(
    observation: dict,
    records: list[dict] | tuple[dict, ...],
) -> dict:
    """Summarize tried controls by visited position using execution history only."""
    current = tuple(observation["player"]["position"])
    visited = {current}
    attempted: dict[tuple[int, int], set[str]] = {}
    for record in records:
        before_position = tuple(record["before"]["player"]["position"])
        after_position = tuple(record["after"]["player"]["position"])
        visited.update((before_position, after_position))
        action = record.get("decision", {}).get("action")
        if action in ACTIONS:
            attempted.setdefault(before_position, set()).add(action)

    by_position = []
    positions_with_gaps = []
    attempted_pair_count = 0
    for position in sorted(visited):
        tried = attempted.get(position, set())
        tried_in_control_order = [control for control in ACTIONS if control in tried]
        untried = [control for control in ACTIONS if control not in tried]
        attempted_pair_count += len(tried_in_control_order)
        by_position.append({
            "position": list(position),
            "attempted": tried_in_control_order,
        })
        if untried:
            positions_with_gaps.append({
                "position": list(position),
                "distance": abs(position[0] - current[0]) + abs(position[1] - current[1]),
                "unattempted": untried,
            })

    positions_with_gaps.sort(key=lambda item: (
        item["distance"], item["position"][0], item["position"][1],
    ))
    pair_total = len(visited) * len(ACTIONS)
    return {
        "controls": list(ACTIONS),
        "visited_positions": by_position,
        "position_control_pair_counts": {
            "attempted": attempted_pair_count,
            "unattempted": pair_total - attempted_pair_count,
        },
        "nearest_visited_positions_with_unattempted_controls": (
            positions_with_gaps[:LEAN_COVERAGE_NEAREST_LIMIT]
        ),
    }


def lean_messages(
    observation: dict,
    episode: int,
    notes: dict[str, LeanNote],
    records: list[dict],
    remaining: dict,
    *,
    focus: str = "mechanics",
    feedback: str = "",
    history_size: int = LEAN_CONTEXT_HISTORY_SIZE,
    sweep_summary: dict | None = None,
) -> list[dict]:
    continuous = _continuous_visible(observation)
    facts = compact_software_facts(
        records,
        min(history_size, LEAN_CONTINUOUS_HISTORY_SIZE) if continuous else history_size,
    )
    if continuous:
        facts = _compact_continuous_payload(facts)
    context = {
        "session_question": EXPLORATION_FOCUSES[focus],
        "episode": episode,
        "observation": (
            continuous_exploration_view(observation)
            if continuous else exploration_view(observation)
        ),
        "notes": [note_dict(note) for note in notes.values()],
        "note_slots_remaining": max(0, LEAN_NOTE_LIMIT - len(notes)),
        "context_compaction": facts["context_compaction"],
        "factual_log": facts["factual_log"],
        "earlier_factual_summary": facts["earlier_factual_summary"],
        "cross_episode_pair_fields": facts["cross_episode_pair_fields"],
        "cross_episode_same_start_pairs": facts["cross_episode_same_start_pairs"],
        "retention_facts": facts["retention_facts"],
        "remaining_budget": remaining,
        "response_contract": {
            "fields": ["action", "reason", "expected_position", "notes"],
            "action": "Choose exactly one currently allowed control.",
            "reason": "Brief text.",
            "expected_position": (
                "Use null for this continuous 3D world."
                if continuous else "Immediate [x,y] prediction or null."
            ),
            "notes": (
                "Zero to six tentative updates: reuse an existing id, or use a new id only when "
                "note_slots_remaining permits; include claim, uncertain|supported|contradicted, "
                "optional mechanic|suspected_bug kind (default mechanic), and completed evidence_steps."
            ),
        },
    }
    if focus == "qa":
        context["qa_checklist"] = (
            LEAN_CONTINUOUS_QA_CHECKLIST if continuous else LEAN_QA_CHECKLIST
        )
        context["exploration_coverage"] = (
            continuous_position_control_coverage(observation, records)
            if continuous else position_control_coverage(observation, records)
        )
        context["anomalies"] = facts.get("anomalies", [])
        context["anomaly_summary"] = facts.get("anomaly_summary", {
            "observed": 0, "unique": 0, "included": 0, "omitted": 0,
        })
        if sweep_summary is not None:
            from .hybrid import compact_sweep_summary

            context["sweep_coverage"] = compact_sweep_summary(sweep_summary)
    if feedback:
        context["validation_feedback"] = feedback
    if continuous:
        return _bounded_continuous_messages(context)
    return [
        {"role": "system", "content": LEAN_PROMPT},
        {"role": "user", "content": json.dumps(context, separators=(",", ":"))},
    ]


def _merge_notes(
    retained: dict[str, LeanNote],
    updates: tuple[LeanNote, ...],
) -> tuple[dict[str, LeanNote], list[dict]]:
    merged = dict(retained)
    warnings = []
    for index, note in enumerate(updates):
        if note.id not in merged and len(merged) >= LEAN_NOTE_LIMIT:
            replace_id = next(
                (identifier for identifier, retained_note in merged.items() if retained_note.status == "uncertain"),
                None,
            )
            if replace_id is None:
                warnings.append(_warning(
                    "note_memory_full",
                    f"Dropped note[{index}]: session memory already contains six note IDs and none is uncertain.",
                ))
                continue
            del merged[replace_id]
            warnings.append(_warning(
                "note_memory_replaced",
                f"Replaced oldest uncertain note {replace_id!r} with new note id {note.id!r} from note[{index}].",
            ))
        merged[note.id] = note
    return merged, warnings


def _truncated_content_diagnostic(content: str, secrets: tuple[str, ...]) -> dict:
    """Keep bounded non-reasoning response edges after whole-string redaction."""
    sanitized = redact_secrets(content, secrets)
    return {
        "head": sanitized[:LEAN_TRUNCATION_DIAGNOSTIC_CHARS],
        "tail": sanitized[-LEAN_TRUNCATION_DIAGNOSTIC_CHARS:],
    }


def _deadline_failure_fields(error):
    """Finite transport telemetry only; provider exception text stays private."""
    if getattr(error, "transport", None) != "isolated-deadline-v1":
        return {}
    fields = {"transport": "isolated-deadline-v1"}
    for name in ("request_timeout_seconds", "transport_elapsed_ms", "cleanup_elapsed_ms"):
        value = getattr(error, name, None)
        if type(value) in (int, float) and isfinite(value) and value >= 0:
            fields[name] = value
    status = getattr(error, "http_status", None)
    if type(status) is int and 100 <= status <= 599:
        fields["http_status"] = status
    reaped = getattr(error, "cleanup_worker_reaped", None)
    if type(reaped) is bool:
        fields["cleanup_worker_reaped"] = reaped
    return fields


def run_lean_exploration(
    game: GameAdapter,
    client: NebiusClient,
    log: SessionLog,
    reset_factory: Callable[[], GameAdapter],
    limits: ExplorationLimits = ExplorationLimits(),
    on_action: Callable[[dict], None] | None = None,
    evaluator: Callable[[dict, str, dict], dict] = evaluate_transition,
    *,
    focus: str = "mechanics",
    direct_fallback_max_tokens: int = DEFAULT_DIRECT_FALLBACK_MAX_TOKENS,
    sleep: Callable[[float], None] = time.sleep,
    sweep_summary: dict | None = None,
    adaptive=None,
) -> dict:
    """Run a bounded lean session with one direct retry for transient failures."""
    from .deadline_transport import RequestAdmissionDeclined

    if focus not in EXPLORATION_FOCUSES:
        raise ValueError("Unknown exploration focus.")
    if type(direct_fallback_max_tokens) is not int or direct_fallback_max_tokens <= 0:
        raise ValueError("Direct fallback max tokens must be a positive integer.")
    started = time.monotonic()
    deadline = started + limits.max_seconds
    deadline_hook = getattr(client, "set_session_deadline", None)
    if callable(deadline_hook):
        deadline_hook(deadline)
    initial_observation = game.observe()
    coverage = Coverage(initial_observation) if adaptive is None else None
    steps = requests = rejected = invalid_streak = resets = episode = responses = 0
    usage_complete = True
    notes: dict[str, LeanNote] = {}
    records: list[dict] = []
    thinking_latencies: list[float] = []
    last_two_thinking_latencies: list[float] = []
    prompt_byte_sizes: list[int] = []
    direct_fallbacks = 0
    retries = 0
    latency_direct = False
    warnings: list[dict] = []
    totals = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    status = "action_limit"
    error = None
    pending_control = None
    admission_stopped = False
    try:
        while True:
            if steps >= limits.max_actions:
                status = "action_limit"
                break
            if requests >= limits.max_requests:
                status = "request_limit"
                break
            remaining_seconds = deadline - time.monotonic()
            if remaining_seconds <= 0:
                status = "time_limit"
                break

            before = game.observe()
            allowed = exploration_view(before)["controls"] if adaptive is None else adaptive.allowed(before)
            schema = lean_schema(allowed) if adaptive is None else adaptive.schema(allowed)
            typical_latency = median(thinking_latencies) if thinking_latencies else None
            remaining = {
                "actions": limits.max_actions - steps,
                "requests": limits.max_requests - requests,
                "seconds": round(remaining_seconds, 1),
                "recent_request_median_seconds": round(typical_latency, 3) if typical_latency is not None else None,
            }
            feedback = "The previous response had no usable legal action. Return one currently allowed action." if invalid_streak else ""
            if adaptive is None:
                messages = lean_messages(
                    before, episode, notes, records, remaining,
                    focus=focus, feedback=feedback,
                    history_size=LEAN_CONTEXT_HISTORY_SIZE,
                    sweep_summary=sweep_summary,
                )
            else:
                messages = adaptive.messages(before, episode, notes, records, remaining,
                                             focus=focus, feedback=feedback)
            remaining_seconds = deadline - time.monotonic()
            admission = getattr(adaptive, "request_admission", None)
            admitted = admission is None or admission(remaining_seconds, client.timeout)
            if remaining_seconds <= 0 or not admitted:
                if adaptive is not None:
                    log.write("unused_context", episode=episode,
                              observation_tick=before["tick"], messages=messages,
                              response_schema=schema, dispatched=False)
                if remaining_seconds > 0 and not admitted:
                    admission_stopped = True
                    log.write("request_admission_stop", remaining_seconds=remaining_seconds,
                              request_timeout=client.timeout, required_seconds=client.timeout + 1,
                              dispatched=False)
                status = "time_limit"
                break

            prompt_bytes = len(json.dumps(messages, separators=(",", ":")).encode("utf-8"))
            schema_bytes = len(json.dumps(schema, separators=(",", ":")).encode("utf-8"))
            retry_of = None
            retry_reason = None
            provider_terminal = False
            while True:
                remaining_seconds = deadline - time.monotonic()
                if admission is not None and not admission(remaining_seconds, client.timeout):
                    log.write("unused_context", episode=episode,
                              observation_tick=before["tick"], messages=messages,
                              response_schema=schema, dispatched=False)
                    if remaining_seconds > 0:
                        admission_stopped = True
                        log.write("request_admission_stop", remaining_seconds=remaining_seconds,
                                  request_timeout=client.timeout, required_seconds=client.timeout + 1,
                                  dispatched=False)
                    status = "time_limit"
                    provider_terminal = True
                    break
                if remaining_seconds <= 0:
                    status = "time_limit"
                    provider_terminal = True
                    break
                if requests >= limits.max_requests:
                    status = "request_limit"
                    provider_terminal = True
                    break

                direct_reason = None
                if retry_of is not None:
                    decision_mode = "direct_retry"
                    direct_reason = retry_reason
                elif latency_direct:
                    decision_mode = "direct_latency"
                    direct_reason = "latency"
                else:
                    if invalid_streak:
                        direct_reason = "repair"
                    elif (
                        typical_latency is not None
                        and remaining_seconds < LEAN_TIME_PRESSURE_MULTIPLIER * typical_latency
                    ):
                        direct_reason = "time_pressure"
                    elif not client.settings.enable_thinking:
                        direct_reason = "configured_direct"
                    decision_mode = "direct_fallback" if direct_reason is not None else "thinking"
                request_enable_thinking = decision_mode == "thinking"
                request_max_tokens = client.max_tokens if request_enable_thinking else direct_fallback_max_tokens
                requests += 1
                request_number = requests
                if retry_of is not None:
                    retries += 1
                if not request_enable_thinking:
                    direct_fallbacks += 1
                prompt_byte_sizes.append(prompt_bytes)
                request_fields = {
                    "request_number": request_number,
                    "episode": episode,
                    "observation_tick": before["tick"],
                    "messages": messages,
                    "response_schema": schema,
                    "temperature": client.temperature,
                    "top_p": client.top_p,
                    "decision_mode": decision_mode,
                    "decision_mode_reason": direct_reason,
                    "enable_thinking": request_enable_thinking,
                    "max_tokens": request_max_tokens,
                    "prompt_bytes": prompt_bytes,
                    "response_schema_bytes": schema_bytes,
                }
                if callable(deadline_hook):
                    request_fields.update(
                        configured_timeout_seconds=client.timeout,
                        actual_timeout_seconds=min(client.timeout, remaining_seconds),
                        remaining_seconds_before_client_call=remaining_seconds,
                    )
                dispatch_fields = {
                    "elapsed_ms": round((time.monotonic() - log.started) * 1000),
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
                try:
                    completion = client.complete(
                        messages,
                        timeout=min(client.timeout, remaining_seconds),
                        response_schema=schema,
                        enable_thinking=request_enable_thinking,
                        max_tokens=request_max_tokens,
                    )
                except RequestAdmissionDeclined as exc:
                    # A durable reservation can consume the last admission
                    # margin. This exception proves the transport never started;
                    # it is an unused context, not an inference attempt.
                    requests -= 1
                    prompt_byte_sizes.pop()
                    if not request_enable_thinking:
                        direct_fallbacks -= 1
                    if retry_of is not None:
                        retries -= 1
                    log.write("unused_context", episode=episode,
                              observation_tick=before["tick"], messages=messages,
                              response_schema=schema, dispatched=False)
                    if exc.remaining_seconds > 0:
                        admission_stopped = True
                        log.write("request_admission_stop", remaining_seconds=exc.remaining_seconds,
                                  request_timeout=exc.request_timeout,
                                  required_seconds=exc.required_seconds, dispatched=False)
                    status = "time_limit"
                    provider_terminal = True
                    break
                except TransientNebiusError as exc:
                    log.write(
                        "request", **request_fields, **dispatch_fields,
                        retry_of=retry_of,
                        reason=exc.reason,
                        failure_reason=exc.reason,
                        attempt_status="failed",
                        possibly_billed=exc.possibly_billed,
                        **_deadline_failure_fields(exc),
                    )
                    transient_error = str(exc)
                    if retry_of is not None or adaptive is not None:
                        status = "provider_unavailable"
                        error = transient_error
                        provider_terminal = True
                        break
                    if requests >= limits.max_requests:
                        status = "request_limit"
                        error = transient_error
                        provider_terminal = True
                        break
                    remaining_seconds = deadline - time.monotonic()
                    if remaining_seconds <= 0:
                        status = "time_limit"
                        error = transient_error
                        provider_terminal = True
                        break
                    delay = (
                        exc.retry_after_seconds
                        if exc.retry_after_seconds is not None
                        else DEFAULT_TRANSIENT_RETRY_BACKOFF_SECONDS
                    )
                    if type(delay) not in {int, float} or not isfinite(delay) or delay < 0:
                        delay = DEFAULT_TRANSIENT_RETRY_BACKOFF_SECONDS
                    delay = min(float(delay), remaining_seconds)
                    if delay > 0:
                        sleep(delay)
                    if time.monotonic() >= deadline:
                        status = "time_limit"
                        error = transient_error
                        provider_terminal = True
                        break
                    retry_of = request_number
                    retry_reason = exc.reason
                    continue
                except NebiusError as exc:
                    log.write(
                        "request", **request_fields, **dispatch_fields,
                        retry_of=retry_of,
                        reason="non_transient",
                        failure_reason="non_transient",
                        attempt_status="failed",
                        possibly_billed=getattr(exc, "possibly_billed", False) is True,
                        **_deadline_failure_fields(exc),
                    )
                    raise
                except KeyboardInterrupt:
                    if adaptive is not None:
                        log.write("request", **request_fields, **dispatch_fields,
                                  retry_of=retry_of, reason="interrupted",
                                  failure_reason="interrupted", attempt_status="interrupted",
                                  possibly_billed=True)
                    raise
                else:
                    log.write(
                        "request", **request_fields, **dispatch_fields,
                        retry_of=retry_of,
                        reason=retry_reason or "initial",
                        failure_reason=None,
                        attempt_status="completed",
                        possibly_billed=False,
                    )
                    break

            if provider_terminal:
                break

            responses += 1
            latency_ms = completion.metadata.get("latency_ms")
            if (
                decision_mode == "thinking"
                and type(latency_ms) in {int, float}
                and isfinite(latency_ms)
                and latency_ms > 0
            ):
                thinking_latencies = [
                    *thinking_latencies, latency_ms / 1000,
                ][-limits.history_size:]
                last_two_thinking_latencies = [
                    *last_two_thinking_latencies, latency_ms / 1000,
                ][-2:]
                if (
                    len(last_two_thinking_latencies) == 2
                    and all(
                        item > LEAN_SLOW_THINKING_TIMEOUT_FRACTION * client.timeout
                        for item in last_two_thinking_latencies
                    )
                ):
                    latency_direct = True
            usage = completion.metadata.get("usage", {})
            usage_complete = usage_complete and all(type(usage.get(name)) is int for name in totals)
            for name in totals:
                totals[name] += usage.get(name, 0)

            decision = None
            decision_warnings: list[dict] = []
            invalid_error = None
            finish_reason = completion.metadata.get("finish_reason")
            try:
                if finish_reason != "stop":
                    raise InvalidDecision("Completion was truncated or interrupted; no action or notes were applied.")
                parser = getattr(adaptive, "parse_decision", LeanDecision.parse)
                decision, decision_warnings = parser(completion.content, allowed, set(range(1, steps + 1)))
                candidate_notes, memory_warnings = _merge_notes(notes, decision.notes)
                decision_warnings = _unique_warnings([*decision_warnings, *memory_warnings])
            except InvalidDecision as exc:
                invalid_error = str(exc)

            safe_answer = (
                json.dumps(redact_secrets(decision_dict(decision), log.secrets), separators=(",", ":"))
                if invalid_error is None else None
            )
            completion_fields = {}
            has_non_reasoning_content = (
                not request_enable_thinking
                or completion.metadata.get("discarded_inline_thinking") is True
                or completion.content.lstrip().startswith("{")
            )
            if finish_reason == "length" and has_non_reasoning_content:
                completion_fields["content_diagnostic"] = _truncated_content_diagnostic(
                    completion.content, log.secrets,
                )
            log.write(
                "completion",
                request_number=requests,
                decision_mode=decision_mode,
                answer=safe_answer,
                answer_characters=len(completion.content),
                **completion.metadata,
                **completion_fields,
            )
            if invalid_error is not None:
                rejected += 1
                invalid_streak += 1
                log.write(
                    "rejected_decision",
                    request_number=requests,
                    error=invalid_error,
                    **completion_fields,
                )
                if direct_reason == "repair" and finish_reason == "length":
                    status, error = "model_output_limit", invalid_error
                    break
                if invalid_streak >= 2:
                    status, error = "invalid_decision", invalid_error
                    break
                if time.monotonic() >= deadline:
                    status = "time_limit"
                    break
                continue
            if time.monotonic() >= deadline:
                status = "time_limit"
                break

            invalid_streak = 0
            request_warnings = []
            for item in decision_warnings:
                recorded = {"request_number": requests, **item}
                warnings.append(recorded)
                request_warnings.append(item)
                log.write("decision_warning", **recorded)
            warning_observer = getattr(adaptive, "observe_decision_warnings", None)
            if warning_observer is not None:
                warning_observer(requests, request_warnings)
            notes = candidate_notes
            current_facts = software_facts(records)
            log.write(
                "memory",
                request_number=requests,
                episode=episode,
                decision_mode=decision_mode,
                hypotheses=[note_dict(note) for note in notes.values()],
                notes_warnings=request_warnings,
                software_facts={
                    "cross_episode_same_start_pairs": current_facts["cross_episode_same_start_pairs"],
                    "retention_facts": current_facts["retention_facts"],
                },
            )

            if decision.action == "finish":
                finish_fields = {}
                if sweep_summary is not None:
                    finish_fields["source"] = "llm"
                finish_record = {
                    "request_number": requests, "episode": episode,
                    "decision_mode": decision_mode,
                    "decision": decision_dict(decision), "observation": before,
                    **finish_fields,
                }
                finish_hook = getattr(adaptive, "finish", None)
                if finish_hook is not None:
                    fact_key = "focused_fact" if getattr(adaptive, "prompt_version", None) == "focused-v1" else "investigation_fact"
                    finish_record[fact_key] = finish_hook(finish_record)
                log.write(
                    "finish", **finish_record,
                )
                status = "finished"
                break
            if time.monotonic() >= deadline:
                status = "time_limit"
                break

            if decision.action == "reset":
                try:
                    reset_game = reset_factory()
                    after = reset_game.observe()
                except BaseException as exc:
                    if adaptive is not None:
                        pending_control = {"action": "reset", "step": steps + 1,
                                           "execution_unknown": True, "error_type": type(exc).__name__}
                        log.write("unacknowledged_control", request_number=requests,
                                  before=before, **pending_control)
                    raise
                old_episode = episode
                episode += 1
                resets += 1
                game = reset_game
                record = {
                    "type": "reset",
                    "episode_before": old_episode,
                    "episode_after": episode,
                    "before": before,
                    "after": after,
                }
            else:
                try:
                    after = game.act(decision.action)
                except BaseException as exc:
                    if adaptive is not None:
                        pending_control = {"action": decision.action, "step": steps + 1,
                                           "execution_unknown": True, "error_type": type(exc).__name__}
                        log.write("unacknowledged_control", request_number=requests,
                                  before=before, **pending_control)
                    raise
                game_evaluator = getattr(game, "evaluate_transition", None)
                active_evaluator = game_evaluator if evaluator is evaluate_transition and callable(game_evaluator) else evaluator
                evaluation_error = None
                if adaptive is not None and active_evaluator is evaluate_transition and not callable(game_evaluator):
                    evaluation = {"moved": before.get("player", {}).get("position") != after.get("player", {}).get("position"),
                                  "events": copy.deepcopy(after.get("events", [])), "contract_violations": []}
                else:
                    try:
                        evaluation = active_evaluator(before, decision.action, after)
                    except Exception as exc:
                        if adaptive is None:
                            raise
                        evaluation = {"events": copy.deepcopy(after.get("events", [])), "contract_violations": []}
                        evaluation_error = type(exc).__name__
                record = {
                    "type": "step",
                    "episode": episode,
                    "episode_before": episode,
                    "episode_after": episode,
                    "before": before,
                    "after": after,
                    "evaluation": evaluation,
                }
                if evaluation_error:
                    record["evaluation_error"] = evaluation_error
            steps += 1
            record.update(
                step=steps,
                request_number=requests,
                decision_mode=decision_mode,
                decision=decision_dict(decision),
            )
            if focus == "qa":
                record["qa_mode"] = True
            if sweep_summary is not None:
                record["source"] = "llm"
            record["software_fact"] = step_fact(record, records)
            if adaptive is not None:
                try:
                    record["experiment_fact"] = adaptive.record(record)
                except Exception as exc:
                    # The acknowledged control must survive a bookkeeping
                    # failure. Preserve its exact raw transition before stop.
                    record["experiment_fact_error"] = type(exc).__name__
                    log.write(record["type"], **{key: value for key, value in record.items() if key != "type"})
                    records.append(record)
                    status, error = "experiment_recording_error", "Experiment bookkeeping failed after the recorded control."
                    break
            log.write(record["type"], **{key: value for key, value in record.items() if key != "type"})
            records.append(record)
            if coverage is not None:
                coverage.record(
                    before,
                    decision.action,
                    after,
                    step=steps,
                    episode=episode - 1 if decision.action == "reset" else episode,
                )
            if on_action:
                on_action(record)
            if record.get("evaluation_error"):
                status, error = "evaluation_error", "The adapter checker failed after the recorded action."
                break
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
        if adaptive is not None:
            from .adaptive import AdaptivePromptBudgetExceeded
            if isinstance(exc, AdaptivePromptBudgetExceeded):
                status = "context_budget_exceeded"
                log.write("context_budget_failure", episode=episode,
                          required_bytes=exc.required_bytes, max_bytes=exc.maximum_bytes,
                          remaining_budget=remaining, feedback=feedback)
    except Exception as exc:
        if adaptive is None:
            raise
        status, error = "environment_error", type(exc).__name__

    facts = software_facts(records, history_size=None)
    if focus == "qa":
        facts.setdefault("anomalies", [])
    adaptive_metrics = adaptive.summary() if adaptive is not None else None
    action_types = set(coverage.action_types) if coverage is not None else set()
    final_observation_status = "observed"
    if adaptive is not None:
        try:
            final_observation = game.observe()
        except Exception:
            final_observation = copy.deepcopy(records[-1]["after"] if records else initial_observation)
            final_observation_status = "unavailable_last_acknowledged"
    else:
        final_observation = game.observe()
    summary = {
        "status": status,
        "partial": status != "finished",
        "steps": steps,
        "focus": focus,
        "game_actions": steps - resets,
        "resets": resets,
        "episodes": episode + 1,
        "inference_requests": requests,
        "completed_inference_responses": responses,
        "direct_fallbacks": direct_fallbacks,
        "retries": retries,
        "usage_complete": usage_complete and responses == requests,
        "rejected_decisions": rejected,
        "usage": totals,
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "final_observation": final_observation,
        "active_experiment": None,
        "experiments": [],
        "coverage": coverage.summary() if coverage is not None else adaptive_metrics,
        "hypotheses": [note_dict(note) for note in notes.values()],
        "parse_warnings": warnings,
        "warning_count": len(warnings),
        "request_prompt_bytes": {
            "first": prompt_byte_sizes[0] if prompt_byte_sizes else None,
            "twentieth": prompt_byte_sizes[19] if len(prompt_byte_sizes) >= 20 else None,
            "last": prompt_byte_sizes[-1] if prompt_byte_sizes else None,
            "maximum": max(prompt_byte_sizes) if prompt_byte_sizes else None,
        },
        "response_schema_bytes": len(json.dumps(lean_schema(), separators=(",", ":")).encode("utf-8")),
        **facts,
        "unassessed_final_step": None,
        "unassessed_final_outcome": None,
        "pilot_observations": {
            "movement_actions": bool(action_types.intersection(DIRECTIONS)),
            "interaction_actions": "interact" in action_types,
            "blocked_or_unchanged_interaction": coverage.blocked_movements + coverage.unchanged_interactions > 0 if coverage is not None else False,
            "evidence_linked_notes": any(note.evidence_steps for note in notes.values()),
            "cross_episode_same_start_pair": bool(facts["cross_episode_same_start_pairs"]),
        },
        "ready_for_live_review": False,
        "review_note": "Software facts verify recorded execution. Tentative note truth, intent, universality, and memory causation require review.",
        "learning_claims": "Tentative model notes linked to recorded steps; not independently verified mechanics or bugs.",
    }
    if adaptive is not None:
        summary["prompt_version"] = adaptive_metrics["prompt_version"]
        if admission_stopped:
            summary["request_admission_stopped"] = True
        summary["adaptive"] = adaptive_metrics
        summary["final_observation_status"] = final_observation_status
        if pending_control is not None:
            summary["unassessed_final_step"] = pending_control["step"]
            summary["unassessed_final_outcome"] = pending_control
        summary["response_schema_bytes"] = adaptive_metrics["context_compaction"]["maximum_schema_bytes"] or None
        summary["pilot_observations"] = {
            "evidence_linked_notes": any(note.evidence_steps for note in notes.values()),
            "scope": "Native controls and observed transitions; no inferred movement families or bug truth.",
        }
    if focus == "qa":
        summary["qa_mode"] = True
    if sweep_summary is not None:
        from .hybrid import compact_sweep_summary

        compact_sweep = compact_sweep_summary(sweep_summary)
        sweep_actions = compact_sweep.get("actions", 0)
        sweep_actions = sweep_actions if type(sweep_actions) is int else 0
        summary["source"] = "llm"
        summary["sweep_coverage"] = compact_sweep
        summary["phase_control_counts"] = {
            "sweep": sweep_actions,
            "llm": steps,
        }
        summary["hybrid_total_controls"] = sweep_actions + steps
    if error:
        summary["error"] = redact_secrets(error, log.secrets) if adaptive is not None else error
    log.write("summary", **summary)
    return summary
