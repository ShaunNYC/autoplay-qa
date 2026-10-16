"""Versioned, game-agnostic JSON-lines protocol helpers.

The wire representation deliberately contains only public game state.  The
reference game may add a namespaced compatibility extension so its existing
Python adapter can round-trip observations exactly; injected-defect metadata is
never part of that extension or any other protocol response.
"""

from __future__ import annotations

from copy import deepcopy
import json
from math import isfinite
from typing import BinaryIO


PROTOCOL_VERSION = "autoplay-qa/1"
MAX_LINE_BYTES = 1024 * 1024
LEGACY_OBSERVATION_EXTENSION = "org.openai.autoplay-qa.legacy-observation"


class ProtocolError(ValueError):
    """A malformed, unsupported, or failed protocol exchange."""

    def __init__(self, code: str, message: str, *, retryable: bool = False):
        super().__init__(message)
        self.code = code
        self.retryable = retryable


def error_response(
    request_id: str | None,
    code: str,
    message: str,
    *,
    retryable: bool = False,
) -> dict:
    return {
        "id": request_id,
        "type": "error",
        "ok": False,
        "error": {
            "code": code,
            "message": message,
            "retryable": retryable,
        },
    }


def encode_message(message: dict) -> bytes:
    if not isinstance(message, dict):
        raise ProtocolError("invalid_message", "Protocol messages must be JSON objects.")
    try:
        payload = json.dumps(
            message,
            ensure_ascii=False,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ProtocolError("invalid_message", f"Message is not valid JSON: {exc}.") from None
    if len(payload) > MAX_LINE_BYTES:
        raise ProtocolError(
            "message_too_large",
            f"Message exceeds the {MAX_LINE_BYTES}-byte limit.",
        )
    return payload + b"\n"


def read_message(stream: BinaryIO) -> dict | None:
    """Read one bounded UTF-8 JSON object; return ``None`` only at clean EOF."""
    line = stream.readline(MAX_LINE_BYTES + 2)
    if line == b"":
        return None
    if len(line) > MAX_LINE_BYTES + 1 or not line.endswith(b"\n"):
        raise ProtocolError(
            "message_too_large",
            f"Message exceeds the {MAX_LINE_BYTES}-byte limit or lacks a newline.",
        )
    try:
        value = json.loads(line.decode("utf-8"))
    except UnicodeDecodeError:
        raise ProtocolError("invalid_utf8", "Message is not valid UTF-8.") from None
    except json.JSONDecodeError as exc:
        raise ProtocolError("invalid_json", f"Message is not valid JSON: {exc.msg}.") from None
    if not isinstance(value, dict):
        raise ProtocolError("invalid_message", "Protocol messages must be JSON objects.")
    return value


def write_message(stream: BinaryIO, message: dict) -> None:
    stream.write(encode_message(message))
    stream.flush()


def validate_request_envelope(message: dict) -> tuple[str, str]:
    request_id = message.get("id")
    request_type = message.get("type")
    if not isinstance(request_id, str) or not request_id:
        raise ProtocolError("invalid_request", "Request id must be a nonempty string.")
    if not isinstance(request_type, str) or not request_type:
        raise ProtocolError("invalid_request", "Request type must be a nonempty string.")
    return request_id, request_type


def _entity(
    identifier: str,
    kind: str,
    position: object,
    state: dict | None = None,
) -> dict:
    result = {"id": identifier, "kind": kind, "state": deepcopy(state or {})}
    if position is not None:
        result["position"] = deepcopy(position)
    return result


def keygate_protocol_observation(observation: dict) -> dict:
    """Project a KeyGate observation into the generic public wire shape."""
    player = observation.get("player")
    player = player if isinstance(player, dict) else {}
    objects = observation.get("objects")
    objects = objects if isinstance(objects, dict) else {}
    gate = objects.get("gate")
    gate = gate if isinstance(gate, dict) else {}
    entities = [
        _entity("key", "key", objects.get("key"), {"available": objects.get("key") is not None}),
        _entity("gate", "gate", gate.get("position"), {"open": gate.get("open") is True}),
        _entity("exit", "exit", objects.get("exit"), {}),
    ]
    public_legacy = {
        key: deepcopy(value)
        for key, value in observation.items()
        if key != "rules"
    }
    result = {
        "tick": observation.get("tick"),
        "status": observation.get("status"),
        "player": {
            "position": deepcopy(player.get("position")),
            "inventory": deepcopy(player.get("inventory", [])),
        },
        "entities": entities,
        "events": deepcopy(observation.get("events", [])),
        "controls": deepcopy(observation.get("actions", [])),
        "grid": deepcopy(observation.get("map", [])),
        "legend": deepcopy(observation.get("legend", {})),
        "extensions": {
            # This public-state-only extension preserves legacy recordings and
            # prompts while the generic fields remain the portable contract.
            LEGACY_OBSERVATION_EXTENSION: public_legacy,
        },
    }
    return result


def adapter_observation(wire: dict, *, game_version: str = "remote-v1") -> dict:
    """Convert a generic wire observation to the existing adapter shape.

    The namespaced compatibility extension is preferred for the reference
    server.  Other games receive safe aliases plus their generic ``entities``
    list, allowing game-agnostic detectors to operate without a grid.
    """
    if not isinstance(wire, dict):
        raise ProtocolError("invalid_observation", "Observation must be a JSON object.")
    extensions = wire.get("extensions")
    legacy = (
        extensions.get(LEGACY_OBSERVATION_EXTENSION)
        if isinstance(extensions, dict)
        else None
    )
    if isinstance(legacy, dict):
        result = deepcopy(legacy)
        if game_version == "key-gate-v1" and "rules" not in result:
            # Rules remain local to the Python host and never cross the wire.
            # Exploration still whitelists them out of model prompts.
            from .game import KEY_GATE_RULES

            restored = {}
            for key, value in result.items():
                if key == "actions":
                    restored["rules"] = list(KEY_GATE_RULES)
                restored[key] = value
            if "rules" not in restored:
                restored["rules"] = list(KEY_GATE_RULES)
            result = restored
        return result

    player = wire.get("player")
    player = deepcopy(player) if isinstance(player, dict) else {}
    player.setdefault("inventory", [])
    entities = wire.get("entities")
    entities = deepcopy(entities) if isinstance(entities, list) else []
    objects: dict[str, object] = {}
    for item in entities:
        if not isinstance(item, dict):
            continue
        identifier = item.get("id")
        if not isinstance(identifier, str) or not identifier:
            continue
        state = item.get("state")
        state = deepcopy(state) if isinstance(state, dict) else {}
        position = deepcopy(item.get("position"))
        objects[identifier] = {"position": position, **state} if state else position

    observation = {
        "game": game_version,
        "tick": wire.get("tick", 0),
        "status": wire.get("status", "unknown"),
        "player": player,
        "entities": entities,
        "objects": objects,
        "events": deepcopy(wire.get("events", [])),
        "actions": deepcopy(wire.get("controls", [])),
    }
    if isinstance(wire.get("grid"), list):
        observation["map"] = deepcopy(wire["grid"])
    if isinstance(wire.get("legend"), dict):
        observation["legend"] = deepcopy(wire["legend"])
    known_fields = {
        "tick", "status", "player", "entities", "events", "controls",
        "grid", "legend", "extensions",
    }
    for field, value in wire.items():
        if field not in known_fields:
            observation[field] = deepcopy(value)
    if isinstance(extensions, dict):
        observation["extensions"] = deepcopy(extensions)
    return observation


def validate_wire_observation(observation: object, capabilities: dict) -> dict:
    if not isinstance(observation, dict):
        raise ProtocolError("invalid_observation", "Response observation must be an object.")
    status = observation.get("status")
    if not isinstance(status, str) or not status:
        raise ProtocolError("invalid_observation", "Observation status must be a nonempty string.")
    player = observation.get("player")
    if not isinstance(player, dict):
        raise ProtocolError("invalid_observation", "Observation player must be an object.")
    inventory = player.get("inventory")
    if not isinstance(inventory, list) or not all(isinstance(item, str) for item in inventory):
        raise ProtocolError("invalid_observation", "Player inventory must be an array of strings.")

    def validate_position(value: object, field: str) -> None:
        if value is None:
            return
        if not isinstance(value, list) or not value or not all(
            type(coordinate) in {int, float} and isfinite(coordinate)
            for coordinate in value
        ):
            raise ProtocolError(
                "invalid_observation",
                f"{field} must be null or an array of finite coordinates.",
            )

    validate_position(player.get("position"), "Player position")
    entities = observation.get("entities")
    if not isinstance(entities, list):
        raise ProtocolError("invalid_observation", "Observation entities must be an array.")
    entity_ids: set[str] = set()
    for entity in entities:
        if not isinstance(entity, dict):
            raise ProtocolError("invalid_observation", "Each entity must be an object.")
        identifier = entity.get("id")
        kind = entity.get("kind")
        if not isinstance(identifier, str) or not identifier or identifier in entity_ids:
            raise ProtocolError(
                "invalid_observation",
                "Entity ids must be nonempty and unique.",
            )
        entity_ids.add(identifier)
        if not isinstance(kind, str) or not kind:
            raise ProtocolError("invalid_observation", "Entity kind must be a nonempty string.")
        if not isinstance(entity.get("state"), dict):
            raise ProtocolError("invalid_observation", "Entity state must be an object.")
        validate_position(entity.get("position"), f"Entity {identifier!r} position")
    if not isinstance(observation.get("events"), list) or not all(
        isinstance(item, str) for item in observation["events"]
    ):
        raise ProtocolError("invalid_observation", "Observation events must be an array of strings.")
    controls = observation.get("controls")
    if (
        not isinstance(controls, list)
        or not all(isinstance(item, str) and item for item in controls)
        or len(set(controls)) != len(controls)
    ):
        raise ProtocolError("invalid_observation", "Observation controls must be an array of strings.")
    grid = observation.get("grid")
    if capabilities.get("grid") is True and not isinstance(grid, list):
        raise ProtocolError("invalid_observation", "Grid capability requires observation.grid.")
    if grid is not None:
        if not isinstance(grid, list) or not all(isinstance(row, str) for row in grid):
            raise ProtocolError("invalid_observation", "Observation grid must be an array of strings.")
        widths = {len(row) for row in grid}
        if len(widths) > 1:
            raise ProtocolError("invalid_observation", "Observation grid rows must have equal length.")
    legend = observation.get("legend")
    if capabilities.get("legend") is True and not isinstance(legend, dict):
        raise ProtocolError("invalid_observation", "Legend capability requires observation.legend.")
    if legend is not None and (
        not isinstance(legend, dict)
        or not all(
            isinstance(glyph, str)
            and len(glyph) == 1
            and isinstance(meaning, str)
            for glyph, meaning in legend.items()
        )
    ):
        raise ProtocolError("invalid_observation", "Observation legend must map glyphs to strings.")
    return observation
