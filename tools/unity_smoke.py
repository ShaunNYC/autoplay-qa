#!/usr/bin/env python3
"""Strict, dependency-free smoke client for the Unity AutoPlay QA sample."""

from __future__ import annotations

import argparse
import collections
import json
import math
import socket
import sys
import time
from typing import Any, BinaryIO, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


PROTOCOL = "autoplay-qa/1"
MAX_LINE_BYTES = 1024 * 1024
REQUIRED_CONTROLS = ("north", "south", "east", "west", "jump", "interact", "wait")
DEFAULT_ACTIONS = (
    "north",
    "north",
    "north",
    "north",
    "east",
    "east",
    "interact",
    "east",
    "east",
    "east",
)
REQUIRED_CAPABILITIES = (
    "entities",
    "reset",
    "seed",
    "defect_injection",
    "continuous",
    "solids_visible",
)
LAYOUT_KINDS = ("floor", "wall", "door", "key", "exit")
SOLID_KINDS = ("floor", "wall", "door")


class SmokeError(RuntimeError):
    """A transport, protocol, or sample-schema failure."""


def _parse_port(text: str) -> int:
    try:
        value = int(text, 10)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("port must be an integer") from exc
    if not 1 <= value <= 65535:
        raise argparse.ArgumentTypeError("port must be between 1 and 65535")
    return value


def _parse_seed(text: str) -> int:
    try:
        value = int(text, 10)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("seed must be an integer") from exc
    if not -(2**31) <= value < 2**31:
        raise argparse.ArgumentTypeError("seed must fit a signed 32-bit integer")
    return value


def _parse_positive_float(text: str) -> float:
    try:
        value = float(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("timeout must be a number") from exc
    if not math.isfinite(value) or value <= 0.0:
        raise argparse.ArgumentTypeError("timeout must be finite and greater than zero")
    return value


def _parse_loopback_host(text: str) -> str:
    if text not in ("127.0.0.1", "localhost", "::1"):
        raise argparse.ArgumentTypeError("host must be a loopback address")
    return text


def _parse_actions(text: str) -> Sequence[str]:
    actions = tuple(part.strip() for part in text.split(","))
    if not actions or any(not action for action in actions):
        raise argparse.ArgumentTypeError("actions must be a comma-separated list")
    return actions


def _reject_constant(value: str) -> None:
    raise SmokeError("response contains forbidden JSON number " + value)


def _unique_object(pairs: Iterable[tuple[str, Any]]) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise SmokeError("response contains duplicate JSON key " + repr(key))
        result[key] = value
    return result


def _decode_json(payload: bytes) -> Mapping[str, Any]:
    try:
        text = payload.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise SmokeError("response is not valid UTF-8") from exc
    try:
        value = json.loads(
            text,
            parse_constant=_reject_constant,
            object_pairs_hook=_unique_object,
        )
    except SmokeError:
        raise
    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        raise SmokeError("response is not one valid JSON value: " + str(exc)) from exc
    _require_finite_json_numbers(value)
    if not isinstance(value, dict):
        raise SmokeError("response JSON value is not an object")
    return value


def _require_finite_json_numbers(value: Any) -> None:
    pending = [("response", value)]
    while pending:
        path, item = pending.pop()
        if isinstance(item, float) and not math.isfinite(item):
            raise SmokeError(path + " contains a non-finite JSON number")
        if isinstance(item, dict):
            pending.extend(
                (path + "." + str(key), child) for key, child in item.items()
            )
        elif isinstance(item, list):
            pending.extend(
                (path + "[" + str(index) + "]", child)
                for index, child in enumerate(item)
            )


def _nonempty_string(value: Any, path: str) -> str:
    if not isinstance(value, str) or not value:
        raise SmokeError(path + " must be a nonempty string")
    return value


def _finite_number(value: Any, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SmokeError(path + " must be a JSON number")
    number = float(value)
    if not math.isfinite(number):
        raise SmokeError(path + " must be finite")
    return number


def _vector(
    value: Any,
    path: str,
    *,
    dimensions: int = 3,
    nonnegative: bool = False,
) -> List[float]:
    if not isinstance(value, list) or len(value) != dimensions:
        raise SmokeError(path + " must be a " + str(dimensions) + "-element array")
    result = []
    for index, item in enumerate(value):
        number = _finite_number(item, path + "[" + str(index) + "]")
        if nonnegative and number < 0.0:
            raise SmokeError(path + " values must be non-negative")
        result.append(number)
    return result


def _string_list(value: Any, path: str, *, unique: bool = False) -> List[str]:
    if not isinstance(value, list):
        raise SmokeError(path + " must be an array")
    result = []
    seen = set()
    for index, item in enumerate(value):
        item = _nonempty_string(item, path + "[" + str(index) + "]")
        if unique and item in seen:
            raise SmokeError(path + " contains duplicate value " + repr(item))
        result.append(item)
        seen.add(item)
    return result


class ProtocolClient:
    def __init__(self, connection: socket.socket):
        self._connection = connection
        self._stream: BinaryIO = connection.makefile("rwb")
        self._sequence = 0
        self._closed = False

    def __enter__(self) -> "ProtocolClient":
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        try:
            self._stream.close()
        finally:
            self._connection.close()

    def request(self, message_type: str, **fields: Any) -> Mapping[str, Any]:
        if self._closed:
            raise SmokeError("cannot send after close")
        self._sequence += 1
        request_id = "smoke-" + str(self._sequence)
        request: Dict[str, Any] = {"id": request_id, "type": message_type}
        request.update(fields)
        encoded = json.dumps(
            request,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        ).encode("utf-8")
        if len(encoded) > MAX_LINE_BYTES:
            raise SmokeError("request exceeds the 1 MiB line limit")

        try:
            self._stream.write(encoded + b"\n")
            self._stream.flush()
            raw = self._stream.readline(MAX_LINE_BYTES + 2)
        except (OSError, socket.timeout) as exc:
            raise SmokeError(message_type + " transport failed: " + str(exc)) from exc

        if not raw:
            raise SmokeError("server closed before responding to " + message_type)
        if not raw.endswith(b"\n"):
            if len(raw) > MAX_LINE_BYTES:
                raise SmokeError("response exceeds the 1 MiB line limit")
            raise SmokeError("response is not newline-terminated")
        payload = raw[:-1]
        if payload.endswith(b"\r"):
            payload = payload[:-1]
        if len(payload) > MAX_LINE_BYTES:
            raise SmokeError("response exceeds the 1 MiB line limit")

        response = _decode_json(payload)
        if response.get("id") != request_id:
            raise SmokeError(
                "response id mismatch: expected "
                + repr(request_id)
                + ", got "
                + repr(response.get("id"))
            )
        if response.get("ok") is False or response.get("type") == "error":
            self._raise_server_error(response)
        if response.get("ok") is not True:
            raise SmokeError("successful response must contain ok:true")
        if response.get("type") != message_type:
            raise SmokeError(
                "response type mismatch: expected "
                + repr(message_type)
                + ", got "
                + repr(response.get("type"))
            )
        if message_type == "close":
            self._closed = True
        return response

    @staticmethod
    def _raise_server_error(response: Mapping[str, Any]) -> None:
        if response.get("type") != "error" or response.get("ok") is not False:
            raise SmokeError("malformed error response envelope")
        error = response.get("error")
        if not isinstance(error, dict):
            raise SmokeError("error response is missing its error object")
        code = _nonempty_string(error.get("code"), "error.code")
        message = _nonempty_string(error.get("message"), "error.message")
        if not isinstance(error.get("retryable"), bool):
            raise SmokeError("error.retryable must be boolean")
        raise SmokeError("server error " + code + ": " + message)


def _connect(host: str, port: int, connect_timeout: float, response_timeout: float) -> socket.socket:
    deadline = time.monotonic() + connect_timeout
    last_error: Optional[OSError] = None
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0.0:
            detail = "" if last_error is None else ": " + str(last_error)
            raise SmokeError("could not connect to tcp://" + host + ":" + str(port) + detail)
        try:
            connection = socket.create_connection(
                (host, port), timeout=min(remaining, response_timeout)
            )
            connection.settimeout(response_timeout)
            return connection
        except OSError as exc:
            last_error = exc
            time.sleep(min(0.1, max(0.0, remaining)))


def _validate_hello(response: Mapping[str, Any]) -> Tuple[List[str], Mapping[str, bool]]:
    if response.get("protocol") != PROTOCOL:
        raise SmokeError("hello response does not select " + PROTOCOL)
    game = response.get("game")
    if not isinstance(game, dict):
        raise SmokeError("hello.game must be an object")
    _nonempty_string(game.get("id"), "hello.game.id")
    _nonempty_string(game.get("version"), "hello.game.version")

    controls = _string_list(response.get("controls"), "hello.controls", unique=True)
    for expected in REQUIRED_CONTROLS:
        if expected not in controls:
            raise SmokeError("hello.controls is missing " + repr(expected))

    capabilities = response.get("capabilities")
    if not isinstance(capabilities, dict):
        raise SmokeError("hello.capabilities must be an object")
    for name, value in capabilities.items():
        _nonempty_string(name, "hello.capabilities key")
        if not isinstance(value, bool):
            raise SmokeError("hello.capabilities." + name + " must be boolean")
    for name in REQUIRED_CAPABILITIES:
        if capabilities.get(name) is not True:
            raise SmokeError("PhysicsKeyDoor must advertise " + name + ":true")
    return controls, capabilities


def _validate_observation(
    response: Mapping[str, Any],
    catalog: Sequence[str],
    capabilities: Mapping[str, bool],
    *,
    require_fresh_layout: bool,
) -> Mapping[str, Any]:
    observation = response.get("observation")
    if not isinstance(observation, dict):
        raise SmokeError("response.observation must be an object")
    _nonempty_string(observation.get("status"), "observation.status")

    tick = observation.get("tick")
    if tick is not None and (isinstance(tick, bool) or not isinstance(tick, int)):
        raise SmokeError("observation.tick must be an integer when present")

    player = observation.get("player")
    if not isinstance(player, dict):
        raise SmokeError("observation.player must be an object")
    position = _vector(player.get("position"), "observation.player.position")
    for index, number in enumerate(position):
        if abs(number - round(number, 2)) > 1e-9:
            raise SmokeError(
                "observation.player.position["
                + str(index)
                + "] is not rounded to two decimal places"
            )
    _vector(player.get("velocity"), "observation.player.velocity")
    if not isinstance(player.get("grounded"), bool):
        raise SmokeError("observation.player.grounded must be boolean")
    _string_list(player.get("inventory"), "observation.player.inventory", unique=True)

    entities = observation.get("entities")
    if not isinstance(entities, list):
        raise SmokeError("observation.entities must be an array")
    entity_ids = set()
    seen_kinds = set()
    for index, entity in enumerate(entities):
        path = "observation.entities[" + str(index) + "]"
        if not isinstance(entity, dict):
            raise SmokeError(path + " must be an object")
        entity_id = _nonempty_string(entity.get("id"), path + ".id")
        if entity_id in entity_ids:
            raise SmokeError("observation.entities contains duplicate id " + repr(entity_id))
        entity_ids.add(entity_id)
        kind = _nonempty_string(entity.get("kind"), path + ".kind")
        if kind not in LAYOUT_KINDS:
            raise SmokeError(path + ".kind is not a PhysicsKeyDoor layout kind")
        seen_kinds.add(kind)
        if not isinstance(entity.get("state"), dict):
            raise SmokeError(path + ".state must be an object")
        if "position" in entity:
            _vector(entity["position"], path + ".position")
        if "size" in entity:
            _vector(entity["size"], path + ".size", nonnegative=True)
        if capabilities.get("solids_visible") and kind in SOLID_KINDS:
            if "position" not in entity or "size" not in entity:
                raise SmokeError(path + " must expose position and size for solids_visible")

    if require_fresh_layout:
        missing = set(LAYOUT_KINDS).difference(seen_kinds)
        if missing:
            raise SmokeError(
                "fresh PhysicsKeyDoor observation is missing entity kinds: "
                + ", ".join(sorted(missing))
            )

    _string_list(observation.get("events"), "observation.events")
    available = _string_list(
        observation.get("controls"), "observation.controls", unique=True
    )
    unknown = set(available).difference(catalog)
    if unknown:
        raise SmokeError(
            "observation.controls contains controls absent from hello: "
            + ", ".join(sorted(unknown))
        )
    if observation["status"] in ("won", "lost") and available:
        raise SmokeError("terminal observation.controls must be empty")
    return observation


def _print_hello(response: Mapping[str, Any]) -> None:
    record = {
        "step": "hello",
        "game": response["game"],
        "controls": response["controls"],
        "capabilities": response["capabilities"],
    }
    print(json.dumps(record, ensure_ascii=False, separators=(",", ":")))


def _print_observation(step: str, observation: Mapping[str, Any]) -> None:
    player = observation["player"]
    counts = collections.Counter(entity["kind"] for entity in observation["entities"])
    selected_states = {
        entity["id"]: entity["state"]
        for entity in observation["entities"]
        if entity["kind"] in ("door", "key", "exit")
    }
    record = {
        "step": step,
        "tick": observation.get("tick"),
        "status": observation["status"],
        "position": player["position"],
        "velocity": player["velocity"],
        "grounded": player["grounded"],
        "inventory": player["inventory"],
        "entity_counts": dict(sorted(counts.items())),
        "states": selected_states,
        "events": observation["events"],
    }
    print(json.dumps(record, ensure_ascii=False, separators=(",", ":")))


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Smoke-test the Unity PhysicsKeyDoor autoplay-qa/1 server."
    )
    parser.add_argument("--host", type=_parse_loopback_host, default="127.0.0.1")
    parser.add_argument("--port", type=_parse_port, default=8765)
    parser.add_argument("--seed", type=_parse_seed, default=5)
    parser.add_argument(
        "--actions",
        type=_parse_actions,
        default=DEFAULT_ACTIONS,
        metavar="A,B,...",
        help="comma-separated action script (default: %(default)s)",
    )
    parser.add_argument(
        "--connect-timeout",
        type=_parse_positive_float,
        default=10.0,
        help="seconds to wait for the player to start listening",
    )
    parser.add_argument(
        "--response-timeout",
        type=_parse_positive_float,
        default=10.0,
        help="seconds to wait for each protocol response",
    )
    return parser


def run(arguments: argparse.Namespace) -> None:
    connection = _connect(
        arguments.host,
        arguments.port,
        arguments.connect_timeout,
        arguments.response_timeout,
    )
    with ProtocolClient(connection) as client:
        hello = client.request("hello", protocol=PROTOCOL)
        catalog, capabilities = _validate_hello(hello)
        _print_hello(hello)

        reset = client.request("reset", seed=arguments.seed)
        observation = _validate_observation(
            reset, catalog, capabilities, require_fresh_layout=True
        )
        if observation["status"] != "playing":
            raise SmokeError("reset did not produce status 'playing'")
        _print_observation("reset", observation)

        observed = client.request("observe")
        observation = _validate_observation(
            observed, catalog, capabilities, require_fresh_layout=True
        )
        _print_observation("observe", observation)

        for index, action in enumerate(arguments.actions, start=1):
            if observation["status"] != "playing":
                break
            if action not in observation["controls"]:
                raise SmokeError(
                    "script action " + repr(action) + " is not currently available"
                )
            acted = client.request("act", control=action)
            observation = _validate_observation(
                acted, catalog, capabilities, require_fresh_layout=False
            )
            _print_observation("act-" + str(index) + ":" + action, observation)

        client.request("close")
        print('{"step":"close","ok":true}')


def main(argv: Optional[Sequence[str]] = None) -> int:
    arguments = _build_parser().parse_args(argv)
    try:
        run(arguments)
    except (SmokeError, OSError) as exc:
        print("unity_smoke: " + str(exc), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
