"""Socket-backed implementation of the existing ``GameAdapter`` boundary."""

from __future__ import annotations

from copy import deepcopy
import socket
import threading
from urllib.parse import urlparse

from .game import evaluate_transition
from .protocol import (
    LEGACY_OBSERVATION_EXTENSION,
    PROTOCOL_VERSION,
    ProtocolError,
    adapter_observation,
    read_message,
    validate_wire_observation,
    write_message,
)


LOCAL_HOSTS = {"127.0.0.1", "localhost", "::1"}


def parse_tcp_endpoint(endpoint: str) -> tuple[str, int]:
    parsed = urlparse(endpoint)
    if parsed.scheme != "tcp" or parsed.username is not None or parsed.password is not None:
        raise ValueError("Game endpoint must use tcp://localhost:PORT.")
    host = parsed.hostname
    try:
        port = parsed.port
    except ValueError:
        raise ValueError("Game endpoint has an invalid port.") from None
    if host not in LOCAL_HOSTS:
        raise ValueError("Remote game endpoints are restricted to localhost.")
    if (
        port is None
        or not 1 <= port <= 65535
        or parsed.path not in {"", "/"}
        or parsed.params
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError("Game endpoint must use tcp://localhost:PORT.")
    return host, port


class RemoteGameAdapter:
    """Synchronous stdlib TCP client with one request in flight at a time."""

    def __init__(self, endpoint: str, *, timeout: float = 5.0, seed: int | None = None):
        if type(timeout) not in {int, float} or timeout <= 0:
            raise ValueError("Remote game timeout must be positive.")
        self.endpoint = endpoint
        self.timeout = float(timeout)
        self.host, self.port = parse_tcp_endpoint(endpoint)
        self._socket = socket.create_connection((self.host, self.port), timeout=self.timeout)
        self._socket.settimeout(self.timeout)
        self._stream = self._socket.makefile("rwb")
        self._lock = threading.Lock()
        self._next_id = 1
        self._closed = False
        self.capabilities: dict[str, bool] = {}
        self.controls: list[str] = []
        self.game_id = "remote"
        self.version = "remote-v1"
        self.seed = 0 if seed is None else seed
        self._observation: dict | None = None
        try:
            hello = self._exchange("hello", protocol=PROTOCOL_VERSION)
            if hello.get("protocol") != PROTOCOL_VERSION:
                raise ProtocolError(
                    "unsupported_protocol",
                    f"Server selected unsupported protocol {hello.get('protocol')!r}.",
                )
            game = hello.get("game")
            if (
                not isinstance(game, dict)
                or not isinstance(game.get("id"), str)
                or not game["id"]
                or not isinstance(game.get("version"), str)
                or not game["version"]
            ):
                raise ProtocolError("invalid_response", "Hello response has no valid game identity.")
            self.game_id = game["id"]
            version = game.get("version")
            self.version = version if isinstance(version, str) and version else self.game_id
            controls = hello.get("controls")
            capabilities = hello.get("capabilities")
            if (
                not isinstance(controls, list)
                or not all(isinstance(item, str) and item for item in controls)
                or len(set(controls)) != len(controls)
            ):
                raise ProtocolError("invalid_response", "Hello controls must be an array of strings.")
            if not isinstance(capabilities, dict) or not all(
                isinstance(name, str) and type(value) is bool
                for name, value in capabilities.items()
            ):
                raise ProtocolError("invalid_response", "Hello capabilities must be boolean fields.")
            required_capabilities = {
                "grid", "legend", "entities", "reset", "seed", "defect_injection",
            }
            if not required_capabilities.issubset(capabilities):
                raise ProtocolError(
                    "invalid_response",
                    "Hello response is missing required capability flags.",
                )
            self.controls = list(controls)
            self.capabilities = dict(capabilities)
            if seed is not None:
                self.reset(seed)
            else:
                self._observation = self._read_observation(self._exchange("observe"))
        except BaseException:
            self._close_transport()
            raise

    def _exchange(self, request_type: str, **fields) -> dict:
        with self._lock:
            if self._closed:
                raise ProtocolError("closed", "Remote game connection is closed.")
            request_id = str(self._next_id)
            self._next_id += 1
            write_message(
                self._stream,
                {"id": request_id, "type": request_type, **fields},
            )
            try:
                response = read_message(self._stream)
            except socket.timeout:
                # A late response would make request IDs ambiguous.  Treat the
                # connection as unusable; callers must reconnect and observe
                # before deciding whether a mutating request can be retried.
                self._close_transport()
                raise ProtocolError("timeout", "Timed out waiting for the game server.", retryable=True) from None
            if response is None:
                self._close_transport()
                raise ProtocolError("connection_closed", "Game server closed the connection.", retryable=True)
            if response.get("id") != request_id:
                raise ProtocolError("invalid_response", "Response id does not match the request id.")
            if response.get("type") == "error" or response.get("ok") is False:
                error = response.get("error")
                error = error if isinstance(error, dict) else {}
                raise ProtocolError(
                    str(error.get("code", "remote_error")),
                    str(error.get("message", "Remote game request failed.")),
                    retryable=error.get("retryable") is True,
                )
            if response.get("type") != request_type or response.get("ok") is not True:
                raise ProtocolError("invalid_response", "Response type or success flag is invalid.")
            return response

    def _read_observation(self, response: dict) -> dict:
        wire = validate_wire_observation(response.get("observation"), self.capabilities)
        if any(control not in self.controls for control in wire["controls"]):
            raise ProtocolError(
                "invalid_observation",
                "Observation advertised a control absent from the hello catalog.",
            )
        observation = adapter_observation(wire, game_version=self.version)
        extensions = wire.get("extensions")
        has_legacy = (
            isinstance(extensions, dict)
            and isinstance(extensions.get(LEGACY_OBSERVATION_EXTENSION), dict)
        )
        if not has_legacy:
            observation["_capabilities"] = deepcopy(self.capabilities)
        return observation

    def observe(self) -> dict:
        # Observe is an actual protocol operation, not a cache read. Unity and
        # other engines may advance visible state between acknowledged inputs.
        self._observation = self._read_observation(self._exchange("observe"))
        return deepcopy(self._observation)

    def act(self, action: str) -> dict:
        if not isinstance(action, str) or not action:
            raise ValueError("Game control must be a nonempty string.")
        response = self._exchange("act", control=action)
        self._observation = self._read_observation(response)
        return deepcopy(self._observation)

    def reset(self, seed: int | None = None) -> "RemoteGameAdapter":
        if self.capabilities.get("reset") is not True:
            raise ValueError("Remote game does not advertise reset support.")
        if seed is None:
            seed = self.seed
        if type(seed) is not int:
            raise ValueError("Game seed must be an integer.")
        fields = {"seed": seed} if self.capabilities.get("seed") is True else {}
        response = self._exchange("reset", **fields)
        self.seed = seed
        self._observation = self._read_observation(response)
        return self

    def evaluate_transition(self, before: dict, action: str, after: dict) -> dict:
        """Use legacy public contracts when available, otherwise generic facts."""
        required = {"tick", "map", "objects", "player", "events"}
        if required.issubset(before) and required.issubset(after):
            return evaluate_transition(before, action, after)
        old = before.get("player", {}).get("position") if isinstance(before.get("player"), dict) else None
        new = after.get("player", {}).get("position") if isinstance(after.get("player"), dict) else None
        return {
            "moved": old is not None and new is not None and old != new,
            "events": deepcopy(after.get("events", [])),
            "contract_violations": [],
        }

    def _close_transport(self) -> None:
        self._closed = True
        try:
            self._stream.close()
        except (AttributeError, OSError):
            pass
        try:
            self._socket.close()
        except (AttributeError, OSError):
            pass

    def close(self) -> None:
        if self._closed:
            return
        try:
            self._exchange("close")
        finally:
            self._close_transport()

    def __enter__(self) -> "RemoteGameAdapter":
        return self

    def __exit__(self, *_args) -> None:
        self.close()

    def __del__(self):
        if hasattr(self, "_closed") and not self._closed:
            self._close_transport()
