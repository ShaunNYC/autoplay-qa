"""Test-only adapter to the unchanged Outpost: Mars TypeScript engine.

The subprocess executes the game's own exported ``execute`` function.  It does
not implement game rules, inject defects, edit source, or supply a solution.
"""

from __future__ import annotations

from copy import deepcopy
from math import isfinite
import os
from pathlib import Path
import queue
import subprocess
import threading
from typing import Callable

from .protocol import (
    PROTOCOL_VERSION,
    ProtocolError,
    adapter_observation,
    read_message,
    validate_wire_observation,
    write_message,
)


class MarsAdapter:
    """One original-engine subprocess; acknowledged controls are never retried."""

    def __init__(
        self,
        project_dir: str | Path,
        *,
        seed: int = 0,
        map_id: str | None = None,
        node: str = "node",
        cache_dir: str | Path | None = None,
        timeout: float = 10,
        semantic_state: bool = False,
    ):
        self.semantic_state = semantic_state
        if type(seed) is not int or abs(seed) > (2**53 - 1):
            raise ValueError("Game seed must be a safe integer.")
        if type(timeout) not in {int, float} or not isfinite(timeout) or timeout <= 0:
            raise ValueError("Game timeout must be positive.")
        if map_id is not None and (not isinstance(map_id, str) or not map_id):
            raise ValueError("Map ID must be a nonempty string.")
        root = Path(__file__).resolve().parent.parent
        cache = Path(cache_dir) if cache_dir is not None else root / ".dashboard" / "mars-engine"
        command = [
            node, str(root / "tools" / "mars_bridge.mjs"),
            "--project", str(Path(project_dir).resolve()),
            "--cache", str(cache.resolve()),
            "--seed", str(seed),
        ]
        if map_id is not None:
            command += ["--map", map_id]
        # The engine needs no provider or bridge credentials from this process.
        child_env = {name: os.environ[name] for name in ("PATH", "TMPDIR", "SystemRoot") if name in os.environ}
        self._process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            env=child_env,
        )
        self._lock = threading.Lock()
        self._responses: queue.Queue[dict | BaseException | None] = queue.Queue()
        self._closed = False
        self._next_id = 1
        self.timeout = float(timeout)
        self.seed = seed
        self.capabilities: dict[str, bool] = {}
        self.controls: list[str] = []
        self._reader = threading.Thread(
            target=self._read_responses,
            args=(self._process.stdout, self._responses),
            daemon=True,
        )
        self._reader.start()
        try:
            hello = self._exchange("hello", protocol=PROTOCOL_VERSION)
            if hello.get("protocol") != PROTOCOL_VERSION:
                raise ProtocolError("unsupported_protocol", "The engine returned an unsupported protocol.")
            game = hello.get("game")
            if not isinstance(game, dict) or not isinstance(game.get("version"), str):
                raise ProtocolError("invalid_response", "The engine returned an invalid identity.")
            self.game_id = game["id"]
            self.version = game["version"]
            controls, capabilities = hello.get("controls"), hello.get("capabilities")
            if not isinstance(controls, list) or not controls or not all(isinstance(item, str) and item for item in controls) or len(set(controls)) != len(controls):
                raise ProtocolError("invalid_response", "The engine returned invalid controls.")
            if not isinstance(capabilities, dict) or not all(isinstance(name, str) and type(value) is bool for name, value in capabilities.items()):
                raise ProtocolError("invalid_response", "The engine returned invalid capabilities.")
            self.controls = list(controls)
            self.capabilities = dict(capabilities)
            self.source_manifest = deepcopy(self._exchange("sourceManifest")["manifest"])
            self.map_id = self.source_manifest["mapId"]
            self._observation = self._observation_from(self._exchange("observe"))
        except BaseException:
            self._close_transport()
            raise

    @staticmethod
    def _read_responses(stream, responses: queue.Queue) -> None:
        # A bound reader target would keep the adapter alive after a factory
        # reset, preventing its finalizer from closing the original subprocess.
        assert stream is not None
        try:
            while True:
                response = read_message(stream)
                responses.put(response)
                if response is None:
                    return
        except BaseException as error:
            responses.put(error)

    def _exchange(self, request_type: str, **fields) -> dict:
        with self._lock:
            if self._closed:
                raise ProtocolError("closed", "Mars engine is closed.")
            request_id = str(self._next_id)
            self._next_id += 1
            assert self._process.stdin is not None
            try:
                write_message(self._process.stdin, {"id": request_id, "type": request_type, **fields})
                response = self._responses.get(timeout=self.timeout)
            except queue.Empty:
                self._close_transport()
                raise ProtocolError("timeout", "Timed out waiting for Mars; the engine was stopped.") from None
            except (BrokenPipeError, OSError):
                self._close_transport()
                raise ProtocolError("connection_closed", "Mars engine stopped before acknowledging the request.") from None
            if isinstance(response, BaseException):
                self._close_transport()
                raise response
            if response is None:
                self._close_transport()
                raise ProtocolError("connection_closed", "Mars engine could not load or has stopped.")
            if response.get("id") != request_id:
                self._close_transport()
                raise ProtocolError("invalid_response", "Mars response did not match the request.")
            if response.get("ok") is not True or response.get("type") == "error":
                error = response.get("error")
                error = error if isinstance(error, dict) else {}
                raise ProtocolError(str(error.get("code", "remote_error")), str(error.get("message", "Mars request failed.")))
            if response.get("type") != request_type:
                raise ProtocolError("invalid_response", "Mars response type did not match the request.")
            return response

    def _observation_from(self, response: dict) -> dict:
        wire = validate_wire_observation(response.get("observation"), self.capabilities)
        observation = adapter_observation(wire, game_version=self.version)
        observation["_capabilities"] = deepcopy(self.capabilities)
        if self.semantic_state:
            # Public telemetry remains visible. Only experiment identity omits
            # these adapter-declared displays of the last execution/coverage.
            observation["_capabilities"]["state_identity"] = {"transient_paths": [
                "entities.execution", "objects.execution",
            ]}
        return observation

    def observe(self) -> dict:
        self._observation = self._observation_from(self._exchange("observe"))
        return deepcopy(self._observation)

    def act(self, action: str) -> dict:
        # Rejected game execution (boundary, duplicate bomb, empty removal)
        # is real feedback. Only unknown adapter controls are rejected here.
        if not isinstance(action, str) or action not in self.controls:
            raise ValueError("Unknown Mars control.")
        self._observation = self._observation_from(self._exchange("act", control=action))
        return deepcopy(self._observation)

    def reset(self, seed: int | None = None) -> "MarsAdapter":
        next_seed = self.seed if seed is None else seed
        if type(next_seed) is not int or abs(next_seed) > (2**53 - 1):
            raise ValueError("Game seed must be a safe integer.")
        self._observation = self._observation_from(self._exchange("reset", seed=next_seed))
        self.seed = next_seed
        # A seed-only adapter may choose another original map and catalog.
        self.source_manifest = deepcopy(self._exchange("sourceManifest")["manifest"])
        self.map_id = self.source_manifest["mapId"]
        self.controls = list(self._observation["actions"])
        return self

    def list_maps(self) -> list[dict]:
        return deepcopy(self._exchange("listmaps")["maps"])

    def evaluate_transition(self, before: dict, action: str, after: dict) -> dict:
        # The KeyGate-specific movement/contracts do not apply to this drone.
        return {
            "moved": before.get("player", {}).get("position") != after.get("player", {}).get("position"),
            "events": deepcopy(after.get("events", [])),
            "contract_violations": [],
        }

    def _close_transport(self) -> None:
        self._closed = True
        if self._process.poll() is None:
            self._process.terminate()
            try:
                self._process.wait(timeout=1)
            except subprocess.TimeoutExpired:
                self._process.kill()
                self._process.wait(timeout=1)
        for stream in (self._process.stdin, self._process.stdout):
            if stream is not None:
                try:
                    stream.close()
                except OSError:
                    pass

    def close(self) -> None:
        if self._closed:
            return
        try:
            self._exchange("close")
        finally:
            self._close_transport()

    def __enter__(self) -> "MarsAdapter":
        return self

    def __exit__(self, *_args) -> None:
        self.close()

    def __del__(self):
        if hasattr(self, "_closed") and not self._closed:
            self._close_transport()


MarsGameAdapter = MarsAdapter


def mars_game_factory(project_dir: str | Path, **options) -> Callable[[], MarsAdapter]:
    """Build an isolated fresh original-engine adapter for replay/reproduction."""
    return lambda: MarsAdapter(project_dir, **options)
