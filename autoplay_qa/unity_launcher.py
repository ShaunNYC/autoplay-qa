"""Managed launcher for a standalone Unity ``autoplay-qa/1`` player.

The injected defect is deliberately process configuration.  This module never
adds it to a protocol request or to the launch metadata exposed to prompt and
recording code.
"""

from __future__ import annotations

import os
from math import isfinite
from pathlib import Path
import socket
import subprocess
import time
from typing import Callable

from .protocol import ProtocolError
from .remote import RemoteGameAdapter


UNITY_DEFECT_CHOICES = (
    "none",
    "floor-gap",
    "door-ghost-collider",
    "key-not-consumed",
    "respawn-leak",
    "wall-tunnel",
)
UNITY_BENCHMARK_DEFECTS = UNITY_DEFECT_CHOICES

_UNITY_DEFECT_DESCRIPTIONS = {
    "floor-gap": "A visible solid floor tile has no working collider.",
    "door-ghost-collider": "An open, visibly non-solid door still blocks movement.",
    "key-not-consumed": "Unlock feedback says the key was used but it remains held.",
    "respawn-leak": "Reset or respawn preserves progress that should be restored.",
    "wall-tunnel": "A visible solid wall segment has no working collider.",
}

DEFAULT_STARTUP_TIMEOUT = 10.0
_CONNECT_RETRY_SECONDS = 0.05
_TERMINATE_TIMEOUT = 3.0


class UnityPlayerLaunchError(OSError):
    """The standalone player could not be started or reached in time."""


def validate_unity_defect(defect: str | None) -> str:
    name = "none" if defect is None else defect
    if name not in UNITY_DEFECT_CHOICES:
        choices = ", ".join(UNITY_DEFECT_CHOICES)
        raise ValueError(
            f"Unknown Unity injected defect {name!r}; choose one of: {choices}."
        )
    return name


def unity_defect_evaluation_metadata(defect: str | None) -> dict:
    """Return harness-only ground truth suitable for a session header."""
    name = validate_unity_defect(defect)
    injected = None if name == "none" else {
        "name": name,
        "description": _UNITY_DEFECT_DESCRIPTIONS[name],
    }
    return {"evaluation": {"injected_defect": injected}}


def _reserve_loopback_port() -> int:
    """Ask the OS for an available loopback TCP port, then release it."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


def _validated_player_path(player_path: str | os.PathLike[str]) -> Path:
    path = Path(player_path).expanduser().resolve()
    if not path.is_file():
        raise ValueError(f"Unity player does not exist or is not a file: {path}")
    if not os.access(path, os.X_OK):
        raise ValueError(f"Unity player is not executable: {path}")
    return path


def _validate_positive_timeout(value: float, label: str) -> float:
    if type(value) not in {int, float} or not isfinite(value) or value <= 0:
        raise ValueError(f"{label} must be positive.")
    return float(value)


def _terminate_process(process: subprocess.Popen | None) -> None:
    if process is None or process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=_TERMINATE_TIMEOUT)
        return
    except subprocess.TimeoutExpired:
        pass
    process.kill()
    try:
        process.wait(timeout=_TERMINATE_TIMEOUT)
    except subprocess.TimeoutExpired:
        # There is no stronger portable operation than kill.  The caller has
        # still made both bounded cleanup attempts and must not hang forever.
        pass


class UnityPlayerGame:
    """A ``GameAdapter`` that also owns its standalone Unity process."""

    def __init__(
        self,
        player_path: str | os.PathLike[str],
        *,
        seed: int,
        defect: str | None = None,
        timeout: float = 5.0,
        startup_timeout: float = DEFAULT_STARTUP_TIMEOUT,
    ):
        if type(seed) is not int or not -(2**31) <= seed < 2**31:
            raise ValueError("Unity player seed must fit a signed 32-bit integer.")
        request_timeout = _validate_positive_timeout(
            timeout, "Unity player request timeout"
        )
        startup_timeout = _validate_positive_timeout(
            startup_timeout, "Unity player startup timeout"
        )
        selected_defect = validate_unity_defect(defect)
        path = _validated_player_path(player_path)
        port = _reserve_loopback_port()
        command = [
            str(path),
            "-batchmode",
            "-nographics",
            "-autoplayqa-port",
            str(port),
            "-autoplayqa-seed",
            str(seed),
        ]
        if selected_defect != "none":
            command.extend(("-autoplayqa-defect", selected_defect))

        self.player_path = path
        self.endpoint = f"tcp://127.0.0.1:{port}"
        self._process: subprocess.Popen | None = None
        self._adapter: RemoteGameAdapter | None = None
        self._closed = False
        try:
            # Unity writes its own player log.  Pipes are intentionally avoided
            # because an unconsumed verbose log can fill and deadlock a player.
            self._process = subprocess.Popen(
                command,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            self._adapter = self._connect(
                seed=seed,
                request_timeout=request_timeout,
                startup_timeout=startup_timeout,
            )
        except BaseException:
            _terminate_process(self._process)
            self._closed = True
            raise

        self.seed = self._adapter.seed
        self.version = self._adapter.version
        self.game_id = self._adapter.game_id
        self.controls = self._adapter.controls
        self.capabilities = self._adapter.capabilities
        self.session_metadata = {
            "unity_player": {
                "path": str(self.player_path),
                # The protocol game version is the stable compatibility
                # version the launched player reports during its handshake.
                "version": self.version,
            }
        }

    def _connect(
        self,
        *,
        seed: int,
        request_timeout: float,
        startup_timeout: float,
    ) -> RemoteGameAdapter:
        deadline = time.monotonic() + startup_timeout
        last_error: OSError | ValueError | None = None
        while True:
            process = self._process
            return_code = process.poll() if process is not None else None
            if return_code is not None:
                raise UnityPlayerLaunchError(
                    "Unity player exited before its protocol listener became ready "
                    f"(exit code {return_code})."
                )
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                detail = f": {last_error}" if last_error is not None else ""
                raise UnityPlayerLaunchError(
                    "Timed out waiting for the Unity player protocol listener" + detail
                )
            attempt_timeout = min(request_timeout, max(_CONNECT_RETRY_SECONDS, remaining))
            try:
                adapter = RemoteGameAdapter(
                    self.endpoint,
                    timeout=attempt_timeout,
                    seed=seed,
                )
            except (OSError, ProtocolError) as exc:
                last_error = exc
                time.sleep(min(_CONNECT_RETRY_SECONDS, max(0.0, remaining)))
                continue

            # Startup attempts use a short bound so a not-yet-ready player does
            # not consume the whole deadline.  Restore the requested bound for
            # normal game operations after the handshake succeeds.
            adapter.timeout = request_timeout
            connection = getattr(adapter, "_socket", None)
            if connection is not None:
                connection.settimeout(request_timeout)
            return adapter

    def observe(self) -> dict:
        return self._require_adapter().observe()

    def act(self, action: str) -> dict:
        return self._require_adapter().act(action)

    def reset(self, seed: int | None = None) -> "UnityPlayerGame":
        adapter = self._require_adapter()
        adapter.reset(seed)
        self.seed = adapter.seed
        return self

    def evaluate_transition(self, before: dict, action: str, after: dict) -> dict:
        return self._require_adapter().evaluate_transition(before, action, after)

    def _require_adapter(self) -> RemoteGameAdapter:
        if self._closed or self._adapter is None:
            raise ValueError("Unity player game is closed.")
        return self._adapter

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        error: BaseException | None = None
        try:
            if self._adapter is not None:
                self._adapter.close()
        except BaseException as exc:
            error = exc
        finally:
            _terminate_process(self._process)
        if error is not None:
            raise error

    def __enter__(self) -> "UnityPlayerGame":
        return self

    def __exit__(self, *_args) -> None:
        self.close()

    def __del__(self):
        if hasattr(self, "_closed") and not self._closed:
            try:
                self.close()
            except BaseException:
                pass


def unity_game_factory(
    player_path: str | os.PathLike[str],
    *,
    timeout: float = 5.0,
    startup_timeout: float = DEFAULT_STARTUP_TIMEOUT,
) -> Callable[[int, str | None], UnityPlayerGame]:
    """Create the factory shape used by benchmark/report/reproduction code."""
    path = Path(player_path)

    def create(seed: int, defect: str | None) -> UnityPlayerGame:
        return UnityPlayerGame(
            path,
            seed=seed,
            defect=defect,
            timeout=timeout,
            startup_timeout=startup_timeout,
        )

    return create
