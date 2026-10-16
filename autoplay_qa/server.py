"""Reference KeyGate server for the AutoPlay QA JSON-lines protocol."""

from __future__ import annotations

import socketserver
import sys
from typing import BinaryIO, Callable

from .game import ACTIONS, KeyGateGame, validate_defect
from .protocol import (
    PROTOCOL_VERSION,
    ProtocolError,
    error_response,
    keygate_protocol_observation,
    read_message,
    validate_request_envelope,
    write_message,
)


CAPABILITIES = {
    "grid": True,
    "legend": True,
    "entities": True,
    "reset": True,
    "seed": True,
    # This says that a harness may configure the server.  It never reveals
    # whether one is active or which behavior was selected.
    "defect_injection": True,
}


class ReferenceProtocolSession:
    def __init__(self, *, seed: int = 0, defect: str = "none"):
        self.seed = seed
        self.defect = validate_defect(defect)
        self.game = KeyGateGame(seed, self.defect)
        self.hello_complete = False
        self.request_ids: set[str] = set()
        self.closed = False

    def _observation(self) -> dict:
        return keygate_protocol_observation(self.game.observe())

    def handle(self, message: dict) -> tuple[dict, bool]:
        request_id, request_type = validate_request_envelope(message)
        if request_id in self.request_ids:
            raise ProtocolError("duplicate_id", "Request id was already used on this connection.")
        self.request_ids.add(request_id)
        if not self.hello_complete and request_type != "hello":
            raise ProtocolError("hello_required", "hello must be the first request.")
        if request_type == "hello":
            if self.hello_complete:
                raise ProtocolError("already_initialized", "hello may be sent only once.")
            if message.get("protocol") != PROTOCOL_VERSION:
                raise ProtocolError(
                    "unsupported_protocol",
                    f"Server supports only {PROTOCOL_VERSION}.",
                )
            self.hello_complete = True
            return {
                "id": request_id,
                "type": "hello",
                "ok": True,
                "protocol": PROTOCOL_VERSION,
                "game": {"id": "key-gate", "version": self.game.version},
                "controls": list(ACTIONS),
                "capabilities": dict(CAPABILITIES),
            }, False
        if request_type == "observe":
            return {
                "id": request_id,
                "type": "observe",
                "ok": True,
                "observation": self._observation(),
            }, False
        if request_type == "act":
            control = message.get("control")
            if not isinstance(control, str):
                raise ProtocolError("invalid_control", "act.control must be a string.")
            if control not in ACTIONS:
                raise ProtocolError("illegal_control", f"Control {control!r} is not currently legal.")
            try:
                self.game.act(control)
            except ValueError as exc:
                raise ProtocolError("illegal_control", str(exc)) from None
            return {
                "id": request_id,
                "type": "act",
                "ok": True,
                "observation": self._observation(),
            }, False
        if request_type == "reset":
            seed = message.get("seed")
            if type(seed) is not int:
                raise ProtocolError("invalid_seed", "reset.seed must be an integer.")
            if seed == self.seed:
                self.game.reset()
            else:
                self.seed = seed
                self.game = KeyGateGame(seed, self.defect)
            return {
                "id": request_id,
                "type": "reset",
                "ok": True,
                "observation": self._observation(),
            }, False
        if request_type == "close":
            self.closed = True
            return {"id": request_id, "type": "close", "ok": True}, True
        raise ProtocolError("unknown_type", f"Unknown request type {request_type!r}.")


def serve_stream(
    reader: BinaryIO,
    writer: BinaryIO,
    *,
    seed: int = 0,
    defect: str = "none",
) -> None:
    """Serve one connection over any binary line-oriented streams."""
    session = ReferenceProtocolSession(seed=seed, defect=defect)
    while not session.closed:
        request_id: str | None = None
        try:
            message = read_message(reader)
            if message is None:
                return
            candidate = message.get("id")
            request_id = candidate if isinstance(candidate, str) else None
            response, should_close = session.handle(message)
        except ProtocolError as exc:
            write_message(
                writer,
                error_response(request_id, exc.code, str(exc), retryable=exc.retryable),
            )
            continue
        write_message(writer, response)
        if should_close:
            return


class _ReferenceHandler(socketserver.StreamRequestHandler):
    def handle(self) -> None:
        server = self.server
        serve_stream(
            self.rfile,
            self.wfile,
            seed=server.initial_seed,
            defect=server.defect,
        )


class ReferenceGameServer(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True

    def __init__(self, address: tuple[str, int], *, seed: int = 0, defect: str = "none"):
        self.initial_seed = seed
        self.defect = validate_defect(defect)
        super().__init__(address, _ReferenceHandler)


def serve_tcp(port: int, *, seed: int = 0, defect: str = "none") -> None:
    if type(port) is not int or not 1 <= port <= 65535:
        raise ValueError("Server port must be between 1 and 65535.")
    with ReferenceGameServer(("127.0.0.1", port), seed=seed, defect=defect) as server:
        print(f"AutoPlay QA server listening on tcp://127.0.0.1:{port}", flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass


def serve_stdio(*, seed: int = 0, defect: str = "none") -> None:
    serve_stream(sys.stdin.buffer, sys.stdout.buffer, seed=seed, defect=defect)
