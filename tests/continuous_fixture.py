"""Socketpair-backed continuous game used by Phase-5b offline tests.

The fixture mirrors the public PhysicsKeyDoor layout and process-selected
defects, but it is deliberately small and deterministic.  Defect selection is
held by the server harness and never appears in protocol messages.
"""

from __future__ import annotations

from copy import deepcopy
import math
import socket
import threading
from typing import BinaryIO

from autoplay_qa.protocol import (
    PROTOCOL_VERSION,
    ProtocolError,
    adapter_observation,
    error_response,
    read_message,
    validate_request_envelope,
    write_message,
)


CONTINUOUS_DEFECTS = (
    "none",
    "floor-gap",
    "door-ghost-collider",
    "key-not-consumed",
    "respawn-leak",
    "wall-tunnel",
)
CONTROLS = ("north", "south", "east", "west", "jump", "interact", "wait")
CAPABILITIES = {
    "grid": False,
    "legend": False,
    "entities": True,
    "reset": True,
    "seed": True,
    "defect_injection": True,
    "continuous": True,
    "solids_visible": True,
}

SPAWN = (-4.0, 1.05, -4.0)
KEY_POSITION = (-4.0, 0.5, 2.0)
DOOR_POSITION = (0.0, 1.25, 2.0)
EXIT_POSITION = (4.0, 0.1, 2.0)
MOVE_DISTANCE = 1.55
PLAYER_RADIUS = 0.5

_DIRECTIONS = {
    "north": (0.0, 1.0),
    "south": (0.0, -1.0),
    "east": (1.0, 0.0),
    "west": (-1.0, 0.0),
}
_FLOOR_GAP_CANDIDATES = (
    # BuildLevel appends candidates in z-major, then x-major order.
    (2, 1), (4, 1), (1, 2), (5, 2),
    (1, 3), (4, 3), (2, 4), (5, 5),
)


def _u32(value: int) -> int:
    return value & 0xFFFFFFFF


def _select_index(seed: int, count: int, salt: int) -> int:
    value = _u32(seed) ^ salt
    value = _u32(value ^ (value >> 16))
    value = _u32(value * 0x7FEB352D)
    value = _u32(value ^ (value >> 15))
    value = _u32(value * 0x846CA68B)
    value = _u32(value ^ (value >> 16))
    return value % count


def _segment_box_entry_fraction(
    start: tuple[float, float],
    end: tuple[float, float],
    center: tuple[float, float],
    size: tuple[float, float],
    *,
    expansion: float = PLAYER_RADIUS,
) -> float | None:
    lower = (
        center[0] - size[0] / 2.0 - expansion,
        center[1] - size[1] / 2.0 - expansion,
    )
    upper = (
        center[0] + size[0] / 2.0 + expansion,
        center[1] + size[1] / 2.0 + expansion,
    )
    enter, leave = 0.0, 1.0
    for axis in range(2):
        delta = end[axis] - start[axis]
        if abs(delta) < 1e-12:
            if not lower[axis] < start[axis] < upper[axis]:
                return None
            continue
        near = (lower[axis] - start[axis]) / delta
        far = (upper[axis] - start[axis]) / delta
        if near > far:
            near, far = far, near
        enter = max(enter, near)
        leave = min(leave, far)
        if enter > leave:
            return None
    # Merely starting at a contact boundary while moving away is not a hit.
    if leave <= 1e-7 or enter >= 1.0 - 1e-7:
        return None
    return max(0.0, enter)


class FakeContinuousGame:
    """Deterministic public-observation analogue of PhysicsKeyDoor."""

    version = "physics-key-door-v1"
    game_id = "physics-key-door"
    controls = list(CONTROLS)
    capabilities = dict(CAPABILITIES)

    def __init__(self, seed: int = 0, defect: str | None = None):
        selected = "none" if defect is None else defect
        if selected not in CONTINUOUS_DEFECTS:
            raise ValueError("Unsupported fake continuous defect.")
        self.seed = seed
        self._defect = selected
        self._floor_entities = self._build_floors()
        self._wall_entities = self._build_walls()
        self._initialize_progress()
        self._select_geometry_defect()
        self._respawn()
        self.tick = 0
        self.events = ["Game started."]

    @staticmethod
    def _build_floors() -> list[dict]:
        return [
            {
                "id": f"floor-{x_index}-{z_index}",
                "kind": "floor",
                "position": [float(-6 + x_index * 2), -0.125, float(-6 + z_index * 2)],
                "size": [2.0, 0.25, 2.0],
                "state": {"solid": True},
            }
            for z_index in range(7)
            for x_index in range(7)
        ]

    @staticmethod
    def _build_walls() -> list[dict]:
        specifications = [
            ("wall-perimeter-north", (0.0, 7.25), (14.5, 0.5)),
            ("wall-perimeter-south", (0.0, -7.25), (14.5, 0.5)),
            ("wall-perimeter-east", (7.25, 0.0), (0.5, 14.5)),
            ("wall-perimeter-west", (-7.25, 0.0), (0.5, 14.5)),
        ]
        specifications.extend(
            (f"wall-interior-{index}", (0.0, z), (0.5, 2.0))
            for index, z in enumerate((-6.0, -4.0, -2.0, 0.0, 4.0, 6.0))
        )
        return [
            {
                "id": identifier,
                "kind": "wall",
                "position": [center[0], 1.25, center[1]],
                "size": [size[0], 2.5, size[1]],
                "state": {"solid": True},
            }
            for identifier, center, size in specifications
        ]

    def _initialize_progress(self) -> None:
        self.key_present = True
        self.has_key = False
        self.door_open = False
        self.won = False

    def _select_geometry_defect(self) -> None:
        candidate = _FLOOR_GAP_CANDIDATES[
            _select_index(self.seed, len(_FLOOR_GAP_CANDIDATES), 0x76A9)
        ]
        self._floor_gap_id = f"floor-{candidate[0]}-{candidate[1]}"
        interior = self._wall_entities[4:]
        self._wall_tunnel_id = interior[
            _select_index(self.seed, len(interior), 0x51F2)
        ]["id"]

    def _respawn(self) -> None:
        self.position = list(SPAWN)
        self.grounded = True
        self._floor_gap_wedge_attempts: int | None = None

    def reset(self, seed: int | None = None) -> "FakeContinuousGame":
        if seed is not None:
            if type(seed) is not int:
                raise ValueError("Seed must be an integer.")
            self.seed = seed
        preserve = self._defect == "respawn-leak"
        if not preserve:
            self._initialize_progress()
        else:
            self.won = False
        self._select_geometry_defect()
        self._respawn()
        self.tick = 0
        self.events = ["Game started."]
        return self

    def _dynamic_entities(self) -> list[dict]:
        return [
            {
                "id": "door",
                "kind": "door",
                "position": list(DOOR_POSITION),
                "size": [0.5, 2.5, 2.0],
                "state": {"open": self.door_open, "solid": not self.door_open},
            },
            {
                "id": "key",
                "kind": "key",
                **({"position": list(KEY_POSITION)} if self.key_present else {}),
                "size": [0.7, 0.7, 0.7],
                "state": {"present": self.key_present},
            },
            {
                "id": "exit",
                "kind": "exit",
                "position": list(EXIT_POSITION),
                "size": [1.5, 0.2, 1.5],
                "state": {"reached": self.won},
            },
        ]

    def wire_observation(self) -> dict:
        return {
            "tick": self.tick,
            "status": "won" if self.won else "playing",
            "player": {
                "position": [round(value, 2) for value in self.position],
                "velocity": [0.0, 0.0, 0.0],
                "grounded": self.grounded,
                "inventory": ["key"] if self.has_key else [],
            },
            "entities": deepcopy(self._floor_entities + self._wall_entities + self._dynamic_entities()),
            "events": list(self.events),
            "controls": [] if self.won else list(CONTROLS),
        }

    def observe(self) -> dict:
        value = adapter_observation(self.wire_observation(), game_version=self.version)
        value["_capabilities"] = dict(CAPABILITIES)
        return value

    def _colliders(self) -> list[tuple[str, tuple[float, float], tuple[float, float]]]:
        colliders = []
        for entity in self._wall_entities:
            if self._defect == "wall-tunnel" and entity["id"] == self._wall_tunnel_id:
                continue
            colliders.append((
                entity["id"],
                (entity["position"][0], entity["position"][2]),
                (entity["size"][0], entity["size"][2]),
            ))
        if not self.door_open or self._defect == "door-ghost-collider":
            colliders.append(("door", (DOOR_POSITION[0], DOOR_POSITION[2]), (0.5, 2.0)))
        return colliders

    def _floor_at(self, x: float, z: float) -> dict | None:
        for entity in self._floor_entities:
            ex, _, ez = entity["position"]
            if abs(x - ex) <= 1.0 and abs(z - ez) <= 1.0:
                return entity
        return None

    def _move(self, action: str) -> None:
        if self._floor_gap_wedge_attempts is not None:
            if self._floor_gap_wedge_attempts == 0:
                # Mirror the Unity failure mode: one later move is swallowed
                # while the capsule remains caught between neighboring floor
                # colliders.  A following move completes the fall so the
                # bounded sweep can recover and continue its generic scan.
                self._floor_gap_wedge_attempts = 1
                return
            self._initialize_progress()
            self._respawn()
            self.events.append("You fell.")
            return

        dx, dz = _DIRECTIONS[action]
        start = (self.position[0], self.position[2])
        end = (start[0] + dx * MOVE_DISTANCE, start[1] + dz * MOVE_DISTANCE)
        first_contact: float | None = None
        for _identifier, center, size in self._colliders():
            contact = _segment_box_entry_fraction(start, end, center, size)
            if contact is not None and (
                first_contact is None or contact < first_contact
            ):
                first_contact = contact

        if first_contact is not None:
            # Unity's rigidbody advances to the collider contact instead of
            # snapping back to the start of the requested move.  Preserve a
            # tiny separation so a following move away is not treated as
            # starting inside the expanded collider.
            travel = max(0.0, first_contact * MOVE_DISTANCE - 0.001)
            self.position[0] = start[0] + dx * travel
            self.position[2] = start[1] + dz * travel
            self.events.append("Movement blocked.")
            return

        floor = self._floor_at(end[0], end[1])
        if (
            self._defect == "floor-gap"
            and floor is not None
            and floor["id"] == self._floor_gap_id
        ):
            if self.seed in {1, 5}:
                self.position = [end[0], 0.10, end[1]]
                self.grounded = False
                self._floor_gap_wedge_attempts = 0
                return
            self._initialize_progress()
            self._respawn()
            self.events.append("You fell.")
            return

        self.position[0] = end[0]
        self.position[2] = end[1]
        self.events.append(f"Moved {action}.")
        if self.key_present and math.hypot(
            self.position[0] - KEY_POSITION[0],
            self.position[2] - KEY_POSITION[2],
        ) <= 1.0:
            self.key_present = False
            self.has_key = True
            self.events.append("Picked up the key.")
        if math.hypot(
            self.position[0] - EXIT_POSITION[0],
            self.position[2] - EXIT_POSITION[2],
        ) <= 1.0:
            self.won = True
            self.events.append("Reached the exit!")

    def _interact(self) -> None:
        distance = math.sqrt(sum(
            (self.position[index] - DOOR_POSITION[index]) ** 2
            for index in range(3)
        ))
        if self.door_open:
            self.events.append("The door is already open.")
        elif distance > 1.5:
            self.events.append("There is nothing to interact with here.")
        elif not self.has_key:
            self.events.append("The door is locked; you need the key.")
        else:
            self.door_open = True
            if self._defect != "key-not-consumed":
                self.has_key = False
            self.events.append("Unlocked the door and used the key.")

    def act(self, action: str) -> dict:
        if action not in CONTROLS or self.won:
            raise ValueError("Control is not currently legal.")
        self.tick += 1
        self.events = []
        if action in _DIRECTIONS:
            self._move(action)
        elif action == "interact":
            self._interact()
        elif action == "jump":
            self.events.append("Jumped.")
        return self.observe()

    @staticmethod
    def evaluate_transition(before: dict, _action: str, after: dict) -> dict:
        old = before.get("player", {}).get("position")
        new = after.get("player", {}).get("position")
        return {
            "moved": old != new,
            "events": deepcopy(after.get("events", [])),
            "contract_violations": [],
        }

    def close(self) -> None:
        return None


class FakeContinuousProtocolSession:
    def __init__(self, *, seed: int = 0, defect: str = "none"):
        self.game = FakeContinuousGame(seed, defect)
        self.initialized = False
        self.ids: set[str] = set()
        self.closed = False

    def handle(self, message: dict) -> tuple[dict, bool]:
        request_id, request_type = validate_request_envelope(message)
        if request_id in self.ids:
            raise ProtocolError("duplicate_id", "Request id was already used.")
        self.ids.add(request_id)
        if not self.initialized and request_type != "hello":
            raise ProtocolError("hello_required", "hello must be first.")
        if request_type == "hello":
            if self.initialized:
                raise ProtocolError("already_initialized", "hello may be sent once.")
            if message.get("protocol") != PROTOCOL_VERSION:
                raise ProtocolError("unsupported_protocol", "Unsupported protocol.")
            self.initialized = True
            return {
                "id": request_id,
                "type": "hello",
                "ok": True,
                "protocol": PROTOCOL_VERSION,
                "game": {"id": self.game.game_id, "version": self.game.version},
                "controls": list(CONTROLS),
                "capabilities": dict(CAPABILITIES),
            }, False
        if request_type == "observe":
            observation = self.game.wire_observation()
        elif request_type == "act":
            control = message.get("control")
            if not isinstance(control, str):
                raise ProtocolError("invalid_control", "act.control must be a string.")
            if control not in self.game.wire_observation()["controls"]:
                raise ProtocolError("illegal_control", "Control is not currently legal.")
            self.game.act(control)
            observation = self.game.wire_observation()
        elif request_type == "reset":
            seed = message.get("seed")
            if type(seed) is not int:
                raise ProtocolError("invalid_seed", "reset.seed must be an integer.")
            self.game.reset(seed)
            observation = self.game.wire_observation()
        elif request_type == "close":
            self.closed = True
            return {"id": request_id, "type": "close", "ok": True}, True
        else:
            raise ProtocolError("unknown_type", "Unknown request type.")
        return {
            "id": request_id,
            "type": request_type,
            "ok": True,
            "observation": observation,
        }, False


def serve_fake_continuous_stream(
    reader: BinaryIO,
    writer: BinaryIO,
    *,
    seed: int = 0,
    defect: str = "none",
) -> None:
    session = FakeContinuousProtocolSession(seed=seed, defect=defect)
    while not session.closed:
        request_id = None
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


class ContinuousStreamServer:
    """Create a fresh fake server session for each patched TCP connection."""

    def __init__(self, *, seed: int = 0, defect: str = "none"):
        self.seed = seed
        self.defect = defect
        self.threads: list[threading.Thread] = []

    def connect(self, _address, timeout=None):
        client, server = socket.socketpair()
        if timeout is not None:
            client.settimeout(timeout)

        def target() -> None:
            try:
                with server:
                    reader = server.makefile("rb")
                    writer = server.makefile("wb")
                    try:
                        serve_fake_continuous_stream(
                            reader, writer, seed=self.seed, defect=self.defect
                        )
                    finally:
                        reader.close()
                        writer.close()
            except OSError:
                pass

        thread = threading.Thread(target=target, daemon=True)
        thread.start()
        self.threads.append(thread)
        return client

    def join(self) -> None:
        for thread in self.threads:
            thread.join(timeout=2)
            if thread.is_alive():
                raise AssertionError("Fake continuous protocol thread did not finish.")
