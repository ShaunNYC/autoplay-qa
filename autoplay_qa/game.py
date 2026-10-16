"""An instrumented, turn-based game. No agent policy or solution path lives here."""

import hashlib
import random
from typing import Protocol


ACTIONS = ("north", "south", "east", "west", "interact", "wait")
DIRECTIONS = {"north": (0, -1), "south": (0, 1), "east": (1, 0), "west": (-1, 0)}
LAYOUT = (
    "#########",
    "#...#...#",
    "#.#.#.#.#",
    "#...D..E#",
    "#.#.#.#.#",
    "#...#...#",
    "#########",
)

DEFECTS = {
    "gate-keeps-key": "Unlocking the gate does not consume the carried key.",
    "wall-clip": "One deterministic interior wall tile permits movement into it.",
    "reset-leak": "Resetting after pickup preserves the key's world and inventory state.",
    "exit-no-win": "The first entry onto the exit does not end the game.",
    "phantom-block": "One deterministic floor tile blocks movement into it.",
}
DEFECT_CHOICES = ("none", *DEFECTS)

KEY_GATE_RULES = (
    "Each movement action attempts one orthogonal tile; walls and locked gates block movement.",
    "Walking onto the key collects it automatically.",
    "interact opens the gate only when one orthogonal tile away and holding the key; it consumes the key.",
    "Opening the gate does not move you. Walking onto the exit wins.",
    "wait advances one tick without movement. Blocked attempts also advance one tick.",
)

_INTERIOR_WALLS = tuple(
    (x, y)
    for y, row in enumerate(LAYOUT[1:-1], start=1)
    for x, glyph in enumerate(row[1:-1], start=1)
    if glyph == "#"
)
_PHANTOM_FLOORS = tuple(
    (x, y)
    for y, row in enumerate(LAYOUT)
    for x, glyph in enumerate(row)
    if glyph == "." and (x, y) not in {(1, 1), (7, 3), (3, 1), (1, 3), (3, 5)}
)


def validate_defect(defect: str | None) -> str:
    """Return the canonical defect name without exposing it through observations."""
    name = "none" if defect is None else defect
    if name not in DEFECT_CHOICES:
        choices = ", ".join(DEFECT_CHOICES)
        raise ValueError(f"Unknown injected defect {name!r}; choose one of: {choices}.")
    return name


def defect_evaluation_metadata(defect: str | None) -> dict:
    """Build the session-only ground truth block used by offline evaluation."""
    name = validate_defect(defect)
    injected = None if name == "none" else {"name": name, "description": DEFECTS[name]}
    return {"evaluation": {"injected_defect": injected}}


def _seeded_position(seed: int, namespace: str, positions: tuple[tuple[int, int], ...]) -> tuple[int, int]:
    digest = hashlib.sha256(f"{namespace}:{seed}".encode("utf-8")).digest()
    return positions[int.from_bytes(digest[:8], "big") % len(positions)]


class GameAdapter(Protocol):
    """Future engines implement observations and acknowledged actions at this boundary."""

    def observe(self) -> dict: ...
    def act(self, action: str) -> dict: ...


class KeyGateGame:
    version = "key-gate-v1"
    door = (4, 3)
    exit = (7, 3)
    key_locations = ((3, 1), (1, 3), (3, 5))

    def __init__(self, seed: int = 0, defect: str | None = None):
        self.seed = seed
        self.defect = validate_defect(defect)
        self._wall_clip_position = _seeded_position(seed, "wall-clip", _INTERIOR_WALLS)
        self._phantom_block_position = _seeded_position(seed, "phantom-block", _PHANTOM_FLOORS)
        self._initialize_state()

    def _initialize_state(self) -> None:
        self.position = (1, 1)
        self.key_position = random.Random(self.seed).choice(self.key_locations)
        self.has_key = False
        self.door_open = False
        self.won = False
        self.tick = 0
        self._exit_entries = 0
        self.events = ["Game started. Find the key, unlock the gate, and reach the exit."]

    def reset(self) -> "KeyGateGame":
        """Start a new episode, retaining only state selected by reset-leak."""
        leaked_key_position = self.key_position
        leaked_has_key = self.has_key
        self._initialize_state()
        if self.defect == "reset-leak":
            self.key_position = leaked_key_position
            self.has_key = leaked_has_key
        return self

    def _adjacent_to_door(self) -> bool:
        return sum(abs(a - b) for a, b in zip(self.position, self.door)) == 1

    def _walkable(self, position: tuple[int, int]) -> bool:
        x, y = position
        if self.defect == "phantom-block" and position == self._phantom_block_position:
            return False
        return (
            0 <= y < len(LAYOUT) and 0 <= x < len(LAYOUT[0])
            and (LAYOUT[y][x] != "#" or (
                self.defect == "wall-clip" and position == self._wall_clip_position
            ))
            and (position != self.door or self.door_open)
        )

    def evaluate_transition(self, before: dict, action: str, after: dict) -> dict:
        """Evaluate a transition while allowing the selected injected behavior."""
        walkable_wall = self._wall_clip_position if self.defect == "wall-clip" else None
        return evaluate_transition(before, action, after, _walkable_wall=walkable_wall)

    def observe(self) -> dict:
        rows = [list(row) for row in LAYOUT]
        rows[self.door[1]][self.door[0]] = "/" if self.door_open else "D"
        if self.key_position is not None:
            x, y = self.key_position
            rows[y][x] = "K"
        x, y = self.position
        rows[y][x] = "@"
        return {
            "game": self.version,
            "tick": self.tick,
            "status": "won" if self.won else "playing",
            "goal": "Collect the key, unlock the gate with interact, and walk to the exit.",
            "map": ["".join(row) for row in rows],
            "coordinates": "Zero-based [x,y]; x increases east, y increases south.",
            "legend": {"#": "wall", ".": "floor", "@": "player", "K": "key", "D": "locked gate", "/": "open gate", "E": "exit"},
            "player": {"position": list(self.position), "inventory": ["key"] if self.has_key else []},
            "objects": {
                "key": list(self.key_position) if self.key_position is not None else None,
                "gate": {"position": list(self.door), "open": self.door_open},
                "exit": list(self.exit),
            },
            "rules": list(KEY_GATE_RULES),
            "actions": [] if self.won else list(ACTIONS),
            "events": list(self.events),
        }

    def act(self, action: str) -> dict:
        if action not in ACTIONS:
            raise ValueError(f"Unknown game action: {action!r}")
        if self.won:
            raise ValueError("The game is already won; no further actions are accepted.")
        self.tick += 1
        self.events = []
        if action in DIRECTIONS:
            dx, dy = DIRECTIONS[action]
            target = (self.position[0] + dx, self.position[1] + dy)
            if not self._walkable(target):
                self.events.append("Movement blocked by a wall or locked gate.")
            else:
                self.position = target
                self.events.append(f"Moved {action} to {list(target)}.")
                if target == self.key_position:
                    self.key_position = None
                    self.has_key = True
                    self.events.append("Collected the key.")
                if target == self.exit:
                    self._exit_entries += 1
                    if self.defect != "exit-no-win" or self._exit_entries >= 2:
                        self.won = True
                        self.events.append("Reached the exit. You won!")
        elif action == "interact":
            if self.door_open:
                self.events.append("The gate is already open.")
            elif not self._adjacent_to_door():
                self.events.append("There is no gate within reach.")
            elif not self.has_key:
                self.events.append("The gate is locked; you need the key.")
            else:
                self.door_open = True
                if self.defect != "gate-keeps-key":
                    self.has_key = False
                self.events.append("Unlocked the gate and used the key.")
        else:
            self.events.append("Waited one tick.")
        return self.observe()


def evaluate_transition(
    before: dict,
    action: str,
    after: dict,
    *,
    _walkable_wall: tuple[int, int] | None = None,
) -> dict:
    """Check developer-supplied contracts; blocked moves alone are not defects."""
    violations = []
    old_position = before["player"]["position"]
    new_position = after["player"]["position"]
    distance = sum(abs(a - b) for a, b in zip(old_position, new_position))
    if after["tick"] != before["tick"] + 1:
        violations.append("An accepted action must advance exactly one tick.")
    if action in DIRECTIONS:
        dx, dy = DIRECTIONS[action]
        target = [old_position[0] + dx, old_position[1] + dy]
        if new_position not in (old_position, target):
            violations.append("A movement action must stay put or move one tile in the requested direction.")
    elif distance:
        violations.append("A non-movement action moved the player.")
    x, y = new_position
    rows = after["map"]
    if (
        not (0 <= y < len(rows) and 0 <= x < len(rows[0]))
        or (LAYOUT[y][x] == "#" and (x, y) != _walkable_wall)
    ):
        violations.append("The player is outside the walkable map.")
    old_gate = before["objects"]["gate"]
    gate = after["objects"]["gate"]
    if not old_gate["open"] and gate["open"]:
        adjacent = sum(abs(a - b) for a, b in zip(old_position, old_gate["position"])) == 1
        if action != "interact" or "key" not in before["player"]["inventory"] or not adjacent:
            violations.append("The gate opened without an adjacent, keyed interaction.")
    if new_position == gate["position"] and not gate["open"]:
        violations.append("The player entered a locked gate.")
    if after["status"] == "won" and new_position != after["objects"]["exit"]:
        violations.append("The game reports victory away from the exit.")
    return {"moved": bool(distance), "events": after["events"], "contract_violations": violations}


def render(observation: dict) -> str:
    player = observation.get("player")
    player = player if isinstance(player, dict) else {}
    items = player.get("inventory", [])
    items = items if isinstance(items, list) else []
    inventory = ", ".join(str(item) for item in items) or "empty"
    rows = observation.get("map", observation.get("grid", []))
    rows = rows if isinstance(rows, list) else []
    prefix = "\n".join(str(row) for row in rows)
    details = (
        f"Tick {observation.get('tick', '?')} | {observation.get('status', 'unknown')} "
        f"| inventory: {inventory}"
    )
    events = observation.get("events", [])
    events = events if isinstance(events, list) else []
    body = details + "\n" + " ".join(str(event) for event in events)
    return prefix + ("\n" if prefix else "") + body
