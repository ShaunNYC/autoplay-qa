"""Inference-free collision coverage used by hybrid QA exploration.

The sweep consumes only the adapter's public observation, advertised controls,
and optional grid/legend capability.  It never consults developer rules,
injected-defect metadata, or a game-specific route.  Records use a separate
``sweep_step`` type so the normal LLM action numbering and replay contract can
remain unchanged.
"""

from __future__ import annotations

from collections import deque
from copy import deepcopy
from dataclasses import dataclass
import math
from statistics import median
from typing import Callable, Mapping, Sequence


SWEEP_SUMMARY_VERSION = 1
SWEEP_ANOMALY_LIMIT = 8

_DIRECTION_ALIASES = (
    ((0, -1), ("north", "up", "n")),
    ((0, 1), ("south", "down", "s")),
    ((1, 0), ("east", "right", "e")),
    ((-1, 0), ("west", "left", "w")),
)
_REVERSE_DELTA = {
    (0, -1): (0, 1),
    (0, 1): (0, -1),
    (1, 0): (-1, 0),
    (-1, 0): (1, 0),
}
_TERMINAL_STATUSES = {
    "won", "lost", "finished", "ended", "closed", "complete", "completed",
    "game_over", "game-over",
}
_CONTINUOUS_FLOOR_KINDS = ("floor", "ground", "platform", "tile")
_CONTINUOUS_DIRECTION_FALLBACKS = {
    (0, -1): (0.0, 1.0),
    (0, 1): (0.0, -1.0),
    (1, 0): (1.0, 0.0),
    (-1, 0): (-1.0, 0.0),
}
_CONTINUOUS_CELL_LIMIT = 96
_CONTINUOUS_POSITION_TOLERANCE = 0.12
_CONTINUOUS_DETECTOR_HISTORY = 32


@dataclass
class SweepResult:
    """A bounded sweep's phase-local transitions and prompt-safe summary."""

    records: list[dict]
    summary: dict


def _capability_value(capabilities: object, name: str) -> bool | None:
    if capabilities is None:
        return None
    if isinstance(capabilities, Mapping):
        if name not in capabilities:
            return None
        value = capabilities[name]
        if isinstance(value, Mapping):
            value = value.get("supported", value.get("enabled", True))
        return bool(value)
    if isinstance(capabilities, (list, tuple, set, frozenset)):
        return name in capabilities
    return None


def _adapter_capabilities(game: object) -> object:
    value = getattr(game, "capabilities", None)
    if callable(value):
        value = value()
    if value is not None:
        return value
    hello = getattr(game, "hello", None)
    if isinstance(hello, Mapping):
        return hello.get("capabilities")
    return None


def _control_name(value: object) -> str | None:
    if isinstance(value, str) and value:
        return value
    if isinstance(value, Mapping):
        for key in ("id", "name", "control"):
            candidate = value.get(key)
            if isinstance(candidate, str) and candidate:
                return candidate
    return None


def _adapter_controls(game: object, observation: dict) -> list[str]:
    candidates: object = observation.get("actions")
    if not isinstance(candidates, (list, tuple)):
        candidates = getattr(game, "controls", None)
        if callable(candidates):
            candidates = candidates()
    if not isinstance(candidates, (list, tuple)):
        hello = getattr(game, "hello", None)
        candidates = hello.get("controls") if isinstance(hello, Mapping) else None
    if not isinstance(candidates, (list, tuple)):
        return []
    controls: list[str] = []
    for candidate in candidates:
        name = _control_name(candidate)
        if name is not None and name not in controls:
            controls.append(name)
    return controls


def _finite_number(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _continuous_vector(value: object) -> tuple[float, ...] | None:
    if not isinstance(value, (list, tuple)) or len(value) not in (2, 3):
        return None
    coordinates = tuple(_finite_number(item) for item in value)
    if any(item is None for item in coordinates):
        return None
    return tuple(item for item in coordinates if item is not None)


def _continuous_player_position(observation: dict) -> tuple[float, ...] | None:
    player = observation.get("player")
    return (
        _continuous_vector(player.get("position"))
        if isinstance(player, Mapping)
        else None
    )


def _horizontal_point(position: tuple[float, ...]) -> tuple[float, float]:
    return (position[0], position[2]) if len(position) == 3 else (position[0], position[1])


def _vertical_coordinate(position: tuple[float, ...]) -> float | None:
    return position[1] if len(position) == 3 else None


def _rounded_point(point: tuple[float, float]) -> list[float]:
    return [round(point[0], 2), round(point[1], 2)]


@dataclass(frozen=True)
class _FloorCell:
    identifier: str
    center: tuple[float, float]
    half_size: tuple[float, float]

    def contains(self, point: tuple[float, float], tolerance: float = 0.0) -> bool:
        return (
            abs(point[0] - self.center[0]) <= self.half_size[0] + tolerance
            and abs(point[1] - self.center[1]) <= self.half_size[1] + tolerance
        )


def _visible_floor_cells(observation: dict) -> list[_FloorCell]:
    entities = observation.get("entities")
    if not isinstance(entities, list):
        return []
    cells: list[_FloorCell] = []
    for index, entity in enumerate(entities):
        if not isinstance(entity, Mapping):
            continue
        kind = entity.get("kind")
        state = entity.get("state")
        if (
            not isinstance(kind, str)
            or not any(token in kind.lower() for token in _CONTINUOUS_FLOOR_KINDS)
            or not isinstance(state, Mapping)
            or state.get("solid") is not True
        ):
            continue
        position = _continuous_vector(entity.get("position"))
        size = _continuous_vector(entity.get("size"))
        if position is None or size is None or len(position) != len(size):
            continue
        horizontal_position = _horizontal_point(position)
        horizontal_size = _horizontal_point(size)
        if horizontal_size[0] <= 0.0 or horizontal_size[1] <= 0.0:
            continue
        identifier = entity.get("id")
        cells.append(_FloorCell(
            identifier if isinstance(identifier, str) and identifier else f"floor-{index}",
            horizontal_position,
            (horizontal_size[0] / 2.0, horizontal_size[1] / 2.0),
        ))
    return sorted(cells, key=lambda cell: (cell.center[1], cell.center[0], cell.identifier))


def _cell_for_point(
    cells: Sequence[_FloorCell],
    point: tuple[float, float],
) -> _FloorCell | None:
    containing = [
        cell for cell in cells
        if cell.contains(point, _CONTINUOUS_POSITION_TOLERANCE)
    ]
    if not containing:
        return None
    return min(containing, key=lambda cell: (
        (cell.center[0] - point[0]) ** 2 + (cell.center[1] - point[1]) ** 2,
        cell.identifier,
    ))


def _direction_controls(controls: Sequence[str]) -> dict[tuple[int, int], str]:
    by_lower = {control.lower(): control for control in controls}
    found: dict[tuple[int, int], str] = {}
    for delta, aliases in _DIRECTION_ALIASES:
        for alias in aliases:
            if alias in by_lower:
                found[delta] = by_lower[alias]
                break
    return found


def _grid_rows(observation: dict) -> tuple[tuple[str, ...], ...] | None:
    raw = observation.get("grid", observation.get("map"))
    if not isinstance(raw, list) or not raw:
        return None
    rows: list[tuple[str, ...]] = []
    for row in raw:
        if isinstance(row, str):
            cells = tuple(row)
        elif isinstance(row, list) and all(isinstance(cell, str) for cell in row):
            cells = tuple(row)
        else:
            return None
        if not cells:
            return None
        rows.append(cells)
    return tuple(rows)


def _position(value: object) -> tuple[int, int] | None:
    if (
        isinstance(value, (list, tuple))
        and len(value) == 2
        and all(type(coordinate) is int for coordinate in value)
    ):
        return value[0], value[1]
    if isinstance(value, Mapping):
        x, y = value.get("x"), value.get("y")
        if type(x) is int and type(y) is int:
            return x, y
    return None


def _player_position(observation: dict) -> tuple[int, int] | None:
    player = observation.get("player")
    return _position(player.get("position")) if isinstance(player, Mapping) else None


def _legend_meaning(legend: Mapping, glyph: str | None) -> str:
    if glyph is None:
        return ""
    value = legend.get(glyph)
    if isinstance(value, str):
        return value.lower()
    if isinstance(value, Mapping):
        text = " ".join(
            str(value[key])
            for key in ("name", "kind", "description")
            if isinstance(value.get(key), str)
        )
        return text.lower()
    return ""


def _event_text(observation: dict) -> str:
    events = observation.get("events")
    if not isinstance(events, list):
        return ""
    parts: list[str] = []
    for event in events:
        if isinstance(event, str):
            parts.append(event)
        elif isinstance(event, Mapping):
            for key in ("text", "message", "feedback"):
                value = event.get(key)
                if isinstance(value, str):
                    parts.append(value)
                    break
    return " ".join(parts).lower()


def _cell(rows: tuple[tuple[str, ...], ...], position: tuple[int, int]) -> str | None:
    x, y = position
    if not 0 <= y < len(rows) or not 0 <= x < len(rows[y]):
        return None
    return rows[y][x]


def _is_wall(legend: Mapping, glyph: str | None) -> bool:
    meaning = _legend_meaning(legend, glyph)
    return glyph == "#" or "wall" in meaning


def _is_floor(legend: Mapping, glyph: str | None) -> bool:
    meaning = _legend_meaning(legend, glyph)
    return glyph == "." or any(word in meaning for word in ("floor", "ground", "walkable"))


def _is_exit(legend: Mapping, glyph: str | None) -> bool:
    meaning = _legend_meaning(legend, glyph)
    return glyph in {"E", "X"} or any(word in meaning for word in ("exit", "goal"))


def _safe_anomaly(value: object) -> dict | None:
    if not isinstance(value, Mapping):
        return None
    allowed = (
        "type", "category", "aspect", "source", "phase", "phase_step",
        "action", "target", "observed_position", "title", "entity_id",
        "evidence_steps",
    )
    return {key: deepcopy(value[key]) for key in allowed if key in value}


def compact_sweep_summary(summary: object) -> dict:
    """Whitelist compact coverage facts before they enter an LLM request."""
    if not isinstance(summary, Mapping):
        raise ValueError("Sweep summary must be an object.")
    result = {
        key: deepcopy(summary[key])
        for key in (
            "version", "source", "status", "actions", "action_budget",
            "movement_attempts", "blocked_attempts", "successful_moves",
            "visited_positions", "visible_non_wall_positions",
            "unvisited_visible_positions", "labeled_wall_targets",
            "labeled_wall_probes", "labeled_floor_probes", "anomaly_count",
            "continuous", "move_distance", "visible_floor_cells",
            "visited_tile_cells", "tried_directions", "interaction_attempts",
            "resets",
        )
        if key in summary
    }
    anomalies = summary.get("anomalies")
    if isinstance(anomalies, list):
        result["anomalies"] = [
            safe for value in anomalies[:SWEEP_ANOMALY_LIMIT]
            if (safe := _safe_anomaly(value)) is not None
        ]
    else:
        result["anomalies"] = []
    if result.get("continuous") is True:
        tried = result.get("tried_directions")
        if isinstance(tried, list):
            result["tried_direction_fields"] = ["cell", "controls"]
            result["tried_directions"] = [
                [deepcopy(item.get("cell")), deepcopy(item.get("controls", []))]
                for item in tried[:_CONTINUOUS_CELL_LIMIT]
                if isinstance(item, Mapping)
            ]
    return result


def _skipped_result(status: str, max_actions: int) -> SweepResult:
    return SweepResult([], {
        "version": SWEEP_SUMMARY_VERSION,
        "source": "sweep",
        "status": status,
        "actions": 0,
        "action_budget": max_actions,
        "movement_attempts": 0,
        "blocked_attempts": 0,
        "successful_moves": 0,
        "visited_positions": 0,
        "visible_non_wall_positions": 0,
        "unvisited_visible_positions": 0,
        "labeled_wall_targets": 0,
        "labeled_wall_probes": 0,
        "labeled_floor_probes": 0,
        "anomaly_count": 0,
        "anomalies": [],
    })


class _SweepRunner:
    def __init__(
        self,
        game: object,
        observation: dict,
        rows: tuple[tuple[str, ...], ...],
        legend: Mapping,
        directions: dict[tuple[int, int], str],
        max_actions: int,
        on_record: Callable[[dict], None] | None,
    ):
        self.game = game
        self.observation = deepcopy(observation)
        self.rows = rows
        self.legend = legend
        self.directions = directions
        self.max_actions = max_actions
        self.on_record = on_record
        self.records: list[dict] = []
        self.anomalies: list[dict] = []
        self.blocked_edges: set[tuple[tuple[int, int], tuple[int, int]]] = set()
        start = _player_position(observation)
        self.visited: set[tuple[int, int]] = {start} if start is not None else set()
        self.successful_moves = 0
        self.blocked_attempts = 0
        self.floor_probes: set[tuple[int, int]] = set()
        self.wall_probes: set[tuple[int, int]] = set()
        self.status = "completed"
        self.error: str | None = None

        self.positions = {
            (x, y)
            for y, row in enumerate(rows)
            for x in range(len(row))
        }
        self.walls = {
            position for position in self.positions
            if _is_wall(legend, _cell(rows, position))
        }
        self.non_walls = self.positions - self.walls
        self.exits = {
            position for position in self.non_walls
            if _is_exit(legend, _cell(rows, position))
        }

    @property
    def action_count(self) -> int:
        return len(self.records)

    def _terminal(self, observation: dict | None = None) -> bool:
        current = self.observation if observation is None else observation
        status = current.get("status")
        return isinstance(status, str) and status.lower() in _TERMINAL_STATUSES

    def _path_toward(self, target: tuple[int, int]) -> list[str] | None:
        start = _player_position(self.observation)
        if start is None:
            return None
        if start == target:
            return []
        queue = deque([start])
        previous: dict[tuple[int, int], tuple[tuple[int, int], str]] = {}
        seen = {start}
        for_current = tuple(self.directions.items())
        while queue:
            current = queue.popleft()
            for delta, action in for_current:
                neighbor = current[0] + delta[0], current[1] + delta[1]
                if (
                    neighbor in seen
                    or neighbor not in self.non_walls
                    or neighbor in self.exits
                    or (current, neighbor) in self.blocked_edges
                ):
                    continue
                previous[neighbor] = current, action
                if neighbor == target:
                    path: list[str] = []
                    cursor = neighbor
                    while cursor != start:
                        parent, path_action = previous[cursor]
                        path.append(path_action)
                        cursor = parent
                    path.reverse()
                    return path
                seen.add(neighbor)
                queue.append(neighbor)
        return None

    def _anomaly(
        self,
        *,
        phase_step: int,
        action: str,
        target: tuple[int, int],
        observed: tuple[int, int] | None,
        aspect: str,
    ) -> dict:
        return {
            "type": "map_object_inconsistency",
            "category": "label_contradiction",
            "aspect": aspect,
            "source": "sweep",
            "phase": "sweep",
            "phase_step": phase_step,
            "action": action,
            "target": list(target),
            "observed_position": list(observed) if observed is not None else None,
        }

    def perform(self, action: str, *, purpose: str) -> dict | None:
        if self.action_count >= self.max_actions:
            self.status = "action_limit"
            return None
        before = deepcopy(self.observation)
        before_position = _player_position(before)
        delta = next((delta for delta, value in self.directions.items() if value == action), None)
        if before_position is None or delta is None:
            self.status = "invalid_observation"
            return None
        target = before_position[0] + delta[0], before_position[1] + delta[1]
        try:
            returned = self.game.act(action)
            after = deepcopy(returned if isinstance(returned, dict) else self.game.observe())
        except (KeyError, OSError, RuntimeError, TypeError, ValueError) as exc:
            self.status = "environment_error"
            self.error = f"{type(exc).__name__}: {exc}"
            return None

        phase_step = self.action_count + 1
        after_position = _player_position(after)
        glyph = _cell(self.rows, target)
        is_wall = _is_wall(self.legend, glyph)
        is_floor = _is_floor(self.legend, glyph)
        if is_wall:
            self.wall_probes.add(target)
        if is_floor:
            self.floor_probes.add(target)

        anomalies: list[dict] = []
        if after_position == target and target != before_position:
            self.successful_moves += 1
            self.visited.add(target)
            if is_wall:
                anomalies.append(self._anomaly(
                    phase_step=phase_step,
                    action=action,
                    target=target,
                    observed=after_position,
                    aspect="entered_wall_labeled_tile",
                ))
        elif after_position == before_position:
            self.blocked_attempts += 1
            self.blocked_edges.add((before_position, target))
            if is_floor and "blocked" in _event_text(after):
                anomalies.append(self._anomaly(
                    phase_step=phase_step,
                    action=action,
                    target=target,
                    observed=after_position,
                    aspect="blocked_on_floor_labeled_tile",
                ))
        elif after_position is not None:
            self.visited.add(after_position)

        self.anomalies.extend(anomalies)
        record = {
            "type": "sweep_step",
            "source": "sweep",
            "phase": "sweep",
            "phase_step": phase_step,
            "step": phase_step,
            "before": before,
            "decision": {
                "action": action,
                "reason": f"Scripted visible-grid {purpose}.",
                "expected_position": None,
                "notes": [],
            },
            "after": after,
            "qa_mode": True,
            "software_fact": {
                "source": "sweep",
                "anomalies": deepcopy(anomalies),
            },
        }
        self.records.append(record)
        self.observation = after
        if self.on_record is not None:
            self.on_record(deepcopy(record))
        if self._terminal(after):
            self.status = "terminal"
        return record

    def navigate(self, target: tuple[int, int]) -> bool:
        while self.status == "completed":
            current = _player_position(self.observation)
            if current == target:
                return True
            path = self._path_toward(target)
            if not path:
                return False
            before = current
            record = self.perform(path[0], purpose="coverage navigation")
            if record is None:
                return False
            after = _player_position(self.observation)
            if after == before and self.action_count >= self.max_actions:
                self.status = "action_limit"
                return False
        return False

    def visit_visible_cells(self) -> None:
        candidates = self.non_walls - self.exits
        while self.status == "completed":
            remaining = candidates - self.visited
            routes = []
            for target in remaining:
                path = self._path_toward(target)
                if path is not None:
                    routes.append((len(path), target[1], target[0], target))
            if not routes:
                return
            target = min(routes)[-1]
            before_count = self.action_count
            self.navigate(target)
            if self.action_count == before_count and target not in self.visited:
                return

    def probe_visible_walls(self) -> None:
        pending = set(self.walls)
        while pending and self.status == "completed":
            routes = []
            for wall in pending:
                for delta, action in self.directions.items():
                    source = wall[0] - delta[0], wall[1] - delta[1]
                    if source not in self.non_walls or source in self.exits:
                        continue
                    path = self._path_toward(source)
                    if path is not None:
                        routes.append((
                            len(path), wall[1], wall[0], source[1], source[0],
                            wall, source, action, delta,
                        ))
            if not routes:
                return
            _, _, _, _, _, wall, source, action, delta = min(routes)
            if not self.navigate(source):
                pending.discard(wall)
                continue
            record = self.perform(action, purpose="labeled-wall collision probe")
            pending.discard(wall)
            if record is None:
                return
            if _player_position(self.observation) == wall and self.status == "completed":
                reverse = self.directions.get(_REVERSE_DELTA[delta])
                if reverse is not None:
                    self.perform(reverse, purpose="collision-probe recovery")

    def run(self) -> SweepResult:
        if self._terminal():
            self.status = "terminal"
        else:
            self.visit_visible_cells()
            self.probe_visible_walls()
        if self.status == "completed" and self.action_count >= self.max_actions:
            self.status = "action_limit"

        candidates = self.non_walls - self.exits
        summary = {
            "version": SWEEP_SUMMARY_VERSION,
            "source": "sweep",
            "status": self.status,
            "actions": self.action_count,
            "action_budget": self.max_actions,
            "movement_attempts": self.action_count,
            "blocked_attempts": self.blocked_attempts,
            "successful_moves": self.successful_moves,
            "visited_positions": len(self.visited),
            "visible_non_wall_positions": len(candidates),
            "unvisited_visible_positions": len(candidates - self.visited),
            "labeled_wall_targets": len(self.walls),
            "labeled_wall_probes": len(self.wall_probes),
            "labeled_floor_probes": len(self.floor_probes),
            "anomaly_count": len(self.anomalies),
            "anomalies": [
                deepcopy(value) for value in self.anomalies[-SWEEP_ANOMALY_LIMIT:]
            ],
        }
        if self.error is not None:
            summary["error"] = self.error
        return SweepResult(self.records, summary)


class _ContinuousSweepRunner:
    """Sample visible continuous floor cells without engine-specific knowledge."""

    def __init__(
        self,
        game: object,
        observation: dict,
        cells: list[_FloorCell],
        directions: dict[tuple[int, int], str],
        controls: Sequence[str],
        max_actions: int,
        on_record: Callable[[dict], None] | None,
    ):
        self.game = game
        self.observation = deepcopy(observation)
        self.cells = cells
        self.directions = directions
        self.controls = tuple(controls)
        self.max_actions = max_actions
        self.on_record = on_record
        self.records: list[dict] = []
        self.anomalies: list[dict] = []
        self.status = "completed"
        self.error: str | None = None
        self.episode = 0
        self.reset_count = 0
        self.interaction_attempts = 0
        self.successful_moves = 0
        self.blocked_attempts = 0
        self.movement_attempts = 0
        self.move_samples: list[float] = []
        self.direction_samples: dict[str, list[tuple[float, float]]] = {
            action: [] for action in directions.values()
        }
        self.blocked_cell_edges: set[tuple[str, str]] = set()
        self.observed_cell_edges: set[frozenset[str]] = set()
        self.visited: set[str] = set()
        self.tried: dict[str, set[str]] = {}
        self.skipped_terminal_pairs: set[tuple[str, str]] = set()
        self.interaction_signatures: set[tuple[str, tuple[tuple[str, str], ...]]] = set()
        self.initial_observation = deepcopy(observation)
        self.restoration_probe_done = False
        self.terminal_cells = self._terminal_floor_cells(observation)
        self.wall_occupied_cells = self._wall_floor_cells(observation)
        start = self._current_cell()
        if start is not None:
            self.visited.add(start.identifier)

    @property
    def action_count(self) -> int:
        return len(self.records)

    @property
    def move_distance(self) -> float:
        return median(self.move_samples) if self.move_samples else 1.0

    def _terminal(self, observation: dict | None = None) -> bool:
        current = self.observation if observation is None else observation
        status = current.get("status")
        return isinstance(status, str) and status.lower() in _TERMINAL_STATUSES

    def _current_position(self) -> tuple[float, ...] | None:
        return _continuous_player_position(self.observation)

    def _current_cell(self) -> _FloorCell | None:
        position = self._current_position()
        return _cell_for_point(self.cells, _horizontal_point(position)) if position else None

    def _terminal_floor_cells(self, observation: dict) -> set[str]:
        entities = observation.get("entities")
        if not isinstance(entities, list):
            return set()
        result = set()
        for entity in entities:
            if not isinstance(entity, Mapping):
                continue
            kind = str(entity.get("kind", "")).lower()
            if not any(token in kind for token in ("exit", "goal", "finish")):
                continue
            position = _continuous_vector(entity.get("position"))
            if position is None:
                continue
            cell = _cell_for_point(self.cells, _horizontal_point(position))
            if cell is not None:
                result.add(cell.identifier)
        return result

    def _wall_floor_cells(self, observation: dict) -> set[str]:
        entities = observation.get("entities")
        if not isinstance(entities, list):
            return set()
        result = set()
        for entity in entities:
            if not isinstance(entity, Mapping) or "wall" not in str(
                entity.get("kind", "")
            ).lower():
                continue
            position = _continuous_vector(entity.get("position"))
            size = _continuous_vector(entity.get("size"))
            if position is None or size is None or len(position) != len(size):
                continue
            center = _horizontal_point(position)
            horizontal_size = _horizontal_point(size)
            for cell in self.cells:
                if (
                    abs(cell.center[0] - center[0]) <= horizontal_size[0] / 2.0
                    and abs(cell.center[1] - center[1]) <= horizontal_size[1] / 2.0
                ):
                    result.add(cell.identifier)
        return result

    def _available_controls(self) -> set[str]:
        raw = self.observation.get("actions", self.observation.get("controls"))
        if not isinstance(raw, (list, tuple)):
            return set(self.controls)
        return {value for value in raw if isinstance(value, str)}

    def _fell(self, before: dict, after: dict) -> bool:
        if any(token in _event_text(after) for token in ("fell", "fallen", "falling")):
            return True
        before_position = _continuous_player_position(before)
        after_position = _continuous_player_position(after)
        before_y = _vertical_coordinate(before_position) if before_position else None
        after_y = _vertical_coordinate(after_position) if after_position else None
        return (
            before_y is not None
            and after_y is not None
            and after_y < before_y - max(2.0, self.move_distance * 1.5)
        )

    @staticmethod
    def _has_anomaly(record: dict, *kinds: str) -> bool:
        wanted = set(kinds)
        fact = record.get("software_fact")
        anomalies = fact.get("anomalies") if isinstance(fact, Mapping) else None
        return isinstance(anomalies, list) and any(
            isinstance(anomaly, Mapping)
            and anomaly.get("type") in wanted
            for anomaly in anomalies
        )

    def _action_vector(self, action: str) -> tuple[float, float] | None:
        samples = self.direction_samples.get(action, [])
        if samples:
            x = median([sample[0] for sample in samples])
            z = median([sample[1] for sample in samples])
            length = math.hypot(x, z)
            if length > 0.0:
                return x / length, z / length
        delta = next(
            (delta for delta, candidate in self.directions.items() if candidate == action),
            None,
        )
        return _CONTINUOUS_DIRECTION_FALLBACKS.get(delta) if delta is not None else None

    def _action_toward(
        self,
        source: tuple[float, float],
        target: tuple[float, float],
    ) -> str | None:
        offset = target[0] - source[0], target[1] - source[1]
        if math.hypot(*offset) <= _CONTINUOUS_POSITION_TOLERANCE:
            return None
        candidates = []
        for order, action in enumerate(self.directions.values()):
            vector = self._action_vector(action)
            if vector is None:
                continue
            projection = offset[0] * vector[0] + offset[1] * vector[1]
            candidates.append((projection, -order, action))
        if not candidates:
            return None
        projection, _, action = max(candidates)
        return action if projection > 0.0 else None

    def _action_enters_terminal_cell(self, action: str) -> bool:
        position = self._current_position()
        vector = self._action_vector(action)
        if position is None or vector is None:
            return False
        point = _horizontal_point(position)
        projected = (
            point[0] + vector[0] * self.move_distance,
            point[1] + vector[1] * self.move_distance,
        )
        cell = _cell_for_point(self.cells, projected)
        return cell is not None and cell.identifier in self.terminal_cells

    def _projected_cell(self, action: str) -> _FloorCell | None:
        position = self._current_position()
        vector = self._action_vector(action)
        if position is None or vector is None:
            return None
        point = _horizontal_point(position)
        return _cell_for_point(self.cells, (
            point[0] + vector[0] * self.move_distance,
            point[1] + vector[1] * self.move_distance,
        ))

    def _block_cell_edge(self, left: _FloorCell | None, right: _FloorCell | None) -> None:
        if left is None or right is None or left.identifier == right.identifier:
            return
        self.blocked_cell_edges.add((left.identifier, right.identifier))
        self.blocked_cell_edges.add((right.identifier, left.identifier))

    @staticmethod
    def _segment_hits_box(
        start: tuple[float, float],
        end: tuple[float, float],
        center: tuple[float, float],
        half_size: tuple[float, float],
    ) -> bool:
        enter, leave = 0.0, 1.0
        for axis in range(2):
            lower = center[axis] - half_size[axis]
            upper = center[axis] + half_size[axis]
            delta = end[axis] - start[axis]
            if abs(delta) <= 1e-12:
                if not lower < start[axis] < upper:
                    return False
                continue
            near = (lower - start[axis]) / delta
            far = (upper - start[axis]) / delta
            if near > far:
                near, far = far, near
            enter = max(enter, near)
            leave = min(leave, far)
            if enter > leave:
                return False
        return leave > 1e-7 and enter < 1.0 - 1e-7

    def _visible_solid_between(self, left: _FloorCell, right: _FloorCell) -> bool:
        entities = self.observation.get("entities")
        if not isinstance(entities, list):
            return False
        for entity in entities:
            if not isinstance(entity, Mapping):
                continue
            state = entity.get("state")
            kind = str(entity.get("kind", "")).lower()
            if (
                not isinstance(state, Mapping)
                or state.get("solid") is not True
                or any(token in kind for token in _CONTINUOUS_FLOOR_KINDS)
            ):
                continue
            position = _continuous_vector(entity.get("position"))
            size = _continuous_vector(entity.get("size"))
            if position is None or size is None or len(position) != len(size):
                continue
            center = _horizontal_point(position)
            horizontal_size = _horizontal_point(size)
            half_size = (
                horizontal_size[0] / 2.0 + 0.45,
                horizontal_size[1] / 2.0 + 0.45,
            )
            if self._segment_hits_box(left.center, right.center, center, half_size):
                return True
        return False

    def _cell_graph(self, *, respect_solids: bool = False) -> dict[str, list[str]]:
        reach = max(0.5, self.move_distance)
        graph = {cell.identifier: [] for cell in self.cells}
        for index, left in enumerate(self.cells):
            for right in self.cells[index + 1:]:
                dx = abs(left.center[0] - right.center[0])
                dz = abs(left.center[1] - right.center[1])
                gap_x = max(0.0, dx - left.half_size[0] - right.half_size[0])
                gap_z = max(0.0, dz - left.half_size[1] - right.half_size[1])
                overlap_x = dx <= left.half_size[0] + right.half_size[0] + 0.05
                overlap_z = dz <= left.half_size[1] + right.half_size[1] + 0.05
                cardinal_neighbor = (
                    overlap_z and dx > max(0.05, dz * 1.5) and gap_x <= reach + 0.05
                ) or (
                    overlap_x and dz > max(0.05, dx * 1.5) and gap_z <= reach + 0.05
                )
                observed_edge = frozenset((left.identifier, right.identifier))
                if cardinal_neighbor and not (
                    respect_solids
                    and observed_edge not in self.observed_cell_edges
                    and self._visible_solid_between(left, right)
                ):
                    graph[left.identifier].append(right.identifier)
                    graph[right.identifier].append(left.identifier)
        by_id = {cell.identifier: cell for cell in self.cells}
        for identifier, neighbors in graph.items():
            neighbors.sort(key=lambda item: (
                by_id[item].center[1], by_id[item].center[0], item,
            ))
        return graph

    def _bfs_cells(self) -> list[_FloorCell]:
        start = self._current_cell()
        if start is None:
            return []
        by_id = {cell.identifier: cell for cell in self.cells}
        graph = self._cell_graph()
        pending = deque([start.identifier])
        seen = {start.identifier}
        ordered: list[_FloorCell] = []
        while pending:
            identifier = pending.popleft()
            if (
                identifier not in self.terminal_cells
                and identifier not in self.wall_occupied_cells
            ):
                ordered.append(by_id[identifier])
            for neighbor in graph.get(identifier, []):
                if (
                    neighbor not in seen
                    and neighbor not in self.terminal_cells
                    and neighbor not in self.wall_occupied_cells
                ):
                    seen.add(neighbor)
                    pending.append(neighbor)
        return ordered

    def _path_to(self, target: _FloorCell) -> list[_FloorCell] | None:
        start = self._current_cell()
        if start is None:
            return None
        if start.identifier == target.identifier:
            return []
        by_id = {cell.identifier: cell for cell in self.cells}
        # Use visible floor adjacency for routing and retire only edges that an
        # executed movement actually found blocked.  This keeps missing wall
        # colliders probeable instead of treating visible solidity as truth.
        graph = self._cell_graph()
        queue = deque([start.identifier])
        previous: dict[str, str] = {}
        seen = {start.identifier}
        while queue:
            current = queue.popleft()
            for neighbor in graph.get(current, []):
                if (
                    neighbor in seen
                    or neighbor in self.terminal_cells
                    or neighbor in self.wall_occupied_cells
                    or (current, neighbor) in self.blocked_cell_edges
                ):
                    continue
                previous[neighbor] = current
                if neighbor == target.identifier:
                    path = [neighbor]
                    while path[-1] != start.identifier:
                        path.append(previous[path[-1]])
                    path.reverse()
                    return [by_id[identifier] for identifier in path[1:]]
                seen.add(neighbor)
                queue.append(neighbor)
        return None

    def _record_anomalies(self, record: dict) -> list[dict]:
        from .continuous import continuous_anomalies

        directional = set(self.directions.values())
        calibration_records = [
            prior for prior in self.records
            if prior.get("type") == "sweep_step"
            and prior.get("decision", {}).get("action") in directional
        ][-_CONTINUOUS_DETECTOR_HISTORY:]
        anomalies = continuous_anomalies(
            record,
            calibration_records,
            tolerance=0.05,
        )
        return deepcopy(anomalies) if isinstance(anomalies, list) else []

    def perform(self, action: str, *, purpose: str) -> dict | None:
        if self.action_count >= self.max_actions:
            self.status = "action_limit"
            return None
        if action not in self._available_controls():
            return None
        before = deepcopy(self.observation)
        before_position = _continuous_player_position(before)
        before_cell = self._current_cell()
        try:
            returned = self.game.act(action)
            after = deepcopy(returned if isinstance(returned, dict) else self.game.observe())
        except (KeyError, OSError, RuntimeError, TypeError, ValueError) as exc:
            self.status = "environment_error"
            self.error = f"{type(exc).__name__}: {exc}"
            return None

        phase_step = self.action_count + 1
        after_position = _continuous_player_position(after)
        after_cell = (
            _cell_for_point(self.cells, _horizontal_point(after_position))
            if after_position is not None
            else None
        )
        if before_cell is not None:
            self.visited.add(before_cell.identifier)
        if after_cell is not None:
            self.visited.add(after_cell.identifier)

        if action in self.direction_samples:
            self.movement_attempts += 1
            if before_cell is not None:
                self.tried.setdefault(before_cell.identifier, set()).add(action)
            if before_position is not None and after_position is not None:
                old = _horizontal_point(before_position)
                new = _horizontal_point(after_position)
                displacement = new[0] - old[0], new[1] - old[1]
                distance = math.hypot(*displacement)
                if distance >= 0.1 and not self._fell(before, after):
                    self.move_samples.append(distance)
                    self.direction_samples[action].append(displacement)
                    self.successful_moves += 1
                    if (
                        before_cell is not None
                        and after_cell is not None
                        and before_cell.identifier != after_cell.identifier
                    ):
                        self.observed_cell_edges.add(frozenset((
                            before_cell.identifier,
                            after_cell.identifier,
                        )))
                else:
                    self.blocked_attempts += 1
        elif action.lower() in {"interact", "use", "activate"}:
            self.interaction_attempts += 1

        record = {
            "type": "sweep_step",
            "source": "sweep",
            "phase": "sweep",
            "phase_step": phase_step,
            "step": phase_step,
            "episode": self.episode,
            "episode_before": self.episode,
            "episode_after": self.episode,
            "before": before,
            "decision": {
                "action": action,
                "reason": f"Scripted visible-entity continuous {purpose}.",
                "expected_position": None,
                "notes": [],
            },
            "after": after,
            "qa_mode": True,
        }
        anomalies = self._record_anomalies(record)
        record["software_fact"] = {
            "source": "sweep",
            "anomalies": deepcopy(anomalies),
        }
        self.records.append(record)
        self.anomalies.extend(anomalies)
        self.observation = after
        if self.on_record is not None:
            self.on_record(deepcopy(record))
        if self._terminal(after):
            self.status = "terminal"
        return record

    def reset(self, *, purpose: str) -> bool:
        if self.action_count >= self.max_actions:
            self.status = "action_limit"
            return False
        reset = getattr(self.game, "reset", None)
        if not callable(reset):
            return False
        before = deepcopy(self.observation)
        old_episode = self.episode
        try:
            returned = reset()
            after = deepcopy(returned if isinstance(returned, dict) else self.game.observe())
        except (KeyError, OSError, RuntimeError, TypeError, ValueError) as exc:
            self.status = "environment_error"
            self.error = f"{type(exc).__name__}: {exc}"
            return False
        self.episode += 1
        self.reset_count += 1
        phase_step = self.action_count + 1
        record = {
            "type": "sweep_reset",
            "source": "sweep",
            "phase": "sweep",
            "phase_step": phase_step,
            "step": phase_step,
            "episode_before": old_episode,
            "episode_after": self.episode,
            "before": before,
            "decision": {
                "action": "reset",
                "reason": f"Scripted continuous {purpose}.",
                "expected_position": None,
                "notes": [],
            },
            "after": after,
            "qa_mode": True,
        }
        from .continuous import continuous_observation_equal

        anomalies = []
        if not continuous_observation_equal(
            self.initial_observation,
            after,
            tolerance=0.05,
            ignore_tick=True,
        ):
            anomalies.append({
                "type": "reset_inconsistency",
                "aspect": "reset_inconsistency",
                "source": "sweep",
                "phase": "sweep",
                "phase_step": phase_step,
                "step": phase_step,
                "evidence_steps": [phase_step],
                "action": "reset",
                "expected": {"observation": deepcopy(self.initial_observation)},
                "actual": {"observation": deepcopy(after)},
            })
        record["software_fact"] = {
            "source": "sweep",
            "anomalies": deepcopy(anomalies),
        }
        self.records.append(record)
        self.anomalies.extend(anomalies)
        self.observation = after
        current = self._current_cell()
        if current is not None:
            self.visited.add(current.identifier)
        if self.on_record is not None:
            self.on_record(deepcopy(record))
        return True

    def navigate(self, target: _FloorCell, *, allow_reset: bool = True) -> bool:
        failures = 0
        while self.status == "completed":
            current = self._current_cell()
            if current is not None and current.identifier == target.identifier:
                return True
            path = self._path_to(target)
            if not path:
                break
            next_cell = path[0]
            position = self._current_position()
            if position is None:
                self.status = "invalid_observation"
                return False
            action = self._action_toward(_horizontal_point(position), next_cell.center)
            if action is None:
                break
            before_cell = current
            before_position = _horizontal_point(position)
            record = self.perform(action, purpose="floor-cell BFS navigation")
            if record is None:
                return False
            if (
                self._fell(record["before"], record["after"])
                or self._has_anomaly(record, "sank_into_visible_floor")
            ):
                self._block_cell_edge(before_cell, next_cell)
                self.reset(purpose="fall-or-sink recovery")
                return False
            if self._has_anomaly(
                record,
                "blocked_no_visible_obstacle",
                "player_stuck_in_geometry",
            ):
                if allow_reset:
                    self.reset(purpose="stuck-navigation recovery")
                return False
            after_position_value = self._current_position()
            after_position = (
                _horizontal_point(after_position_value)
                if after_position_value is not None
                else before_position
            )
            if math.dist(before_position, after_position) < 0.1:
                failures += 1
                if before_cell is not None:
                    self.blocked_cell_edges.add((
                        before_cell.identifier,
                        next_cell.identifier,
                    ))
                if failures >= 2:
                    break
            else:
                failures = 0
        return False

    def _nearby_dynamic_signature(
        self,
        cell: _FloorCell,
    ) -> tuple[tuple[str, str], ...]:
        position = self._current_position()
        if position is None:
            return ()
        point = _horizontal_point(position)
        entities = self.observation.get("entities")
        if not isinstance(entities, list):
            return ()
        reach = max(1.0, min(1.75, self.move_distance + 0.2))
        nearby = []
        for index, entity in enumerate(entities):
            if not isinstance(entity, Mapping):
                continue
            kind = entity.get("kind")
            state = entity.get("state")
            if (
                isinstance(kind, str)
                and any(token in kind.lower() for token in (*_CONTINUOUS_FLOOR_KINDS, "wall"))
                and isinstance(state, Mapping)
                and state.get("solid") is True
                and set(state).issubset({"solid"})
            ):
                continue
            entity_position = _continuous_vector(entity.get("position"))
            entity_size = _continuous_vector(entity.get("size"))
            if entity_position is None:
                continue
            center = _horizontal_point(entity_position)
            half = (
                tuple(value / 2.0 for value in _horizontal_point(entity_size))
                if entity_size is not None and len(entity_size) == len(entity_position)
                else (0.0, 0.0)
            )
            distance = math.hypot(
                max(0.0, abs(point[0] - center[0]) - half[0]),
                max(0.0, abs(point[1] - center[1]) - half[1]),
            )
            if distance <= reach:
                identifier = entity.get("id")
                nearby.append((
                    identifier if isinstance(identifier, str) else f"entity-{index}",
                    repr(state) if isinstance(state, Mapping) else "{}",
                ))
        return tuple(sorted(nearby))

    def interact_nearby(self, cell: _FloorCell) -> None:
        interact = next(
            (control for control in self.controls if control.lower() in {"interact", "use", "activate"}),
            None,
        )
        if interact is None or self.status != "completed":
            return
        signature = self._nearby_dynamic_signature(cell)
        key = cell.identifier, signature
        if not signature or key in self.interaction_signatures:
            return
        self.interaction_signatures.add(key)
        record = self.perform(interact, purpose="nearby-object interaction probe")
        if record is not None:
            # A visible interaction can change which previously blocked routes
            # are traversable (for example, an opened door).
            self.blocked_cell_edges.clear()

    @staticmethod
    def _dynamic_progress(observation: dict) -> tuple[object, object]:
        player = observation.get("player")
        inventory = (
            deepcopy(player.get("inventory", []))
            if isinstance(player, Mapping) else []
        )
        entities = observation.get("entities")
        dynamic = []
        if isinstance(entities, list):
            for entity in entities:
                if not isinstance(entity, Mapping):
                    continue
                kind = str(entity.get("kind", "")).lower()
                state = entity.get("state")
                if (
                    any(token in kind for token in (*_CONTINUOUS_FLOOR_KINDS, "wall"))
                    and isinstance(state, Mapping)
                    and state.get("solid") is True
                    and set(state).issubset({"solid"})
                ):
                    continue
                dynamic.append((
                    entity.get("id"),
                    deepcopy(entity.get("position")),
                    deepcopy(state),
                ))
        return inventory, dynamic

    def _progress_changed(self) -> bool:
        return self._dynamic_progress(self.observation) != self._dynamic_progress(
            self.initial_observation
        )

    def run(self) -> SweepResult:
        if self._terminal():
            self.status = "terminal"
        order = self._bfs_cells() if self.status == "completed" else []
        if not order and self.status == "completed":
            self.status = "invalid_observation"
        for _pass in range(3):
            pass_action_start = self.action_count
            for target in order:
                if self.status != "completed":
                    break
                completed = all(
                    action in self.tried.get(target.identifier, set())
                    or (target.identifier, action) in self.skipped_terminal_pairs
                    for action in self.directions.values()
                )
                if completed:
                    continue
                if not self.navigate(target):
                    continue
                self.interact_nearby(target)
                for action in self.directions.values():
                    if self.status != "completed":
                        break
                    if action in self.tried.get(target.identifier, set()):
                        continue
                    if not self.navigate(target):
                        break
                    if self._action_enters_terminal_cell(action):
                        self.skipped_terminal_pairs.add((target.identifier, action))
                        continue
                    projected = self._projected_cell(action)
                    record = self.perform(action, purpose="four-direction floor-cell probe")
                    if record is None:
                        break
                    if (
                        self._fell(record["before"], record["after"])
                        or self._has_anomaly(record, "sank_into_visible_floor")
                    ):
                        self._block_cell_edge(target, projected)
                        self.reset(purpose="fall-or-sink recovery")
                        break
                    current = self._current_cell()
                    if (
                        current is not None
                        and current.identifier in self.wall_occupied_cells
                        and self.status == "completed"
                    ):
                        continuation = self.perform(
                            action,
                            purpose="visible-solid boundary continuation probe",
                        )
                        if continuation is not None and (
                            self._fell(
                                continuation["before"], continuation["after"]
                            )
                            or self._has_anomaly(
                                continuation, "sank_into_visible_floor"
                            )
                        ):
                            self.reset(purpose="fall-or-sink recovery")
                            break
            if self.status != "completed" or self.action_count == pass_action_start:
                break

        if (
            not self.restoration_probe_done
            and self.action_count < self.max_actions
            and self._progress_changed()
        ):
            self.restoration_probe_done = self.reset(
                purpose="visible-state restoration probe"
            )

        if self.status == "completed" and self.action_count >= self.max_actions:
            self.status = "action_limit"
        visited_cells = [
            cell for cell in self.cells if cell.identifier in self.visited
        ][:_CONTINUOUS_CELL_LIMIT]
        tried_directions = [
            {
                "cell": _rounded_point(cell.center),
                "controls": [
                    action for action in self.directions.values()
                    if action in self.tried.get(cell.identifier, set())
                ],
            }
            for cell in visited_cells
        ]
        summary = {
            "version": SWEEP_SUMMARY_VERSION,
            "source": "sweep",
            "status": self.status,
            "actions": self.action_count,
            "action_budget": self.max_actions,
            "movement_attempts": self.movement_attempts,
            "blocked_attempts": self.blocked_attempts,
            "successful_moves": self.successful_moves,
            "visited_positions": len(self.visited),
            "continuous": True,
            "move_distance": round(self.move_distance, 3),
            "visible_floor_cells": len(self.cells),
            "visited_tile_cells": [
                _rounded_point(cell.center) for cell in visited_cells
            ],
            "tried_directions": tried_directions,
            "interaction_attempts": self.interaction_attempts,
            "resets": self.reset_count,
            "anomaly_count": len(self.anomalies),
            "anomalies": [
                deepcopy(value) for value in self.anomalies[-SWEEP_ANOMALY_LIMIT:]
            ],
        }
        if self.error is not None:
            summary["error"] = self.error
        return SweepResult(self.records, summary)


def run_collision_sweep(
    game: object,
    *,
    max_actions: int = 400,
    capabilities: object = None,
    controls: Sequence[object] | None = None,
    on_record: Callable[[dict], None] | None = None,
) -> SweepResult:
    """Run a public-observation collision sweep with a hard action bound.

    A missing advertised grid/legend capability is a normal, zero-action skip.
    When an adapter does not expose capabilities (the built-in compatibility
    case), the presence of ``map``/``grid`` and ``legend`` is used instead.
    """
    if type(max_actions) is not int or max_actions <= 0:
        raise ValueError("Sweep action budget must be a positive integer.")
    observation = deepcopy(game.observe())
    if not isinstance(observation, dict):
        return _skipped_result("invalid_observation", max_actions)

    advertised = _adapter_capabilities(game) if capabilities is None else capabilities
    continuous_capability = _capability_value(advertised, "continuous")
    solids_visible_capability = _capability_value(advertised, "solids_visible")
    if continuous_capability is True and solids_visible_capability is True:
        if controls is None:
            control_names = _adapter_controls(game, observation)
        else:
            control_names = [
                name for value in controls
                if (name := _control_name(value)) is not None
            ]
        directions = _direction_controls(control_names)
        if len(directions) != 4:
            return _skipped_result("controls_unsupported", max_actions)
        if _continuous_player_position(observation) is None:
            return _skipped_result("invalid_observation", max_actions)
        cells = _visible_floor_cells(observation)
        if not cells:
            return _skipped_result("invalid_observation", max_actions)
        return _ContinuousSweepRunner(
            game,
            observation,
            cells,
            directions,
            control_names,
            max_actions,
            on_record,
        ).run()

    grid_capability = _capability_value(advertised, "grid")
    legend_capability = _capability_value(advertised, "legend")
    if grid_capability is False or legend_capability is False:
        return _skipped_result("capability_missing", max_actions)

    rows = _grid_rows(observation)
    legend = observation.get("legend")
    if rows is None or not isinstance(legend, Mapping):
        return _skipped_result("capability_missing", max_actions)
    if _player_position(observation) is None:
        return _skipped_result("invalid_observation", max_actions)

    if controls is None:
        control_names = _adapter_controls(game, observation)
    else:
        control_names = [
            name for value in controls
            if (name := _control_name(value)) is not None
        ]
    directions = _direction_controls(control_names)
    if len(directions) != 4:
        return _skipped_result("controls_unsupported", max_actions)

    return _SweepRunner(
        game,
        observation,
        rows,
        legend,
        directions,
        max_actions,
        on_record,
    ).run()
