"""Visible-evidence anomaly checks for continuous games.

The checks in this module are deliberately game- and transport-agnostic.  They
consume only recorded observations, advertised capabilities, the executed
control, and earlier transitions from the same session.  In particular they do
not consume injected-defect metadata or a supplied route.
"""

from __future__ import annotations

from bisect import insort
from copy import deepcopy
import math
import statistics
from typing import Iterable, Sequence


FLOAT_TOLERANCE = 0.05
DEFAULT_MOVE_DISTANCE = 1.0
DEFAULT_PLAYER_RADIUS = 0.5
BLOCKED_MOVE_FRACTION = 0.30
MIN_BLOCKED_DISTANCE = 0.12
SANK_HEIGHT_DROP = 0.40
STUCK_DISPLACEMENT = 0.12

_CARDINAL_DIRECTIONS = {
    "north": (0.0, 1.0),
    "south": (0.0, -1.0),
    "east": (1.0, 0.0),
    "west": (-1.0, 0.0),
}
_OPPOSITES = {
    "north": "south",
    "south": "north",
    "east": "west",
    "west": "east",
}


def is_continuous_visible(observation: object) -> bool:
    """Return whether an observation explicitly advertises both prerequisites."""
    if not isinstance(observation, dict):
        return False
    capabilities = observation.get("_capabilities")
    return (
        isinstance(capabilities, dict)
        and capabilities.get("continuous") is True
        and capabilities.get("solids_visible") is True
    )


def continuous_observation_equal(
    left: object,
    right: object,
    *,
    tolerance: float = FLOAT_TOLERANCE,
    ignore_tick: bool = False,
) -> bool:
    """Compare JSON-like visible state with bounded numeric tolerance.

    Boolean values remain distinct from numbers, matching JSON semantics.
    Structure, strings, and feedback must still match exactly.
    """
    if isinstance(left, bool) or isinstance(right, bool):
        return type(left) is type(right) and left == right
    if _is_number(left) and _is_number(right):
        return abs(float(left) - float(right)) <= tolerance + 1e-9
    if isinstance(left, dict) and isinstance(right, dict):
        left_keys = set(left)
        right_keys = set(right)
        if ignore_tick:
            left_keys.discard("tick")
            right_keys.discard("tick")
        if left_keys != right_keys:
            return False
        return all(
            continuous_observation_equal(
                left[key], right[key], tolerance=tolerance, ignore_tick=False
            )
            for key in left_keys
        )
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        return len(left) == len(right) and all(
            continuous_observation_equal(
                left_item,
                right_item,
                tolerance=tolerance,
                ignore_tick=False,
            )
            for left_item, right_item in zip(left, right)
        )
    return type(left) is type(right) and left == right


def continuous_anomalies(
    record: dict,
    prior_records: Sequence[dict] = (),
    *,
    tolerance: float = FLOAT_TOLERANCE,
    typical_distance: float | None = None,
    movement_direction: tuple[float, float] | None = None,
    standing_height: float | None = None,
) -> list[dict]:
    """Derive continuous-world anomalies for one executed transition.

    Returned dictionaries use the same shape as the existing software anomaly
    facts.  The function is a no-op unless both the before and after
    observations explicitly advertise ``continuous`` and ``solids_visible``.
    """
    if not isinstance(record, dict):
        return []
    before = record.get("before")
    after = record.get("after")
    if not (is_continuous_visible(before) and is_continuous_visible(after)):
        return []
    if not isinstance(before, dict) or not isinstance(after, dict):
        return []

    decision = record.get("decision")
    decision = decision if isinstance(decision, dict) else {}
    action = decision.get("action")
    if typical_distance is None and action in _CARDINAL_DIRECTIONS:
        typical_distance = typical_unobstructed_move_distance(
            prior_records, tolerance=tolerance
        )
    elif typical_distance is None:
        typical_distance = DEFAULT_MOVE_DISTANCE
    if movement_direction is None and action in _CARDINAL_DIRECTIONS:
        movement_direction = _movement_direction(action, prior_records)
    if standing_height is None:
        standing_height = typical_grounded_height(prior_records)
    if standing_height is None:
        standing_height = _grounded_player_height(before)
    step = record.get("step")
    if type(step) is not int:
        step = 0
    base = {
        "step": step,
        "evidence_steps": [step],
        "action": action,
    }

    anomalies: list[dict] = []
    if record.get("type") not in {"reset", "sweep_reset"}:
        sunk_floor = None
        if standing_height is not None and not _below_standing_height(
            before,
            standing_height=standing_height,
        ):
            sunk_floor = _sunk_floor(
                after,
                standing_height=standing_height,
                tolerance=tolerance,
            )
        if sunk_floor is not None:
            after_position = _player_position(after)
            height_drop = (
                standing_height - after_position[1]
                if after_position is not None else None
            )
            anomalies.append({
                **base,
                "type": "sank_into_visible_floor",
                "aspect": "sank_into_visible_floor",
                "expected": {
                    "floor_entity": _entity_summary(sunk_floor),
                    "visible_floor_supports_player": True,
                    "standing_height": round(standing_height, 4),
                    "maximum_height_drop": SANK_HEIGHT_DROP,
                },
                "actual": {
                    "before_position": deepcopy(_player_position(before)),
                    "after_position": deepcopy(after_position),
                    "height_drop": (
                        round(height_drop, 4)
                        if height_drop is not None else None
                    ),
                    "before_grounded": _player_grounded(before),
                    "after_grounded": _player_grounded(after),
                    "feedback": _visible_events(after),
                },
            })

        fallen = _fall_feedback(after)
        if action in _CARDINAL_DIRECTIONS and not fallen:
            crossed = _crossed_solid(before, after, tolerance=tolerance)
            if crossed is not None:
                anomalies.append({
                    **base,
                    "type": "passed_through_solid",
                    "aspect": "passed_through_solid",
                    "expected": {
                        "solid_entity": _entity_summary(crossed),
                        "movement": "does not enter visible solid bounds",
                    },
                    "actual": {
                        "before_position": deepcopy(_player_position(before)),
                        "after_position": deepcopy(_player_position(after)),
                        "solid_entity": _entity_summary(crossed),
                        "feedback": _visible_events(after),
                    },
                })

            blocked = _blocked_without_visible_obstacle(
                record,
                prior_records,
                tolerance=tolerance,
                typical_distance=typical_distance,
                movement_direction=movement_direction,
                standing_height=standing_height,
            )
            if blocked is not None:
                anomalies.append({
                    **base,
                    "type": "blocked_no_visible_obstacle",
                    "aspect": "blocked_no_visible_obstacle",
                    "expected": {
                        "minimum_unobstructed_displacement": blocked["threshold"],
                        "typical_unobstructed_displacement": blocked["typical"],
                        "visible_solid_ahead": False,
                    },
                    "actual": {
                        "displacement": blocked["displacement"],
                        "before_position": deepcopy(_player_position(before)),
                        "after_position": deepcopy(_player_position(after)),
                        "feedback": _visible_events(after),
                    },
                })

            stuck = _stuck_in_geometry(
                record,
                standing_height=standing_height,
                tolerance=tolerance,
            )
            if stuck is not None:
                anomalies.append({
                    **base,
                    "type": "player_stuck_in_geometry",
                    "aspect": "player_stuck_in_geometry",
                    "expected": {
                        "floor_entity": _entity_summary(stuck["floor"]),
                        "movement_escapes_geometry": True,
                    },
                    "actual": {
                        "displacement": stuck["displacement"],
                        "before_position": deepcopy(_player_position(before)),
                        "after_position": deepcopy(_player_position(after)),
                        "standing_height": round(standing_height, 4),
                        "feedback": _visible_events(after),
                    },
                })

        floor = _fallen_through_floor(
            record,
            prior_records,
            tolerance=tolerance,
            typical_distance=typical_distance,
            movement_direction=movement_direction,
        )
        if floor is not None:
            anomalies.append({
                **base,
                "type": "fell_through_visible_floor",
                "aspect": "fell_through_visible_floor",
                "expected": {
                    "floor_entity": _entity_summary(floor),
                    "visible_floor_supports_player": True,
                },
                "actual": {
                    "before_position": deepcopy(_player_position(before)),
                    "after_position": deepcopy(_player_position(after)),
                    "feedback": _visible_events(after),
                },
            })
    return anomalies


def typical_grounded_height(
    prior_records: Sequence[dict],
) -> float | None:
    """Return the median visibly grounded player height from earlier rows."""
    heights: list[float] = []
    for prior in prior_records:
        if not isinstance(prior, dict):
            continue
        for observation in (prior.get("before"), prior.get("after")):
            height = _grounded_player_height(observation)
            if height is not None:
                heights.append(height)
    return float(statistics.median(heights)) if heights else None


def typical_unobstructed_move_distance(
    prior_records: Sequence[dict],
    *,
    tolerance: float = FLOAT_TOLERANCE,
) -> float:
    """Return the session median for earlier visibly free cardinal moves."""
    distances: list[float] = []
    for prior in prior_records:
        if not isinstance(prior, dict) or prior.get("type") not in {
            "step", "sweep_step"
        }:
            continue
        before = prior.get("before")
        after = prior.get("after")
        if not (is_continuous_visible(before) and is_continuous_visible(after)):
            continue
        decision = prior.get("decision")
        action = decision.get("action") if isinstance(decision, dict) else None
        if action not in _CARDINAL_DIRECTIONS or _fall_feedback(after):
            continue
        distance = _horizontal_displacement(before, after)
        if distance is None or distance <= max(MIN_BLOCKED_DISTANCE, tolerance * 2):
            continue
        if _crossed_solid(before, after, tolerance=tolerance) is not None:
            continue
        start = _player_position(before)
        direction = _normalized_horizontal_delta(prior)
        if (
            start is None
            or direction is None
            or _solid_ahead(
                before,
                start,
                direction,
                distance + max(0.15, tolerance * 2.0),
                tolerance=tolerance,
            )
        ):
            continue
        distances.append(distance)
    return (
        float(statistics.median(distances))
        if distances
        else DEFAULT_MOVE_DISTANCE
    )


class ContinuousSessionDetector:
    """Incrementally derive anomalies without rescanning session history.

    Call :meth:`anomalies` in execution order.  Calibration is updated only
    after checking the current row, so each verdict uses earlier moves exactly
    as the public ``continuous_anomalies`` function does.
    """

    def __init__(self, *, tolerance: float = FLOAT_TOLERANCE):
        self.tolerance = tolerance
        self._free_distances: list[float] = []
        self._standing_heights: list[float] = []
        self._direction_samples: dict[str, list[tuple[float, float]]] = {
            action: [] for action in _CARDINAL_DIRECTIONS
        }

    @property
    def typical_distance(self) -> float:
        count = len(self._free_distances)
        if not count:
            return DEFAULT_MOVE_DISTANCE
        middle = count // 2
        if count % 2:
            return self._free_distances[middle]
        return (
            self._free_distances[middle - 1] + self._free_distances[middle]
        ) / 2.0

    @property
    def typical_standing_height(self) -> float | None:
        count = len(self._standing_heights)
        if not count:
            return None
        middle = count // 2
        if count % 2:
            return self._standing_heights[middle]
        return (
            self._standing_heights[middle - 1]
            + self._standing_heights[middle]
        ) / 2.0

    def direction(self, action: object) -> tuple[float, float] | None:
        if not isinstance(action, str):
            return None
        samples = self._direction_samples.get(action, [])
        if not samples:
            opposite = self._direction_samples.get(_OPPOSITES.get(action, ""), [])
            samples = [(-item[0], -item[1]) for item in opposite]
        if samples:
            dx = statistics.median(item[0] for item in samples)
            dz = statistics.median(item[1] for item in samples)
            length = math.hypot(dx, dz)
            if length > 1e-9:
                return dx / length, dz / length
        return _CARDINAL_DIRECTIONS.get(action)

    def anomalies(self, record: dict) -> list[dict]:
        decision = record.get("decision") if isinstance(record, dict) else None
        action = decision.get("action") if isinstance(decision, dict) else None
        anomalies = continuous_anomalies(
            record,
            (),
            tolerance=self.tolerance,
            typical_distance=self.typical_distance,
            movement_direction=self.direction(action),
            standing_height=self.typical_standing_height,
        )
        self.observe(record)
        return anomalies

    def observe(self, record: dict) -> None:
        """Add one completed row to later calibration without producing facts."""
        if not isinstance(record, dict) or record.get("type") not in {
            "step", "sweep_step", "reset", "sweep_reset"
        }:
            return
        before = record.get("before")
        after = record.get("after")
        if not (is_continuous_visible(before) and is_continuous_visible(after)):
            return
        for observation in (before, after):
            height = _grounded_player_height(observation)
            if height is not None:
                insort(self._standing_heights, height)
        if record.get("type") not in {"step", "sweep_step"}:
            return
        decision = record.get("decision")
        action = decision.get("action") if isinstance(decision, dict) else None
        if action not in _CARDINAL_DIRECTIONS or _fall_feedback(after):
            return
        direction = _normalized_horizontal_delta(record)
        if direction is None:
            return
        self._direction_samples[action].append(direction)
        distance = _horizontal_displacement(before, after)
        start = _player_position(before)
        if (
            distance is None
            or distance <= max(MIN_BLOCKED_DISTANCE, self.tolerance * 2)
            or start is None
            or _crossed_solid(before, after, tolerance=self.tolerance) is not None
            or _solid_ahead(
                before,
                start,
                direction,
                distance + max(0.15, self.tolerance * 2.0),
                tolerance=self.tolerance,
            )
        ):
            return
        insort(self._free_distances, distance)


def _is_number(value: object) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
    )


def _vector(value: object, dimensions: int = 3) -> tuple[float, ...] | None:
    if (
        not isinstance(value, (list, tuple))
        or len(value) < dimensions
        or not all(_is_number(item) for item in value[:dimensions])
    ):
        return None
    return tuple(float(item) for item in value[:dimensions])


def _player_position(observation: object) -> tuple[float, float, float] | None:
    if not isinstance(observation, dict):
        return None
    player = observation.get("player")
    if not isinstance(player, dict):
        return None
    value = _vector(player.get("position"))
    return value if value is not None and len(value) == 3 else None


def _player_grounded(observation: object) -> bool | None:
    if not isinstance(observation, dict):
        return None
    player = observation.get("player")
    if not isinstance(player, dict):
        return None
    grounded = player.get("grounded")
    return grounded if isinstance(grounded, bool) else None


def _grounded_player_height(observation: object) -> float | None:
    if _player_grounded(observation) is not True:
        return None
    position = _player_position(observation)
    return position[1] if position is not None else None


def _visible_events(observation: object) -> list:
    if not isinstance(observation, dict):
        return []
    events = observation.get("events")
    return deepcopy(events) if isinstance(events, list) else []


def _fall_feedback(observation: object) -> bool:
    return any(
        isinstance(event, str) and "fell" in event.lower()
        for event in _visible_events(observation)
    )


def _player_radius(observation: dict) -> float:
    player = observation.get("player")
    player = player if isinstance(player, dict) else {}
    radius = player.get("radius")
    if _is_number(radius) and float(radius) >= 0.0:
        return float(radius)
    size = _vector(player.get("size"))
    if size is not None and all(value >= 0.0 for value in size):
        return min(size[0], size[2]) / 2.0
    return DEFAULT_PLAYER_RADIUS


def _solid_entities(observation: dict, *, include_floor: bool) -> Iterable[dict]:
    entities = observation.get("entities")
    if not isinstance(entities, list):
        return ()
    values = []
    for entity in entities:
        if not isinstance(entity, dict):
            continue
        state = entity.get("state")
        if not isinstance(state, dict) or state.get("solid") is not True:
            continue
        kind = str(entity.get("kind", "")).lower()
        if not include_floor and "floor" in kind:
            continue
        position = _vector(entity.get("position"))
        size = _vector(entity.get("size"))
        if (
            position is None
            or size is None
            or any(value < 0.0 for value in size)
        ):
            continue
        values.append(entity)
    return values


def _bounds(entity: dict, expansion: float = 0.0) -> tuple[tuple[float, ...], tuple[float, ...]] | None:
    position = _vector(entity.get("position"))
    size = _vector(entity.get("size"))
    if position is None or size is None:
        return None
    lower = tuple(
        position[index] - (size[index] / 2.0) - expansion
        for index in range(3)
    )
    upper = tuple(
        position[index] + (size[index] / 2.0) + expansion
        for index in range(3)
    )
    return lower, upper


def _strictly_inside(
    point: tuple[float, ...],
    lower: tuple[float, ...],
    upper: tuple[float, ...],
    tolerance: float,
) -> bool:
    return all(
        lower[index] + tolerance < point[index] < upper[index] - tolerance
        for index in range(3)
    )


def _segment_interval(
    start: tuple[float, ...],
    end: tuple[float, ...],
    lower: tuple[float, ...],
    upper: tuple[float, ...],
    *,
    tolerance: float,
) -> tuple[float, float] | None:
    """Return the parametric interval inside an AABB, if it has real depth."""
    enter, leave = 0.0, 1.0
    for index in range(3):
        delta = end[index] - start[index]
        if abs(delta) <= 1e-12:
            if not lower[index] < start[index] < upper[index]:
                return None
            continue
        near = (lower[index] - start[index]) / delta
        far = (upper[index] - start[index]) / delta
        if near > far:
            near, far = far, near
        enter = max(enter, near)
        leave = min(leave, far)
        if leave - enter <= 1e-9:
            return None
    if leave <= tolerance or enter >= 1.0 - tolerance:
        return None
    return enter, leave


def _crossed_solid(
    before: dict,
    after: dict,
    *,
    tolerance: float,
) -> dict | None:
    start = _player_position(before)
    end = _player_position(after)
    if start is None or end is None:
        return None
    if math.hypot(end[0] - start[0], end[2] - start[2]) <= tolerance:
        return None
    # Shrinking the nominal radius by the numeric tolerance treats a player
    # resting exactly against a collider as legal while still detecting real
    # penetration and tunnelling.
    expansion = max(0.0, _player_radius(before) - tolerance)
    for entity in _solid_entities(before, include_floor=False):
        entity_bounds = _bounds(entity, expansion)
        if entity_bounds is None:
            continue
        lower, upper = entity_bounds
        start_inside = _strictly_inside(start, lower, upper, tolerance)
        end_inside = _strictly_inside(end, lower, upper, tolerance)
        if end_inside and not start_inside:
            return entity
        interval = _segment_interval(
            start, end, lower, upper, tolerance=min(0.01, tolerance)
        )
        if interval is not None and not start_inside:
            return entity
    return None


def _horizontal_displacement(before: object, after: object) -> float | None:
    start = _player_position(before)
    end = _player_position(after)
    if start is None or end is None:
        return None
    return math.hypot(end[0] - start[0], end[2] - start[2])


def _normalized_horizontal_delta(record: dict) -> tuple[float, float] | None:
    start = _player_position(record.get("before"))
    end = _player_position(record.get("after"))
    if start is None or end is None:
        return None
    dx, dz = end[0] - start[0], end[2] - start[2]
    length = math.hypot(dx, dz)
    if length <= MIN_BLOCKED_DISTANCE:
        return None
    return dx / length, dz / length


def _movement_direction(
    action: str,
    prior_records: Sequence[dict],
) -> tuple[float, float] | None:
    samples: list[tuple[float, float]] = []
    inverse_samples: list[tuple[float, float]] = []
    for prior in prior_records:
        if not isinstance(prior, dict):
            continue
        decision = prior.get("decision")
        prior_action = decision.get("action") if isinstance(decision, dict) else None
        delta = _normalized_horizontal_delta(prior)
        if delta is None or _fall_feedback(prior.get("after")):
            continue
        if prior_action == action:
            samples.append(delta)
        elif prior_action == _OPPOSITES.get(action):
            inverse_samples.append((-delta[0], -delta[1]))
    chosen = samples or inverse_samples
    if chosen:
        dx = statistics.median(item[0] for item in chosen)
        dz = statistics.median(item[1] for item in chosen)
        length = math.hypot(dx, dz)
        if length > 1e-9:
            return dx / length, dz / length
    return _CARDINAL_DIRECTIONS.get(action)


def _solid_ahead(
    observation: dict,
    start: tuple[float, float, float],
    direction: tuple[float, float],
    reach: float,
    *,
    tolerance: float,
) -> bool:
    end = (
        start[0] + direction[0] * reach,
        start[1],
        start[2] + direction[1] * reach,
    )
    expansion = max(0.0, _player_radius(observation) - tolerance)
    for entity in _solid_entities(observation, include_floor=False):
        entity_bounds = _bounds(entity, expansion)
        if entity_bounds is None:
            continue
        lower, upper = entity_bounds
        if _strictly_inside(start, lower, upper, tolerance):
            continue
        if _strictly_inside(end, lower, upper, tolerance):
            return True
        if _segment_interval(start, end, lower, upper, tolerance=0.0) is not None:
            return True
    return False


def _blocked_without_visible_obstacle(
    record: dict,
    prior_records: Sequence[dict],
    *,
    tolerance: float,
    typical_distance: float | None = None,
    movement_direction: tuple[float, float] | None = None,
    standing_height: float | None = None,
) -> dict | None:
    decision = record.get("decision")
    action = decision.get("action") if isinstance(decision, dict) else None
    if action not in _CARDINAL_DIRECTIONS:
        return None
    before = record.get("before")
    after = record.get("after")
    displacement = _horizontal_displacement(before, after)
    start = _player_position(before)
    if displacement is None or start is None:
        return None
    if _player_grounded(before) is False:
        return None
    if (
        standing_height is not None
        and start[1] <= standing_height - SANK_HEIGHT_DROP
    ):
        return None
    typical = (
        typical_distance
        if typical_distance is not None
        else typical_unobstructed_move_distance(
            prior_records, tolerance=tolerance
        )
    )
    threshold = max(MIN_BLOCKED_DISTANCE, typical * BLOCKED_MOVE_FRACTION)
    if displacement > threshold + tolerance:
        return None
    direction = (
        movement_direction
        if movement_direction is not None
        else _movement_direction(action, prior_records)
    )
    if direction is None or _solid_ahead(
        before, start, direction, typical, tolerance=tolerance
    ):
        return None
    return {
        "displacement": round(displacement, 4),
        "typical": round(typical, 4),
        "threshold": round(threshold, 4),
    }


def _floor_under(
    observation: dict,
    x: float,
    z: float,
    *,
    tolerance: float,
    expansion: float = 0.0,
) -> dict | None:
    for entity in _solid_entities(observation, include_floor=True):
        if "floor" not in str(entity.get("kind", "")).lower():
            continue
        entity_bounds = _bounds(entity)
        if entity_bounds is None:
            continue
        lower, upper = entity_bounds
        # A capsule overlaps a floor footprint when its horizontal circle is
        # within its radius of the tile's AABB.  Expanding each AABB axis would
        # incorrectly include square corner regions beyond that radius.
        dx = max(lower[0] - x, 0.0, x - upper[0])
        dz = max(lower[2] - z, 0.0, z - upper[2])
        if math.hypot(dx, dz) <= expansion + tolerance:
            return entity
    return None


def _sunk_floor(
    observation: dict,
    *,
    standing_height: float,
    tolerance: float,
) -> dict | None:
    position = _player_position(observation)
    if position is None or not _below_standing_height(
        observation,
        standing_height=standing_height,
    ):
        return None
    return _floor_under(
        observation,
        position[0],
        position[2],
        tolerance=tolerance,
        expansion=_player_radius(observation),
    )


def _below_standing_height(
    observation: object,
    *,
    standing_height: float,
) -> bool:
    position = _player_position(observation)
    return (
        position is not None
        and position[1] <= standing_height - SANK_HEIGHT_DROP
    )


def _stuck_in_geometry(
    record: dict,
    *,
    standing_height: float | None,
    tolerance: float,
) -> dict | None:
    if standing_height is None:
        return None
    before = record.get("before")
    after = record.get("after")
    if not isinstance(before, dict) or not isinstance(after, dict):
        return None
    start = _player_position(before)
    end = _player_position(after)
    if start is None or end is None:
        return None
    floor = _sunk_floor(
        before,
        standing_height=standing_height,
        tolerance=tolerance,
    )
    if floor is None or _sunk_floor(
        after,
        standing_height=standing_height,
        tolerance=tolerance,
    ) is None:
        return None
    displacement = math.dist(start, end)
    threshold = max(STUCK_DISPLACEMENT, tolerance * 2.0)
    if displacement > threshold + tolerance:
        return None
    return {
        "floor": floor,
        "displacement": round(displacement, 4),
    }


def _fallen_through_floor(
    record: dict,
    prior_records: Sequence[dict],
    *,
    tolerance: float,
    typical_distance: float | None = None,
    movement_direction: tuple[float, float] | None = None,
) -> dict | None:
    before = record.get("before")
    after = record.get("after")
    if not isinstance(before, dict) or not isinstance(after, dict):
        return None
    start = _player_position(before)
    end = _player_position(after)
    if start is None or end is None:
        return None

    if _fall_feedback(after):
        decision = record.get("decision")
        action = decision.get("action") if isinstance(decision, dict) else None
        candidates: list[tuple[float, float]] = []
        if action in _CARDINAL_DIRECTIONS:
            direction = (
                movement_direction
                if movement_direction is not None
                else _movement_direction(action, prior_records)
            )
            if direction is not None:
                reach = (
                    typical_distance
                    if typical_distance is not None
                    else typical_unobstructed_move_distance(
                        prior_records, tolerance=tolerance
                    )
                )
                candidates.append((
                    start[0] + direction[0] * reach,
                    start[2] + direction[1] * reach,
                ))
        # The last reported pre-action position is still useful when an engine
        # respawns immediately and exposes no intermediate falling position.
        candidates.append((start[0], start[2]))
        for x, z in candidates:
            floor = _floor_under(before, x, z, tolerance=tolerance)
            if floor is not None:
                return floor

    floor = _floor_under(after, end[0], end[2], tolerance=tolerance)
    if floor is None:
        return None
    bounds = _bounds(floor)
    if bounds is None:
        return None
    floor_top = bounds[1][1]
    if end[1] < floor_top - tolerance:
        return floor
    return None


def _entity_summary(entity: dict) -> dict:
    return {
        key: deepcopy(entity.get(key))
        for key in ("id", "kind", "position", "size", "state")
        if key in entity
    }
