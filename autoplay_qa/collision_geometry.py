"""Pure narrow-phase checks for reported axis-aligned solid-box candidates.

The legacy continuous detector uses expanded AABBs as a broad phase.  Those
boxes have square corners; an upright capsule has round corners.  These helpers
refine reporting/reproduction only and do not change recorded detector facts.

Endpoints define a straight observed chord, not a measured physics trajectory.
An intersection therefore remains a candidate, not proof of a collision bug.
With no visible capsule height, the horizontal test is a necessary condition
only: it can rule out a corner miss without inventing vertical dimensions.
"""

from __future__ import annotations

import math
from typing import Sequence

from .continuous import _player_radius


def _number(value: object) -> bool:
    if type(value) not in (int, float):
        return False
    try:
        return math.isfinite(float(value))
    except (ValueError, TypeError, OverflowError):
        return False


def _vector(value: object, dimensions: int = 3) -> tuple[float, ...] | None:
    if (not isinstance(value, (list, tuple)) or len(value) != dimensions
            or not all(_number(item) for item in value)):
        return None
    return tuple(float(item) for item in value)


def segment_box_distance_squared(start: Sequence[float], end: Sequence[float],
                                 lower: Sequence[float], upper: Sequence[float]) -> float:
    """Exact minimum distance from a segment to an axis-aligned 2D/3D box.

    Squared distance is piecewise quadratic.  Coordinate/face crossings split
    [0,1] into at most seven intervals in 3D; endpoints and each interval's one
    quadratic minimum suffice.  There is no sampling or iterative search.
    """
    if not isinstance(start, (list, tuple)):
        raise ValueError("Segment-box distance requires coordinate vectors.")
    dimensions = len(start)
    if dimensions not in (2, 3):
        raise ValueError("Segment-box distance requires two or three dimensions.")
    vectors = [_vector(value, dimensions) for value in (start, end, lower, upper)]
    if any(value is None for value in vectors):
        raise ValueError("Segment and bounds must contain finite coordinates.")
    start, end, lower, upper = vectors
    if any(low > high for low, high in zip(lower, upper)):
        raise ValueError("Box bounds are reversed.")
    delta = tuple(last - first for first, last in zip(start, end))
    if not all(math.isfinite(value) for value in delta):
        raise ValueError("Segment extent is not numerically representable.")
    breaks = {0.0, 1.0}
    for axis, velocity in enumerate(delta):
        if velocity:
            for boundary in (lower[axis], upper[axis]):
                point = (boundary - start[axis]) / velocity
                if 0.0 < point < 1.0:
                    breaks.add(point)
    intervals = sorted(breaks)

    def distance(parameter: float) -> float:
        gaps = []
        for first, velocity, low, high in zip(start, delta, lower, upper):
            value = first + parameter * velocity
            gap = low - value if value < low else value - high if value > high else 0.0
            gaps.append(gap * gap)
        return math.fsum(gaps)

    closest = min(distance(0.0), distance(1.0))
    for left, right in zip(intervals, intervals[1:]):
        midpoint = (left + right) / 2.0
        quadratic, linear = [], []
        for first, velocity, low, high in zip(start, delta, lower, upper):
            value = first + midpoint * velocity
            boundary = low if value < low else high if value > high else None
            if boundary is not None:
                quadratic.append(velocity * velocity)
                linear.append(velocity * (first - boundary))
        coefficient = math.fsum(quadratic)
        candidate = left
        if coefficient:
            optimum = -math.fsum(linear) / coefficient
            if math.isfinite(optimum):
                candidate = min(right, max(left, optimum))
        closest = min(closest, distance(left), distance(right), distance(candidate))
    if not math.isfinite(closest):
        raise ValueError("Distance is not numerically representable.")
    return closest


def _segment_enters_open_box(start, end, lower, upper) -> bool:
    """Point/zero-radius case: touching a face alone is not penetration."""
    enter, leave = 0.0, 1.0
    for first, last, low, high in zip(start, end, lower, upper):
        velocity = last - first
        if not velocity:
            if not low < first < high:
                return False
            continue
        near, far = (low - first) / velocity, (high - first) / velocity
        enter, leave = max(enter, min(near, far)), min(leave, max(near, far))
        if enter >= leave:
            return False
    return enter < leave


def swept_capsule_box_overlap(start: Sequence[float], end: Sequence[float],
                              lower: Sequence[float], upper: Sequence[float], *,
                              radius: float, height: float | None = None) -> bool:
    """Check a translating upright capsule against an axis-aligned box.

    An explicit full height gives the exact straight-sweep volume.  Eliminating
    the capsule's vertical centerline parameter expands only the box's Y range
    by max(height/2-radius, 0), then compares Euclidean distance to the actual
    radius.  Unknown height uses the exact horizontal necessary condition.
    """
    vectors = [_vector(value) for value in (start, end, lower, upper)]
    if any(value is None for value in vectors) or not _number(radius) or radius < 0:
        raise ValueError("Capsule and box need finite, nonnegative geometry.")
    start, end, lower, upper = vectors
    if any(low > high for low, high in zip(lower, upper)):
        raise ValueError("Box bounds are reversed.")
    radius = float(radius)
    if height is None:
        axes = (0, 2)
        start, end, lower, upper = [tuple(vector[axis] for axis in axes)
                                    for vector in (start, end, lower, upper)]
    else:
        if not _number(height) or height < 2.0 * radius:
            raise ValueError("Capsule height must be at least its diameter.")
        half_line = float(height) / 2.0 - radius
        lower = (lower[0], lower[1] - half_line, lower[2])
        upper = (upper[0], upper[1] + half_line, upper[2])
    if radius == 0.0:
        return _segment_enters_open_box(start, end, lower, upper)
    squared = segment_box_distance_squared(start, end, lower, upper)
    radius_squared = radius * radius
    if not math.isfinite(radius_squared):
        raise ValueError("Capsule radius is not numerically representable.")
    # Floating arithmetic at exact tangency is not evidence of penetration.
    # This is relative numeric precision, not a shrunken physical radius.
    return squared < radius_squared and not math.isclose(squared, radius_squared, rel_tol=1e-12, abs_tol=0.0)


def passed_solid_geometry_supported(record: dict, anomaly: dict) -> bool:
    """Require rounded geometry support for a broad-phase solid candidate.

    Only current visible solid entities supply geometry.  A stored anomaly's
    entity ID selects the target; its copied bounds are never trusted instead
    of the transition's observation.  Missing/malformed geometry is unverified.
    This helper does not relax the legacy detector's input contract or turn an
    earlier malformed-recording failure into a clean-game result.
    """
    before, after = record.get("before"), record.get("after")
    if not isinstance(before, dict) or not isinstance(after, dict):
        return False
    player = before.get("player")
    after_player = after.get("player")
    if not isinstance(player, dict) or not isinstance(after_player, dict):
        return False
    start, end = _vector(player.get("position")), _vector(after_player.get("position"))
    if start is None or end is None:
        return False
    height = player.get("height")
    if height is None:
        size = _vector(player.get("size"))
        height = size[1] if size is not None else None
    identifier = None
    for field in ("actual", "expected"):
        details = anomaly.get(field)
        solid = details.get("solid_entity") if isinstance(details, dict) else None
        if isinstance(solid, dict) and solid.get("id") is not None:
            identifier = solid["id"]
            break
    entities = before.get("entities")
    if not isinstance(entities, list):
        return False
    for entity in entities:
        if not isinstance(entity, dict) or (identifier is not None and entity.get("id") != identifier):
            continue
        state = entity.get("state")
        if not isinstance(state, dict) or state.get("solid") is not True:
            continue
        center, size = _vector(entity.get("position")), _vector(entity.get("size"))
        if center is None or size is None or any(value < 0.0 for value in size):
            continue
        lower = tuple(value - extent / 2.0 for value, extent in zip(center, size))
        upper = tuple(value + extent / 2.0 for value, extent in zip(center, size))
        try:
            if swept_capsule_box_overlap(start, end, lower, upper,
                                          radius=_player_radius(before), height=height):
                return True
        except (ValueError, OverflowError):
            continue
    return False
