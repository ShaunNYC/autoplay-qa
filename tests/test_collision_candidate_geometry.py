"""Independent narrow-phase reporting checks; no model or original game edits."""

from copy import deepcopy
import math
import random
import unittest

from autoplay_qa.collision_geometry import (
    passed_solid_geometry_supported, segment_box_distance_squared,
    swept_capsule_box_overlap,
)
from autoplay_qa.continuous import continuous_anomalies
from autoplay_qa.reporting import (
    _anomaly_has_issue_evidence, _trace_reproduces, extract_issues,
    extract_observations,
)
from tests.test_continuous_reporting import entity, observation, recording, transition


WALL = entity("visible-obstacle", "wall", [0, 1.25, 4], [0.5, 2.5, 2], solid=True)
LOWER, UPPER = [-0.25, 0, 3], [0.25, 2.5, 5]


def scene(before_position, after_position, *, radius=None, height=None):
    before = observation(before_position, entities=[WALL])
    after = observation(after_position, entities=[WALL], tick=1)
    for obs in (before, after):
        if radius is not None:
            obs["player"]["radius"] = radius
        if height is not None:
            obs["player"]["height"] = height
    row = transition(18, "north", before, after)
    return row, recording(before, [row])


def raw_solid_candidate(row):
    return next(value for value in continuous_anomalies(row) if value["type"] == "passed_through_solid")


def trace(row):
    return [{"source_step": row["step"], "action": row["decision"]["action"],
             "_before_observation": row["before"], "_after_observation": row["after"]}]


class SegmentBoxDistanceTests(unittest.TestCase):
    def test_saved_corner_counterexample_has_endpoint_distance_outside_radius(self):
        start, end = [0.65, 1, 2.2], [0.74, 0.99, 2.87]
        squared = segment_box_distance_squared(start, end, LOWER, UPPER)
        self.assertAlmostEqual(squared, 0.257, places=12)
        self.assertAlmostEqual(math.sqrt(squared), 0.506951674225463, places=12)
        self.assertFalse(swept_capsule_box_overlap(start, end, LOWER, UPPER, radius=0.5))
        self.assertFalse(swept_capsule_box_overlap(start, end, LOWER, UPPER, radius=0.5, height=2))

    def test_piecewise_quadratic_minimum_can_be_between_face_crossings(self):
        # The nearest point has t=.2, outside both the left and upper faces.
        squared = segment_box_distance_squared([-1, 0, 2], [2, 0, 3], [0, -1, 0], [1, 1, 1])
        self.assertAlmostEqual(squared, 1.6)

    def test_random_segments_agree_with_independent_dense_distance_upper_bound(self):
        rng = random.Random(731)
        low, high = [-0.7, -1.1, 0.4], [0.8, 0.6, 1.5]
        for _ in range(32):
            start = [rng.uniform(-3, 3) for _ in range(3)]
            end = [rng.uniform(-3, 3) for _ in range(3)]
            exact = math.sqrt(segment_box_distance_squared(start, end, low, high))
            dense = min(math.sqrt(sum(
                max(low[axis] - (start[axis] + (end[axis] - start[axis]) * step / 400),
                    0, (start[axis] + (end[axis] - start[axis]) * step / 400) - high[axis]) ** 2
                for axis in range(3)
            )) for step in range(401))
            self.assertLessEqual(exact, dense + 1e-12)
            self.assertLessEqual(dense - exact, math.dist(start, end) / 400 + 1e-12)

    def test_crossing_initial_overlap_terminal_overlap_and_stationary_overlap(self):
        for start, end in (
            ([-1.5, 1, 4], [1.5, 1, 4]),
            ([0.6, 1, 4], [1.5, 1, 4]),
            ([1.5, 1, 4], [0.6, 1, 4]),
            ([0.6, 1, 4], [0.6, 1, 4]),
        ):
            with self.subTest(start=start, end=end):
                self.assertTrue(swept_capsule_box_overlap(start, end, LOWER, UPPER, radius=0.5, height=2))

    def test_tangency_is_not_penetration_but_small_real_overlap_is_kept(self):
        self.assertFalse(swept_capsule_box_overlap([0.75, 1, 3], [0.75, 1, 5], LOWER, UPPER, radius=0.5))
        self.assertFalse(swept_capsule_box_overlap([0.65, 1, 2], [0.65, 1, 2.7], LOWER, UPPER, radius=0.5))
        self.assertTrue(swept_capsule_box_overlap([0.649, 1, 2], [0.649, 1, 2.7], LOWER, UPPER, radius=0.5))

    def test_player_position_rounding_toward_saved_corner_still_misses(self):
        # +/- .005 player-coordinate rounding, choosing the nearer endpoints.
        low_round = [0.645, 1, 2.205]
        high_round = [0.735, 0.99, 2.875]
        horizontal_sq = segment_box_distance_squared(
            [low_round[0], low_round[2]], [high_round[0], high_round[2]],
            [LOWER[0], LOWER[2]], [UPPER[0], UPPER[2]],
        )
        self.assertAlmostEqual(horizontal_sq, 0.25085)
        self.assertGreater(math.sqrt(horizontal_sq), 0.5)
        self.assertFalse(swept_capsule_box_overlap(low_round, high_round, LOWER, UPPER, radius=0.5))

    def test_explicit_height_uses_capsule_line_and_rounded_vertical_caps(self):
        low, high = [-1, -1, -1], [1, 1, 1]
        self.assertFalse(swept_capsule_box_overlap([-2, 2, 0], [2, 2, 0], low, high, radius=0.5, height=2))
        self.assertTrue(swept_capsule_box_overlap([-2, 1.99, 0], [2, 1.99, 0], low, high, radius=0.5, height=2))
        self.assertFalse(swept_capsule_box_overlap([1.4, 1.9, -0.1], [1.4, 1.9, 0.1], low, high, radius=0.5, height=2))
        # Without public height, XZ is only a necessary condition, deliberately.
        self.assertTrue(swept_capsule_box_overlap([1.4, 1.9, -0.1], [1.4, 1.9, 0.1], low, high, radius=0.5))

    def test_zero_radius_point_crossing_and_boundary_only_contact(self):
        self.assertTrue(swept_capsule_box_overlap([-1, 1, 4], [1, 1, 4], LOWER, UPPER, radius=0))
        self.assertFalse(swept_capsule_box_overlap([0.25, 1, 3], [0.25, 1, 5], LOWER, UPPER, radius=0))

    def test_invalid_geometry_is_rejected(self):
        for invalid in (float("nan"), float("inf"), True, 10**1000):
            with self.assertRaises(ValueError):
                segment_box_distance_squared([invalid, 0, 0], [1, 0, 0], LOWER, UPPER)
        with self.assertRaises(ValueError):
            swept_capsule_box_overlap([0, 1, 4], [0, 1, 4], LOWER, UPPER, radius=-1)
        with self.assertRaises(ValueError):
            swept_capsule_box_overlap([0, 1, 4], [0, 1, 4], LOWER, UPPER, radius=0.5, height=0.9)


class CollisionCandidateReportingTests(unittest.TestCase):
    def test_saved_false_candidate_becomes_observation_and_raw_detector_is_unchanged(self):
        row, records = scene([0.65, 1, 2.2], [0.74, 0.99, 2.87])
        raw = raw_solid_candidate(row)
        row["software_fact"] = {"anomalies": [deepcopy(raw)]}
        before = deepcopy(records)
        self.assertFalse(passed_solid_geometry_supported(row, raw))
        self.assertEqual(extract_issues(records), [])
        observations = extract_observations(records)
        self.assertEqual(len(observations), 1)
        self.assertEqual(observations[0]["type"], "passed_through_solid")
        self.assertIn("rounded capsule/box", observations[0]["reason"])
        self.assertEqual(records, before)
        self.assertEqual(raw_solid_candidate(row), raw)

    def test_real_wall_cross_and_terminal_penetration_remain_reportable(self):
        for start, end in (([-1.5, 1, 4], [1.5, 1, 4]),
                           ([-1.5, 1, 4], [-0.4, 1, 4]),
                           ([0.73, 1, 4], [0.3, 1, 4])):
            with self.subTest(start=start, end=end):
                row, records = scene(start, end)
                raw = raw_solid_candidate(row)
                self.assertTrue(passed_solid_geometry_supported(row, raw))
                issues = extract_issues(records)
                self.assertEqual(len(issues), 1)
                self.assertEqual(issues[0]["type"], "passed_through_solid")
                self.assertTrue(_trace_reproduces(issues[0], records, row["before"], trace(row)))

    def test_old_raw_false_issue_cannot_reproduce_by_repeating_broad_phase_alone(self):
        row, records = scene([0.65, 1, 2.2], [0.74, 0.99, 2.87])
        self.assertTrue(raw_solid_candidate(row))
        old_issue = {"type": "passed_through_solid", "evidence_steps": [18]}
        self.assertFalse(_trace_reproduces(old_issue, records, row["before"], trace(row)))
        self.assertFalse(_anomaly_has_issue_evidence("passed_through_solid", raw_solid_candidate(row), [18], {18: row}, row["before"]))

    def test_tangent_reporting_rejected_without_shrinking_real_radius(self):
        tangent, tangent_records = scene([0.65, 1, 2], [0.65, 1, 2.7])
        self.assertTrue(raw_solid_candidate(tangent))
        self.assertEqual(extract_issues(tangent_records), [])
        _, slight_overlap = scene([0.649, 1, 2], [0.649, 1, 2.7])
        self.assertEqual(len(extract_issues(slight_overlap)), 1)

    def test_visible_entity_bounds_and_solidity_override_copied_anomaly_geometry(self):
        row, _ = scene([0.65, 1, 2.2], [0.74, 0.99, 2.87])
        raw = raw_solid_candidate(row)
        raw["actual"]["solid_entity"]["position"] = [0.7, 1, 2.5]
        raw["actual"]["solid_entity"]["size"] = [10, 10, 10]
        self.assertFalse(passed_solid_geometry_supported(row, raw))
        raw["actual"]["solid_entity"]["id"] = "not-visible"
        self.assertFalse(passed_solid_geometry_supported(row, raw))

    def test_public_player_radius_and_height_are_used(self):
        row, _ = scene([0.65, 1, 2.2], [0.74, 0.99, 2.87], radius=0.6, height=2)
        raw = raw_solid_candidate(row)
        self.assertTrue(passed_solid_geometry_supported(row, raw))
        row["before"]["player"]["radius"] = 0.4
        self.assertFalse(passed_solid_geometry_supported(row, raw))

    def test_malformed_public_geometry_does_not_crash_candidate_guard(self):
        row, _ = scene([-1.5, 1, 4], [1.5, 1, 4])
        raw = raw_solid_candidate(row)
        for path in ("player", "solid"):
            malformed = deepcopy(row)
            if path == "player":
                malformed["before"]["player"]["position"][0] = 10**1000
            else:
                malformed["before"]["entities"][0]["size"][0] = 10**1000
            self.assertFalse(passed_solid_geometry_supported(malformed, raw))

    def test_genuine_floor_sink_is_unaffected(self):
        floor = entity("visible-floor", "floor", [0, -0.125, 0], [6, 0.25, 2], solid=True)
        before = observation([0, 1, 0], entities=[floor])
        after = observation([1.55, 0.4, 0], entities=[floor], tick=1)
        row = transition(1, "east", before, after)
        self.assertEqual([issue["type"] for issue in extract_issues(recording(before, [row]))], ["sank_into_visible_floor"])


if __name__ == "__main__":
    unittest.main()
