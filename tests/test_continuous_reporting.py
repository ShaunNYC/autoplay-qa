"""Continuous visible-geometry detectors and offline reporting semantics."""

from copy import deepcopy
from pathlib import Path
import unittest

from autoplay_qa.continuous import (
    ContinuousSessionDetector,
    continuous_anomalies,
    continuous_observation_equal,
    typical_unobstructed_move_distance,
)
from autoplay_qa.reporting import (
    evaluate_recording,
    extract_issues,
    reproduce_issue,
)
from tests.continuous_fixture import FakeContinuousGame


CAPABILITIES = {
    "continuous": True,
    "solids_visible": True,
    "grid": False,
    "legend": False,
}


def entity(identifier, kind, position, size, *, solid=None, **state):
    if solid is not None:
        state["solid"] = solid
    return {
        "id": identifier,
        "kind": kind,
        "position": list(position),
        "size": list(size),
        "state": state,
    }


def observation(
    position,
    *,
    entities=(),
    events=(),
    inventory=(),
    tick=0,
    capabilities=True,
    grounded=True,
):
    value = {
        "game": "continuous-fixture-v1",
        "tick": tick,
        "status": "playing",
        "player": {
            "position": list(position),
            "velocity": [0.0, 0.0, 0.0],
            "grounded": grounded,
            "inventory": list(inventory),
        },
        "entities": deepcopy(list(entities)),
        "events": list(events),
        "actions": ["north", "south", "east", "west", "interact"],
    }
    if capabilities:
        value["_capabilities"] = deepcopy(CAPABILITIES)
    return value


def transition(step, action, before, after, *, episode=0, kind="step"):
    return {
        "type": kind,
        "step": step,
        "episode": episode,
        "episode_before": episode - 1 if kind in {"reset", "sweep_reset"} else episode,
        "episode_after": episode,
        "source": "sweep" if kind.startswith("sweep_") else "llm",
        "before": deepcopy(before),
        "decision": {"action": action, "reason": "fixture", "notes": []},
        "after": deepcopy(after),
    }


def recording(initial, rows, defect=None):
    return [{
        "type": "session",
        "schema_version": 2,
        "game_version": "continuous-fixture-v1",
        "seed": 0,
        "initial_observation": deepcopy(initial),
        "evaluation": {
            "injected_defect": (
                {"name": defect, "description": "fixture"}
                if defect is not None else None
            ),
        },
    }, *rows]


class TunnelGame:
    version = "continuous-fixture-v1"

    def __init__(self, seed=0, defect=None):
        self.seed = seed
        self.defect = defect
        self.tick = 0
        self.position = [-1.5, 1.0, 0.0]
        self.wall = entity("wall", "wall", [0.0, 1.0, 0.0], [0.5, 2.0, 2.0], solid=True)

    def observe(self):
        return observation(
            self.position, entities=[self.wall], tick=self.tick
        )

    def act(self, action):
        if action != "east":
            raise ValueError(action)
        self.tick += 1
        self.position = [1.5, 1.0, 0.0]
        return self.observe()

    def reset(self):
        self.tick = 0
        self.position = [-1.5, 1.0, 0.0]
        return self.observe()


def tunnel_factory(seed, defect):
    return TunnelGame(seed, defect)


class ContinuousDetectorTests(unittest.TestCase):
    def test_passed_through_solid_uses_visible_aabb_and_ignores_legal_contact(self):
        wall = entity(
            "wall-interior", "wall", [0.0, 1.0, 0.0], [0.5, 2.0, 2.0], solid=True
        )
        before = observation([-1.5, 1.0, 0.0], entities=[wall])
        crossed = observation([1.5, 1.0, 0.0], entities=[wall], tick=1)
        stopped = observation([-0.75, 1.0, 0.0], entities=[wall], tick=1)

        anomalies = continuous_anomalies(transition(1, "east", before, crossed))
        legal = continuous_anomalies(transition(1, "east", before, stopped))

        self.assertEqual([item["type"] for item in anomalies], ["passed_through_solid"])
        self.assertNotIn("passed_through_solid", [item["type"] for item in legal])

    def test_blocked_move_uses_session_median_and_requires_no_solid_ahead(self):
        first = transition(
            1,
            "east",
            observation([-3.0, 1.0, 0.0]),
            observation([-1.45, 1.0, 0.0], tick=1),
        )
        ghost = entity(
            "door", "door", [0.0, 1.25, 0.0], [0.5, 2.5, 2.0],
            solid=False, open=True,
        )
        before = observation([-1.45, 1.0, 0.0], entities=[ghost], tick=1)
        after = observation([-1.44, 1.0, 0.0], entities=[ghost], tick=2)
        blocked = continuous_anomalies(
            transition(2, "east", before, after), [first]
        )

        self.assertAlmostEqual(typical_unobstructed_move_distance([first]), 1.55)
        self.assertEqual(
            [item["type"] for item in blocked], ["blocked_no_visible_obstacle"]
        )

        solid_door = deepcopy(ghost)
        solid_door["state"] = {"solid": True, "open": False}
        visible_obstacle = continuous_anomalies(
            transition(
                2,
                "east",
                observation([-0.75, 1.0, 0.0], entities=[solid_door]),
                observation([-0.75, 1.0, 0.0], entities=[solid_door], tick=2),
            ),
            [first],
        )
        self.assertNotIn(
            "blocked_no_visible_obstacle",
            [item["type"] for item in visible_obstacle],
        )

    def test_fall_feedback_is_checked_against_visible_solid_floor(self):
        floor = entity(
            "floor-0", "floor", [0.0, -0.125, 0.0], [2.0, 0.25, 2.0], solid=True
        )
        before = observation([0.0, 1.0, 0.0], entities=[floor])
        respawned = observation(
            [-4.0, 1.0, -4.0], entities=[floor], events=["You fell."], tick=1
        )

        anomalies = continuous_anomalies(
            transition(1, "north", before, respawned)
        )

        self.assertEqual(
            [item["type"] for item in anomalies], ["fell_through_visible_floor"]
        )

    def test_sink_fires_on_height_crossing_and_stuck_suppresses_blocked(self):
        floor = entity(
            "floor-0", "floor", [0.0, -0.125, 0.0], [2.0, 0.25, 2.0],
            solid=True,
        )
        calibration = transition(
            1,
            "east",
            observation([-1.55, 1.0, 0.0], entities=[floor]),
            observation([0.0, 1.0, 0.0], entities=[floor], tick=1),
        )
        sinking = transition(
            2,
            "east",
            observation([0.0, 1.0, 0.0], entities=[floor], tick=1),
            observation(
                [1.4, 0.2, 0.0],
                entities=[floor],
                tick=2,
                grounded=False,
            ),
        )
        wedged = transition(
            3,
            "north",
            sinking["after"],
            observation(
                [1.4, 0.2, 0.0],
                entities=[floor],
                tick=3,
                grounded=False,
            ),
        )

        sink_types = [
            item["type"]
            for item in continuous_anomalies(sinking, [calibration])
        ]
        stuck_types = [
            item["type"]
            for item in continuous_anomalies(wedged, [calibration, sinking])
        ]

        self.assertEqual(sink_types, ["sank_into_visible_floor"])
        self.assertEqual(stuck_types, ["player_stuck_in_geometry"])
        self.assertNotIn("blocked_no_visible_obstacle", stuck_types)

        outside_rounded_corner = transition(
            2,
            "east",
            observation([0.0, 1.0, 0.0], entities=[floor], tick=1),
            observation(
                [1.4, 0.2, 1.4],
                entities=[floor],
                tick=2,
                grounded=False,
            ),
        )
        self.assertNotIn(
            "sank_into_visible_floor",
            {
                item["type"]
                for item in continuous_anomalies(
                    outside_rounded_corner, [calibration]
                )
            },
        )

        detector = ContinuousSessionDetector()
        observed = [
            detector.anomalies(row) for row in (calibration, sinking, wedged)
        ]
        self.assertEqual(
            [[item["type"] for item in values] for values in observed],
            [[], ["sank_into_visible_floor"], ["player_stuck_in_geometry"]],
        )

    def test_all_detectors_require_both_capabilities(self):
        wall = entity("wall", "wall", [0, 1, 0], [0.5, 2, 2], solid=True)
        before = observation([-1.5, 1, 0], entities=[wall], capabilities=False)
        after = observation([1.5, 1, 0], entities=[wall], capabilities=False)
        self.assertEqual(
            continuous_anomalies(transition(1, "east", before, after)), []
        )

    def test_numeric_tolerance_preserves_json_boolean_distinction(self):
        self.assertTrue(continuous_observation_equal(
            {"position": [1.0, 2.0]}, {"position": [1.04, 1.96]}
        ))
        self.assertTrue(continuous_observation_equal(1.0, 1.05))
        self.assertFalse(continuous_observation_equal(True, 1))

    def test_incremental_detector_matches_full_session_calibration(self):
        first = transition(
            1,
            "east",
            observation([-3.0, 1.0, 0.0]),
            observation([-1.45, 1.0, 0.0], tick=1),
        )
        ghost = entity(
            "door", "door", [0.0, 1.25, 0.0], [0.5, 2.5, 2.0],
            solid=False, open=True,
        )
        second = transition(
            2,
            "east",
            observation([-1.45, 1.0, 0.0], entities=[ghost], tick=1),
            observation([-1.44, 1.0, 0.0], entities=[ghost], tick=2),
        )
        detector = ContinuousSessionDetector()
        prior = []
        for row in (first, second):
            self.assertEqual(
                detector.anomalies(row), continuous_anomalies(row, prior)
            )
            prior.append(row)


class ContinuousReportingTests(unittest.TestCase):
    def test_titles_evaluation_and_reproduction_for_wall_tunnel(self):
        game = TunnelGame(defect="wall-tunnel")
        before = game.observe()
        after = game.act("east")
        records = recording(
            before,
            [transition(1, "east", before, after)],
            defect="wall-tunnel",
        )

        issues = extract_issues(records)
        result = evaluate_recording(
            records, verify_reproduction=False, game_factory=tunnel_factory
        )
        reproduced = reproduce_issue(records, 1, game_factory=tunnel_factory)

        self.assertEqual(len(issues), 1)
        self.assertEqual(issues[0]["title"], "Player passed through a solid wall")
        self.assertEqual(result["detection"], "hit")
        self.assertTrue(reproduced["reproduced"])
        self.assertEqual(reproduced["actions"], ["east"])

    def test_repeated_continuous_sightings_merge_into_one_issue(self):
        wall = entity(
            "wall", "wall", [0.0, 1.0, 0.0], [0.5, 2.0, 2.0], solid=True
        )
        first_before = observation([-1.5, 1.0, 0.0], entities=[wall])
        first_after = observation([1.5, 1.0, 0.0], entities=[wall], tick=1)
        second_before = observation([1.5, 1.0, 0.0], entities=[wall], tick=1)
        second_after = observation([-1.5, 1.0, 0.0], entities=[wall], tick=2)
        records = recording(first_before, [
            transition(1, "east", first_before, first_after),
            transition(2, "west", second_before, second_after),
        ])

        issues = extract_issues(records)

        self.assertEqual(len(issues), 1)
        self.assertEqual(issues[0]["type"], "passed_through_solid")
        self.assertEqual(issues[0]["evidence_steps"], [1, 2])

    def test_block_fall_and_key_defect_mappings_use_visible_evidence(self):
        free = transition(
            1,
            "east",
            observation([-3.0, 1.0, 0.0]),
            observation([-1.45, 1.0, 0.0], tick=1),
        )
        ghost = entity(
            "door", "door", [0.0, 1.25, 0.0], [0.5, 2.5, 2.0],
            solid=False, open=True,
        )
        blocked_before = observation([-1.45, 1.0, 0.0], entities=[ghost], tick=1)
        blocked_after = observation([-1.45, 1.0, 0.0], entities=[ghost], tick=2)
        blocked_records = recording(
            free["before"],
            [free, transition(2, "east", blocked_before, blocked_after)],
            defect="door-ghost-collider",
        )

        floor = entity(
            "floor", "floor", [0.0, -0.125, 0.0], [2.0, 0.25, 2.0], solid=True
        )
        fall_before = observation([0.0, 1.0, 0.0], entities=[floor])
        fall_after = observation(
            [-4.0, 1.0, -4.0], entities=[floor], events=["You fell."], tick=1
        )
        fall_records = recording(
            fall_before,
            [transition(1, "north", fall_before, fall_after)],
            defect="floor-gap",
        )

        door = entity(
            "door", "door", [0, 1.25, 0], [0.5, 2.5, 2], solid=False, open=True
        )
        key_before = observation([0, 1, 0], entities=[door], inventory=["key"])
        key_after = observation(
            [0, 1, 0],
            entities=[door],
            inventory=["key"],
            events=["Unlocked the door and used the key."],
            tick=1,
        )
        key_records = recording(
            key_before,
            [transition(1, "interact", key_before, key_after)],
            defect="key-not-consumed",
        )

        cases = (
            (blocked_records, "Movement blocked with no visible obstacle"),
            (fall_records, "Player fell through a visible floor tile"),
            (key_records, "Key remained in inventory after feedback said it was used"),
        )
        for records, title in cases:
            with self.subTest(title=title):
                issues = extract_issues(records)
                result = evaluate_recording(records, verify_reproduction=False)
                self.assertIn(title, [issue["title"] for issue in issues])
                self.assertEqual(result["detection"], "hit")
                self.assertEqual(result["false_positive_count"], 0)

    def test_sink_and_stuck_both_map_to_floor_gap(self):
        floor = entity(
            "floor", "floor", [0.0, -0.125, 0.0], [2.0, 0.25, 2.0],
            solid=True,
        )
        initial = observation([-1.0, 1.0, 0.0], entities=[floor])
        normal = observation([0.0, 1.0, 0.0], entities=[floor], tick=1)
        sunk = observation(
            [0.8, 0.1, 0.0],
            entities=[floor],
            tick=2,
            grounded=False,
        )
        wedged = observation(
            [0.8, 0.1, 0.0],
            entities=[floor],
            tick=3,
            grounded=False,
        )
        records = recording(initial, [
            transition(1, "east", initial, normal),
            transition(2, "east", normal, sunk),
            transition(3, "north", sunk, wedged),
        ], defect="floor-gap")

        issues = extract_issues(records)
        result = evaluate_recording(records, verify_reproduction=False)

        self.assertEqual(
            {issue["type"] for issue in issues},
            {"sank_into_visible_floor", "player_stuck_in_geometry"},
        )
        self.assertEqual(result["detection"], "hit")
        self.assertEqual(result["false_positive_count"], 0)

    def test_unity_floor_gap_recordings_reclassify_wedge_at_first_sink(self):
        root = Path(__file__).resolve().parents[1]
        fixtures = [root / "diagnostics" / "unity-floor-gap" / f"floor-gap-seed{seed}.jsonl"
                    for seed in (1, 5)]
        if not all(path.is_file() for path in fixtures):
            self.skipTest("Optional historical Unity diagnostic recordings are unavailable.")
        for seed, first_sink, minimized_count in ((1, 223, 11), (5, 138, 5)):
            with self.subTest(seed=seed):
                path = (
                    root / "diagnostics" / "unity-floor-gap"
                    / f"floor-gap-seed{seed}.jsonl"
                )
                issues = extract_issues(path)
                result = evaluate_recording(path, verify_reproduction=False)
                sink = next(
                    issue for issue in issues
                    if issue["type"] == "sank_into_visible_floor"
                )
                reproduced = reproduce_issue(
                    path,
                    sink["number"],
                    game_factory=lambda replay_seed, defect: FakeContinuousGame(
                        replay_seed, defect
                    ),
                )

                self.assertEqual(sink["evidence_steps"][0], first_sink)
                self.assertNotIn(
                    "blocked_no_visible_obstacle",
                    {issue["type"] for issue in issues},
                )
                self.assertEqual(result["detection"], "hit")
                self.assertEqual(result["false_positive_count"], 0)
                self.assertTrue(reproduced["reproduced"])
                self.assertEqual(
                    reproduced["original_action_count"], first_sink
                )
                self.assertEqual(
                    reproduced["minimized_action_count"], minimized_count
                )
                self.assertEqual(
                    reproduced["minimization_strategy"],
                    "visible_target_search",
                )

    def test_collected_generic_key_is_not_treated_as_still_in_world(self):
        collected_key = {
            "id": "key",
            "kind": "key",
            "size": [0.7, 0.7, 0.7],
            "state": {"present": False},
        }
        before = observation(
            [0, 1, 0], entities=[collected_key], inventory=["key"]
        )
        before["objects"] = {"key": {"position": None, "present": False}}
        after = deepcopy(before)
        after["tick"] = 1
        records = recording(
            before, [transition(1, "wait", before, after)]
        )

        self.assertEqual(extract_issues(records), [])

    def test_reset_and_nondeterminism_apply_float_tolerance_to_continuous_state(self):
        initial = observation([0.0, 1.0, 0.0])
        near_reset = observation([0.04, 1.03, -0.02])
        reset_row = transition(
            1, "reset", initial, near_reset, episode=1, kind="sweep_reset"
        )
        self.assertEqual(extract_issues(recording(initial, [reset_row])), [])

        leaked = observation([0.06, 1.0, 0.0], inventory=["key"])
        leaked_row = transition(
            1, "reset", initial, leaked, episode=1, kind="sweep_reset"
        )
        leaked_records = recording(initial, [leaked_row], defect="respawn-leak")
        leaked_issues = extract_issues(leaked_records)
        self.assertEqual([issue["type"] for issue in leaked_issues], ["reset_inconsistency"])
        self.assertEqual(
            evaluate_recording(leaked_records, verify_reproduction=False)["detection"],
            "hit",
        )

        first_after = observation([1.0, 1.0, 0.0], tick=1)
        first = transition(1, "east", initial, first_after, episode=0)
        reset = transition(
            2, "reset", first_after, near_reset, episode=1, kind="reset"
        )
        repeated_after = observation([1.04, 1.0, 0.0], tick=1)
        repeated = transition(3, "east", near_reset, repeated_after, episode=1)
        self.assertEqual(
            extract_issues(recording(initial, [first, reset, repeated])), []
        )

        different_after = observation([1.11, 1.0, 0.0], tick=1)
        different = transition(3, "east", near_reset, different_after, episode=1)
        issues = extract_issues(recording(initial, [first, reset, different]))
        self.assertIn(
            "cross_episode_nondeterminism", [issue["type"] for issue in issues]
        )


if __name__ == "__main__":
    unittest.main()
