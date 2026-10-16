"""Rule-free position/control coverage supplied to lean QA requests."""

import copy
import hashlib
import json
import unittest

from autoplay_qa.game import ACTIONS, DEFECTS, KeyGateGame
from autoplay_qa.lean import (
    LEAN_COVERAGE_NEAREST_LIMIT,
    LEAN_QA_CHECKLIST,
    lean_messages,
    lean_schema,
    position_control_coverage,
)
from autoplay_qa.session import _LeanReplay


def _snapshot(position):
    observation = KeyGateGame(0).observe()
    observation["player"]["position"] = list(position)
    return observation


def _record(before_position, action, after_position=None, *, step=1):
    return {
        "type": "step",
        "step": step,
        "episode": 0,
        "episode_before": 0,
        "episode_after": 0,
        "before": _snapshot(before_position),
        "after": _snapshot(after_position or before_position),
        "decision": {
            "action": action,
            "reason": "coverage fixture",
            "expected_position": None,
            "notes": [],
        },
        "qa_mode": True,
    }


class PositionControlCoverageTests(unittest.TestCase):
    def test_reports_distinct_attempted_pairs_for_every_visited_position(self):
        records = [
            _record((1, 1), "south", (1, 2), step=1),
            _record((1, 2), "south", (1, 3), step=2),
            _record((1, 3), "north", (1, 2), step=3),
            _record((1, 2), "south", (1, 3), step=4),
            _record((1, 3), "reset", (1, 1), step=5),
        ]

        coverage = position_control_coverage(_snapshot((1, 1)), records)

        self.assertEqual(coverage["controls"], list(ACTIONS))
        self.assertEqual(coverage["visited_positions"], [
            {"position": [1, 1], "attempted": ["south"]},
            {"position": [1, 2], "attempted": ["south"]},
            {"position": [1, 3], "attempted": ["north"]},
        ])
        # Repeating south at [1,2] is still one pair. Session controls are not
        # game-control exploration pairs.
        self.assertEqual(
            coverage["position_control_pair_counts"],
            {"attempted": 3, "unattempted": 15},
        )
        nearest = coverage["nearest_visited_positions_with_unattempted_controls"]
        self.assertEqual([item["position"] for item in nearest], [[1, 1], [1, 2], [1, 3]])
        self.assertEqual([item["distance"] for item in nearest], [0, 1, 2])
        self.assertNotIn("south", nearest[0]["unattempted"])
        self.assertIn("east", nearest[0]["unattempted"])

    def test_nearest_gap_list_is_capped_and_sorted_from_current_position(self):
        records = [
            _record((x, 0), "north", (x + 1, 0), step=x + 1)
            for x in range(10)
        ]

        coverage = position_control_coverage(_snapshot((10, 0)), records)
        nearest = coverage["nearest_visited_positions_with_unattempted_controls"]

        self.assertEqual(len(coverage["visited_positions"]), 11)
        self.assertEqual(len(nearest), LEAN_COVERAGE_NEAREST_LIMIT)
        self.assertEqual(
            [item["position"] for item in nearest],
            [[10, 0], [9, 0], [8, 0], [7, 0], [6, 0], [5, 0], [4, 0], [3, 0]],
        )
        self.assertLess(
            len(json.dumps(coverage, separators=(",", ":")).encode("utf-8")),
            2_500,
        )

    def test_qa_request_contains_coverage_and_exact_checklist_instruction(self):
        observation = KeyGateGame(0).observe()
        context = json.loads(lean_messages(
            observation,
            0,
            {},
            [],
            {"actions": 24, "requests": 28, "seconds": 300},
            focus="qa",
        )[1]["content"])

        self.assertEqual(
            context["exploration_coverage"]["position_control_pair_counts"],
            {"attempted": 0, "unattempted": len(ACTIONS)},
        )
        self.assertEqual(
            context["exploration_coverage"]["visited_positions"],
            [{"position": [1, 1], "attempted": []}],
        )
        self.assertIn(
            "Prefer unattempted controls at visited positions before finishing.",
            context["qa_checklist"].splitlines(),
        )
        self.assertIn(
            "Prefer unattempted controls at visited positions before finishing.",
            LEAN_QA_CHECKLIST.splitlines(),
        )

        mechanics_context = json.loads(lean_messages(
            observation,
            0,
            {},
            [],
            {"actions": 24, "requests": 28, "seconds": 300},
            focus="mechanics",
        )[1]["content"])
        self.assertNotIn("exploration_coverage", mechanics_context)
        self.assertNotIn("qa_checklist", mechanics_context)

    def test_requests_exclude_rules_defects_and_hidden_untried_outcomes(self):
        game = KeyGateGame(0, "wall-clip")
        before = game.observe()
        developer_rules = list(before["rules"])
        after = game.act("wait")
        record = _record(tuple(before["player"]["position"]), "wait", tuple(after["player"]["position"]))
        record["before"] = copy.deepcopy(before)
        record["after"] = copy.deepcopy(after)

        hidden_values = (
            "SECRET_DEVELOPER_GOAL",
            "SECRET_PASSABILITY_TABLE",
            "SECRET_UNTRIED_NORTH_WOULD_SUCCEED",
            "SECRET_UNTRIED_EAST_WOULD_BLOCK",
        )
        current = copy.deepcopy(after)
        current["goal"] = hidden_values[0]
        current["passability"] = hidden_values[1]
        current["untried_outcomes"] = [hidden_values[2], hidden_values[3]]
        record["evaluation"] = {
            "defect": "wall-clip",
            "untried_outcomes": [hidden_values[2], hidden_values[3]],
        }
        record["before"]["rules"] = [*developer_rules, "SECRET_RULE_BEFORE"]
        record["after"]["rules"] = [*developer_rules, "SECRET_RULE_AFTER"]

        messages = lean_messages(
            current,
            0,
            {},
            [record],
            {"actions": 23, "requests": 27, "seconds": 290},
            focus="qa",
        )
        serialized = json.dumps(messages)
        context = json.loads(messages[1]["content"])
        coverage_text = json.dumps(context["exploration_coverage"])

        for hidden in (*hidden_values, "SECRET_RULE_BEFORE", "SECRET_RULE_AFTER"):
            self.assertNotIn(hidden, serialized)
        for rule in developer_rules:
            self.assertNotIn(rule, serialized)
        for name, description in DEFECTS.items():
            self.assertNotIn(name, serialized)
            self.assertNotIn(description, serialized)
        for forbidden in ("map", "tile", "wall", "floor", "passab", "outcome", "feedback", "result"):
            self.assertNotIn(forbidden, coverage_text.lower())
        self.assertEqual(
            context["exploration_coverage"]["visited_positions"],
            [{"position": [1, 1], "attempted": ["wait"]}],
        )

    def test_response_schema_bytes_are_unchanged(self):
        encoded = json.dumps(lean_schema(), separators=(",", ":")).encode("utf-8")

        self.assertEqual(len(encoded), 867)
        self.assertEqual(
            hashlib.sha256(encoded).hexdigest(),
            "71d0712cd5e40b952acda97a49df7c83cb7fae4a593f1af0957520f27aa67713",
        )

    def test_replay_recomputes_new_coverage_without_affecting_old_requests(self):
        observation = KeyGateGame(0).observe()
        messages = lean_messages(
            observation,
            0,
            {},
            [],
            {"actions": 24, "requests": 28, "seconds": 300},
            focus="qa",
        )
        context = json.loads(messages[1]["content"])
        request = {"response_schema": lean_schema(context["observation"]["controls"])}

        _LeanReplay({"limits": {"history_size": 8}}).request(
            context, observation, request,
        )

        tampered = copy.deepcopy(context)
        tampered["exploration_coverage"]["position_control_pair_counts"]["attempted"] = 1
        with self.assertRaisesRegex(ValueError, "exploration_coverage"):
            _LeanReplay({"limits": {"history_size": 8}}).request(
                tampered, observation, request,
            )


if __name__ == "__main__":
    unittest.main()
