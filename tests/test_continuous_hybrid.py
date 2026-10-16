"""Continuous visible-solid sweep and lean prompt coverage."""

from __future__ import annotations

import copy
import json
import math
import unittest

from autoplay_qa.hybrid import compact_sweep_summary, run_collision_sweep
from autoplay_qa.lean import (
    LEAN_CONTINUOUS_PROMPT_MAX_BYTES,
    cross_episode_pairs,
    lean_messages,
    step_fact,
)


class _ContinuousGame:
    capabilities = {
        "grid": False,
        "legend": False,
        "entities": True,
        "reset": True,
        "seed": True,
        "continuous": True,
        "solids_visible": True,
    }
    controls = ["north", "south", "east", "west", "interact", "wait"]

    def __init__(self, defect: str = "none"):
        self.defect = defect
        self.reset_calls = 0
        self.interactions = 0
        self._reset_state()

    def _reset_state(self):
        self.tick = 0
        self.position = [-2.0, 1.0, -2.0]
        self.inventory = []
        self.key_present = True
        self.door_open = False
        self.events = ["Game started."]

    def reset(self):
        self.reset_calls += 1
        self._reset_state()
        return self

    def _entities(self):
        entities = []
        for z in (-2.0, 0.0, 2.0):
            for x in (-2.0, 0.0, 2.0):
                entities.append({
                    "id": f"floor-{x:g}-{z:g}",
                    "kind": "floor",
                    "position": [x, -0.125, z],
                    "size": [2.0, 0.25, 2.0],
                    "state": {"solid": True},
                })
        entities.extend([
            {
                "id": "wall-north", "kind": "wall",
                "position": [0.0, 1.25, 3.25], "size": [6.5, 2.5, 0.5],
                "state": {"solid": True},
            },
            {
                "id": "wall-south", "kind": "wall",
                "position": [0.0, 1.25, -3.25], "size": [6.5, 2.5, 0.5],
                "state": {"solid": True},
            },
            {
                "id": "wall-east", "kind": "wall",
                "position": [3.25, 1.25, 0.0], "size": [0.5, 2.5, 6.5],
                "state": {"solid": True},
            },
            {
                "id": "wall-west", "kind": "wall",
                "position": [-3.25, 1.25, 0.0], "size": [0.5, 2.5, 6.5],
                "state": {"solid": True},
            },
            {
                "id": "door", "kind": "door",
                "position": [0.0, 1.25, 0.0], "size": [0.5, 2.5, 1.5],
                "state": {"open": self.door_open, "solid": not self.door_open},
            },
            {
                "id": "key", "kind": "key",
                "position": [-2.0, 0.5, 0.0] if self.key_present else None,
                "size": [0.7, 0.7, 0.7],
                "state": {"present": self.key_present},
            },
            {
                "id": "exit", "kind": "exit",
                "position": [2.0, 0.1, 2.0], "size": [1.0, 0.2, 1.0],
                "state": {"reached": False},
            },
        ])
        return entities

    def observe(self):
        return {
            "game": "continuous-fixture-v1",
            "tick": self.tick,
            "status": "playing",
            "player": {
                "position": list(self.position),
                "velocity": [0.0, 0.0, 0.0],
                "grounded": True,
                "inventory": list(self.inventory),
            },
            "entities": self._entities(),
            "events": list(self.events),
            "actions": list(self.controls),
            "_capabilities": dict(self.capabilities),
        }

    def act(self, control):
        self.tick += 1
        self.events = []
        directions = {
            "north": (0.0, 1.55),
            "south": (0.0, -1.55),
            "east": (1.55, 0.0),
            "west": (-1.55, 0.0),
        }
        if control in directions:
            dx, dz = directions[control]
            candidate = [self.position[0] + dx, 1.0, self.position[2] + dz]
            if abs(candidate[0]) <= 2.7 and abs(candidate[2]) <= 2.7:
                if self.defect == "floor-gap" and (
                    abs(candidate[0]) <= 1.0 and 1.0 <= candidate[2] <= 3.0
                ):
                    self.position = [-2.0, 1.0, -2.0]
                    self.events = ["You fell."]
                else:
                    self.position = candidate
            else:
                self.events = ["Blocked by a wall."]
            if self.key_present and math.dist(
                (self.position[0], self.position[2]), (-2.0, 0.0)
            ) < 0.8:
                self.key_present = False
                self.inventory = ["key"]
                self.events.append("Picked up the key.")
        elif control == "interact":
            self.interactions += 1
            if (
                "key" in self.inventory
                and math.dist((self.position[0], self.position[2]), (0.0, 0.0)) <= 1.8
            ):
                self.door_open = True
                self.inventory = []
                self.events = ["Unlocked the door and used the key."]
            else:
                self.events = ["There is nothing to interact with here."]
        return self.observe()


def _unity_sized_observation():
    game = _ContinuousGame()
    observation = game.observe()
    floors = []
    for z_index in range(7):
        for x_index in range(7):
            floors.append({
                "id": f"floor-{x_index}-{z_index}",
                "kind": "floor",
                "position": [-6.0 + x_index * 2.0, -0.125, -6.0 + z_index * 2.0],
                "size": [2.0, 0.25, 2.0],
                "state": {"solid": True},
            })
    walls = [
        {
            "id": f"wall-{index}", "kind": "wall",
            "position": [0.0, 1.25, -7.25 + index * 1.5],
            "size": [0.5, 2.5, 2.0], "state": {"solid": True},
        }
        for index in range(10)
    ]
    dynamic = [
        {
            "id": "door", "kind": "door", "position": [0.0, 1.25, 2.0],
            "size": [0.5, 2.5, 2.0], "state": {"open": False, "solid": True},
        },
        {
            "id": "key", "kind": "key", "position": [-4.0, 0.5, 2.0],
            "size": [0.7, 0.7, 0.7], "state": {"present": True},
        },
        {
            "id": "exit", "kind": "exit", "position": [4.0, 0.1, 2.0],
            "size": [1.5, 0.2, 1.5], "state": {"reached": False},
        },
    ]
    observation["entities"] = [*floors, *walls, *dynamic]
    observation["player"]["position"] = [-4.0, 1.05, -4.0]
    observation["private_ground_truth"] = "SECRET_DEFECT_NAME"
    return observation


def _step(before, after, step, *, episode=0, action="north"):
    return {
        "type": "step",
        "step": step,
        "episode": episode,
        "episode_before": episode,
        "episode_after": episode,
        "before": copy.deepcopy(before),
        "after": copy.deepcopy(after),
        "decision": {
            "action": action,
            "reason": "continuous fixture",
            "expected_position": None,
            "notes": [],
        },
        "qa_mode": True,
    }


class ContinuousSweepTests(unittest.TestCase):
    def test_sweep_samples_floor_cells_directions_and_nearby_objects(self):
        game = _ContinuousGame()

        result = run_collision_sweep(game, max_actions=120)

        self.assertTrue(result.summary["continuous"])
        self.assertLessEqual(result.summary["actions"], 120)
        self.assertGreater(result.summary["visited_positions"], 1)
        self.assertGreater(result.summary["movement_attempts"], 4)
        self.assertGreater(result.summary["interaction_attempts"], 0)
        self.assertGreater(game.interactions, 0)
        self.assertTrue(result.summary["visited_tile_cells"])
        self.assertTrue(any(
            item["controls"] for item in result.summary["tried_directions"]
        ))
        self.assertTrue(all(
            record["source"] == "sweep" for record in result.records
        ))

    def test_fall_causes_an_explicit_bounded_sweep_reset(self):
        game = _ContinuousGame("floor-gap")

        result = run_collision_sweep(game, max_actions=120)

        self.assertLessEqual(result.summary["actions"], 120)
        self.assertGreater(result.summary["resets"], 0)
        self.assertGreater(game.reset_calls, 0)
        self.assertIn("sweep_reset", {record["type"] for record in result.records})
        self.assertIn(
            "fell_through_visible_floor",
            {item.get("type") for item in result.summary["anomalies"]},
        )


class ContinuousPromptTests(unittest.TestCase):
    def test_prompt_compacts_static_geometry_and_stays_below_twelve_kb(self):
        observation = _unity_sized_observation()
        records = []
        for step in range(1, 9):
            before = copy.deepcopy(observation)
            after = copy.deepcopy(observation)
            after["tick"] = step
            after["player"]["position"][0] += 0.2 * step
            records.append(_step(before, after, step, action="wait"))
        sweep = {
            "version": 1,
            "source": "sweep",
            "status": "action_limit",
            "actions": 100,
            "action_budget": 100,
            "movement_attempts": 90,
            "blocked_attempts": 10,
            "successful_moves": 80,
            "visited_positions": 49,
            "continuous": True,
            "move_distance": 1.55,
            "visible_floor_cells": 49,
            "visited_tile_cells": [[x, z] for z in range(7) for x in range(7)],
            "tried_directions": [
                {"cell": [x, z], "controls": ["north", "south", "east", "west"]}
                for z in range(7) for x in range(7)
            ],
            "interaction_attempts": 4,
            "resets": 1,
            "anomaly_count": 0,
            "anomalies": [],
        }

        messages = lean_messages(
            observation,
            0,
            {},
            records,
            {"actions": 24, "requests": 28, "seconds": 300},
            focus="qa",
            sweep_summary=sweep,
        )
        context = json.loads(messages[1]["content"])
        encoded = len(json.dumps(messages, separators=(",", ":")).encode("utf-8"))

        self.assertLess(encoded, LEAN_CONTINUOUS_PROMPT_MAX_BYTES)
        self.assertEqual(context["observation"]["static_solids"]["count"], 59)
        self.assertEqual(
            {entity["id"] for entity in context["observation"]["dynamic_entities"]},
            {"door", "key", "exit"},
        )
        self.assertEqual(
            context["exploration_coverage"]["coordinate_space"],
            "continuous_xz_floor_cells",
        )
        self.assertTrue(context["sweep_coverage"]["visited_tile_cells"])
        self.assertNotIn("SECRET_DEFECT_NAME", json.dumps(messages))
        self.assertNotIn("private_ground_truth", json.dumps(messages))
        self.assertNotIn("entities", context["factual_log"][0]["before"])

    def test_compact_sweep_rows_are_whitelisted(self):
        summary = {
            "continuous": True,
            "visited_tile_cells": [[0.0, 0.0]],
            "tried_directions": [{
                "cell": [0.0, 0.0],
                "controls": ["north"],
                "private": "SECRET",
            }],
            "evaluation": {"defect": "SECRET"},
            "anomalies": [],
        }

        compact = compact_sweep_summary(summary)

        self.assertEqual(compact["tried_direction_fields"], ["cell", "controls"])
        self.assertEqual(compact["tried_directions"], [[[0.0, 0.0], ["north"]]])
        self.assertNotIn("SECRET", json.dumps(compact))


class ContinuousToleranceTests(unittest.TestCase):
    def test_reset_and_cross_episode_comparisons_tolerate_small_float_jitter(self):
        initial = _ContinuousGame().observe()
        reset_after = copy.deepcopy(initial)
        reset_after["player"]["position"][0] += 0.04
        reset_record = {
            "type": "reset",
            "step": 1,
            "episode_before": 0,
            "episode_after": 1,
            "before": copy.deepcopy(initial),
            "after": reset_after,
            "decision": {
                "action": "reset", "reason": "fixture",
                "expected_position": None, "notes": [],
            },
            "qa_mode": True,
        }

        self.assertEqual(step_fact(reset_record)["anomalies"], [])

        first_after = copy.deepcopy(initial)
        first_after["player"]["position"][2] += 1.55
        repeated_before = copy.deepcopy(initial)
        repeated_before["player"]["position"][0] += 0.04
        repeated_after = copy.deepcopy(first_after)
        repeated_after["player"]["position"][0] += 0.04
        records = [
            _step(initial, first_after, 1, episode=0),
            _step(repeated_before, repeated_after, 2, episode=1),
        ]

        self.assertEqual(cross_episode_pairs(records)[0]["result"], "same")


if __name__ == "__main__":
    unittest.main()
