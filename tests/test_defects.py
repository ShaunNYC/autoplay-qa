import json
import unittest

from autoplay_qa.game import (
    DEFECTS,
    DIRECTIONS,
    LAYOUT,
    KeyGateGame,
    defect_evaluation_metadata,
    evaluate_transition,
)


def approach(target, game):
    """Place a test game on a normal tile one move away from target."""
    for action, (dx, dy) in DIRECTIONS.items():
        origin = (target[0] - dx, target[1] - dy)
        x, y = origin
        if (
            0 <= y < len(LAYOUT)
            and 0 <= x < len(LAYOUT[0])
            and LAYOUT[y][x] != "#"
            and origin != game.door
            and origin != game.key_position
        ):
            game.position = origin
            return action, origin
    raise AssertionError(f"No ordinary approach tile for {target!r}")


class DefectTests(unittest.TestCase):
    def test_defect_names_are_validated_and_metadata_is_session_shaped(self):
        self.assertEqual(KeyGateGame().defect, "none")
        self.assertEqual(KeyGateGame(defect=None).defect, "none")
        with self.assertRaisesRegex(ValueError, "Unknown injected defect"):
            KeyGateGame(defect="not-a-defect")
        self.assertEqual(
            defect_evaluation_metadata("none"),
            {"evaluation": {"injected_defect": None}},
        )
        for name, description in DEFECTS.items():
            self.assertEqual(
                defect_evaluation_metadata(name),
                {
                    "evaluation": {
                        "injected_defect": {"name": name, "description": description}
                    }
                },
            )

    def test_defect_ground_truth_is_absent_from_observations(self):
        clean = KeyGateGame(11).observe()
        for name, description in DEFECTS.items():
            observation = KeyGateGame(11, name).observe()
            self.assertEqual(observation, clean)
            payload = json.dumps(observation)
            self.assertNotIn(name, payload)
            self.assertNotIn(description, payload)
            self.assertEqual(observation["legend"], clean["legend"])
            self.assertEqual(observation["rules"], clean["rules"])

    def test_gate_keeps_key_after_unlocking(self):
        game = KeyGateGame(0, "gate-keeps-key")
        game.act("south")
        game.act("south")
        game.act("east")
        game.act("east")
        before = game.observe()
        after = game.act("interact")
        self.assertTrue(after["objects"]["gate"]["open"])
        self.assertEqual(after["player"]["inventory"], ["key"])
        self.assertEqual(after["events"], ["Unlocked the gate and used the key."])
        self.assertEqual(game.evaluate_transition(before, "interact", after)["contract_violations"], [])

    def test_wall_clip_selects_one_deterministic_interior_wall(self):
        positions = []
        for seed in range(8):
            first = KeyGateGame(seed, "wall-clip")
            second = KeyGateGame(seed, "wall-clip")
            self.assertEqual(first._wall_clip_position, second._wall_clip_position)
            x, y = first._wall_clip_position
            self.assertEqual(LAYOUT[y][x], "#")
            self.assertGreater(x, 0)
            self.assertGreater(y, 0)
            self.assertLess(x, len(LAYOUT[0]) - 1)
            self.assertLess(y, len(LAYOUT) - 1)
            positions.append(first._wall_clip_position)
        self.assertGreater(len(set(positions)), 1)

    def test_wall_clip_moves_into_only_the_selected_wall_without_contract_failure(self):
        game = KeyGateGame(0, "wall-clip")
        target = game._wall_clip_position
        action, origin = approach(target, game)
        before = game.observe()
        after = game.act(action)
        self.assertEqual(after["player"]["position"], list(target))
        self.assertEqual(game.evaluate_transition(before, action, after)["contract_violations"], [])
        self.assertTrue(evaluate_transition(before, action, after)["contract_violations"])

        control = KeyGateGame(0)
        control.position = origin
        blocked = control.act(action)
        self.assertEqual(blocked["player"]["position"], list(origin))

    def test_reset_leak_carries_key_world_and_inventory_state(self):
        before_pickup = KeyGateGame(0, "reset-leak")
        initial = before_pickup.observe()
        before_pickup.act("east")
        before_pickup.reset()
        self.assertEqual(before_pickup.observe(), initial)

        game = KeyGateGame(0, "reset-leak")
        game.act("south")
        game.act("south")
        self.assertIsNone(game.key_position)
        self.assertTrue(game.has_key)
        identity = id(game)
        returned = game.reset()
        self.assertEqual(id(returned), identity)
        self.assertEqual(returned.observe()["player"], {"position": [1, 1], "inventory": ["key"]})
        self.assertIsNone(returned.observe()["objects"]["key"])
        self.assertEqual(returned.tick, 0)
        self.assertFalse(returned.door_open)
        self.assertFalse(returned.won)

        clean = KeyGateGame(0)
        clean.act("south")
        clean.act("south")
        clean.reset()
        self.assertEqual(clean.observe(), KeyGateGame(0).observe())

    def test_exit_no_win_requires_leaving_and_entering_again(self):
        game = KeyGateGame(0, "exit-no-win")
        game.position = (6, 3)
        first = game.act("east")
        self.assertEqual(first["player"]["position"], [7, 3])
        self.assertEqual(first["status"], "playing")
        self.assertNotIn("won", " ".join(first["events"]).lower())
        game.act("west")
        second = game.act("east")
        self.assertEqual(second["status"], "won")
        self.assertIn("Reached the exit. You won!", second["events"])

    def test_phantom_block_selects_one_deterministic_floor_and_uses_normal_feedback(self):
        positions = []
        for seed in range(8):
            first = KeyGateGame(seed, "phantom-block")
            second = KeyGateGame(seed, "phantom-block")
            self.assertEqual(first._phantom_block_position, second._phantom_block_position)
            x, y = first._phantom_block_position
            self.assertEqual(LAYOUT[y][x], ".")
            positions.append(first._phantom_block_position)
        self.assertGreater(len(set(positions)), 1)

        game = KeyGateGame(0, "phantom-block")
        target = game._phantom_block_position
        action, origin = approach(target, game)
        blocked = game.act(action)
        self.assertEqual(blocked["player"]["position"], list(origin))
        self.assertEqual(blocked["events"], ["Movement blocked by a wall or locked gate."])

        control = KeyGateGame(0)
        control.position = origin
        moved = control.act(action)
        self.assertEqual(moved["player"]["position"], list(target))


if __name__ == "__main__":
    unittest.main()
