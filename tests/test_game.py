import copy
from collections import deque
import unittest

from autoplay_qa.game import ACTIONS, KeyGateGame, evaluate_transition


class GameTests(unittest.TestCase):
    def test_wall_does_not_move_player_but_advances_tick(self):
        game = KeyGateGame()
        before = game.observe()
        after = game.act("north")
        self.assertEqual(after["player"]["position"], [1, 1])
        self.assertEqual(after["tick"], 1)
        self.assertEqual(evaluate_transition(before, "north", after)["contract_violations"], [])

    def test_collect_key_unlock_and_win(self):
        game = KeyGateGame(0)
        for action in ["south", "south"]:
            game.act(action)
        self.assertEqual(game.observe()["player"]["inventory"], ["key"])
        game.act("east")
        game.act("east")
        blocked = game.act("east")
        self.assertEqual(blocked["player"]["position"], [3, 3])
        opened = game.act("interact")
        self.assertTrue(opened["objects"]["gate"]["open"])
        self.assertEqual(opened["player"]["inventory"], [])
        for _ in range(4):
            game.act("east")
        self.assertTrue(game.won)
        with self.assertRaises(ValueError):
            game.act("wait")

    def test_gate_requires_adjacency_and_key(self):
        game = KeyGateGame()
        game.act("interact")
        self.assertFalse(game.door_open)
        for action in ["east", "east", "south", "south"]:
            game.act(action)
        game.act("interact")
        self.assertFalse(game.door_open)
        self.assertIn("need the key", game.events[0])

    def test_invalid_action_does_not_advance_game(self):
        game = KeyGateGame()
        with self.assertRaises(ValueError):
            game.act("teleport")
        self.assertEqual(game.tick, 0)

    def test_observations_are_detached(self):
        game = KeyGateGame()
        observation = game.observe()
        observation["player"]["position"][0] = 999
        observation["events"].clear()
        self.assertEqual(game.observe()["player"]["position"], [1, 1])
        self.assertTrue(game.events)

    def test_checker_detects_unauthorized_gate_open_and_wrong_movement(self):
        game = KeyGateGame()
        before = game.observe()
        after = copy.deepcopy(before)
        after["tick"] = 1
        after["objects"]["gate"]["open"] = True
        self.assertTrue(evaluate_transition(before, "wait", after)["contract_violations"])
        after = game.act("south")
        self.assertTrue(evaluate_transition(before, "east", after)["contract_violations"])

    def test_each_spawn_has_a_solution_and_contracts_hold(self):
        # Exhaustive state-space search verifies game solvability. This is a test
        # oracle, not a runtime controller and never stands in for model inference.
        for seed in [0, 1, 5]:
            queue = deque([KeyGateGame(seed)])
            seen = set()
            found = False
            while queue:
                game = queue.popleft()
                state = (game.position, game.key_position, game.has_key, game.door_open)
                if state in seen:
                    continue
                seen.add(state)
                if game.won:
                    found = True
                    break
                for action in ACTIONS:
                    successor = copy.deepcopy(game)
                    before = successor.observe()
                    after = successor.act(action)
                    self.assertEqual(evaluate_transition(before, action, after)["contract_violations"], [])
                    queue.append(successor)
            self.assertTrue(found, f"Unsolvable game seed {seed}")


if __name__ == "__main__":
    unittest.main()

