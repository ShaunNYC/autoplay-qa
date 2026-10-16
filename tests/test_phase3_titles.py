"""Specific, visible-evidence titles for Phase-3 QA issues."""

from copy import deepcopy
import unittest

from autoplay_qa.game import KeyGateGame
from autoplay_qa.reporting import extract_issues


def recording_for(defect: str, actions: list[str]) -> list[dict]:
    game = KeyGateGame(0, defect=defect)
    initial = deepcopy(game.observe())
    records = [{
        "type": "session",
        "schema_version": 2,
        "game_version": game.version,
        "seed": 0,
        "focus": "qa",
        "initial_observation": initial,
        # This deliberately supplies no injected-defect metadata: the report
        # title must come from executed, player-visible evidence alone.
        "evaluation": {"injected_defect": None},
    }]
    episode = 0
    for step, action in enumerate(actions, 1):
        before = deepcopy(game.observe())
        episode_before = episode
        if action == "reset":
            after = deepcopy(game.reset())
            episode += 1
            record_type = "reset"
        else:
            after = deepcopy(game.act(action))
            record_type = "step"
        records.append({
            "type": record_type,
            "step": step,
            "episode": episode,
            "episode_before": episode_before,
            "episode_after": episode,
            "before": before,
            "decision": {
                "action": action,
                "reason": "title fixture",
                "expected_position": None,
                "notes": [],
            },
            "after": after,
        })
    return records


class SpecificIssueTitleTests(unittest.TestCase):
    def test_titles_come_from_visible_anomaly_details(self):
        cases = {
            "gate-keeps-key": (
                ["south", "south", "east", "east", "interact"],
                "Key remained in inventory after feedback said it was used",
            ),
            "exit-no-win": (
                [
                    "south", "south", "east", "east", "interact",
                    "east", "east", "east", "east",
                ],
                "Entering the exit tile did not end the game",
            ),
            "phantom-block": (
                ["east", "east", "south"],
                "Movement into a floor tile was blocked",
            ),
            "wall-clip": (
                ["east", "east", "east"],
                "Movement into a wall tile succeeded",
            ),
            "reset-leak": (
                ["south", "south", "reset"],
                "Reset did not restore the initial state",
            ),
        }

        for defect, (actions, expected_title) in cases.items():
            with self.subTest(defect=defect):
                issues = extract_issues(recording_for(defect, actions))

                self.assertEqual(len(issues), 1)
                self.assertEqual(issues[0]["title"], expected_title)


if __name__ == "__main__":
    unittest.main()
