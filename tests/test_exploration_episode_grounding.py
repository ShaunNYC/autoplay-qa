"""Actual episode transitions take precedence over unexecuted reset plans."""

import json
import unittest

from autoplay_qa.exploration import (
    Coverage, Hypothesis, exploration_messages, outcome_view,
)
from autoplay_qa.game import KeyGateGame


class EpisodeGroundingTests(unittest.TestCase):
    def test_reset_in_plan_prediction_and_note_cannot_change_actual_episode(self):
        game = KeyGateGame()
        before = game.observe()
        after = game.act("south")
        coverage = Coverage(before)
        coverage.record(before, "south", after, step=1, episode=0)
        history = [{"step": 1, "episode_before": 0, "episode_after": 0,
                    "experiment_id": "model-chosen", "action": "south",
                    "prediction": "After reset, movement repeats.",
                    "before": outcome_view(before), "after": outcome_view(after)}]
        active = {"id": "model-chosen", "goal": "Does a chosen effect repeat?",
                  "plan": "Reset and repeat the chosen control."}
        notes = {"h1": Hypothesis("h1", "The planned reset already happened.", "uncertain", (1,))}
        messages = exploration_messages(after, 0, active, notes, history, coverage,
                                        1, {"actions": 23}, focus="reset-retention")
        context = json.loads(messages[1]["content"])
        completed = context["completed_outcome_to_assess"]
        self.assertEqual(completed["action_that_already_ran"], "south")
        self.assertEqual((completed["episode_before"], completed["episode_after"]), (0, 0))
        self.assertEqual(completed["before_state"]["player"]["position"], [1, 1])
        self.assertEqual(completed["after_state"]["player"]["position"], [1, 2])
        self.assertEqual(context["coverage"]["reset_use_count"], 0)
        self.assertEqual(context["active_experiment"]["plan"], active["plan"])
        self.assertEqual(context["hypotheses"][0]["claim"], notes["h1"].claim)
        prompt = " ".join(messages[0]["content"].split())
        self.assertIn("Only action executes; plans/predictions are unexecuted intent", prompt)
        self.assertIn("Writing reset in a plan does not reset anything", prompt)
        self.assertIn("Episode 0 without a recorded reset means no reset occurred", prompt)
        self.assertIn("Different conditions do not refute conditioned repeatability", prompt)
        self.assertIn("cite control trials, not reset setup", prompt)

    def test_actual_reset_reports_recorded_episode_change_and_neutral_start(self):
        game = KeyGateGame()
        initial = game.observe()
        before = game.act("south")
        after = KeyGateGame().observe()
        coverage = Coverage(initial)
        coverage.record(initial, "south", before, step=1, episode=0)
        coverage.record(before, "reset", after, step=2, episode=0)
        history = [{"step": 2, "episode_before": 0, "episode_after": 1,
                    "experiment_id": "model-chosen", "action": "reset",
                    "prediction": "The seeded start is restored.",
                    "before": outcome_view(before), "after": outcome_view(after)}]
        messages = exploration_messages(after, 1, None, {}, history, coverage,
                                        2, {"actions": 22}, focus="reset-retention")
        context = json.loads(messages[1]["content"])
        completed = context["completed_outcome_to_assess"]
        self.assertEqual(completed["action_that_already_ran"], "reset")
        self.assertEqual((completed["episode_before"], completed["episode_after"]), (0, 1))
        self.assertEqual(context["episode"], 1)
        self.assertEqual(context["coverage"]["reset_use_count"], 1)
        self.assertEqual(completed["after_state"]["events"], ["Environment started."])
        self.assertEqual(completed["after_state"]["player"]["position"], [1, 1])

    def test_grounding_and_probe_question_withhold_rules_and_leave_controls_free(self):
        observation = KeyGateGame().observe()
        observation["rules"] = ["SECRET_RULE"]
        observation["goal"] = "SECRET_OBJECTIVE"
        observation["events"] = ["SECRET_STRATEGY"]
        observation["neighbors"] = {"east": {"passable": True}}
        for focus in ("mechanics", "reset-retention"):
            messages = exploration_messages(observation, 0, None, {}, [], Coverage(observation),
                                            None, {"actions": 24}, focus=focus)
            with self.subTest(focus=focus):
                context = json.loads(messages[1]["content"])
                self.assertEqual(context["observation"]["controls"], [*observation["actions"], "reset", "finish"])
                self.assertEqual(context["observation"]["events"], ["Environment started."])
                for spoiler in ("SECRET_", '"passable"', "Collect the key", "key before", "stand next"):
                    self.assertNotIn(spoiler, json.dumps(messages))
                if focus == "reset-retention":
                    self.assertIn("Choose an intentional boundary/refusal probe", context["session_question"])


if __name__ == "__main__":
    unittest.main()
