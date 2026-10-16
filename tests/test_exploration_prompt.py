"""Prompt compression must preserve the learner's exploration contract."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from autoplay_qa.cli import begin_session
from autoplay_qa.config import Settings
from autoplay_qa.exploration import (
    EXPLORATION_FOCUSES, EXPLORATION_PROMPT, EXPLORATION_SCHEMA, Coverage,
    ExplorationDecision, Hypothesis, exploration_messages, outcome_view, run_exploration,
)
from autoplay_qa.game import KeyGateGame
from autoplay_qa.nebius import InvalidDecision, NebiusClient
from autoplay_qa.session import SessionLog, replay
from tests.test_exploration import decision, hypothesis, response


class ExplorationPromptTests(unittest.TestCase):
    def test_hypothesis_status_labels_are_not_assessments_of_a_result(self):
        for label in ("supported", "untested", "uncertain", "verified"):
            value = decision("finish", 1)
            value["assessment"]["outcome"] = label
            with self.subTest(label=label), self.assertRaises(InvalidDecision):
                ExplorationDecision.parse(json.dumps(value), ["finish"], {1}, 1)

    def test_prompt_stays_compact_without_dropping_response_fields(self):
        self.assertLessEqual(len(EXPLORATION_PROMPT.split()), 300)
        self.assertLessEqual(len(EXPLORATION_FOCUSES["reset-retention"].split()), 65)
        ordered_fields = (
            "assessment", "hypotheses", "experiment_id", "experiment_goal",
            "plan", "action", "prediction", "reason",
        )
        self.assertEqual(tuple(EXPLORATION_SCHEMA["properties"]), ordered_fields)
        self.assertIn(
            "assessment, hypotheses,\nexperiment_id, experiment_goal, plan, action, prediction, reason",
            EXPLORATION_PROMPT,
        )
        # Brief deliberation is advisory; no unsupported provider guarantee.
        self.assertNotIn("enforced", EXPLORATION_PROMPT)
        self.assertNotIn("reasoning_tokens", EXPLORATION_PROMPT)

    def test_question_discipline_keeps_generic_mechanics_controls_unrestricted(self):
        game = KeyGateGame()
        observation = game.observe()
        active = {"id": "chosen-question", "goal": "A model-selected question", "plan": "A tentative plan"}
        messages = exploration_messages(observation, 0, active, {}, [], Coverage(observation),
                                        None, {"actions": 24}, focus="mechanics")
        context = json.loads(messages[1]["content"])
        self.assertEqual(context["active_experiment"], active)
        self.assertEqual(context["observation"]["controls"], [*observation["actions"], "reset", "finish"])
        self.assertNotIn("question_progress", context)
        prompt = " ".join(messages[0]["content"].split())
        self.assertIn("Each tests one narrow falsifiable effect", prompt)
        self.assertIn("until answered or explicitly inconclusive", prompt)
        self.assertIn("a few internal sentences", prompt)
        self.assertIn("Preserve unchanged claim wording and add trial references", prompt)
        self.assertIn("Mark refuted claims contradicted or refine them", prompt)
        self.assertIn("Across-reset comparisons cite original and repeated trials", prompt)
        self.assertNotIn("[1,1]", prompt)
        for spoiler in ("key before", "Collect the key", '"passable"'):
            self.assertNotIn(spoiler, json.dumps(messages))

    def test_reset_focus_asks_independent_questions_without_selecting_controls(self):
        observation = KeyGateGame().observe()
        messages = exploration_messages(observation, 0, None, {}, [], Coverage(observation),
                                        None, {"actions": 24}, focus="reset-retention")
        context = json.loads(messages[1]["content"])
        question = context["session_question"]
        self.assertEqual(question.count("?"), 2)
        self.assertIn("two independent questions with distinct experiment IDs", question)
        self.assertIn("Choose effects, comparable conditions, controls, and reset timing yourself", question)
        self.assertIn("actual original/repeated action steps", question)
        self.assertIn("both comparisons are assessed", question)
        self.assertIn("explain remaining uncertainty", question)
        self.assertNotIn("[1,1]", question)
        self.assertEqual(context["observation"]["controls"], [*observation["actions"], "reset", "finish"])

    def test_false_tentative_note_does_not_replace_actual_refused_feedback(self):
        game = KeyGateGame()
        coverage = Coverage(game.observe())
        initial = game.observe()
        before = game.act("south")
        coverage.record(initial, "south", before, step=1, episode=0)
        after = game.act("interact")
        coverage.record(before, "interact", after, step=2, episode=0)
        history = [{"step": 2, "episode_before": 0, "episode_after": 0,
                    "experiment_id": "chosen-interaction", "action": "interact",
                    "prediction": "The key enters inventory.",
                    "before": outcome_view(before), "after": outcome_view(after)}]
        notes = {"interaction": Hypothesis("interaction", "Interaction picked up the key.",
                                           "uncertain", (2,))}
        messages = exploration_messages(after, 0, None, notes, history, coverage, 2,
                                        {"actions": 22}, focus="reset-retention")
        context = json.loads(messages[1]["content"])
        actual = context["completed_outcome_to_assess"]
        self.assertEqual(actual["action_that_already_ran"], "interact")
        self.assertEqual(actual["after_state"]["events"], ["There is no gate within reach."])
        self.assertEqual(actual["after_state"]["player"]["inventory"], [])
        self.assertEqual(actual["observed_changes"]["inventory_before"], [])
        self.assertEqual(actual["observed_changes"]["inventory_after"], [])
        self.assertEqual(context["hypotheses"][0]["claim"], "Interaction picked up the key.")
        prompt = " ".join(messages[0]["content"].split())
        self.assertIn("Actual states and recorded feedback override tentative notes/plans", prompt)
        self.assertIn("predictions are never observations", prompt)

    def test_assessed_movement_and_refused_interaction_survive_a_chosen_reset(self):
        movement = lambda evidence: hypothesis(
            identifier="move", status="supported", evidence=evidence,
            claim="East from [1,1] moves the player to [2,1] without inventory or object changes.",
        )
        interaction = lambda evidence: hypothesis(
            identifier="interaction", status="supported", evidence=evidence,
            claim="Interact from [2,1] with empty inventory is refused and leaves player and objects unchanged.",
        )
        chosen = [
            decision("east", experiment="movement", goal="Does east from [1,1] change position?"),
            decision("interact", 1, [movement([1])], experiment="interaction",
                     goal="Does interact from [2,1] have an effect?",
                     prediction="No reachable gate; interaction is refused without state changes."),
            decision("reset", 2, [interaction([2])], experiment="restart-comparison"),
            decision("east", 3, experiment="movement"),
            decision("interact", 4, [movement([1, 4])], experiment="interaction",
                     prediction="No reachable gate; interaction is refused without state changes."),
            decision("finish", 5, [interaction([2, 5])], experiment="restart-comparison"),
        ]
        chosen[1]["assessment"]["outcome"] = "Player moved east to [2,1]; inventory and objects were unchanged."
        chosen[2]["assessment"]["outcome"] = "Interaction was refused: no gate within reach; player and objects were unchanged."
        chosen[3]["assessment"]["outcome"] = "Reset restored the initial player, inventory, and objects while notes persisted."
        chosen[4]["assessment"]["outcome"] = "East again moved the player to [2,1], matching the earlier trial at step 1."
        chosen[5]["assessment"]["outcome"] = "Interaction was refused again, matching the unchanged state and feedback at step 2."

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fixture.jsonl"
            game = KeyGateGame()
            client = NebiusClient(Settings("unit-test-key"))
            with SessionLog(path) as log:
                begin_session(log, game, "unit-test-fixtures", schema_version=2,
                              task="explore", focus="reset-retention")
                with patch.object(client, "_request", side_effect=[response(item) for item in chosen]) as transport:
                    summary = run_exploration(game, client, log, lambda: KeyGateGame(), focus="reset-retention")
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            requests = [json.loads(row["messages"][1]["content"]) for row in rows if row["type"] == "request"]
            controls = [row["decision"]["action"] for row in rows if row["type"] in {"step", "reset", "finish"}]
            self.assertEqual(controls, [item["action"] for item in chosen])
            self.assertEqual(transport.call_count, 6)
            self.assertEqual(requests[0]["hypotheses"], [])
            self.assertEqual([note["evidence_steps"] for note in requests[3]["hypotheses"]], [[1], [2]])
            self.assertEqual(requests[3]["episode"], 1)
            self.assertEqual(requests[3]["observation"]["player"]["position"], [1, 1])
            self.assertEqual(requests[3]["question_progress"]["families_observed"],
                             {"movement": False, "interaction": False})
            baselines = requests[3]["question_progress"]["baseline_conditions"]
            self.assertEqual({row["step"] for row in baselines}, {1, 2})
            self.assertEqual(requests[5]["question_progress"]["families_observed"],
                             {"movement": True, "interaction": False})
            for request in requests:
                progress = request["question_progress"]
                self.assertNotIn("recommendation", progress)
                self.assertNotIn("next_action", progress)
                self.assertNotIn("passable", json.dumps(progress))
            self.assertEqual(summary["status"], "finished")
            self.assertEqual(summary["resets"], 1)
            self.assertIsNone(summary["unassessed_final_step"])
            self.assertEqual({note["id"]: note["evidence_steps"] for note in summary["hypotheses"]},
                             {"move": (1, 4), "interaction": (2, 5)})
            self.assertEqual(replay(path)["actions_replayed"], 5)


if __name__ == "__main__":
    unittest.main()
