"""Explicit transport fixtures for the strict controller, never live evidence."""

import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from autoplay_qa.cli import begin_session
from autoplay_qa.comparisons import COMPARISON_CONTRACT_VERSION
from autoplay_qa.config import Settings
from autoplay_qa.exploration import ExplorationLimits, run_exploration
from autoplay_qa.game import KeyGateGame
from autoplay_qa.nebius import NebiusClient, NebiusError
from autoplay_qa.session import SessionLog, replay
from tests.test_exploration import decision, response


def note(control, refs=(), pair=None, *, status="uncertain", claim=None):
    return {"id": control, "claim": claim or f"{control} leaves the initial player and objects unchanged.",
            "status": status, "evidence_steps": list(refs), "control": control,
            "comparison": pair}


def pair(original, repeated, old_episode=0, new_episode=1):
    return {"original_step": original, "repeated_step": repeated,
            "original_episode": old_episode, "repeated_episode": new_episode,
            "result": "same"}


class StrictExplorationTests(unittest.TestCase):
    def run_fixture(self, choices, *, limits=ExplorationLimits(), seed=0, clock=None, evaluator=None):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        path = Path(directory.name) / "fixture.jsonl"
        game = KeyGateGame(seed)
        client = NebiusClient(Settings("unit-test-key"))
        choices = iter(choices)

        def transport(_path, body=None, timeout=None):
            context = json.loads(body["messages"][1]["content"])
            choice = next(choices)
            if isinstance(choice, BaseException):
                raise choice
            value = copy.deepcopy(choice)
            receipt = context["comparison_contract"]["latest_control_receipt"]
            value["assessment"] = ({
                **{field: receipt[field] for field in ("step", "control", "episode_before", "episode_after")},
                "outcome": "Fixture: actual feedback and state match the predicted effect.",
            } if context["pending_assessment_step"] is not None else None)
            assessment_patch = value.pop("assessment_patch", {})
            if value["assessment"] is not None:
                value["assessment"].update(assessment_patch)
            return response(value, finish=value.pop("fixture_finish", "stop"))

        kwargs = {"evaluator": evaluator} if evaluator is not None else {}
        clock_patch = patch("autoplay_qa.exploration.time.monotonic", side_effect=clock) if clock else patch("autoplay_qa.exploration.time.monotonic", wraps=__import__("time").monotonic)
        with SessionLog(path) as log:
            begin_session(log, game, "unit-test-fixtures", schema_version=2, task="explore",
                          focus="reset-retention", comparison_contract=COMPARISON_CONTRACT_VERSION)
            with patch.object(client, "_request", side_effect=transport) as mocked, clock_patch:
                summary = run_exploration(game, client, log, lambda: KeyGateGame(seed), limits,
                                          focus="reset-retention", comparison_contract=True, **kwargs)
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        contexts = [json.loads(row["messages"][1]["content"]) for row in rows if row["type"] == "request"]
        return summary, rows, contexts, path, mocked

    @staticmethod
    def comparisons():
        return [
            decision("north", experiment="movement", prediction="Intentionally test whether movement into the visible wall is blocked."),
            decision("interact", hypotheses=[note("north", [1])], experiment="interaction"),
            decision("reset", hypotheses=[note("interact", [2])], experiment="interaction"),
            decision("north", experiment="movement"),
            decision("interact", hypotheses=[note("north", [1, 4], pair(1, 4), status="supported")], experiment="interaction"),
            decision("finish", hypotheses=[note("interact", [2, 5], pair(2, 5), status="supported")], experiment="interaction"),
        ]

    def test_both_comparisons_have_actual_receipts_and_retained_notes(self):
        from autoplay_qa.review import review_session

        summary, rows, contexts, path, _ = self.run_fixture(self.comparisons())
        self.assertEqual((summary["status"], summary["steps"], summary["resets"]), ("finished", 5, 1))
        self.assertIsNone(summary["unassessed_final_step"])
        receipts = summary["comparison_receipts"]
        self.assertEqual({item["control"] for item in receipts}, {"north", "interact"})
        self.assertTrue(all(item["verified"] is False for item in receipts))
        self.assertEqual(contexts[3]["episode"], 1)
        self.assertEqual(contexts[2]["hypotheses"], contexts[3]["hypotheses"][:1])
        self.assertEqual(contexts[3]["hypotheses"][1]["evidence_steps"], [2])
        self.assertEqual(contexts[4]["comparison_contract"]["eligible_pairs"][0]["repeated_step"], 4)
        for memory in (row for row in rows if row["type"] == "memory"):
            self.assertIn("comparison_receipts", memory)
        self.assertEqual(replay(path)["actions_replayed"], 5)
        report = review_session(path, include_evidence=True)
        self.assertEqual({item["action"] for item in report["reset_learning"]["retest_pairs"]}, {"north", "interact"})

    def test_bad_pair_is_atomic_then_single_repair_executes_same_pending_action(self):
        choices = self.comparisons()
        bad = copy.deepcopy(choices[4])
        bad["experiment_id"] = "REJECTED_EXPERIMENT"
        bad["hypotheses"][0]["claim"] = "REJECTED_CLAIM"
        bad["hypotheses"][0]["comparison"]["original_episode"] = 1
        choices.insert(4, bad)
        summary, rows, contexts, path, mocked = self.run_fixture(choices)
        self.assertEqual((summary["status"], summary["rejected_decisions"], mocked.call_count), ("finished", 1, 7))
        self.assertEqual(contexts[4]["hypotheses"], contexts[5]["hypotheses"])
        self.assertEqual(contexts[4]["comparison_contract"], contexts[5]["comparison_contract"])
        self.assertEqual(contexts[4]["observation"], contexts[5]["observation"])
        self.assertNotIn("REJECTED_CLAIM", json.dumps(contexts[5]))
        self.assertFalse(any(row["type"] in {"memory", "step", "reset", "finish"} and row["request_number"] == 5 for row in rows))
        self.assertEqual(replay(path)["actions_replayed"], 5)

    def test_reset_is_assessable_before_any_game_trial_and_cannot_be_trial_evidence(self):
        good = [decision("reset"), decision("finish")]
        summary, _, contexts, path, _ = self.run_fixture(good)
        self.assertEqual((summary["status"], summary["resets"]), ("finished", 1))
        self.assertEqual(contexts[1]["comparison_contract"]["completed_trials"], [])
        item_schema = contexts[1]["response_schema"]["properties"]["hypotheses"]["items"]["properties"]
        self.assertEqual(item_schema["comparison"], {"type": "null"})
        self.assertEqual(item_schema["evidence_steps"]["maxItems"], 0)
        self.assertEqual(replay(path)["resets_replayed"], 1)
        malformed = decision("north", hypotheses=[note("north", [1])])
        summary, _, _, path, _ = self.run_fixture([decision("reset"), malformed, malformed])
        self.assertEqual((summary["status"], summary["steps"], summary["rejected_decisions"]), ("invalid_decision", 1, 2))
        self.assertEqual(summary["hypotheses"], [])
        self.assertEqual(summary["unassessed_final_step"], 1)
        self.assertEqual(replay(path)["actions_replayed"], 1)

    def test_wrong_control_receipt_cannot_assess_a_reset_as_movement(self):
        malformed = {**decision("finish"), "assessment_patch": {"control": "north"}}
        summary, rows, _, path, _ = self.run_fixture([decision("reset"), malformed, malformed])
        self.assertEqual((summary["status"], summary["steps"]), ("invalid_decision", 1))
        self.assertEqual(len([row for row in rows if row["type"] == "memory"]), 1)
        self.assertEqual(summary["unassessed_final_step"], 1)
        self.assertEqual(replay(path)["actions_replayed"], 1)

    def test_budget_and_interruption_keep_unassessed_actual_outcome(self):
        for stop, expected in ((NebiusError("Fixture unavailable"), "inference_error"), (KeyboardInterrupt(), "interrupted")):
            with self.subTest(expected=expected):
                summary, _, _, path, mocked = self.run_fixture([decision("north"), stop])
                self.assertEqual((summary["status"], summary["steps"], mocked.call_count), (expected, 1, 2))
                self.assertEqual(summary["unassessed_final_step"], 1)
                self.assertFalse(summary["usage_complete"])
                self.assertEqual(replay(path)["actions_replayed"], 1)
        summary, _, _, path, _ = self.run_fixture([decision("reset")], limits=ExplorationLimits(max_actions=1))
        self.assertEqual((summary["status"], summary["steps"], summary["unassessed_final_step"]), ("action_limit", 1, 1))
        self.assertEqual(replay(path)["actions_replayed"], 1)

    def test_deadline_rejects_completed_response_without_memory_or_action(self):
        # Both controller and client's latency measurement use this shared clock.
        summary, rows, _, path, _ = self.run_fixture([decision("north")],
            limits=ExplorationLimits(max_seconds=2), clock=[0, 0, 0, 1, 3, 3, 3, 3, 3, 3])
        self.assertEqual((summary["status"], summary["steps"]), ("time_limit", 0))
        self.assertFalse(any(row["type"] == "memory" for row in rows))
        self.assertEqual(replay(path)["actions_replayed"], 0)

    def test_terminal_controls_and_request_limit_keep_strict_bookkeeping(self):
        actions = ["south", "south", "east", "east", "interact", "east", "east", "east", "east"]
        choices = [decision(action, hypotheses=[note("south", [1])] if index == 1 else [])
                   for index, action in enumerate(actions)]
        choices.extend([decision("north"), decision("finish")])
        summary, _, contexts, path, _ = self.run_fixture(choices)
        self.assertEqual((summary["status"], summary["steps"], summary["rejected_decisions"]), ("finished", 9, 1))
        self.assertEqual(contexts[-1]["observation"]["controls"], ["reset", "finish"])
        self.assertEqual(replay(path)["status"], "won")
        summary, _, _, path, mocked = self.run_fixture([decision("north"), decision("finish")],
                                                       limits=ExplorationLimits(max_requests=2))
        self.assertEqual((summary["status"], summary["rejected_decisions"], mocked.call_count), ("request_limit", 1, 2))
        self.assertEqual(summary["unassessed_final_step"], 1)
        self.assertEqual(replay(path)["actions_replayed"], 1)

    def test_fact_context_contains_only_recorded_visible_data(self):
        def evaluator(_before, _action, _after):
            return {"contract_violations": [], "PRIVATE_CHECKER_DETAIL": "PRIVATE_CONTRACT"}

        _, rows, contexts, _, _ = self.run_fixture(self.comparisons(), evaluator=evaluator)
        text = json.dumps(contexts)
        for withheld in ('"rules"', '"goal"', '"passable"', "PRIVATE_CHECKER_DETAIL", "PRIVATE_CONTRACT", "Find the key", "unlock the gate"):
            # experiment_goal is legitimate model intent, but no observation goal.
            if withheld == '"goal"':
                for context in contexts:
                    self.assertNotIn("goal", context["observation"])
                    for trial in context["comparison_contract"]["completed_trials"]:
                        self.assertNotIn("goal", trial["before"])
            else:
                self.assertNotIn(withheld, text)
        self.assertTrue(any("PRIVATE_CHECKER_DETAIL" in row.get("evaluation", {}) for row in rows))

    def test_first_learning_schema_allows_proposals_beside_required_observed_note(self):
        choices = [decision("north"), decision("finish", hypotheses=[
            note("north", [1]), note("interact", status="untested"),
        ])]
        summary, _, contexts, path, _ = self.run_fixture(choices)
        self.assertEqual(summary["status"], "finished")
        array = contexts[1]["response_schema"]["properties"]["hypotheses"]
        self.assertNotIn("contains", contexts[0]["response_schema"]["properties"]["hypotheses"])
        self.assertEqual(array["contains"]["required"], ["status", "evidence_steps"])
        self.assertNotIn("untested", array["contains"]["properties"]["status"]["enum"])
        self.assertEqual(array["contains"]["properties"]["evidence_steps"]["minItems"], 1)
        self.assertNotIn("additionalProperties", array["contains"])
        self.assertIn("untested", array["items"]["properties"]["status"]["enum"])
        self.assertEqual({note["status"] for note in summary["hypotheses"]}, {"uncertain", "untested"})
        self.assertEqual(replay(path)["actions_replayed"], 1)

    def test_schema_binds_pair_choices_to_control_and_allows_first_interaction(self):
        choices = self.comparisons()
        # Before completing the interaction repeat, it can remain tentative
        # even though a movement comparison already exists.
        choices[4]["hypotheses"].append(note("interact", [2]))
        summary, _, contexts, path, _ = self.run_fixture(choices)
        self.assertEqual(summary["status"], "finished")
        items = contexts[4]["response_schema"]["properties"]["hypotheses"]["items"]
        for branch in items["anyOf"]:
            self.assertEqual(set(branch["required"]), {"id", "claim", "status", "evidence_steps", "control", "comparison"})
            self.assertEqual(set(branch["properties"]), set(branch["required"]))
            self.assertFalse(branch["additionalProperties"])
            self.assertEqual(branch["properties"]["evidence_steps"]["type"], "array")
            control = branch["properties"]["control"]["enum"][0]
            self.assertEqual(branch["properties"]["evidence_steps"]["maxItems"], 8 if control in {"north", "interact"} else 0)
            self.assertEqual(branch["properties"]["evidence_steps"]["items"]["type"], "integer")
        north = [branch["properties"] for branch in items["anyOf"]
                 if branch["properties"]["control"]["enum"] == ["north"]]
        interaction = [branch["properties"] for branch in items["anyOf"]
                       if branch["properties"]["control"]["enum"] == ["interact"]]
        self.assertEqual(len(north), 2)
        self.assertEqual(len(interaction), 1)
        self.assertEqual(interaction[0]["comparison"], {"type": "null"})
        self.assertEqual(interaction[0]["status"]["enum"], ["untested", "uncertain"])
        self.assertEqual(interaction[0]["evidence_steps"]["items"]["enum"], [2])
        paired = north[1]["comparison"]["anyOf"][0]
        self.assertEqual(paired["properties"]["original_step"]["enum"], [1])
        self.assertEqual(paired["properties"]["repeated_step"]["enum"], [4])
        self.assertEqual(contexts[4]["observation"]["controls"], ["north", "south", "east", "west", "interact", "wait", "reset", "finish"])
        self.assertEqual(replay(path)["actions_replayed"], 5)

    def test_strict_shape_repair_names_all_required_fields_and_keeps_state(self):
        incomplete = {"id": "east", "claim": "An untested idea", "status": "untested", "evidence_steps": []}
        choices = [decision("south", hypotheses=[incomplete]), decision("finish")]
        summary, rows, contexts, path, _ = self.run_fixture(choices)
        self.assertEqual((summary["status"], summary["steps"], summary["rejected_decisions"]), ("finished", 0, 1))
        rejection = next(row for row in rows if row["type"] == "rejected_decision")
        self.assertIn("evidence_steps, control, and comparison", rejection["error"])
        self.assertNotIn("An untested idea", rejection["error"])
        self.assertEqual(contexts[0]["observation"], contexts[1]["observation"])
        self.assertEqual(contexts[0]["hypotheses"], contexts[1]["hypotheses"])
        self.assertEqual(replay(path)["actions_replayed"], 0)


if __name__ == "__main__":
    unittest.main()
