"""Controller repair fixtures; these are not evidence of live model learning."""

import copy
import json
import unittest

from autoplay_qa.exploration import CONTROLS, MEASURED_PROMPT, ExplorationLimits
from autoplay_qa.review import review_session
from autoplay_qa.session import replay
from tests import test_exploration_grounded as grounded_fixtures
from tests.test_measured_exploration import measured_note, scoped_choice, two_scopes
from tests.test_strict_exploration import pair


PRIVATE_FIELD = "PRIVATE_COMPARISON_FIELD_DO_NOT_ECHO"
PRIVATE_VALUE = "PRIVATE_COMPARISON_VALUE_DO_NOT_ECHO"


def first_trial_choices(comparison):
    observed = measured_note("north", [1], identifier="movement-note")
    first = scoped_choice("north", controls=["north", "reset"], target="north", experiment="movement")
    repaired = scoped_choice("reset", controls=["reset"], mode="continue", target="north", experiment="movement", updates=[observed])
    malformed = copy.deepcopy(repaired)
    malformed["hypotheses"][0]["comparison"] = comparison
    malformed["hypotheses"][0]["claim"] = "This rejected proposal must never enter persistent memory."
    finish = scoped_choice("finish", target="north", experiment="movement")
    return first, malformed, repaired, finish


def north_pair_prefix():
    return [
        scoped_choice("north", controls=["north", "reset", "north"], target="north", experiment="movement"),
        scoped_choice("reset", controls=["reset", "north"], mode="continue", target="north", experiment="movement", updates=[measured_note("north", [1], identifier="movement-note")]),
        scoped_choice("north", controls=["north"], mode="continue", target="north", experiment="movement"),
    ]


class MeasuredComparisonRepairTests(unittest.TestCase):
    run_fixture = grounded_fixtures.GroundedControllerTests.run_fixture

    def assert_atomic_retry_context(self, original, retry):
        for key in (
            "observation", "episode", "active_experiment", "hypotheses",
            "pending_assessment_step", "comparison_contract", "reflection_contract",
            "coverage", "research_context", "experiment_progress",
        ):
            self.assertEqual(original[key], retry[key], key)

    def assert_null_guidance(self, context):
        feedback = context["validation_feedback"]
        self.assertIn("comparison must be JSON null", feedback)
        self.assertIn("this note's cited evidence", feedback)
        self.assertNotIn(PRIVATE_FIELD, feedback)
        self.assertNotIn(PRIVATE_VALUE, feedback)

    def test_first_trial_comparison_shape_gets_one_atomic_repair_using_json_null(self):
        for malformed in ({}, {PRIVATE_FIELD: PRIVATE_VALUE}, PRIVATE_VALUE, [PRIVATE_VALUE]):
            with self.subTest(shape=type(malformed).__name__):
                summary, rows, contexts, path, transport = self.run_fixture(first_trial_choices(malformed), version=4)
                self.assertEqual((summary["status"], summary["steps"], summary["resets"], summary["inference_requests"], summary["rejected_decisions"]), ("finished", 2, 1, 4, 1))
                self.assert_atomic_retry_context(contexts[1], contexts[2])
                self.assert_null_guidance(contexts[2])
                self.assertEqual(contexts[2]["hypotheses"], [])
                self.assertEqual(contexts[2]["pending_assessment_step"], 1)
                self.assertEqual(contexts[2]["experiment_progress"]["experiments"][0]["assessed_trial_count"], 0)
                self.assertEqual([row["request_number"] for row in rows if row["type"] == "memory"], [1, 3, 4])
                executed = [row for row in rows if row["type"] in {"step", "reset"}]
                self.assertEqual([(row["request_number"], row["decision"]["action"]) for row in executed], [(1, "north"), (3, "reset")])
                rejected = next(row for row in rows if row["type"] == "completion" and row["request_number"] == 2)
                self.assertIsNone(rejected["answer"])
                accepted = next(row for row in rows if row["type"] == "memory" and row["request_number"] == 3)
                self.assertEqual(accepted["assessment"]["step"], 1)
                self.assertIsNone(accepted["hypotheses"][0]["comparison"])
                self.assertEqual(accepted["hypotheses"][0]["evidence_steps"], [1])
                self.assertEqual(accepted["evidence_measurements"][0]["measurements"][0]["step"], 1)
                self.assertNotIn("rejected proposal", json.dumps(summary["hypotheses"]))
                self.assertNotIn(PRIVATE_FIELD, json.dumps(rows))
                self.assertNotIn(PRIVATE_VALUE, json.dumps(rows))
                self.assertIsNone(summary["unassessed_final_step"])
                self.assertEqual(replay(path)["actions_replayed"], 2)
                self.assertFalse(review_session(path)["goal_checks"]["focused_question_observed"])

                requests = [row for row in rows if row["type"] == "request"]
                self.assertEqual(requests[1]["response_schema"], requests[2]["response_schema"])
                for row, context, call in zip(requests, contexts, transport.call_args_list):
                    self.assertEqual(row["messages"][0]["content"], MEASURED_PROMPT)
                    self.assertEqual(context["observation"]["controls"], list(CONTROLS))
                    self.assertNotIn("goal", context["observation"])
                    text = json.dumps(context)
                    for withheld in ('"rules"', '"passable"', '"passability"', "Find the key", "Collect the key, unlock the gate"):
                        self.assertNotIn(withheld, text)
                    body = call.args[1]
                    self.assertEqual(body["model"], "nvidia/Nemotron-3_5-Lightning")
                    self.assertEqual(body["max_tokens"], 8192)
                    self.assertEqual(body["chat_template_kwargs"], {"enable_thinking": True})
                    self.assertEqual(call.args[2], 45)
                    self.assertEqual(body["response_format"]["json_schema"]["schema"], row["response_schema"])
                self.assertEqual(contexts[0]["observation"]["events"], ["Environment started."])
                limits = ExplorationLimits()
                self.assertEqual((limits.max_actions, limits.max_requests, limits.max_seconds), (24, 28, 300))

    def test_genuine_cited_pair_repairs_exact_fields_without_null_only_guidance(self):
        linked = measured_note("north", [1, 3], identifier="movement-note", comparison=pair(1, 3), status="supported")
        repaired = scoped_choice("finish", target="north", experiment="movement", updates=[linked])
        malformed = copy.deepcopy(repaired)
        malformed["hypotheses"][0]["comparison"] = {"original_step": 1}
        summary, rows, contexts, path, _ = self.run_fixture([*north_pair_prefix(), malformed, repaired], version=4)
        self.assertEqual((summary["status"], summary["steps"], summary["resets"], summary["rejected_decisions"]), ("finished", 3, 1, 1))
        self.assert_atomic_retry_context(contexts[3], contexts[4])
        feedback = contexts[4]["validation_feedback"]
        self.assertIn("eligible", feedback)
        self.assertIn("exact five fields", feedback)
        self.assertNotIn("comparison must be JSON null", feedback)
        for field in ("original_step", "repeated_step", "original_episode", "repeated_episode", "result"):
            self.assertIn(field, feedback)
        self.assertEqual(contexts[3]["comparison_contract"]["eligible_pairs"], [{**pair(1, 3), "control": "north"}])
        self.assertEqual([row["request_number"] for row in rows if row["type"] == "memory"], [1, 2, 3, 5])
        self.assertEqual(summary["hypotheses"][0]["comparison"], pair(1, 3))
        self.assertEqual(summary["hypotheses"][0]["evidence_steps"], [1, 3])
        self.assertEqual(replay(path)["validated_comparisons"], 1)
        self.assertTrue(review_session(path)["goal_checks"]["reset_retested_control_families"]["movement"])

    def test_existing_pair_does_not_fill_uncited_evidence_for_a_repair(self):
        observed = measured_note("north", [1], identifier="movement-note")
        repaired = scoped_choice("finish", target="north", experiment="movement", updates=[observed])
        malformed = copy.deepcopy(repaired)
        malformed["hypotheses"][0]["comparison"] = {}
        summary, _, contexts, path, _ = self.run_fixture([*north_pair_prefix(), malformed, repaired], version=4)
        self.assertEqual(len(contexts[3]["comparison_contract"]["eligible_pairs"]), 1)
        self.assert_null_guidance(contexts[4])
        self.assertEqual(summary["hypotheses"][0]["evidence_steps"], [1])
        self.assertIsNone(summary["hypotheses"][0]["comparison"])
        self.assertEqual(summary["comparison_receipts"], [])
        self.assertEqual(replay(path)["validated_comparisons"], 0)

    def test_a_pair_for_another_control_cannot_supply_first_interaction_comparison(self):
        values = two_scopes()
        malformed = copy.deepcopy(values[4])
        malformed["hypotheses"][0]["comparison"] = {}
        values.insert(4, malformed)
        summary, _, contexts, path, _ = self.run_fixture(values, version=4)
        self.assertEqual((summary["status"], summary["steps"], summary["rejected_decisions"]), ("finished", 6, 1))
        self.assertEqual(contexts[4]["comparison_contract"]["eligible_pairs"], [{**pair(1, 3), "control": "north"}])
        self.assertEqual(contexts[4]["reflection_contract"]["pending_outcome"]["control"], "interact")
        self.assert_atomic_retry_context(contexts[4], contexts[5])
        self.assert_null_guidance(contexts[5])
        self.assertEqual(replay(path)["validated_comparisons"], 2)

    def test_extra_trial_from_different_conditions_prevents_pair_guidance_for_those_citations(self):
        # North1/3 is comparable, but east4 changes the start of north5.
        # Retaining all three north citations must not silently discard north5.
        heterogeneous = measured_note("north", [1, 3, 5], identifier="movement-note")
        repaired = scoped_choice("finish", controls=["finish"], mode="continue", target="north", experiment="movement", updates=[heterogeneous])
        malformed = copy.deepcopy(repaired)
        malformed["hypotheses"][0]["comparison"] = {}
        values = [
            *north_pair_prefix(),
            scoped_choice("east", controls=["east", "north", "finish"], target="north", experiment="movement"),
            scoped_choice("north", controls=["north", "finish"], mode="continue", target="north", experiment="movement"),
            malformed, repaired,
        ]
        summary, rows, contexts, path, _ = self.run_fixture(values, version=4)
        self.assertEqual((summary["status"], summary["steps"], summary["rejected_decisions"]), ("finished", 5, 1))
        self.assertEqual(contexts[5]["comparison_contract"]["eligible_pairs"], [{**pair(1, 3), "control": "north"}])
        north_trials = [row for row in rows if row["type"] == "step" and row["decision"]["action"] == "north"]
        self.assertEqual([row["before"]["player"]["position"] for row in north_trials], [[1, 1], [1, 1], [2, 1]])
        self.assert_atomic_retry_context(contexts[5], contexts[6])
        self.assert_null_guidance(contexts[6])
        self.assertEqual(summary["hypotheses"][0]["evidence_steps"], [1, 3, 5])
        self.assertIsNone(summary["hypotheses"][0]["comparison"])
        self.assertEqual(replay(path)["validated_comparisons"], 0)

    def test_same_control_trials_in_different_experiments_are_not_a_declared_pair(self):
        first_note = measured_note("north", [1], identifier="first-note")
        second_note = measured_note("north", [3], identifier="second-note")
        first = scoped_choice("north", controls=["north", "reset", "north"], target="north", experiment="first")
        reset = scoped_choice("reset", controls=["reset", "north"], mode="continue", target="north", experiment="first", updates=[first_note])
        retarget = scoped_choice("north", controls=["north"], mode="retarget", target="north", experiment="second", revision=[2], revision_reason="Keep the queued control but measure a separate question under the restarted conditions.")
        goal = "Does this recorded north action leave position unchanged from the visible start?"
        retarget["experiment_goal"] = goal
        repaired = scoped_choice("finish", target="north", experiment="second", updates=[second_note])
        repaired["experiment_goal"] = goal
        malformed = copy.deepcopy(repaired)
        malformed["hypotheses"][0]["evidence_steps"] = [1, 3]
        malformed["hypotheses"][0]["comparison"] = {}
        summary, _, contexts, path, _ = self.run_fixture([first, reset, retarget, malformed, repaired], version=4)
        self.assertEqual((summary["status"], summary["steps"], summary["rejected_decisions"]), ("finished", 3, 1))
        self.assertEqual(contexts[3]["comparison_contract"]["eligible_pairs"], [{**pair(1, 3), "control": "north"}])
        self.assertEqual([entry["id"] for entry in contexts[3]["experiment_progress"]["experiments"]], ["first", "second"])
        self.assertTrue(all(entry["eligible_pair_count"] == 0 for entry in contexts[3]["experiment_progress"]["experiments"]))
        self.assert_atomic_retry_context(contexts[3], contexts[4])
        self.assert_null_guidance(contexts[4])
        self.assertEqual(summary["research_context"]["scope_switches"], 1)
        self.assertEqual(replay(path)["validated_comparisons"], 0)

    def test_two_invalid_shapes_stop_after_the_single_repair_without_reset_or_memory_commit(self):
        first, malformed, repaired, finish = first_trial_choices({})
        summary, rows, contexts, path, transport = self.run_fixture([first, malformed, malformed, repaired, finish], version=4)
        self.assertEqual((summary["status"], summary["steps"], summary["resets"], summary["inference_requests"], summary["rejected_decisions"]), ("invalid_decision", 1, 0, 3, 2))
        self.assertEqual(transport.call_count, 3)
        self.assert_null_guidance(contexts[2])
        self.assertEqual([row["request_number"] for row in rows if row["type"] == "memory"], [1])
        self.assertFalse(any(row["type"] in {"reset", "finish"} for row in rows))
        self.assertEqual(summary["hypotheses"], [])
        self.assertEqual(summary["evidence_measurements"], [])
        self.assertEqual(summary["unassessed_final_step"], 1)
        self.assertEqual(summary["research_context"]["pending_plan"]["controls"], ["reset"])
        self.assertEqual(replay(path)["reflected_steps"], 0)

    def test_request_budget_can_end_before_repair_without_executing_queued_reset(self):
        first, malformed, repaired, finish = first_trial_choices({})
        summary, rows, _, path, transport = self.run_fixture([first, malformed, repaired, finish], version=4, limits=ExplorationLimits(max_requests=2))
        self.assertEqual((summary["status"], summary["steps"], summary["resets"], summary["inference_requests"], summary["rejected_decisions"]), ("request_limit", 1, 0, 2, 1))
        self.assertEqual(transport.call_count, 2)
        self.assertTrue(summary["usage_complete"])
        self.assertEqual(summary["hypotheses"], [])
        self.assertEqual(summary["unassessed_final_step"], 1)
        self.assertFalse(any(row["type"] in {"reset", "finish"} for row in rows))
        self.assertEqual(replay(path)["actions_replayed"], 1)


if __name__ == "__main__":
    unittest.main()
