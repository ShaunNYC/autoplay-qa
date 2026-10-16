"""Controlled fixtures test mechanics of the protocol, not live model learning."""

import copy
import json
import time
import unittest
from unittest.mock import patch

from autoplay_qa.exploration import MEASURED_PROMPT, ExplorationLimits
from autoplay_qa.nebius import NebiusError
from autoplay_qa.review import review_session
from autoplay_qa.session import replay
from tests import test_exploration_grounded as grounded_fixtures
from tests.test_strict_exploration import note, pair


def measured_note(control, refs, *, condition_step=None, before=(1, 1), after=(1, 1), comparison=None, status="uncertain", identifier=None):
    value = note(control, refs, comparison, status=status, claim=f"The sampled {control} action has this recorded position effect under its initial conditions.")
    if identifier:
        value["id"] = identifier
    value["effect"] = {"condition_step": condition_step or refs[0], "facet": "position", "before": list(before), "after": list(after)} if refs else None
    return value


def scoped_choice(action, **kwargs):
    value = grounded_fixtures.choice(action, **kwargs)
    target = value["experiment_control"]
    value["experiment_goal"] = f"Does the sampled {target} effect repeat from equivalent conditions after reset?"
    return value


def two_scopes():
    movement = measured_note("north", [1], identifier="movement-note")
    movement_pair = measured_note("north", [1, 3], identifier="movement-note", comparison=pair(1, 3), status="supported")
    interaction = measured_note("interact", [4], identifier="interaction-note")
    interaction_pair = measured_note("interact", [4, 6], identifier="interaction-note", comparison=pair(4, 6, 1, 2), status="supported")
    return [
        scoped_choice("north", controls=["north", "reset", "north", "interact"], target="north", experiment="movement"),
        scoped_choice("reset", controls=["reset", "north", "interact"], mode="continue", target="north", experiment="movement", updates=[movement]),
        scoped_choice("north", controls=["north", "interact"], mode="continue", target="north", experiment="movement"),
        scoped_choice("interact", controls=["interact"], mode="retarget", revision=[3], revision_reason="Keep my queued control, but measure the separate interaction question now.", updates=[movement_pair]),
        scoped_choice("reset", controls=["reset", "interact", "finish"], updates=[interaction]),
        scoped_choice("interact", controls=["interact", "finish"], mode="continue"),
        scoped_choice("finish", controls=["finish"], mode="continue", updates=[interaction_pair]),
    ]


class MeasuredExplorationTests(unittest.TestCase):
    run_fixture = grounded_fixtures.GroundedControllerTests.run_fixture

    def test_scope_switch_keeps_queued_control_and_both_measured_notes_across_resets(self):
        for seed in (0, 5):
            with self.subTest(seed=seed):
                summary, rows, contexts, path, _ = self.run_fixture(two_scopes(), version=4, seed=seed)
                self.assertEqual(summary["status"], "finished")
                self.assertEqual((summary["steps"], summary["resets"]), (6, 2))
                self.assertIsNone(summary["unassessed_final_step"])
                self.assertEqual(summary["research_context"]["contract_version"], 2)
                self.assertEqual(summary["research_context"]["scope_switches"], 1)
                self.assertEqual([r["decision"]["action"] for r in rows if r["type"] in {"step", "reset"}], ["north", "reset", "north", "interact", "reset", "interact"])
                self.assertEqual(contexts[3]["research_context"]["pending_plan"]["controls"], ["interact"])
                switched = next(r for r in rows if r["type"] == "step" and r["step"] == 4)
                self.assertEqual((switched["decision"]["experiment_id"], switched["decision"]["action_role"]), ("interaction", "trial"))
                self.assertEqual(contexts[2]["hypotheses"][0]["effect"], {"condition_step": 1, "facet": "position", "before": [1, 1], "after": [1, 1]})
                self.assertTrue(all(row["messages"][0]["content"] == MEASURED_PROMPT for row in rows if row["type"] == "request"))
                self.assertTrue(summary["evidence_measurements"])
                self.assertEqual(replay(path)["actions_replayed"], 6)
                self.assertTrue(review_session(path)["goal_checks"]["focused_question_observed"])
                evidence = review_session(path, include_evidence=True)
                self.assertFalse(evidence["learning_claims_verified"])

    def test_another_experiments_note_cannot_replace_the_interaction_note(self):
        values = two_scopes()
        values.insert(4, scoped_choice("reset", controls=["reset", "interact", "finish"], updates=[measured_note("north", [1, 3], identifier="movement-note", comparison=pair(1, 3), status="supported")]))
        summary, rows, contexts, path, _ = self.run_fixture(values, version=4)
        self.assertEqual(summary["status"], "finished")
        self.assertEqual(summary["rejected_decisions"], 1)
        rejected = next(r for r in rows if r["type"] == "rejected_decision")
        self.assertIn("missing trial steps [4]", rejected["error"])
        self.assertEqual(contexts[4]["hypotheses"], contexts[5]["hypotheses"])
        self.assertEqual(contexts[4]["research_context"], contexts[5]["research_context"])
        self.assertEqual(replay(path)["actions_replayed"], 6)

    def test_sampled_measurement_cannot_remember_a_forecast_as_actual_before_state(self):
        actual = measured_note("east", [1], before=(1, 1), after=(2, 1))
        wrong = copy.deepcopy(actual)
        wrong["effect"]["before"] = [9, 9]
        values = [scoped_choice("east", target="east", experiment="east"), scoped_choice("finish", target="east", experiment="east", updates=[wrong]), scoped_choice("finish", target="east", experiment="east", updates=[actual])]
        summary, rows, contexts, path, _ = self.run_fixture(values, version=4)
        self.assertEqual(summary["status"], "finished")
        self.assertEqual(summary["rejected_decisions"], 1)
        self.assertEqual(summary["hypotheses"][0]["effect"]["before"], [1, 1])
        self.assertEqual(contexts[1]["hypotheses"], contexts[2]["hypotheses"])
        self.assertEqual(replay(path)["actions_replayed"], 1)

    def test_retarget_with_changed_tail_is_atomic_and_repaired_without_executing_it(self):
        values = two_scopes()
        malformed = copy.deepcopy(values[3])
        malformed["plan"]["controls"] = ["interact", "reset"]
        values.insert(3, malformed)
        summary, rows, contexts, path, _ = self.run_fixture(values, version=4)
        self.assertEqual(summary["status"], "finished")
        self.assertEqual(summary["rejected_decisions"], 1)
        self.assertEqual(contexts[3]["research_context"], contexts[4]["research_context"])
        self.assertEqual(summary["research_context"]["scope_switches"], 1)
        self.assertEqual(replay(path)["actions_replayed"], 6)

    def test_changed_goal_of_existing_scope_is_rejected_before_memory_or_action(self):
        values = two_scopes()
        malformed = copy.deepcopy(values[1])
        malformed["experiment_goal"] = "A different unrelated question."
        values.insert(1, malformed)
        summary, rows, contexts, path, _ = self.run_fixture(values, version=4)
        self.assertEqual(summary["status"], "finished")
        self.assertEqual(summary["rejected_decisions"], 1)
        self.assertEqual(contexts[1]["hypotheses"], contexts[2]["hypotheses"])
        self.assertEqual(summary["resets"], 2)
        self.assertEqual(replay(path)["actions_replayed"], 6)

    def test_actual_changes_and_forecast_differences_are_independently_bound(self):
        values = two_scopes()
        malformed = copy.deepcopy(values[1])
        malformed["fixture_assessment_patch"] = {"actual_changed_facets": ["position"]}
        values.insert(1, malformed)
        summary, rows, contexts, path, _ = self.run_fixture(values, version=4)
        self.assertEqual(summary["status"], "finished")
        self.assertEqual(summary["rejected_decisions"], 1)
        self.assertEqual(contexts[1]["reflection_contract"]["pending_outcome"]["actual_changed_facets"], [])
        self.assertEqual(replay(path)["actions_replayed"], 6)

    def test_compact_refs_preserve_required_fields_and_omit_spoilers(self):
        from autoplay_qa.schema_refs import expand_schema

        _, rows, contexts, _, _ = self.run_fixture(two_scopes(), version=4)
        for row in rows:
            if row["type"] != "request":
                continue
            schema = row["response_schema"]
            self.assertIn("$defs", schema)
            full = expand_schema(schema)
            self.assertEqual(len(full["required"]), 10)
            self.assertIn("effect", full["properties"]["hypotheses"]["items"]["required"])
            self.assertLess(len(json.dumps(schema)), len(json.dumps(full)))
        for context in contexts:
            self.assertTrue(context["experiment_progress"])
            self.assertNotIn("rules", context["observation"])
            self.assertNotIn("goal", context["observation"])
            self.assertNotIn("passable", json.dumps(context["observation"]))
        self.assertEqual(contexts[0]["observation"]["events"], ["Environment started."])

    def test_required_note_is_conditional_on_this_pending_assessment_and_not_counted_as_accepted(self):
        _, _, contexts, _, _ = self.run_fixture(two_scopes(), version=4)
        requirements = contexts[1]["response_contract"]["hypotheses"]["required_trial_notes_after_valid_assessment"]
        self.assertEqual(requirements["pending_assessment_step"], 1)
        self.assertEqual(requirements["experiments"], [{"experiment_id": "movement", "control": "north", "evidence_trial_steps": [1]}])
        self.assertEqual(contexts[1]["experiment_progress"]["experiments"][0]["assessed_trial_count"], 0)
        self.assertEqual(contexts[1]["experiment_progress"]["experiments"][0]["note_count"], 0)
        self.assertEqual(contexts[0]["response_contract"]["hypotheses"]["required_trial_notes_after_valid_assessment"]["experiments"], [])

    def test_plain_and_object_requests_keep_conditional_note_obligations(self):
        for mode in ("plain", "json_object"):
            with self.subTest(mode=mode):
                summary, _, contexts, path, _ = self.run_fixture(two_scopes(), version=4, mode=mode)
                self.assertEqual(summary["status"], "finished")
                self.assertEqual(contexts[1]["required_trial_notes_after_valid_assessment"]["pending_assessment_step"], 1)
                self.assertIn("response_schema", contexts[1])
                self.assertEqual(replay(path)["actions_replayed"], 6)

    def test_budget_api_error_and_interruption_never_execute_queued_interaction(self):
        for failure, status in ((NebiusError("Fixture network failure."), "inference_error"), (KeyboardInterrupt(), "interrupted")):
            with self.subTest(status=status):
                summary, _, _, path, _ = self.run_fixture([two_scopes()[0], failure], version=4)
                self.assertEqual(summary["status"], status)
                self.assertEqual(summary["steps"], 1)
                self.assertEqual(summary["unassessed_final_step"], 1)
                self.assertEqual(summary["research_context"]["pending_plan"]["controls"], ["reset", "north", "interact"])
                self.assertFalse(summary["usage_complete"])
                self.assertEqual(replay(path)["actions_replayed"], 1)
        summary, _, _, path, _ = self.run_fixture(two_scopes(), version=4, limits=ExplorationLimits(max_actions=1))
        self.assertEqual(summary["status"], "action_limit")
        self.assertEqual(summary["unassessed_final_step"], 1)
        self.assertEqual(replay(path)["actions_replayed"], 1)

    def test_schema_preparation_exhausting_deadline_never_starts_an_inference_request(self):
        from autoplay_qa import exploration

        clock = {"now": time.monotonic()}
        original = exploration.turn_schema

        def slow_preparation(*args, **kwargs):
            schema = original(*args, **kwargs)
            clock["now"] += 2
            return schema

        with patch.object(exploration, "turn_schema", side_effect=slow_preparation):
            summary, rows, _, path, transport = self.run_fixture([], version=4, clock=lambda: clock["now"], limits=ExplorationLimits(max_seconds=1))
        self.assertEqual(summary["status"], "time_limit")
        self.assertEqual((summary["steps"], summary["inference_requests"]), (0, 0))
        self.assertTrue(summary["usage_complete"])
        self.assertFalse(any(row["type"] == "request" for row in rows))
        transport.assert_not_called()
        self.assertEqual(replay(path)["actions_replayed"], 0)


if __name__ == "__main__":
    unittest.main()
