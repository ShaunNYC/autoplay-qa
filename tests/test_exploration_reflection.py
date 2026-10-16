"""Controller fixtures for fallible forecasts and actual interaction evidence."""

import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from autoplay_qa.cli import begin_session, main
from autoplay_qa.config import Settings
from autoplay_qa.exploration import CONTROLS, ExplorationLimits, REFLECTION_PROMPT, run_exploration
from autoplay_qa.game import KeyGateGame
from autoplay_qa.nebius import NebiusClient, NebiusError
from autoplay_qa.reflection import outcome_projection
from autoplay_qa.review import review_session
from autoplay_qa.session import SessionLog, replay
from tests.test_exploration import decision, response
from tests.test_strict_exploration import note, pair


def choice(action, *, target="interact", role=None, experiment="interaction", updates=(), expected=None, assessment_patch=None):
    value = decision(action, hypotheses=updates, experiment=experiment,
                     goal="Does the chosen interaction effect repeat under the recorded starting conditions?",
                     prediction="Only the next selected control's immediate visible effect.")
    value.update(experiment_control=target, action_role=role or ("finish" if action == "finish" else "setup" if action == "reset" else "trial"),
                 expected_outcome=expected, fixture_assessment_patch=assessment_patch or {})
    return value


def interaction_choices():
    return [choice("interact"), choice("reset", updates=[note("interact", [1], claim="Interact from the recorded initial state is refused without state changes.")]),
            choice("interact"), choice("finish", updates=[note("interact", [1, 3], pair(1, 3), status="supported",
                                                              claim="Interact from the recorded initial state is refused without state changes.")])]


class ReflectionControllerTests(unittest.TestCase):
    def run_fixture(self, choices, *, limits=ExplorationLimits(), focus="interaction-retention", seed=0, clock=None, response_format="json_schema"):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        path = Path(directory.name) / "fixture.jsonl"
        client = NebiusClient(Settings("unit-test-key", response_format=response_format))
        game = KeyGateGame(seed)
        choices = iter(choices)

        def transport(_path, body=None, timeout=None):
            context = json.loads(body["messages"][1]["content"])
            value = next(choices)
            if isinstance(value, BaseException):
                raise value
            value = copy.deepcopy(value)
            receipt = context["reflection_contract"]["pending_outcome"]
            value["assessment"] = ({
                **{key: receipt[key] for key in ("step", "control", "episode_before", "episode_after", "position_before", "observed_outcome", "observed_events", "prediction_verdict")},
                "outcome": f"Recorded {receipt['control']} feedback compared with the original five-facet forecast: {receipt['prediction_verdict']}.",
            } if receipt is not None else None)
            changes = value.pop("fixture_assessment_patch", {})
            if value["assessment"] is not None:
                value["assessment"].update(changes)
            if value["action"] != "finish" and value["expected_outcome"] is None:
                # A fixture forecasts unchanged public state; it never executes
                # a proposed future action to manufacture a correct prediction.
                value["expected_outcome"] = outcome_projection(context["observation"])
            return response(value, finish=value.pop("fixture_finish", "stop"))

        time_patch = patch("autoplay_qa.exploration.time.monotonic", side_effect=clock) if clock else patch("autoplay_qa.exploration.time.monotonic", wraps=__import__("time").monotonic)
        with SessionLog(path) as log:
            begin_session(log, game, "unit-test-fixtures", schema_version=2, task="explore", focus=focus,
                          comparison_contract=1, reflection_contract=1)
            with patch.object(client, "_request", side_effect=transport) as mocked, time_patch:
                summary = run_exploration(game, client, log, lambda: KeyGateGame(seed), limits, focus=focus,
                                          comparison_contract=True, reflection_contract=True)
        rows = [json.loads(s) for s in path.read_text().splitlines()]
        contexts = [json.loads(row["messages"][1]["content"]) for row in rows if row["type"] == "request"]
        return summary, rows, contexts, path, mocked

    def test_refused_interaction_comparison_is_retained_and_narrow_focus_stays_separate(self):
        summary, rows, contexts, path, _ = self.run_fixture(interaction_choices())
        self.assertEqual((summary["status"], summary["steps"], summary["resets"]), ("finished", 3, 1))
        self.assertIsNone(summary["unassessed_final_step"])
        self.assertEqual(summary["experiment_bindings"], {"interaction": "interact"})
        before_reset = next(row for row in rows if row["type"] == "memory" and row["request_number"] == 2)
        self.assertEqual(before_reset["hypotheses"], contexts[2]["hypotheses"])
        self.assertEqual(contexts[2]["reflection_contract"]["experiments"]["experiments"][0]["trial_steps"], [1])
        self.assertEqual(contexts[2]["reflection_contract"]["pending_outcome"]["observed_events"], ["Environment started."])
        self.assertTrue(all(row["assessed"] for row in summary["reflection_receipts"]))
        self.assertTrue(all(row["prediction_verdict"] == "matched" for row in summary["reflection_receipts"]))
        self.assertEqual(replay(path)["actions_replayed"], 3)
        report = review_session(path, include_evidence=True)
        self.assertTrue(report["goal_checks"]["focused_question_observed"])
        self.assertFalse(report["goal_checks"]["reset_retested_control_families"]["movement"])
        self.assertEqual(report["goal_checks"]["next_stage_candidate"], "exploration_validation")
        self.assertEqual([row["decision"]["action_role"] for row in rows if row["type"] in {"step", "reset"}], ["trial", "setup", "trial"])

    def test_wrong_future_forecast_executes_and_is_assessed_as_learning_evidence(self):
        choices = interaction_choices()
        incorrect = outcome_projection(KeyGateGame().observe())
        incorrect.update(key=None, inventory=["key"], gate_open=True)
        choices[0]["expected_outcome"] = incorrect
        summary, _, contexts, path, _ = self.run_fixture(choices)
        self.assertEqual(summary["status"], "finished")
        self.assertEqual(summary["rejected_decisions"], 0)
        self.assertEqual(contexts[1]["reflection_contract"]["pending_outcome"]["expected_outcome"], incorrect)
        self.assertEqual(contexts[1]["reflection_contract"]["pending_outcome"]["prediction_verdict"], "mismatched")
        self.assertEqual(summary["reflection_receipts"][0]["prediction_verdict"], "mismatched")
        self.assertEqual(summary["reflection_receipts"][2]["prediction_verdict"], "matched")
        self.assertEqual(replay(path)["actions_replayed"], 3)

    def test_false_match_rejected_atomically_then_repaired_without_new_game_step(self):
        choices = interaction_choices()
        incorrect = outcome_projection(KeyGateGame().observe())
        incorrect["inventory"] = ["key"]
        choices[0]["expected_outcome"] = incorrect
        bad = copy.deepcopy(choices[1])
        bad["fixture_assessment_patch"] = {"prediction_verdict": "matched"}
        bad["experiment_id"] = "REJECTED_EXPERIMENT"
        choices.insert(1, bad)
        summary, rows, contexts, path, mocked = self.run_fixture(choices)
        self.assertEqual((summary["status"], summary["rejected_decisions"], mocked.call_count), ("finished", 1, 5))
        self.assertEqual(contexts[1]["reflection_contract"], contexts[2]["reflection_contract"])
        self.assertEqual(contexts[1]["observation"], contexts[2]["observation"])
        self.assertEqual(contexts[1]["hypotheses"], contexts[2]["hypotheses"])
        self.assertNotIn("REJECTED_EXPERIMENT", summary["experiment_bindings"])
        self.assertFalse(any(row["type"] in {"memory", "step", "reset", "finish"} and row["request_number"] == 2 for row in rows))
        self.assertEqual(replay(path)["actions_replayed"], 3)

    def test_setup_is_not_a_game_trial_and_experiment_targets_cannot_change(self):
        bad = choice("north", target="north", experiment="interaction")
        valid = choice("finish", updates=[note("interact", [1])])
        summary, _, contexts, path, _ = self.run_fixture([choice("interact"), bad, valid])
        self.assertEqual((summary["steps"], summary["rejected_decisions"]), (1, 1))
        self.assertIn("retain its declared experiment_control", contexts[2]["validation_feedback"])
        self.assertEqual(summary["experiment_bindings"], {"interaction": "interact"})
        self.assertEqual(replay(path)["actions_replayed"], 1)

    def test_budget_interruption_and_api_error_preserve_pending_typed_result(self):
        for exception, expected in ((NebiusError("Fixture unavailable"), "inference_error"), (KeyboardInterrupt(), "interrupted")):
            with self.subTest(status=expected):
                summary, _, _, path, _ = self.run_fixture([choice("interact"), exception])
                self.assertEqual((summary["status"], summary["unassessed_final_step"]), (expected, 1))
                self.assertFalse(summary["usage_complete"])
                self.assertFalse(summary["reflection_receipts"][0]["assessed"])
                self.assertEqual(replay(path)["actions_replayed"], 1)
        summary, _, _, path, _ = self.run_fixture([choice("reset")], limits=ExplorationLimits(max_actions=1))
        self.assertEqual((summary["status"], summary["unassessed_final_step"]), ("action_limit", 1))
        self.assertEqual(summary["reflection_receipts"][0]["observed_events"], ["Environment started."])
        self.assertEqual(replay(path)["actions_replayed"], 1)

    def test_deadline_after_response_changes_neither_binding_memory_nor_game(self):
        summary, rows, _, path, _ = self.run_fixture([choice("interact")], limits=ExplorationLimits(max_seconds=2),
                                                   clock=[0, 0, 0, 1, 3, 3, 3, 3, 3, 3])
        self.assertEqual((summary["status"], summary["steps"]), ("time_limit", 0))
        self.assertEqual(summary["experiment_bindings"], {})
        self.assertEqual(summary["reflection_receipts"], [])
        self.assertFalse(any(row["type"] == "memory" for row in rows))
        self.assertEqual(replay(path)["actions_replayed"], 0)

    def test_reflection_context_holds_no_spoilers_or_selected_route(self):
        _, rows, contexts, _, _ = self.run_fixture(interaction_choices())
        for context in contexts:
            self.assertEqual(context["observation"]["controls"], list(CONTROLS))
            self.assertEqual(set(context["reflection_contract"]), {"contract_version", "pending_outcome", "experiments"})
            text = json.dumps(context["reflection_contract"])
            for withheld in ('"rules"', '"goal"', '"passable"', "Find the key", "unlock the gate"):
                self.assertNotIn(withheld, text)
        first_request = next(row for row in rows if row["type"] == "request")
        schema = first_request["response_schema"]
        self.assertEqual(list(schema["properties"]).index("experiment_control"), 3)
        self.assertEqual(schema["properties"]["action"]["enum"], list(CONTROLS))
        self.assertLessEqual(len(REFLECTION_PROMPT.split()), 350)
        branches = schema["anyOf"]
        trials = [branch["properties"] for branch in branches if branch["properties"]["action_role"]["enum"] == ["trial"]]
        self.assertEqual({branch["action"]["enum"][0] for branch in trials}, set(CONTROLS) - {"reset", "finish"})
        for branch in trials:
            self.assertEqual(branch["action"]["enum"], branch["experiment_control"]["enum"])
            self.assertEqual(len(branch["action"]["enum"]), 1)
            self.assertEqual(branch["expected_outcome"]["type"], "object")
        setup = next(branch["properties"] for branch in branches if branch["properties"]["action_role"]["enum"] == ["setup"])
        self.assertEqual(set(setup["action"]["enum"]), set(CONTROLS) - {"finish"})
        self.assertEqual(set(setup["experiment_control"]["enum"]), set(CONTROLS) - {"reset", "finish"})
        self.assertEqual(setup["expected_outcome"]["type"], "object")
        final = branches[-1]["properties"]
        self.assertEqual(final["action"]["enum"], ["finish"])
        self.assertEqual(final["expected_outcome"], {"type": "null"})
        self.assertEqual(final["action_role"]["enum"], ["finish"])
        for branch in branches:
            self.assertEqual(set(branch["required"]), set(branch["properties"]))
            self.assertFalse(branch["additionalProperties"])
        requests = [row for row in rows if row["type"] == "request"]
        later = requests[1]["response_schema"]["properties"]["assessment"]
        self.assertEqual(later["properties"]["prediction_verdict"]["enum"], [contexts[1]["reflection_contract"]["pending_outcome"]["prediction_verdict"]])

    def test_interaction_focus_cli_preserves_defaults_and_enables_versioned_contract(self):
        with patch("autoplay_qa.cli.explore", return_value=0) as handler:
            self.assertEqual(main(["explore", "--focus", "interaction-retention"]), 0)
        args = handler.call_args.args[0]
        self.assertEqual((args.max_actions, args.max_requests, args.max_seconds, args.max_tokens, args.timeout), (24, 28, 300, 8192, 45))

    def test_first_observed_note_prefix_retains_proposals_and_cannot_cite_untried_control(self):
        choices = interaction_choices()
        choices[1]["hypotheses"].append(note("east", status="untested"))
        summary, rows, contexts, path, _ = self.run_fixture(choices)
        self.assertEqual(summary["status"], "finished")
        requests = [row for row in rows if row["type"] == "request"]
        self.assertNotIn("prefixItems", requests[0]["response_schema"]["properties"]["hypotheses"])
        array = requests[1]["response_schema"]["properties"]["hypotheses"]
        first = array["prefixItems"][0]
        self.assertEqual(first["properties"]["status"]["enum"], ["uncertain"])
        self.assertEqual(first["properties"]["evidence_steps"]["minItems"], 1)
        self.assertEqual({branch["properties"]["control"]["enum"][0] for branch in first["anyOf"]}, {"interact"})
        self.assertEqual(set(first["required"]), {"id", "claim", "status", "evidence_steps", "control", "comparison"})
        self.assertIn("untested", array["items"]["properties"]["status"]["enum"])
        self.assertEqual(summary["hypotheses"][1]["status"], "untested")
        self.assertEqual(replay(path)["actions_replayed"], 3)

    def test_guided_mode_explains_compact_contract_and_still_transmits_formal_schema(self):
        _, rows, contexts, _, transport = self.run_fixture(interaction_choices())
        requests = [row for row in rows if row["type"] == "request"]
        for row, context, call in zip(requests, contexts, transport.call_args_list):
            self.assertNotIn("response_schema", context)
            contract = context["response_contract"]
            self.assertEqual(contract["required_fields"], row["response_schema"]["required"])
            self.assertEqual(contract["action"], row["response_schema"]["properties"]["action"]["enum"])
            self.assertEqual(call.args[1]["response_format"]["json_schema"]["schema"], row["response_schema"])
            self.assertLess(len(json.dumps(contract)), len(json.dumps(row["response_schema"])) / 3)
            self.assertIn("authoritative", contract["expected_outcome"]["rule"])
            self.assertTrue(all("prediction" not in outcome for outcome in context["recent_outcomes"]))
            if context["completed_outcome_to_assess"]:
                completed = context["completed_outcome_to_assess"]
                self.assertNotIn("prediction_made_before_it", completed)
                self.assertEqual(completed["expected_outcome"], context["reflection_contract"]["pending_outcome"]["expected_outcome"])
                self.assertIn("prediction_explanation", completed)
        self.assertFalse(contexts[0]["response_contract"]["hypotheses"]["first_observed_update_required"])
        self.assertTrue(contexts[1]["response_contract"]["hypotheses"]["first_observed_update_required"])
        progress = contexts[-1]["question_progress"]["declared_trial_progress"]["experiments"][0]
        self.assertEqual(progress["trial_episodes"], [0, 1])
        self.assertEqual(progress["trial_count"], 2)
        self.assertEqual(progress["assessed_cross_episode_pair_count"], 0)

    def test_plain_and_object_modes_keep_full_schema_and_all_controls(self):
        for mode in ("plain", "json_object"):
            with self.subTest(mode=mode):
                summary, rows, contexts, path, _ = self.run_fixture(interaction_choices(), response_format=mode)
                self.assertEqual(summary["status"], "finished")
                requests = [row for row in rows if row["type"] == "request"]
                for row, context in zip(requests, contexts):
                    self.assertNotIn("response_contract", context)
                    self.assertEqual(context["response_schema"], row["response_schema"])
                    self.assertEqual(context["observation"]["controls"], list(CONTROLS))
                self.assertEqual(replay(path)["actions_replayed"], 3)


if __name__ == "__main__":
    unittest.main()
