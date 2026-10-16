"""End-to-end fixtures for one forecast and model-selected executable plans."""

import copy
from dataclasses import asdict
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from autoplay_qa.cli import begin_session
from autoplay_qa.config import Settings
from autoplay_qa.exploration import CONTROLS, ExplorationDecision, ExplorationLimits, GROUNDED_PROMPT, STATE_PROMPT, run_exploration
from autoplay_qa.forecast import DECISION_FIELDS, PREDICTION_SOURCE, outcome_projection
from autoplay_qa.game import KeyGateGame
from autoplay_qa.nebius import InvalidDecision, NebiusClient, NebiusError
from autoplay_qa.review import review_session
from autoplay_qa.session import SessionLog, replay
from tests.test_exploration import response
from tests.test_strict_exploration import note, pair


def plan(controls, mode="new", *, evidence=(), reason=""):
    return {"mode": mode, "controls": list(controls), "revision_reason": reason, "evidence_steps": list(evidence)}


def choice(action, controls=None, *, mode="new", target="interact", experiment="interaction", updates=(), expected=None, revision=(), revision_reason="", assessment_patch=None):
    return {
        "action": action, "reason": "Probe the selected effect under the visible conditions.",
        "plan": plan(controls or [action], mode, evidence=revision, reason=revision_reason),
        "experiment_id": experiment, "experiment_control": target,
        "experiment_goal": "Does this chosen effect repeat from the same conditions in another episode?",
        "action_role": "finish" if action == "finish" else "trial" if action == target else "setup",
        "expected_outcome": expected, "hypotheses": list(updates), "assessment": None,
        "fixture_assessment_patch": assessment_patch or {},
    }


def both_questions():
    movement = note("north", [1], claim="North from the recorded start is blocked without position or inventory change.")
    movement_pair = note("north", [1, 3], pair(1, 3), status="supported", claim=movement["claim"])
    interaction = note("interact", [4], claim="Interact from the recorded start is refused without state change.")
    interaction_pair = note("interact", [4, 6], pair(4, 6, 1, 2), status="supported", claim=interaction["claim"])
    # Distinct IDs for genuinely different sampled control-family questions.
    movement["id"] = movement_pair["id"] = "movement-note"
    interaction["id"] = interaction_pair["id"] = "interaction-note"
    return [
        choice("north", ["north", "reset", "north"], target="north", experiment="movement"),
        choice("reset", ["reset", "north"], mode="continue", target="north", experiment="movement", updates=[movement]),
        choice("north", ["north"], mode="continue", target="north", experiment="movement"),
        choice("interact", ["interact", "reset", "interact", "finish"], updates=[movement_pair]),
        choice("reset", ["reset", "interact", "finish"], mode="continue", updates=[interaction]),
        choice("interact", ["interact", "finish"], mode="continue"),
        choice("finish", ["finish"], mode="continue", updates=[interaction_pair]),
    ]


class GroundedControllerTests(unittest.TestCase):
    def run_fixture(self, choices, *, seed=0, limits=ExplorationLimits(), clock=None, mode="json_schema", version=2):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        path = Path(directory.name) / "fixture.jsonl"
        game = KeyGateGame(seed)
        client = NebiusClient(Settings("unit-test-key", response_format=mode))
        choices = iter(choices)

        def transport(_path, body=None, timeout=None):
            context = json.loads(body["messages"][1]["content"])
            value = next(choices)
            if isinstance(value, BaseException):
                raise value
            value = copy.deepcopy(value)
            receipt = context["reflection_contract"]["pending_outcome"]
            value["assessment"] = ({
                "step": receipt["step"], "prediction_verdict": receipt["prediction_verdict"],
                "mismatched_facets": receipt["mismatched_facets"],
                "outcome": f"Actual feedback: {receipt['observed_events']}. The original forecast differed in {receipt['mismatched_facets']}.",
            } if receipt else None)
            if receipt is not None and version == 4:
                value["assessment"]["actual_changed_facets"] = receipt["actual_changed_facets"]
            changes = value.pop("fixture_assessment_patch", {})
            if value["assessment"] is not None:
                value["assessment"].update(changes)
            if value["action"] != "finish" and value["expected_outcome"] is None:
                value["expected_outcome"] = outcome_projection(context["observation"])
            return response(value, finish=value.pop("fixture_finish", "stop"))

        time_patch = patch("autoplay_qa.exploration.time.monotonic", side_effect=clock) if clock else patch("autoplay_qa.exploration.time.monotonic", wraps=__import__("time").monotonic)
        with SessionLog(path) as log:
            begin_session(log, game, "unit-test-fixtures", schema_version=2, task="explore", focus="reset-retention", comparison_contract=1, reflection_contract=version)
            with patch.object(client, "_request", side_effect=transport) as mocked, time_patch:
                summary = run_exploration(game, client, log, lambda: KeyGateGame(seed), limits, focus="reset-retention", comparison_contract=True, reflection_contract=version)
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        contexts = [json.loads(row["messages"][1]["content"]) for row in rows if row["type"] == "request"]
        return summary, rows, contexts, path, mocked

    def test_both_families_retain_actual_learning_and_self_selected_plans_across_resets(self):
        for seed in (0, 5):
            with self.subTest(seed=seed):
                summary, rows, contexts, path, _ = self.run_fixture(both_questions(), seed=seed)
                self.assertEqual((summary["status"], summary["steps"], summary["resets"]), ("finished", 6, 2))
                self.assertIsNone(summary["unassessed_final_step"])
                self.assertEqual([r["decision"]["action"] for r in rows if r["type"] in {"step", "reset"}], ["north", "reset", "north", "interact", "reset", "interact"])
                self.assertEqual(contexts[1]["research_context"]["pending_plan"]["controls"], ["reset", "north"])
                self.assertEqual(contexts[2]["research_context"]["pending_plan"]["controls"], ["north"])
                self.assertNotIn("plan", contexts[2]["active_experiment"])
                self.assertEqual(contexts[1]["hypotheses"], [])
                self.assertEqual(contexts[2]["hypotheses"][0]["evidence_steps"], [1])
                self.assertEqual(contexts[5]["hypotheses"][1]["evidence_steps"], [4])
                self.assertTrue(summary["research_context"]["session_finished"])
                self.assertIsNone(summary["research_context"]["pending_plan"])
                self.assertEqual(replay(path)["actions_replayed"], 6)
                checks = review_session(path)["goal_checks"]
                self.assertTrue(checks["focused_question_observed"])
                self.assertEqual(checks["reset_retested_control_families"], {"movement": True, "interaction": True})

    def test_wrong_forecast_executes_and_compact_assessment_names_only_real_mismatched_facet(self):
        expected = outcome_projection(KeyGateGame().observe())
        expected["key"] = None
        values = [choice("interact", expected=expected), choice("finish", updates=[note("interact", [1])])]
        summary, rows, contexts, path, _ = self.run_fixture(values)
        self.assertEqual(summary["status"], "finished")
        receipt = contexts[1]["reflection_contract"]["pending_outcome"]
        self.assertEqual(receipt["mismatched_facets"], ["key"])
        self.assertFalse(receipt["expected_outcome"]["gate_open"])
        step = next(row for row in rows if row["type"] == "step")
        self.assertEqual(set(step["decision"]), DECISION_FIELDS)
        self.assertEqual(step["prediction_source"], PREDICTION_SOURCE)
        self.assertIn("gate_open=false", step["prediction"])
        self.assertNotIn("prediction", step["decision"])
        self.assertEqual(set(next(row for row in rows if row["type"] == "memory" and row["assessment"])["assessment"]), {"step", "outcome", "prediction_verdict", "mismatched_facets"})
        self.assertEqual(replay(path)["actions_replayed"], 1)

    def test_repeating_instead_of_recorded_reset_is_atomic_and_only_one_repair_runs(self):
        first = choice("interact", ["interact", "reset", "interact"])
        malformed = choice("interact", ["interact"], mode="continue", updates=[note("interact", [1], claim="Malformed proposal must never commit.")])
        repaired = choice("reset", ["reset", "interact"], mode="continue", updates=[note("interact", [1])])
        finish = choice("finish", mode="revise", revision=[2], revision_reason="Stop this fixture after the actual reset while retaining the earlier observation.")
        summary, rows, contexts, path, _ = self.run_fixture([first, malformed, repaired, finish])
        self.assertEqual((summary["status"], summary["steps"], summary["rejected_decisions"]), ("finished", 2, 1))
        self.assertEqual([r["decision"]["action"] for r in rows if r["type"] in {"step", "reset"}], ["interact", "reset"])
        self.assertEqual(contexts[1]["research_context"], contexts[2]["research_context"])
        self.assertEqual(contexts[1]["hypotheses"], contexts[2]["hypotheses"])
        self.assertFalse(any("Malformed proposal" in item["claim"] for item in summary["hypotheses"]))
        self.assertEqual(replay(path)["actions_replayed"], 2)

    def test_voluntary_evidence_linked_revision_keeps_all_current_controls_available(self):
        values = [choice("interact", ["interact", "reset", "interact"]),
                  choice("east", mode="revise", revision=[1], revision_reason="The interaction was refused; investigate a different visible condition.", updates=[note("interact", [1])]),
                  choice("finish")]
        summary, _, contexts, path, _ = self.run_fixture(values)
        self.assertEqual((summary["status"], summary["steps"]), ("finished", 2))
        self.assertEqual(contexts[1]["observation"]["controls"], list(CONTROLS))
        self.assertEqual(summary["research_context"]["plan_revisions"], 1)
        self.assertEqual(replay(path)["actions_replayed"], 2)

    def test_model_written_prediction_or_source_is_rejected_before_any_state_update(self):
        value = choice("interact")
        value.pop("fixture_assessment_patch")
        value["expected_outcome"] = outcome_projection(KeyGateGame().observe())
        for extra in ("prediction", "prediction_source"):
            with self.subTest(extra=extra), self.assertRaises(InvalidDecision):
                ExplorationDecision.parse(json.dumps({**value, extra: "another forecast"}), list(CONTROLS), set(), None, comparison_records={}, reflection_records={}, reflection_version=2)
        parsed = ExplorationDecision.parse(json.dumps(value), list(CONTROLS), set(), None, comparison_records={}, reflection_records={}, reflection_version=2)
        self.assertEqual(set(asdict(parsed)), DECISION_FIELDS)
        self.assertIn("Expected after interact", parsed.prediction)

    def test_false_mismatched_facets_get_one_atomic_repair(self):
        expected = outcome_projection(KeyGateGame().observe())
        expected["key"] = None
        bad = choice("finish", updates=[note("interact", [1])], assessment_patch={"mismatched_facets": ["gate_open"]})
        good = choice("finish", updates=[note("interact", [1])])
        summary, _, contexts, path, _ = self.run_fixture([choice("interact", expected=expected), bad, good])
        self.assertEqual((summary["status"], summary["steps"], summary["rejected_decisions"]), ("finished", 1, 1))
        self.assertEqual(contexts[1]["reflection_contract"], contexts[2]["reflection_contract"])
        self.assertEqual(replay(path)["actions_replayed"], 1)

    def test_api_interruption_and_budget_preserve_actual_pending_plan_without_autorun(self):
        for failure, status in ((NebiusError("Fixture network error"), "inference_error"), (KeyboardInterrupt(), "interrupted")):
            with self.subTest(status=status):
                summary, rows, _, path, _ = self.run_fixture([choice("interact", ["interact", "reset", "interact"]), failure])
                self.assertEqual((summary["status"], summary["steps"]), (status, 1))
                self.assertEqual(summary["research_context"]["pending_plan"]["controls"], ["reset", "interact"])
                self.assertEqual(summary["unassessed_final_step"], 1)
                self.assertFalse(summary["usage_complete"])
                self.assertFalse(any(r["type"] == "reset" for r in rows))
                self.assertEqual(replay(path)["actions_replayed"], 1)
        summary, _, _, path, _ = self.run_fixture([choice("interact", ["interact", "reset", "interact"])], limits=ExplorationLimits(max_requests=1))
        self.assertEqual((summary["status"], summary["steps"]), ("request_limit", 1))
        self.assertEqual(summary["research_context"]["pending_plan"]["controls"], ["reset", "interact"])
        self.assertEqual(replay(path)["actions_replayed"], 1)

    def test_deadline_rejects_proposed_plan_without_consuming_or_committing_it(self):
        summary, rows, _, path, _ = self.run_fixture([choice("interact", ["interact", "reset", "interact"])], limits=ExplorationLimits(max_seconds=2), clock=[0, 0, 0, 1, 3, 3, 3, 3, 3, 3])
        self.assertEqual((summary["status"], summary["steps"]), ("time_limit", 0))
        self.assertIsNone(summary["research_context"]["pending_plan"])
        self.assertEqual(summary["experiment_bindings"], {})
        self.assertFalse(any(row["type"] == "memory" for row in rows))
        self.assertEqual(replay(path)["actions_replayed"], 0)

    def test_grounded_messages_withhold_hints_and_do_not_supply_a_plan_or_future_answer(self):
        _, rows, contexts, path, transport = self.run_fixture(both_questions())
        self.assertLessEqual(len(GROUNDED_PROMPT.split()), 450)
        for row, context, call in zip([r for r in rows if r["type"] == "request"], contexts, transport.call_args_list):
            schema = row["response_schema"]
            self.assertEqual(set(schema["required"]), DECISION_FIELDS)
            self.assertNotIn("prediction", schema["properties"])
            self.assertEqual(call.args[1]["response_format"]["json_schema"]["schema"], schema)
            self.assertNotIn("prediction", context["response_contract"]["string_limits"])
            text = json.dumps(context)
            self.assertNotIn("goal", context["observation"])
            for withheld in ('"rules"', '"passable"', 'Find the key', 'unlock the gate'):
                self.assertNotIn(withheld, text)
            self.assertNotIn("prediction_explanation", text)
            self.assertFalse(context["research_context"]["claims_verified"])
            self.assertEqual(context["observation"]["controls"], list(CONTROLS))
        self.assertIsNone(contexts[0]["research_context"]["pending_plan"])
        self.assertEqual(replay(path)["actions_replayed"], 6)

    def test_v3_actual_changes_are_distinct_from_forecast_mismatches_and_layout_is_labeled(self):
        expected = outcome_projection(KeyGateGame().observe())
        expected["key"] = None
        summary, rows, contexts, path, _ = self.run_fixture([choice("interact", expected=expected), choice("finish", updates=[note("interact", [1])])], version=3)
        self.assertEqual(summary["status"], "finished")
        receipt = contexts[1]["reflection_contract"]["pending_outcome"]
        self.assertEqual(receipt["before_outcome"]["key"], [1, 3])
        self.assertEqual(receipt["observed_outcome"]["key"], [1, 3])
        self.assertEqual(receipt["before_outcome"]["inventory"], [])
        self.assertEqual(receipt["actual_changed_facets"], [])
        self.assertEqual(receipt["mismatched_facets"], ["key"])
        completions = [row for row in rows if row["type"] == "completion"]
        for completion in completions:
            self.assertEqual(set(completion["response_field_order"]), DECISION_FIELDS)
            for order in completion["hypothesis_field_orders"]:
                self.assertEqual(set(order), {"id", "claim", "status", "evidence_steps", "control", "comparison"})
        self.assertEqual(replay(path)["actions_replayed"], 1)

    def test_v3_schema_conditions_text_on_chosen_controls_evidence_and_plan(self):
        summary, rows, contexts, path, _ = self.run_fixture(both_questions(), version=3, seed=5)
        self.assertEqual(summary["status"], "finished")
        for request in (r for r in rows if r["type"] == "request"):
            schema = request["response_schema"]
            fields = list(schema["properties"])
            self.assertLess(fields.index("plan"), fields.index("action"))
            self.assertLess(fields.index("action"), fields.index("experiment_control"))
            self.assertLess(fields.index("experiment_control"), fields.index("experiment_id"))
            self.assertLess(fields.index("experiment_control"), fields.index("experiment_goal"))
            for note_schema in (schema["properties"]["hypotheses"]["items"], *schema["properties"]["hypotheses"].get("prefixItems", [])):
                for variant in (note_schema, *note_schema["anyOf"]):
                    names = list(variant["properties"])
                    self.assertLess(names.index("control"), names.index("claim"))
                    self.assertLess(names.index("evidence_steps"), names.index("claim"))
        self.assertLessEqual(len(STATE_PROMPT.split()), 500)
        self.assertIn("visible WORLD", contexts[0]["response_contract"]["expected_outcome"]["types"])
        self.assertEqual(replay(path)["actions_replayed"], 6)

    def test_scoped_schema_keeps_continuation_identity_and_allows_voluntary_revisions(self):
        _, rows, contexts, _, _ = self.run_fixture(both_questions(), version=3)
        requests = [r for r in rows if r["type"] == "request"]
        for request, context in zip(requests[1:], contexts[1:]):
            bindings = {item["id"]: item["control"] for item in context["reflection_contract"]["experiments"]["experiments"]}
            pending = context["research_context"]["pending_plan"]
            revision_heads = set()
            for variant in request["response_schema"]["anyOf"]:
                properties = variant["properties"]
                mode = properties["plan"]["properties"]["mode"]["enum"][0]
                identifier = properties["experiment_id"]
                if mode == "continue":
                    self.assertEqual(identifier["enum"], [pending["experiment_id"]])
                    self.assertEqual(properties["experiment_control"]["enum"], [bindings[pending["experiment_id"]]])
                    self.assertEqual(properties["plan"]["properties"]["controls"]["enum"], [pending["controls"]])
                elif "enum" in identifier:
                    for existing in identifier["enum"]:
                        self.assertEqual(properties["experiment_control"]["enum"], [bindings[existing]])
                else:
                    self.assertEqual(identifier["not"]["enum"], sorted(bindings))
                if mode == "revise":
                    revision_heads.update(properties["action"]["enum"])
                self.assertEqual(set(variant["required"]), DECISION_FIELDS)
                self.assertFalse(variant["additionalProperties"])
            self.assertEqual(revision_heads, set(context["observation"]["controls"]))

    def test_planned_control_can_continue_old_question_or_explicitly_revise_to_new_question(self):
        values = [choice("east", ["east", "interact"], target="east", experiment="movement"),
                  choice("interact", ["interact"], mode="continue", target="east", experiment="movement", updates=[note("east", [1])]),
                  choice("finish")]
        _, rows, _, path, _ = self.run_fixture(values, version=3)
        schema = [r for r in rows if r["type"] == "request"][1]["response_schema"]
        continuations = [v["properties"] for v in schema["anyOf"] if v["properties"]["plan"]["properties"]["mode"]["enum"] == ["continue"]]
        self.assertTrue(continuations)
        for branch in continuations:
            self.assertEqual(branch["action"]["enum"], ["interact"])
            self.assertEqual(branch["action_role"]["enum"], ["setup"])
            self.assertEqual(branch["experiment_control"]["enum"], ["east"])
            self.assertEqual(branch["experiment_id"]["enum"], ["movement"])
        fresh_interaction = [v["properties"] for v in schema["anyOf"] if v["properties"]["plan"]["properties"]["mode"]["enum"] == ["revise"] and v["properties"]["action_role"]["enum"] == ["trial"] and v["properties"]["action"]["enum"] == ["interact"]]
        self.assertTrue(fresh_interaction)
        self.assertTrue(all(v["experiment_id"]["not"]["enum"] == ["movement"] for v in fresh_interaction))
        self.assertEqual(replay(path)["actions_replayed"], 2)


if __name__ == "__main__":
    unittest.main()
