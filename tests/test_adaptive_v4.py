"""Safe repair feedback and definition-scoped trials, using only stub HTTP."""

import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from autoplay_qa.adaptive import adaptive_input_bytes, expand_observation
from autoplay_qa.adaptive_v3 import AdaptiveContextV3
from autoplay_qa.adaptive_v4 import AdaptiveContextV4, definition_signature
from autoplay_qa.cli import begin_session
from autoplay_qa.config import Settings
from autoplay_qa.experience import public_observation
from autoplay_qa.exploration import ExplorationLimits
from autoplay_qa.game import KeyGateGame
from autoplay_qa.investigation import public_field
from autoplay_qa.lean import decision_dict, run_lean_exploration
from autoplay_qa.nebius import NebiusClient
from autoplay_qa.session import SessionLog, replay
from tests.test_adaptive_context import build, observation, record
from tests.test_adaptive_v3 import question, forecast, decision
from tests.test_lean_exploration import response


def acknowledged(context, step, before, after, *, update=None, predictions=()):
    build(context, before)
    parsed, warnings = context.parse_decision(json.dumps(decision(update=update, predictions=predictions)), context.allowed(before), set())
    context.observe_decision_warnings(step, warnings)
    row = record(step, before, parsed.action, after)
    row["request_number"] = step
    row["decision"] = decision_dict(parsed)
    context.record(row)
    return row, warnings


class AdaptiveV4FeedbackTests(unittest.TestCase):
    def test_version_is_opt_in_and_v3_context_has_no_new_protocol_fields(self):
        current = observation()
        _, old = build(AdaptiveContextV3(current), current)
        _, new = build(AdaptiveContextV4(current), current)
        self.assertEqual(old["prompt_version"], "adaptive-v3")
        self.assertEqual(new["prompt_version"], "adaptive-v4")
        self.assertNotIn("metadata_feedback", old)
        self.assertNotIn("metadata_feedback_omitted", old["context_compaction"])
        self.assertNotIn("definition_scope", old["investigation_memory"])
        self.assertEqual(new["metadata_feedback"]["status"], "no_previous_response")

    def test_feedback_uses_only_safe_fixed_explanations_and_current_public_examples(self):
        initial = observation(entities=[{"id": "old-panel", "state": {"open": False}}])
        context = AdaptiveContextV4(initial, secrets=("runtime-secret-123",))
        context.observe_decision_warnings(1, [
            {"code": "invalid_prediction", "message": "runtime-secret-123 /Users/private/source.py sk-abcdefghijklmnopqrstuv"},
            {"code": "runtime-secret-123", "message": "do something hidden"},
        ])
        current = observation(entities=[{"id": "new-panel", "state": {"open": True}}])
        messages, value = build(context, current)
        feedback = value["metadata_feedback"]
        self.assertEqual([item["code"] for item in feedback["issues"]], ["invalid_prediction"])
        self.assertLessEqual(len(feedback["public_selector_examples"]), 6)
        self.assertTrue(feedback["public_selector_examples"])
        for path in feedback["public_selector_examples"]:
            public_field(current, path)
            self.assertNotIn("old-panel", path)
        serialized = json.dumps(messages) + json.dumps(context.summary())
        for private in ("runtime-secret-123", "/Users/private", "sk-abcdefghijklmnopqrstuv", "do something hidden"):
            self.assertNotIn(private, serialized)

    def test_warning_packet_is_bounded_and_ordinary_valid_response_clears_old_repair(self):
        current = observation()
        context = AdaptiveContextV4(current)
        codes = ["invalid_investigation_update", "invalid_predictions", "too_many_predictions", "invalid_prediction",
                 "investigation_requires_revision", "investigation_update_not_applied", "invalid_expected_position"]
        context.observe_decision_warnings(2, [{"code": code, "message": "x" * 10_000} for code in codes])
        _, value = build(context, current)
        self.assertEqual(len(value["metadata_feedback"]["issues"]), 6)
        self.assertEqual(value["metadata_feedback"]["issues_omitted_count"], 1)
        self.assertLess(len(json.dumps(value["metadata_feedback"])), 1800)
        context.observe_decision_warnings(3, [])
        _, value = build(context, current)
        self.assertEqual(value["metadata_feedback"]["status"], "accepted_metadata")
        self.assertEqual(value["metadata_feedback"]["issues"], [])
        self.assertEqual(value["metadata_feedback"]["public_selector_examples"], [])

    def test_selector_examples_exclude_declared_telemetry_private_values_and_array_indexes(self):
        current = observation(entities=[{"id": "panel", "state": {"open": False, "execution": {"count": 5}, "secret": "hidden"}}])
        current["_capabilities"]["state_identity"] = {"transient_paths": ["entities.panel.state.execution"]}
        context = AdaptiveContextV4(current)
        context.observe_decision_warnings(1, [{"code": "invalid_prediction"}])
        _, value = build(context, current)
        paths = value["metadata_feedback"]["public_selector_examples"]
        for path in paths:
            self.assertNotIn(".execution", path)
            self.assertNotIn(".secret", path)
            self.assertNotIn(".0", path)
            self.assertNotIn(path, {"tick", "events", "controls"})

    def test_selector_field_names_cannot_echo_known_runtime_credentials(self):
        current = observation()
        current["player"]["runtime-credential-123"] = True
        context = AdaptiveContextV4(current, secrets=("runtime-credential-123",))
        context.observe_decision_warnings(1, [{"code": "invalid_prediction"}])
        examples, _ = context._selector_examples(current)
        self.assertFalse(any("runtime-credential-123" in path for path in examples))


class AdaptiveV4DefinitionTests(unittest.TestCase):
    def test_continue_cannot_repurpose_id_and_failed_update_cannot_count_as_old_question(self):
        first = observation()
        context = AdaptiveContextV4(first)
        one, _ = acknowledged(context, 1, first, observation(tick=1), update=question())
        changed = {"id": "q1", "decision": "continue", "question": "Does a different visible component react?"}
        two, warnings = acknowledged(context, 2, one["after"], observation(tick=2), update=changed)
        self.assertIn("investigation_requires_revision", {item["code"] for item in warnings})
        self.assertEqual(two["investigation_fact"]["update"]["status"], "dropped")
        self.assertIsNone(two["investigation_fact"]["investigation_id"])
        self.assertIsNone(two["investigation_fact"]["condition_trial"])
        self.assertEqual(context._investigations[0]["last_transition"]["step"], 1)
        _, value = build(context, two["after"])
        self.assertIn("investigation_requires_revision", [item["code"] for item in value["metadata_feedback"]["issues"]])
        self.assertEqual(context.summary()["executed_controls"], 2)

    def test_explicit_revision_separates_same_id_trials_and_keeps_negative_citations(self):
        first = observation()
        context = AdaptiveContextV4(first)
        one, _ = acknowledged(context, 1, first, observation(tick=1), update=question(), predictions=[forecast()])
        two, _ = acknowledged(context, 2, one["after"], observation(tick=2))
        revision = {"id": "q1", "decision": "revise", "expected_result": "The selected field stays unchanged"}
        three, warnings = acknowledged(context, 3, two["after"], observation(tick=3), update=revision)
        self.assertEqual(warnings, [])
        old = one["investigation_fact"]["condition_trial"]
        new = three["investigation_fact"]["condition_trial"]
        self.assertEqual(two["investigation_fact"]["condition_trial"]["previous_same_condition_action_count"], 1)
        self.assertNotEqual(old["definition_signature"], new["definition_signature"])
        self.assertEqual(new["previous_same_condition_action_count"], 0)
        self.assertEqual(new["other_conditions_for_action"], 0)
        self.assertEqual(context._investigations[0]["counterexamples"], [{"run_id": "session", "step": 1}])
        _, value = build(context, three["after"])
        self.assertEqual(value["investigation_memory"]["definition_scope"]["active_definition_signature"], new["definition_signature"])
        self.assertEqual(value["investigation_memory"]["current_condition_trials"]["trials"][0]["attempts"], 1)
        self.assertEqual(context.summary()["investigation_definitions"]["observed_definitions"], 2)

    def test_changed_facets_and_explanations_also_require_revision(self):
        for edits in ({"condition_facets": ["player.position"]}, {"explanations": ["Another explanation"]}, {"expected_result": "A changed forecast"}):
            with self.subTest(edits=edits):
                current = observation()
                context = AdaptiveContextV4(current)
                acknowledged(context, 1, current, observation(tick=1), update=question())
                update = {"id": "q1", "decision": "continue", **edits}
                row, warnings = acknowledged(context, 2, observation(tick=1), observation(tick=2), update=update)
                self.assertEqual(row["investigation_fact"]["update"]["status"], "dropped")
                self.assertIn("investigation_requires_revision", [item["code"] for item in warnings])

    def test_definition_signature_ignores_condition_values_and_explanation_order(self):
        first = observation()
        context = AdaptiveContextV4(first)
        acknowledged(context, 1, first, observation(tick=1), update=question())
        row = copy.deepcopy(context._investigations[0])
        other = copy.deepcopy(row)
        other["conditions"]["player.facing"] = "West"
        other["explanations"].reverse()
        self.assertEqual(definition_signature(row), definition_signature(other))
        other["question"] = "A different question"
        self.assertNotEqual(definition_signature(row), definition_signature(other))


class AdaptiveV4LoopTests(unittest.TestCase):
    def fixture(self, replies):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        path = Path(temporary.name) / "recording.jsonl"
        game = KeyGateGame(0)
        context = AdaptiveContextV4(game.observe(), run_id="v4-loop")
        client = NebiusClient(Settings("fake-key"))
        with SessionLog(path) as log:
            begin_session(log, game, "fake", schema_version=2, task="explore", focus="qa", prompt_version="adaptive-v4",
                          adaptive_config={"run_id": "v4-loop", "max_prompt_bytes": 24000, "request_timeout": client.timeout})
            with patch.object(client, "_request", side_effect=replies) as http:
                result = run_lean_exploration(game, client, log, game.reset, ExplorationLimits(), focus="qa", adaptive=context)
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        return result, path, rows, http

    def test_real_loop_repairs_bad_optional_metadata_on_next_request_and_replays(self):
        bad = decision("east")
        bad.update(investigation_update=["bad"], predictions="bad")
        good = decision("east", update=question(facets=["player.position"]), predictions=[forecast()])
        result, path, rows, http = self.fixture([response(bad), response(good), response(decision("finish"))])
        self.assertEqual((result["steps"], result["rejected_decisions"], http.call_count), (2, 0, 3))
        requests = [json.loads(row["messages"][1]["content"]) for row in rows if row["type"] == "request"]
        self.assertEqual(requests[1]["metadata_feedback"]["status"], "needs_metadata_repair")
        self.assertEqual({item["code"] for item in requests[1]["metadata_feedback"]["issues"]}, {"invalid_investigation_update", "invalid_predictions"})
        self.assertEqual(requests[2]["metadata_feedback"]["status"], "accepted_metadata")
        self.assertTrue(replay(path)["adaptive_facts_validated"])

    def test_real_loop_repurpose_is_unassociated_until_explicit_revision_and_replays(self):
        start = decision("east", update=question(facets=["player.position"]))
        bad = decision("east", update={"id": "q1", "decision": "continue", "question": "A different current observation question"})
        revised = decision("east", update={"id": "q1", "decision": "revise", "question": "A different current observation question"})
        result, path, rows, _ = self.fixture([response(start), response(bad), response(revised), response(decision("finish"))])
        controls = [row for row in rows if row["type"] == "step"]
        self.assertIsNone(controls[1]["investigation_fact"]["investigation_id"])
        self.assertNotEqual(controls[0]["investigation_fact"]["condition_trial"]["definition_signature"], controls[2]["investigation_fact"]["condition_trial"]["definition_signature"])
        self.assertEqual(result["adaptive"]["investigation_definitions"]["rejected_definition_changes"], 1)
        self.assertTrue(replay(path)["adaptive_facts_validated"])

    def test_parser_dropped_update_does_not_falsely_continue_active_question(self):
        start = decision("east", update=question(facets=["player.position"]))
        malformed = decision("east")
        malformed["investigation_update"] = ["not an update"]
        result, path, rows, _ = self.fixture([response(start), response(malformed), response(decision("east")), response(decision("finish"))])
        controls = [row for row in rows if row["type"] == "step"]
        self.assertIsNone(controls[1]["decision"]["investigation_update"])
        self.assertEqual([row["investigation_fact"]["investigation_id"] for row in controls], ["q1", None, "q1"])
        self.assertIsNone(controls[1]["investigation_fact"]["condition_trial"])
        self.assertEqual(result["steps"], 3)
        self.assertEqual(result["adaptive"]["investigation"]["acknowledged_controls_with_investigation"], 2)
        self.assertEqual(result["adaptive"]["investigation_definitions"]["invalid_updates_unassociated"], 1)
        self.assertTrue(replay(path)["adaptive_facts_validated"])

    def test_replay_rejects_forged_feedback_examples(self):
        bad = decision("east")
        bad["predictions"] = "bad"
        _, path, rows, _ = self.fixture([response(bad), response(decision("finish"))])
        request = [row for row in rows if row["type"] == "request"][1]
        context = json.loads(request["messages"][1]["content"])
        context["metadata_feedback"]["public_selector_examples"] = ["invented.public.path"]
        request["messages"][1]["content"] = json.dumps(context, separators=(",", ":"))
        path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
        with self.assertRaises(ValueError):
            replay(path)


class AdaptiveV4BudgetTests(unittest.TestCase):
    def test_feedback_pressure_retains_exact_current_observation_and_full_schema_under24k(self):
        current = observation(entities=[{"id": "display", "state": {"text": "X" * 7_000}}])
        context = AdaptiveContextV4(current)
        context.observe_decision_warnings(1, [{"code": "invalid_prediction"}, {"code": "invalid_investigation_update"}])
        messages, value = build(context, current)
        self.assertEqual(expand_observation(value["observation"]), public_observation(current))
        self.assertEqual(context.schema(context.allowed(current))["properties"]["action"]["enum"], context.allowed(current))
        self.assertLessEqual(adaptive_input_bytes(messages, context.schema(context.allowed(current))), 24_000)

    def test_request_admission_requires_actual_timeout_plus_one_second(self):
        self.assertTrue(AdaptiveContextV4.request_admission(46, 45))
        self.assertFalse(AdaptiveContextV4.request_admission(45.999, 45))
        self.assertTrue(AdaptiveContextV4.request_admission(4, 3))
        for remaining, timeout in ((True, 1), (20, False), (float("nan"), 5), (100, float("inf")), (-1, 5), (20, 0)):
            self.assertFalse(AdaptiveContextV4.request_admission(remaining, timeout))

    def test_feedback_example_and_whole_packet_omissions_reconstruct_exactly(self):
        current = observation(controls=["tap"])
        warnings = [{"code": code} for code in ("invalid_investigation_update", "invalid_predictions", "too_many_predictions",
                    "invalid_prediction", "investigation_requires_revision", "investigation_update_not_applied", "invalid_expected_position")]
        for maximum, whole_omitted in ((9_000, True), (10_000, False)):
            with self.subTest(maximum=maximum):
                context = AdaptiveContextV4(current, max_prompt_bytes=maximum)
                context.observe_decision_warnings(1, warnings)
                messages, value = build(context, current)
                self.assertEqual(value["context_compaction"]["metadata_feedback_omitted"], whole_omitted)
                self.assertGreater(value["context_compaction"]["metadata_feedback_examples_omitted"], 0)
                class Recorded:
                    def retrieve(self, *_args, **_kwargs):
                        return copy.deepcopy(value["retrieved_experiments"])
                    def experiment_index(self, *_args, **_kwargs):
                        return copy.deepcopy(value["exploration_coverage"]["experiment_agenda"]["prior_run_index"])
                    def investigation_index(self, *_args, **_kwargs):
                        return copy.deepcopy(value["investigation_memory"]["prior_run_index"])
                reconstructed = AdaptiveContextV4(current, max_prompt_bytes=maximum, store=Recorded())
                reconstructed.observe_decision_warnings(1, warnings)
                replayed = reconstructed.messages(current, 0, {}, [], value["remaining_budget"], replay_compaction=value["context_compaction"])
                self.assertEqual(messages, replayed)
                self.assertLessEqual(adaptive_input_bytes(messages, context.schema(context.allowed(current))), maximum)


if __name__ == "__main__":
    unittest.main()
