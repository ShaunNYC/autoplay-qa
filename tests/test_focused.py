"""Focused experiment selection: native feedback and fake HTTP only.

These tests establish contracts and evidence integrity, not that a model learns
or that deterministic test actions improve bug discovery.
"""

from contextlib import ExitStack
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from autoplay_qa.adaptive import AdaptivePromptBudgetExceeded, adaptive_input_bytes, expand_observation
from autoplay_qa.adaptive_protocol import context_class
from autoplay_qa.cli import begin_session
from autoplay_qa.config import Settings
from autoplay_qa.experience import ExperienceStore, ScopeMismatch, experiment_fact, public_observation
from autoplay_qa.exploration import ExplorationLimits
from autoplay_qa.focused import FocusedContext, focused_schema
from autoplay_qa.game import KeyGateGame
from autoplay_qa.lean import InvalidDecision, decision_dict, run_lean_exploration
from autoplay_qa.nebius import NebiusClient, TransientNebiusError
from autoplay_qa.review import review_session
from autoplay_qa.session import SessionLog, replay
from tests.test_adaptive_context import observation, record
from tests.test_lean_exploration import response


def answer(action="east", *, question=None, prediction=None):
    return {"action": action, "reason": "Offline contract witness.", "question": question, "prediction": prediction}


def forecast(path="player.position", *, relation="equals", value=(2, 1)):
    return {"path": path, "relation": relation,
            "value_json": json.dumps(value) if relation == "equals" else None}


def build(context, current, records=()):
    messages = context.messages(current, 0, {}, list(records),
                                {"actions": 24, "requests": 28, "seconds": 300}, focus="qa")
    return messages, json.loads(messages[1]["content"])


def acknowledge(context, game, step, *, action="east", question=None, prediction=None):
    before = deepcopy(game.observe())
    build(context, before)
    parsed, warnings = context.parse_decision(json.dumps(answer(action, question=question, prediction=prediction)),
                                             context.allowed(before), set(range(1, step)))
    after = game.reset().observe() if action == "reset" else game.act(action)
    row = record(step, before, action, after, kind="reset" if action == "reset" else "step")
    row["decision"] = decision_dict(parsed)
    row["request_number"] = step
    context.record(row)
    return row, warnings


class OfflineFocusedCase(unittest.TestCase):
    def setUp(self):
        self.guards = ExitStack()
        self.addCleanup(self.guards.close)
        self.guards.enter_context(patch.object(Settings, "load", side_effect=AssertionError("Tests must not load credentials.")))
        self.guards.enter_context(patch("urllib.request.urlopen", side_effect=AssertionError("Tests must not dispatch provider HTTP.")))


class FocusedContractTests(OfflineFocusedCase):
    def test_small_contract_uses_only_native_dynamic_controls_and_adapter_coordinates(self):
        current = observation(position=(1.25, -2.5, 8.75), controls=["turn(+37)", "thrust(0.1)", "inspect(panel)"])
        current["coordinates"] = "Public adapter frame: metres [z,x,y], heading clockwise radians."
        current["player"]["facing"] = {"yaw_radians": .5}
        current["_capabilities"]["reset"] = False
        context = FocusedContext(current)
        messages, packet = build(context, current)
        allowed = [*current["controls"], "finish"]
        schema = context.schema(allowed)
        self.assertEqual(schema, focused_schema(allowed))
        self.assertEqual(set(schema["properties"]), {"action", "reason", "question", "prediction"})
        self.assertEqual(set(schema["required"]), set(schema["properties"]))
        self.assertFalse(schema["additionalProperties"])
        self.assertEqual(schema["properties"]["action"]["enum"], allowed)
        self.assertEqual(expand_observation(packet["observation"]), public_observation(current))
        self.assertEqual(packet["prompt_version"], "focused-v1")
        for legacy in ('"north"', '"south"', '"east"', '"west"', "expected_position", "investigation_update"):
            self.assertNotIn(legacy, json.dumps([messages, schema]))
        parsed, warnings = context.parse_decision(json.dumps(answer(
            "thrust(0.1)", prediction=forecast(value=(1.35, -2.5, 8.75)))), allowed, set())
        self.assertEqual(warnings, [])
        self.assertEqual(json.loads(decision_dict(parsed)["prediction"]["value_json"]), [1.35, -2.5, 8.75])

    def test_bad_optional_question_and_prediction_are_dropped_independently_of_legal_action(self):
        current = KeyGateGame(0).observe()
        context = FocusedContext(current)
        build(context, current)
        cases = (
            (answer(question={"bad": "shape"}, prediction=forecast()), None, forecast()),
            (answer(question="Will this public field change?", prediction={"path": "not.present"}),
             "Will this public field change?", None),
            (answer(question=["bad"], prediction="bad"), None, None),
            ({**answer(question="A valid focused question."), "expected_position": [9, 9], "notes": [{"bogus": True}]},
             "A valid focused question.", None),
        )
        for payload, question, prediction in cases:
            with self.subTest(payload=payload):
                parsed, warnings = context.parse_decision(json.dumps(payload), context.allowed(current), set())
                self.assertEqual(parsed.action, "east")
                self.assertIsNone(parsed.expected_position)
                self.assertEqual(parsed.notes, ())
                self.assertEqual(decision_dict(parsed)["question"], question)
                actual_prediction = decision_dict(parsed)["prediction"]
                if prediction is None:
                    self.assertIsNone(actual_prediction)
                else:
                    self.assertEqual(actual_prediction["path"], prediction["path"])
                    self.assertEqual(actual_prediction["relation"], prediction["relation"])
                    self.assertEqual(json.loads(actual_prediction["value_json"]), json.loads(prediction["value_json"]))
                self.assertTrue(warnings)

    def test_absent_optional_metadata_keeps_legal_action_but_missing_or_illegal_action_is_rejected(self):
        current = KeyGateGame(0).observe()
        context = FocusedContext(current)
        build(context, current)
        parsed, _ = context.parse_decision(json.dumps({"action": "east", "reason": "Visible action."}), context.allowed(current), set())
        self.assertEqual(parsed.action, "east")
        for payload in ({"reason": "No action."}, answer("teleport"), [answer()], "not json"):
            content = payload if isinstance(payload, str) else json.dumps(payload)
            with self.subTest(payload=payload), self.assertRaises(InvalidDecision):
                context.parse_decision(content, context.allowed(current), set())

    def test_predictions_cannot_target_private_missing_or_transient_fields(self):
        current = observation(controls=["inspect"], entities=[{"id": "panel", "state": {"open": False, "telemetry": 3}}])
        current["_capabilities"]["state_identity"] = {"transient_paths": ["entities.panel.state.telemetry"]}
        current["evaluation"] = {"target": "PRIVATE_TARGET"}
        current["player"]["secret"] = "PRIVATE_SECRET"
        context = FocusedContext(current, secrets=("PRIVATE_SECRET",))
        build(context, current)
        for path in ("evaluation.target", "player.secret", "entities.panel.state.telemetry", "tick", "events", "player.position.0", "player.missing"):
            with self.subTest(path=path):
                parsed, warnings = context.parse_decision(json.dumps(answer("inspect", prediction=forecast(path))),
                                                         ["inspect", "finish"], set())
                self.assertEqual(parsed.action, "inspect")
                self.assertIsNone(decision_dict(parsed)["prediction"])
                self.assertTrue(warnings)


class FocusedEvidenceTests(OfflineFocusedCase):
    def test_questions_change_only_after_acknowledgement_and_invalid_or_null_updates_retain_them(self):
        game = KeyGateGame(0)
        context = FocusedContext(game.observe())
        build(context, game.observe())
        context.parse_decision(json.dumps(answer(question="Unexecuted question.")), context.allowed(game.observe()), set())
        _, before = build(context, game.observe())
        self.assertIsNone(before["retrieved_experiments"]["question"])
        first, _ = acknowledge(context, game, 1, question="Does this observed control change position?")
        self.assertEqual(first["focused_fact"]["question_revision"], 1)
        self.assertEqual(first["focused_fact"]["question_update"], "created")
        second, warnings = acknowledge(context, game, 2, action="west", question=["malformed"])
        self.assertTrue(warnings)
        self.assertEqual(second["focused_fact"]["question"], first["focused_fact"]["question"])
        self.assertEqual(second["focused_fact"]["question_revision"], 1)
        third, _ = acknowledge(context, game, 3, action="north")
        self.assertEqual(third["focused_fact"]["question_revision"], 1)
        fourth, _ = acknowledge(context, game, 4, question="Will another condition change the result?")
        self.assertEqual(fourth["focused_fact"]["question_revision"], 2)
        self.assertEqual(fourth["focused_fact"]["question_update"], "revised")

    def test_public_forecasts_use_actual_native_results_and_keep_counterexamples_at_original_question_revision(self):
        game = KeyGateGame(0)
        context = FocusedContext(game.observe(), run_id="forecast-run")
        first, _ = acknowledge(context, game, 1, question="Does position change as predicted?", prediction=forecast(value=(8, 8)))
        check = first["focused_fact"]["prediction_check"]
        self.assertEqual((check["before"], check["after"], check["status"]), ([1, 1], [2, 1], "mismatch"))
        self.assertTrue(first["focused_fact"]["counterexample"])
        second, _ = acknowledge(context, game, 2, question="Does the next control preserve inventory?",
                                prediction=forecast("player.inventory", relation="unchanged"))
        self.assertEqual(second["focused_fact"]["prediction_check"]["status"], "matched")
        self.assertFalse(second["focused_fact"]["counterexample"])
        _, packet = build(context, game.observe(), [first, second])
        card = packet["retrieved_experiments"]
        self.assertEqual(card["question_revision"], 2)
        self.assertEqual(card["forecast_check"]["citation"], {"run_id": "forecast-run", "step": 2})
        self.assertEqual(card["counterexamples"][0]["citation"], {"run_id": "forecast-run", "step": 1})
        self.assertEqual(card["counterexamples"][0]["question_revision"], 1)
        self.assertEqual(card["counterexamples"][0]["prediction_check"]["status"], "mismatch")
        before = deepcopy(context.summary())
        context.record(first)
        context.record(second)
        self.assertEqual(context.summary(), before)

    def test_vanished_public_field_is_not_comparable_rather_than_a_refuted_prediction(self):
        before = observation(controls=["close"], entities=[{"id": "panel", "state": {"open": True}}])
        after = observation(controls=["open"], tick=1, entities=[])
        context = FocusedContext(before)
        build(context, before)
        parsed, warnings = context.parse_decision(json.dumps(answer("close", prediction=forecast("entities.panel.state.open", relation="unchanged"))),
                                                 context.allowed(before), set())
        self.assertEqual(warnings, [])
        row = record(1, before, "close", after)
        row["decision"] = decision_dict(parsed)
        context.record(row)
        self.assertEqual(row["focused_fact"]["prediction_check"]["status"], "not_comparable")
        self.assertFalse(row["focused_fact"]["counterexample"])

    def test_real_prior_fact_retains_exact_source_and_readonly_history_with_private_metadata_excluded(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "observed.sqlite3"
            game = KeyGateGame(0)
            initial = deepcopy(game.observe())
            after = game.act("north")
            with ExperienceStore(path, "actual-grid", game.version, initial) as store:
                store.record(initial, "north", after, "prior-native", 1)
            original = path.read_bytes()
            with ExperienceStore(path, "actual-grid", game.version, initial, read_only=True) as store:
                context = FocusedContext(initial, store=store, run_id="current-native")
                _, packet = build(context, initial)
                fact, = packet["retrieved_experiments"]["facts"]
                expected = experiment_fact(initial, "north", after, run_id="prior-native", step=1)
                self.assertEqual(fact["action"], expected["action"])
                self.assertEqual(fact["before_state_id"], expected["before_state_id"])
                self.assertEqual(fact["after_state_id"], expected["after_state_id"])
                self.assertEqual(fact["outcome"], expected["outcome"])
                self.assertEqual(fact["citations"], [expected["citation"]])
                current = KeyGateGame(0)
                acknowledge(context, current, 1)
                self.assertEqual(store.summary()["transitions"], 1)
            self.assertEqual(path.read_bytes(), original)
            changed = deepcopy(initial)
            changed["coordinates"] = "A different public coordinate frame."
            with self.assertRaises(ScopeMismatch):
                with ExperienceStore(path, "actual-grid", game.version, changed, read_only=True):
                    pass

    def test_current_and_prior_same_numbered_citations_do_not_share_prior_aggregate_counts(self):
        with tempfile.TemporaryDirectory() as directory:
            game = KeyGateGame(0)
            initial = deepcopy(game.observe())
            path = Path(directory) / "observed.sqlite3"
            with ExperienceStore(path, "actual-grid", game.version, initial) as store:
                prior_after = KeyGateGame(0).act("north")
                store.record(initial, "north", prior_after, "prior-run", 1)
                context = FocusedContext(initial, store=store, run_id="current-run")
                acknowledge(context, game, 1, action="north")
                _, packet = build(context, game.observe())
                card = packet["retrieved_experiments"]
                fact, = card["facts"]
                self.assertEqual(fact["citations"], [{"run_id": "prior-run", "step": 1}])
                for aggregate in ("attempts", "unchanged_count", "different_outcomes_for_action"):
                    self.assertNotIn(aggregate, fact)
                self.assertEqual(card["last_trial"]["citation"], {"run_id": "current-run", "step": 1})
                self.assertEqual(store.summary()["transitions"], 2)

    def test_evidence_card_is_bounded_and_omissions_are_explicit_without_slicing_current_observation(self):
        with tempfile.TemporaryDirectory() as directory:
            game = KeyGateGame(0)
            initial = deepcopy(game.observe())
            path = Path(directory) / "observed.sqlite3"
            actions = ("north", "west", "east", "east", "south", "south", "west", "west", "north", "north", "wait", "interact")
            with ExperienceStore(path, "actual-grid", game.version, initial) as store:
                for step, action in enumerate(actions, 1):
                    before = deepcopy(game.observe())
                    store.record(before, action, game.act(action), "native-collection", step)
            with ExperienceStore(path, "actual-grid", game.version, initial, read_only=True) as store:
                context = FocusedContext(initial, store=store, run_id="bounded-evaluation")
                messages, packet = build(context, initial)
                card = packet["retrieved_experiments"]
                self.assertLessEqual(len(json.dumps(card, ensure_ascii=False, separators=(",", ":")).encode()), 4000)
                self.assertLessEqual(len(card["facts"]), 2)
                self.assertGreater(card["omitted_count"], 0)
                self.assertLessEqual(adaptive_input_bytes(messages, context.schema(context.allowed(initial))), 24000)
                self.assertEqual(expand_observation(packet["observation"]), public_observation(initial))

    def test_budget_pressure_does_not_keep_only_one_side_of_a_native_same_condition_outcome_contrast(self):
        with tempfile.TemporaryDirectory() as directory:
            game = KeyGateGame(0, defect="exit-no-win")
            initial = deepcopy(game.observe())
            for action in ("south", "south", "east", "east", "interact", "east", "east", "east"):
                game.act(action)
            first_before = deepcopy(game.observe())
            first_after = game.act("east")
            game.act("west")
            second_before = deepcopy(game.observe())
            second_after = game.act("east")
            self.assertEqual((first_after["status"], second_after["status"]), ("playing", "won"))
            path = Path(directory) / "native-contrast.sqlite3"
            with ExperienceStore(path, "actual-grid", game.version, initial) as store:
                first = store.record(first_before, "east", first_after, "native-contrast", 1)
                second = store.record(second_before, "east", second_after, "native-contrast", 2)
                self.assertEqual(first["before_state_id"], second["before_state_id"])
            with ExperienceStore(path, "actual-grid", game.version, initial, read_only=True) as store:
                context = FocusedContext(first_before, store=store, run_id="contrast-evaluation")
                messages, packet = build(context, first_before)
                self.assertEqual(len(packet["retrieved_experiments"]["facts"]), 2)
                baseline = adaptive_input_bytes(messages, context.schema(context.allowed(first_before)))
                pressured = deepcopy(first_before)
                # Synthetic public-event padding tests context accounting only;
                # both contrasting source outcomes above are native feedback.
                pressured["events"] = ["X" * max(1, 8000 - baseline + 200)]
                bounded = FocusedContext(pressured, store=store, run_id="contrast-evaluation", max_prompt_bytes=8000)
                bounded_messages, bounded_packet = build(bounded, pressured)
                self.assertEqual(bounded_packet["retrieved_experiments"]["facts"], [])
                self.assertEqual(bounded_packet["retrieved_experiments"]["omitted_count"], 2)
                self.assertLessEqual(adaptive_input_bytes(bounded_messages, bounded.schema(bounded.allowed(pressured))), 8000)
                self.assertEqual(expand_observation(bounded_packet["observation"]), public_observation(pressured))

    def test_complete_current_observation_that_exceeds_budget_fails_explicitly(self):
        current = observation(controls=["inspect"])
        current["events"] = ["A" * 25_000]
        context = FocusedContext(current)
        with self.assertRaises(AdaptivePromptBudgetExceeded):
            build(context, current)
        self.assertEqual(current["events"], ["A" * 25_000])

    def test_unavailable_history_is_explicit_and_does_not_leak_exception_text_as_evidence(self):
        class UnavailableStore:
            def retrieve(self, *args, **kwargs):
                raise OSError("PRIVATE_SECRET /Users/private/project/experience.sqlite3")

            def summary(self):
                return {}

        current = KeyGateGame(0).observe()
        context = FocusedContext(current, store=UnavailableStore(), secrets=("PRIVATE_SECRET",))
        messages, packet = build(context, current)
        card = packet["retrieved_experiments"]
        self.assertEqual(card["facts"], [])
        self.assertTrue(card["unavailable"])
        serialized = json.dumps([messages, context.summary()])
        self.assertNotIn("PRIVATE_SECRET", serialized)
        self.assertNotIn("/Users/private", serialized)


class FocusedLoopTests(OfflineFocusedCase):
    def fixture(self, replies, *, store=None, run_id="focused-loop", game=None, max_actions=24):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        path = Path(temporary.name) / "focused.jsonl"
        game = game or KeyGateGame(0)
        context = FocusedContext(game.observe(), store=store, run_id=run_id)
        client = NebiusClient(Settings("OFFLINE_FOCUSED_FAKE_KEY"))
        with SessionLog(path) as log:
            begin_session(log, game, "fake", schema_version=2, task="explore", focus="qa", prompt_version="focused-v1",
                          adaptive_config={"run_id": run_id, "max_prompt_bytes": 24000, "request_timeout": client.timeout})
            with patch.object(client, "_request", side_effect=replies) as http:
                result = run_lean_exploration(game, client, log, game.reset, ExplorationLimits(max_actions=max_actions), focus="qa", adaptive=context)
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        return result, path, rows, http, context

    def test_real_loop_executes_legal_actions_despite_bad_optional_metadata_and_replays(self):
        bad = answer(question={"invalid": True}, prediction="invalid")
        good = answer(question="Does this public position change?", prediction=forecast(value=(3, 1)))
        result, path, rows, http, _ = self.fixture([response(bad), response(good), response(answer("finish"))])
        self.assertEqual((result["status"], result["steps"], result["rejected_decisions"], http.call_count), ("finished", 2, 0, 3))
        self.assertGreaterEqual(result["warning_count"], 2)
        self.assertIsNone(result["unassessed_final_outcome"])
        self.assertEqual([row["decision"]["action"] for row in rows if row["type"] == "step"], ["east", "east"])
        self.assertTrue(replay(path)["adaptive_facts_validated"])
        self.assertTrue(review_session(path)["execution_replayed"])

    def test_real_later_session_receives_only_factual_prior_evidence_and_cannot_mutate_readonly_archive(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "history.sqlite3"
            initial = KeyGateGame(0).observe()
            with ExperienceStore(path, "actual-grid", KeyGateGame.version, initial) as store:
                first, _, _, _, _ = self.fixture([response(answer("north", question="Prior tentative question.")), response(answer("finish"))],
                                                store=store, run_id="collection")
                self.assertEqual(first["steps"], 1)
            original = hashlib.sha256(path.read_bytes()).hexdigest()
            with ExperienceStore(path, "actual-grid", KeyGateGame.version, initial, read_only=True) as store:
                _, recording, rows, _, _ = self.fixture([response(answer()), response(answer("finish"))], store=store, run_id="evaluation")
            first_request = next(row for row in rows if row["type"] == "request")
            card = json.loads(first_request["messages"][1]["content"])["retrieved_experiments"]
            self.assertIsNone(card["question"])
            self.assertIsNone(card["forecast_check"])
            self.assertEqual(card["counterexamples"], [])
            self.assertEqual(card["facts"][0]["citations"], [{"run_id": "collection", "step": 1}])
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), original)
            self.assertTrue(replay(recording)["adaptive_facts_validated"])
            self.assertFalse(replay(recording)["prior_run_memory_execution_verified"])

    def test_actor_reset_returns_real_native_spawn_and_preserves_executed_control_counts(self):
        result, path, rows, http, _ = self.fixture([
            response(answer("east")), response(answer("reset", prediction=forecast(value=(1, 1)))), response(answer("finish"))])
        reset, = [row for row in rows if row["type"] == "reset"]
        self.assertEqual(reset["after"]["player"]["position"], [1, 1])
        self.assertEqual(reset["after"]["tick"], 0)
        self.assertEqual(reset["focused_fact"]["prediction_check"]["status"], "matched")
        self.assertEqual((result["steps"], result["game_actions"], result["resets"], http.call_count), (2, 1, 1, 3))
        self.assertTrue(replay(path)["recording_complete"])

    def test_finish_can_update_question_without_inventing_executed_evidence_or_forecast_result(self):
        result, path, rows, _, context = self.fixture([
            response(answer("finish", question="This remains untested.", prediction=forecast()))])
        finish, = [row for row in rows if row["type"] == "finish"]
        receipt = finish["focused_fact"]
        self.assertEqual(receipt["question"], "This remains untested.")
        self.assertIsNone(receipt["citation"])
        self.assertIsNone(receipt["prediction_check"])
        self.assertFalse(receipt["counterexample"])
        self.assertEqual(result["steps"], 0)
        _, packet = build(context, KeyGateGame(0).observe())
        self.assertIsNone(packet["retrieved_experiments"]["last_trial"])
        self.assertTrue(replay(path)["recording_complete"])

    def test_truncated_response_cannot_apply_question_prediction_or_control(self):
        result, path, rows, http, context = self.fixture([
            response(answer(question="Must never become active.", prediction=forecast()), finish="length"),
            response(answer("finish"))])
        self.assertEqual((result["steps"], result["rejected_decisions"], http.call_count), (0, 1, 2))
        self.assertFalse(any(row["type"] in {"step", "reset"} for row in rows))
        _, packet = build(context, KeyGateGame(0).observe())
        self.assertIsNone(packet["retrieved_experiments"]["question"])
        self.assertTrue(replay(path)["recording_complete"])

    def test_provider_failure_is_terminal_without_retry_or_forced_action(self):
        result, path, rows, http, _ = self.fixture([TransientNebiusError("offline", reason="timeout")])
        self.assertEqual((result["steps"], result["retries"], http.call_count), (0, 0, 1))
        self.assertEqual(result["status"], "provider_unavailable")
        self.assertFalse(any(row["type"] in {"step", "reset"} for row in rows))
        self.assertTrue(replay(path)["recording_complete"])

    def test_action_limit_keeps_explicit_last_fact_ack_pending_even_when_forecast_is_checked(self):
        result, path, _, http, _ = self.fixture([response(answer(prediction=forecast()))], max_actions=1)
        self.assertEqual((result["status"], result["steps"], http.call_count), ("action_limit", 1, 1))
        focused = result["adaptive"]["focused"]
        self.assertEqual(focused["forecast_check"]["status"], "matched")
        self.assertEqual(focused["final_pending_fact_ack"], {"run_id": "focused-loop", "step": 1})
        self.assertTrue(replay(path)["recording_complete"])

    def test_strict_replay_rejects_changed_public_observation_native_schema_and_focused_receipt(self):
        for target in ("observation", "schema", "focused_fact", "question"):
            with self.subTest(target=target):
                _, path, rows, _, _ = self.fixture([
                    response(answer(question="Does this public position change?", prediction=forecast())), response(answer("finish"))])
                request = next(row for row in rows if row["type"] == "request")
                if target == "schema":
                    request["response_schema"]["properties"]["action"]["enum"].append("teleport")
                elif target in {"observation", "question"}:
                    packet = json.loads(request["messages"][1]["content"])
                    if target == "observation":
                        packet["observation"]["player"]["position"] = [7, 3]
                    else:
                        packet["retrieved_experiments"]["question"] = "An invented earlier question."
                    request["messages"][1]["content"] = json.dumps(packet)
                else:
                    step = next(row for row in rows if row["type"] == "step")
                    step["focused_fact"]["prediction_check"]["status"] = "mismatch"
                path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
                with self.assertRaises(ValueError):
                    replay(path)


if __name__ == "__main__":
    unittest.main()
