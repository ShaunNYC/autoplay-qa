"""Explicit inference fixtures; real exploration must be verified separately."""

import copy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from autoplay_qa.cli import begin_session, exploration_exit_code, main
from autoplay_qa.config import Settings
from autoplay_qa.exploration import (
    EXPLORATION_PROMPT, EXPLORATION_SCHEMA, Coverage, ExplorationDecision,
    ExplorationLimits, exploration_messages, exploration_view, run_exploration,
    turn_schema,
)
from autoplay_qa.game import ACTIONS, KeyGateGame, evaluate_transition
from autoplay_qa.nebius import DECISION_SCHEMA, InvalidDecision, NebiusClient, NebiusError
from autoplay_qa.session import SessionLog, replay


def hypothesis(identifier="h1", status="untested", evidence=(), claim="A tentative mechanic"):
    return {"id": identifier, "claim": claim, "status": status, "evidence_steps": list(evidence)}


def decision(action="north", pending=None, hypotheses=(), experiment="e1", goal="Investigate a control", prediction="Observe whether state changes"):
    return {
        "action": action, "reason": "Fixture experiment decision", "plan": "Continue this experiment",
        "experiment_id": experiment, "experiment_goal": goal, "prediction": prediction,
        "assessment": {"step": pending, "outcome": "Fixture comparison with the recorded outcome"} if pending is not None else None,
        "hypotheses": list(hypotheses),
    }


def response(value, finish="stop"):
    return {
        "id": "unit-test-exploration-fixture", "model": "nvidia/unit-test-nemotron",
        "choices": [{"message": {"content": json.dumps(value)}, "finish_reason": finish}],
        "usage": {"prompt_tokens": 20, "completion_tokens": 10, "total_tokens": 30},
    }


class DecisionTests(unittest.TestCase):
    def parse(self, value, allowed=None, completed=None, pending=None):
        return ExplorationDecision.parse(json.dumps(value), allowed or [*ACTIONS, "reset", "finish"], completed or set(), pending)

    def test_grounded_update_and_assessment(self):
        parsed = self.parse(decision("south", 1, [hypothesis(status="supported", evidence=[1])]), completed={1}, pending=1)
        self.assertEqual(parsed.assessment.step, 1)
        self.assertEqual(parsed.hypotheses[0].evidence_steps, (1,))

    def test_nonexistent_future_boolean_and_noninteger_evidence_are_rejected(self):
        for evidence in ([2], [0], [-1], [True], [1.0], [1, 1]):
            with self.subTest(evidence=evidence), self.assertRaises(InvalidDecision):
                self.parse(decision("south", 1, [hypothesis(status="supported", evidence=evidence)]), completed={1}, pending=1)

    def test_duplicate_ids_and_unsupported_status_are_rejected(self):
        for updates in ([hypothesis(), hypothesis()], [hypothesis(status="verified")], [hypothesis(status="supported")]):
            with self.subTest(updates=updates), self.assertRaises(InvalidDecision):
                self.parse(decision(hypotheses=updates))

    def test_memory_and_claim_bounds(self):
        for updates in ([hypothesis(claim="x" * 241)], [hypothesis(identifier=f"h{i}") for i in range(9)]):
            with self.subTest(updates=updates), self.assertRaises(InvalidDecision):
                self.parse(decision(hypotheses=updates))

    def test_required_assessment_refers_to_the_latest_global_step(self):
        for value in (decision(pending=None), decision(pending=1), decision(pending=True)):
            with self.subTest(value=value), self.assertRaises(InvalidDecision):
                self.parse(value, completed={1, 2}, pending=2)
        with self.assertRaises(InvalidDecision):
            self.parse(decision(pending=1))

    def test_pending_placeholders_do_not_count_as_assessing_an_observed_outcome(self):
        for text in ("pending", "Pending execution of the recorded action, no outcome is available yet.", "matched}, hypotheses: [], experiment_id:"):
            value = decision(pending=1)
            value["assessment"]["outcome"] = text
            with self.subTest(text=text), self.assertRaises(InvalidDecision):
                self.parse(value, completed={1}, pending=1)

    def test_concise_comparisons_are_accepted_without_artificial_padding(self):
        for outcome in ("Stayed put as predicted.", "matched prediction", "contradicted"):
            value = decision(pending=1)
            value["assessment"]["outcome"] = outcome
            self.assertEqual(self.parse(value, completed={1}, pending=1).assessment.outcome, outcome)

    def test_terminal_controls_and_unknown_actions(self):
        for action in ("north", "teleport"):
            with self.subTest(action=action), self.assertRaises(InvalidDecision):
                self.parse(decision(action), allowed=["reset", "finish"])
        self.assertEqual(self.parse(decision("finish"), allowed=["reset", "finish"]).action, "finish")


class ObservationTests(unittest.TestCase):
    def test_view_withholds_spoilers_and_does_not_mutate_snapshot(self):
        observation = KeyGateGame().observe()
        original = copy.deepcopy(observation)
        observation["goal"] = "SECRET_OBJECTIVE"
        observation["rules"] = ["SECRET_RULE"]
        observation["events"] = ["SECRET_START_INSTRUCTION"]
        observation["neighbors"] = {"north": {"passable": False}}
        view = exploration_view(observation)
        text = json.dumps(view)
        for value in ("SECRET_", "passable", '"rules"', '"goal"'):
            self.assertNotIn(value, text)
        self.assertEqual(view["events"], ["Environment started."])
        view["player"]["position"][0] = 999
        self.assertEqual(observation["player"], original["player"])

    def test_actual_feedback_is_visible_and_terminal_controls_are_limited(self):
        game = KeyGateGame()
        after = game.act("interact")
        self.assertEqual(exploration_view(after)["events"], after["events"])
        game.won = True
        self.assertEqual(exploration_view(game.observe())["controls"], ["reset", "finish"])

    def test_novelty_ignores_ticks_and_reset_does_not_create_new_states(self):
        game = KeyGateGame()
        before = game.observe()
        coverage = Coverage(before)
        for _ in range(3):
            after = game.act("wait")
            coverage.record(before, "wait", after)
            before = after
        self.assertEqual(coverage.context(before)["distinct_observed_states"], 1)
        self.assertEqual(coverage.context(before)["prior_attempts_in_current_state"], {"wait": 3})
        coverage.record(before, "reset", KeyGateGame().observe())
        self.assertEqual(coverage.summary()["distinct_observed_states"], 1)

    def test_prompt_is_independent_of_the_playthrough_policy(self):
        text = json.dumps(exploration_messages(KeyGateGame().observe(), 0, None, {}, [], Coverage(KeyGateGame().observe()), None, {"actions": 24}))
        for phrase in ("Collect the key", "collect the key", "interact opens", '"passable"', '"rules"', "key before", "stand next"):
            self.assertNotIn(phrase, text)
        self.assertIn(EXPLORATION_PROMPT, json.loads(text)[0]["content"])
        self.assertNotIn("adds 1", EXPLORATION_PROMPT)

    def test_client_accepts_the_exploration_schema_without_changing_the_default(self):
        client = NebiusClient(Settings("unit-test-key"))
        with patch.object(client, "_request", return_value=response(decision())) as transport:
            client.complete([], response_schema=EXPLORATION_SCHEMA)
            self.assertEqual(transport.call_args.args[1]["response_format"]["json_schema"]["schema"], EXPLORATION_SCHEMA)
            client.complete([])
            self.assertEqual(transport.call_args.args[1]["response_format"]["json_schema"]["schema"], DECISION_SCHEMA)

    def test_turn_schema_constrains_history_references_without_supplying_mechanics(self):
        initial = turn_schema(["north", "reset", "finish"], set(), None)
        self.assertEqual(initial["properties"]["assessment"], {"type": "null"})
        self.assertEqual(initial["properties"]["hypotheses"]["items"]["properties"]["evidence_steps"]["maxItems"], 0)
        later = turn_schema(["reset", "finish"], {1, 2}, 2, True)
        self.assertEqual(later["properties"]["action"]["enum"], ["reset", "finish"])
        self.assertEqual(later["properties"]["assessment"]["properties"]["step"]["enum"], [2])
        self.assertEqual(later["properties"]["hypotheses"]["minItems"], 1)
        self.assertEqual(later["properties"]["hypotheses"]["items"]["properties"]["evidence_steps"]["items"]["enum"], [1, 2])
        self.assertNotIn("key", json.dumps(later))
        self.assertIn("anyOf", EXPLORATION_SCHEMA["properties"]["assessment"])

    def test_exploration_sampling_does_not_change_the_playthrough_default(self):
        baseline = NebiusClient(Settings("unit-test-key"))
        explorer = NebiusClient(Settings("unit-test-key"), temperature=1.0, top_p=0.95)
        for client in (baseline, explorer):
            with patch.object(client, "_request", return_value=response(decision())) as transport:
                client.complete([])
                payload = transport.call_args.args[1]
                self.assertEqual(payload["temperature"], client.temperature)
                self.assertEqual(payload.get("top_p"), client.top_p)
        self.assertEqual(baseline.temperature, 0.2)
        self.assertIsNone(baseline.top_p)


class ControllerTests(unittest.TestCase):
    def run_fixture(self, responses, limits=ExplorationLimits(), game=None, factory=None, evaluator=evaluate_transition, response_format="json_schema"):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        path = Path(directory.name) / "test-only.jsonl"
        game = game or KeyGateGame(0)
        client = NebiusClient(Settings("unit-test-key", response_format=response_format))
        with SessionLog(path, ("unit-test-key",)) as log:
            begin_session(log, game, "unit-test-fixtures", schema_version=2, task="explore")
            with patch.object(client, "_request", side_effect=responses) as transport:
                summary = run_exploration(game, client, log, factory or (lambda: KeyGateGame(0)), limits, evaluator=evaluator)
        records = [json.loads(line) for line in path.read_text().splitlines()]
        return summary, records, path, transport

    def test_learning_survives_reset_and_evidence_numbers_do_not_restart(self):
        responses = [
            response(decision("north", hypotheses=[hypothesis()])),
            response(decision("reset", 1, [hypothesis(status="supported", evidence=[1])])),
            response(decision("interact", 2, experiment="e2", goal="Investigate interactions")),
            response(decision("finish", 3, [hypothesis(identifier="h2", status="supported", evidence=[3])], experiment="e2", goal="Investigate interactions")),
        ]
        summary, records, path, transport = self.run_fixture(responses)
        self.assertEqual(summary["status"], "finished")
        self.assertEqual(summary["steps"], 3)
        self.assertEqual(summary["game_actions"], 2)
        self.assertEqual(summary["resets"], 1)
        self.assertEqual(summary["episodes"], 2)
        self.assertIsNone(summary["unassessed_final_step"])
        self.assertEqual(len(summary["hypotheses"]), 2)
        contexts = [json.loads(r["messages"][1]["content"]) for r in records if r["type"] == "request"]
        self.assertEqual(contexts[2]["observation"]["tick"], 0)
        self.assertEqual(contexts[2]["pending_assessment_step"], 2)
        self.assertEqual(contexts[2]["hypotheses"][0]["evidence_steps"], [1])
        self.assertEqual(contexts[1]["completed_outcome_to_assess"]["observed_changes"]["position_delta"], {"dx": 0, "dy": 0})
        self.assertEqual(contexts[3]["coverage"]["prior_attempts_in_current_state"], {"north": 1, "reset": 1, "interact": 1})
        self.assertEqual(summary["coverage"]["distinct_observed_states"], 1)
        self.assertEqual(transport.call_count, 4)
        result = replay(path)
        self.assertEqual(result["actions_replayed"], 3)
        self.assertEqual(result["resets_replayed"], 1)
        self.assertFalse(result["learning_claims_verified"])

    def test_invalid_memory_update_is_atomic_and_uses_one_repair(self):
        responses = [
            response(decision("north", hypotheses=[hypothesis(claim="Original memory")])),
            response(decision("east", 1, [hypothesis(status="supported", evidence=[2], claim="REJECTED_MEMORY")], experiment="invalid-goal")),
            response(decision("finish", 1, [hypothesis(status="uncertain", evidence=[1], claim="Original memory")])),
        ]
        summary, records, _, _ = self.run_fixture(responses)
        self.assertEqual(summary["steps"], 1)
        self.assertEqual(summary["rejected_decisions"], 1)
        self.assertEqual(summary["hypotheses"][0]["claim"], "Original memory")
        contexts = [json.loads(r["messages"][1]["content"]) for r in records if r["type"] == "request"]
        self.assertEqual(contexts[1]["observation"], contexts[2]["observation"])
        self.assertEqual(contexts[1]["hypotheses"], contexts[2]["hypotheses"])
        self.assertEqual(contexts[2]["active_experiment"]["id"], "e1")
        self.assertNotIn("REJECTED_MEMORY", json.dumps(contexts[2]))
        self.assertIn("validation_feedback", contexts[2])

    def test_dynamic_contract_reaches_the_model_in_every_response_mode(self):
        for mode in ("json_schema", "json_object", "plain"):
            with self.subTest(mode=mode):
                summary, records, _, transport = self.run_fixture([
                    response(decision("south")),
                    response(decision("finish", 1, [hypothesis(status="supported", evidence=[1])])),
                ], response_format=mode)
                self.assertEqual(summary["status"], "finished")
                requests = [r for r in records if r["type"] == "request"]
                context = json.loads(requests[1]["messages"][1]["content"])
                self.assertEqual(context["response_schema"], requests[1]["response_schema"])
                self.assertEqual(context["response_schema"]["properties"]["hypotheses"]["minItems"], 1)
                self.assertEqual(transport.call_args.args[1]["messages"], requests[1]["messages"])

    def test_session_memory_cannot_grow_past_eight_ids(self):
        responses = [
            response(decision("wait", hypotheses=[hypothesis(identifier=f"h{i}") for i in range(8)])),
            response(decision("wait", 1, [hypothesis(identifier="h9")])),
            response(decision("finish", 1, [hypothesis(identifier="h0", status="uncertain", evidence=[1])])),
        ]
        summary, _, _, _ = self.run_fixture(responses)
        self.assertEqual(summary["steps"], 1)
        self.assertEqual(summary["rejected_decisions"], 1)
        self.assertEqual(len(summary["hypotheses"]), 8)

    def test_repeated_invalid_responses_execute_nothing_and_omit_trace_text(self):
        bad = response(decision(hypotheses=[hypothesis(status="supported", evidence=[1])]))
        summary, records, _, _ = self.run_fixture([bad, bad])
        self.assertEqual(summary["status"], "invalid_decision")
        self.assertEqual(summary["steps"], 0)
        self.assertEqual(summary["hypotheses"], [])
        self.assertTrue(all(r["answer"] is None for r in records if r["type"] == "completion"))

    def test_truncation_does_not_apply_memory_or_actions(self):
        summary, _, _, _ = self.run_fixture([response(decision(hypotheses=[hypothesis()]), finish="length")] * 2)
        self.assertEqual(summary["status"], "invalid_decision")
        self.assertEqual(summary["hypotheses"], [])
        self.assertEqual(summary["steps"], 0)

    def test_reset_counts_toward_action_budget_and_leaves_last_outcome_unassessed(self):
        summary, _, path, _ = self.run_fixture([response(decision("reset"))], ExplorationLimits(max_actions=1))
        self.assertEqual(summary["status"], "action_limit")
        self.assertEqual(summary["steps"], 1)
        self.assertEqual(summary["game_actions"], 0)
        self.assertEqual(summary["resets"], 1)
        self.assertEqual(summary["unassessed_final_step"], 1)
        self.assertEqual(summary["unassessed_final_outcome"]["action"], "reset")
        self.assertEqual(summary["unassessed_final_outcome"]["observed_events"], ["Environment started."])
        self.assertTrue(summary["partial"])
        self.assertEqual(replay(path)["actions_replayed"], 1)

    def test_request_limit_bounds_repairs(self):
        summary, _, _, _ = self.run_fixture([response(decision("teleport"))], ExplorationLimits(max_requests=1))
        self.assertEqual(summary["status"], "request_limit")
        self.assertEqual(summary["steps"], 0)

    def test_completion_after_deadline_changes_neither_game_nor_learning(self):
        clock = [0.0]
        def late_response(*_):
            clock[0] = 10.0
            return response(decision(hypotheses=[hypothesis()]))
        with patch("time.monotonic", side_effect=lambda: clock[0]):
            summary, records, _, _ = self.run_fixture(late_response, ExplorationLimits(max_seconds=1))
        self.assertEqual(summary["status"], "time_limit")
        self.assertEqual(summary["steps"], 0)
        self.assertEqual(summary["hypotheses"], [])
        self.assertFalse(any(r["type"] == "memory" for r in records))

    def test_network_timeout_at_run_deadline_is_a_partial_budget_completion(self):
        clock = [0.0]
        def late_timeout(*_):
            clock[0] = 10.0
            raise NebiusError("Timed out; request may be billed")
        with patch("time.monotonic", side_effect=lambda: clock[0]):
            summary, _, _, _ = self.run_fixture(late_timeout, ExplorationLimits(max_seconds=1))
        self.assertEqual(summary["status"], "time_limit")
        self.assertEqual(exploration_exit_code(summary["status"]), 0)
        self.assertFalse(summary["usage_complete"])
        self.assertEqual(summary["completed_inference_responses"], 0)

    def test_deadline_during_recording_prevents_a_new_game_action(self):
        clock = [0.0]
        write_record = SessionLog.write
        def slow_record(log, kind, **fields):
            write_record(log, kind, **fields)
            if kind == "memory":
                clock[0] = 10.0
        with patch("time.monotonic", side_effect=lambda: clock[0]), patch.object(SessionLog, "write", new=slow_record):
            summary, _, path, _ = self.run_fixture([
                response(decision("north", hypotheses=[hypothesis()])),
            ], ExplorationLimits(max_seconds=1))
        self.assertEqual(summary["status"], "time_limit")
        self.assertEqual(summary["steps"], 0)
        self.assertEqual(len(summary["hypotheses"]), 1)
        self.assertIsNone(summary["unassessed_final_step"])
        self.assertEqual(replay(path)["actions_replayed"], 0)

    def test_api_failure_and_interruption_keep_recorded_progress(self):
        for failure, status in ((NebiusError("Authentication failed"), "inference_error"), (KeyboardInterrupt(), "interrupted")):
            with self.subTest(failure=type(failure).__name__):
                summary, records, _, transport = self.run_fixture(failure)
                self.assertEqual(summary["status"], status)
                self.assertEqual(summary["steps"], 0)
                self.assertEqual(records[-1]["type"], "summary")
                self.assertEqual(transport.call_count, 1)

    def test_contract_checker_details_never_enter_the_learner_prompt(self):
        def violated(*_):
            return {"contract_violations": ["SECRET_ORACLE_RULE"]}
        summary, records, _, _ = self.run_fixture([response(decision())], evaluator=violated)
        self.assertEqual(summary["status"], "contract_failure")
        self.assertNotIn("SECRET_ORACLE_RULE", json.dumps([r["messages"] for r in records if r["type"] == "request"]))

    def test_interruption_after_learning_preserves_replayable_progress(self):
        summary, _, path, transport = self.run_fixture([
            response(decision("north")),
            response(decision("interact", 1, [hypothesis(status="supported", evidence=[1])])),
            KeyboardInterrupt(),
        ])
        self.assertEqual(summary["status"], "interrupted")
        self.assertEqual(exploration_exit_code(summary["status"]), 130)
        self.assertEqual(summary["steps"], 2)
        self.assertEqual(summary["hypotheses"][0]["evidence_steps"], (1,))
        self.assertEqual(summary["unassessed_final_step"], 2)
        self.assertFalse(summary["usage_complete"])
        self.assertEqual(transport.call_count, 3)
        self.assertEqual(replay(path)["actions_replayed"], 2)

    def test_reset_factory_difference_does_not_raise_an_environment_error(self):
        summary, records, _, _ = self.run_fixture(
            [response(decision("reset"))],
            limits=ExplorationLimits(max_actions=1),
            factory=lambda: KeyGateGame(5),
        )
        self.assertEqual(summary["status"], "action_limit")
        self.assertEqual(summary["resets"], 1)
        self.assertEqual(sum(row["type"] == "reset" for row in records), 1)

    def test_separate_sessions_start_without_learning(self):
        for _ in range(2):
            _, records, _, _ = self.run_fixture([response(decision("finish"))])
            context = json.loads(next(r for r in records if r["type"] == "request")["messages"][1]["content"])
            self.assertEqual(context["hypotheses"], [])
            self.assertIsNone(context["active_experiment"])

    def test_history_is_bounded_but_memory_remains(self):
        responses = [response(decision("wait", hypotheses=[hypothesis()]))]
        responses.extend(response(decision("wait", pending=i, hypotheses=[hypothesis(status="supported", evidence=[1])] if i == 1 else [])) for i in range(1, 8))
        responses.append(response(decision("finish", pending=8, hypotheses=[hypothesis(status="supported", evidence=[8])])))
        _, records, _, _ = self.run_fixture(responses)
        context = json.loads([r for r in records if r["type"] == "request"][-1]["messages"][1]["content"])
        self.assertEqual([r["step"] for r in context["recent_outcomes"]], [3, 4, 5, 6, 7, 8])
        self.assertEqual(len(context["hypotheses"]), 1)

    def test_terminal_state_only_accepts_reset_or_finish(self):
        actions = ["south", "south", "east", "east", "interact", "east", "east", "east", "east"]
        responses = [response(decision(action, pending=i or None, hypotheses=[hypothesis(status="uncertain", evidence=[1])] if i == 1 else [])) for i, action in enumerate(actions)]
        responses.extend([response(decision("north", pending=9)), response(decision("finish", pending=9, hypotheses=[hypothesis(status="supported", evidence=[9])]))])
        summary, records, path, _ = self.run_fixture(responses)
        self.assertEqual(summary["steps"], 9)
        self.assertEqual(summary["rejected_decisions"], 1)
        terminal = json.loads([r for r in records if r["type"] == "request"][-1]["messages"][1]["content"])
        self.assertEqual(terminal["observation"]["controls"], ["reset", "finish"])
        self.assertEqual(replay(path)["status"], "won")

    def test_replay_detects_reset_and_summary_tampering(self):
        _, records, path, _ = self.run_fixture([response(decision("reset")), response(decision("finish", pending=1))])
        original = copy.deepcopy(records)
        reset = next(r for r in records if r["type"] == "reset")
        reset["after"]["tick"] = 5
        path.write_text("\n".join(json.dumps(r) for r in records))
        with self.assertRaisesRegex(ValueError, "diverged after"):
            replay(path)
        original[-1]["resets"] = 0
        path.write_text("\n".join(json.dumps(r) for r in original))
        with self.assertRaisesRegex(ValueError, "reset counts"):
            replay(path)

    def test_credentials_are_redacted_in_learning_records(self):
        summary, _, path, _ = self.run_fixture([response(decision("finish", hypotheses=[hypothesis(claim="unit-test-key")]))])
        self.assertEqual(summary["status"], "finished")
        self.assertNotIn("unit-test-key", path.read_text())
        self.assertIn("[REDACTED]", path.read_text())

    def test_replay_detects_stale_request_observations_after_reset(self):
        _, records, path, _ = self.run_fixture([
            response(decision("south")),
            response(decision("reset", 1, [hypothesis(status="uncertain", evidence=[1])])),
            response(decision("finish", 2)),
        ])
        original = copy.deepcopy(records)
        request = [r for r in records if r["type"] == "request"][-1]
        context = json.loads(request["messages"][1]["content"])
        context["observation"]["player"]["position"] = [1, 2]
        request["messages"][1]["content"] = json.dumps(context)
        path.write_text("\n".join(json.dumps(r) for r in records))
        with self.assertRaisesRegex(ValueError, "request observation differs"):
            replay(path)
        original[-1]["inference_requests"] += 1
        path.write_text("\n".join(json.dumps(r) for r in original))
        with self.assertRaisesRegex(ValueError, "inference request count differs"):
            replay(path)

    def test_switching_experiments_requires_a_grounded_note_without_inventing_it(self):
        responses = [
            response(decision("north")),
            response(decision("interact", 1, experiment="e2")),
            response(decision("interact", 1, [hypothesis(status="uncertain", evidence=[1])], experiment="e2")),
        ]
        summary, records, _, _ = self.run_fixture(responses, ExplorationLimits(max_actions=2))
        self.assertEqual(summary["steps"], 2)
        self.assertEqual(summary["inference_requests"], 3)
        self.assertEqual(summary["rejected_decisions"], 1)
        self.assertEqual(summary["hypotheses"][0]["status"], "uncertain")
        self.assertEqual(summary["unassessed_final_step"], 2)
        self.assertTrue(summary["ready_for_live_review"])
        context = json.loads([r for r in records if r["type"] == "request"][-1]["messages"][1]["content"])
        self.assertEqual(context["hypotheses"], [])
        self.assertEqual(context["recent_outcomes"][0]["details_in"], "completed_outcome_to_assess")
        self.assertNotIn("map", context["completed_outcome_to_assess"]["before_state"])

    def test_finish_cannot_silently_discard_learning_after_experiments(self):
        summary, _, _, _ = self.run_fixture([
            response(decision("north")),
            response(decision("finish", 1)),
            response(decision("finish", 1, [hypothesis(status="supported", evidence=[1])])),
        ])
        self.assertEqual(summary["status"], "finished")
        self.assertEqual(summary["rejected_decisions"], 1)
        self.assertEqual(summary["hypotheses"][0]["evidence_steps"], (1,))


class CliTests(unittest.TestCase):
    def test_exit_codes_distinguish_partial_runs_errors_and_interruptions(self):
        for status in ("finished", "action_limit", "request_limit", "time_limit"):
            self.assertEqual(exploration_exit_code(status), 0)
        for status in ("invalid_decision", "inference_error", "contract_failure", "environment_error"):
            self.assertEqual(exploration_exit_code(status), 2)
        self.assertEqual(exploration_exit_code("interrupted"), 130)

    def test_explore_cli_and_default_limits(self):
        with patch("autoplay_qa.cli.Settings.load", return_value=Settings("unit-test-key")), patch("autoplay_qa.cli.explore", return_value=0) as handler:
            self.assertEqual(main(["explore"]), 0)
        args = handler.call_args.args[0]
        self.assertEqual((args.max_actions, args.max_requests, args.max_seconds), (24, 28, 300))
        self.assertEqual(args.max_tokens, 8192)

    def test_explore_cli_prints_a_local_summary_without_exposing_a_key(self):
        for key in ("unit-test-key", 'unit-test-key"\\escaped'):
            with self.subTest(key_type="escaped" if "\\" in key else "plain"), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "test-only.jsonl"
                client = NebiusClient(Settings(key))
                note = hypothesis(status="uncertain", claim=key)
                finished = {
                    "action": "finish", "reason": "End the fixture session",
                    "expected_position": None, "notes": [note],
                }
                with patch("autoplay_qa.cli.Settings.load", return_value=Settings(key)), patch("autoplay_qa.cli.NebiusClient", return_value=client), patch.object(client, "_request", return_value=response(finished)), patch("sys.stdout", new_callable=io.StringIO) as output:
                    self.assertEqual(main(["explore", "--log", str(path)]), 0)
                    text = output.getvalue()
                self.assertIn("Learning summary", text)
                self.assertNotIn(key, text)
                self.assertNotIn(json.dumps(key)[1:-1], text)
                self.assertNotIn(json.dumps(key)[1:-1], path.read_text())
                completion = next(r for r in map(json.loads, path.read_text().splitlines()) if r["type"] == "completion")
                self.assertEqual(json.loads(completion["answer"])["notes"][0]["claim"], "[REDACTED]")
                self.assertTrue(replay(path)["recording_complete"])


if __name__ == "__main__":
    unittest.main()
