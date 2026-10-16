"""Focus changes the investigation question, never selects a game control."""

import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from autoplay_qa.cli import begin_session, main
from autoplay_qa.config import Settings
from autoplay_qa.exploration import (
    Coverage, ExplorationLimits, exploration_messages, outcome_view,
    run_exploration, turn_schema,
)
from autoplay_qa.game import KeyGateGame
from autoplay_qa.nebius import Completion, NebiusClient
from autoplay_qa.session import SessionLog, replay
from tests.test_exploration import decision, hypothesis, response


class ExplorationFocusTests(unittest.TestCase):
    def test_focus_leaves_controls_and_observations_unrestricted_and_withholds_rules(self):
        observation = KeyGateGame().observe()
        observation["rules"] = ["SECRET_RULE"]
        observation["goal"] = "SECRET_OBJECTIVE"
        coverage = Coverage(observation)
        contexts = []
        for focus in ("mechanics", "reset-retention"):
            messages = exploration_messages(observation, 0, None, {}, [], coverage, None,
                                            {"actions": 24}, focus=focus)
            context = json.loads(messages[1]["content"])
            contexts.append(context)
            for spoiler in ("SECRET_", '"passable"', "Collect the key", "key before", "stand next"):
                self.assertNotIn(spoiler, json.dumps(messages))
        self.assertEqual(contexts[0]["observation"], contexts[1]["observation"])
        self.assertEqual(contexts[0]["coverage"], contexts[1]["coverage"])
        self.assertNotEqual(contexts[0]["session_question"], contexts[1]["session_question"])
        self.assertIn("reset", contexts[1]["observation"]["controls"])
        self.assertIn("finish", contexts[1]["observation"]["controls"])
        self.assertIn("north", contexts[1]["observation"]["controls"])

    def test_compact_history_preserves_changes_refusals_and_all_six_steps(self):
        game = KeyGateGame()
        coverage = Coverage(game.observe())
        history = []
        for step, action in enumerate(("south", "south", "east", "east", "interact", "reset", "west"), 1):
            before = game.observe()
            if action == "reset":
                game = KeyGateGame()
                after = game.observe()
            else:
                after = game.act(action)
            coverage.record(before, action, after)
            history.append({"step": step, "episode_before": int(step > 6), "episode_after": int(step >= 6),
                            "experiment_id": "fixture", "action": action, "prediction": "Fixture prediction",
                            "before": outcome_view(before), "after": outcome_view(after)})
        history = history[-6:]
        original = copy.deepcopy(history)
        messages = exploration_messages(game.observe(), 1, None, {}, history, coverage, 7,
                                        {"actions": 17}, response_schema=turn_schema(["finish"], set(range(1, 8)), 7))
        context = json.loads(messages[1]["content"])
        recent = context["recent_outcomes"]
        self.assertEqual([item["step"] for item in recent], list(range(2, 8)))
        pickup = next(item for item in recent if item["step"] == 3)
        self.assertEqual(pickup["observed_changes"]["inventory_after"], ["key"])
        opened = next(item for item in recent if item["step"] == 5)
        self.assertTrue(opened["observed_changes"]["changed_objects"]["gate"]["after"]["open"])
        reset = next(item for item in recent if item["step"] == 6)
        self.assertEqual((reset["episode_before"], reset["episode_after"]), (0, 1))
        latest = context["completed_outcome_to_assess"]
        self.assertEqual(latest["step"], 7)
        self.assertEqual(latest["before_state"]["player"]["position"], latest["after_state"]["player"]["position"])
        self.assertEqual(latest["after_state"]["events"], history[-1]["after"]["events"])
        self.assertEqual(recent[-1]["details_in"], "completed_outcome_to_assess")
        self.assertLess(len(json.dumps(recent)), len(json.dumps(original)))
        self.assertEqual(history, original)

    def test_reset_focus_preserves_evidence_and_does_not_override_decisions(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fixture.jsonl"
            game = KeyGateGame()
            client = NebiusClient(Settings("unit-test-key"))
            responses = [
                response(decision("interact", experiment="chosen-by-model")),
                response(decision("reset", 1, [hypothesis(status="supported", evidence=[1])])),
                response(decision("interact", 2)),
                response(decision("finish", 3, [hypothesis(status="supported", evidence=[1, 3])])),
            ]
            with SessionLog(path) as log:
                begin_session(log, game, "unit-test-fixtures", schema_version=2, task="explore", focus="reset-retention")
                with patch.object(client, "_request", side_effect=responses):
                    summary = run_exploration(game, client, log, lambda: KeyGateGame(), focus="reset-retention")
            records = [json.loads(line) for line in path.read_text().splitlines()]
            controls = [r["decision"]["action"] for r in records if r["type"] in {"step", "reset", "finish"}]
            self.assertEqual(controls, ["interact", "reset", "interact", "finish"])
            requests = [json.loads(r["messages"][1]["content"]) for r in records if r["type"] == "request"]
            self.assertEqual(requests[2]["hypotheses"][0]["evidence_steps"], [1])
            self.assertEqual(summary["hypotheses"][0]["evidence_steps"], (1, 3))
            self.assertEqual(summary["focus"], "reset-retention")
            self.assertEqual(replay(path)["resets_replayed"], 1)

    def test_unknown_focus_stops_before_any_request_or_action(self):
        game = KeyGateGame()
        client = NebiusClient(Settings("unit-test-key"))
        with tempfile.TemporaryDirectory() as directory, SessionLog(Path(directory) / "fixture.jsonl") as log:
            with patch.object(client, "complete") as inference, self.assertRaises(ValueError):
                run_exploration(game, client, log, lambda: KeyGateGame(), focus="unknown")
            inference.assert_not_called()
            self.assertEqual(game.tick, 0)

    def test_cross_episode_metrics_report_trials_without_choosing_controls(self):
        game = KeyGateGame()
        initial = game.observe()
        coverage = Coverage(initial)
        moved = game.act("south")
        coverage.record(initial, "south", moved, step=1, episode=0)
        restored = KeyGateGame()
        coverage.record(moved, "reset", restored.observe(), step=2, episode=0)
        repeated = restored.act("south")
        coverage.record(initial, "south", repeated, step=3, episode=1)
        before = restored.observe()
        after = restored.act("south")
        coverage.record(before, "south", after, step=4, episode=1)
        context = coverage.context(after)
        self.assertEqual(context["reset_use_count"], 1)
        self.assertEqual(context["cross_episode_retests"], [{
            "action": "south", "position": [1, 1],
            "trials": [{"episode": 0, "steps": [1]}, {"episode": 1, "steps": [3]}],
        }])
        self.assertNotIn("next_action", context)
        self.assertNotIn("passable", json.dumps(context))
        self.assertEqual(coverage.summary()["distinct_observed_states"], 3)

    def test_observed_latency_informs_budget_without_overriding_model_controls(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fixture.jsonl"
            game = KeyGateGame()
            client = NebiusClient(Settings("unit-test-key"))
            original_complete = client.complete

            def timed_completion(*args, **kwargs):
                completion = original_complete(*args, **kwargs)
                completion.metadata["latency_ms"] = 30_000
                return completion

            with SessionLog(path) as log:
                begin_session(log, game, "unit-test-fixtures", schema_version=2, task="explore")
                with patch.object(client, "_request", side_effect=[
                    response(decision("north")),
                    response(decision("finish", 1, [hypothesis(status="uncertain", evidence=[1])])),
                ]), patch.object(client, "complete", side_effect=timed_completion):
                    summary = run_exploration(game, client, log, lambda: KeyGateGame())
            records = [json.loads(line) for line in path.read_text().splitlines()]
            requests = [json.loads(r["messages"][1]["content"]) for r in records if r["type"] == "request"]
            self.assertIsNone(requests[0]["remaining_budget"]["recent_request_median_seconds"])
            self.assertEqual(requests[1]["remaining_budget"]["recent_request_median_seconds"], 30)
            self.assertEqual(requests[1]["remaining_budget"]["approximate_remaining_decisions"], 9)
            self.assertEqual(requests[0]["observation"]["controls"], requests[1]["observation"]["controls"])
            self.assertEqual(summary["status"], "finished")
            self.assertEqual(summary["steps"], 1)
            self.assertEqual(replay(path)["actions_replayed"], 1)

    def test_cli_focus_defaults_and_bounds_are_preserved(self):
        for arguments, focus in ((["explore"], "mechanics"), (["explore", "--focus", "reset-retention"], "reset-retention")):
            with patch("autoplay_qa.cli.explore", return_value=0) as handler:
                self.assertEqual(main(arguments), 0)
            args = handler.call_args.args[0]
            self.assertEqual(args.focus, focus)
            self.assertEqual((args.max_actions, args.max_requests, args.max_seconds, args.max_tokens), (24, 28, 300, 8192))

    def test_action_limit_does_not_advertise_a_reflection_after_the_last_action(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fixture.jsonl"
            game = KeyGateGame()
            client = NebiusClient(Settings("unit-test-key"))
            completions = [Completion(json.dumps(value), {
                "finish_reason": "stop", "usage": {}, "latency_ms": 30_000,
            }) for value in (decision("north"), decision("finish", 1, [hypothesis(status="uncertain", evidence=[1])]))]
            with SessionLog(path) as log:
                begin_session(log, game, "unit-test-fixtures", schema_version=2, task="explore")
                with patch.object(client, "complete", side_effect=completions):
                    summary = run_exploration(game, client, log, lambda: KeyGateGame(), ExplorationLimits(max_actions=2))
            records = [json.loads(line) for line in path.read_text().splitlines()]
            request = [r for r in records if r["type"] == "request"][1]
            budget = json.loads(request["messages"][1]["content"])["remaining_budget"]
            self.assertEqual(budget["actions"], 1)
            self.assertEqual(budget["approximate_remaining_decisions"], 1)
            self.assertEqual(summary["status"], "finished")
            self.assertIsNone(summary["unassessed_final_step"])

    def test_review_cli_replays_without_credentials_or_inference(self):
        import io

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "human.jsonl"
            game = KeyGateGame()
            with SessionLog(path) as log:
                begin_session(log, game, "human")
                log.write("summary", status="quit", steps=0, inference_requests=0, final_observation=game.observe())
            with patch("autoplay_qa.cli.Settings.load", side_effect=AssertionError("Must not load a credential")), patch("sys.stdout", new_callable=io.StringIO) as output:
                self.assertEqual(main(["review", str(path)]), 0)
            result = json.loads(output.getvalue())
            self.assertTrue(result["execution_replayed"])
            self.assertFalse(result["bugs_verified"])


if __name__ == "__main__":
    unittest.main()
