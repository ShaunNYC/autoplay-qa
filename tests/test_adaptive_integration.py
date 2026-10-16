"""Real loop/replay bindings and durable authorization with stub inference."""

from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from autoplay_qa.adaptive import AdaptiveContext
from autoplay_qa.adaptive_protocol import context_class
from autoplay_qa.adaptive_budget import SupplementalAllowance
from autoplay_qa.cli import begin_session
from autoplay_qa.config import Settings
from autoplay_qa.dashboard_runner import BudgetExceeded, TotalAllowance
from autoplay_qa.experience import ExperienceStore
from autoplay_qa.exploration import ExplorationLimits
from autoplay_qa.game import KeyGateGame
from autoplay_qa.lean import run_lean_exploration
from autoplay_qa.nebius import NebiusClient, TransientNebiusError
from autoplay_qa.review import review_session
from autoplay_qa.session import SessionLog, replay
from tests.test_lean_exploration import decision, response


class AdaptiveIntegrationTests(unittest.TestCase):
    def fixture(self, actions, *, store=None, maximum=24000, game=None,
                version="adaptive-v1", run_id="fixture-run"):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        path = Path(directory.name) / "adaptive.jsonl"
        game = game or KeyGateGame(0)
        context = context_class(version)(game.observe(), store=store, run_id=run_id, max_prompt_bytes=maximum)
        client = NebiusClient(Settings("fake-key"))
        replies = [item if isinstance(item, BaseException) else response(decision(item)) for item in actions]
        with SessionLog(path) as log:
            begin_session(log, game, "fake", schema_version=2, task="explore", focus="qa",
                          prompt_version=version,
                          adaptive_config={"run_id": run_id, "max_prompt_bytes": maximum})
            with patch.object(client, "_request", side_effect=replies) as http:
                result = run_lean_exploration(game, client, log, game.reset,
                                              ExplorationLimits(max_actions=24), focus="qa", adaptive=context)
        return result, path, http

    def test_actual_loop_replays_generic_facts_and_review(self):
        result, path, http = self.fixture(["north", "north", "east", "reset", "finish"])
        self.assertEqual(result["steps"], 4)
        self.assertEqual(result["adaptive"]["repeated_unchanged_state_action_attempts"], 1)
        self.assertTrue(replay(path)["adaptive_facts_validated"])
        self.assertEqual(review_session(path)["prompt_version"], "adaptive-v1")
        self.assertEqual(http.call_count, 5)

    def test_persistent_history_is_received_in_another_run_without_relabeling(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "memory.sqlite"
            initial = KeyGateGame(0).observe()
            with ExperienceStore(path, project_id="game-A", game_version=KeyGateGame.version,
                                 initial_observation=initial) as store:
                self.fixture(["north", "finish"], store=store)
            with ExperienceStore(path, project_id="game-A", game_version=KeyGateGame.version,
                                 initial_observation=initial, read_only=True) as store:
                result, recording, _ = self.fixture(["east", "finish"], store=store)
            rows = [json.loads(line) for line in recording.read_text().splitlines()]
            context = json.loads(next(row for row in rows if row["type"] == "request")["messages"][1]["content"])
            self.assertTrue(context["retrieved_experiments"]["facts"])
            self.assertFalse(replay(recording)["prior_run_memory_execution_verified"])

    def test_tampered_experiment_fact_is_rejected(self):
        _, path, _ = self.fixture(["north", "finish"])
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        next(row for row in rows if row["type"] == "step")["experiment_fact"]["action"] = "west"
        path.write_text("\n".join(json.dumps(row) for row in rows))
        with self.assertRaises(ValueError):
            replay(path)

    def test_tampered_current_state_or_native_action_schema_is_rejected(self):
        for target in ("observation", "schema"):
            _, path, _ = self.fixture(["east", "finish"])
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            request = next(row for row in rows if row["type"] == "request")
            if target == "schema":
                request["response_schema"]["properties"]["action"]["enum"].append("teleport")
            else:
                context = json.loads(request["messages"][1]["content"])
                context["observation"]["player"]["position"] = [7, 3]
                request["messages"][1]["content"] = json.dumps(context)
            path.write_text("\n".join(json.dumps(row) for row in rows))
            with self.assertRaises(ValueError):
                replay(path)

    def test_adaptive_provider_failure_is_terminal_without_retry(self):
        result, path, http = self.fixture([TransientNebiusError("offline", reason="timeout")])
        self.assertEqual(result["status"], "provider_unavailable")
        self.assertEqual(result["steps"], 0)
        self.assertEqual(http.call_count, 1)
        self.assertTrue(replay(path)["recording_complete"])

    def test_v2_loop_and_historical_version_dispatch_replay(self):
        for version in ("adaptive-v1", "adaptive-v2"):
            with self.subTest(version=version):
                result, path, http = self.fixture(["north", "north", "east", "reset", "finish"],
                                                   version=version)
                self.assertEqual(result["prompt_version"], version)
                self.assertEqual(result["steps"], 4)
                self.assertTrue(replay(path)["adaptive_facts_validated"])
                self.assertEqual(review_session(path)["prompt_version"], version)
                self.assertEqual(http.call_count, 5)

    def test_v2_warm_aggregate_inputs_replay_without_database_mutation(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "memory.sqlite"
            initial = KeyGateGame(0).observe()
            with ExperienceStore(path, project_id="game-A", game_version=KeyGateGame.version,
                                 initial_observation=initial) as store:
                self.fixture(["north", "finish"], store=store, version="adaptive-v2", run_id="training")
            with ExperienceStore(path, project_id="game-A", game_version=KeyGateGame.version,
                                 initial_observation=initial, read_only=True) as store:
                before = store.summary()
                result, recording, _ = self.fixture(["north", "east", "finish"], store=store,
                                                     version="adaptive-v2", run_id="evaluation")
                self.assertEqual(store.summary(), before)
            self.assertTrue(replay(recording)["adaptive_facts_validated"])
            self.assertFalse(replay(recording)["prior_run_memory_execution_verified"])
            self.assertEqual(result["adaptive"]["repeated_unchanged_state_action_attempts"], 0)

    def test_unsupported_adaptive_version_fails_closed(self):
        _, path, _ = self.fixture(["east", "finish"])
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        rows[0]["prompt_version"] = "adaptive-future"
        path.write_text("\n".join(json.dumps(row) for row in rows))
        with self.assertRaises(ValueError):
            replay(path)


class SupplementalBudgetTests(unittest.TestCase):
    def test_old_unknown_hold_and_old_runner_stay_compatible(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "allowance.json"
            old = TotalAllowance(path)
            old.reserve("old-unknown", 1)
            old.settle("old-unknown", {"usage_known": False})
            before = deepcopy(json.loads(path.read_text())["jobs"]["old-unknown"])
            with self.assertRaises(ValueError):
                SupplementalAllowance(path)
            new = SupplementalAllowance(path, authorize=True)
            new.reserve("new", 2)
            new.settle("new", {"usage_known": True, "accounted_cost_usd": 0.03})
            self.assertEqual(json.loads(path.read_text())["jobs"]["old-unknown"], before)
            self.assertEqual(new.summary()["occupied_usd"], 0.03)
            TotalAllowance(path)  # Existing on-line process can still open it.

    def test_restarting_or_renaming_cannot_renew_the_fifteen_dollars(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "allowance.json"
            TotalAllowance(path)
            grant = SupplementalAllowance(path, authorize=True)
            grant.reserve("unknown", 15)
            grant.settle("unknown", {"usage_known": False})
            restart = SupplementalAllowance(path, authorize=True)
            with self.assertRaises(BudgetExceeded):
                restart.reserve("another", 0.01)
            with self.assertRaises(ValueError):
                SupplementalAllowance(path, round_id="renamed", authorize=True)
            self.assertEqual(restart.summary()["occupied_usd"], 15)

    def test_cross_round_settlement_and_duplicate_dispatch_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "allowance.json"
            old = TotalAllowance(path)
            old.reserve("old", 1)
            grant = SupplementalAllowance(path, authorize=True)
            with self.assertRaises(ValueError):
                grant.settle("old", {"usage_known": True, "accounted_cost_usd": 0})
            grant.reserve("new", 1)
            with self.assertRaises(ValueError):
                grant.reserve("new", 1)


if __name__ == "__main__":
    unittest.main()
