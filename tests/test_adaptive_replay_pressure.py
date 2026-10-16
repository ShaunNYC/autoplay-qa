"""Exercise the actual stubbed agent loop after optional context data is cut."""

import copy
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

from autoplay_qa.adaptive import AdaptiveContext, adaptive_input_bytes
from autoplay_qa.cli import begin_session
from autoplay_qa.config import Settings
from autoplay_qa.exploration import ExplorationLimits
from autoplay_qa.game import KeyGateGame
from autoplay_qa.lean import run_lean_exploration
from autoplay_qa.nebius import NebiusClient
from autoplay_qa.review import review_session
from autoplay_qa.session import SessionLog, replay
from tests.test_lean_exploration import decision, response


class PressureStore:
    """Large declared external packet; local execution remains independently real."""

    def __init__(self, *, sqlite_failure=False):
        self.sqlite_failure = sqlite_failure

    def record(self, *args, **kwargs):
        if self.sqlite_failure:
            raise sqlite3.OperationalError("database is locked at /Users/private/experience.db")

    def retrieve(self, *args, **kwargs):
        if self.sqlite_failure:
            raise sqlite3.OperationalError("database is locked at /Users/private/experience.db")
        return {
            "scope_id": "opaque-pressure-scope", "current_state_id": "retained-input",
            "facts": [{
                "action": "north", "attempts": i + 1,
                "outcome": {"observed_text": "Old recorded feedback " + "x" * 620},
                "citations": [{"run_id": "previous-run", "step": i + 1}],
            } for i in range(8)],
            "omitted_count": 3,
        }

    def summary(self):
        if self.sqlite_failure:
            raise sqlite3.OperationalError("database is locked at /Users/private/experience.db")
        return {"retained_input": True}


class AdaptiveReplayPressureTests(unittest.TestCase):
    def fixture(self, *, store=None, maximum=8000):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        path = Path(directory.name) / "pressure.jsonl"
        game = KeyGateGame(0)
        adaptive = AdaptiveContext(game.observe(), store=store or PressureStore(),
                                   run_id="pressure-run", max_prompt_bytes=maximum)
        client = NebiusClient(Settings("fake-key"))
        notes = [{"id": f"n{i}", "claim": "Tentative mechanics hypothesis " + "z" * 160,
                  "status": "uncertain", "evidence_steps": []} for i in range(6)]
        replies = [response(decision("north", notes=notes))]
        replies += [response(decision(action)) for action in ("north", "east", "west", "north", "finish")]
        with SessionLog(path) as log:
            begin_session(log, game, "fake", schema_version=2, task="explore", focus="qa",
                          prompt_version="adaptive-v1",
                          adaptive_config={"run_id": "pressure-run", "max_prompt_bytes": maximum})
            with patch.object(client, "_request", side_effect=replies) as http:
                result = run_lean_exploration(game, client, log, game.reset,
                                              ExplorationLimits(max_actions=24), focus="qa", adaptive=adaptive)
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        return result, path, rows, http

    def test_optional_working_set_omissions_replay_with_exact_local_facts(self):
        result, path, rows, http = self.fixture()
        self.assertEqual(result["status"], "finished")
        self.assertEqual(result["steps"], 5)
        self.assertEqual(http.call_count, 6)
        requests = [row for row in rows if row["type"] == "request"]
        contexts = [json.loads(row["messages"][1]["content"]) for row in requests]
        for request in requests:
            self.assertLessEqual(adaptive_input_bytes(request["messages"], request["response_schema"]), 8000)
        self.assertTrue(any(value["context_compaction"]["notes_omitted"] for value in contexts))
        self.assertTrue(any(value["context_compaction"]["history_omitted"] for value in contexts))
        self.assertTrue(any(value["context_compaction"]["memory_budget_omitted"] for value in contexts))
        for value in contexts:
            receipt = value["context_compaction"]
            memory = value["retrieved_experiments"]
            self.assertEqual(receipt["memory_candidates"], len(memory["facts"]) + receipt["memory_budget_omitted"])
            self.assertEqual(memory["omitted_count"], receipt["memory_omitted"])
        self.assertTrue(replay(path)["adaptive_facts_validated"])
        self.assertFalse(replay(path)["prior_run_memory_execution_verified"])
        self.assertEqual(review_session(path)["prompt_version"], "adaptive-v1")

    def test_corrupt_or_inconsistent_compaction_receipts_are_rejected(self):
        for corruption in ("negative", "local_count", "memory_count"):
            _, path, rows, _ = self.fixture()
            request = next(row for row in rows if row["type"] == "request" and json.loads(row["messages"][1]["content"])["context_compaction"]["completed_steps"])
            context = json.loads(request["messages"][1]["content"])
            if corruption == "negative":
                context["context_compaction"]["history_included"] = -1
            elif corruption == "local_count":
                context["context_compaction"]["completed_steps"] += 1
            else:
                context["retrieved_experiments"]["context_candidates"] += 1
            request["messages"][1]["content"] = json.dumps(context, separators=(",", ":"))
            path.write_text("\n".join(json.dumps(row) for row in rows))
            with self.subTest(corruption=corruption), self.assertRaises(ValueError):
                replay(path)

    def test_sqlite_failures_keep_executed_controls_and_typed_warnings(self):
        result, path, rows, http = self.fixture(store=PressureStore(sqlite_failure=True), maximum=24000)
        self.assertEqual(result["status"], "finished")
        self.assertEqual(result["steps"], 5)
        self.assertEqual(http.call_count, 6)
        self.assertEqual(sum(row["type"] == "step" for row in rows), 5)
        self.assertEqual(result["adaptive"]["executed_controls"], 5)
        warnings = result["adaptive"]["memory"]["warnings"]
        self.assertEqual({warning["code"] for warning in warnings}, {
            "experience_record_failed", "experience_retrieval_failed", "experience_summary_failed",
        })
        self.assertEqual({warning["error_type"] for warning in warnings}, {"OperationalError"})
        self.assertNotIn("/Users/private/", json.dumps(result))
        self.assertTrue(replay(path)["adaptive_facts_validated"])


if __name__ == "__main__":
    unittest.main()
