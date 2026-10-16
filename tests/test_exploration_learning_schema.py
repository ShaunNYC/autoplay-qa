"""First learning requires one recorded note, not evidence for every proposal."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from autoplay_qa.cli import begin_session
from autoplay_qa.config import Settings
from autoplay_qa.exploration import (
    CONTROLS, EXPLORATION_SCHEMA, ExplorationDecision, run_exploration, turn_schema,
)
from autoplay_qa.game import KeyGateGame
from autoplay_qa.nebius import NebiusClient
from autoplay_qa.session import SessionLog, replay
from tests.test_exploration import decision, hypothesis, response


class FirstLearningSchemaTests(unittest.TestCase):
    def run_fixture(self, decisions):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        path = Path(directory.name) / "fixture.jsonl"
        game = KeyGateGame()
        client = NebiusClient(Settings("unit-test-key"))
        with SessionLog(path) as log:
            begin_session(log, game, "unit-test-fixtures", schema_version=2, task="explore")
            with patch.object(client, "_request", side_effect=[response(item) for item in decisions]) as transport:
                summary = run_exploration(game, client, log, lambda: KeyGateGame())
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        contexts = [json.loads(row["messages"][1]["content"]) for row in rows if row["type"] == "request"]
        return summary, rows, contexts, path, transport

    def test_uncertain_note_and_untested_proposal_fit_first_learning_schema(self):
        schema = turn_schema(list(CONTROLS), {1}, 1, require_learning=True)
        updates = [hypothesis("movement", "uncertain", [1]), hypothesis("interaction", "untested", [])]
        value = decision("finish", 1, updates)
        parsed = ExplorationDecision.parse(json.dumps(value), list(CONTROLS), {1}, 1)
        self.assertEqual(parsed.hypotheses[0].evidence_steps, (1,))
        self.assertEqual(parsed.hypotheses[1].evidence_steps, ())
        array = schema["properties"]["hypotheses"]
        self.assertEqual(array["minItems"], 1)
        self.assertIn("At least one update", array["description"])
        item = array["items"]["properties"]
        for update in updates:
            self.assertIn(update["status"], item["status"]["enum"])
            self.assertGreaterEqual(len(update["evidence_steps"]), item["evidence_steps"].get("minItems", 0))
            for step in update["evidence_steps"]:
                self.assertIn(step, item["evidence_steps"]["items"]["enum"])
        self.assertNotIn("description", EXPLORATION_SCHEMA["properties"]["hypotheses"])
        self.assertNotIn("minItems", turn_schema(list(CONTROLS), {1}, 1)["properties"]["hypotheses"])

    def test_controller_accepts_mixed_first_learning_updates(self):
        summary, rows, contexts, path, transport = self.run_fixture([
            decision("south", hypotheses=[hypothesis("movement")]),
            decision("finish", 1, [hypothesis("movement", "uncertain", [1]),
                                  hypothesis("interaction", "untested", [])]),
        ])
        self.assertEqual(summary["status"], "finished")
        self.assertEqual(summary["steps"], 1)
        self.assertEqual(summary["rejected_decisions"], 0)
        self.assertEqual(transport.call_count, 2)
        self.assertIsNone(summary["unassessed_final_step"])
        self.assertEqual({note["id"]: note["status"] for note in summary["hypotheses"]},
                         {"movement": "uncertain", "interaction": "untested"})
        self.assertIn("untested", contexts[1]["response_schema"]["properties"]["hypotheses"]["items"]["properties"]["status"]["enum"])
        self.assertEqual(replay(path)["actions_replayed"], 1)

    def test_all_untested_update_is_rejected_atomically_and_repaired_once(self):
        original = hypothesis("movement", claim="Original memory")
        rejected = hypothesis("movement", claim="REJECTED_MEMORY")
        summary, rows, contexts, path, transport = self.run_fixture([
            decision("south", hypotheses=[original]),
            decision("east", 1, [rejected], experiment="REJECTED_EXPERIMENT"),
            decision("finish", 1, [hypothesis("movement", "uncertain", [1], "Original memory")]),
        ])
        self.assertEqual(summary["status"], "finished")
        self.assertEqual(summary["steps"], 1)
        self.assertEqual(summary["rejected_decisions"], 1)
        self.assertEqual(transport.call_count, 3)
        self.assertEqual(contexts[1]["observation"], contexts[2]["observation"])
        self.assertEqual(contexts[1]["hypotheses"], contexts[2]["hypotheses"])
        self.assertEqual(contexts[2]["active_experiment"]["id"], "e1")
        self.assertNotIn("REJECTED_MEMORY", json.dumps(contexts[2]))
        self.assertIn("Record a first tentative learned hypothesis", contexts[2]["validation_feedback"])
        rejected_completion = next(row for row in rows if row["type"] == "completion" and row["request_number"] == 2)
        self.assertIsNone(rejected_completion["answer"])
        self.assertFalse(any(row["type"] in {"memory", "step", "reset", "finish"} and row["request_number"] == 2 for row in rows))
        self.assertEqual(replay(path)["actions_replayed"], 1)

    def test_repeated_all_untested_updates_stop_after_only_one_repair(self):
        summary, rows, contexts, path, transport = self.run_fixture([
            decision("south", hypotheses=[hypothesis("movement", claim="Original memory")]),
            decision("east", 1, [hypothesis("movement", claim="REJECTED_MEMORY")]),
            decision("east", 1, [hypothesis("movement", claim="REJECTED_MEMORY")]),
            decision("finish", 1, [hypothesis("movement", "uncertain", [1])]),
        ])
        self.assertEqual(summary["status"], "invalid_decision")
        self.assertEqual(summary["steps"], 1)
        self.assertEqual(summary["rejected_decisions"], 2)
        self.assertEqual(transport.call_count, 3)
        self.assertEqual(summary["hypotheses"][0]["claim"], "Original memory")
        self.assertEqual(summary["unassessed_final_step"], 1)
        self.assertEqual(replay(path)["actions_replayed"], 1)


if __name__ == "__main__":
    unittest.main()
