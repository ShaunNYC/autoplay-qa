"""Model notes remain tentative beside actual replayed action receipts."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from autoplay_qa.cli import begin_session
from autoplay_qa.config import Settings
from autoplay_qa.exploration import run_exploration
from autoplay_qa.game import KeyGateGame
from autoplay_qa.nebius import NebiusClient
from autoplay_qa.review import review_session
from autoplay_qa.session import SessionLog
from tests.test_review import decision, note, response


class HypothesisEvidenceTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.path = Path(directory.name) / "evidence.jsonl"

    def explore(self, decisions):
        game = KeyGateGame(0)
        client = NebiusClient(Settings("unit-test-key"))
        with SessionLog(self.path, ("unit-test-key",)) as log:
            begin_session(log, game, "unit-test-fixtures", schema_version=2, task="explore")
            with patch.object(client, "_request", side_effect=[response(value) for value in decisions]):
                run_exploration(game, client, log, lambda: KeyGateGame(0))
        return [json.loads(line) for line in self.path.read_text().splitlines()]

    def rewrite(self, records):
        self.path.write_text("\n".join(json.dumps(record) for record in records) + "\n")

    def test_false_pickup_claim_shows_actual_controls_and_empty_inventory(self):
        claim = "The key entered inventory."
        wrong = "The key entered inventory, matching the prediction."
        first = decision("north", notes=[note(claim=claim)])
        second = decision("reset", 1, [note([1], claim)])
        second["assessment"]["outcome"] = wrong
        final = decision("finish", 2, [note([2, 1], claim)])
        final["assessment"]["outcome"] = "The reset collected the key, matching the prediction."
        records = self.explore([first, second, final])
        records[-1]["hypotheses"] = [note([1], "A fabricated summary claim")]
        self.rewrite(records)

        ordinary = review_session(self.path)
        self.assertEqual(review_session(self.path, include_evidence=False), ordinary)
        detailed = review_session(self.path, include_evidence=True)
        self.assertEqual({key: value for key, value in detailed.items() if key != "hypothesis_evidence"}, ordinary)
        self.assertNotIn("hypothesis_evidence", ordinary)
        self.assertNotIn(claim, json.dumps(ordinary))
        evidence = detailed["hypothesis_evidence"]
        self.assertTrue(evidence["available"])
        self.assertTrue(evidence["recording_complete"])
        self.assertFalse(evidence["claims_verified"])
        self.assertFalse(evidence["assessments_verified"])
        self.assertIn("Learner-visible", evidence["state_projection"])
        self.assertEqual(len(evidence["notes"]), 1)
        learned = evidence["notes"][0]
        self.assertEqual(learned["claim"], claim)
        self.assertEqual(learned["model_status"], "supported")
        self.assertTrue(learned["tentative"])
        self.assertFalse(learned["verified"])
        self.assertEqual(learned["evidence_steps"], [2, 1])
        receipts = learned["cited_steps"]
        self.assertEqual([(item["step"], item["control"], item["episode_before"], item["episode_after"]) for item in receipts], [(2, "reset", 0, 1), (1, "north", 0, 0)])
        self.assertEqual(receipts[1]["model_assessment"], {"step": 1, "outcome": wrong})
        self.assertEqual(receipts[1]["model_prediction"], first["prediction"])
        self.assertEqual(receipts[0]["model_assessment"], final["assessment"])
        for receipt in receipts:
            for state in (receipt["before"], receipt["after"]):
                self.assertEqual(set(state), {"status", "player", "objects", "events"})
                self.assertEqual(state["player"]["inventory"], [])
                self.assertEqual(state["objects"], KeyGateGame(0).observe()["objects"])
        self.assertIn("blocked", receipts[1]["after"]["events"][0].lower())

    def test_cross_reset_receipts_preserve_trials_and_model_assessment_text(self):
        wrong = "The repeated north moved onto the key, matching my prediction."
        final = decision("finish", 3, [note([1, 3])])
        final["assessment"]["outcome"] = wrong
        self.explore([
            decision("north", notes=[note()]),
            decision("reset", 1, [note([1])]),
            decision("north", 2),
            final,
        ])
        evidence = review_session(self.path, include_evidence=True)["hypothesis_evidence"]
        learned = evidence["notes"][0]
        self.assertEqual(learned["evidence_steps"], [1, 3])
        original, repeated = learned["cited_steps"]
        self.assertEqual([item["control"] for item in (original, repeated)], ["north", "north"])
        self.assertEqual([(item["episode_before"], item["episode_after"]) for item in (original, repeated)], [(0, 0), (1, 1)])
        self.assertEqual(original["before"], repeated["before"])
        self.assertEqual(original["after"], repeated["after"])
        self.assertEqual(repeated["model_assessment"], {"step": 3, "outcome": wrong})
        self.assertEqual(repeated["after"]["player"]["position"], [1, 1])
        self.assertEqual(repeated["after"]["player"]["inventory"], [])
        self.assertFalse(learned["verified"])
        self.assertFalse(evidence["assessments_verified"])

    def test_episode_start_and_reset_receipts_use_neutral_learner_events(self):
        records = self.explore([
            decision("north", notes=[note()]),
            decision("reset", 1, [note([1])]),
            decision("finish", 2, [note([1, 2])]),
        ])
        self.assertNotEqual(records[0]["initial_observation"]["events"], ["Environment started."])
        evidence = review_session(self.path, include_evidence=True)["hypothesis_evidence"]
        first, reset = evidence["notes"][0]["cited_steps"]
        self.assertEqual(first["before"]["events"], ["Environment started."])
        self.assertEqual(reset["after"]["events"], ["Environment started."])
        actual_reset = next(record for record in records if record["type"] == "reset")
        self.assertEqual(reset["before"]["events"], actual_reset["before"]["events"])
        self.assertNotIn("Find the key", json.dumps(evidence))

    def test_schema_one_explicitly_has_no_hypothesis_evidence(self):
        game = KeyGateGame(0)
        with SessionLog(self.path) as log:
            begin_session(log, game, "human")
            log.write("summary", status="finished", steps=0, final_observation=game.observe())
        evidence = review_session(self.path, include_evidence=True)["hypothesis_evidence"]
        self.assertFalse(evidence["available"])
        self.assertTrue(evidence["recording_complete"])
        self.assertEqual(evidence["notes"], [])
        self.assertIn("Schema 1", evidence["reason"])

    def test_incomplete_capture_does_not_invent_missing_assessment(self):
        records = self.explore([
            decision("north", notes=[note()]),
            decision("finish", 1, [note([1])]),
        ])
        finish_index = next(index for index, record in enumerate(records) if record["type"] == "finish")
        incomplete = records[:finish_index]
        for record in incomplete:
            if record["type"] == "memory":
                record.pop("assessment", None)
            if record["type"] in {"step", "reset"}:
                record["decision"].pop("assessment", None)
        # A completion still contains model comparison text. It must not be
        # promoted to an accepted assessment when capture metadata is missing.
        self.assertTrue(any(record["type"] == "completion" and "Matched the recorded result" in record.get("answer", "") for record in incomplete))
        self.rewrite(incomplete)
        result = review_session(self.path, include_evidence=True)
        self.assertFalse(result["recording_complete"])
        self.assertTrue(result["partial"])
        evidence = result["hypothesis_evidence"]
        self.assertFalse(evidence["recording_complete"])
        receipt = evidence["notes"][0]["cited_steps"][0]
        self.assertIsNone(receipt["model_assessment"])
        self.assertEqual(receipt["model_prediction"], "The observable state may change")
        self.assertEqual(receipt["control"], "north")

    def test_replay_divergence_prevents_evidence_construction(self):
        records = self.explore([
            decision("north", notes=[note()]),
            decision("reset", 1, [note([1])]),
            decision("finish", 2, [note([1, 2])]),
        ])
        reset = next(record for record in records if record["type"] == "reset")
        reset["after"]["player"]["position"] = [2, 1]
        self.rewrite(records)
        with patch("autoplay_qa.review._recorded_hypothesis_evidence") as builder:
            with self.assertRaisesRegex(ValueError, "Replay diverged after action 2"):
                review_session(self.path, include_evidence=True)
            builder.assert_not_called()

    def test_later_execution_does_not_validate_a_future_memory_reference(self):
        records = self.explore([
            decision("north", notes=[note()]),
            decision("east", 1, [note([1])]),
            decision("finish", 2, [note([1, 2])]),
        ])
        last_action = next(index for index, record in enumerate(records) if record["type"] == "step" and record["step"] == 2)
        incomplete = records[:last_action + 1]
        latest_memory = [record for record in incomplete if record["type"] == "memory"][-1]
        latest_memory["hypotheses"][0]["evidence_steps"] = [2]
        self.rewrite(incomplete)
        result = review_session(self.path, include_evidence=True)
        self.assertTrue(result["execution_replayed"])
        self.assertEqual(result["coverage"]["game_actions"], 2)
        learned = result["hypothesis_evidence"]["notes"][0]
        self.assertEqual(learned["evidence_steps"], [2])
        self.assertEqual(learned["cited_steps"], [{
            "step": 2, "available": False,
            "reason": "Referenced action was not completed when this memory was recorded.",
        }])
        self.assertFalse(learned["verified"])


if __name__ == "__main__":
    unittest.main()
