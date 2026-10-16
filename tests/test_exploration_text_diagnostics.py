"""Safe text diagnostics identify malformed fields without echoing their data."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from autoplay_qa.cli import begin_session
from autoplay_qa.config import Settings
from autoplay_qa.exploration import _text, run_exploration
from autoplay_qa.game import KeyGateGame
from autoplay_qa.nebius import InvalidDecision, NebiusClient
from autoplay_qa.session import SessionLog, replay
from tests.test_exploration import decision, response


class TextDiagnosticTests(unittest.TestCase):
    def test_non_string_reports_only_bounded_json_type(self):
        marker = "PRIVATE_INVALID_PAYLOAD"
        for value, kind in ((None, "null"), (True, "boolean"), (123456789, "integer"),
                            (12345.6789, "number"), ([marker], "array"), ({marker: marker}, "object")):
            with self.subTest(kind=kind), self.assertRaises(InvalidDecision) as caught:
                _text(value, "reason", 240)
            message = str(caught.exception)
            self.assertEqual(message, f"reason must be a string; received type {kind}.")
            self.assertNotIn(marker, message)
            self.assertNotIn("123456789", message)
            self.assertNotIn("12345.6789", message)
        unknown = type(marker, (), {"__str__": lambda self: marker})()
        with self.assertRaises(InvalidDecision) as caught:
            _text(unknown, "reason", 240)
        self.assertEqual(str(caught.exception), "reason must be a string; received type non-string.")

    def test_empty_and_over_limit_counts_distinguish_rejections_without_content(self):
        with self.assertRaises(InvalidDecision) as empty:
            _text(" \t\n\u2003", "hypothesis.claim", 240)
        self.assertEqual(str(empty.exception), "hypothesis.claim is empty after trimming whitespace; expected 1-240 characters.")
        marker = "PRIVATE_INVALID_PAYLOAD"
        value = " \n" + marker + "x" * (241 - len(marker)) + "\t "
        with self.assertRaises(InvalidDecision) as oversized:
            _text(value, "reason", 240)
        self.assertEqual(str(oversized.exception), "reason has 241 trimmed characters; limit is 240.")
        self.assertNotIn(marker, str(oversized.exception))

    def test_accepted_whitespace_unicode_and_character_bounds_are_preserved(self):
        self.assertEqual(_text(" \nA\t ", "reason", 240), "A")
        self.assertEqual(_text("\u2003" + "界" * 240 + "\u2003", "reason", 240), "界" * 240)
        self.assertEqual(_text("🙂", "hypothesis.id", 32), "🙂")

    def test_diagnostic_repair_remains_atomic_without_recording_invalid_value(self):
        marker = "PRIVATE_INVALID_PAYLOAD"
        malformed = decision("south", experiment="REJECTED_EXPERIMENT")
        malformed["reason"] = marker + "x" * (241 - len(marker))
        valid = decision("finish")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fixture.jsonl"
            game = KeyGateGame()
            client = NebiusClient(Settings("unit-test-key"))
            with SessionLog(path) as log:
                begin_session(log, game, "unit-test-fixtures", schema_version=2, task="explore")
                with patch.object(client, "_request", side_effect=[response(malformed), response(valid)]) as transport:
                    summary = run_exploration(game, client, log, lambda: KeyGateGame())
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            contexts = [json.loads(row["messages"][1]["content"]) for row in rows if row["type"] == "request"]
            self.assertEqual(summary["status"], "finished")
            self.assertEqual(summary["steps"], 0)
            self.assertEqual(summary["hypotheses"], [])
            self.assertEqual(summary["rejected_decisions"], 1)
            self.assertEqual(transport.call_count, 2)
            self.assertEqual(contexts[0]["observation"], contexts[1]["observation"])
            self.assertIsNone(contexts[1]["active_experiment"])
            self.assertEqual(contexts[1]["validation_feedback"], "reason has 241 trimmed characters; limit is 240.")
            self.assertNotIn(marker, path.read_text())
            self.assertIsNone(next(row for row in rows if row["type"] == "completion")["answer"])
            self.assertEqual(replay(path)["actions_replayed"], 0)


if __name__ == "__main__":
    unittest.main()
