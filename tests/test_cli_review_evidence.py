"""Evidence inspection stays local and keeps ordinary review compatible."""

import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from autoplay_qa.cli import begin_session, main
from autoplay_qa.game import KeyGateGame
from autoplay_qa.review import review_session
from autoplay_qa.session import SessionLog


class EvidenceReviewCliTests(unittest.TestCase):
    def test_evidence_flag_reads_legacy_record_without_credentials_or_inference(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "legacy.jsonl"
            game = KeyGateGame()
            with SessionLog(path) as log:
                begin_session(log, game, "human")
                log.write("summary", status="finished", steps=0, final_observation=game.observe())
            ordinary = review_session(path)
            with patch("autoplay_qa.cli.Settings.load", side_effect=AssertionError("Review must not load credentials")) as load, \
                    patch("autoplay_qa.nebius.NebiusClient._request", side_effect=AssertionError("Review must not call inference")) as transport:
                with patch("sys.stdout", new_callable=io.StringIO) as output:
                    self.assertEqual(main(["review", str(path)]), 0)
                self.assertEqual(json.loads(output.getvalue()), ordinary)
                self.assertNotIn("hypothesis_evidence", ordinary)
                with patch("sys.stdout", new_callable=io.StringIO) as output:
                    self.assertEqual(main(["review", str(path), "--evidence"]), 0)
                detailed = json.loads(output.getvalue())
                self.assertFalse(detailed["hypothesis_evidence"]["available"])
                self.assertEqual({key: value for key, value in detailed.items() if key != "hypothesis_evidence"}, ordinary)
                load.assert_not_called()
                transport.assert_not_called()

    def test_evidence_flag_keeps_unreadable_record_an_error(self):
        with tempfile.TemporaryDirectory() as directory:
            missing = Path(directory) / "missing.jsonl"
            with patch("autoplay_qa.cli.Settings.load") as load, \
                    patch("autoplay_qa.nebius.NebiusClient._request") as transport, \
                    patch("sys.stdout", new_callable=io.StringIO) as output, \
                    patch("sys.stderr", new_callable=io.StringIO) as errors:
                self.assertEqual(main(["review", str(missing), "--evidence"]), 2)
            self.assertEqual(output.getvalue(), "")
            self.assertIn("Error:", errors.getvalue())
            load.assert_not_called()
            transport.assert_not_called()


if __name__ == "__main__":
    unittest.main()
