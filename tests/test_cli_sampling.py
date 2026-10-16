"""Supported sampling overrides apply only to exploration and remain recorded."""

import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from autoplay_qa.cli import main
from autoplay_qa.config import Settings
from tests.test_exploration import decision, response
from tests.test_integration import fixture_response


class ExplorationSamplingCliTests(unittest.TestCase):
    def test_default_and_custom_temperature_reach_request_and_recording(self):
        for arguments, expected in (([], 1.0), (["--temperature", "0.2"], 0.2),
                                    (["--temperature", "0"], 0.0), (["--temperature", "2"], 2.0)):
            with self.subTest(arguments=arguments), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "fixture.jsonl"
                finished = {**decision("finish"), "experiment_control": "north", "action_role": "finish", "expected_outcome": None}
                finished.pop("prediction")
                finished["plan"] = {"mode": "new", "controls": ["finish"], "revision_reason": "", "evidence_steps": []}
                with patch("autoplay_qa.cli.Settings.load", return_value=Settings("unit-test-key")), \
                        patch("autoplay_qa.nebius.NebiusClient._request", return_value=response(finished)) as transport, \
                        patch("sys.stdout", new_callable=io.StringIO):
                    self.assertEqual(main(["explore", "--log", str(path), *arguments]), 0)
                body = transport.call_args.args[1]
                self.assertEqual(body["temperature"], expected)
                self.assertEqual(body["top_p"], 0.95)
                self.assertEqual(body["max_tokens"], 8192)
                self.assertTrue(body["chat_template_kwargs"]["enable_thinking"])
                self.assertEqual(transport.call_count, 1)
                records = [json.loads(line) for line in path.read_text().splitlines()]
                header = records[0]
                request = next(row for row in records if row["type"] == "request")
                self.assertEqual(header["temperature"], expected)
                self.assertEqual(request["temperature"], expected)
                self.assertEqual((header["top_p"], request["top_p"]), (0.95, 0.95))
                self.assertEqual(header["limits"], {"max_actions": 24, "max_requests": 28,
                                                  "max_seconds": 300, "history_size": 6})

    def test_invalid_temperature_fails_before_loading_credentials_or_inference(self):
        for value in ("-0.01", "2.01", "nan", "NaN", "inf", "-inf", "1e999", "not-a-number"):
            with self.subTest(value=value), patch("autoplay_qa.cli.Settings.load") as load, \
                    patch("autoplay_qa.nebius.NebiusClient._request") as transport, \
                    patch("sys.stderr", new_callable=io.StringIO), self.assertRaises(SystemExit) as error:
                main(["explore", "--temperature=" + value])
            self.assertEqual(error.exception.code, 2)
            load.assert_not_called()
            transport.assert_not_called()

    def test_temperature_flag_does_not_change_the_playthrough_interface_or_sampling(self):
        with patch("autoplay_qa.cli.Settings.load") as load, \
                patch("sys.stderr", new_callable=io.StringIO), self.assertRaises(SystemExit) as error:
            main(["run", "--temperature", "0.8"])
        self.assertEqual(error.exception.code, 2)
        load.assert_not_called()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "baseline.jsonl"
            with patch("autoplay_qa.cli.Settings.load", return_value=Settings("unit-test-key")), \
                    patch("autoplay_qa.nebius.NebiusClient._request", return_value=fixture_response()) as transport, \
                    patch("sys.stdout", new_callable=io.StringIO):
                self.assertEqual(main(["run", "--max-steps", "1", "--log", str(path)]), 3)
            body = transport.call_args.args[1]
            self.assertEqual(body["temperature"], 0.2)
            self.assertNotIn("top_p", body)
            self.assertEqual(body["max_tokens"], 8192)
            self.assertTrue(body["chat_template_kwargs"]["enable_thinking"])


if __name__ == "__main__":
    unittest.main()
