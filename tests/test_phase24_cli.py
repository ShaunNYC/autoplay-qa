"""Stub-only coverage for the opt-in Phase 2.4 inference preflight."""

import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from autoplay_qa.cli import PREFLIGHT_MAX_TOKENS, main
from autoplay_qa.config import Settings
from autoplay_qa.nebius import NebiusError


MODEL = "nvidia/Nemotron-3_5-Lightning"


def catalog_response():
    return {"data": [{"id": MODEL}]}


def completion_response(content, *, finish_reason="stop"):
    return {
        "id": "phase24-cli-stub",
        "model": MODEL,
        "choices": [{
            "message": {"content": json.dumps(content)},
            "finish_reason": finish_reason,
        }],
        "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8},
    }


class InferenceProbeCliTests(unittest.TestCase):
    def test_doctor_probe_is_opt_in_and_makes_one_tiny_direct_completion(self):
        settings = Settings("secret-unit-key")

        def transport(path, body=None, timeout=None):
            if path == "models":
                return catalog_response()
            self.assertEqual(path, "chat/completions")
            return completion_response({"ok": True})

        with patch("autoplay_qa.cli.Settings.load", return_value=settings), \
                patch("autoplay_qa.nebius.NebiusClient._request", side_effect=transport) as request, \
                patch("sys.stdout", new_callable=io.StringIO) as output:
            self.assertEqual(main(["doctor"]), 0)
            self.assertEqual(
                [call.args[0] for call in request.call_args_list],
                ["models"],
            )
            self.assertNotIn("secret-unit-key", output.getvalue())

        with patch("autoplay_qa.cli.Settings.load", return_value=settings), \
                patch("autoplay_qa.nebius.NebiusClient._request", side_effect=transport) as request, \
                patch("sys.stdout", new_callable=io.StringIO) as output:
            self.assertEqual(main(["doctor", "--probe-inference"]), 0)

        paths = [call.args[0] for call in request.call_args_list]
        self.assertEqual(paths.count("chat/completions"), 1)
        probe_body = next(
            call.args[1]
            for call in request.call_args_list
            if call.args[0] == "chat/completions"
        )
        self.assertFalse(probe_body["chat_template_kwargs"]["enable_thinking"])
        self.assertLessEqual(probe_body["max_tokens"], 16)
        self.assertEqual(probe_body["max_tokens"], PREFLIGHT_MAX_TOKENS)
        self.assertIn("latency:", output.getvalue())
        self.assertIn("finish_reason: stop", output.getvalue())
        self.assertNotIn("secret-unit-key", output.getvalue())

    def test_explore_preflight_error_aborts_before_creating_recording(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "must-not-exist.jsonl"
            with patch("autoplay_qa.cli.Settings.load", return_value=Settings("unit-key")), \
                    patch("autoplay_qa.cli.probe_inference", side_effect=NebiusError("stub outage")) as probe, \
                    patch("autoplay_qa.nebius.NebiusClient._request", side_effect=AssertionError("no exploration request")) as transport, \
                    patch("sys.stderr", new_callable=io.StringIO) as errors:
                self.assertEqual(main(["explore", "--preflight", "--log", str(path)]), 4)

            probe.assert_called_once()
            transport.assert_not_called()
            self.assertFalse(path.exists())
            self.assertIn("Preflight failed", errors.getvalue())
            self.assertIn("exploration was not started", errors.getvalue())

    def test_explore_preflight_slow_response_aborts_before_creating_recording(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "must-not-exist.jsonl"
            result = {"latency_ms": 15_001, "finish_reason": "stop"}
            with patch("autoplay_qa.cli.Settings.load", return_value=Settings("unit-key")), \
                    patch("autoplay_qa.cli.probe_inference", return_value=result) as probe, \
                    patch("autoplay_qa.nebius.NebiusClient._request", side_effect=AssertionError("no exploration request")) as transport, \
                    patch("sys.stdout", new_callable=io.StringIO) as output, \
                    patch("sys.stderr", new_callable=io.StringIO) as errors:
                self.assertEqual(main(["explore", "--preflight", "--log", str(path)]), 4)

            probe.assert_called_once()
            transport.assert_not_called()
            self.assertFalse(path.exists())
            self.assertIn("15.001s", output.getvalue())
            self.assertIn("finish_reason stop", output.getvalue())
            self.assertIn("exceeded 15 seconds", errors.getvalue())

    def test_explore_preflight_passes_then_starts_lean_session(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "session.jsonl"
            finished = {
                "action": "finish",
                "reason": "Probe passed; finish fixture",
                "expected_position": None,
                "notes": [],
            }
            with patch("autoplay_qa.cli.Settings.load", return_value=Settings("unit-key")), \
                    patch(
                        "autoplay_qa.cli.probe_inference",
                        return_value={"latency_ms": 15_000, "finish_reason": "stop"},
                    ) as probe, patch(
                        "autoplay_qa.nebius.NebiusClient._request",
                        return_value=completion_response(finished),
                    ) as transport, patch(
                        "sys.stdout", new_callable=io.StringIO,
                    ) as output:
                self.assertEqual(main(["explore", "--preflight", "--log", str(path)]), 0)

            probe.assert_called_once()
            self.assertEqual(transport.call_count, 1)
            self.assertTrue(path.exists())
            self.assertIn("Inference preflight: latency 15.000s", output.getvalue())
            records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(records[0]["prompt_version"], "lean-v1")
            self.assertEqual(records[-1]["status"], "finished")

    def test_legacy_preflight_is_rejected_before_credentials_or_transport(self):
        with patch("autoplay_qa.cli.Settings.load") as load, \
                patch("autoplay_qa.nebius.NebiusClient._request") as transport, \
                patch("sys.stderr", new_callable=io.StringIO) as errors:
            self.assertEqual(main([
                "explore", "--mode", "legacy-v27", "--preflight",
            ]), 2)
        load.assert_not_called()
        transport.assert_not_called()
        self.assertIn("only with --mode lean-v1", errors.getvalue())


if __name__ == "__main__":
    unittest.main()
