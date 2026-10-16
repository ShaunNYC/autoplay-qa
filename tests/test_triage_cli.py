"""Offline CLI end-to-end coverage for bounded Ultra reports and dry runs."""

import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from autoplay_qa.cli import main
from autoplay_qa.config import Settings
from autoplay_qa.nebius import NebiusError
from autoplay_qa.triage import DEFAULT_TRIAGE_MODEL
from tests.triage_fixture import (
    PRIVATE_SENTINELS,
    assessment,
    grid_recording,
    provider_reply,
    write_recording,
)


class TriageCliTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.source = write_recording(self.root / "session.jsonl", grid_recording())
        self.report = self.root / "report.md"
        self.result = self.root / "result.json"
        self.audit = self.root / "audit.jsonl"

    def arguments(self, *extra):
        return [
            "triage", str(self.source),
            "--out", str(self.report),
            "--json-out", str(self.result),
            "--log", str(self.audit),
            *extra,
        ]

    def invoke(self, arguments):
        with patch("sys.stdout", new_callable=io.StringIO) as output, \
                patch("sys.stderr", new_callable=io.StringIO) as errors:
            code = main(arguments)
        return code, output.getvalue(), errors.getvalue()

    def read_result(self):
        return json.loads(self.result.read_text(encoding="utf-8"))

    def assert_private_evidence_not_sent(self, body):
        prompt = json.dumps(body["messages"])
        for sentinel in PRIVATE_SENTINELS:
            self.assertNotIn(sentinel, prompt)

    def test_dry_run_creates_inspectable_reports_without_credentials_or_http(self):
        source_before = self.source.read_bytes()
        with patch("autoplay_qa.cli.Settings.load", side_effect=AssertionError("No credentials for dry-run")) as settings, \
                patch("autoplay_qa.nebius.NebiusClient._request", side_effect=AssertionError("No HTTP for dry-run")) as transport:
            code, output, errors = self.invoke(self.arguments("--dry-run"))

        self.assertEqual(code, 0)
        self.assertEqual(errors, "")
        settings.assert_not_called()
        transport.assert_not_called()
        self.assertEqual(self.source.read_bytes(), source_before)
        result = self.read_result()
        self.assertEqual(result["status"], "dry_run")
        self.assertEqual(result["summary"]["candidate_count"], 1)
        self.assertEqual(result["summary"]["requests"], 0)
        self.assertEqual(result["summary"]["final_pending_issues"], [1])
        self.assertEqual(result["results"][0]["status"], "pending")
        self.assertIsNone(result["results"][0].get("assessment"))
        self.assertEqual(result["source_counts"]["actions"], 5)
        self.assertEqual(result["source_counts"]["resets"], 0)
        self.assertEqual(result["source_counts"]["inference_requests"], 0)
        self.assertEqual(result["source_counts"]["llm"]["game_actions"], 5)
        self.assertEqual(result["source_counts"]["sweep"]["game_actions"], 0)
        self.assertTrue(self.report.exists())
        self.assertTrue(self.audit.exists())
        self.assertIn(str(self.report), output)
        self.assertIn(str(self.result), output)
        self.assertIn(str(self.audit), output)
        audit = [json.loads(line) for line in self.audit.read_text().splitlines()]
        self.assertEqual(audit[0]["type"], "triage_source")
        self.assertEqual(sum(row["type"] == "triage_session" for row in audit), 1)
        self.assertEqual(audit[-1]["type"], "triage_artifacts")

    def test_default_report_destinations_are_adjacent_to_recording(self):
        with patch("autoplay_qa.cli.Settings.load") as settings, \
                patch("autoplay_qa.nebius.NebiusClient._request") as transport:
            code, _, _ = self.invoke([
                "triage", str(self.source), "--dry-run", "--log", str(self.audit),
            ])
        self.assertEqual(code, 0)
        self.assertTrue(self.source.with_suffix(".triage.md").exists())
        self.assertTrue(self.source.with_suffix(".triage.json").exists())
        settings.assert_not_called()
        transport.assert_not_called()

    def test_clean_recording_needs_no_credentials_even_without_dry_run(self):
        write_recording(self.source, grid_recording(defective=False))
        with patch("autoplay_qa.cli.Settings.load", side_effect=AssertionError("Clean run needs no key")) as settings, \
                patch("autoplay_qa.nebius.NebiusClient._request", side_effect=AssertionError("Clean run has no inference")) as transport:
            code, _, errors = self.invoke(self.arguments())
        self.assertEqual(code, 0)
        self.assertEqual(errors, "")
        result = self.read_result()
        self.assertEqual(result["status"], "no_candidates")
        self.assertEqual(result["summary"]["requests"], 0)
        self.assertEqual(result["summary"]["final_pending_issues"], [])
        self.assertEqual(result["results"], [])
        settings.assert_not_called()
        transport.assert_not_called()

    def test_live_command_uses_fake_http_and_overrides_exploration_model(self):
        settings = Settings("offline-cli-key", model="nvidia/Nemotron-3_5-Lightning")
        env_path = self.root / "chosen.env"
        with patch("autoplay_qa.cli.Settings.load", return_value=settings) as load, \
                patch("autoplay_qa.nebius.NebiusClient._request", return_value=provider_reply()) as transport:
            code, _, errors = self.invoke(self.arguments("--env-file", str(env_path)))

        self.assertEqual(code, 0)
        self.assertEqual(errors, "")
        load.assert_called_once_with(env_path)
        self.assertEqual(transport.call_count, 1)
        self.assertEqual(transport.call_args.args[0], "chat/completions")
        body = transport.call_args.args[1]
        self.assertEqual(body["model"], DEFAULT_TRIAGE_MODEL)
        self.assert_private_evidence_not_sent(body)
        result = self.read_result()
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["model"], DEFAULT_TRIAGE_MODEL)
        self.assertEqual(result["summary"]["requests"], 1)
        self.assertEqual(result["results"][0]["assessment"]["verdict"], "bug")

    def test_fake_failure_is_saved_with_no_automatic_retry(self):
        with patch("autoplay_qa.cli.Settings.load", return_value=Settings("offline-cli-key")), \
                patch("autoplay_qa.nebius.NebiusClient._request", side_effect=NebiusError("Offline API failure.")) as transport:
            code, _, errors = self.invoke(self.arguments())
        self.assertEqual(code, 3)
        self.assertEqual(errors, "")
        self.assertEqual(transport.call_count, 1)
        result = self.read_result()
        self.assertEqual(result["results"][0]["status"], "failed")
        self.assertFalse(result["summary"]["usage_known"])
        self.assertGreater(result["summary"]["accounted_cost_usd"], 0)
        self.assertTrue(self.report.exists())
        self.assertIn("failed", self.report.read_text().lower())

    def test_tiny_cost_cap_persists_pending_evidence_without_request(self):
        with patch("autoplay_qa.cli.Settings.load", return_value=Settings("offline-cli-key")), \
                patch("autoplay_qa.nebius.NebiusClient._request") as transport:
            code, _, errors = self.invoke(self.arguments("--max-usd", "0.000000001"))
        self.assertEqual(code, 3)
        self.assertEqual(errors, "")
        transport.assert_not_called()
        result = self.read_result()
        self.assertEqual(result["summary"]["requests"], 0)
        self.assertEqual(result["summary"]["final_pending_issues"], [1])
        self.assertEqual(result["results"][0]["status"], "pending")
        self.assertIn("evidence_packet", result["results"][0])

    def test_invalid_limits_stop_before_credentials_and_artifact_writes(self):
        for flag, value in (
            ("--max-requests", "0"), ("--max-tokens", "0"),
            ("--max-seconds", "nan"), ("--timeout", "-1"),
            ("--max-usd", "inf"), ("--input-price", "-0.01"),
            ("--output-price", "nan"),
        ):
            with self.subTest(flag=flag, value=value), \
                    patch("autoplay_qa.cli.Settings.load") as settings, \
                    patch("autoplay_qa.nebius.NebiusClient._request") as transport, \
                    patch("sys.stderr", new_callable=io.StringIO):
                with self.assertRaises(SystemExit) as raised:
                    main(self.arguments(flag, value))
                self.assertEqual(raised.exception.code, 2)
                settings.assert_not_called()
                transport.assert_not_called()
        self.assertFalse(self.report.exists())
        self.assertFalse(self.result.exists())
        self.assertFalse(self.audit.exists())

    def test_source_output_collision_is_rejected_and_recording_preserved(self):
        source_before = self.source.read_bytes()
        with patch("autoplay_qa.cli.Settings.load") as settings, \
                patch("autoplay_qa.nebius.NebiusClient._request") as transport:
            code, _, errors = self.invoke([
                "triage", str(self.source), "--out", str(self.source),
                "--json-out", str(self.result), "--log", str(self.audit),
            ])
        self.assertEqual(code, 2)
        self.assertIn("different paths", errors)
        self.assertEqual(self.source.read_bytes(), source_before)
        settings.assert_not_called()
        transport.assert_not_called()

    def test_existing_audit_log_is_rejected_before_spending(self):
        self.audit.write_text("Preserved audit\n", encoding="utf-8")
        with patch("autoplay_qa.cli.Settings.load") as settings, \
                patch("autoplay_qa.nebius.NebiusClient._request") as transport:
            code, _, errors = self.invoke(self.arguments())
        self.assertEqual(code, 2)
        self.assertIn("audit log already exists", errors)
        self.assertEqual(self.audit.read_text(), "Preserved audit\n")
        settings.assert_not_called()
        transport.assert_not_called()

    def test_missing_recording_is_an_error_without_inference(self):
        self.source.unlink()
        with patch("autoplay_qa.cli.Settings.load") as settings, \
                patch("autoplay_qa.nebius.NebiusClient._request") as transport:
            code, _, errors = self.invoke(self.arguments())
        self.assertEqual(code, 2)
        self.assertIn("Error:", errors)
        settings.assert_not_called()
        transport.assert_not_called()

    def test_alternate_model_requires_explicit_input_and_output_pricing(self):
        with patch("autoplay_qa.cli.Settings.load") as settings, \
                patch("autoplay_qa.nebius.NebiusClient._request") as transport:
            code, _, errors = self.invoke(self.arguments(
                "--model", "nvidia/Nemotron-other-model", "--input-price", "1",
            ))
        self.assertEqual(code, 2)
        self.assertIn("both", errors)
        settings.assert_not_called()
        transport.assert_not_called()

    def test_alternate_model_and_explicit_pricing_apply_to_fake_request(self):
        model = "nvidia/Nemotron-other-model"
        with patch("autoplay_qa.cli.Settings.load", return_value=Settings("offline-cli-key")), \
                patch("autoplay_qa.nebius.NebiusClient._request", return_value=provider_reply()) as transport:
            code, _, errors = self.invoke(self.arguments(
                "--model", model, "--input-price", "2", "--output-price", "4",
            ))
        self.assertEqual(code, 0)
        self.assertEqual(errors, "")
        self.assertEqual(transport.call_args.args[1]["model"], model)
        self.assertAlmostEqual(self.read_result()["summary"]["known_cost_usd"], 0.0004)

    def test_dry_run_optional_replay_marks_execution_without_judgement(self):
        with patch("autoplay_qa.cli.Settings.load") as settings, \
                patch("autoplay_qa.nebius.NebiusClient._request") as transport:
            code, _, errors = self.invoke(self.arguments("--dry-run", "--verify-repro"))
        self.assertEqual(code, 0)
        self.assertEqual(errors, "")
        settings.assert_not_called()
        transport.assert_not_called()
        result = self.read_result()
        self.assertEqual(result["summary"]["requests"], 0)
        row = result["results"][0]
        self.assertIsNone(row.get("assessment"))
        self.assertTrue(row["evidence_packet"]["reproduction"]["verified"])
        self.assertFalse(row["evidence_packet"]["reproduction"]["minimization_complete"])
        self.assertEqual(row["evidence_packet"]["reproduction"]["minimization_strategy"], "not_requested")
        self.assertIn("not claimed to be minimal", self.report.read_text().lower())

    def test_secret_echo_is_removed_from_json_markdown_stdout_and_audit(self):
        secret = 'offline-"cli\\escaped-key'
        encoded_secret = json.dumps(secret)[1:-1]
        reply = provider_reply(assessment(rationale=f"Echo {secret} then {encoded_secret}"))
        with patch("autoplay_qa.cli.Settings.load", return_value=Settings(secret)), \
                patch("autoplay_qa.nebius.NebiusClient._request", return_value=reply):
            code, output, errors = self.invoke(self.arguments())
        self.assertEqual(code, 0)
        self.assertEqual(errors, "")
        for value in (
            self.result.read_text(), self.report.read_text(),
            self.audit.read_text(), output,
        ):
            self.assertNotIn(secret, value)
            self.assertNotIn(encoded_secret, value)
        self.assertIn("[REDACTED]", self.report.read_text())
        self.assertIn("[REDACTED]", self.audit.read_text())


if __name__ == "__main__":
    unittest.main()
