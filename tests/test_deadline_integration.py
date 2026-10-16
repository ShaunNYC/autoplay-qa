"""Deadline admission and typed failures across the real exploration loop."""

from types import SimpleNamespace
import unittest
from unittest.mock import patch

from autoplay_qa.deadline_transport import RequestAdmissionDeclined
from autoplay_qa.nebius import NebiusClient, NebiusError, TransientNebiusError
from autoplay_qa.session import replay
from tests import test_investigation_integration as integration_tests
from tests.test_lean_exploration import decision, response


class DeadlineLoopTests(unittest.TestCase):
    fixture = integration_tests.InvestigationIntegrationTests.fixture

    def test_time_spent_after_admission_does_not_become_an_inference_attempt(self):
        clock = SimpleNamespace(now=0)
        clock.monotonic = lambda: clock.now

        class ReservedClient(NebiusClient):
            calls = 0

            def set_session_deadline(self, deadline):
                self.session_deadline = deadline

            def complete(self, *args, **kwargs):
                self.calls += 1
                if self.calls == 2:
                    # Simulate durable reservation consuming the last window.
                    clock.now = 265
                    raise RequestAdmissionDeclined(self.session_deadline - clock.now, self.timeout)
                return super().complete(*args, **kwargs)

        with patch("autoplay_qa.lean.time", clock), patch.object(integration_tests, "NebiusClient", ReservedClient):
            result, path, rows, http = self.fixture([response(decision("east"))], version="adaptive-v4")
        self.assertEqual((result["steps"], result["inference_requests"], http.call_count), (1, 1, 1))
        self.assertEqual(result["direct_fallbacks"], 0)
        self.assertTrue(result["usage_complete"])
        self.assertTrue(result["request_admission_stopped"])
        self.assertIsNone(result["unassessed_final_outcome"])
        self.assertEqual([row["request_number"] for row in rows if row["type"] == "request"], [1])
        stop = next(row for row in rows if row["type"] == "request_admission_stop")
        self.assertEqual((stop["remaining_seconds"], stop["required_seconds"]), (35, 46))
        request = next(row for row in rows if row["type"] == "request")
        self.assertEqual(request["actual_timeout_seconds"], 45)
        self.assertEqual(request["remaining_seconds_before_client_call"], 300)
        self.assertTrue(replay(path)["adaptive_facts_validated"])
        # Only the accepted control may become factual experience.
        self.assertEqual(result["adaptive"]["executed_controls"], 1)

    def test_typed_timeout_is_logged_and_never_retried(self):
        class TimedOutClient(NebiusClient):
            def set_session_deadline(self, deadline):
                self.session_deadline = deadline

            def complete(self, *args, **kwargs):
                error = TransientNebiusError("Safe timeout message", reason="timeout")
                error.transport = "isolated-deadline-v1"
                error.request_timeout_seconds = 45
                error.transport_elapsed_ms = 45000
                raise error

        with patch.object(integration_tests, "NebiusClient", TimedOutClient):
            result, path, rows, http = self.fixture([], version="adaptive-v4")
        self.assertEqual((result["steps"], result["inference_requests"], result["retries"]), (0, 1, 0))
        self.assertEqual(result["status"], "provider_unavailable")
        self.assertFalse(result["usage_complete"])
        self.assertEqual(http.call_count, 0)
        request = next(row for row in rows if row["type"] == "request")
        self.assertEqual(request["failure_reason"], "timeout")
        self.assertTrue(request["possibly_billed"])
        self.assertEqual(request["transport_elapsed_ms"], 45000)
        self.assertTrue(replay(path)["recording_complete"])
        self.assertFalse(any(row["type"] in {"step", "reset", "finish"} for row in rows))

    def test_whole_session_expiring_in_bookkeeping_still_records_no_dispatch(self):
        clock = SimpleNamespace(now=0)
        clock.monotonic = lambda: clock.now

        class ExpiredClient(NebiusClient):
            def set_session_deadline(self, deadline):
                self.session_deadline = deadline

            def complete(self, *args, **kwargs):
                clock.now = 301
                raise RequestAdmissionDeclined(self.session_deadline - clock.now, self.timeout)

        with patch("autoplay_qa.lean.time", clock), patch.object(integration_tests, "NebiusClient", ExpiredClient):
            result, path, rows, http = self.fixture([], version="adaptive-v4")
        self.assertEqual((result["steps"], result["inference_requests"], http.call_count), (0, 0, 0))
        self.assertEqual(result["status"], "time_limit")
        self.assertFalse(result.get("request_admission_stopped", False))
        self.assertTrue(any(row["type"] == "unused_context" for row in rows))
        self.assertFalse(any(row["type"] in {"request", "request_admission_stop"} for row in rows))
        self.assertTrue(replay(path)["recording_complete"])

    def test_postdispatch_protocol_failure_keeps_possible_billing_explicit(self):
        class BrokenWorkerClient(NebiusClient):
            def set_session_deadline(self, deadline):
                pass

            def complete(self, *args, **kwargs):
                error = NebiusError("The isolated inference worker returned unusable private IPC.")
                error.possibly_billed = True
                error.transport = "isolated-deadline-v1"
                raise error

        with patch.object(integration_tests, "NebiusClient", BrokenWorkerClient):
            result, path, rows, _ = self.fixture([], version="adaptive-v4")
        request = next(row for row in rows if row["type"] == "request")
        self.assertTrue(request["possibly_billed"])
        self.assertEqual((result["inference_requests"], result["retries"], result["steps"]), (1, 0, 0))
        self.assertTrue(replay(path)["recording_complete"])


if __name__ == "__main__":
    unittest.main()
