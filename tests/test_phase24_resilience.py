"""Offline coverage for lean-v1 provider resilience and latency adaptation."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from autoplay_qa.cli import begin_session
from autoplay_qa.config import Settings
from autoplay_qa.exploration import ExplorationLimits
from autoplay_qa.game import KeyGateGame
from autoplay_qa.lean import (
    DEFAULT_TRANSIENT_RETRY_BACKOFF_SECONDS,
    LEAN_PROMPT_VERSION,
    run_lean_exploration,
)
from autoplay_qa.nebius import NebiusClient, TransientNebiusError
from autoplay_qa.reporting import evaluate_recording, generate_report
from autoplay_qa.review import review_session
from autoplay_qa.session import SessionLog, replay


def completion(action="finish"):
    value = {
        "action": action,
        "reason": "Phase 2.4 fixture",
        "expected_position": None,
        "notes": [],
    }
    return {
        "id": "phase24-fixture",
        "model": "nvidia/unit-test-nemotron",
        "choices": [{"message": {"content": json.dumps(value)}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3},
    }


class LeanProviderResilienceTests(unittest.TestCase):
    def run_fixture(self, replies, *, limits=None, timeout=45):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        path = Path(directory.name) / "phase24.jsonl"
        game = KeyGateGame(0)
        client = NebiusClient(
            Settings("unit-test-key"), timeout=timeout, temperature=1.0, top_p=0.95,
        )
        sleeps = []
        with SessionLog(path, ("unit-test-key",)) as log:
            begin_session(
                log,
                game,
                "phase24-fixture",
                schema_version=2,
                task="explore",
                focus="mechanics",
                prompt_version=LEAN_PROMPT_VERSION,
            )
            with patch.object(client, "_request", side_effect=replies) as transport:
                summary = run_lean_exploration(
                    game,
                    client,
                    log,
                    game.reset,
                    limits or ExplorationLimits(max_actions=2, max_requests=4, max_seconds=120),
                    sleep=sleeps.append,
                )
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        return summary, rows, path, transport, sleeps

    def test_timeout_retries_once_in_direct_mode_then_succeeds(self):
        timeout = TransientNebiusError(
            "fixture timeout; may be billed", reason="timeout",
        )
        summary, rows, path, transport, sleeps = self.run_fixture([
            timeout,
            completion(),
        ])

        self.assertEqual(summary["status"], "finished")
        self.assertEqual(summary["inference_requests"], 2)
        self.assertEqual(summary["completed_inference_responses"], 1)
        self.assertEqual(summary["retries"], 1)
        self.assertEqual(summary["direct_fallbacks"], 1)
        self.assertFalse(summary["usage_complete"])
        self.assertNotIn("error", summary)
        self.assertEqual(sleeps, [DEFAULT_TRANSIENT_RETRY_BACKOFF_SECONDS])
        self.assertEqual(transport.call_count, 2)

        requests = [row for row in rows if row["type"] == "request"]
        self.assertEqual(
            [row["decision_mode"] for row in requests],
            ["thinking", "direct_retry"],
        )
        self.assertEqual([row["retry_of"] for row in requests], [None, 1])
        self.assertEqual([row["reason"] for row in requests], ["timeout", "timeout"])
        self.assertEqual([row["attempt_status"] for row in requests], ["failed", "completed"])
        self.assertEqual([row["possibly_billed"] for row in requests], [True, False])
        bodies = [call.args[1] for call in transport.call_args_list]
        self.assertEqual(
            [(body["chat_template_kwargs"]["enable_thinking"], body["max_tokens"]) for body in bodies],
            [(True, 8192), (False, 1536)],
        )
        self.assertTrue(replay(path)["lean_facts_validated"])

    def test_two_transient_failures_end_provider_unavailable_with_complete_artifacts(self):
        summary, rows, path, transport, sleeps = self.run_fixture([
            TransientNebiusError("fixture timeout", reason="timeout"),
            TransientNebiusError("fixture reset", reason="connection_reset"),
            completion(),
        ])

        self.assertEqual(summary["status"], "provider_unavailable")
        self.assertEqual(summary["inference_requests"], 2)
        self.assertEqual(summary["completed_inference_responses"], 0)
        self.assertEqual(summary["retries"], 1)
        self.assertEqual(transport.call_count, 2)
        self.assertEqual(sleeps, [DEFAULT_TRANSIENT_RETRY_BACKOFF_SECONDS])
        self.assertEqual(rows[-1]["type"], "summary")
        self.assertEqual(rows[-1]["status"], "provider_unavailable")
        requests = [row for row in rows if row["type"] == "request"]
        self.assertEqual([row["reason"] for row in requests], ["timeout", "connection_reset"])
        self.assertEqual([row["possibly_billed"] for row in requests], [True, False])

        replayed = replay(path)
        self.assertTrue(replayed["recording_complete"])
        self.assertEqual(replayed["exploration_status"], "provider_unavailable")
        reviewed = review_session(path)
        self.assertTrue(reviewed["recording_complete"])
        self.assertEqual(reviewed["status"], "provider_unavailable")
        self.assertEqual(reviewed["requests"]["without_completion"], 2)
        self.assertIn("AutoPlay QA bug report", generate_report(path))
        self.assertEqual(evaluate_recording(path)["detection"], "clean")

    def test_rate_limit_respects_retry_after_without_real_sleep(self):
        summary, rows, _, _, sleeps = self.run_fixture([
            TransientNebiusError(
                "fixture rate limit", reason="rate_limited", retry_after_seconds=3.25,
            ),
            completion(),
        ])

        self.assertEqual(summary["status"], "finished")
        self.assertEqual(summary["retries"], 1)
        self.assertEqual(sleeps, [3.25])
        requests = [row for row in rows if row["type"] == "request"]
        self.assertEqual([row["reason"] for row in requests], ["rate_limited", "rate_limited"])
        self.assertFalse(any(row["possibly_billed"] for row in requests))

    def test_http_5xx_gets_one_short_direct_retry(self):
        summary, rows, _, transport, sleeps = self.run_fixture([
            TransientNebiusError("fixture 503", reason="http_5xx"),
            completion(),
        ])

        self.assertEqual(summary["status"], "finished")
        self.assertEqual(summary["retries"], 1)
        self.assertEqual(transport.call_count, 2)
        self.assertEqual(sleeps, [DEFAULT_TRANSIENT_RETRY_BACKOFF_SECONDS])
        requests = [row for row in rows if row["type"] == "request"]
        self.assertEqual(requests[0]["reason"], "http_5xx")
        self.assertEqual(requests[1]["decision_mode"], "direct_retry")

    def test_two_slow_thinking_completions_latch_direct_latency_mode(self):
        clock = [0.0]
        replies = iter([completion("wait"), completion("wait"), completion("wait")])

        def timed_transport(*_args):
            value = next(replies)
            if clock[0] < 56:
                clock[0] += 28
            return value

        with patch("time.monotonic", side_effect=lambda: clock[0]):
            summary, rows, path, transport, sleeps = self.run_fixture(
                timed_transport,
                limits=ExplorationLimits(max_actions=3, max_requests=4, max_seconds=200),
                timeout=45,
            )

        self.assertEqual(summary["status"], "action_limit")
        self.assertEqual(summary["retries"], 0)
        self.assertEqual(sleeps, [])
        self.assertEqual(transport.call_count, 3)
        requests = [row for row in rows if row["type"] == "request"]
        self.assertEqual(
            [row["decision_mode"] for row in requests],
            ["thinking", "thinking", "direct_latency"],
        )
        self.assertEqual(requests[2]["decision_mode_reason"], "latency")
        bodies = [call.args[1] for call in transport.call_args_list]
        self.assertEqual(
            [body["chat_template_kwargs"]["enable_thinking"] for body in bodies],
            [True, True, False],
        )
        self.assertTrue(replay(path)["lean_facts_validated"])


if __name__ == "__main__":
    unittest.main()
