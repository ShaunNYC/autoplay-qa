"""Offline deadline/IPC/process tests; fake children never call a provider."""

import io
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import unittest
from unittest.mock import Mock, patch

from autoplay_qa.config import Settings
from autoplay_qa.deadline_transport import (
    DEADLINE_TRANSPORT_VERSION, DeadlineNebiusClient, MAX_WORKER_OUTPUT_BYTES,
    RequestAdmissionDeclined, _failure_packet, _worker_main, _worker_result,
)
from autoplay_qa.nebius import Completion, NebiusClient, NebiusError, TransientNebiusError


_SUCCESS_PROGRAM = '''
from autoplay_qa.deadline_transport import _worker_main
from autoplay_qa.nebius import Completion
class FakeClient:
    def __init__(self, settings, **kwargs):
        self.settings = settings
    def complete(self, messages, **kwargs):
        return Completion('{"action":"wait","reason":"Observe"}', {
            "finish_reason":"stop", "usage":{"prompt_tokens":2,"completion_tokens":3,"total_tokens":5},
            "latency_ms":1, "raw_content_characters":38,
        })
raise SystemExit(_worker_main(client_factory=FakeClient))
'''


class FakeChildClient(DeadlineNebiusClient):
    def __init__(self, program, **kwargs):
        super().__init__(Settings("fake-private-key-123"), **kwargs)
        self.program = program
        self.children = []

    def _spawn_worker(self):
        process = subprocess.Popen([sys.executable, "-c", self.program], stdin=subprocess.PIPE,
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            cwd=str(Path(__file__).resolve().parents[1]), start_new_session=(os.name == "posix"))
        self.children.append(process)
        return process


def worker_payload(*, deadline=100.0):
    return {"version": DEADLINE_TRANSPORT_VERSION, "nonce": "a" * 32, "deadline": deadline,
        "settings": {"api_key": "fake-private-key-123", "base_url": "https://api.tokenfactory.nebius.com/v1/",
                     "model": "nvidia/unit-test-nemotron", "response_format": "json_schema", "enable_thinking": True},
        "client": {"timeout": 45, "max_tokens": 8192, "temperature": 1.0, "top_p": 0.95},
        "request": {"messages": [{"role": "user", "content": "Public observation"}],
                    "response_schema": {"type": "object"}, "enable_thinking": True, "max_tokens": 8192}}


class DeadlineWorkerTests(unittest.TestCase):
    def test_worker_passes_full_client_contract_exactly_once(self):
        fake = Mock()
        fake.complete.return_value = Completion('{"action":"wait"}', {"finish_reason": "stop", "usage": {"total_tokens": 3}})
        factory = Mock(return_value=fake)
        result = _worker_result(worker_payload(), client_factory=factory, clock=lambda: 1.0)
        self.assertEqual(result["status"], "completed")
        self.assertEqual(factory.call_count, 1)
        self.assertEqual(fake.complete.call_count, 1)
        kwargs = fake.complete.call_args.kwargs
        self.assertEqual((kwargs["timeout"], kwargs["max_tokens"], kwargs["enable_thinking"]), (45, 8192, True))
        self.assertEqual(kwargs["response_schema"], {"type": "object"})

    def test_already_expired_worker_input_cannot_dispatch(self):
        factory = Mock()
        result = _worker_result(worker_payload(deadline=1.0), client_factory=factory, clock=lambda: 1.0)
        self.assertEqual(result["error"]["category"], "timeout")
        factory.assert_not_called()

    def test_expiry_during_client_setup_cannot_dispatch(self):
        fake = Mock()
        result = _worker_result(worker_payload(deadline=2.0), client_factory=Mock(return_value=fake), clock=Mock(side_effect=[1.0, 2.0]))
        self.assertEqual(result["error"]["category"], "timeout")
        fake.complete.assert_not_called()

    def test_late_worker_completion_is_unusable_even_when_usage_is_present(self):
        fake = Mock()
        fake.complete.return_value = Completion('{"action":"wait"}', {"finish_reason": "stop", "usage": {"total_tokens": 3}})
        result = _worker_result(worker_payload(deadline=2.0), client_factory=Mock(return_value=fake), clock=Mock(side_effect=[1.0, 1.0, 2.0]))
        self.assertEqual(result["status"], "error")
        self.assertEqual(result["error"]["category"], "timeout")
        self.assertNotIn("completion", result)

    def test_error_protocol_preserves_categories_but_never_exception_text(self):
        failures = [TransientNebiusError("fake-private-key-123 private timeout", reason="timeout"),
                    TransientNebiusError("private reset", reason="connection_reset"),
                    TransientNebiusError("Nebius HTTP 429: private body", reason="rate_limited", retry_after_seconds=2.5),
                    TransientNebiusError("Nebius HTTP 503: private body", reason="http_5xx"),
                    NebiusError("Nebius HTTP 401: fake-private-key-123 provider text"),
                    RuntimeError("fake-private-key-123 /Users/private/project raw thinking")]
        for failure in failures:
            with self.subTest(kind=type(failure).__name__):
                fake = Mock()
                fake.complete.side_effect = failure
                result = _worker_result(worker_payload(), client_factory=Mock(return_value=fake), clock=lambda: 1.0)
                self.assertEqual(result["status"], "error")
                self.assertEqual(fake.complete.call_count, 1)
                encoded = json.dumps(result)
                for private in ("fake-private-key-123", "provider text", "private body", "raw thinking", "/Users/private"):
                    self.assertNotIn(private, encoded)
        self.assertEqual(_failure_packet(failures[2])["retry_after_seconds"], 2.5)
        self.assertEqual(_failure_packet(failures[3])["http_status"], 503)

    def test_truncated_or_non_json_trace_never_leaves_worker_as_content(self):
        for finish, content in (("length", '{"reason":"private unfinished thinking'), ("stop", "<think>private unfinished thinking")):
            with self.subTest(finish=finish):
                fake = Mock()
                fake.complete.return_value = Completion(content, {"finish_reason": finish, "raw_content_characters": len(content),
                    "thinking": "private raw thinking", "returned_model": "fake-private-key-123"})
                result = _worker_result(worker_payload(), client_factory=Mock(return_value=fake), clock=lambda: 1.0)
                self.assertEqual(result["completion"]["content"], "")
                metadata = result["completion"]["metadata"]
                self.assertEqual(metadata["finish_reason"], finish)
                self.assertEqual(metadata["raw_content_characters"], len(content))
                self.assertTrue(metadata["discarded_unusable_content"])
                self.assertNotIn("thinking", metadata)
                self.assertNotIn("fake-private-key-123", json.dumps(result))

    def test_worker_main_emits_only_finite_envelope_even_if_dependency_prints(self):
        class NoisyClient:
            def __init__(self, *_args, **_kwargs):
                print("fake-private-key-123 private setup")
            def complete(self, *_args, **_kwargs):
                print("raw reasoning from dependency", file=sys.stderr)
                raise RuntimeError("fake-private-key-123 private exception")
        target = io.BytesIO()
        _worker_main(io.BytesIO(json.dumps(worker_payload()).encode()), target, client_factory=NoisyClient, clock=lambda: 1.0)
        result = json.loads(target.getvalue())
        self.assertEqual(result["status"], "error")
        self.assertEqual(result["error"]["category"], "worker_failed")
        self.assertNotIn(b"fake-private-key-123", target.getvalue())
        self.assertNotIn(b"raw reasoning", target.getvalue())

    def test_malformed_worker_input_never_constructs_provider_client(self):
        factory = Mock()
        for raw in (b"not-json fake-private-key-123", b"[]", b'{"version":"wrong"}'):
            with self.subTest(raw=raw[:8]):
                target = io.BytesIO()
                _worker_main(io.BytesIO(raw), target, client_factory=factory)
                result = json.loads(target.getvalue())
                self.assertEqual(result["error"]["category"], "worker_failed")
        factory.assert_not_called()

    def test_metadata_cannot_smuggle_raw_text_or_invalid_usage_through_diagnostics(self):
        fake = Mock()
        fake.complete.return_value = Completion('{"action":"wait"}', {
            "completion_id": "private raw reasoning " * 1000, "returned_model": "fake-private-key-123",
            "finish_reason": "stop", "usage": {"prompt_tokens": True, "completion_tokens": -1, "total_tokens": 3},
            "thinking": "untrusted provider trace", "reasoning_tokens": "not-a-count",
        })
        result = _worker_result(worker_payload(), client_factory=Mock(return_value=fake), clock=lambda: 1.0)
        metadata = result["completion"]["metadata"]
        self.assertIsNone(metadata["completion_id"])
        self.assertIsNone(metadata["returned_model"])
        self.assertEqual(metadata["usage"], {"total_tokens": 3})
        self.assertNotIn("thinking", metadata)
        self.assertNotIn("reasoning_tokens", metadata)
        self.assertNotIn("private raw reasoning", json.dumps(result))


class DeadlineParentTests(unittest.TestCase):
    def test_real_worker_entrypoint_rejects_expired_input_without_provider_dispatch(self):
        client = DeadlineNebiusClient(Settings("fake-private-key-123"))
        process = client._spawn_worker()
        try:
            output, _ = process.communicate(input=json.dumps(worker_payload(deadline=time.monotonic() - 1)).encode(), timeout=5)
        finally:
            client._kill_and_reap(process)
        result = json.loads(output)
        self.assertEqual(process.returncode, 0)
        self.assertEqual(result["nonce"], "a" * 32)
        self.assertEqual(result["error"]["category"], "timeout")
        self.assertNotIn(b"fake-private-key-123", output)

    def test_successful_fake_child_retains_completion_contract_and_reaps(self):
        client = FakeChildClient(_SUCCESS_PROGRAM)
        result = client.complete([{"role": "user", "content": "public observation"}], timeout=5,
            response_schema={"type": "object"}, enable_thinking=True, max_tokens=8192)
        self.assertEqual(json.loads(result.content)["action"], "wait")
        self.assertEqual(result.metadata["usage"], {"prompt_tokens": 2, "completion_tokens": 3, "total_tokens": 5})
        self.assertEqual(result.metadata["transport"], DEADLINE_TRANSPORT_VERSION)
        self.assertEqual(result.metadata["request_timeout_seconds"], 5)
        self.assertGreaterEqual(result.metadata["latency_ms"], result.metadata["transport_elapsed_ms"])
        self.assertGreaterEqual(result.metadata["cleanup_elapsed_ms"], 0)
        self.assertTrue(result.metadata["cleanup_worker_reaped"])
        self.assertEqual(len(client.children), 1)
        self.assertIsNotNone(client.children[0].poll())

    def test_slow_worker_is_killed_reaped_and_timeout_is_unknown_without_retry(self):
        program = "import sys,time; sys.stdin.buffer.read(); time.sleep(5)"
        client = FakeChildClient(program)
        started = time.monotonic()
        with self.assertRaises(TransientNebiusError) as raised:
            client.complete([], timeout=0.15)
        self.assertEqual(raised.exception.reason, "timeout")
        self.assertTrue(raised.exception.possibly_billed)
        self.assertEqual(raised.exception.request_timeout_seconds, 0.15)
        self.assertLess(time.monotonic() - started, 2.0)
        self.assertEqual(len(client.children), 1)
        self.assertIsNotNone(client.children[0].poll())
        self.assertNotIn("fake-private-key-123", str(raised.exception))

    def test_slow_partial_output_cannot_extend_deadline_by_trickling(self):
        program = '''
import sys,time
sys.stdin.buffer.read()
for character in '{"version":"slow-private-output":':
    sys.stdout.write(character); sys.stdout.flush(); time.sleep(0.03)
'''
        client = FakeChildClient(program)
        started = time.monotonic()
        with self.assertRaises(TransientNebiusError) as raised:
            client.complete([], timeout=0.18)
        self.assertEqual(raised.exception.reason, "timeout")
        self.assertLess(time.monotonic() - started, 2.0)
        self.assertIsNotNone(client.children[0].poll())
        self.assertNotIn("slow-private-output", str(raised.exception))

    def test_truncated_malformed_or_failed_worker_does_not_leak_output(self):
        for program in ("import sys;sys.stdin.buffer.read();print('fake-private-key-123 raw provider error')",
                        "import sys;sys.stdin.buffer.read();print('{');sys.exit(2)"):
            with self.subTest(program=program[-8:]):
                client = FakeChildClient(program)
                with self.assertRaises(NebiusError) as raised:
                    client.complete([], timeout=5)
                self.assertNotIn("fake-private-key-123", str(raised.exception))
                self.assertNotIn("raw provider", str(raised.exception))
                self.assertTrue(raised.exception.possibly_billed)
                self.assertTrue(raised.exception.cleanup_worker_reaped)
                self.assertEqual(len(client.children), 1)
                self.assertIsNotNone(client.children[0].poll())

    def test_oversized_worker_output_is_rejected(self):
        program = f"import sys;sys.stdin.buffer.read();sys.stdout.write('X'*{MAX_WORKER_OUTPUT_BYTES + 1})"
        client = FakeChildClient(program)
        with self.assertRaises(NebiusError) as raised:
            client.complete([], timeout=5)
        self.assertIn("private IPC", str(raised.exception))
        self.assertIsNotNone(client.children[0].poll())

    def test_deeply_nested_malformed_ipc_is_safe_protocol_error(self):
        program = "import sys;sys.stdin.buffer.read();sys.stdout.write('['*2000+'0'+']'*2000)"
        client = FakeChildClient(program)
        with self.assertRaises(NebiusError) as raised:
            client.complete([], timeout=5)
        self.assertIn("private IPC", str(raised.exception))
        self.assertIsNotNone(client.children[0].poll())

    def test_valid_but_late_private_result_is_rejected_by_parent(self):
        clock = [1.0]
        class LateProcess:
            returncode = 0
            stdin = stdout = stderr = None
            def communicate(self, input=None, timeout=None):
                if input is not None:
                    self.packet = json.loads(input)
                    clock[0] = 2.1
                return json.dumps({"version": DEADLINE_TRANSPORT_VERSION, "nonce": self.packet["nonce"], "status": "completed",
                    "completion": {"content": '{"action":"wait"}', "metadata": {"finish_reason": "stop", "usage": {"total_tokens": 3}}}}).encode(), None
            def poll(self):
                return 0
        client = DeadlineNebiusClient(Settings("fake-private-key-123"))
        with patch("autoplay_qa.deadline_transport.time.monotonic", side_effect=lambda: clock[0]), patch.object(client, "_spawn_worker", return_value=LateProcess()):
            with self.assertRaises(TransientNebiusError) as raised:
                client.complete([], timeout=1)
        self.assertEqual(raised.exception.reason, "timeout")
        self.assertTrue(raised.exception.possibly_billed)

    def test_response_deadline_check_and_measurement_share_one_timestamp(self):
        from autoplay_qa import deadline_transport
        clock = [0.0]
        def monotonic():
            sampled = clock[0]
            if sampled == .99:
                # Simulate descheduling immediately after the deadline check.
                clock[0] = 1.001
            return sampled
        class FinishedProcess:
            returncode = 0
            stdin = stdout = stderr = None
            def communicate(self, input=None, timeout=None):
                if input is not None:
                    self.packet = json.loads(input)
                    clock[0] = .9
                return json.dumps({"version": DEADLINE_TRANSPORT_VERSION, "nonce": self.packet["nonce"], "status": "completed",
                    "completion": {"content": '{"action":"wait"}', "metadata": {"finish_reason": "stop", "usage": {"total_tokens": 3}}}}).encode(), None
            def poll(self):
                return 0
        safe_completion = deadline_transport._safe_completion
        def parsed(*args):
            safe = safe_completion(*args)
            clock[0] = .99
            return safe
        client = DeadlineNebiusClient(Settings("fake-private-key-123"))
        with patch("autoplay_qa.deadline_transport.time.monotonic", side_effect=monotonic), \
                patch.object(client, "_spawn_worker", return_value=FinishedProcess()), \
                patch("autoplay_qa.deadline_transport._safe_completion", side_effect=parsed):
            completion = client.complete([], timeout=1)
        self.assertEqual(completion.metadata["transport_elapsed_ms"], 990)
        self.assertEqual(completion.metadata["latency_ms"], 1001)
        self.assertTrue(completion.metadata["cleanup_worker_reaped"])

    def test_interrupt_kills_and_reaps_worker_before_propagating(self):
        client = FakeChildClient("import sys,time;sys.stdin.buffer.read();time.sleep(5)")
        original_spawn = client._spawn_worker
        def interrupting_spawn():
            process = original_spawn()
            communicate = process.communicate
            invoked = [False]
            def interrupted(*args, **kwargs):
                if not invoked[0]:
                    invoked[0] = True
                    raise KeyboardInterrupt
                return communicate(*args, **kwargs)
            process.communicate = interrupted
            return process
        with patch.object(client, "_spawn_worker", side_effect=interrupting_spawn):
            with self.assertRaises(KeyboardInterrupt):
                client.complete([], timeout=5)
        self.assertEqual(len(client.children), 1)
        self.assertIsNotNone(client.children[0].poll())

    def test_credentials_and_prompts_are_not_process_arguments_or_environment(self):
        client = DeadlineNebiusClient(Settings("fake-private-key-123"))
        with patch.dict(os.environ, {"NEBIUS_API_KEY": "fake-private-key-123"}), patch("autoplay_qa.deadline_transport.subprocess.Popen") as spawn:
            client._spawn_worker()
        args, kwargs = spawn.call_args
        self.assertNotIn("fake-private-key-123", json.dumps(args))
        self.assertNotIn("NEBIUS_API_KEY", kwargs["env"])
        self.assertEqual(kwargs["stdin"], subprocess.PIPE)
        self.assertEqual(kwargs["stdout"], subprocess.PIPE)
        self.assertEqual(kwargs["stderr"], subprocess.DEVNULL)
        self.assertFalse(kwargs.get("shell", False))

    def test_safe_error_packets_reconstruct_original_categories_without_retry(self):
        for reason, status, retry in (("timeout", None, None), ("connection_reset", None, None),
                                      ("rate_limited", 429, 2.5), ("http_5xx", 503, None)):
            with self.subTest(reason=reason):
                program = """
import json,sys
p=json.load(sys.stdin)
result={"version":p["version"],"nonce":p["nonce"],"status":"error","error":%s}
sys.stdout.write(json.dumps(result))
""" % repr({"category": reason, "http_status": status, "retry_after_seconds": retry})
                client = FakeChildClient(program)
                with self.assertRaises(TransientNebiusError) as raised:
                    client.complete([], timeout=5)
                self.assertEqual(raised.exception.reason, reason)
                self.assertEqual(raised.exception.retry_after_seconds, retry)
                self.assertEqual(raised.exception.possibly_billed, reason == "timeout")
                self.assertEqual(len(client.children), 1)
                if status:
                    self.assertIn(str(status), str(raised.exception))


class DeadlineAdmissionTests(unittest.TestCase):
    def test_session_recheck_after_bookkeeping_declines_before_spawn(self):
        client = DeadlineNebiusClient(Settings("fake-private-key-123"))
        client.set_session_deadline(120.0)
        with patch("autoplay_qa.deadline_transport.time.monotonic", return_value=100.0), patch.object(client, "_spawn_worker") as spawn:
            with self.assertRaises(RequestAdmissionDeclined) as raised:
                client.complete([])
        spawn.assert_not_called()
        self.assertEqual(raised.exception.remaining_seconds, 20)
        self.assertEqual(raised.exception.request_timeout, 45)
        self.assertEqual(raised.exception.required_seconds, 46)
        self.assertTrue(raised.exception.no_provider_dispatched)
        self.assertFalse(raised.exception.possibly_billed)

    def test_cleanup_communicate_and_fallback_share_one_total_window(self):
        clock = [0.0]
        waits = []
        class PartialCleanup:
            pid = 123456
            stdin = stdout = stderr = None
            returncode = None
            def poll(self):
                return self.returncode
            def kill(self):
                pass
            def communicate(self, timeout):
                self.communicate_timeout = timeout
                clock[0] += .75
                raise subprocess.TimeoutExpired("fake-worker", timeout)
            def wait(self, timeout):
                waits.append(timeout)
                clock[0] += timeout
                self.returncode = -9
        process = PartialCleanup()
        with patch("autoplay_qa.deadline_transport.time.monotonic", side_effect=lambda: clock[0]), patch("autoplay_qa.deadline_transport.os.killpg"):
            self.assertTrue(DeadlineNebiusClient._kill_and_reap(process))
        self.assertEqual(process.communicate_timeout, 1.0)
        self.assertEqual(waits, [.25])
        self.assertLessEqual(clock[0], 1.0)

    def test_full_window_boundary_is_admitted_and_timeout_override_is_honored(self):
        client = FakeChildClient(_SUCCESS_PROGRAM)
        client.set_session_deadline(time.monotonic() + 10)
        result = client.complete([], timeout=5)
        self.assertEqual(result.metadata["request_timeout_seconds"], 5)
        self.assertEqual(len(client.children), 1)

    def test_invalid_timeout_values_are_rejected_without_worker(self):
        for value in (0, -1, True, float("inf"), float("nan")):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    DeadlineNebiusClient(Settings("fake-key"), timeout=value)
        client = DeadlineNebiusClient(Settings("fake-key"))
        with patch.object(client, "_spawn_worker") as spawn:
            for value in (0, -1, True, float("inf"), float("nan")):
                with self.assertRaises(ValueError):
                    client.complete([], timeout=value)
            for value in (True, float("nan"), float("inf")):
                with self.assertRaises(ValueError):
                    client.set_session_deadline(value)
        spawn.assert_not_called()

    def test_legacy_client_transport_is_unchanged(self):
        client = NebiusClient(Settings("fake-key"))
        with patch.object(client._opener, "open", side_effect=TimeoutError), patch("autoplay_qa.deadline_transport.subprocess.Popen") as spawn:
            with self.assertRaises(TransientNebiusError) as raised:
                client.complete([])
        self.assertEqual(raised.exception.reason, "timeout")
        spawn.assert_not_called()


if __name__ == "__main__":
    unittest.main()
