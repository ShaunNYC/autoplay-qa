"""Local runner accepts bounded jobs and persists spending before fake HTTP."""

from copy import deepcopy
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

from autoplay_qa.config import Settings
from autoplay_qa.dashboard_runner import (
    BudgetedClient, CostLedger, ReceiptStore, TotalAllowance, execute_job, validate_job,
    recover_interrupted_receipts,
    _Bridge, DashboardTransport,
)
from autoplay_qa.game import KeyGateGame
from autoplay_qa.hybrid import run_collision_sweep
from autoplay_qa.nebius import NebiusClient, NebiusError
from autoplay_qa.triage import TriagePricing
from tests.triage_fixture import PRIVATE_SENTINELS, grid_recording, provider_reply, write_recording


JOB = {
    "id": "fixture-job-001", "requestId": "fixture-request-001", "leaseId": "fixture-lease-001",
    "createdAt": "2026-10-14T16:00:00Z", "game": "grid", "profile": "clean", "seed": 0,
    "maxUsd": 1, "cancelRequested": False,
}
MESSAGES = [{"role": "user", "content": "Only use visible controls."}]
SCHEMA = {"type": "object", "properties": {"action": {"type": "string"}}}


class DashboardTransportTests(unittest.TestCase):
    def test_bridge_identifies_itself_and_authenticates_without_real_http(self):
        token = "OFFLINE_RUNNER_SECRET_" + "x" * 32
        transport = DashboardTransport("https://dashboard.example", token)
        requests = []

        class Response:
            def __enter__(self): return self
            def __exit__(self, *args): return False
            def read(self, limit): return b'{"job":null}'

        def open_request(request, timeout):
            requests.append(request)
            self.assertEqual(timeout, 15)
            return Response()

        with patch.object(transport.opener, "open", side_effect=open_request):
            self.assertEqual(transport.post("claim", {"runnerId": "offline_fixture"}), {"job": None})
        self.assertEqual(len(requests), 1)
        self.assertEqual(requests[0].get_header("User-agent"), "AutoPlayQA-DashboardRunner/0.1")
        self.assertEqual(requests[0].get_header("Authorization"), "Bearer " + token)


class DashboardJobTests(unittest.TestCase):
    def test_only_allowlisted_game_profile_seed_and_budget_are_accepted(self):
        for game in ("grid", "unity"):
            for profile in ("clean", "wall", "floor", "key", "reset", "door"):
                for seed in (0, 1, 5):
                    with self.subTest(game=game, profile=profile, seed=seed):
                        job = {**JOB, "game": game, "profile": profile, "seed": seed}
                        self.assertEqual(validate_job(job), job)

    def test_remote_jobs_cannot_supply_commands_paths_endpoints_or_provider_keys(self):
        for field, value in (
            ("command", "curl https://example.invalid"),
            ("args", ["--max-tokens", "100000"]),
            ("unityProject", "/private/other-user/game"),
            ("path", "../../other-user"),
            ("endpoint", "https://example.invalid/"),
            ("api_key", "PRIVATE_PROVIDER_KEY"),
            ("maxRequests", 1000),
        ):
            with self.subTest(field=field), self.assertRaises(ValueError):
                validate_job({**JOB, field: value})

    def test_bad_job_value_fails_before_mutating_input_or_opening_http(self):
        values = {
            "game": ["shell", "Grid", None],
            "profile": ["../../wall", "floor-gap", None],
            "seed": [True, 2, -1, "0", 0.0],
            "maxUsd": [True, 0, -0.1, 10.01, float("nan"), float("inf"), "1"],
            "cancelRequested": ["false", 0, None],
            "id": ["../../receipt", "", "job\nprivate"],
            "requestId": ["../request", "", None],
            "leaseId": ["../lease", "", None],
        }
        with patch("autoplay_qa.nebius.NebiusClient._request", side_effect=AssertionError("Validation cannot infer")) as transport:
            for field, invalid in values.items():
                with self.subTest(field=field, invalid=invalid):
                    job = {**JOB, field: invalid}
                    original = deepcopy(job)
                    with self.assertRaises(ValueError):
                        validate_job(job)
                    if field != "maxUsd" or not isinstance(invalid, float) or invalid == invalid:
                        self.assertEqual(job, original)
        transport.assert_not_called()

    def test_missing_fields_and_nonobject_jobs_are_rejected(self):
        for field in JOB:
            with self.subTest(field=field), self.assertRaises(ValueError):
                validate_job({key: value for key, value in JOB.items() if key != field})
        for value in (None, [], "job", 123):
            with self.subTest(value=value), self.assertRaises(ValueError):
                validate_job(value)

    def test_exact_newly_authorized_ten_dollar_job_cap_is_supported(self):
        self.assertEqual(validate_job({**JOB, "maxUsd": 10})["maxUsd"], 10)


class DashboardLedgerTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "spending.json"
        self.pricing = TriagePricing()

    def reserve(self, ledger):
        return ledger.reserve_request(
            "nvidia/Nemotron-3-Ultra-550b-a55b", MESSAGES, SCHEMA, 8192, self.pricing,
        )

    def test_open_reservation_survives_restart_as_unknown_usage(self):
        first = CostLedger(self.path, cap_usd=1)
        self.reserve(first)
        original = first.summary()
        recovered = CostLedger(self.path, cap_usd=1).summary()
        self.assertEqual(recovered, original)
        self.assertEqual(recovered["requests"], 1)
        self.assertFalse(recovered["usage_known"])
        self.assertEqual(recovered["unknown_usage_requests"], 1)
        self.assertGreater(recovered["accounted_cost_usd"], 0)
        self.assertEqual(recovered["known_cost_usd"], 0)

    def test_complete_coherent_usage_releases_reservation_to_configured_actual_cost(self):
        ledger = CostLedger(self.path, cap_usd=1)
        identifier = self.reserve(ledger)
        reserved = ledger.summary()["accounted_cost_usd"]
        ledger.settle_request(identifier, {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150})
        known = ledger.summary()
        self.assertLess(known["accounted_cost_usd"], reserved)
        self.assertAlmostEqual(known["accounted_cost_usd"], self.pricing.cost(100, 50), places=8)
        self.assertEqual(known["known_cost_usd"], known["accounted_cost_usd"])
        self.assertTrue(known["usage_known"])
        self.assertEqual(known["unknown_usage_requests"], 0)
        self.assertEqual(CostLedger(self.path, cap_usd=1).summary(), known)

    def test_missing_negative_bool_and_incoherent_usage_keep_full_reservation(self):
        malformed = (
            None, {}, {"total_tokens": 150},
            {"prompt_tokens": -1, "completion_tokens": 50, "total_tokens": 49},
            {"prompt_tokens": True, "completion_tokens": 50, "total_tokens": 51},
            {"prompt_tokens": 100.0, "completion_tokens": 50, "total_tokens": 150},
            {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 140},
            {"prompt_tokens": "100", "completion_tokens": 50, "total_tokens": 150},
        )
        for index, usage in enumerate(malformed):
            with self.subTest(usage=usage):
                ledger = CostLedger(self.path.with_name(f"invalid-{index}.json"), cap_usd=1)
                identifier = self.reserve(ledger)
                before = ledger.summary()["accounted_cost_usd"]
                ledger.settle_request(identifier, usage)
                summary = ledger.summary()
                self.assertEqual(summary["accounted_cost_usd"], before)
                self.assertFalse(summary["usage_known"])
                self.assertEqual(summary["unknown_usage_requests"], 1)

    def test_unknown_usage_reservation_can_exhaust_cap_after_restart(self):
        probe = CostLedger(self.path.with_name("probe.json"), cap_usd=1)
        self.reserve(probe)
        reservation = probe.summary()["accounted_cost_usd"]
        ledger = CostLedger(self.path, cap_usd=reservation + 0.00000001)
        self.reserve(ledger)
        resumed = CostLedger(self.path, cap_usd=reservation + 0.00000001)
        with self.assertRaises((ValueError, NebiusError)):
            self.reserve(resumed)
        self.assertEqual(resumed.summary()["requests"], 1)

    def test_price_or_cap_cannot_make_an_invalid_ledger_free(self):
        for cap in (0, -1, True, float("nan"), float("inf")):
            with self.subTest(cap=cap), self.assertRaises(ValueError):
                CostLedger(self.path, cap_usd=cap)

    def test_settlement_is_idempotent_and_cannot_release_unknown_request(self):
        ledger = CostLedger(self.path, cap_usd=1)
        identifier = self.reserve(ledger)
        usage = {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150}
        ledger.settle_request(identifier, usage)
        once = ledger.summary()
        ledger.settle_request(identifier, usage)
        self.assertEqual(ledger.summary(), once)
        with self.assertRaises((ValueError, KeyError)):
            ledger.settle_request("not-a-request", usage)
        self.assertEqual(ledger.summary(), once)


class DashboardBudgetedClientTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "spending.json"
        self.ledger = CostLedger(self.path, cap_usd=1)
        self.client = NebiusClient(
            Settings("FAKE_PROVIDER_KEY"), timeout=45, max_tokens=8192,
            temperature=1.0, top_p=0.95,
        )
        self.pricing = TriagePricing()
        self.wrapped = BudgetedClient(self.client, self.ledger, self.pricing)

    def test_wrapper_keeps_lightning_settings_and_sampling_defaults(self):
        self.assertEqual(self.wrapped.settings.model, "nvidia/Nemotron-3_5-Lightning")
        self.assertTrue(self.wrapped.settings.enable_thinking)
        self.assertEqual(self.wrapped.timeout, 45)
        self.assertEqual(self.wrapped.max_tokens, 8192)
        self.assertEqual(self.wrapped.temperature, 1.0)
        self.assertEqual(self.wrapped.top_p, 0.95)

    def test_reservation_is_on_disk_before_the_wrapped_fake_http_begins(self):
        seen = []

        def fake_request(_path, _body, *_args, **_kwargs):
            summary = CostLedger(self.path, cap_usd=1).summary()
            seen.append(summary)
            self.assertEqual(summary["requests"], 1)
            self.assertEqual(summary["unknown_usage_requests"], 1)
            self.assertGreater(summary["accounted_cost_usd"], 0)
            return provider_reply()

        with patch.object(self.client, "_request", side_effect=fake_request) as transport:
            self.wrapped.complete(MESSAGES, response_schema=SCHEMA)

        self.assertEqual(len(seen), 1)
        self.assertEqual(transport.call_count, 1)
        self.assertTrue(self.ledger.summary()["usage_known"])
        persisted = self.path.read_text(encoding="utf-8")
        self.assertNotIn("FAKE_PROVIDER_KEY", persisted)
        self.assertNotIn("Only use visible controls.", persisted)

    def test_provider_failure_is_single_attempt_and_keeps_unknown_cost(self):
        with patch.object(self.client, "_request", side_effect=NebiusError("Failure echoes FAKE_PROVIDER_KEY")) as transport:
            with self.assertRaises(NebiusError):
                self.wrapped.complete(MESSAGES, response_schema=SCHEMA)
        self.assertEqual(transport.call_count, 1)
        summary = self.ledger.summary()
        self.assertEqual(summary["requests"], 1)
        self.assertEqual(summary["unknown_usage_requests"], 1)
        self.assertFalse(summary["usage_known"])
        self.assertGreater(summary["accounted_cost_usd"], 0)
        self.assertNotIn("FAKE_PROVIDER_KEY", self.path.read_text())

    def test_missing_provider_usage_keeps_reservation_even_when_content_is_valid(self):
        with patch.object(self.client, "_request", return_value=provider_reply(usage=False)):
            self.wrapped.complete(MESSAGES, response_schema=SCHEMA)
        summary = self.ledger.summary()
        self.assertFalse(summary["usage_known"])
        self.assertEqual(summary["unknown_usage_requests"], 1)
        self.assertGreater(summary["accounted_cost_usd"], 0)

    def test_shared_cap_stops_next_phase_before_http(self):
        wrapped = BudgetedClient(self.client, CostLedger(self.path.with_name("tiny.json"), cap_usd=0.00000001), self.pricing)
        with patch.object(self.client, "_request") as transport:
            with self.assertRaises((ValueError, NebiusError)):
                wrapped.complete(MESSAGES, response_schema=SCHEMA)
        transport.assert_not_called()

    def test_cancellation_prevents_reservation_and_provider_request(self):
        wrapped = BudgetedClient(self.client, self.ledger, self.pricing, cancel_check=lambda: True)
        with patch.object(self.client, "_request") as transport:
            with self.assertRaises((KeyboardInterrupt, ValueError, NebiusError)):
                wrapped.complete(MESSAGES, response_schema=SCHEMA)
        transport.assert_not_called()
        self.assertEqual(self.ledger.summary()["requests"], 0)


class DashboardReceiptTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.state = Path(self.directory.name) / "runner"

    def test_receipt_is_created_before_work_and_contains_current_lease_identity(self):
        store = ReceiptStore(self.state)
        directory = store.start(JOB)
        self.assertIsInstance(directory, Path)
        receipt = json.loads((directory / "receipt.json").read_text())
        self.assertEqual(receipt["jobId"], JOB["id"])
        self.assertEqual(receipt["leaseId"], JOB["leaseId"])
        self.assertEqual(receipt["status"], "running")

    def test_crash_recovery_never_reexecutes_the_same_received_job(self):
        ReceiptStore(self.state).start(JOB)
        restarted = ReceiptStore(self.state)
        with self.assertRaises(FileExistsError):
            restarted.start(JOB)
        with self.assertRaises(FileExistsError):
            restarted.start({**JOB, "leaseId": "fixture-new-lease"})

    def test_finished_job_cannot_be_run_again_with_the_same_id(self):
        store = ReceiptStore(self.state)
        directory = store.start(JOB)
        store.finish(JOB["id"], "completed")
        receipt = json.loads((directory / "receipt.json").read_text())
        self.assertEqual(receipt["status"], "completed")
        with self.assertRaises(FileExistsError):
            ReceiptStore(self.state).start(JOB)

    def test_path_traversal_job_is_rejected_before_receipt_creation(self):
        store = ReceiptStore(self.state)
        with self.assertRaises(ValueError):
            store.start({**JOB, "id": "../outside"})
        self.assertFalse((self.state.parent / "outside").exists())

    def test_interrupted_lease_recovery_publishes_saved_evidence_without_repeating_work(self):
        directory = ReceiptStore(self.state).start(JOB)
        write_recording(directory / "recording.jsonl", grid_recording())
        ledger = CostLedger(directory / "cost.json", cap_usd=1)
        ledger.reserve_request("nvidia/Nemotron-3_5-Lightning", MESSAGES, SCHEMA, 8192, TriagePricing())
        recovered = []
        with patch.object(NebiusClient, "_request", side_effect=AssertionError("Recovery cannot infer")) as transport, \
                patch("autoplay_qa.dashboard_runner.KeyGateGame", side_effect=AssertionError("Recovery cannot restart a game")) as game:
            count = recover_interrupted_receipts(self.state, lambda job, value: recovered.append((job, value)))
            self.assertEqual(recover_interrupted_receipts(self.state, lambda *_: self.fail("Duplicate recovery")), 0)
        transport.assert_not_called()
        game.assert_not_called()
        self.assertEqual(count, 1)
        job, result = recovered[0]
        self.assertEqual(job["leaseId"], JOB["leaseId"])
        self.assertEqual(result["summary"]["status"], "interrupted")
        self.assertEqual(result["summary"]["source"], "live")
        self.assertEqual(result["summary"]["controls"], 5)
        self.assertEqual(result["summary"]["requests"], 1)
        self.assertFalse(result["summary"]["costKnown"])
        self.assertGreater(result["summary"]["estimatedCostUsd"], 0)
        self.assertEqual(json.loads((directory / "receipt.json").read_text())["status"], "interrupted")

    def test_hard_stop_sweep_journal_recovers_canonical_negative_ids_with_no_private_labels(self):
        directory = ReceiptStore(self.state).start(JOB)
        records = grid_recording()
        write_recording(directory / "recording.jsonl", records[:1])
        for index, row in enumerate(records[1:], 1):
            row.update(type="sweep_step", source="sweep", phase_step=index, step=index - 401)
        write_recording(directory / "sweep-progress.jsonl", records)
        recovered = []
        with patch.object(NebiusClient, "_request", side_effect=AssertionError("Recovery cannot infer")):
            self.assertEqual(recover_interrupted_receipts(self.state, lambda job, value: recovered.append(value)), 1)
        result = recovered[0]
        self.assertEqual([row["step"] for row in result["frames"]], [-5, -4, -3, -2, -1])
        self.assertEqual([row["phaseStep"] for row in result["frames"]], [1, 2, 3, 4, 5])
        self.assertEqual(result["summary"]["controls"], 5)
        self.assertEqual(result["summary"]["requests"], 0)
        self.assertIn(-1, result["issues"][0]["evidenceSteps"])
        self.assertIn(-1, result["issues"][0]["reproduction"]["sourceSteps"])
        serialized = json.dumps(result)
        for sentinel in (*PRIVATE_SENTINELS[:2], PRIVATE_SENTINELS[3]):
            self.assertNotIn(sentinel, serialized)
        self.assertLess(len(serialized.encode()), 1_600_000)

    def test_recovered_terminal_upload_cannot_be_lost_behind_a_prior_server_revision(self):
        remote = {"revision": 7, "status": "running", "updates": []}

        class FakeTransport:
            def post(self, endpoint, payload):
                if endpoint in {"control", "heartbeat"}:
                    return {"cancelRequested": False, "revision": remote["revision"]}
                if endpoint != "update":
                    raise AssertionError("Only evidence/control HTTP is allowed in recovery.")
                remote["updates"].append(payload["revision"])
                if payload["revision"] > remote["revision"]:
                    remote["revision"] = payload["revision"]
                    remote["status"] = payload["snapshot"]["summary"]["status"]
                return {"cancelRequested": False, "revision": remote["revision"]}

        bridge = _Bridge(FakeTransport(), "fixture-runner-001", False, threading.Event())
        # The startup recovery path obtains this baseline from /runner/control.
        bridge.begin(JOB, revision=remote["revision"])
        bridge.publish({"summary": {"status": "interrupted"}})
        with patch.object(NebiusClient, "_request", side_effect=AssertionError("Recovery cannot infer")):
            bridge.flush()
        self.assertEqual(remote["status"], "interrupted")
        self.assertGreater(remote["revision"], 7)
        self.assertFalse(bridge.upload_failure)

    def test_failed_recovery_delivery_retries_saved_evidence_without_restarting_inference(self):
        directory = ReceiptStore(self.state).start(JOB)
        write_recording(directory / "recording.jsonl", grid_recording())
        attempted = []

        def failed_publish(job, value):
            attempted.append((job, value))
            return False

        with patch.object(NebiusClient, "_request", side_effect=AssertionError("Delivery retry cannot infer")) as transport, \
                patch("autoplay_qa.dashboard_runner.KeyGateGame", side_effect=AssertionError("Delivery retry cannot restart a game")) as game:
            self.assertEqual(recover_interrupted_receipts(self.state, failed_publish), 1)
            receipt = json.loads((directory / "receipt.json").read_text())
            self.assertTrue(receipt["deliveryPending"])
            delivered = []
            self.assertEqual(recover_interrupted_receipts(self.state, lambda job, value: delivered.append((job, value))), 1)
            self.assertEqual(recover_interrupted_receipts(self.state, lambda *_: self.fail("Already delivered")), 0)
        transport.assert_not_called()
        game.assert_not_called()
        self.assertEqual(len(attempted), 1)
        self.assertEqual(len(delivered), 1)
        self.assertEqual(delivered[0], attempted[0])
        self.assertFalse(json.loads((directory / "receipt.json").read_text())["deliveryPending"])


class DashboardAllowanceTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "allowance.json"

    def test_entire_job_caps_are_reserved_against_the_shared_ten_dollar_allowance(self):
        first = TotalAllowance(self.path, cap_usd=10)
        first.reserve("first-job", 7)
        resumed = TotalAllowance(self.path, cap_usd=10)
        with self.assertRaises(NebiusError):
            resumed.reserve("second-job", 4)
        resumed.reserve("second-job", 3)
        with self.assertRaises(NebiusError):
            resumed.reserve("third-job", 0.05)

    def test_unknown_job_usage_retains_the_whole_allowance_after_restart(self):
        allowance = TotalAllowance(self.path, cap_usd=10)
        allowance.reserve("first-job", 10)
        allowance.settle("first-job", {"usage_known": False, "accounted_cost_usd": 0.01})
        with self.assertRaises(NebiusError):
            TotalAllowance(self.path, cap_usd=10).reserve("second-job", 0.05)

    def test_known_usage_releases_the_unused_cap_and_settlement_is_idempotent(self):
        allowance = TotalAllowance(self.path, cap_usd=10)
        allowance.reserve("first-job", 10)
        allowance.settle("first-job", {"usage_known": True, "accounted_cost_usd": 0.2})
        allowance.settle("first-job", {"usage_known": True, "accounted_cost_usd": 0})
        resumed = TotalAllowance(self.path, cap_usd=10)
        with self.assertRaises(NebiusError):
            resumed.reserve("second-job", 10)
        resumed.reserve("second-job", 9.8)

    def test_duplicate_job_or_changed_session_cap_cannot_create_new_allowance(self):
        allowance = TotalAllowance(self.path, cap_usd=10)
        allowance.reserve("first-job", 1)
        with self.assertRaises(ValueError):
            allowance.reserve("first-job", 1)
        for cap in (11, 0, -1, True, float("nan"), float("inf")):
            with self.subTest(cap=cap), self.assertRaises(ValueError):
                TotalAllowance(self.path, cap_usd=cap)
        with self.assertRaises(ValueError):
            TotalAllowance(self.path, cap_usd=5)


class DashboardJobExecutionTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.state = Path(self.directory.name) / "runner"
        self.settings = Settings("OFFLINE_RUNNER_KEY", enable_thinking=False, model="nvidia/Nemotron-Other")
        self.games = []

    def game_factory(self, seed, defect):
        game = KeyGateGame(seed, None if defect == "none" else defect)
        self.games.append(game)
        return game

    def test_full_fake_job_preserves_defaults_and_publishes_monotonic_live_progress(self):
        factory_calls, updates, requests = [], [], []

        def client_factory(settings, **kwargs):
            factory_calls.append((settings, kwargs))
            return NebiusClient(settings, **kwargs)

        def fake_request(_client, _path, body, *_args, **_kwargs):
            requests.append(body)
            return provider_reply({
                "action": "finish", "reason": "Enough recorded evidence for this offline test.",
                "expected_position": None, "notes": [],
            })

        with patch.object(NebiusClient, "_request", autospec=True, side_effect=fake_request), \
                patch("autoplay_qa.dashboard_runner.run_collision_sweep", wraps=run_collision_sweep) as sweep:
            result = execute_job(
                JOB, state_dir=self.state, settings=self.settings,
                game_factory=self.game_factory, client_factory=client_factory, publish=updates.append,
            )

        self.assertEqual(result["summary"]["status"], "completed")
        self.assertEqual(result["summary"]["source"], "live")
        self.assertEqual(result["summary"]["requests"], 1)
        self.assertLessEqual(result["summary"]["controls"], 424)
        self.assertEqual(result["summary"]["resets"], 0)
        self.assertTrue(result["summary"]["costKnown"])
        self.assertEqual(sweep.call_args.kwargs["max_actions"], 400)
        self.assertEqual(len(self.games), 2)
        self.assertTrue(updates)
        self.assertEqual(updates[-1], result)
        self.assertEqual(updates[-1]["summary"]["phase"], "complete")
        self.assertEqual([entry["summary"]["requests"] for entry in updates], sorted(entry["summary"]["requests"] for entry in updates))
        for settings, kwargs in factory_calls:
            self.assertTrue(settings.enable_thinking)
            self.assertEqual(settings.response_format, "json_schema")
            self.assertEqual(kwargs, {"timeout": 45, "max_tokens": 8192, "temperature": 1.0, "top_p": 0.95})
        self.assertEqual(factory_calls[0][0].model, "nvidia/Nemotron-3_5-Lightning")
        self.assertEqual(requests[0]["max_tokens"], 8192)
        self.assertEqual(requests[0]["temperature"], 1.0)
        self.assertEqual(requests[0]["top_p"], 0.95)
        self.assertTrue(requests[0]["chat_template_kwargs"]["enable_thinking"])
        self.assertNotIn("OFFLINE_RUNNER_KEY", json.dumps(updates))
        self.assertNotIn("injected_defect", json.dumps(updates))
        self.assertNotIn("evaluation", json.dumps(updates))
        receipt = json.loads((self.state / JOB["id"] / "receipt.json").read_text())
        self.assertEqual(receipt["status"], "completed")
        self.assertEqual(receipt["requests"], 1)

    def test_cancellation_before_execution_never_creates_a_game_or_dispatches_inference(self):
        updates = []
        with patch.object(NebiusClient, "_request") as transport:
            result = execute_job(
                JOB, state_dir=self.state, settings=self.settings,
                game_factory=self.game_factory, publish=updates.append, cancel_check=lambda: True,
            )
        transport.assert_not_called()
        self.assertEqual(self.games, [])
        self.assertEqual(result["summary"]["status"], "cancelled")
        self.assertEqual(result["summary"]["controls"], 0)
        self.assertEqual(result["summary"]["resets"], 0)
        self.assertEqual(result["summary"]["requests"], 0)
        self.assertEqual(updates[-1], result)

    def test_existing_durable_receipt_refuses_a_second_full_job_before_http(self):
        ReceiptStore(self.state).start(JOB)
        with patch.object(NebiusClient, "_request") as transport:
            with self.assertRaises(FileExistsError):
                execute_job(JOB, state_dir=self.state, settings=self.settings, game_factory=self.game_factory)
        transport.assert_not_called()
        self.assertEqual(self.games, [])

    def test_cancelled_partial_sweep_keeps_every_executed_control_in_local_recording(self):
        state = {"cancelled": False, "actions": 0, "resets": 0}

        class TrackedGame:
            def __init__(self, seed, defect):
                self.inner = KeyGateGame(seed, None if defect == "none" else defect)

            def __getattr__(self, name):
                return getattr(self.inner, name)

            def act(self, action):
                result = self.inner.act(action)
                state["actions"] += 1
                if state["actions"] == 5:
                    state["cancelled"] = True
                return result

            def reset(self, *args, **kwargs):
                state["resets"] += 1
                self.inner.reset(*args, **kwargs)
                return self

        with patch.object(NebiusClient, "_request") as transport:
            result = execute_job(
                JOB, state_dir=self.state, settings=self.settings, game_factory=TrackedGame,
                cancel_check=lambda: state["cancelled"],
            )
        transport.assert_not_called()
        self.assertEqual(result["summary"]["status"], "cancelled")
        self.assertEqual(state["actions"], 5)
        self.assertEqual(state["resets"], 0)
        self.assertEqual(result["summary"]["controls"], 5)
        rows = [json.loads(line) for line in (self.state / JOB["id"] / "recording.jsonl").read_text().splitlines()]
        executed = [row for row in rows if row["type"] in {"sweep_step", "sweep_reset", "step", "reset"}]
        self.assertEqual(len(executed), 5)
        self.assertEqual([row["step"] for row in executed], [frame["step"] for frame in result["frames"]])
        self.assertEqual([row["phase_step"] for row in executed], [1, 2, 3, 4, 5])

    def test_cancellation_after_provider_response_stops_a_proposed_reset_before_execution(self):
        state = {"cancelled": False, "resets": 0}

        class TrackedGame:
            def __init__(self, seed, defect):
                self.inner = KeyGateGame(seed, None if defect == "none" else defect)

            def __getattr__(self, name):
                return getattr(self.inner, name)

            def reset(self, *args, **kwargs):
                state["resets"] += 1
                self.inner.reset(*args, **kwargs)
                return self

        def fake_request(_client, _path, _body, *_args, **_kwargs):
            state["cancelled"] = True
            return provider_reply({
                "action": "reset", "reason": "The model proposes a new session start.",
                "expected_position": None, "notes": [],
            })

        with patch.object(NebiusClient, "_request", autospec=True, side_effect=fake_request) as transport:
            result = execute_job(
                JOB, state_dir=self.state, settings=self.settings, game_factory=TrackedGame,
                cancel_check=lambda: state["cancelled"],
            )
        self.assertEqual(transport.call_count, 1)
        self.assertEqual(result["summary"]["status"], "cancelled")
        self.assertEqual(result["summary"]["requests"], 1)
        self.assertEqual(result["summary"]["resets"], 0)
        self.assertEqual(state["resets"], 0)
        self.assertTrue(result["summary"]["costKnown"])
        self.assertFalse(any(frame["phase"] == "llm" for frame in result["frames"]))


if __name__ == "__main__":
    unittest.main()
