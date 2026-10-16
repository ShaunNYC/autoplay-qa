"""Conformance and parity tests for the localhost JSON-lines adapter."""

from __future__ import annotations

import json
import os
import socket
import threading
import unittest
from unittest.mock import patch

from autoplay_qa.benchmark import run_scripted_benchmark
from autoplay_qa.game import DEFECT_CHOICES, KeyGateGame
from autoplay_qa.lean import step_fact
from autoplay_qa.protocol import PROTOCOL_VERSION, read_message, write_message
from autoplay_qa.remote import RemoteGameAdapter, parse_tcp_endpoint
from autoplay_qa.reporting import extract_issues
from autoplay_qa.server import serve_stream


class _StreamServer:
    """Provide real sockets without binding a sandbox-forbidden TCP port."""

    def __init__(self, *, seed=0, defect="none"):
        self.seed = seed
        self.defect = defect
        self.threads: list[threading.Thread] = []

    def connect(self, _address, timeout=None):
        client, server = socket.socketpair()
        if timeout is not None:
            client.settimeout(timeout)

        def target():
            try:
                with server:
                    reader = server.makefile("rb")
                    writer = server.makefile("wb")
                    try:
                        serve_stream(
                            reader,
                            writer,
                            seed=self.seed,
                            defect=self.defect,
                        )
                    finally:
                        reader.close()
                        writer.close()
            except OSError:
                # A test-side close may race a final flush after a failed case.
                pass

        thread = threading.Thread(target=target, daemon=True)
        thread.start()
        self.threads.append(thread)
        return client

    def join(self):
        for thread in self.threads:
            thread.join(timeout=2)
            if thread.is_alive():
                raise AssertionError("Protocol server thread did not finish.")


class ProtocolConformanceTests(unittest.TestCase):
    def test_all_commands_and_error_envelopes_on_one_threaded_session(self):
        client, server = socket.socketpair()

        def target():
            with server:
                reader = server.makefile("rb")
                writer = server.makefile("wb")
                try:
                    serve_stream(reader, writer, seed=0, defect="wall-clip")
                finally:
                    reader.close()
                    writer.close()

        thread = threading.Thread(target=target, daemon=True)
        thread.start()
        with client:
            stream = client.makefile("rwb")
            try:
                stream.write(b"not-json\n")
                stream.flush()
                self.assertEqual(read_message(stream)["error"]["code"], "invalid_json")

                stream.write(b"[]\n")
                stream.flush()
                self.assertEqual(read_message(stream)["error"]["code"], "invalid_message")

                stream.write(b"\xff\n")
                stream.flush()
                self.assertEqual(read_message(stream)["error"]["code"], "invalid_utf8")

                write_message(stream, {"type": "observe"})
                missing_id = read_message(stream)
                self.assertIsNone(missing_id["id"])
                self.assertEqual(missing_id["error"]["code"], "invalid_request")

                write_message(stream, {"id": "1", "type": "observe"})
                error = read_message(stream)
                self.assertEqual(error["error"]["code"], "hello_required")

                write_message(stream, {
                    "id": "2", "type": "hello", "protocol": "autoplay-qa/999",
                })
                unsupported = read_message(stream)
                self.assertEqual(unsupported["error"]["code"], "unsupported_protocol")

                write_message(stream, {
                    "id": "3", "type": "hello", "protocol": PROTOCOL_VERSION,
                })
                hello = read_message(stream)
                self.assertTrue(hello["ok"])
                self.assertEqual(hello["protocol"], PROTOCOL_VERSION)
                self.assertEqual(hello["game"]["id"], "key-gate")
                self.assertTrue(hello["capabilities"]["grid"])

                write_message(stream, {
                    "id": "4", "type": "hello", "protocol": PROTOCOL_VERSION,
                })
                second_hello = read_message(stream)
                self.assertEqual(second_hello["error"]["code"], "already_initialized")

                write_message(stream, {"id": "5", "type": "observe"})
                observed = read_message(stream)
                self.assertEqual(observed["observation"]["player"]["position"], [1, 1])
                serialized = json.dumps(observed)
                self.assertNotIn("wall-clip", serialized)
                self.assertNotIn('"rules"', serialized)

                write_message(stream, {"id": "5", "type": "observe"})
                duplicate = read_message(stream)
                self.assertEqual(duplicate["error"]["code"], "duplicate_id")

                write_message(stream, {"id": "6", "type": "act"})
                missing_control = read_message(stream)
                self.assertEqual(missing_control["error"]["code"], "invalid_control")

                write_message(stream, {
                    "id": "7", "type": "act", "control": "teleport",
                })
                illegal = read_message(stream)
                self.assertEqual(illegal["error"]["code"], "illegal_control")

                write_message(stream, {
                    "id": "8", "type": "act", "control": "wait",
                })
                acted = read_message(stream)
                self.assertEqual(acted["observation"]["tick"], 1)

                write_message(stream, {"id": "9", "type": "reset"})
                missing_seed = read_message(stream)
                self.assertEqual(missing_seed["error"]["code"], "invalid_seed")

                write_message(stream, {"id": "10", "type": "reset", "seed": 5})
                reset = read_message(stream)
                self.assertEqual(reset["observation"]["tick"], 0)

                write_message(stream, {"id": "11", "type": "unknown"})
                unknown = read_message(stream)
                self.assertEqual(unknown["error"]["code"], "unknown_type")

                write_message(stream, {"id": "12", "type": "close"})
                self.assertTrue(read_message(stream)["ok"])
            finally:
                stream.close()
        thread.join(timeout=2)
        self.assertFalse(thread.is_alive())

    def test_endpoint_parser_rejects_non_local_hosts(self):
        self.assertEqual(parse_tcp_endpoint("tcp://127.0.0.1:1234"), ("127.0.0.1", 1234))
        with self.assertRaisesRegex(ValueError, "localhost"):
            parse_tcp_endpoint("tcp://example.com:1234")

    def test_no_grid_skips_map_detectors_but_keeps_feedback_inventory_checks(self):
        before = {
            "game": "generic-v1",
            "tick": 0,
            "status": "playing",
            "player": {"position": [0, 0], "inventory": ["key"]},
            "entities": [{
                "id": "gate", "kind": "gate", "position": [1, 0],
                "state": {"open": False},
            }],
            "events": ["Environment started."],
            "actions": ["interact"],
        }
        after = json.loads(json.dumps(before))
        after.update(tick=1, events=["Unlocked the gate and used the key."])
        after["entities"][0]["state"]["open"] = True
        record = {
            "type": "step",
            "step": 1,
            "episode": 0,
            "episode_before": 0,
            "episode_after": 0,
            "before": before,
            "after": after,
            "decision": {
                "action": "interact",
                "reason": "fixture",
                "expected_position": None,
                "notes": [],
            },
            "qa_mode": True,
        }

        aspects = {
            anomaly.get("aspect")
            for anomaly in step_fact(record)["anomalies"]
        }

        self.assertIn("key_use_feedback_inventory", aspects)
        self.assertNotIn("player_position_glyph", aspects)
        self.assertFalse(any("object_glyph" in str(value) for value in aspects))

    def test_explicit_false_grid_capability_overrides_incidental_grid_fields(self):
        capabilities = {"grid": False, "legend": False, "entities": True}
        before = {
            "game": "generic-v1",
            "tick": 0,
            "status": "playing",
            "player": {"position": [0, 0], "inventory": []},
            "entities": [],
            "events": ["Environment started."],
            "actions": ["east"],
            "map": ["@#"],
            "legend": {"@": "player", "#": "wall"},
            "_capabilities": capabilities,
        }
        after = json.loads(json.dumps(before))
        after.update(tick=1, events=["Moved east."], map=["#@"])
        after["player"]["position"] = [1, 0]
        record = {
            "type": "step",
            "step": 1,
            "episode": 0,
            "episode_before": 0,
            "episode_after": 0,
            "before": before,
            "after": after,
            "decision": {
                "action": "east",
                "reason": "fixture",
                "expected_position": None,
                "notes": [],
            },
            "qa_mode": True,
        }

        aspects = {
            anomaly.get("aspect")
            for anomaly in step_fact(record)["anomalies"]
        }
        records = [{
            "type": "session",
            "schema_version": 2,
            "seed": 0,
            "focus": "qa",
            "initial_observation": before,
        }, record]

        self.assertNotIn("entered_wall_labeled_tile", aspects)
        self.assertEqual(extract_issues(records), [])


class RemoteParityTests(unittest.TestCase):
    def test_reference_remote_matches_builtin_scripted_observations(self):
        transport = _StreamServer(seed=0)
        local = KeyGateGame(0)
        with patch("autoplay_qa.remote.socket.create_connection", transport.connect):
            with RemoteGameAdapter("tcp://127.0.0.1:12345", seed=0) as remote:
                self.assertEqual(remote.observe(), local.observe())
                self.assertEqual(
                    json.dumps(remote.observe(), separators=(",", ":")),
                    json.dumps(local.observe(), separators=(",", ":")),
                )
                for action in (
                    "south", "south", "east", "east", "interact", "east",
                ):
                    before_remote = remote.observe()
                    before_local = local.observe()
                    self.assertEqual(before_remote, before_local)
                    after_remote = remote.act(action)
                    after_local = local.act(action)
                    self.assertEqual(after_remote, after_local)
                    self.assertEqual(
                        remote.evaluate_transition(before_remote, action, after_remote),
                        local.evaluate_transition(before_local, action, after_local),
                    )
        transport.join()

    def test_full_scripted_benchmark_runs_over_protocol(self):
        # The full 3x6 matrix is useful as an opt-in integration soak, but it
        # repeatedly runs report/reproduction minimization and made this unit
        # module take almost three minutes.  Keep the same socketpair-backed
        # end-to-end coverage on one representative case by default.
        full_matrix = os.environ.get("AUTOPLAY_QA_FULL_PROTOCOL_MATRIX") == "1"
        defects = DEFECT_CHOICES if full_matrix else ("wall-clip",)
        seeds = (0, 1, 5) if full_matrix else (0,)
        transports = {
            defect: _StreamServer(defect=defect)
            for defect in defects
        }
        ports = {
            defect: 12000 + index
            for index, defect in enumerate(defects)
        }
        by_port = {ports[defect]: transport for defect, transport in transports.items()}

        def connect(address, timeout=None):
            return by_port[address[1]].connect(address, timeout=timeout)

        def factory(seed, defect):
            name = defect or "none"
            return RemoteGameAdapter(
                f"tcp://127.0.0.1:{ports[name]}",
                seed=seed,
            )

        with patch("autoplay_qa.remote.socket.create_connection", side_effect=connect):
            result = run_scripted_benchmark(
                seeds=seeds,
                defects=defects,
                game_factory=factory,
            )

        for transport in transports.values():
            transport.join()
        self.assertTrue(result["passed"])
        self.assertEqual(len(result["rows"]), len(seeds) * len(defects))
        self.assertTrue(all(
            row["detection"] == ("clean" if row["defect"] == "none" else "hit")
            for row in result["rows"]
        ))
        self.assertTrue(all(
            row["repro_verified"]
            for row in result["rows"]
            if row["defect"] != "none"
        ))


if __name__ == "__main__":
    unittest.main()
