"""Socketpair-backed Phase-5b continuous-game pipeline coverage."""

from __future__ import annotations

import json
import os
import socket
import threading
import unittest
from unittest.mock import patch

from autoplay_qa.benchmark import run_scripted_benchmark
from autoplay_qa.lean import LEAN_CONTINUOUS_PROMPT_MAX_BYTES, lean_messages
from autoplay_qa.protocol import PROTOCOL_VERSION, read_message, write_message
from autoplay_qa.remote import RemoteGameAdapter
from autoplay_qa.reporting import evaluate_recording, extract_issues, reproduce_issue
from tests.continuous_fixture import (
    CONTINUOUS_DEFECTS,
    SPAWN,
    ContinuousStreamServer,
    FakeContinuousGame,
    serve_fake_continuous_stream,
)


EXPECTED_TITLES = {
    "door-ghost-collider": "Movement blocked with no visible obstacle",
    "key-not-consumed": "Key remained in inventory after feedback said it was used",
    "respawn-leak": "Reset did not restore the initial state",
    "wall-tunnel": "Player passed through a solid wall",
}
FLOOR_GAP_TITLES = {
    0: "Player fell through a visible floor tile",
    1: "Player sank into a visible floor tile",
    5: "Player sank into a visible floor tile",
}


class ContinuousProtocolTests(unittest.TestCase):
    def test_seeded_geometry_selection_matches_unity_creation_order(self):
        expected = {
            0: ("floor-4-1", "wall-interior-0"),
            1: ("floor-5-5", "wall-interior-4"),
            5: ("floor-2-4", "wall-interior-5"),
        }
        for seed, selected in expected.items():
            with self.subTest(seed=seed):
                game = FakeContinuousGame(seed, "floor-gap")
                wall_game = FakeContinuousGame(seed, "wall-tunnel")
                self.assertEqual(
                    (game._floor_gap_id, wall_game._wall_tunnel_id), selected
                )

    def test_floor_gap_wedge_reports_first_sink_and_reproduces_short_path(self):
        game = FakeContinuousGame(5, "floor-gap")
        initial = game.observe()
        records = [{
            "type": "session",
            "schema_version": 2,
            "game_version": game.version,
            "seed": 5,
            "mode": "scripted-benchmark",
            "task": "explore",
            "focus": "qa",
            "initial_observation": initial,
            "evaluation": {
                "injected_defect": {
                    "name": "floor-gap",
                    "description": "fixture",
                },
            },
        }]
        actions = ("north", "north", "north", "north", "east", "north", "north")
        for step, action in enumerate(actions, 1):
            before = game.observe()
            after = game.act(action)
            records.append({
                "type": "sweep_step",
                "source": "sweep",
                "step": step,
                "episode": 0,
                "episode_before": 0,
                "episode_after": 0,
                "before": before,
                "decision": {
                    "action": action,
                    "reason": "Short public-control floor-gap fixture path.",
                    "expected_position": None,
                    "notes": [],
                },
                "after": after,
                "qa_mode": True,
            })

        sunk = records[5]["after"]
        wedged = records[6]["after"]
        fallen = records[7]["after"]
        self.assertEqual(sunk["player"]["position"], [-2.45, 0.1, 2.2])
        self.assertFalse(sunk["player"]["grounded"])
        self.assertEqual(sunk["events"], [])
        self.assertEqual(wedged["player"], sunk["player"])
        self.assertEqual(wedged["events"], [])
        self.assertEqual(fallen["player"]["position"], list(SPAWN))
        self.assertTrue(fallen["player"]["grounded"])
        self.assertEqual(fallen["events"], ["You fell."])
        recovered = game.act("north")
        self.assertEqual(recovered["player"]["position"], [-4.0, 1.05, -2.45])

        seed_one = FakeContinuousGame(1, "floor-gap")
        seed_one_path = (
            "north", "north", "north", "north", "east", "east", "interact",
            "east", "east", "north", "east",
        )
        for action in seed_one_path:
            seed_one.act(action)
        self.assertEqual(seed_one.observe()["player"]["position"], [3.75, 0.1, 3.75])
        self.assertFalse(seed_one.observe()["player"]["grounded"])
        seed_one.reset()
        self.assertEqual(seed_one.observe()["player"]["position"], list(SPAWN))
        self.assertTrue(seed_one.observe()["player"]["grounded"])

        issues = extract_issues(records)
        issue_types = {issue["type"] for issue in issues}
        self.assertIn("sank_into_visible_floor", issue_types)
        self.assertIn("player_stuck_in_geometry", issue_types)
        self.assertNotIn("blocked_no_visible_obstacle", issue_types)
        sink_issue = next(
            issue for issue in issues
            if issue["type"] == "sank_into_visible_floor"
        )
        self.assertEqual(sink_issue["evidence_steps"], [5])
        stuck_issue = next(
            issue for issue in issues
            if issue["type"] == "player_stuck_in_geometry"
        )
        self.assertEqual(stuck_issue["evidence_steps"], [6])

        evaluation = evaluate_recording(records, verify_reproduction=False)
        self.assertEqual(evaluation["detection"], "hit")
        self.assertEqual(evaluation["false_positive_count"], 0)

        reproduced = reproduce_issue(
            records,
            sink_issue["number"],
            game_factory=lambda seed, defect: FakeContinuousGame(seed, defect),
        )
        self.assertTrue(reproduced["reproduced"])
        self.assertEqual(reproduced["original_action_count"], 5)
        self.assertEqual(reproduced["minimized_action_count"], 5)
        self.assertEqual(reproduced["actions"], list(actions[:5]))

    def test_wire_contract_and_prompt_do_not_expose_harness_defect(self):
        client, server = socket.socketpair()

        def target() -> None:
            with server:
                reader = server.makefile("rb")
                writer = server.makefile("wb")
                try:
                    serve_fake_continuous_stream(
                        reader, writer, seed=5, defect="wall-tunnel"
                    )
                finally:
                    reader.close()
                    writer.close()

        thread = threading.Thread(target=target, daemon=True)
        thread.start()
        with client:
            stream = client.makefile("rwb")
            try:
                write_message(stream, {
                    "id": "1", "type": "hello", "protocol": PROTOCOL_VERSION,
                })
                hello = read_message(stream)
                self.assertTrue(hello["capabilities"]["continuous"])
                self.assertTrue(hello["capabilities"]["solids_visible"])
                self.assertFalse(hello["capabilities"]["grid"])

                write_message(stream, {"id": "2", "type": "observe"})
                observed = read_message(stream)
                self.assertEqual(len(observed["observation"]["entities"]), 62)
                wire_text = json.dumps([hello, observed], separators=(",", ":"))
                for defect in CONTINUOUS_DEFECTS[1:]:
                    self.assertNotIn(defect, wire_text)

                write_message(stream, {"id": "3", "type": "close"})
                self.assertTrue(read_message(stream)["ok"])
            finally:
                stream.close()
        thread.join(timeout=2)
        self.assertFalse(thread.is_alive())

        transport = ContinuousStreamServer(seed=5, defect="wall-tunnel")
        with patch(
            "autoplay_qa.remote.socket.create_connection", transport.connect
        ):
            with RemoteGameAdapter(
                "tcp://127.0.0.1:13000", seed=5
            ) as adapter:
                observation = adapter.observe()
                messages = lean_messages(
                    observation,
                    0,
                    {},
                    [],
                    {"actions": 24, "requests": 28, "seconds": 300},
                    focus="qa",
                )
        transport.join()
        prompt = json.dumps(messages, separators=(",", ":"))
        self.assertLess(
            len(prompt.encode("utf-8")), LEAN_CONTINUOUS_PROMPT_MAX_BYTES
        )
        for defect in CONTINUOUS_DEFECTS[1:]:
            self.assertNotIn(defect, prompt)

    def test_scripted_report_reproduce_evaluate_matrix_over_protocol(self):
        # Keep one clean control and one representative defect in the ordinary
        # unit suite.  The exact 3x6 director matrix remains available as an
        # opt-in integration soak; each row intentionally performs a full 7x7
        # sampled sweep plus fresh-process reproduction.
        full_matrix = (
            os.environ.get("AUTOPLAY_QA_FULL_CONTINUOUS_MATRIX") == "1"
        )
        seeds = (0, 1, 5) if full_matrix else (0,)
        defects = CONTINUOUS_DEFECTS if full_matrix else ("none", "wall-tunnel")
        transports = {
            defect: ContinuousStreamServer(defect=defect)
            for defect in defects
        }
        ports = {
            defect: 13010 + index
            for index, defect in enumerate(defects)
        }
        by_port = {
            ports[defect]: transport
            for defect, transport in transports.items()
        }

        def connect(address, timeout=None):
            return by_port[address[1]].connect(address, timeout=timeout)

        def factory(seed, defect):
            name = defect or "none"
            return RemoteGameAdapter(
                f"tcp://127.0.0.1:{ports[name]}", seed=seed
            )

        try:
            with patch(
                "autoplay_qa.remote.socket.create_connection",
                side_effect=connect,
            ):
                result = run_scripted_benchmark(
                    seeds=seeds,
                    defects=defects,
                    game_factory=factory,
                    sweep_actions=400,
                )
        finally:
            for transport in transports.values():
                transport.join()

        self.assertTrue(result["passed"])
        self.assertEqual(
            len(result["rows"]), len(seeds) * len(defects)
        )
        for row in result["rows"]:
            with self.subTest(defect=row["defect"], seed=row["seed"]):
                if row["defect"] == "none":
                    self.assertEqual(row["detection"], "clean")
                    self.assertEqual(row["false_positive_count"], 0)
                    self.assertIsNone(row["matched_title"])
                else:
                    self.assertEqual(row["detection"], "hit")
                    self.assertTrue(row["repro_verified"])
                    if row["defect"] == "floor-gap" and row["seed"] in {1, 5}:
                        self.assertIn(
                            FLOOR_GAP_TITLES[row["seed"]],
                            row["matched_title"].split("; "),
                        )
                    elif row["defect"] == "floor-gap":
                        self.assertEqual(
                            row["matched_title"], FLOOR_GAP_TITLES[row["seed"]]
                        )
                    else:
                        self.assertEqual(
                            row["matched_title"], EXPECTED_TITLES[row["defect"]]
                        )


if __name__ == "__main__":
    unittest.main()
