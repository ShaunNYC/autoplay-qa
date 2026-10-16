"""Offline CLI coverage for injected sessions and Phase-2 artifacts."""

import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from autoplay_qa.cli import main
from autoplay_qa.config import Settings
from autoplay_qa.game import DEFECTS, KeyGateGame, defect_evaluation_metadata
from autoplay_qa.lean import software_facts, step_fact
from autoplay_qa.reporting import evaluate_recording, reproduce_issue


def completion(value):
    return {
        "id": "phase2-cli-fixture",
        "model": "nvidia/unit-test-nemotron",
        "choices": [{"message": {"content": json.dumps(value)}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    }


def write_gate_recording(path: Path) -> None:
    game = KeyGateGame(0, "gate-keeps-key")
    records = [{
        "type": "session",
        "schema_version": 2,
        "game_version": game.version,
        "seed": game.seed,
        "mode": "unit-test",
        "task": "explore",
        "focus": "qa",
        "initial_observation": game.observe(),
        **defect_evaluation_metadata(game.defect),
    }]
    actions = ["south", "south", "east", "east", "interact"]
    action_records = []
    for number, action in enumerate(actions, 1):
        before = game.observe()
        after = game.act(action)
        record = {
            "type": "step",
            "step": number,
            "episode": 0,
            "episode_before": 0,
            "episode_after": 0,
            "before": before,
            "decision": {
                "action": action,
                "reason": "fixture",
                "expected_position": after["player"]["position"],
                "notes": [],
            },
            "after": after,
            "evaluation": game.evaluate_transition(before, action, after),
            "qa_mode": True,
        }
        record["software_fact"] = step_fact(record, action_records)
        action_records.append(record)
        records.append(record)
    note = {
        "id": "gate-key",
        "claim": "The key remained in inventory after the gate opened.",
        "status": "supported",
        "kind": "suspected_bug",
        "evidence_steps": [5],
    }
    records.extend([
        {"type": "memory", "hypotheses": [note]},
        {"type": "summary", "steps": 5, "hypotheses": [note]},
    ])
    path.write_text(
        "\n".join(json.dumps(record, separators=(",", ":")) for record in records) + "\n",
        encoding="utf-8",
    )


class DefectRequestPrivacyTests(unittest.TestCase):
    def test_play_accepts_every_defect_without_network(self):
        for name in DEFECTS:
            with self.subTest(defect=name), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "play.jsonl"
                with patch("builtins.input", return_value="q"), \
                        patch("autoplay_qa.nebius.NebiusClient._request", side_effect=AssertionError("play is offline")) as transport, \
                        patch("sys.stdout", new_callable=io.StringIO):
                    self.assertEqual(main([
                        "play", "--defect", name, "--log", str(path),
                    ]), 0)
                header = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
                self.assertEqual(header["evaluation"]["injected_defect"]["name"], name)
                transport.assert_not_called()

    def test_every_defect_is_session_only_and_absent_from_explore_requests(self):
        probes = {
            "gate-keeps-key": ["south", "south", "east", "east", "interact"],
            "wall-clip": ["east", "east", "east"],
            "reset-leak": ["south", "south", "reset"],
            "exit-no-win": [
                "south", "south", "east", "east", "interact",
                "east", "east", "east", "east",
            ],
            "phantom-block": ["east", "east", "south"],
        }
        finished = {
            "action": "finish",
            "reason": "Enough testing",
            "expected_position": None,
            "notes": [],
        }
        for name, description in DEFECTS.items():
            with self.subTest(defect=name), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "explore.jsonl"
                decisions = [{
                    "action": action,
                    "reason": "Probe the visible game state.",
                    "expected_position": None,
                    "notes": [],
                } for action in probes[name]] + [finished]
                with patch("autoplay_qa.cli.Settings.load", return_value=Settings("unit-test-key")), \
                        patch(
                            "autoplay_qa.nebius.NebiusClient._request",
                            side_effect=[completion(decision) for decision in decisions],
                        ) as transport, \
                        patch("sys.stdout", new_callable=io.StringIO):
                    self.assertEqual(main([
                        "explore", "--focus", "qa", "--defect", name,
                        "--log", str(path),
                    ]), 0)

                self.assertEqual(transport.call_count, len(decisions))
                for call in transport.call_args_list:
                    payload = json.dumps(call.args[1])
                    self.assertNotIn(name, payload)
                    self.assertNotIn(description, payload)
                rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
                self.assertEqual(
                    rows[0]["evaluation"]["injected_defect"],
                    {"name": name, "description": description},
                )
                for request in (row for row in rows if row["type"] == "request"):
                    serialized = json.dumps(request)
                    self.assertNotIn(name, serialized)
                    self.assertNotIn(description, serialized)
                self.assertNotIn(name, json.dumps(rows[1:]))
                self.assertNotIn(description, json.dumps(rows[1:]))

    def test_run_request_also_excludes_private_defect_metadata(self):
        value = {"action": "wait", "reason": "fixture", "plan": "continue"}
        for name, description in DEFECTS.items():
            with self.subTest(defect=name), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "run.jsonl"
                with patch("autoplay_qa.cli.Settings.load", return_value=Settings("unit-test-key")), \
                        patch("autoplay_qa.nebius.NebiusClient._request", return_value=completion(value)) as transport, \
                        patch("sys.stdout", new_callable=io.StringIO):
                    self.assertEqual(main([
                        "run", "--defect", name, "--max-steps", "1", "--log", str(path),
                    ]), 3)
                payload = json.dumps(transport.call_args.args[1])
                self.assertNotIn(name, payload)
                self.assertNotIn(description, payload)


class OfflineArtifactCliTests(unittest.TestCase):
    def test_all_real_defects_evaluate_and_reproduce(self):
        cases = {
            "gate-keeps-key": (
                ["south", "south", "east", "east", "interact"],
                "The gate opened but did not consume the key.",
            ),
            "wall-clip": (
                ["east", "east", "east"],
                "The player walked through a wall.",
            ),
            "reset-leak": (
                ["south", "south", "reset"],
                "Reset did not restore the key and inventory state.",
            ),
            "exit-no-win": (
                ["south", "south", "east", "east", "interact", "east", "east", "east", "east"],
                "The exit did not trigger a win on first entry.",
            ),
            "phantom-block": (
                ["east", "east", "south"],
                "A floor tile blocked movement.",
            ),
        }
        for defect, (actions, claim) in cases.items():
            with self.subTest(defect=defect):
                game = KeyGateGame(0, defect)
                rows = [{
                    "type": "session",
                    "schema_version": 2,
                    "game_version": game.version,
                    "seed": 0,
                    "mode": "unit-test",
                    "task": "explore",
                    "focus": "qa",
                    "initial_observation": game.observe(),
                    **defect_evaluation_metadata(defect),
                }]
                action_rows = []
                episode = 0
                for number, action in enumerate(actions, 1):
                    before = game.observe()
                    if action == "reset":
                        after = game.reset().observe()
                        row = {
                            "type": "reset",
                            "episode_before": episode,
                            "episode_after": episode + 1,
                        }
                        episode += 1
                    else:
                        after = game.act(action)
                        row = {
                            "type": "step",
                            "episode": episode,
                            "episode_before": episode,
                            "episode_after": episode,
                            "evaluation": game.evaluate_transition(before, action, after),
                        }
                    row.update(
                        step=number,
                        before=before,
                        after=after,
                        decision={
                            "action": action,
                            "reason": "fixture",
                            "expected_position": after["player"]["position"],
                            "notes": [],
                        },
                        qa_mode=True,
                    )
                    row["software_fact"] = step_fact(row, action_rows)
                    action_rows.append(row)
                    rows.append(row)
                note = {
                    "id": "bug",
                    "claim": claim,
                    "status": "supported",
                    "kind": "suspected_bug",
                    "evidence_steps": [len(actions)],
                }
                rows.extend([
                    {"type": "memory", "hypotheses": [note]},
                    {
                        "type": "summary",
                        "steps": len(actions),
                        "hypotheses": [note],
                        **software_facts(action_rows, history_size=None),
                    },
                ])

                reproduced = reproduce_issue(rows, 1)
                evaluated = evaluate_recording(rows)

                self.assertTrue(reproduced["reproduced"])
                self.assertEqual(reproduced["exit_code"], 0)
                self.assertEqual(evaluated["detection"], "hit")
                self.assertEqual(evaluated["false_positive_count"], 0)
                self.assertTrue(evaluated["repro_verified"])

    def test_report_reproduce_and_evaluate_are_offline(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "gate.jsonl"
            out = Path(directory) / "bugs.md"
            write_gate_recording(path)
            offline = patch(
                "autoplay_qa.cli.Settings.load",
                side_effect=AssertionError("offline commands must not load credentials"),
            )
            transport = patch(
                "autoplay_qa.nebius.NebiusClient._request",
                side_effect=AssertionError("offline commands must not call inference"),
            )
            with offline as load, transport as request, patch("sys.stdout", new_callable=io.StringIO) as output:
                self.assertEqual(main(["report", str(path)]), 0)
                report = output.getvalue()
                self.assertIn("## Issue 1:", report)
                self.assertNotIn("gate-keeps-key", report)

                output.seek(0)
                output.truncate(0)
                self.assertEqual(main(["report", str(path), "--out", str(out)]), 0)
                self.assertTrue(out.is_file())

                output.seek(0)
                output.truncate(0)
                self.assertEqual(main(["reproduce", str(path), "--issue", "1"]), 0)
                reproduced = json.loads(output.getvalue())
                self.assertTrue(reproduced["reproduced"])
                self.assertEqual(reproduced["actions"], ["south", "south", "east", "east", "interact"])

                output.seek(0)
                output.truncate(0)
                self.assertEqual(main(["evaluate", str(path)]), 0)
                evaluated = json.loads(output.getvalue())
                self.assertEqual(evaluated["recordings"][0]["detection"], "hit")
                self.assertTrue(evaluated["recordings"][0]["repro_verified"])
            load.assert_not_called()
            request.assert_not_called()


if __name__ == "__main__":
    unittest.main()
