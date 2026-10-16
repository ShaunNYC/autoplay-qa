"""Portable offline integration checks against an optional external Mars engine.

The direct-engine oracle and the QA bridge build the real source independently.
Fixture programs below are host-side test inputs, not live model decisions and
not evidence of AI exploration. No HTTP or paid inference is used here.
Place the separately obtained project in external/mars, or configure
AUTOPLAY_MARS_PROJECT. The default missing external dependency is skipped.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from autoplay_qa.cli import begin_session
from autoplay_qa.config import Settings
from autoplay_qa.exploration import ExplorationLimits
from autoplay_qa.lean import lean_messages, run_lean_exploration, step_fact
from autoplay_qa.nebius import NebiusClient
from autoplay_qa.reporting import extract_issues
from autoplay_qa.session import SessionLog


ROOT = Path(__file__).resolve().parents[1]
ORIGINAL_PROJECT = Path(os.environ.get(
    "AUTOPLAY_MARS_PROJECT", ROOT / "external/mars",
))
SOURCE = ORIGINAL_PROJECT / "src/mars/engine.ts"
NODE = shutil.which("node") or "/usr/local/bin/node"


def _block(kind: str, **fields: object) -> dict:
    return {"id": "offline-fixture", "kind": kind, **fields}


def _oracle(requests: list[dict]) -> dict:
    completed = subprocess.run(
        [NODE, str(ROOT / "tools/mars_conformance.mjs"),
         "--project-dir", str(ORIGINAL_PROJECT)],
        input=json.dumps(requests), capture_output=True, text=True,
        timeout=30, check=False,
    )
    if completed.returncode:
        raise AssertionError("Independent original-engine oracle failed: " + completed.stderr[-1500:])
    return json.loads(completed.stdout)


def _step(before: dict, action: str, after: dict, number: int = 1) -> dict:
    return {
        "type": "step", "step": number, "episode": 0,
        "episode_before": 0, "episode_after": 0,
        "before": before, "after": after,
        "decision": {"action": action, "reason": "offline conformance fixture",
                     "expected_position": None, "notes": []},
        "qa_mode": True,
    }


def _response(action: str) -> dict:
    return {
        "id": "offline-mars-fixture", "model": "nvidia/offline-fixture",
        "choices": [{"message": {"content": json.dumps({
            "action": action, "reason": "offline conformance fixture",
            "expected_position": None, "notes": [],
        })}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
    }


class MarsIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not SOURCE.is_file():
            if "AUTOPLAY_MARS_PROJECT" in os.environ:
                raise AssertionError("Configured AUTOPLAY_MARS_PROJECT does not contain the original engine.")
            raise unittest.SkipTest("External Mars project unavailable; configure AUTOPLAY_MARS_PROJECT to run original-engine integration checks.")
        if not Path(NODE).is_file():
            raise AssertionError("Node is required to test the configured original Mars engine.")
        cls.MarsAdapter = importlib.import_module("autoplay_qa.mars_adapter").MarsAdapter
        cls.source_bytes = SOURCE.read_bytes()
        cls.oracle = _oracle([])

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="mars-adapter-test-")
        self.addCleanup(self.temporary.cleanup)
        self.cache_dir = Path(self.temporary.name) / "bundle"

    def game(self, map_id="matrix-square", seed=0):
        game = self.MarsAdapter(
            ORIGINAL_PROJECT, map_id=map_id, seed=seed,
            node=NODE, cache_dir=self.cache_dir,
        )
        self.addCleanup(game.close)
        return game

    def assertPositionMatches(self, observation: dict, state: dict):
        self.assertEqual(observation["player"]["position"], [state["x"], state["y"]])

    def assertOriginalStateMatches(self, observation: dict, state: dict):
        """Compare exposed drone/cells to the independent engine's actual state."""
        self.assertPositionMatches(observation, state)
        self.assertEqual(observation["player"]["facing"],
                         ("North", "East", "South", "West")[state["facing"]])
        self.assertEqual(observation["player"]["variableV"], state["v"])
        self.assertEqual(observation["player"]["inventory"],
                         [f"blocks:{state['supply']}", f"bombs:{state['bombSupply']}"])
        cells = [entity for entity in observation["entities"] if entity["kind"] == "matrix-cell"]
        self.assertEqual([cell["state"]["builtHeight"] for cell in cells], state["tiles"])
        self.assertEqual([cell["state"]["bomb"] for cell in cells], state["bombs"])
        self.assertEqual([cell["state"]["visited"] for cell in cells], state["visited"])

    def test_real_source_hash_matches_independent_oracle_and_stays_unchanged(self):
        game = self.game()
        manifest = game.source_manifest
        encoded = json.dumps(manifest)
        digest = hashlib.sha256(self.source_bytes).hexdigest()
        self.assertEqual(self.oracle["source"]["sha256"], digest)
        self.assertIn(digest, encoded)
        self.assertIn("src/mars/engine.ts", encoded)
        self.assertNotIn(str(ORIGINAL_PROJECT), encoded)
        game.act("bomb")
        game.act("forward")
        game.reset()
        self.assertEqual(SOURCE.read_bytes(), self.source_bytes)

    def test_initial_drone_position_matches_all_18_original_maps(self):
        self.assertEqual(len(self.oracle["maps"]), 18)
        for original_map in self.oracle["maps"]:
            with self.subTest(map_id=original_map["id"]):
                game = self.game(original_map["id"])
                observation = game.observe()
                self.assertEqual(observation["player"]["position"],
                                 [original_map["start"]["x"], original_map["start"]["y"]])
                self.assertEqual(observation["tick"], 0)
                game.close()

    def test_turns_and_forward_follow_original_orientation(self):
        for map_id in ("matrix-square", "matrix-tall", "matrix-deploy-square", "matrix-deploy-wide"):
            for turn in ("left", "right"):
                with self.subTest(map_id=map_id, turn=turn):
                    oracle = _oracle([{"mapId": map_id, "program": [_block(turn), _block("forward")]}])["executions"][0]
                    game = self.game(map_id)
                    game.act(turn)
                    observed = game.act("forward")
                    self.assertOriginalStateMatches(observed, oracle["final"])
                    self.assertEqual(observed["tick"], 2)
                    if "error" in oracle["execution"]:
                        self.assertIn(oracle["execution"]["error"], observed["events"])
                    game.close()

    def test_boundary_block_is_feedback_not_a_host_bug(self):
        game = self.game()
        before = game.act("left")
        after = game.act("forward")
        self.assertEqual(after["player"]["position"], before["player"]["position"])
        self.assertTrue(any("blocked" in event.lower() for event in after["events"]))
        self.assertEqual(after["status"], "playing")
        self.assertEqual(step_fact(_step(before, "forward", after))["anomalies"], [])

    def test_duplicate_bomb_error_preserves_placement_and_inventory(self):
        game = self.game()
        before = game.act("bomb")
        after = game.act("bomb")
        expected = _oracle([{"mapId": "matrix-square", "program": [_block("bomb"), _block("bomb")]}])["executions"][0]
        self.assertEqual(expected["bombed"], 1)
        self.assertEqual(expected["final"]["bombSupply"], 24)
        self.assertOriginalStateMatches(after, expected["final"])
        self.assertEqual(after["player"]["inventory"], before["player"]["inventory"])
        self.assertIn(expected["execution"]["error"], after["events"])
        self.assertEqual(after["status"], "playing")
        self.assertEqual(step_fact(_step(before, "bomb", after))["anomalies"], [])

    def test_repeat_keeps_partial_progress_when_original_program_errors(self):
        game = self.game()
        before = game.observe()
        after = game.act("repeat(2,bomb)")
        expected = _oracle([{"mapId": "matrix-square", "program": [
            _block("repeat", value=2, body=[_block("bomb")]),
        ]}])["executions"][0]
        self.assertEqual(expected["execution"]["actions"], 1)
        self.assertOriginalStateMatches(after, expected["final"])
        self.assertIn(expected["execution"]["error"], after["events"])
        self.assertEqual(after["tick"], 1)
        self.assertEqual(after["status"], "playing")
        self.assertEqual(step_fact(_step(before, "repeat(2,bomb)", after))["anomalies"], [])

    def test_move_to_runs_real_intermediate_motion_without_placing_bombs(self):
        game = self.game()
        observed = game.act("moveTo(3,2)")
        expected = _oracle([{"mapId": "matrix-square", "program": [_block("moveTo", x=3, y=2)]}])["executions"][0]
        self.assertOriginalStateMatches(observed, expected["final"])
        self.assertEqual(expected["execution"]["actions"], 5)
        self.assertEqual(expected["visited"], 6)
        self.assertEqual(expected["bombed"], 0)
        for frame in expected["execution"]["frames"][1:]:
            self.assertIn(frame["message"], observed["events"])

    def test_zero_repeat_and_empty_remove_are_normal_observed_results(self):
        for action in ("repeat(0,forward)", "remove"):
            with self.subTest(action=action):
                game = self.game()
                before = game.observe()
                after = game.act(action)
                self.assertEqual(after["player"]["position"], before["player"]["position"])
                self.assertEqual(after["tick"], 1)
                self.assertEqual(after["status"], "playing")
                self.assertEqual(step_fact(_step(before, action, after))["anomalies"], [])
                game.close()

    def test_original_rock_destination_and_partially_blocked_path_match_engine(self):
        for map_id, x, y in (("m6", 2, 2), ("m8", 2, 1)):
            with self.subTest(map_id=map_id, target=(x, y)):
                expected = _oracle([{"mapId": map_id, "program": [_block("moveTo", x=x, y=y)]}])["executions"][0]
                game = self.game(map_id)
                before = game.observe()
                after = game.act(f"moveTo({x},{y})")
                self.assertOriginalStateMatches(after, expected["final"])
                self.assertIn("rock", expected["execution"]["error"].lower())
                self.assertIn(expected["execution"]["error"], after["events"])
                self.assertEqual(step_fact(_step(before, f"moveTo({x},{y})", after))["anomalies"], [])
                if map_id == "m6":
                    self.assertEqual(after["player"]["position"], before["player"]["position"])
                else:
                    self.assertEqual(after["player"]["position"], [2, 3])
                    self.assertEqual(expected["execution"]["actions"], 3)
                game.close()

    def test_place_remove_roundtrip_and_reset_restore_all_public_state(self):
        game = self.game()
        initial = game.observe()
        game.act("place")
        game.act("remove")
        game.act("bomb")
        game.act("forward")
        game.act("set(1)")
        game.reset()
        self.assertEqual(game.observe(), initial)
        game.act("bomb")
        game.reset(seed=5)
        self.assertEqual(game.observe(), initial)

    def test_observations_are_copies_and_illegal_controls_do_not_mutate_game(self):
        game = self.game()
        initial = game.observe()
        damaged = game.observe()
        damaged["player"]["position"][0] = 123
        damaged["events"].append("caller-added secret")
        if damaged.get("entities"):
            damaged["entities"][0]["state"]["callerMutation"] = True
        self.assertEqual(game.observe(), initial)
        for illegal in ("teleport", "moveTo(-1,0)", "__proto__", "repeat(999999,bomb)"):
            with self.subTest(control=illegal), self.assertRaises(ValueError):
                game.act(illegal)
            self.assertEqual(game.observe(), initial)

    def test_unknown_map_is_rejected_instead_of_falling_back_to_first_map(self):
        with self.assertRaises(ValueError):
            self.game("not-an-original-map")

    def test_seed_selection_is_repeatable_and_updates_the_control_catalog(self):
        game = self.game(map_id=None, seed=0)
        selected = []
        for seed, original_map in enumerate(self.oracle["maps"]):
            with self.subTest(seed=seed):
                game.reset(seed=seed)
                selected.append(game.map_id)
                self.assertEqual(game.map_id, original_map["id"])
                initial = game.observe()
                self.assertIn(f"moveTo({original_map['width'] - 1},{original_map['height'] - 1})", game.controls)
                game.act("right")
                game.reset()
                self.assertEqual(game.observe(), initial)
        self.assertEqual(len(set(selected)), 18)

    def test_provider_and_runner_credentials_are_not_in_the_game_child_environment(self):
        adapter_module = importlib.import_module("autoplay_qa.mars_adapter")
        original_popen = subprocess.Popen
        with patch.dict(os.environ, {"NEBIUS_API_KEY": "offline-provider-secret",
                                     "DASHBOARD_RUNNER_TOKEN": "offline-runner-secret"}):
            with patch.object(adapter_module.subprocess, "Popen", wraps=original_popen) as launch:
                game = self.game()
                child_environment = launch.call_args.kwargs["env"]
                self.assertNotIn("NEBIUS_API_KEY", child_environment)
                self.assertNotIn("DASHBOARD_RUNNER_TOKEN", child_environment)
                encoded = json.dumps(launch.call_args.args[0]) + json.dumps(game.observe())
                self.assertNotIn("offline-provider-secret", encoded)
                self.assertNotIn("offline-runner-secret", encoded)

    def test_context_exit_stops_its_engine_even_after_a_caller_exception(self):
        game = self.game()
        with self.assertRaisesRegex(RuntimeError, "test-side interruption"):
            with game:
                game.act("forward")
                raise RuntimeError("test-side interruption")
        self.assertIsNotNone(game._process.poll())
        game.close()
        with self.assertRaises(ValueError):
            game.observe()

    def test_visible_feedback_and_controls_do_not_expose_source_or_route(self):
        game = self.game()
        observation = game.act("moveTo(1,1)")
        messages = lean_messages(observation, 0, {}, [],
                                 {"actions": 23, "requests": 27, "seconds": 299}, focus="qa")
        context = json.loads(messages[1]["content"])
        view = context["observation"]
        encoded = json.dumps(messages)
        self.assertIn("bomb", view["controls"])
        self.assertIn("moveTo(1,1)", view["controls"])
        self.assertEqual(view["player"]["position"], [1, 1])
        for private in ("hint", "lesson", "rules", "program", "functions", "frames", "trace", "sourceManifest", "source_manifest", "evaluation", "injected_defect"):
            self.assertNotIn('"' + private + '"', json.dumps(view))
        self.assertNotIn(str(ORIGINAL_PROJECT), encoded)
        self.assertNotIn("src/mars/engine.ts", encoded)
        self.assertNotIn("serpentine", encoded.lower())
        self.assertNotIn("nested row/column", encoded.lower())
        self.assertNotIn("passable", encoded.lower())

    def test_lean_loop_uses_stub_http_and_records_actual_program_feedback(self):
        game = self.game()
        client = NebiusClient(Settings("offline-test-secret"))
        recording = Path(self.temporary.name) / "offline.jsonl"
        with SessionLog(recording, ("offline-test-secret",)) as log:
            begin_session(log, game, "offline-mars-integration", schema_version=2,
                          task="explore", prompt_version="lean-v1", focus="qa")
            with patch.object(client, "_request", side_effect=[_response("bomb"), _response("bomb"), _response("reset"), _response("finish")]) as transport:
                summary = run_lean_exploration(
                    game, client, log, game.reset,
                    ExplorationLimits(max_actions=4, max_requests=4),
                    evaluator=game.evaluate_transition, focus="qa",
                )
        records = [json.loads(line) for line in recording.read_text().splitlines()]
        self.assertEqual(transport.call_count, 4)
        self.assertEqual(summary["inference_requests"], 4)
        self.assertEqual(summary["resets"], 1)
        self.assertEqual(summary["steps"], 3)
        self.assertEqual(summary["status"], "finished")
        self.assertEqual(summary["warning_count"], 0)
        self.assertIsNone(summary["unassessed_final_outcome"])
        self.assertTrue(summary["usage_complete"])
        self.__class__.lean_fixture_metrics = {
            key: summary[key] for key in (
                "status", "steps", "game_actions", "resets", "inference_requests",
                "warning_count", "unassessed_final_outcome", "usage_complete",
                "request_prompt_bytes", "response_schema_bytes", "elapsed_seconds",
            )
        }
        steps = [record for record in records if record["type"] == "step"]
        self.assertEqual(len(steps), 2)
        self.assertTrue(any("already has a bomb" in event for event in steps[1]["after"]["events"]))
        self.assertEqual(extract_issues(records), [])
        for request in (record for record in records if record["type"] == "request"):
            encoded = json.dumps(request["messages"])
            self.assertNotIn(str(ORIGINAL_PROJECT), encoded)
            self.assertNotIn("serpentine", encoded.lower())
            self.assertNotIn("src/mars/engine.ts", encoded)
        self.assertNotIn("offline-test-secret", recording.read_text())


if __name__ == "__main__":
    unittest.main()
