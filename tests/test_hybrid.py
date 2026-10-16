"""Inference-free hybrid sweep and lean-context integration."""

from __future__ import annotations

import io
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

from autoplay_qa.cli import _write_sweep_phase, begin_session, main
from autoplay_qa.config import Settings
from autoplay_qa.exploration import ExplorationLimits
from autoplay_qa.game import KeyGateGame, defect_evaluation_metadata
from autoplay_qa.hybrid import (
    compact_sweep_summary,
    run_collision_sweep,
)
from autoplay_qa.lean import LEAN_PROMPT_VERSION, lean_messages, run_lean_exploration
from autoplay_qa.nebius import Completion
from autoplay_qa.reporting import evaluate_recording, extract_issues, generate_report
from autoplay_qa.scoreboard import score_recording
from autoplay_qa.session import SessionLog


class _GridAliasAdapter:
    """Expose the built-in game through the protocol's generic grid alias."""

    capabilities = {"grid": True, "legend": True}
    controls = ["north", "south", "east", "west", "interact", "wait"]

    def __init__(self, game):
        self.game = game

    @staticmethod
    def _generic(observation):
        value = dict(observation)
        value["grid"] = value.pop("map")
        value.pop("actions", None)
        return value

    def observe(self):
        return self._generic(self.game.observe())

    def act(self, control):
        return self._generic(self.game.act(control))


class _NoGridAdapter:
    capabilities = {"grid": False, "legend": False}
    controls = ["north", "south", "east", "west"]

    def __init__(self):
        self.actions = 0

    def observe(self):
        return {
            "status": "playing",
            "player": {"position": [0, 0], "inventory": []},
            "events": [],
        }

    def act(self, _control):
        self.actions += 1
        raise AssertionError("A capability-skipped sweep must not act.")


class _StubClient:
    def __init__(self, decisions):
        self.decisions = iter(decisions)
        self.settings = SimpleNamespace(enable_thinking=False)
        self.timeout = 45
        self.max_tokens = 8192
        self.temperature = 1.0
        self.top_p = 0.95

    def complete(self, messages, **_kwargs):
        self.last_messages = messages
        decision = next(self.decisions)
        return Completion(
            json.dumps(decision),
            {
                "latency_ms": 1,
                "finish_reason": "stop",
                "usage": {
                    "prompt_tokens": 1,
                    "completion_tokens": 1,
                    "total_tokens": 2,
                },
            },
        )


def _decision(action):
    return {
        "action": action,
        "reason": "offline hybrid fixture",
        "expected_position": None,
        "notes": [],
    }


class CollisionSweepTests(unittest.TestCase):
    def test_wall_clip_is_found_for_required_seeds_and_clean_has_no_anomaly(self):
        for seed in (0, 1, 5):
            with self.subTest(seed=seed, defect="wall-clip"):
                result = run_collision_sweep(
                    KeyGateGame(seed, "wall-clip"), max_actions=400
                )
                self.assertLessEqual(result.summary["actions"], 400)
                self.assertEqual(result.summary["anomaly_count"], 1)
                self.assertIn(
                    "entered_wall_labeled_tile",
                    {item["aspect"] for item in result.summary["anomalies"]},
                )
                self.assertTrue(result.records)
                self.assertTrue(all(
                    record["type"] == "sweep_step"
                    and record["source"] == "sweep"
                    for record in result.records
                ))
                serialized = json.dumps(result.summary)
                self.assertNotIn("wall-clip", serialized)
                self.assertNotIn("defect", serialized.lower())

            with self.subTest(seed=seed, defect="none"):
                result = run_collision_sweep(
                    KeyGateGame(seed), max_actions=400
                )
                self.assertEqual(result.summary["anomaly_count"], 0)
                self.assertEqual(result.summary["anomalies"], [])

    def test_action_budget_is_a_hard_bound(self):
        result = run_collision_sweep(
            KeyGateGame(0, "wall-clip"), max_actions=5
        )

        self.assertEqual(result.summary["actions"], 5)
        self.assertEqual(len(result.records), 5)
        self.assertEqual(result.summary["status"], "action_limit")

    def test_grid_alias_and_advertised_capabilities_are_supported(self):
        result = run_collision_sweep(
            _GridAliasAdapter(KeyGateGame(5, "wall-clip")),
            max_actions=400,
        )

        self.assertEqual(result.summary["anomaly_count"], 1)
        self.assertEqual(
            result.summary["anomalies"][0]["aspect"],
            "entered_wall_labeled_tile",
        )

    def test_missing_grid_capability_is_a_zero_action_skip(self):
        adapter = _NoGridAdapter()

        result = run_collision_sweep(adapter, max_actions=400)

        self.assertEqual(result.summary["status"], "capability_missing")
        self.assertEqual(result.summary["actions"], 0)
        self.assertEqual(result.records, [])
        self.assertEqual(adapter.actions, 0)


class HybridLeanContextTests(unittest.TestCase):
    def test_cli_closes_sweep_adapter_before_opening_llm_adapter(self):
        state = {"active": 0, "maximum": 0, "created": 0}

        class TrackedGame:
            version = "key-gate-v1"

            def __init__(self):
                self.inner = KeyGateGame(0)
                self.seed = 0
                self.controls = [
                    "north", "south", "east", "west", "interact", "wait",
                ]
                self.closed = False
                state["created"] += 1
                state["active"] += 1
                state["maximum"] = max(state["maximum"], state["active"])

            def observe(self):
                return self.inner.observe()

            def act(self, control):
                return self.inner.act(control)

            def reset(self):
                self.inner.reset()
                return self

            def evaluate_transition(self, before, action, after):
                return self.inner.evaluate_transition(before, action, after)

            def close(self):
                if not self.closed:
                    self.closed = True
                    state["active"] -= 1

        client = _StubClient([_decision("finish")])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "hybrid-cli.jsonl"
            with patch(
                "autoplay_qa.cli.Settings.load",
                return_value=Settings("unit-key"),
            ), patch(
                "autoplay_qa.cli.NebiusClient",
                return_value=client,
            ), patch(
                "autoplay_qa.cli._create_game",
                side_effect=lambda _args: TrackedGame(),
            ), patch("sys.stdout", new_callable=io.StringIO):
                exit_code = main([
                    "explore", "--focus", "qa", "--hybrid",
                    "--log", str(path),
                ])

        self.assertEqual(exit_code, 0)
        self.assertEqual(state["created"], 2)
        self.assertEqual(state["maximum"], 1)
        self.assertEqual(state["active"], 0)

    def test_default_messages_are_unchanged_when_no_sweep_is_supplied(self):
        observation = KeyGateGame(0).observe()
        arguments = (
            observation,
            0,
            {},
            [],
            {"actions": 24, "requests": 28, "seconds": 300},
        )

        implicit = lean_messages(*arguments, focus="qa")
        explicit = lean_messages(*arguments, focus="qa", sweep_summary=None)

        self.assertEqual(implicit, explicit)
        self.assertNotIn("sweep_coverage", json.loads(implicit[1]["content"]))

    def test_prompt_summary_is_whitelisted_and_contains_no_ground_truth(self):
        sweep = run_collision_sweep(
            KeyGateGame(0, "wall-clip"), max_actions=400
        ).summary
        sweep["evaluation"] = {"injected_defect": "SECRET_DEFECT"}
        sweep["anomalies"][0]["private_rule"] = "SECRET_RULE"

        messages = lean_messages(
            KeyGateGame(0).observe(),
            0,
            {},
            [],
            {"actions": 24, "requests": 28, "seconds": 300},
            focus="qa",
            sweep_summary=sweep,
        )
        context = json.loads(messages[1]["content"])

        self.assertEqual(
            context["sweep_coverage"], compact_sweep_summary(sweep)
        )
        serialized = json.dumps(messages)
        self.assertNotIn("SECRET_DEFECT", serialized)
        self.assertNotIn("SECRET_RULE", serialized)
        self.assertNotIn("evaluation", context["sweep_coverage"])

    def test_lean_phase_keeps_its_budget_and_gets_llm_source_labels(self):
        game = KeyGateGame(0)
        sweep = run_collision_sweep(
            KeyGateGame(0, "wall-clip"), max_actions=400
        ).summary
        client = _StubClient([_decision("wait"), _decision("finish")])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "hybrid.jsonl"
            with SessionLog(path) as log:
                result = run_lean_exploration(
                    game,
                    client,
                    log,
                    lambda: KeyGateGame(0),
                    ExplorationLimits(
                        max_actions=24,
                        max_requests=28,
                        max_seconds=300,
                    ),
                    focus="qa",
                    sleep=lambda _seconds: None,
                    sweep_summary=sweep,
                )
            records = [
                json.loads(line)
                for line in path.read_text(encoding="utf-8").splitlines()
            ]

        request = next(record for record in records if record["type"] == "request")
        context = json.loads(request["messages"][1]["content"])
        step = next(record for record in records if record["type"] == "step")
        finish = next(record for record in records if record["type"] == "finish")

        self.assertEqual(result["steps"], 1)
        self.assertEqual(result["inference_requests"], 2)
        self.assertEqual(result["source"], "llm")
        self.assertEqual(result["sweep_coverage"], compact_sweep_summary(sweep))
        self.assertIn("sweep_coverage", context)
        self.assertEqual(step["source"], "llm")
        self.assertEqual(finish["source"], "llm")

    def test_offline_hybrid_recordings_hit_wall_clip_and_keep_clean_zero_fp(self):
        rows = []
        for defect in ("none", "wall-clip"):
            for seed in (0, 1, 5):
                with self.subTest(defect=defect, seed=seed):
                    game = KeyGateGame(seed, defect)
                    sweep = run_collision_sweep(
                        KeyGateGame(seed, defect), max_actions=400
                    )
                    client = _StubClient([_decision("finish")])
                    with tempfile.TemporaryDirectory() as directory:
                        path = Path(directory) / "hybrid.jsonl"
                        with SessionLog(path) as log:
                            begin_session(
                                log,
                                game,
                                "offline-stub",
                                schema_version=2,
                                task="explore",
                                focus="qa",
                                prompt_version=LEAN_PROMPT_VERSION,
                                limits={"history_size": 8},
                                hybrid=True,
                                **defect_evaluation_metadata(defect),
                            )
                            _write_sweep_phase(log, sweep)
                            run_lean_exploration(
                                game,
                                client,
                                log,
                                lambda: KeyGateGame(seed, defect),
                                ExplorationLimits(24, 28, 300),
                                focus="qa",
                                sleep=lambda _seconds: None,
                                sweep_summary=sweep.summary,
                            )
                        evaluation = evaluate_recording(path)
                        issues = extract_issues(path)
                        report = generate_report(path) if defect == "wall-clip" and seed == 0 else None
                        score = score_recording(path) if defect == "wall-clip" and seed == 0 else None

                    rows.append({
                        "defect": defect,
                        "seed": seed,
                        "detection": evaluation["detection"],
                        "false_positives": evaluation["false_positive_count"],
                    })
                    if defect == "none":
                        self.assertEqual(evaluation["detection"], "clean")
                        self.assertEqual(evaluation["false_positive_count"], 0)
                        self.assertEqual(issues, [])
                    else:
                        self.assertEqual(evaluation["detection"], "hit")
                        self.assertTrue(evaluation["repro_verified"])
                        self.assertEqual(len(issues), 1)
                        self.assertEqual(issues[0]["phase_sources"], ["sweep"])
                        if seed == 0:
                            self.assertIn("Phase source: `sweep`.", report)
                            self.assertEqual(score["phase_control_counts"], {"sweep": 44, "llm": 0})
                            self.assertEqual(score["actions"], 44)
        self.assertEqual(len(rows), 6)


if __name__ == "__main__":
    unittest.main()
