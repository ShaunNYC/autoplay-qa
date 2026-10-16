"""Offline tests for the Phase-3a live-results scoreboard."""

import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from autoplay_qa.cli import main
from autoplay_qa.scoreboard import (
    _retained_suspected_bug_ids,
    build_scoreboard,
    findings_paragraph,
    format_scoreboard,
    score_recording,
    scripted_baseline_row,
)


def write_fixture(directory: str) -> Path:
    path = Path(directory) / "fixture.jsonl"
    suspected_one = {
        "id": "bug-one",
        "kind": "suspected_bug",
        "claim": "Accepted transient flag.",
        "status": "uncertain",
        "evidence_steps": [1],
    }
    rows = [
        {"type": "session", "schema_version": 2, "seed": 0},
        {"type": "request", "request_number": 1, "decision_mode": "thinking"},
        {
            "type": "completion", "request_number": 1,
            "decision_mode": "thinking", "latency_ms": 100,
            "usage": {"total_tokens": 5},
        },
        {"type": "memory", "hypotheses": [suspected_one]},
        {
            "type": "step", "step": 1,
            "decision": {"action": "wait", "notes": [suspected_one]},
        },
        {
            "type": "request", "request_number": 2,
            "decision_mode": "thinking", "attempt_status": "failed",
            "failure_reason": "timeout",
        },
        {
            "type": "request", "request_number": 3,
            "decision_mode": "direct_retry", "attempt_status": "completed",
        },
        {
            "type": "completion", "request_number": 3,
            "decision_mode": "direct_retry", "latency_ms": 20,
            "usage": {"total_tokens": 7},
        },
        {"type": "reset", "step": 2, "decision": {"action": "reset", "notes": []}},
        {
            "type": "summary", "steps": 2, "game_actions": 1,
            "resets": 1, "inference_requests": 3,
            "usage": {"total_tokens": 12}, "usage_complete": False,
            "elapsed_seconds": 2.5, "hypotheses": [],
        },
    ]
    path.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )
    return path


def evaluation(*, defect="gate-keeps-key", detection="hit"):
    return {
        "injected_defect": defect,
        "detection": detection,
        "false_positive_count": 0,
        "matched_issue_numbers": [1] if detection == "hit" else [],
    }


class RecordingScoreTests(unittest.TestCase):
    def test_counts_budgets_modes_and_final_retained_bug_notes(self):
        with tempfile.TemporaryDirectory() as directory:
            path = write_fixture(directory)
            with patch(
                "autoplay_qa.scoreboard.evaluate_recording",
                return_value=evaluation(),
            ) as evaluate, patch(
                "autoplay_qa.scoreboard.reproduce_issue",
                return_value={"reproduced": True, "minimized_action_count": 5},
            ) as reproduce:
                row = score_recording(path)

        evaluate.assert_called_once()
        self.assertFalse(evaluate.call_args.kwargs["verify_reproduction"])
        reproduce.assert_called_once()
        self.assertEqual(row["recording"], "fixture.jsonl")
        self.assertEqual(row["detection"], "hit")
        self.assertTrue(row["repro_verified"])
        self.assertEqual(row["minimized_steps"], 5)
        self.assertEqual((row["actions"], row["resets"], row["requests"]), (1, 1, 3))
        self.assertEqual(row["tokens"], 12)
        self.assertFalse(row["usage_complete"])
        self.assertEqual(row["seconds"], 2.5)
        self.assertEqual(row["decision_mode_counts"], {"thinking": 2, "direct_retry": 1})
        # The transient accepted proposal is not the model's final judgment.
        self.assertEqual(row["model_suspected_bug_count"], 0)
        self.assertEqual(row["failed_requests"], 1)
        self.assertEqual(row["completions"], 2)

    def test_partial_recording_falls_back_to_last_memory_snapshot(self):
        bug = {"id": "retained", "kind": "suspected_bug"}
        records = [
            {"type": "session"},
            {"type": "memory", "hypotheses": [bug]},
            {"type": "memory", "hypotheses": [bug, {"id": "m", "kind": "mechanic"}]},
        ]

        self.assertEqual(_retained_suspected_bug_ids(records), {"retained"})

    def test_markdown_has_required_columns_and_marks_incomplete_tokens(self):
        row = {
            "recording": "fixture|name.jsonl", "defect": "clean",
            "detection": "clean", "false_positive_count": 0,
            "repro_verified": None, "minimized_steps": None,
            "actions": 1, "resets": 0, "requests": 1,
            "tokens": 12, "usage_complete": False, "seconds": 1.25,
            "decision_mode_counts": {"thinking": 1},
            "model_suspected_bug_count": 0, "failed_requests": 0,
            "completion_latencies_ms": [100], "interior_wall_attempts": 0,
            "scripted": False,
        }
        rendered = format_scoreboard({
            "rows": [row], "live_rows": [row], "scripted_baseline": None,
        })

        self.assertIn("| recording | defect | detection | FP | repro verified |", rendered)
        self.assertIn("model suspected_bug count", rendered)
        self.assertIn("fixture\\|name.jsonl", rendered)
        self.assertIn("12+", rendered)
        self.assertIn("thinking=1", rendered)
        self.assertIn("known token minimum", rendered)


class ScriptedBaselineTests(unittest.TestCase):
    def test_aggregates_seed_zero_benchmark_into_one_comparison_row(self):
        benchmark = {
            "defects": ["none", "wall-clip"],
            "passed": True,
            "rows": [
                {
                    "defect": "none", "detection": "clean",
                    "false_positive_count": 0, "repro_verified": False,
                },
                {
                    "defect": "wall-clip", "detection": "hit",
                    "false_positive_count": 0, "repro_verified": True,
                },
            ],
        }
        controls = [
            {"type": "session"}, {"type": "step"},
            {"type": "reset"}, {"type": "summary"},
        ]
        with patch(
            "autoplay_qa.benchmark.run_scripted_benchmark",
            return_value=benchmark,
        ) as run, patch(
            "autoplay_qa.benchmark.build_scripted_recording",
            return_value=controls,
        ) as build:
            row = scripted_baseline_row()

        run.assert_called_once_with(seeds=(0,))
        self.assertEqual(build.call_count, 2)
        self.assertEqual(row["recording"], "scripted baseline (seed 0)")
        self.assertEqual(row["detection"], "1/1 hit")
        self.assertTrue(row["repro_verified"])
        self.assertEqual((row["actions"], row["resets"], row["requests"]), (2, 2, 0))


class FindingsTests(unittest.TestCase):
    def test_reports_hits_coverage_miss_clean_fp_model_role_and_transport(self):
        def live(defect, detection, *, steps=None, modes=None, walls=1):
            return {
                "defect": defect, "detection": detection,
                "minimized_steps": steps, "false_positive_count": 0,
                "decision_mode_counts": modes or {}, "requests": sum((modes or {}).values()),
                "completion_latencies_ms": [1000], "failed_requests": 0,
                "model_suspected_bug_count": 0,
                "interior_wall_attempts": walls,
            }

        rows = [
            live("reset-leak", "hit", steps=3, modes={"thinking": 2}),
            live("gate-keeps-key", "hit", steps=5, modes={"direct_retry": 1}),
            live("exit-no-win", "hit", steps=9, modes={"direct_fallback": 2}),
            live("phantom-block", "hit", steps=8, modes={"direct_latency": 1}),
            live("wall-clip", "miss", walls=0),
            live("clean", "clean"),
            live("clean", "clean"),
        ]
        baseline = {
            "benchmark_hits": 5, "benchmark_defects": 5,
            "false_positive_count": 0,
        }
        findings = findings_paragraph({
            "live_rows": rows, "scripted_baseline": baseline,
        })

        self.assertIn("4 of 5 injected defects", findings)
        self.assertIn("never attempted a move into an interior wall", findings)
        self.assertIn("2 clean controls produced 0 false positives", findings)
        self.assertIn("thinking 2", findings)
        self.assertIn("direct_retry 1", findings)
        self.assertIn("direct_fallback 2", findings)
        self.assertIn("direct_latency 1", findings)
        self.assertIn("LLM contributed exploration but not judgment", findings)
        self.assertIn("0 final suspected_bug notes", findings)
        self.assertIn("scripted baseline detected 5 of 5 defects", findings)


class ScoreboardCliTests(unittest.TestCase):
    def test_cli_is_offline_and_can_write_markdown(self):
        with tempfile.TemporaryDirectory() as directory:
            source = write_fixture(directory)
            destination = Path(directory) / "results.md"
            with patch(
                "autoplay_qa.scoreboard.evaluate_recording",
                return_value=evaluation(defect=None, detection="clean"),
            ), patch(
                "autoplay_qa.cli.Settings.load",
                side_effect=AssertionError("scoreboard must not load credentials"),
            ) as settings, patch(
                "autoplay_qa.nebius.NebiusClient._request",
                side_effect=AssertionError("scoreboard must not call inference"),
            ) as request, patch("sys.stdout", new_callable=io.StringIO) as output:
                code = main([
                    "scoreboard", str(source), "--no-scripted-baseline",
                    "--out", str(destination),
                ])

            self.assertEqual(code, 0)
            settings.assert_not_called()
            request.assert_not_called()
            self.assertTrue(destination.exists())
            rendered = destination.read_text(encoding="utf-8")
            self.assertIn("fixture.jsonl", rendered)
            self.assertIn("## Findings", rendered)
            self.assertIn("Wrote QA scoreboard:", output.getvalue())

    def test_build_requires_at_least_one_recording(self):
        with self.assertRaisesRegex(ValueError, "At least one recording"):
            build_scoreboard([], include_scripted_baseline=False)


if __name__ == "__main__":
    unittest.main()
