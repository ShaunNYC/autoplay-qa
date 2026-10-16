"""Deterministic offline benchmark coverage."""

import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from autoplay_qa.benchmark import (
    SCRIPTED_SEEDS,
    build_scripted_recording,
    format_benchmark_table,
    run_scripted_benchmark,
)
from autoplay_qa.cli import main
from autoplay_qa.game import DEFECT_CHOICES
from autoplay_qa.reporting import evaluate_recording


class ScriptedPolicyTests(unittest.TestCase):
    def test_raw_policy_hits_every_defect_and_keeps_clean_controls_clean(self):
        for defect in DEFECT_CHOICES:
            for seed in SCRIPTED_SEEDS:
                with self.subTest(defect=defect, seed=seed):
                    records = build_scripted_recording(seed, defect)
                    controls = [
                        record for record in records
                        if record.get("type") in {"step", "reset"}
                    ]
                    self.assertEqual(
                        sum(record["type"] == "reset" for record in controls), 1
                    )
                    self.assertTrue(all(
                        "software_fact" not in record for record in controls
                    ))
                    self.assertTrue(all(
                        record["decision"]["expected_position"] is None
                        for record in controls
                    ))

                    result = evaluate_recording(
                        records, verify_reproduction=False
                    )
                    if defect == "none":
                        self.assertEqual(result["detection"], "clean")
                        self.assertEqual(result["issue_count"], 0)
                        self.assertEqual(result["false_positive_count"], 0)
                    else:
                        self.assertEqual(result["detection"], "hit")
                        self.assertGreater(result["issue_count"], 0)
                        self.assertEqual(result["false_positive_count"], 0)

    def test_runner_invokes_report_reproduce_and_evaluate(self):
        evaluation = {
            "detection": "hit",
            "hit": True,
            "false_positive_count": 0,
            "repro_verified": True,
            "issue_count": 1,
        }
        with patch(
            "autoplay_qa.benchmark.generate_report", return_value="# report\n"
        ) as report, patch(
            "autoplay_qa.benchmark.extract_issues",
            return_value=[{"number": 1}],
        ), patch(
            "autoplay_qa.benchmark.reproduce_issue",
            return_value={"reproduced": True},
        ) as reproduce, patch(
            "autoplay_qa.benchmark.evaluate_recording", return_value=evaluation
        ) as evaluate:
            result = run_scripted_benchmark(seeds=(0,), defects=("wall-clip",))

        self.assertTrue(result["passed"])
        report.assert_called_once()
        self.assertFalse(report.call_args.kwargs["minimize"])
        reproduce.assert_called_once()
        self.assertFalse(reproduce.call_args.kwargs["minimize"])
        evaluate.assert_called_once()

    def test_record_dir_writes_full_recording_and_ground_truth_report(self):
        records = [
            {
                "type": "session",
                "game_version": "fixture-v1",
                "seed": 5,
                "initial_observation": {},
                "evaluation": {
                    "injected_defect": {"name": "floor-gap"},
                },
            },
            {"type": "sweep_step", "step": 1, "source": "sweep"},
            {"type": "request", "request": 1, "source": "llm"},
            {"type": "response", "request": 1, "source": "llm"},
            {"type": "summary", "status": "finished"},
        ]
        evaluation = {
            "detection": "hit",
            "hit": True,
            "false_positive_count": 0,
            "repro_verified": True,
            "issue_count": 0,
        }
        report_text = (
            "# AutoPlay QA bug report\n\n"
            "Ground truth: `floor-gap`\n\n"
            "## Appendix: Observations (not issues)\n"
        )
        with tempfile.TemporaryDirectory() as directory, patch(
            "autoplay_qa.benchmark.build_scripted_recording",
            return_value=records,
        ), patch(
            "autoplay_qa.benchmark.extract_issues", return_value=[]
        ), patch(
            "autoplay_qa.benchmark.generate_report", return_value=report_text
        ) as report, patch(
            "autoplay_qa.benchmark.evaluate_recording", return_value=evaluation
        ):
            result = run_scripted_benchmark(
                seeds=(5,),
                defects=("floor-gap",),
                record_dir=Path(directory),
            )
            recording_path = Path(directory) / "floor-gap-seed5.jsonl"
            report_path = Path(directory) / "floor-gap-seed5.md"
            serialized = recording_path.read_text(encoding="utf-8")

            self.assertTrue(serialized.endswith("\n"))
            self.assertEqual(
                [json.loads(line) for line in serialized.splitlines()],
                records,
            )
            self.assertEqual(report_path.read_text(encoding="utf-8"), report_text)
            self.assertEqual(result["artifacts"], [{
                "defect": "floor-gap",
                "seed": 5,
                "recording": str(recording_path.resolve()),
                "report": str(report_path.resolve()),
            }])

        self.assertTrue(report.call_args.kwargs["with_ground_truth"])
        self.assertEqual(report.call_args.kwargs["reproduction_results"], [])


class BenchmarkCliTests(unittest.TestCase):
    def test_scripted_cli_is_offline_and_prints_requested_columns(self):
        result = {
            "mode": "scripted",
            "seeds": [0],
            "defects": ["none", "wall-clip"],
            "passed": True,
            "rows": [
                {
                    "defect": "none",
                    "seed": 0,
                    "detection": "clean",
                    "false_positive_count": 0,
                    "repro_verified": False,
                },
                {
                    "defect": "wall-clip",
                    "seed": 0,
                    "detection": "hit",
                    "false_positive_count": 0,
                    "repro_verified": True,
                },
            ],
        }
        with patch(
            "autoplay_qa.benchmark.run_scripted_benchmark",
            return_value=result,
        ) as run, patch(
            "autoplay_qa.cli.Settings.load",
            side_effect=AssertionError("benchmark must not load credentials"),
        ) as settings, patch(
            "autoplay_qa.nebius.NebiusClient._request",
            side_effect=AssertionError("benchmark must not call inference"),
        ) as request, patch("sys.stdout", new_callable=io.StringIO) as output:
            self.assertEqual(main(["benchmark", "--scripted"]), 0)

        run.assert_called_once_with()
        settings.assert_not_called()
        request.assert_not_called()
        rendered = output.getvalue()
        self.assertIn("hit/miss", rendered)
        self.assertIn("FP", rendered)
        self.assertIn("repro verified", rendered)
        self.assertIn("wall-clip", rendered)
        self.assertIn("Scripted benchmark: PASS", rendered)

    def test_unity_filters_and_record_dir_are_forwarded_and_paths_printed(self):
        with tempfile.TemporaryDirectory() as directory:
            record_dir = Path(directory) / "diagnostics"
            recording = record_dir / "floor-gap-seed1.jsonl"
            report = record_dir / "floor-gap-seed1.md"
            result = {
                "mode": "scripted",
                "seeds": [1, 5],
                "defects": ["floor-gap"],
                "passed": True,
                "rows": [{
                    "defect": "floor-gap",
                    "seed": 1,
                    "detection": "hit",
                    "false_positive_count": 0,
                    "repro_verified": True,
                }],
                "artifacts": [{
                    "defect": "floor-gap",
                    "seed": 1,
                    "recording": str(recording),
                    "report": str(report),
                }],
            }
            game_factory = object()
            with patch(
                "autoplay_qa.cli.unity_game_factory",
                return_value=game_factory,
            ), patch(
                "autoplay_qa.benchmark.run_scripted_benchmark",
                return_value=result,
            ) as run, patch(
                "sys.stdout", new_callable=io.StringIO
            ) as output:
                exit_code = main([
                    "benchmark",
                    "--scripted",
                    "--unity-player", "/fixture/player",
                    "--defects", "floor-gap",
                    "--seeds", "1,5",
                    "--record-dir", str(record_dir),
                ])

        self.assertEqual(exit_code, 0)
        run.assert_called_once_with(
            defects=("floor-gap",),
            seeds=(1, 5),
            record_dir=record_dir,
            game_factory=game_factory,
            sweep_actions=400,
        )
        rendered = output.getvalue()
        self.assertIn(f"Recording: {recording}", rendered)
        self.assertIn(f"Report: {report}", rendered)

    def test_filter_parsers_reject_empty_or_unknown_values(self):
        cases = (
            ["benchmark", "--scripted", "--defects", "floor-gap,"],
            ["benchmark", "--scripted", "--defects", "not-a-defect"],
            ["benchmark", "--scripted", "--seeds", "1,nope"],
        )
        for arguments in cases:
            with self.subTest(arguments=arguments), patch(
                "sys.stderr", new_callable=io.StringIO
            ), self.assertRaises(SystemExit) as error:
                main(arguments)
            self.assertEqual(error.exception.code, 2)

    def test_table_marks_clean_reproduction_not_applicable(self):
        result = {
            "passed": True,
            "rows": [{
                "defect": "none",
                "seed": 5,
                "detection": "clean",
                "false_positive_count": 0,
                "repro_verified": False,
            }],
        }

        table = format_benchmark_table(result)

        self.assertIn("none", table)
        self.assertIn("clean", table)
        self.assertIn("n/a", table)


if __name__ == "__main__":
    unittest.main()
