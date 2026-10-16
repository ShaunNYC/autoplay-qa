"""Offline lifecycle and CLI coverage for the managed Unity player launcher."""

from __future__ import annotations

import argparse
import io
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import Mock, patch

from autoplay_qa.cli import (
    _evaluation_metadata,
    _recording_game_factory,
    begin_session,
    main,
)
from autoplay_qa.session import SessionLog
from autoplay_qa.unity_launcher import (
    UNITY_BENCHMARK_DEFECTS,
    UnityPlayerGame,
    UnityPlayerLaunchError,
    unity_game_factory,
)


class _FakeProcess:
    def __init__(self):
        self.returncode = None
        self.terminate_calls = 0
        self.kill_calls = 0
        self.wait_calls = []

    def poll(self):
        return self.returncode

    def terminate(self):
        self.terminate_calls += 1

    def kill(self):
        self.kill_calls += 1

    def wait(self, timeout=None):
        self.wait_calls.append(timeout)
        self.returncode = -15 if not self.kill_calls else -9
        return self.returncode


class _FakeSocket:
    def __init__(self):
        self.timeouts = []

    def settimeout(self, value):
        self.timeouts.append(value)


class _FakeAdapter:
    version = "physics-key-door-v1"
    game_id = "physics-key-door"
    controls = ["north", "interact"]
    capabilities = {"continuous": True, "solids_visible": True}

    def __init__(self, endpoint, *, timeout, seed):
        self.endpoint = endpoint
        self.timeout = timeout
        self.seed = seed
        self._socket = _FakeSocket()
        self.closed = False
        self.actions = []

    def observe(self):
        return {
            "status": "playing",
            "player": {"position": [0.0, 1.0, 0.0], "inventory": []},
            "entities": [],
            "events": [],
            "controls": list(self.controls),
        }

    def act(self, action):
        self.actions.append(action)
        return self.observe()

    def reset(self, seed=None):
        if seed is not None:
            self.seed = seed
        return self

    def evaluate_transition(self, before, action, after):
        return {"moved": False, "events": [], "contract_violations": []}

    def close(self):
        self.closed = True


class UnityLauncherTests(unittest.TestCase):
    def executable(self, directory: str) -> Path:
        path = Path(directory) / "AutoPlayQA-Physics"
        path.write_text("fixture", encoding="utf-8")
        path.chmod(path.stat().st_mode | 0o100)
        return path

    def test_launch_uses_process_only_defect_and_closes_adapter_then_process(self):
        with tempfile.TemporaryDirectory() as directory:
            player = self.executable(directory)
            process = _FakeProcess()
            adapter = _FakeAdapter("tcp://127.0.0.1:43210", timeout=0.1, seed=5)
            with patch(
                "autoplay_qa.unity_launcher._reserve_loopback_port",
                return_value=43210,
            ), patch(
                "autoplay_qa.unity_launcher.subprocess.Popen",
                return_value=process,
            ) as popen, patch(
                "autoplay_qa.unity_launcher.RemoteGameAdapter",
                return_value=adapter,
            ) as remote:
                game = UnityPlayerGame(
                    player,
                    seed=5,
                    defect="door-ghost-collider",
                    timeout=7,
                )

                command = popen.call_args.args[0]
                self.assertEqual(command[:3], [str(player.resolve()), "-batchmode", "-nographics"])
                self.assertEqual(command[command.index("-autoplayqa-port") + 1], "43210")
                self.assertEqual(command[command.index("-autoplayqa-seed") + 1], "5")
                self.assertEqual(
                    command[command.index("-autoplayqa-defect") + 1],
                    "door-ghost-collider",
                )
                remote.assert_called_once()
                self.assertEqual(remote.call_args.args, ("tcp://127.0.0.1:43210",))
                self.assertEqual(remote.call_args.kwargs["seed"], 5)
                self.assertNotIn("defect", remote.call_args.kwargs)
                self.assertNotIn(
                    "door-ghost-collider", json.dumps(game.session_metadata)
                )
                self.assertEqual(
                    game.session_metadata,
                    {
                        "unity_player": {
                            "path": str(player.resolve()),
                            "version": "physics-key-door-v1",
                        }
                    },
                )

                game.close()

            self.assertTrue(adapter.closed)
            self.assertEqual(process.terminate_calls, 1)
            self.assertEqual(process.kill_calls, 0)
            self.assertEqual(adapter._socket.timeouts, [7.0])

    def test_clean_launch_omits_defect_argument(self):
        with tempfile.TemporaryDirectory() as directory:
            player = self.executable(directory)
            process = _FakeProcess()
            with patch(
                "autoplay_qa.unity_launcher._reserve_loopback_port",
                return_value=40001,
            ), patch(
                "autoplay_qa.unity_launcher.subprocess.Popen",
                return_value=process,
            ) as popen, patch(
                "autoplay_qa.unity_launcher.RemoteGameAdapter",
                side_effect=lambda endpoint, timeout, seed: _FakeAdapter(
                    endpoint, timeout=timeout, seed=seed
                ),
            ):
                game = UnityPlayerGame(player, seed=0)
                game.close()
            self.assertNotIn("-autoplayqa-defect", popen.call_args.args[0])

    def test_startup_timeout_terminates_a_still_running_child(self):
        with tempfile.TemporaryDirectory() as directory:
            player = self.executable(directory)
            process = _FakeProcess()
            with patch(
                "autoplay_qa.unity_launcher._reserve_loopback_port",
                return_value=40002,
            ), patch(
                "autoplay_qa.unity_launcher.subprocess.Popen",
                return_value=process,
            ), patch(
                "autoplay_qa.unity_launcher.time.monotonic",
                side_effect=[0.0, 0.2],
            ):
                with self.assertRaisesRegex(
                    UnityPlayerLaunchError, "Timed out waiting"
                ):
                    UnityPlayerGame(player, seed=1, startup_timeout=0.1)
            self.assertEqual(process.terminate_calls, 1)

    def test_connection_is_retried_until_the_player_is_ready(self):
        with tempfile.TemporaryDirectory() as directory:
            player = self.executable(directory)
            process = _FakeProcess()
            adapter = _FakeAdapter("tcp://127.0.0.1:40003", timeout=1, seed=2)
            with patch(
                "autoplay_qa.unity_launcher._reserve_loopback_port",
                return_value=40003,
            ), patch(
                "autoplay_qa.unity_launcher.subprocess.Popen",
                return_value=process,
            ), patch(
                "autoplay_qa.unity_launcher.RemoteGameAdapter",
                side_effect=[OSError("not listening"), adapter],
            ) as remote, patch(
                "autoplay_qa.unity_launcher.time.monotonic",
                side_effect=[0.0, 0.01, 0.02],
            ), patch("autoplay_qa.unity_launcher.time.sleep") as sleep:
                game = UnityPlayerGame(player, seed=2, timeout=4, startup_timeout=1)
                game.close()

            self.assertEqual(remote.call_count, 2)
            self.assertEqual(
                [call.args[0] for call in remote.call_args_list],
                ["tcp://127.0.0.1:40003", "tcp://127.0.0.1:40003"],
            )
            sleep.assert_called_once()
            self.assertEqual(adapter._socket.timeouts, [4.0])

    def test_close_kills_child_when_graceful_termination_times_out(self):
        process = _FakeProcess()
        waits = iter((subprocess.TimeoutExpired("player", 3), -9))

        def wait(timeout=None):
            result = next(waits)
            if isinstance(result, BaseException):
                raise result
            process.returncode = result
            return result

        process.wait = wait
        with patch("autoplay_qa.unity_launcher._terminate_process") as terminate:
            # Build without starting a real process; close must still delegate
            # cleanup after closing the adapter.
            game = object.__new__(UnityPlayerGame)
            game._closed = False
            game._adapter = _FakeAdapter("fixture", timeout=1, seed=0)
            game._process = process
            game.close()
            terminate.assert_called_once_with(process)

        # Exercise the bounded TERM -> KILL fallback itself.
        from autoplay_qa.unity_launcher import _terminate_process

        process.returncode = None
        _terminate_process(process)
        self.assertEqual(process.terminate_calls, 1)
        self.assertEqual(process.kill_calls, 1)

    def test_factory_relaunches_with_requested_recording_seed_and_defect(self):
        factory = unity_game_factory("/fixture/player", timeout=9, startup_timeout=12)
        sentinel = object()
        with patch(
            "autoplay_qa.unity_launcher.UnityPlayerGame", return_value=sentinel
        ) as launch:
            self.assertIs(factory(5, "floor-gap"), sentinel)
        launch.assert_called_once_with(
            Path("/fixture/player"),
            seed=5,
            defect="floor-gap",
            timeout=9,
            startup_timeout=12,
        )

    def test_supported_matrix_is_clean_plus_five_unity_defects(self):
        self.assertEqual(
            UNITY_BENCHMARK_DEFECTS,
            (
                "none",
                "floor-gap",
                "door-ghost-collider",
                "key-not-consumed",
                "respawn-leak",
                "wall-tunnel",
            ),
        )


class UnityLauncherCliTests(unittest.TestCase):
    def recording(self, directory: str, player: Path) -> Path:
        path = Path(directory) / "unity.jsonl"
        header = {
            "type": "session",
            "game_version": "physics-key-door-v1",
            "seed": 5,
            "unity_player": {
                "path": str(player),
                "version": "physics-key-door-v1",
            },
            "evaluation": {"injected_defect": {"name": "floor-gap"}},
        }
        path.write_text(json.dumps(header) + "\n", encoding="utf-8")
        return path

    def test_begin_session_records_player_path_and_version(self):
        class Game:
            version = "physics-key-door-v1"
            seed = 5
            session_metadata = {
                "unity_player": {
                    "path": "/fixture/AutoPlayQA-Physics",
                    "version": "physics-key-door-v1",
                }
            }

            def observe(self):
                return {"status": "playing"}

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "session.jsonl"
            with SessionLog(path) as log:
                begin_session(log, Game(), "fixture")
            header = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
        self.assertEqual(header["unity_player"], Game.session_metadata["unity_player"])

    def test_recording_factory_reuses_stored_player_path(self):
        with tempfile.TemporaryDirectory() as directory:
            player = Path(directory) / "player"
            recording = self.recording(directory, player)
            args = argparse.Namespace(
                game=None,
                unity_player=None,
                game_timeout=6,
                unity_start_timeout=11,
            )
            sentinel = object()
            with patch(
                "autoplay_qa.cli.unity_game_factory", return_value=sentinel
            ) as factory:
                self.assertIs(
                    _recording_game_factory(args, [recording]), sentinel
                )
            factory.assert_called_once_with(
                player.resolve(), timeout=6, startup_timeout=11
            )

    def test_artifact_commands_forward_recorded_player_factory(self):
        with tempfile.TemporaryDirectory() as directory:
            player = Path(directory) / "player"
            recording = self.recording(directory, player)
            sentinel = Mock(name="recorded-unity-factory")
            cases = (
                (
                    ["report", str(recording)],
                    "autoplay_qa.reporting.generate_report",
                    "# fixture report\n",
                ),
                (
                    ["reproduce", str(recording), "--issue", "1"],
                    "autoplay_qa.reporting.reproduce_issue",
                    {"exit_code": 0},
                ),
                (
                    ["evaluate", str(recording)],
                    "autoplay_qa.reporting.evaluate_recordings",
                    {},
                ),
                (
                    ["scoreboard", str(recording), "--no-scripted-baseline"],
                    "autoplay_qa.scoreboard.generate_scoreboard",
                    "# fixture scoreboard\n",
                ),
            )
            for argv, target, returned in cases:
                with self.subTest(command=argv[0]), patch(
                    "autoplay_qa.cli.unity_game_factory",
                    return_value=sentinel,
                ), patch(target, return_value=returned) as artifact, patch(
                    "sys.stdout", new_callable=io.StringIO
                ):
                    self.assertEqual(main(argv), 0)
                    self.assertIs(
                        artifact.call_args.kwargs["game_factory"], sentinel
                    )

    def test_continuous_defect_is_only_in_evaluation_metadata(self):
        args = argparse.Namespace(
            game=None,
            unity_player=Path("/fixture/player"),
            defect="key-not-consumed",
        )
        metadata = _evaluation_metadata(args)
        self.assertEqual(
            metadata["evaluation"]["injected_defect"]["name"],
            "key-not-consumed",
        )

    def test_cli_rejects_game_and_unity_player_together(self):
        with patch("sys.stderr", new_callable=io.StringIO) as errors:
            code = main([
                "evaluate",
                "fixture.jsonl",
                "--game",
                "tcp://127.0.0.1:1234",
                "--unity-player",
                "/fixture/player",
            ])
        self.assertEqual(code, 2)
        self.assertIn("either --game or --unity-player", errors.getvalue())

    def test_scripted_benchmark_accepts_unity_defect_and_uses_launcher_factory(self):
        sentinel_factory = Mock(name="unity-factory")
        result = {"passed": True, "rows": [], "mode": "scripted"}
        with patch(
            "autoplay_qa.cli.unity_game_factory",
            return_value=sentinel_factory,
        ) as factory, patch(
            "autoplay_qa.benchmark.run_scripted_benchmark",
            return_value=result,
        ) as benchmark, patch(
            "autoplay_qa.benchmark.format_benchmark_table",
            return_value="fixture table",
        ), patch("sys.stdout", new_callable=io.StringIO):
            code = main([
                "benchmark",
                "--scripted",
                "--unity-player",
                "/fixture/player",
            ])
        self.assertEqual(code, 0)
        factory.assert_called_once_with(
            Path("/fixture/player"), timeout=5, startup_timeout=10
        )
        benchmark.assert_called_once_with(
            defects=UNITY_BENCHMARK_DEFECTS,
            game_factory=sentinel_factory,
            sweep_actions=400,
        )


if __name__ == "__main__":
    unittest.main()
