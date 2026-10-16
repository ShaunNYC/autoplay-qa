"""Explicit response fixtures. These tests provide NO evidence of live inference."""

import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import urllib.error

from autoplay_qa.cli import begin_session
from autoplay_qa.config import ConfigurationError, Settings, read_env_file
from autoplay_qa.game import KeyGateGame
from autoplay_qa.nebius import Completion, Decision, InvalidDecision, NebiusClient, NebiusError, encode_observation
from autoplay_qa.runner import RunLimits, run_agent
from autoplay_qa.session import SessionLog, replay


def fixture_response(action="south", content=None, finish="stop"):
    return {
        "id": "unit-test-fixture", "model": "nvidia/unit-test-nemotron",
        "choices": [{"message": {"content": content if content is not None else json.dumps({"action": action, "reason": "Fixture decision", "plan": "Fixture plan"})}, "finish_reason": finish}],
        "usage": {"prompt_tokens": 20, "completion_tokens": 10, "total_tokens": 30},
    }


class ConfigurationTests(unittest.TestCase):
    def test_missing_key_fails_without_leaking_or_network(self):
        with patch.dict("os.environ", {}, clear=True):
            with self.assertRaisesRegex(ConfigurationError, "NEBIUS_API_KEY is missing"):
                Settings.load(Path("/nonexistent-autoplay-test-env"))

    def test_dotenv_is_literal_and_environment_takes_precedence(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / ".env"
            path.write_text("# config\nNEBIUS_API_KEY='file-secret'\nIGNORED=$(whoami)\n")
            self.assertEqual(read_env_file(path), {"NEBIUS_API_KEY": "file-secret"})
            with patch.dict("os.environ", {"NEBIUS_API_KEY": "environment-secret"}, clear=True):
                settings = Settings.load(path)
                self.assertEqual(settings.api_key, "environment-secret")
                self.assertNotIn("environment-secret", repr(settings))

    def test_credentials_cannot_be_sent_to_an_unrelated_host(self):
        for url in ["http://api.tokenfactory.nebius.com/v1", "https://example.com/v1", "https://api.tokenfactory.nebius.com.evil.example/v1", "https://api.tokenfactory.nebius.com/v1?key=secret"]:
            with self.subTest(url=url), patch.dict("os.environ", {"NEBIUS_API_KEY": "test-secret", "NEBIUS_BASE_URL": url}, clear=True):
                with self.assertRaises(ConfigurationError):
                    Settings.load(Path("/nonexistent-autoplay-test-env"))

    def test_model_must_be_nvidia_nemotron(self):
        with patch.dict("os.environ", {"NEBIUS_API_KEY": "test-secret", "NEBIUS_MODEL": "other/model"}, clear=True):
            with self.assertRaises(ConfigurationError):
                Settings.load(Path("/nonexistent-autoplay-test-env"))

    def test_thinking_mode_is_explicit_and_defaults_to_working_live_profile(self):
        with patch.dict("os.environ", {"NEBIUS_API_KEY": "test-secret"}, clear=True):
            self.assertTrue(Settings.load(Path("/nonexistent-autoplay-test-env")).enable_thinking)
        with patch.dict("os.environ", {"NEBIUS_API_KEY": "test-secret", "NEBIUS_ENABLE_THINKING": "false"}, clear=True):
            self.assertFalse(Settings.load(Path("/nonexistent-autoplay-test-env")).enable_thinking)


class NebiusTests(unittest.TestCase):
    def test_observation_encoding_reports_local_tiles_without_mutating_snapshot(self):
        observation = KeyGateGame(0).observe()
        encoded = encode_observation(observation)
        self.assertFalse(encoded["neighbors"]["north"]["passable"])
        self.assertTrue(encoded["neighbors"]["south"]["passable"])
        self.assertEqual(encoded["neighbors"]["east"]["destination"], {"x": 2, "y": 1})
        self.assertEqual(encoded["named_positions"]["key"], {"x": 1, "y": 3})
        self.assertNotIn("neighbors", observation)
        self.assertNotIn("next_action", encoded)

    def test_http_request_is_openai_compatible_and_includes_schema(self):
        client = NebiusClient(Settings("unit-test-key"))
        with patch.object(client._opener, "open", return_value=io.BytesIO(json.dumps(fixture_response()).encode())) as transport:
            completion = client.complete([{"role": "user", "content": "observation"}])
        request = transport.call_args.args[0]
        body = json.loads(request.data)
        self.assertEqual(request.full_url, "https://api.tokenfactory.nebius.com/v1/chat/completions")
        self.assertEqual(request.get_header("Authorization"), "Bearer unit-test-key")
        self.assertEqual(body["model"], "nvidia/Nemotron-3_5-Lightning")
        self.assertEqual(body["response_format"]["type"], "json_schema")
        self.assertFalse(body["stream"])
        self.assertEqual(body["chat_template_kwargs"], {"enable_thinking": True})
        self.assertEqual(completion.metadata["usage"]["total_tokens"], 30)
        self.assertEqual(Decision.parse(completion.content, ["south"]).action, "south")

    def test_http_401_is_actionable_and_redacts_provider_body(self):
        client = NebiusClient(Settings("unit-test-key"))
        failure = urllib.error.HTTPError("url", 401, "error", {}, io.BytesIO(b"unit-test-key"))
        with patch.object(client._opener, "open", side_effect=failure):
            with self.assertRaisesRegex(NebiusError, "Authentication failed") as context:
                client.list_models()
        self.assertNotIn("unit-test-key", str(context.exception))

    def test_network_error_has_no_hidden_retry(self):
        client = NebiusClient(Settings("unit-test-key"))
        with patch.object(client._opener, "open", side_effect=urllib.error.URLError("DNS error")) as transport:
            with self.assertRaisesRegex(NebiusError, "connection failed"):
                client.list_models()
        self.assertEqual(transport.call_count, 1)

    def test_malformed_response_and_refusal_are_rejected(self):
        client = NebiusClient(Settings("unit-test-key"))
        for value in [{"choices": []}, {"choices": [{"message": {"refusal": "cannot"}}]}]:
            with self.subTest(value=value), patch.object(client, "_request", return_value=value):
                with self.assertRaises(NebiusError):
                    client.complete([])

    def test_inline_thinking_is_separated_from_the_final_action(self):
        client = NebiusClient(Settings("unit-test-key", enable_thinking=True))
        answer = json.dumps({"action": "south", "reason": "Final explanation", "plan": "Final plan"})
        response = fixture_response(content='A trace may mention {"action":"east"}.</think>\n' + answer)
        with patch.object(client, "_request", return_value=response) as transport:
            completion = client.complete([])
        self.assertEqual(Decision.parse(completion.content, ["south"]).action, "south")
        self.assertEqual(completion.content, answer)
        self.assertTrue(completion.metadata["discarded_inline_thinking"])
        self.assertTrue(transport.call_args.args[1]["chat_template_kwargs"]["enable_thinking"])

    def test_invalid_action_and_embedded_json_cannot_be_executed(self):
        for value in ["bad", '{"action":"teleport","reason":"x","plan":"y"}', '{"action":"south","reason":"x","plan":"y","extra":1}', 'prefix {"action":"south","reason":"x","plan":"y"}', '{"action":[],"reason":"x","plan":"y"}']:
            with self.subTest(value=value), self.assertRaises(InvalidDecision):
                Decision.parse(value, ["south"])


class RunnerTests(unittest.TestCase):
    def run_fixture(self, responses, limits=RunLimits(), error=None):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        path = Path(directory.name) / "test-only.jsonl"
        game = KeyGateGame(0)
        client = NebiusClient(Settings("unit-test-key"))
        with SessionLog(path, ("unit-test-key",)) as log:
            begin_session(log, game, "unit-test-fixtures")
            with patch.object(client, "_request", side_effect=error or responses) as transport:
                summary = run_agent(game, client, log, limits)
        records = [json.loads(line) for line in path.read_text().splitlines()]
        return game, summary, records, path, transport

    def test_multiple_actions_use_fresh_observation_and_persistent_plan(self):
        game, summary, records, path, transport = self.run_fixture([fixture_response(), fixture_response()], RunLimits(max_steps=2))
        self.assertEqual(game.position, (1, 3))
        self.assertTrue(game.has_key)
        self.assertEqual(summary["steps"], 2)
        self.assertEqual(summary["usage"]["total_tokens"], 60)
        contexts = [json.loads(record["messages"][1]["content"]) for record in records if record["type"] == "request"]
        self.assertEqual(contexts[0]["observation"]["tick"], 0)
        self.assertEqual(contexts[1]["observation"]["tick"], 1)
        self.assertEqual(contexts[1]["observation"]["player"]["position"], [1, 2])
        self.assertEqual(contexts[1]["previous_plan"], "Fixture plan")
        self.assertEqual(contexts[1]["recent_actions_and_outcomes"][0]["action"], "south")
        self.assertEqual(replay(path)["actions_replayed"], 2)

    def test_invalid_response_gets_one_repair_without_advancing_game(self):
        _, summary, records, _, _ = self.run_fixture([fixture_response(content="bad"), fixture_response()], RunLimits(max_steps=1))
        self.assertEqual(summary["steps"], 1)
        self.assertEqual(summary["inference_requests"], 2)
        self.assertEqual(summary["rejected_decisions"], 1)
        contexts = [json.loads(record["messages"][1]["content"]) for record in records if record["type"] == "request"]
        self.assertEqual(contexts[0]["observation"], contexts[1]["observation"])
        self.assertIn("validation_feedback", contexts[1])

    def test_repeated_invalid_decisions_stop_with_zero_actions(self):
        game, summary, _, _, _ = self.run_fixture([fixture_response(content="bad")] * 2)
        self.assertEqual(game.tick, 0)
        self.assertEqual(summary["status"], "invalid_decision")
        self.assertEqual(summary["inference_requests"], 2)

    def test_truncated_completion_is_not_executed(self):
        game, summary, _, _, _ = self.run_fixture([fixture_response(finish="length")] * 2)
        self.assertEqual(game.tick, 0)
        self.assertEqual(summary["status"], "invalid_decision")

    def test_free_text_thinking_is_not_persisted_as_an_answer(self):
        _, _, records, path, _ = self.run_fixture([fixture_response(content="unit-test private trace", finish="length")] * 2)
        self.assertNotIn("unit-test private trace", path.read_text())
        completions = [record for record in records if record["type"] == "completion"]
        self.assertTrue(all(record["answer"] is None for record in completions))
        self.assertTrue(all(record["answer_characters"] > 0 for record in completions))

    def test_request_budget_stops_invalid_repairs(self):
        game, summary, _, _, _ = self.run_fixture([fixture_response(content="bad")], RunLimits(max_requests=1))
        self.assertEqual(summary["status"], "request_limit")
        self.assertEqual(game.tick, 0)

    def test_api_failure_is_recorded_without_game_action(self):
        game, summary, records, _, transport = self.run_fixture([], error=NebiusError("Authentication failed"))
        self.assertEqual(summary["status"], "inference_error")
        self.assertEqual(game.tick, 0)
        self.assertEqual(transport.call_count, 1)
        self.assertEqual(records[-1]["type"], "summary")

    def test_fixture_win_stops_immediately_and_replays(self):
        actions = ["south", "south", "east", "east", "interact", "east", "east", "east", "east"]
        _, summary, _, path, transport = self.run_fixture([fixture_response(action) for action in actions])
        self.assertEqual(summary["status"], "won")
        self.assertEqual(transport.call_count, 9)
        self.assertEqual(replay(path)["status"], "won")

    def test_replay_detects_modified_observation(self):
        _, _, records, path, _ = self.run_fixture([fixture_response()], RunLimits(max_steps=1))
        step = next(record for record in records if record["type"] == "step")
        step["after"]["player"]["position"] = [3, 5]
        path.write_text("\n".join(json.dumps(record) for record in records))
        with self.assertRaisesRegex(ValueError, "diverged after"):
            replay(path)

    def test_recording_redacts_echoed_credentials(self):
        _, _, _, path, _ = self.run_fixture([fixture_response(content=json.dumps({"action": "south", "reason": "unit-test-key", "plan": "plan"}))], RunLimits(max_steps=1))
        self.assertNotIn("unit-test-key", path.read_text())
        self.assertIn("[REDACTED]", path.read_text())

    def test_replay_checks_summary_and_identifies_incomplete_logs(self):
        _, _, records, path, _ = self.run_fixture([fixture_response()], RunLimits(max_steps=1))
        records[-1]["steps"] = 99
        path.write_text("\n".join(json.dumps(record) for record in records))
        with self.assertRaisesRegex(ValueError, "session summary"):
            replay(path)
        path.write_text("\n".join(json.dumps(record) for record in records[:-1]))
        result = replay(path)
        self.assertEqual(result["actions_replayed"], 1)
        self.assertFalse(result["recording_complete"])

    def test_response_after_deadline_is_logged_but_not_executed(self):
        def late_response(*_args):
            clock[0] = 10.0
            return fixture_response()

        clock = [0.0]
        # Patch the shared monotonic clock, not sleep or a real billed request.
        with patch("time.monotonic", side_effect=lambda: clock[0]):
            game, summary, records, _, _ = self.run_fixture(late_response, RunLimits(max_seconds=1))
        self.assertEqual(summary["status"], "time_limit")
        self.assertEqual(game.tick, 0)
        self.assertEqual(summary["inference_requests"], 1)
        self.assertTrue(any(record["type"] == "completion" for record in records))


if __name__ == "__main__":
    unittest.main()
