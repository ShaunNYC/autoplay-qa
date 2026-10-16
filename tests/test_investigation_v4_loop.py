"""Repair feedback and deadline admission across the actual loop and replay."""

import copy
import json
from types import SimpleNamespace
from unittest.mock import Mock, patch
import unittest

from autoplay_qa.adaptive_replay import AdaptiveReplay
from autoplay_qa.game import KeyGateGame
from autoplay_qa.lean import software_facts
from autoplay_qa.session import replay
from tests import test_investigation_integration as integration_tests
from tests.test_lean_exploration import decision, response


class InvestigationV4LoopTests(unittest.TestCase):
    fixture = integration_tests.InvestigationIntegrationTests.fixture

    def test_parse_failure_is_visible_to_next_response_and_does_not_block_control(self):
        bad = decision("east")
        bad["predictions"] = [{"path": "player.position.0", "relation": "equals", "value_json": "2"}]
        repaired = decision("west")
        repaired["predictions"] = [{"path": "player.position", "relation": "changes", "value_json": None}]
        result, path, rows, http = self.fixture([
            response(bad), response(repaired), response(decision("finish")),
        ], version="adaptive-v4")
        self.assertEqual((result["steps"], result["rejected_decisions"], http.call_count), (2, 0, 3))
        contexts = [json.loads(row["messages"][1]["content"]) for row in rows if row["type"] == "request"]
        feedback = contexts[1]["metadata_feedback"]
        self.assertEqual(feedback["request_number"], 1)
        self.assertIn("invalid_prediction", [issue["code"] for issue in feedback["issues"]])
        self.assertTrue(feedback["public_selector_examples"])
        self.assertEqual(contexts[2]["metadata_feedback"]["issues"], [])
        self.assertTrue(replay(path)["adaptive_facts_validated"])

    def test_v4_tampered_prediction_or_feedback_receipts_fail_replay(self):
        for target in ("trial", "feedback"):
            with self.subTest(target=target):
                result, path, rows, _ = self.fixture([
                    response(decision("east")), response(decision("finish")),
                ], version="adaptive-v4")
                changed = copy.deepcopy(rows)
                if target == "trial":
                    next(row for row in changed if row["type"] == "step")["investigation_fact"]["prediction_checks"] = [{"forged": True}]
                else:
                    request = [row for row in changed if row["type"] == "request"][1]
                    context = json.loads(request["messages"][1]["content"])
                    context["metadata_feedback"]["request_number"] = 999
                    request["messages"][1]["content"] = json.dumps(context)
                path.write_text("\n".join(json.dumps(row) for row in changed) + "\n")
                with self.assertRaises(ValueError):
                    replay(path)

    def test_insufficient_time_stops_before_http_and_preserves_replay(self):
        for times in ([0, 0, 265], [0, 0, 250, 265]):
            clock = SimpleNamespace(monotonic=Mock(side_effect=[*times, *([265] * 100)]))
            with self.subTest(times=times), patch("autoplay_qa.lean.time", clock):
                result, path, rows, http = self.fixture([], version="adaptive-v4")
            self.assertEqual((result["steps"], result["inference_requests"], http.call_count), (0, 0, 0))
            self.assertEqual(result["status"], "time_limit")
            self.assertTrue(result["request_admission_stopped"])
            self.assertIsNone(result["unassessed_final_outcome"])
            stop = next(row for row in rows if row["type"] == "request_admission_stop")
            self.assertFalse(stop["dispatched"])
            self.assertEqual(stop["required_seconds"], 46)
            self.assertTrue(replay(path)["recording_complete"])
            stop["remaining_seconds"] = 60
            path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
            with self.assertRaises(ValueError):
                replay(path)

    def test_v3_deadline_policy_remains_compatible(self):
        clock = SimpleNamespace(monotonic=Mock(side_effect=[0, 0, *([265] * 100)]))
        with patch("autoplay_qa.lean.time", clock):
            result, path, rows, http = self.fixture([response(decision("finish"))])
        self.assertEqual(result["status"], "finished")
        self.assertEqual(http.call_count, 1)
        self.assertFalse(any(row["type"] == "request_admission_stop" for row in rows))
        self.assertTrue(replay(path)["recording_complete"])

    def test_terminal_stop_rejects_reuse_of_a_previously_rejected_request(self):
        clock = SimpleNamespace(monotonic=Mock(side_effect=[*([0] * 6), *([265] * 100)]))
        with patch("autoplay_qa.lean.time", clock):
            result, path, rows, http = self.fixture(
                [response(decision("illegal"))], version="adaptive-v4",
            )
        self.assertEqual((result["steps"], result["rejected_decisions"], http.call_count), (0, 1, 1))
        self.assertTrue(result["request_admission_stopped"])
        self.assertTrue(replay(path)["recording_complete"])

        _, _, valid_rows, _ = self.fixture([
            response(decision("east")), response(decision("finish")),
        ], version="adaptive-v4")
        forged_tail = [copy.deepcopy(row) for row in valid_rows
                       if row.get("request_number") == 1
                       and row["type"] in {"completion", "memory", "step"}]
        # Reuse the old request number and recompute every derived summary so
        # only the terminal event ordering exposes the forgery.
        checker = AdaptiveReplay(rows[0])
        game = KeyGateGame(0)
        for row in rows[1:-1]:
            if row["type"] in {"request", "unused_context"}:
                checker.request(json.loads(row["messages"][1]["content"]), game.observe(), row)
            elif row["type"] == "completion":
                checker.completion(row)
            elif row["type"] == "request_admission_stop":
                checker.admission_stop(row)
        checker.completion(forged_tail[0])
        checker.memory(forged_tail[1], 0)
        checker.control(forged_tail[2])
        game.act("east")
        checker.executed(forged_tail[2])
        summary = copy.deepcopy(rows[-1])
        summary.update(steps=1, game_actions=1, final_observation=game.observe(),
                       adaptive=checker.context.summary(), coverage=checker.context.summary())
        summary.update(software_facts([forged_tail[2]], history_size=None,
                                     include_non_qa_reset_anomalies=True))
        forged = [*rows[:-1], *forged_tail, summary]
        path.write_text("\n".join(json.dumps(row) for row in forged) + "\n")
        with self.assertRaisesRegex(ValueError, "after terminal request admission stop"):
            replay(path)


if __name__ == "__main__":
    unittest.main()
