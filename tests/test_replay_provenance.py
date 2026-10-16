"""Recording integrity fixtures; no inference or hypothesis verification."""

import copy
import json
from pathlib import Path
import tempfile
import unittest

from autoplay_qa.exploration import exploration_view
from autoplay_qa.game import KeyGateGame, evaluate_transition
from autoplay_qa.session import replay


def recording(actions, *, schema=2):
    game = KeyGateGame(0)
    header = {
        "type": "session", "schema_version": schema,
        "game_version": game.version, "seed": game.seed,
        "mode": "unit-test-fixtures", "initial_observation": game.observe(),
    }
    if schema == 2:
        header["task"] = "explore"
    records = [header]
    episode = steps = resets = 0
    for request, action in enumerate(actions, 1):
        before = game.observe()
        if schema == 2:
            records.extend([
                {
                    "type": "request", "request_number": request,
                    "episode": episode, "observation_tick": before["tick"],
                    "messages": [
                        {"role": "system", "content": "Unit-test recording fixture"},
                        {"role": "user", "content": json.dumps({"episode": episode, "observation": exploration_view(before)})},
                    ],
                },
                {"type": "completion", "request_number": request, "answer": json.dumps({"action": action})},
                {"type": "memory", "request_number": request, "episode": episode, "hypotheses": []},
            ])
        if action == "finish":
            records.append({"type": "finish", "request_number": request, "episode": episode, "decision": {"action": action}, "observation": before})
            break
        steps += 1
        if action == "reset":
            game = KeyGateGame(0)
            after = game.observe()
            record = {"type": "reset", "episode_before": episode, "episode_after": episode + 1}
            episode += 1
            resets += 1
        else:
            after = game.act(action)
            record = {"type": "step", "episode": episode, "evaluation": evaluate_transition(before, action, after)}
        record.update(step=steps, request_number=request, decision={"action": action}, before=before, after=after)
        records.append(record)
    summary = {"type": "summary", "status": "finished" if actions[-1] == "finish" else "action_limit", "steps": steps, "final_observation": game.observe()}
    if schema == 2:
        summary.update(resets=resets, game_actions=steps - resets, episodes=episode + 1, inference_requests=len(actions))
    records.append(summary)
    return records


class ReplayProvenanceTests(unittest.TestCase):
    def replay_records(self, records):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "unit-test-recording.jsonl"
            path.write_text("\n".join(json.dumps(item) for item in records) + "\n", encoding="utf-8")
            return replay(path)

    def test_valid_controls_keep_request_links_across_reset(self):
        result = self.replay_records(recording(["north", "reset", "interact", "finish"]))
        self.assertEqual(result["actions_replayed"], 3)
        self.assertEqual(result["resets_replayed"], 1)
        self.assertEqual(result["exploration_status"], "finished")
        self.assertFalse(result["learning_claims_verified"])

    def test_each_control_rejects_stale_future_missing_or_noninteger_request(self):
        original = recording(["north", "reset", "finish"])
        for kind in ("step", "reset", "finish"):
            for wrong in (0, 99, None, True, 1.0, "1"):
                records = copy.deepcopy(original)
                target = next(item for item in records if item["type"] == kind)
                if wrong is None:
                    target.pop("request_number")
                else:
                    target["request_number"] = wrong
                with self.subTest(kind=kind, request=wrong), self.assertRaisesRegex(ValueError, f"{kind} request association"):
                    self.replay_records(records)

    def test_reset_and_finish_cannot_reference_a_previous_real_request(self):
        for kind in ("reset", "finish"):
            records = recording(["north", "reset", "finish"])
            target = next(item for item in records if item["type"] == kind)
            target["request_number"] -= 1
            with self.subTest(kind=kind), self.assertRaisesRegex(ValueError, f"{kind} request association"):
                self.replay_records(records)

    def test_control_without_a_request_is_rejected(self):
        records = recording(["north"])
        records = [item for item in records if item["type"] not in {"request", "completion", "memory"}]
        with self.assertRaisesRegex(ValueError, "step request association"):
            self.replay_records(records)

    def test_boolean_request_number_cannot_impersonate_first_integer_request(self):
        records = recording(["north"])
        target = next(item for item in records if item["type"] == "request")
        target["request_number"] = True
        with self.assertRaisesRegex(ValueError, "request accounting"):
            self.replay_records(records)

    def test_one_request_cannot_execute_a_second_game_reset_or_finish_control(self):
        for second_action in ("interact", "reset", "finish"):
            records = recording(["north", second_action])
            records = [item for item in records if not (item.get("request_number") == 2 and item["type"] in {"request", "completion", "memory"})]
            for item in records:
                if item["type"] in {"step", "reset", "finish"} and item.get("request_number") == 2:
                    item["request_number"] = 1
            records[-1]["inference_requests"] = 1
            with self.subTest(action=second_action), self.assertRaisesRegex(ValueError, "multiple controls for request 1"):
                self.replay_records(records)

    def test_completion_and_memory_must_belong_to_current_request(self):
        for kind in ("completion", "memory"):
            records = recording(["north", "finish"])
            target = next(item for item in records if item["type"] == kind and item["request_number"] == 2)
            target["request_number"] = 1
            with self.subTest(kind=kind), self.assertRaisesRegex(ValueError, f"{kind} request association"):
                self.replay_records(records)

    def test_repair_requests_can_execute_one_control_after_a_rejection(self):
        records = recording(["north", "finish"])
        start = copy.deepcopy(records[1])
        records.insert(1, start)
        records.insert(2, {"type": "completion", "request_number": 1, "answer": None})
        records.insert(3, {"type": "rejected_decision", "request_number": 1, "error": "Unit-test rejection"})
        for item in records[4:]:
            if "request_number" in item:
                item["request_number"] += 1
        records[-1]["inference_requests"] += 1
        result = self.replay_records(records)
        self.assertEqual(result["actions_replayed"], 1)
        self.assertEqual(result["exploration_status"], "finished")

    def test_schema_one_preserves_legacy_request_number_behavior(self):
        records = recording(["north", "interact"], schema=1)
        for item in records:
            if item["type"] == "step":
                item.pop("request_number")
        result = self.replay_records(records)
        self.assertEqual(result["actions_replayed"], 2)
        self.assertTrue(result["recording_complete"])


if __name__ == "__main__":
    unittest.main()
