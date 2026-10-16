"""Phase 2 fixtures for lean QA notes and software-derived anomaly facts."""

import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from autoplay_qa.cli import begin_session
from autoplay_qa.config import Settings
from autoplay_qa.exploration import ExplorationLimits
from autoplay_qa.game import KeyGateGame, defect_evaluation_metadata
from autoplay_qa.lean import (
    LEAN_PROMPT_VERSION,
    LeanDecision,
    LeanNote,
    _merge_notes,
    lean_messages,
    lean_schema,
    note_dict,
    run_lean_exploration,
    software_facts,
    step_fact,
)
from autoplay_qa.nebius import NebiusClient
from autoplay_qa.session import SessionLog, replay


def _decision(action, expected=None, notes=()):
    return {
        "action": action,
        "reason": "phase 2 fixture",
        "expected_position": expected,
        "notes": list(notes),
    }


def _response(value):
    return {
        "id": "phase-2-fixture",
        "model": "unit-test-model",
        "choices": [{"message": {"content": json.dumps(value)}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    }


def _record(game, action, *, step=1, episode=0, expected=None, record_type="step"):
    before = game.observe()
    after = game.act(action) if record_type == "step" else game.reset().observe()
    return {
        "type": record_type,
        "step": step,
        "episode": episode,
        "episode_before": episode,
        "episode_after": episode + (record_type == "reset"),
        "before": before,
        "after": after,
        "decision": _decision(action, expected),
        "qa_mode": True,
    }


class Phase2NoteTests(unittest.TestCase):
    def test_schema_kind_is_optional_defaults_to_mechanic_and_stays_tiny(self):
        schema = lean_schema()
        encoded = json.dumps(schema, separators=(",", ":")).encode("utf-8")
        note_schema = schema["properties"]["notes"]["items"]

        self.assertLess(len(encoded), 1_200)
        self.assertEqual(len(encoded), 867)
        self.assertEqual(note_schema["properties"]["kind"]["enum"], ["mechanic", "suspected_bug"])
        self.assertNotIn("kind", note_schema["required"])

        parsed, warnings = LeanDecision.parse(
            json.dumps(_decision("wait", notes=[{
                "id": "n1", "claim": "Waiting advances feedback", "status": "uncertain", "evidence_steps": [],
            }])),
            ["wait"],
            set(),
        )
        self.assertEqual(parsed.notes[0].kind, "mechanic")
        self.assertEqual(note_dict(parsed.notes[0])["kind"], "mechanic")
        self.assertEqual(warnings, [])

        parsed, warnings = LeanDecision.parse(
            json.dumps(_decision("wait", notes=[{
                "id": "bug", "claim": "Odd result", "status": "uncertain",
                "kind": "verdict", "evidence_steps": [],
            }])),
            ["wait"],
            set(),
        )
        self.assertEqual(parsed.notes, ())
        self.assertEqual([warning["code"] for warning in warnings], ["invalid_note_kind"])

    def test_prompt_shows_slots_and_tells_model_to_reuse_ids(self):
        notes = {
            f"n{index}": LeanNote(f"n{index}", f"claim {index}", "supported", (), "mechanic")
            for index in range(6)
        }
        messages = lean_messages(
            KeyGateGame(0).observe(),
            0,
            notes,
            [],
            {"actions": 4, "requests": 5, "seconds": 30},
            focus="qa",
        )
        context = json.loads(messages[1]["content"])

        self.assertEqual(context["note_slots_remaining"], 0)
        self.assertEqual(context["anomalies"], [])
        self.assertIn("reuse an existing id", messages[0]["content"].lower())
        self.assertIn("reuse an existing id", context["response_contract"]["notes"].lower())
        self.assertIn("qa playtester", context["session_question"].lower())

    def test_full_memory_replaces_oldest_uncertain_then_drops_without_one(self):
        retained = {
            "oldest": LeanNote("oldest", "old", "uncertain", ()),
            "fixed": LeanNote("fixed", "fixed", "supported", ()),
            "newer": LeanNote("newer", "newer", "uncertain", ()),
            "n3": LeanNote("n3", "three", "supported", ()),
            "n4": LeanNote("n4", "four", "contradicted", ()),
            "n5": LeanNote("n5", "five", "supported", ()),
        }
        merged, warnings = _merge_notes(
            retained,
            (LeanNote("replacement", "replacement", "supported", (), "suspected_bug"),),
        )

        self.assertNotIn("oldest", merged)
        self.assertIn("newer", merged)
        self.assertEqual(merged["replacement"].kind, "suspected_bug")
        self.assertEqual([warning["code"] for warning in warnings], ["note_memory_replaced"])

        no_uncertain = {
            identifier: LeanNote(identifier, item.claim, "supported", item.evidence_steps, item.kind)
            for identifier, item in retained.items()
        }
        merged, warnings = _merge_notes(
            no_uncertain,
            (LeanNote("dropped", "drop", "uncertain", ()),),
        )
        self.assertNotIn("dropped", merged)
        self.assertEqual([warning["code"] for warning in warnings], ["note_memory_full"])


class Phase2AnomalyTests(unittest.TestCase):
    def test_prediction_and_floor_block_are_rule_free_step_facts(self):
        game = KeyGateGame(0, "phantom-block")
        game.position = (3, 1)  # The seeded phantom tile is immediately south at [3,2].
        record = _record(game, "south", expected=[3, 2])

        fact = step_fact(record)
        types = [item["type"] for item in fact["anomalies"]]
        self.assertIn("prediction_surprise", types)
        floor = next(item for item in fact["anomalies"] if item.get("aspect") == "blocked_on_floor_labeled_tile")
        self.assertEqual(floor["expected"]["legend_meaning"], "floor")
        self.assertEqual(floor["actual"]["position"], [3, 1])
        serialized = json.dumps(fact)
        self.assertNotIn("phantom-block", serialized)
        self.assertNotIn("defect", serialized.lower())

    def test_wall_entry_is_a_map_inconsistency(self):
        game = KeyGateGame(0, "wall-clip")
        game.position = (3, 1)  # The seeded walkable wall is east at [4,1].
        record = _record(game, "east", expected=[4, 1])

        fact = step_fact(record)
        wall = next(item for item in fact["anomalies"] if item.get("aspect") == "entered_wall_labeled_tile")
        self.assertEqual(wall["expected"]["legend_meaning"], "wall")
        self.assertEqual(wall["actual"]["position"], [4, 1])

    def test_reset_difference_and_cross_episode_difference_are_factual(self):
        game = KeyGateGame(0, "reset-leak")
        first = _record(game, "south", step=1, expected=[1, 2])
        second = _record(game, "south", step=2, expected=[1, 3])
        reset = _record(game, "reset", step=3, episode=0, record_type="reset")
        reset["decision"] = _decision("reset", [1, 1])
        facts = software_facts([first, second, reset], history_size=None)
        reset_fact = next(item for item in facts["anomalies"] if item["type"] == "reset_inconsistency")
        self.assertIn("map", reset_fact["different_fields_ignoring_tick"])
        self.assertEqual(reset_fact["actual"]["facets"]["inventory"], ["key"])

        original_game = KeyGateGame(0)
        original = _record(original_game, "east", step=1, episode=0, expected=[2, 1])
        repeated = copy.deepcopy(original)
        repeated.update(step=2, episode=1, episode_before=1, episode_after=1)
        repeated["after"] = copy.deepcopy(repeated["before"])
        repeated["after"]["tick"] = 1
        repeated["after"]["events"] = ["Movement blocked by a wall or locked gate."]
        nondeterministic = software_facts([original, repeated], history_size=None)
        cross = next(item for item in nondeterministic["anomalies"] if item["type"] == "cross_episode_nondeterminism")
        self.assertEqual(cross["evidence_steps"], [1, 2])
        self.assertEqual(cross["prior"]["facets"]["position"], [2, 1])
        self.assertEqual(cross["actual"]["facets"]["position"], [1, 1])

    def test_non_qa_facts_keep_the_phase1_shape(self):
        game = KeyGateGame(0)
        record = _record(game, "east", expected=[2, 1])
        record.pop("qa_mode")
        facts = software_facts([record], history_size=None)

        self.assertNotIn("anomalies", facts)
        self.assertNotIn("anomalies", facts["factual_log"][0])


class Phase2ControllerTests(unittest.TestCase):
    def test_qa_reset_uses_supplied_bound_factory_and_records_inconsistency(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        path = Path(directory.name) / "qa-reset.jsonl"
        game = KeyGateGame(0, "reset-leak")
        client = NebiusClient(Settings("unit-key"))
        replies = [
            _response(_decision("south", [1, 2])),
            _response(_decision("south", [1, 3])),
            _response(_decision("reset", [1, 1])),
            _response(_decision("finish")),
        ]
        forbidden_factory_calls = 0

        def bound_reset():
            nonlocal forbidden_factory_calls
            forbidden_factory_calls += 1
            return game.reset()

        with SessionLog(path, ("unit-key",)) as log:
            begin_session(
                log, game, "phase-2", schema_version=2, task="explore",
                focus="qa", prompt_version=LEAN_PROMPT_VERSION,
                **defect_evaluation_metadata("reset-leak"),
            )
            with patch.object(client, "_request", side_effect=replies):
                summary = run_lean_exploration(
                    game,
                    client,
                    log,
                    bound_reset,
                    ExplorationLimits(max_actions=4, max_requests=5, max_seconds=30),
                    focus="qa",
                )

        self.assertEqual(forbidden_factory_calls, 1)
        self.assertEqual(summary["status"], "finished")
        self.assertTrue(summary["qa_mode"])
        self.assertIn("reset_inconsistency", [item["type"] for item in summary["anomalies"]])
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        reset = next(row for row in rows if row["type"] == "reset")
        self.assertTrue(reset["qa_mode"])
        self.assertIn("reset_inconsistency", [item["type"] for item in reset["software_fact"]["anomalies"]])
        self.assertTrue(replay(path)["lean_facts_validated"])

    def test_reset_mismatch_without_defect_name_is_an_anomaly_in_any_focus(self):
        class MismatchedResetAdapter:
            version = "mismatched-reset-fixture"
            seed = 0

            def __init__(self, *, mismatched=False):
                self._game = KeyGateGame(self.seed)
                if mismatched:
                    self._game.position = (2, 1)
                    self._game.events = ["Reset produced a visibly different starting state."]

            def observe(self):
                return self._game.observe()

            def act(self, action):
                return self._game.act(action)

        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        path = Path(directory.name) / "generic-reset-mismatch.jsonl"
        game = MismatchedResetAdapter()
        self.assertFalse(hasattr(game, "defect"))
        client = NebiusClient(Settings("unit-key"))
        replies = [
            _response(_decision("reset", [1, 1])),
            _response(_decision("finish")),
        ]

        with SessionLog(path, ("unit-key",)) as log:
            begin_session(
                log, game, "phase-2", schema_version=2, task="explore",
                focus="mechanics", prompt_version=LEAN_PROMPT_VERSION,
            )
            with patch.object(client, "_request", side_effect=replies):
                summary = run_lean_exploration(
                    game,
                    client,
                    log,
                    lambda: MismatchedResetAdapter(mismatched=True),
                    ExplorationLimits(max_actions=2, max_requests=3, max_seconds=30),
                    focus="mechanics",
                )

        self.assertEqual(summary["status"], "finished")
        self.assertIn("reset_inconsistency", [item["type"] for item in summary["anomalies"]])
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        reset = next(row for row in rows if row["type"] == "reset")
        self.assertIn(
            "reset_inconsistency",
            [item["type"] for item in reset["software_fact"]["anomalies"]],
        )

    def test_historical_non_qa_reset_recording_replays_with_its_old_fact_contract(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        path = Path(directory.name) / "historical-reset.jsonl"
        game = KeyGateGame(0, "reset-leak")
        client = NebiusClient(Settings("unit-key"))
        replies = [
            _response(_decision("south", [1, 2])),
            _response(_decision("south", [1, 3])),
            _response(_decision("reset", [1, 1])),
            _response(_decision("finish")),
        ]

        with SessionLog(path, ("unit-key",)) as log:
            begin_session(
                log, game, "phase-2", schema_version=2, task="explore",
                focus="mechanics", prompt_version=LEAN_PROMPT_VERSION,
                **defect_evaluation_metadata("reset-leak"),
            )
            with patch.object(client, "_request", side_effect=replies):
                summary = run_lean_exploration(
                    game,
                    client,
                    log,
                    game.reset,
                    ExplorationLimits(max_actions=4, max_requests=5, max_seconds=30),
                    focus="mechanics",
                )
        self.assertEqual(summary["status"], "finished")

        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        for row in rows:
            if row["type"] == "request":
                properties = row["response_schema"]["properties"]
                properties["reason"]["maxLength"] = 240
                note_properties = properties["notes"]["items"]["properties"]
                note_properties["claim"]["maxLength"] = 240
                note_properties["evidence_steps"].pop("maxItems")
                row["response_schema_bytes"] = len(json.dumps(
                    row["response_schema"], separators=(",", ":"),
                ).encode("utf-8"))
            elif row["type"] == "reset":
                row["software_fact"].pop("anomalies")
            elif row["type"] == "summary":
                row.pop("anomalies")
                next(
                    fact for fact in row["factual_log"] if fact["action"] == "reset"
                ).pop("anomalies")
                row["response_schema_bytes"] = 854
        path.write_text(
            "\n".join(json.dumps(row, separators=(",", ":")) for row in rows) + "\n",
            encoding="utf-8",
        )

        replayed = replay(path)
        self.assertTrue(replayed["recording_complete"])
        self.assertEqual(replayed["exploration_status"], "finished")


if __name__ == "__main__":
    unittest.main()
