"""Versioned investigation metadata must survive the real action/replay loop."""

import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from autoplay_qa.adaptive_protocol import context_class
from autoplay_qa.cli import begin_session
from autoplay_qa.config import Settings
from autoplay_qa.exploration import ExplorationLimits
from autoplay_qa.experience import ExperienceStore
from autoplay_qa.game import KeyGateGame
from autoplay_qa.lean import run_lean_exploration
from autoplay_qa.nebius import NebiusClient
from autoplay_qa.review import review_session
from autoplay_qa.session import SessionLog, replay
from tests.test_lean_exploration import decision, response


class InvestigationIntegrationTests(unittest.TestCase):
    def fixture(self, replies, *, store=None, run_id="inquiry-integration", version="adaptive-v3"):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        path = Path(temporary.name) / "recording.jsonl"
        game = KeyGateGame(0)
        context = context_class(version)(
            game.observe(), store=store, run_id=run_id,
        )
        client = NebiusClient(Settings("fake-key"))
        with SessionLog(path) as log:
            begin_session(log, game, "fake", schema_version=2, task="explore", focus="qa",
                          prompt_version=version,
                          adaptive_config={"run_id": run_id, "max_prompt_bytes": 24000})
            with patch.object(client, "_request", side_effect=replies) as http:
                result = run_lean_exploration(
                    game, client, log, game.reset, ExplorationLimits(),
                    focus="qa", adaptive=context,
                )
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        return result, path, rows, http

    def test_missing_optional_inquiry_never_blocks_controls_or_native_replay(self):
        result, path, rows, http = self.fixture([
            response(decision(action)) for action in ("north", "north", "east", "reset", "finish")
        ])
        self.assertEqual((result["steps"], result["resets"], http.call_count), (4, 1, 5))
        self.assertEqual(result["status"], "finished")
        self.assertIsNone(result["unassessed_final_outcome"])
        request = next(row for row in rows if row["type"] == "request")
        self.assertIn("investigation_update", request["response_schema"]["properties"])
        self.assertIn("predictions", request["response_schema"]["properties"])
        controls = [row for row in rows if row["type"] in {"step", "reset", "finish"}]
        self.assertTrue(all("investigation_fact" in row for row in controls))
        self.assertTrue(replay(path)["adaptive_facts_validated"])
        self.assertEqual(review_session(path)["prompt_version"], "adaptive-v3")

    def test_invalid_optional_metadata_keeps_the_legal_action_and_warning(self):
        invalid = decision("east")
        invalid.update(investigation_update=["not an investigation"], predictions="not forecasts")
        result, path, rows, http = self.fixture([response(invalid), response(decision("finish"))])
        self.assertEqual((result["steps"], result["rejected_decisions"], http.call_count), (1, 0, 2))
        self.assertGreater(result["warning_count"], 0)
        self.assertEqual(next(row for row in rows if row["type"] == "step")["decision"]["action"], "east")
        self.assertTrue(replay(path)["adaptive_facts_validated"])

    def test_receipt_tampering_cannot_be_hidden_by_recompute_mutation(self):
        for target in ("step", "finish"):
            with self.subTest(target=target):
                _, path, rows, _ = self.fixture([
                    response(decision("east")), response(decision("finish")),
                ])
                corrupted = copy.deepcopy(rows)
                row = next(row for row in corrupted if row["type"] == target)
                row["investigation_fact"]["invented_host_fact"] = True
                path.write_text("\n".join(json.dumps(row) for row in corrupted) + "\n")
                with self.assertRaises(ValueError):
                    replay(path)

    def test_truncated_answer_does_not_create_an_experiment(self):
        result, path, rows, _ = self.fixture([
            response(decision("east"), finish="length"), response(decision("finish")),
        ])
        self.assertEqual((result["steps"], result["rejected_decisions"]), (0, 1))
        self.assertFalse(any(row["type"] in {"step", "reset"} for row in rows))
        self.assertTrue(replay(path)["adaptive_facts_validated"])

    def test_frozen_question_recovery_replays_without_archive_writes(self):
        with tempfile.TemporaryDirectory() as temporary:
            database = Path(temporary) / "acquisition.sqlite3"
            frozen = Path(temporary) / "frozen.sqlite3"
            initial = KeyGateGame(0).observe()
            probe = decision("east")
            probe.update(investigation_update={
                "id": "movement-question", "decision": "continue",
                "question": "Does this control change the observed position?",
                "explanations": ["Position changes", "Position stays unchanged"],
                "expected_result": "Observe position after the control.",
                "condition_facets": ["player.position"],
            }, predictions=[{"path": "player.position", "relation": "changes", "value_json": None}])
            with ExperienceStore(database, "integration-game", KeyGateGame.version, initial) as store:
                self.fixture([response(probe), response(decision("west")), response(decision("finish"))],
                             store=store, run_id="acquisition")
                store.snapshot(frozen)
            before_hash = hashlib.sha256(frozen.read_bytes()).hexdigest()
            with ExperienceStore(frozen, "integration-game", KeyGateGame.version, initial, read_only=True) as store:
                result, path, rows, _ = self.fixture([
                    response(decision("east")), response(decision("finish")),
                ], store=store, run_id="evaluation")
            self.assertEqual(hashlib.sha256(frozen.read_bytes()).hexdigest(), before_hash)
            self.assertEqual(result["adaptive"]["investigation"]["restored_unfinished"], 1)
            first_request = next(row for row in rows if row["type"] == "request")
            source = json.loads(first_request["messages"][1]["content"])["investigation_memory"]["prior_run_index"]
            self.assertEqual(source["source_run_id"], "acquisition")
            self.assertTrue(source["applicability"]["movement-question"]["all_selected_match"])
            self.assertTrue(replay(path)["adaptive_facts_validated"])
            self.assertFalse(replay(path)["prior_run_memory_execution_verified"])


if __name__ == "__main__":
    unittest.main()
