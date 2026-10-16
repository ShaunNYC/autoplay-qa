"""Progress review fixtures never make inference requests."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from autoplay_qa.cli import begin_session
from autoplay_qa.config import Settings
from autoplay_qa.exploration import ExplorationLimits, run_exploration
from autoplay_qa.game import KeyGateGame, evaluate_transition
from autoplay_qa.nebius import NebiusClient, NebiusError
from autoplay_qa.review import retention_progress, review_session
from autoplay_qa.session import SessionLog


def note(evidence=(), claim="North stays blocked from the initial state"):
    return {"id": "h1", "claim": claim, "status": "supported" if evidence else "untested", "evidence_steps": list(evidence)}


def decision(action, pending=None, notes=(), experiment="movement"):
    return {
        "action": action, "reason": "Explicit test fixture", "plan": "Inspect recorded feedback",
        "experiment_id": experiment, "experiment_goal": "Fixture question",
        "prediction": "The observable state may change",
        "assessment": {"step": pending, "outcome": "Matched the recorded result"} if pending else None,
        "hypotheses": list(notes),
    }


def response(value, finish="stop"):
    return {
        "id": "explicit-test-fixture", "model": "test-only-model",
        "choices": [{"message": {"content": json.dumps(value)}, "finish_reason": finish}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    }


class ReviewTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.path = Path(directory.name) / "review.jsonl"

    def explore(self, values, limits=ExplorationLimits()):
        game = KeyGateGame(0)
        client = NebiusClient(Settings("test-key"))
        with SessionLog(self.path, ("test-key",)) as log:
            begin_session(log, game, "unit-test-fixtures", schema_version=2, task="explore")
            with patch.object(client, "_request", side_effect=values):
                run_exploration(game, client, log, lambda: KeyGateGame(0), limits)
        return self.records()

    def records(self):
        return [json.loads(line) for line in self.path.read_text().splitlines()]

    def rewrite(self, records):
        self.path.write_text("\n".join(json.dumps(record) for record in records) + "\n")

    def reset_fixture(self, after_reset="north", final_evidence=(1, 3)):
        return self.explore([
            response(decision("north", notes=[note()])),
            response(decision("reset", 1, [note([1])])),
            response(decision(after_reset, 2, experiment="repeat")),
            response(decision("finish", 3, [note(final_evidence)], experiment="repeat")),
        ])

    def both_families_fixture(self, final="finish", tail=(), limits=ExplorationLimits()):
        interaction_note = {**note(), "id": "h2", "claim": "Interaction leaves the initial state unchanged"}
        records = self.explore([
            response(decision("north", notes=[note()])),
            response(decision("interact", 1, [note([1]), interaction_note], experiment="interaction")),
            response(decision("reset", 2, [{**interaction_note, "status": "supported", "evidence_steps": [2]}])),
            response(decision("north", 3)),
            response(decision("interact", 4, [note([1, 4])], experiment="interaction")),
            response(decision(final, 5, [{**interaction_note, "status": "supported", "evidence_steps": [2, 5]}])),
            *tail,
        ], limits=limits)
        records[0].update(focus="reset-retention", prompt_version="fixture-v3")
        self.rewrite(records)
        return records

    def test_actual_request_retention_and_evidence_for_both_trials(self):
        self.reset_fixture()
        result = review_session(self.path)
        learning = result["reset_learning"]
        self.assertTrue(result["recording_complete"])
        self.assertFalse(result["partial"])
        self.assertEqual(result["coverage"]["action_counts"], {"north": 2, "reset": 1})
        self.assertEqual(result["coverage"]["experiment_count"], 2)
        self.assertEqual(result["coverage"]["distinct_observed_states"], 1)
        self.assertTrue(learning["memory_supplied_after_reset"])
        self.assertEqual(learning["retained_hypothesis_ids"], ["h1"])
        self.assertEqual(learning["post_reset_assessed_steps"], [3])
        self.assertTrue(learning["behavioral_retest_observed"])
        self.assertEqual(learning["retest_pairs"], [{"before_step": 1, "after_step": 3, "action": "north", "hypothesis_id": "h1"}])
        self.assertTrue(result["review_needed"])
        self.assertFalse(result["learning_claims_verified"])
        self.assertFalse(result["bugs_verified"])

    def test_mechanical_retention_does_not_imply_behavioral_retest(self):
        self.reset_fixture(after_reset="interact")
        result = review_session(self.path)
        self.assertTrue(result["reset_learning"]["memory_supplied_after_reset"])
        self.assertEqual(result["reset_learning"]["post_reset_assessed_steps"], [3])
        self.assertFalse(result["reset_learning"]["behavioral_retest_observed"])
        self.assertTrue(result["coverage"]["interaction"])
        self.assertEqual(result["coverage"]["unchanged_interactions"], 1)

    def test_repeated_action_needs_both_trial_references(self):
        self.reset_fixture(final_evidence=[1])
        result = review_session(self.path)
        self.assertTrue(result["reset_learning"]["memory_supplied_after_reset"])
        self.assertEqual(result["reset_learning"]["post_reset_assessed_steps"], [3])
        self.assertFalse(result["reset_learning"]["behavioral_retest_observed"])

    def test_matching_id_with_changed_content_does_not_prove_retention(self):
        for field, value in (("claim", "An unrelated claim"), ("status", "uncertain"), ("evidence_steps", [])):
            with self.subTest(field=field):
                if self.path.exists():
                    self.path.unlink()
                records = self.reset_fixture()
                for record in records:
                    if record["type"] == "request" and record["episode"] > 0:
                        context = json.loads(record["messages"][1]["content"])
                        context["hypotheses"][0][field] = value
                        record["messages"][1]["content"] = json.dumps(context)
                self.rewrite(records)
                result = review_session(self.path)
                self.assertFalse(result["reset_learning"]["memory_supplied_after_reset"])
                self.assertFalse(result["reset_learning"]["behavioral_retest_observed"])

    def test_same_action_from_different_state_does_not_count_as_retest(self):
        self.explore([
            response(decision("north", notes=[note()])),
            response(decision("reset", 1, [note([1])])),
            response(decision("east", 2)),
            response(decision("north", 3)),
            response(decision("finish", 4, [note([1, 4])])),
        ])
        result = review_session(self.path)
        self.assertTrue(result["reset_learning"]["memory_supplied_after_reset"])
        self.assertFalse(result["reset_learning"]["behavioral_retest_observed"])

    def test_missing_assessment_keeps_retest_unproven(self):
        records = self.reset_fixture()
        for record in records:
            if record["type"] == "memory" and (record.get("assessment") or {}).get("step") == 3:
                record["assessment"] = None
        self.rewrite(records)
        result = review_session(self.path)
        self.assertFalse(result["reset_learning"]["behavioral_retest_observed"])
        self.assertEqual(result["learning"]["unassessed_final_step"], 3)

    def test_latency_uses_recorded_completions_and_nearest_rank(self):
        records = self.reset_fixture()
        for record, latency in zip((r for r in records if r["type"] == "completion"), [100, 200, 300, 400]):
            record["latency_ms"] = latency
        self.rewrite(records)
        result = review_session(self.path)
        self.assertEqual(result["requests"]["latency_ms"], {"count": 4, "median": 250, "p95": 400, "max": 400})
        completions = [r for r in records if r["type"] == "completion"]
        del completions[0]["latency_ms"]
        completions[1]["latency_ms"] = True
        completions[2]["latency_ms"] = -1
        self.rewrite(records)
        result = review_session(self.path)
        self.assertEqual(result["requests"]["latency_ms"], {"count": 1, "median": 400, "p95": 400, "max": 400})
        self.assertEqual(result["requests"]["latency_unavailable"], 3)

    def test_multiple_resets_preserve_each_retention_boundary(self):
        self.explore([
            response(decision("north", notes=[note()])),
            response(decision("reset", 1, [note([1])])),
            response(decision("north", 2)),
            response(decision("reset", 3, [note([1, 3])])),
            response(decision("north", 4)),
            response(decision("finish", 5, [note([1, 3, 5])])),
        ])
        result = review_session(self.path)
        self.assertEqual(result["coverage"]["episodes"], 3)
        resets = result["reset_learning"]["resets"]
        self.assertEqual([reset["reset_step"] for reset in resets], [2, 4])
        self.assertEqual([reset["post_reset_assessed_steps"] for reset in resets], [[3], [5]])
        self.assertEqual(len(result["reset_learning"]["retest_pairs"]), 3)

    def test_hypothesis_future_references_do_not_become_valid_later(self):
        records = self.explore([
            response(decision("north", notes=[note()])),
            response(decision("east", 1, [note([1])])),
        ], limits=ExplorationLimits(max_actions=2))
        last_memory = [record for record in records if record["type"] == "memory"][-1]
        last_memory["hypotheses"][0]["evidence_steps"] = [2]
        self.rewrite(records)
        result = review_session(self.path)
        self.assertEqual(result["learning"]["retained_hypotheses"], 1)
        self.assertEqual(result["learning"]["evidence_linked_hypotheses"], 0)

    def test_diagnostics_distinguish_known_and_missing_usage(self):
        records = self.reset_fixture()
        completions = [record for record in records if record["type"] == "completion"]
        completions[0]["reasoning_tokens"] = 12
        completions[1]["reasoning_tokens"] = None
        for record in completions[2:]:
            record.pop("reasoning_tokens", None)
        self.rewrite(records)
        diagnostics = review_session(self.path)["requests"]["diagnostics"]
        self.assertEqual(diagnostics["reasoning_tokens"], {"recorded_responses": 1, "total": 12})
        self.assertEqual(diagnostics["output_tokens"], {"recorded_responses": 4, "total": 4})
        self.assertEqual(diagnostics["recorded_decision_characters"]["recorded_responses"], 4)

    def test_goal_checks_preserve_unavailable_focus_and_partial_coverage(self):
        self.reset_fixture()
        result = review_session(self.path)
        checks = result["goal_checks"]
        self.assertIsNone(result["focus"])
        self.assertIsNone(checks["focused_question_observed"])
        self.assertFalse(checks["mechanics_coverage_observed"])
        self.assertTrue(checks["reset_retention_observed"])
        self.assertEqual(checks["reset_retested_control_families"], {"movement": True, "interaction": False})
        self.assertEqual(checks["next_stage_candidate"], "exploration_validation")
        records = self.records()
        records[0].update(focus="reset-retention", prompt_version="fixture-v3")
        self.rewrite(records)
        focused = review_session(self.path)
        self.assertEqual(focused["prompt_version"], "fixture-v3")
        self.assertFalse(focused["goal_checks"]["focused_question_observed"])

    def test_goal_checks_observe_both_families_without_verifying_claims(self):
        self.both_families_fixture()
        result = review_session(self.path)
        checks = result["goal_checks"]
        self.assertTrue(checks["mechanics_coverage_observed"])
        self.assertTrue(checks["reset_retention_observed"])
        self.assertEqual(checks["reset_retested_control_families"], {"movement": True, "interaction": True})
        self.assertTrue(checks["focused_question_observed"])
        self.assertEqual(checks["next_stage_candidate"], "controlled_defect_reproduction")
        self.assertTrue(result["review_needed"])
        self.assertFalse(result["learning_claims_verified"])
        self.assertFalse(result["bugs_verified"])

    def test_retention_progress_reuses_matches_and_actual_cited_conditions(self):
        self.both_families_fixture()
        reviewed = review_session(self.path)
        with patch("autoplay_qa.review.session.replay", side_effect=AssertionError("No engine replay in learner progress")):
            progress = retention_progress(self.path)
        self.assertEqual(set(progress), {"families_observed", "evidence_pairs", "baseline_conditions", "experiment_ids_seen"})
        self.assertEqual(progress["families_observed"], {"movement": True, "interaction": True})
        self.assertEqual(progress["evidence_pairs"], reviewed["reset_learning"]["retest_pairs"])
        self.assertEqual(progress["experiment_ids_seen"], ["interaction", "movement"])
        self.assertEqual(
            [(item["hypothesis_id"], item["step"], item["episode"], item["action"]) for item in progress["baseline_conditions"]],
            [("h1", 1, 0, "north"), ("h1", 4, 1, "north"), ("h2", 2, 0, "interact"), ("h2", 5, 1, "interact")],
        )
        for item in progress["baseline_conditions"]:
            self.assertEqual(item["player_position"], [1, 1])
            self.assertEqual(item["inventory"], [])
            self.assertEqual(item["objects"], KeyGateGame(0).observe()["objects"])

    def test_retention_progress_accepts_flushed_recording_without_summary(self):
        records = self.both_families_fixture()
        self.rewrite(records[:-1])
        progress = retention_progress(self.path)
        self.assertEqual(progress["families_observed"], {"movement": True, "interaction": True})

    def test_retention_progress_different_conditions_do_not_count(self):
        records = self.both_families_fixture()
        next(record for record in records if record["type"] == "step" and record["step"] == 4)["before"]["player"]["position"] = [2, 1]
        self.rewrite(records)
        progress = retention_progress(self.path)
        self.assertEqual(progress["families_observed"], {"movement": False, "interaction": True})
        self.assertEqual([pair["action"] for pair in progress["evidence_pairs"]], ["interact"])

    def test_retention_progress_requires_post_trial_assessment(self):
        records = self.both_families_fixture()
        for record in records:
            if record["type"] == "memory" and (record.get("assessment") or {}).get("step") == 5:
                record["assessment"] = None
        self.rewrite(records)
        progress = retention_progress(self.path)
        self.assertEqual(progress["families_observed"], {"movement": True, "interaction": False})
        self.assertEqual([pair["action"] for pair in progress["evidence_pairs"]], ["north"])

    def test_retention_requires_valid_assessments_of_both_original_trials(self):
        for step, missing_family in ((1, "movement"), (2, "interaction")):
            for outcome in (None, "supported"):
                with self.subTest(step=step, outcome=outcome):
                    if self.path.exists():
                        self.path.unlink()
                    records = self.both_families_fixture()
                    memory = next(
                        record for record in records
                        if record["type"] == "memory" and (record.get("assessment") or {}).get("step") == step
                    )
                    memory["assessment"] = None if outcome is None else {"step": step, "outcome": outcome}
                    self.rewrite(records)
                    reviewed = review_session(self.path)
                    expected = {"movement": True, "interaction": True, missing_family: False}
                    self.assertTrue(reviewed["execution_replayed"])
                    self.assertTrue(reviewed["reset_learning"]["memory_supplied_after_reset"])
                    self.assertEqual(reviewed["reset_learning"]["post_reset_assessed_steps"], [4, 5])
                    self.assertIsNone(reviewed["learning"]["unassessed_final_step"])
                    self.assertNotIn(step, reviewed["learning"]["assessed_steps"])
                    self.assertEqual(reviewed["goal_checks"]["reset_retested_control_families"], expected)
                    self.assertFalse(reviewed["goal_checks"]["focused_question_observed"])
                    self.assertEqual(reviewed["goal_checks"]["next_stage_candidate"], "exploration_validation")
                    self.assertEqual(retention_progress(self.path)["families_observed"], expected)

    def test_bare_status_labels_do_not_promote_legacy_assessments(self):
        records = self.both_families_fixture()
        assessment = next(
            record["assessment"] for record in records
            if record["type"] == "memory" and (record.get("assessment") or {}).get("step") == 5
        )
        for label in ("supported", "untested", "uncertain", "verified", " Supported. "):
            with self.subTest(label=label):
                assessment["outcome"] = label
                self.rewrite(records)
                reviewed = review_session(self.path)
                self.assertTrue(reviewed["execution_replayed"])
                self.assertFalse(reviewed["goal_checks"]["focused_question_observed"])
                self.assertEqual(reviewed["learning"]["unassessed_final_step"], 5)
                self.assertEqual(reviewed["goal_checks"]["next_stage_candidate"], "exploration_validation")
                progress = retention_progress(self.path)
                self.assertEqual(progress["families_observed"], {"movement": True, "interaction": False})
        for comparison in ("matched", "Contradicted."):
            with self.subTest(comparison=comparison):
                assessment["outcome"] = comparison
                self.rewrite(records)
                self.assertTrue(review_session(self.path)["goal_checks"]["focused_question_observed"])
                self.assertEqual(retention_progress(self.path)["families_observed"], {"movement": True, "interaction": True})

    def test_retention_progress_baselines_reject_future_evidence_at_memory_update(self):
        records = self.explore([
            response(decision("north", notes=[note()])),
            response(decision("east", 1, [note([1])])),
        ], limits=ExplorationLimits(max_actions=2))
        last_memory = [record for record in records if record["type"] == "memory"][-1]
        for references in ([2], [1, 99]):
            with self.subTest(references=references):
                last_memory["hypotheses"][0]["evidence_steps"] = references
                self.rewrite(records)
                progress = retention_progress(self.path)
                self.assertEqual(progress["baseline_conditions"], [])
                self.assertEqual(progress["families_observed"], {"movement": False, "interaction": False})

    def test_retention_progress_excludes_withheld_data_and_recommendations(self):
        records = self.both_families_fixture()
        hidden = "DEVELOPER-ONLY-INSTRUCTIONS"
        for record in records:
            if record["type"] not in {"step", "reset"}:
                continue
            before = record["before"]
            before.update(rules=[hidden], goal=hidden, passability=hidden, events=[hidden])
            before["player"]["strategy"] = hidden
            before["objects"]["developer_hints"] = hidden
            before["objects"]["gate"]["passable"] = hidden
        self.rewrite(records)
        progress = retention_progress(self.path)
        self.assertNotIn(hidden, json.dumps(progress))
        self.assertEqual(set(progress), {"families_observed", "evidence_pairs", "baseline_conditions", "experiment_ids_seen"})
        for item in progress["baseline_conditions"]:
            self.assertEqual(set(item), {"hypothesis_id", "step", "episode", "action", "player_position", "inventory", "objects"})
            self.assertEqual(set(item["objects"]), {"key", "gate", "exit"})
            self.assertEqual(set(item["objects"]["gate"]), {"position", "open"})

    def test_retention_progress_requires_exploration_header(self):
        records = self.both_families_fixture()
        for field, value in (("schema_version", 1), ("task", "playthrough"), ("type", "step")):
            with self.subTest(field=field):
                original = records[0][field]
                records[0][field] = value
                self.rewrite(records)
                with self.assertRaisesRegex(ValueError, "schema-2 exploration"):
                    retention_progress(self.path)
                records[0][field] = original

    def test_missing_focused_control_family_blocks_stage_candidate(self):
        records = self.explore([
            response(decision("north", notes=[note()])),
            response(decision("interact", 1, [note([1])], experiment="interaction")),
            response(decision("reset", 2)),
            response(decision("north", 3)),
            response(decision("finish", 4, [note([1, 4])])),
        ])
        records[0]["focus"] = "reset-retention"
        self.rewrite(records)
        result = review_session(self.path)
        self.assertEqual(result["status"], "finished")
        checks = result["goal_checks"]
        self.assertTrue(checks["mechanics_coverage_observed"])
        self.assertTrue(checks["reset_retention_observed"])
        self.assertFalse(checks["focused_question_observed"])
        self.assertEqual(checks["next_stage_candidate"], "exploration_validation")

    def test_mechanics_or_legacy_focus_cannot_skip_interaction_retention(self):
        records = self.explore([
            response(decision("north", notes=[note()])),
            response(decision("interact", 1, [note([1])], experiment="interaction")),
            response(decision("reset", 2)),
            response(decision("north", 3)),
            response(decision("finish", 4, [note([1, 4])])),
        ])
        for focus in ("mechanics", None):
            with self.subTest(focus=focus):
                records[0]["focus"] = focus
                self.rewrite(records)
                result = review_session(self.path)
                checks = result["goal_checks"]
                self.assertTrue(result["recording_complete"])
                self.assertIsNone(result["learning"]["unassessed_final_step"])
                self.assertTrue(checks["mechanics_coverage_observed"])
                self.assertTrue(checks["reset_retention_observed"])
                self.assertEqual(checks["reset_retested_control_families"], {"movement": True, "interaction": False})
                self.assertEqual(checks["focused_question_observed"], True if focus else None)
                self.assertEqual(checks["next_stage_candidate"], "exploration_validation")

    def test_error_status_blocks_stage_candidate_despite_complete_observations(self):
        records = self.both_families_fixture(final="wait", tail=[NebiusError("Fixture API failure")])
        for status in (
            "invalid_decision", "model_output_limit", "inference_error", "provider_unavailable",
            "contract_failure", "environment_error", "interrupted",
        ):
            with self.subTest(status=status):
                records[-1]["status"] = status
                self.rewrite(records)
                result = review_session(self.path)
                checks = result["goal_checks"]
                self.assertTrue(result["partial"])
                self.assertTrue(checks["mechanics_coverage_observed"])
                self.assertTrue(checks["reset_retention_observed"])
                self.assertTrue(checks["focused_question_observed"])
                self.assertEqual(checks["next_stage_candidate"], "exploration_validation")

    def test_budget_boundary_with_unassessed_final_outcome_keeps_validation(self):
        self.both_families_fixture(final="wait", limits=ExplorationLimits(max_actions=6))
        result = review_session(self.path)
        self.assertEqual(result["status"], "action_limit")
        self.assertTrue(result["partial"])
        self.assertTrue(result["goal_checks"]["focused_question_observed"])
        self.assertEqual(result["learning"]["unassessed_final_step"], 6)
        self.assertEqual(result["goal_checks"]["next_stage_candidate"], "exploration_validation")
        self.assertTrue(result["review_needed"])

    def test_incomplete_recording_keeps_validation_despite_complete_comparisons(self):
        records = self.both_families_fixture()
        self.rewrite(records[:-1])
        result = review_session(self.path)
        self.assertFalse(result["recording_complete"])
        self.assertTrue(result["partial"])
        self.assertIsNone(result["status"])
        self.assertIsNone(result["learning"]["unassessed_final_step"])
        self.assertTrue(result["goal_checks"]["focused_question_observed"])
        self.assertEqual(result["goal_checks"]["next_stage_candidate"], "exploration_validation")

    def test_live_like_truncations_and_incomplete_focused_question_keep_validation(self):
        records = self.explore([
            response(decision("north", notes=[note()])),
            response(decision("interact", 1, [note([1])], experiment="interaction")),
            response(decision("reset", 2)),
            response(decision("north", 3)),
            response(decision("east", 4, [note([1, 4])])),
            response(decision("finish", 5), finish="length"),
            response(decision("finish", 5), finish="length"),
        ])
        records[0]["focus"] = "reset-retention"
        self.rewrite(records)
        result = review_session(self.path)
        self.assertEqual(result["status"], "invalid_decision")
        self.assertTrue(result["partial"])
        self.assertEqual(result["requests"]["truncated_completions"], 2)
        self.assertTrue(result["goal_checks"]["mechanics_coverage_observed"])
        self.assertTrue(result["goal_checks"]["reset_retention_observed"])
        self.assertFalse(result["goal_checks"]["focused_question_observed"])
        self.assertEqual(result["goal_checks"]["next_stage_candidate"], "exploration_validation")

    def test_mechanics_focus_coverage_does_not_require_reset(self):
        records = self.explore([
            response(decision("north", notes=[note()])),
            response(decision("interact", 1, [note([1])], experiment="interaction")),
            response(decision("finish", 2)),
        ])
        records[0]["focus"] = "mechanics"
        self.rewrite(records)
        checks = review_session(self.path)["goal_checks"]
        self.assertTrue(checks["mechanics_coverage_observed"])
        self.assertTrue(checks["focused_question_observed"])
        self.assertFalse(checks["reset_retention_observed"])
        self.assertEqual(checks["next_stage_candidate"], "exploration_validation")

    def test_rejected_truncations_and_requests_without_completion(self):
        self.explore([
            response(decision("north"), finish="length"),
            response(decision("north", notes=[note()])),
            NebiusError("Test fixture network failure"),
        ])
        result = review_session(self.path)
        self.assertEqual(result["requests"]["attempted"], 3)
        self.assertEqual(result["requests"]["completed"], 2)
        self.assertEqual(result["requests"]["without_completion"], 1)
        self.assertEqual(result["requests"]["rejected_decisions"], 1)
        self.assertEqual(result["requests"]["truncated_completions"], 1)

    def test_schema1_is_supported_without_learning_or_latency_metadata(self):
        game = KeyGateGame(0)
        with SessionLog(self.path) as log:
            begin_session(log, game, "human")
            for step in (1, 2):
                before = game.observe()
                after = game.act("wait")
                log.write("step", step=step, before=before, decision={"action": "wait"}, after=after, evaluation=evaluate_transition(before, "wait", after))
            log.write("summary", status="quit", steps=2, inference_requests=0, final_observation=game.observe())
        result = review_session(self.path)
        self.assertEqual(result["schema_version"], 1)
        self.assertEqual(result["requests"]["latency_ms"], {"count": 0, "median": None, "p95": None, "max": None})
        self.assertEqual(result["learning"], {"available": False})
        self.assertEqual(result["reset_learning"], {"available": False})
        self.assertEqual(result["coverage"]["distinct_observed_states"], 1)

    def test_old_schema2_without_optional_metadata_and_incomplete_recording(self):
        records = self.reset_fixture()
        for record in records:
            record.pop("focus", None)
            if record["type"] == "completion":
                record.pop("latency_ms", None)
        self.rewrite(records[:-1])
        result = review_session(self.path)
        self.assertFalse(result["recording_complete"])
        self.assertTrue(result["partial"])
        self.assertIsNone(result["status"])
        self.assertTrue(result["reset_learning"]["behavioral_retest_observed"])
        self.assertEqual(result["requests"]["latency_unavailable"], 4)

    def test_reset_divergence_is_rejected_before_review(self):
        records = self.reset_fixture()
        next(record for record in records if record["type"] == "reset")["after"]["tick"] = 1
        self.rewrite(records)
        with self.assertRaisesRegex(ValueError, "diverged after"):
            review_session(self.path)

    def test_output_omits_prompts_raw_answers_claims_and_error_text(self):
        records = self.reset_fixture()
        secret = "SHOULD-NOT-APPEAR-IN-THE-REVIEW"
        records[0]["endpoint"] = secret
        records[-1]["error"] = secret
        for record in records:
            if record["type"] == "completion":
                record["answer"] = secret
            if record["type"] == "memory":
                record["hypotheses"][0]["claim"] = secret
        self.rewrite(records)
        before = self.path.read_bytes()
        result = review_session(self.path)
        self.assertNotIn(secret, json.dumps(result))
        self.assertEqual(self.path.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
