"""Persistent tentative investigations, separate from observed execution facts."""

from copy import deepcopy
from contextlib import closing
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest

from autoplay_qa.experience import EvidenceConflict, ExperienceStore, ScopeMismatch, state_key
from autoplay_qa.investigation import byte_size, copy_conditions
from tests.test_experience import observation, result


def reference(run="train", step=1):
    return {"run_id": run, "step": step}


def inquiry(initial, *, identifier="boundary", run="train", step=1,
            status="active", counterexamples=(), facets=None):
    facets = facets or ["player.position", "player.facing"]
    return {
        "id": identifier, "status": status,
        "question": "Does feedback depend on this observed condition?",
        "explanations": ["The visible boundary may reject this control.",
                         "The direction or another condition may matter."],
        "expected_result": "The selected public condition stays unchanged.",
        "condition_facets": facets, "conditions": copy_conditions(initial, facets),
        "evidence": [reference(run, step)],
        "counterexamples": deepcopy(list(counterexamples)),
        "last_decision": {"active": "continue", "pending": "defer", "closed": "close"}[status],
        "origin_state_id": state_key(initial), "last_transition": reference(run, step),
    }


class InvestigationStoreTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="investigation-store-tests-")
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "memory.sqlite3"
        self.initial = observation()
        self.memory = ExperienceStore(self.path, "project", "build-1", self.initial)
        self.addCleanup(self.memory.close)
        self.memory.record(self.initial, "forward", result(self.initial), "train", 1)

    def packet(self, observation=None, **options):
        return self.memory.investigation_index(observation or self.initial,
                                               exclude_run_id="evaluation", **options)

    def test_restart_retains_tentative_question_provenance_and_factual_anchor(self):
        row = inquiry(self.initial)
        receipt = self.memory.save_investigations("train", 1, [row])
        self.assertTrue(receipt["persisted"])
        expected = self.packet()
        self.memory.close()
        self.memory = ExperienceStore(self.path, "project", "build-1", self.initial)
        self.addCleanup(self.memory.close)
        actual = self.packet()
        self.assertEqual(actual, expected)
        self.assertEqual(actual["source_run_id"], "train")
        self.assertEqual(actual["revision"], 1)
        self.assertEqual(actual["investigations"], [row])
        self.assertTrue(actual["complete"])
        self.assertEqual(actual["evidence_facts"][0]["citation"], reference())
        self.assertEqual(actual["evidence_facts"][0]["outcome"]["events"], ["No change"])
        self.assertNotIn("confirmed", json.dumps(actual))

    def test_facts_feedback_and_existing_outputs_are_not_modified_by_model_summary(self):
        before_summary, before_facts = self.memory.summary(), self.memory.retrieve(self.initial)
        self.memory.save_investigations("train", 1, [inquiry(self.initial)])
        before_packet = self.packet()
        self.memory.record_feedback("train", 1, "confirmed", comment="PRIVATE_EVALUATION_LABEL")
        self.assertEqual(self.memory.summary(), before_summary)
        self.assertEqual(self.memory.retrieve(self.initial), before_facts)
        self.assertEqual(self.packet(), before_packet)
        self.assertNotIn("PRIVATE_EVALUATION_LABEL", json.dumps(self.packet()))

    def test_exact_selected_conditions_differ_from_full_state_and_are_not_truth(self):
        row = inquiry(self.initial, facets=["player.position", "entities.gate.state.open"])
        self.memory.save_investigations("train", 1, [row])
        changed = result(self.initial, facing="East")
        applicable = self.packet(changed)["applicability"][row["id"]]
        self.assertFalse(applicable["exact_origin_state_match"])
        self.assertTrue(applicable["all_selected_match"])
        self.assertEqual(applicable["matching_facets"], row["condition_facets"])
        changed["entities"][0]["state"]["open"] = True
        changed["player"].pop("position")
        mismatched = self.packet(changed)["applicability"][row["id"]]
        self.assertFalse(mismatched["all_selected_match"])
        self.assertEqual(mismatched["missing_facets"], ["player.position"])
        self.assertEqual(mismatched["mismatching_facets"], ["entities.gate.state.open"])

    def test_revisions_preserve_counterexamples_even_when_origin_is_revised(self):
        original = inquiry(self.initial, counterexamples=[reference()])
        self.memory.save_investigations("train", 1, [original])
        later = result(self.initial, facing="East")
        self.memory.record(self.initial, "right", later, "train", 2)
        revised = inquiry(later, step=2)
        revised["last_decision"] = "revise"
        revised["question"] = "Does the revised condition explain the different feedback?"
        saved = self.memory.save_investigations("train", 2, [revised])
        self.assertEqual(saved["counterexamples_retained"], 1)
        stored = self.packet(later)["investigations"][0]
        self.assertEqual(stored["counterexamples"], [reference()])
        self.assertEqual(stored["origin_state_id"], state_key(later))

    def test_cross_run_counterexample_adoption_requires_explicit_snapshot_lineage(self):
        old = inquiry(self.initial, counterexamples=[reference()])
        self.memory.save_investigations("train", 1, [old])
        unrelated = inquiry(self.initial)
        self.memory.save_investigations("unrelated", 1, [unrelated])
        self.assertEqual(self.packet()["investigations"][0]["counterexamples"], [])
        adopted = inquiry(self.initial)
        saved = self.memory.save_investigations("next-training", 1, [adopted],
                                                parent_snapshot={"run_id": "train", "revision": 1})
        self.assertEqual(saved["counterexamples_retained"], 1)
        self.assertEqual(self.packet()["investigations"][0]["counterexamples"], [reference()])

    def test_negative_evidence_overflow_rejects_revision_without_forgetting(self):
        for step in range(2, 6):
            self.memory.record(self.initial, "forward", result(self.initial), "train", step)
        row = inquiry(self.initial, counterexamples=[reference(step=step) for step in range(1, 5)])
        self.memory.save_investigations("train", 1, [row])
        revised = inquiry(self.initial, step=5, counterexamples=[reference(step=5)])
        with self.assertRaises(ValueError):
            self.memory.save_investigations("train", 2, [revised])
        packet = self.packet()
        self.assertEqual(packet["revision"], 1)
        self.assertEqual(packet["investigations"][0]["counterexamples"], row["counterexamples"])

    def test_active_pending_closed_limits_are_validated_before_persistence(self):
        valid = [inquiry(self.initial)] + [inquiry(self.initial, identifier=f"pending-{index}", status="pending") for index in range(3)]
        self.memory.save_investigations("train", 1, valid)
        self.assertEqual(len(self.packet(max_bytes=16_000)["investigations"]), 4)
        invalid_sets = (
            [inquiry(self.initial, identifier="a"), inquiry(self.initial, identifier="b")],
            [inquiry(self.initial, identifier=f"p-{index}", status="pending") for index in range(4)],
            [inquiry(self.initial, identifier=f"c-{index}", status="closed") for index in range(2)],
            valid + [inquiry(self.initial, identifier="closed", status="closed")],
        )
        for rows in invalid_sets:
            with self.assertRaises(ValueError):
                self.memory.save_investigations("train", 2, rows)
        self.assertEqual(self.packet(max_bytes=16_000)["revision"], 1)

    def test_idempotent_revisions_conflicts_and_backwards_revision_do_not_overwrite(self):
        row = inquiry(self.initial)
        self.memory.save_investigations("train", 1, [row])
        self.assertTrue(self.memory.save_investigations("train", 1, [row])["already_recorded"])
        changed = deepcopy(row)
        changed["expected_result"] = "A different tentative forecast."
        self.memory.save_investigations("train", 3, [changed])
        self.assertTrue(self.memory.save_investigations("train", 1, [row])["already_recorded"])
        with self.assertRaises(EvidenceConflict):
            self.memory.save_investigations("train", 1, [changed])
        with self.assertRaises(ValueError):
            self.memory.save_investigations("train", 2, [row])
        self.assertEqual(self.packet()["revision"], 3)

    def test_unknown_future_or_foreign_scope_citations_and_origins_are_rejected(self):
        for mutate in (
            lambda row: row.update(origin_state_id="f" * 64),
            lambda row: row.update(last_transition=reference(step=2)),
            lambda row: row.update(evidence=[reference("not-executed")]),
            lambda row: row.update(counterexamples=[reference("not-executed")]),
        ):
            row = inquiry(self.initial)
            mutate(row)
            with self.assertRaises(ValueError):
                self.memory.save_investigations("train", 1, [row])
        other = ExperienceStore(self.path, "project", "build-2", self.initial)
        self.addCleanup(other.close)
        with self.assertRaises(ValueError):
            other.save_investigations("other", 1, [inquiry(self.initial)])
        self.assertEqual(other.investigation_index(self.initial, exclude_run_id="evaluation")["investigations"], [])

    def test_parent_snapshot_must_exist_in_same_scope_and_another_run(self):
        row = inquiry(self.initial)
        self.memory.save_investigations("train", 1, [row])
        for parent in ({"run_id": "missing", "revision": 1},
                       {"run_id": "train", "revision": 2},
                       {"run_id": "next", "revision": 1},
                       {"run_id": "train", "revision": True}):
            with self.assertRaises(ValueError):
                self.memory.save_investigations("next", 1, [row], parent_snapshot=parent)
        other = ExperienceStore(self.path, "other-project", "build-1", self.initial)
        self.addCleanup(other.close)
        with self.assertRaises(ValueError):
            other.save_investigations("next", 1, [], parent_snapshot={"run_id": "train", "revision": 1})

    def test_latest_empty_snapshot_prevents_old_question_resurrection(self):
        self.memory.save_investigations("train", 1, [inquiry(self.initial)])
        self.memory.save_investigations("later", 1, [])
        packet = self.packet()
        self.assertEqual(packet["source_run_id"], "later")
        self.assertEqual(packet["investigations"], [])
        self.assertTrue(packet["complete"])
        prior = self.memory.investigation_index(self.initial, exclude_run_id="later")
        self.assertEqual(prior["source_run_id"], "train")
        self.assertEqual(len(prior["investigations"]), 1)

    def test_freeze_snapshot_contains_questions_but_evaluation_writes_are_noops(self):
        self.memory.save_investigations("train", 1, [inquiry(self.initial)])
        path = self.memory.snapshot(Path(self.temp.name) / "frozen.sqlite3")
        with ExperienceStore(path, "project", "build-1", self.initial, read_only=True) as frozen:
            before_bytes = path.read_bytes()
            before = frozen.investigation_index(self.initial, exclude_run_id="evaluation")
            unknown = result(self.initial, position=[2, 0])
            evaluation_row = inquiry(unknown, run="evaluation", step=99)
            # These valid local evaluation citations are intentionally absent
            # from the frozen training database; persistence does not check them.
            receipt = frozen.save_investigations("evaluation", 1, [evaluation_row])
            self.assertFalse(receipt["persisted"])
            self.assertEqual(frozen.investigation_index(self.initial, exclude_run_id="evaluation"), before)
            self.assertEqual(path.read_bytes(), before_bytes)

    def test_old_frozen_archive_without_optional_table_is_explicitly_unavailable(self):
        path = self.memory.snapshot(Path(self.temp.name) / "old-frozen.sqlite3")
        with closing(sqlite3.connect(path)) as database:
            database.execute("DROP TABLE experience_investigation_snapshots")
            database.commit()
        with ExperienceStore(path, "project", "build-1", self.initial, read_only=True) as frozen:
            packet = frozen.investigation_index(self.initial, exclude_run_id="evaluation")
            self.assertEqual(packet["status"], "unavailable")
            self.assertEqual(packet["reason"], "archive_schema_unavailable")
            self.assertFalse(packet["complete"])

    def test_byte_budget_omits_whole_questions_and_anchors_with_explicit_counts(self):
        rows = [inquiry(self.initial)] + [inquiry(self.initial, identifier=f"p-{index}", status="pending") for index in range(3)]
        self.memory.save_investigations("train", 1, rows)
        packet = self.packet(max_bytes=1_024)
        self.assertLessEqual(byte_size(packet), 1_024)
        self.assertEqual(len(packet["investigations"]) + packet["omitted_count"], 4)
        self.assertFalse(packet["complete"])
        self.assertEqual(set(packet["applicability"]), {row["id"] for row in packet["investigations"]})
        references = {(ref["run_id"], ref["step"]) for row in packet["investigations"]
                      for ref in [row["last_transition"], *row["evidence"], *row["counterexamples"]]}
        self.assertEqual(packet["omitted_fact_count"], len(references) - len(packet["evidence_facts"]))

    def test_privacy_metadata_and_runtime_credentials_cannot_be_persisted(self):
        secret = "runtimeCredentialValue123"
        private = ExperienceStore(self.path, "project", "build-1", self.initial, secrets=(secret,))
        self.addCleanup(private.close)
        for mutate in (
            lambda row: row.update(question=secret),
            lambda row: row.update(evaluation={"label": "injected"}),
            lambda row: row.update(counterexamples=[{"run_id": "sk-abcdefghijklmnopqrstuv", "step": 1}]),
            lambda row: row["conditions"].update({"player.position": {"private_rule": "secret"}}),
            lambda row: row.update(condition_facets=["_capabilities.state_identity"], conditions={"_capabilities.state_identity": "private"}),
        ):
            row = inquiry(self.initial)
            mutate(row)
            with self.assertRaises(ValueError):
                private.save_investigations("train", 1, [row])
        with self.assertRaises(ValueError):
            private.save_investigations(secret, 1, [])
        self.assertNotIn(secret, json.dumps(private.investigation_index(self.initial, exclude_run_id="evaluation")))

    def test_closed_store_bad_budget_and_scope_mismatch_raise_without_fake_empty_success(self):
        with self.assertRaises(ScopeMismatch):
            self.packet(observation(game="other-game"))
        for budget in (True, 0, 1_023):
            with self.assertRaises(ValueError):
                self.packet(max_bytes=budget)
        self.memory.close()
        with self.assertRaises(ValueError):
            self.packet()
        with self.assertRaises(ValueError):
            self.memory.save_investigations("train", 1, [])


if __name__ == "__main__":
    unittest.main()
