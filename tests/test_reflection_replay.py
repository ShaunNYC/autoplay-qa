"""Explicit local receipts validate forecasts and reflection, never model prose."""

import copy
import json
from pathlib import Path
import tempfile
import unittest

from autoplay_qa.comparisons import comparison_context
from autoplay_qa.reflection import experiment_context, outcome_projection, reflection_receipt
from autoplay_qa.review import review_session
from autoplay_qa.session import replay
from tests.test_comparison_replay import strict_recording


def reflection_recording(actions=("north", "interact", "reset", "north", "interact", "finish"), *, updates=None, mismatch=False):
    """Extend explicit comparison fixtures using their actual completed states."""
    records = strict_recording(actions, updates=updates)
    records[0]["reflection_contract"] = 1
    records[0]["prompt_version"] = "explicit-reflection-fixture"
    outcomes = {
        row["request_number"]: row["after"]
        for row in records if row["type"] in {"step", "reset"}
    }
    ledger, bindings, assessed, accepted = {}, {}, set(), {}
    pending = None
    active = None
    for row in records[1:]:
        kind = row["type"]
        number = row.get("request_number")
        if kind == "request":
            context = json.loads(row["messages"][1]["content"])
            context["active_experiment"] = copy.deepcopy(active)
            context["comparison_contract"] = comparison_context(ledger, assessed, pending)
            context["reflection_contract"] = {
                "contract_version": 1,
                "pending_outcome": reflection_receipt(ledger, pending),
                "experiments": experiment_context(bindings, ledger, assessed),
            }
            row["messages"][1]["content"] = json.dumps(context)
        elif kind == "completion":
            decision = json.loads(row["answer"])
            action = decision["action"]
            target = "interact" if decision["experiment_id"] == "interaction" else "north"
            decision.update(
                expected_outcome=None if action == "finish" else outcome_projection(outcomes[number]),
                experiment_control=target,
                action_role="finish" if action == "finish" else "setup" if action == "reset" else "trial",
            )
            if mismatch and number == 1:
                decision["expected_outcome"]["position"] = [98, 99]
            receipt = reflection_receipt(ledger, pending)
            if receipt is not None:
                decision["assessment"] = {
                    **{field: value for field, value in receipt.items() if field != "expected_outcome"},
                    "outcome": decision["assessment"]["outcome"],
                }
            accepted[number] = decision
            row["answer"] = json.dumps(decision)
        elif kind == "memory":
            decision = accepted[number]
            receipt = reflection_receipt(ledger, pending)
            bindings[decision["experiment_id"]] = decision["experiment_control"]
            active = {
                "id": decision["experiment_id"], "goal": decision["experiment_goal"],
                "plan": decision["plan"], "control": decision["experiment_control"],
                "action_role": decision["action_role"],
            }
            row.update(
                assessment=copy.deepcopy(decision["assessment"]), active_experiment=copy.deepcopy(active),
                experiment_bindings=copy.deepcopy(bindings), reflection_receipt=copy.deepcopy(receipt),
            )
            if pending is not None:
                assessed.add(pending)
            pending = None
        elif kind in {"step", "reset", "finish"}:
            row["decision"] = copy.deepcopy(accepted[number])
            if kind != "finish":
                ledger[row["step"]] = row
                pending = row["step"]
        elif kind == "summary":
            row.update(
                active_experiment=copy.deepcopy(active), experiment_bindings=copy.deepcopy(bindings),
                reflection_receipts=[
                    {**reflection_receipt(ledger, step), "assessed": step in assessed}
                    for step in sorted(ledger)
                ],
            )
    return records


class ReflectionReplayTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.path = Path(directory.name) / "explicit-reflection-fixture.jsonl"

    def write(self, records):
        self.path.write_text("\n".join(json.dumps(row) for row in records) + "\n")

    def replay(self, records):
        self.write(records)
        return replay(self.path)

    def answer(self, records, request):
        row = next(row for row in records if row["type"] == "completion" and row["request_number"] == request)
        return row, json.loads(row["answer"])

    def memory(self, records, request):
        return next(row for row in records if row["type"] == "memory" and row["request_number"] == request)

    def test_complete_reflection_is_replayed_without_proving_claims(self):
        result = self.replay(reflection_recording())
        self.assertEqual(result["reflection_contract"], 1)
        self.assertTrue(result["reflection_bindings_validated"])
        self.assertEqual(result["reflected_steps"], 5)
        self.assertEqual(result["validated_comparisons"], 2)
        self.assertFalse(result["learning_claims_verified"])
        reviewed = review_session(self.path, include_evidence=True)
        self.assertEqual(reviewed["reflection_contract"]["prediction_matches"], 5)
        self.assertEqual(reviewed["reflection_contract"]["prediction_mismatches"], 0)
        self.assertEqual(reviewed["reflection_contract"]["assessed_steps"], [1, 2, 3, 4, 5])
        self.assertFalse(reviewed["reflection_contract"]["freeform_assessments_verified"])
        self.assertEqual(reviewed["coverage"]["tested_experiment_count"], 2)
        self.assertEqual(reviewed["goal_checks"]["reset_retested_control_families"], {"movement": True, "interaction": True})
        cited = reviewed["hypothesis_evidence"]["notes"][0]["cited_steps"][0]
        self.assertEqual(cited["action_role"], "trial")
        self.assertEqual(cited["experiment_control"], "north")
        self.assertTrue(cited["typed_reflection_validated"])
        self.assertFalse(reviewed["hypothesis_evidence"]["assessments_verified"])

    def test_incorrect_forecast_is_valid_and_bound_to_original_prediction(self):
        result = self.replay(reflection_recording(mismatch=True))
        self.assertEqual(result["reflected_steps"], 5)
        reviewed = review_session(self.path)
        self.assertEqual(reviewed["reflection_contract"]["prediction_mismatches"], 1)
        receipt = reviewed["reflection_contract"]["recorded_reflections"][0]
        self.assertEqual(receipt["expected_outcome"]["position"], [98, 99])
        self.assertNotEqual(receipt["expected_outcome"], receipt["observed_outcome"])

    def test_typed_assessment_cannot_invent_state_feedback_or_prediction_match(self):
        alterations = (
            ("position_before", [9, 9]),
            ("observed_outcome", {"position": [1, 1], "inventory": ["key"], "key": None, "gate_open": False, "status": "playing"}),
            ("observed_events", ["A predicted pickup became an observed pickup."]),
            ("prediction_verdict", "mismatched"),
        )
        for field, value in alterations:
            records = reflection_recording()
            completion, decision = self.answer(records, 2)
            decision["assessment"][field] = value
            completion["answer"] = json.dumps(decision)
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "actual recorded reflection receipt"):
                self.replay(records)

    def test_typed_reflection_does_not_verify_free_assessment_prose(self):
        records = reflection_recording()
        completion, decision = self.answer(records, 2)
        decision["assessment"]["outcome"] = "An incorrect freeform claim of a pickup."
        completion["answer"] = json.dumps(decision)
        self.memory(records, 2)["assessment"]["outcome"] = decision["assessment"]["outcome"]
        next(row for row in records if row["type"] == "step" and row["request_number"] == 2)["decision"] = decision
        self.replay(records)
        reviewed = review_session(self.path, include_evidence=True)
        self.assertFalse(reviewed["reflection_contract"]["freeform_assessments_verified"])

    def test_receipts_and_bindings_are_mandatory_even_initially_empty(self):
        for field in ("experiment_bindings", "reflection_receipt"):
            records = reflection_recording()
            self.memory(records, 1).pop(field)
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "reflection memory"):
                self.replay(records)

    def test_memory_receipt_and_binding_cannot_be_tampered_independently(self):
        for mutation in ("receipt", "bindings", "active_role"):
            records = reflection_recording()
            memory = self.memory(records, 2)
            if mutation == "receipt":
                memory["reflection_receipt"]["observed_outcome"]["inventory"] = ["key"]
            elif mutation == "bindings":
                memory["experiment_bindings"]["movement"] = "south"
            else:
                memory["active_experiment"]["action_role"] = "setup"
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                self.replay(records)

    def test_recorded_forecast_cannot_be_changed_after_reflection(self):
        records = reflection_recording()
        for row in records:
            if row.get("request_number") != 1:
                continue
            if row["type"] == "completion":
                decision = json.loads(row["answer"])
                decision["expected_outcome"]["inventory"] = ["key"]
                row["answer"] = json.dumps(decision)
            elif row["type"] == "step":
                row["decision"]["expected_outcome"]["inventory"] = ["key"]
        with self.assertRaisesRegex(ValueError, "request facts"):
            self.replay(records)

    def test_trial_target_and_reused_experiment_are_bound(self):
        for mutation in ("wrong_target", "changed_target", "reset_trial", "finish_trial", "finish_forecast"):
            records = reflection_recording()
            request = 3 if mutation == "reset_trial" else 6 if mutation.startswith("finish_") else 4
            completion, decision = self.answer(records, request)
            if mutation in {"wrong_target", "changed_target"}:
                decision["experiment_control"] = "south"
                if mutation == "changed_target":
                    decision["action_role"] = "setup"
            elif mutation in {"reset_trial", "finish_trial"}:
                decision["action_role"] = "trial"
            else:
                decision["expected_outcome"] = outcome_projection(records[0]["initial_observation"])
            completion["answer"] = json.dumps(decision)
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                self.replay(records)

    def test_setup_controls_and_different_experiment_ids_cannot_form_comparison(self):
        for mutation in ("setup", "different_question"):
            records = reflection_recording()
            completion, decision = self.answer(records, 4)
            if mutation == "setup":
                decision["action_role"] = "setup"
            else:
                decision["experiment_id"] = "movement-two"
            completion["answer"] = json.dumps(decision)
            memory = self.memory(records, 4)
            memory["active_experiment"]["action_role"] = decision["action_role"]
            memory["active_experiment"]["id"] = decision["experiment_id"]
            if mutation == "different_question":
                memory["experiment_bindings"]["movement-two"] = "north"
            row = next(row for row in records if row["type"] == "step" and row["request_number"] == 4)
            row["decision"] = decision
            request = next(row for row in records if row["type"] == "request" and row["request_number"] == 5)
            context = json.loads(request["messages"][1]["content"])
            context["active_experiment"] = memory["active_experiment"]
            ledger = {row["step"]: row for row in records if row["type"] in {"step", "reset"} and row["step"] <= 4}
            context["reflection_contract"]["experiments"] = experiment_context(memory["experiment_bindings"], ledger, {1, 2, 3})
            request["messages"][1]["content"] = json.dumps(context)
            with self.subTest(mutation=mutation), self.assertRaisesRegex(ValueError, "pair steps|same actual experiment"):
                self.replay(records)

    def test_request_reflection_facts_are_reconstructed(self):
        for mutation in ("snapshot", "verdict", "future_trial", "boolean_version"):
            records = reflection_recording()
            request = next(row for row in records if row["type"] == "request" and row["request_number"] == 2)
            context = json.loads(request["messages"][1]["content"])
            facts = context["reflection_contract"]
            if mutation == "snapshot":
                facts["pending_outcome"]["position_before"] = [8, 8]
            elif mutation == "verdict":
                facts["pending_outcome"]["prediction_verdict"] = "mismatched"
            elif mutation == "future_trial":
                facts["experiments"]["experiments"][0]["trial_steps"].append(99)
            else:
                facts["contract_version"] = True
            request["messages"][1]["content"] = json.dumps(context)
            with self.subTest(mutation=mutation), self.assertRaisesRegex(ValueError, "request facts"):
                self.replay(records)

    def with_trial_progress(self, records):
        from autoplay_qa.reflection import trial_progress

        ledger, bindings, assessed = {}, {}, set()
        for row in records:
            if row["type"] == "request":
                context = json.loads(row["messages"][1]["content"])
                context.setdefault("question_progress", {})["declared_trial_progress"] = trial_progress(bindings, ledger, assessed)
                row["messages"][1]["content"] = json.dumps(context)
            elif row["type"] == "memory":
                bindings = row["experiment_bindings"]
                if row["assessment"] is not None:
                    assessed.add(row["assessment"]["step"])
            elif row["type"] in {"step", "reset"}:
                ledger[row["step"]] = row
        return records

    def test_optional_trial_progress_replays_and_does_not_assess_pending_step(self):
        records = self.with_trial_progress(reflection_recording())
        result = self.replay(records)
        self.assertEqual(result["reflected_steps"], 5)
        request = next(row for row in records if row["type"] == "request" and row["request_number"] == 6)
        context = json.loads(request["messages"][1]["content"])
        progress = {item["id"]: item for item in context["question_progress"]["declared_trial_progress"]["experiments"]}
        self.assertEqual(progress["interaction"]["trial_count"], 2)
        self.assertEqual(progress["interaction"]["assessed_trial_count"], 1)
        self.assertEqual(progress["interaction"]["assessed_cross_episode_pair_count"], 0)
        self.assertEqual(progress["movement"]["assessed_cross_episode_pair_count"], 1)

    def test_optional_trial_progress_cannot_invent_trials_or_verified_claims(self):
        for mutation in ("extra_trial", "claim_truth", "boolean_count"):
            records = self.with_trial_progress(reflection_recording())
            request = next(row for row in records if row["type"] == "request" and row["request_number"] == 2)
            context = json.loads(request["messages"][1]["content"])
            progress = context["question_progress"]["declared_trial_progress"]
            if mutation == "extra_trial":
                progress["experiments"][0]["trial_count"] = 99
            elif mutation == "claim_truth":
                progress["claims_verified"] = True
            else:
                progress["experiments"][0]["trial_count"] = True
            request["messages"][1]["content"] = json.dumps(context)
            with self.subTest(mutation=mutation), self.assertRaisesRegex(ValueError, "declared trial progress"):
                self.replay(records)

    def test_summary_bindings_receipts_and_assessed_flags_are_checked(self):
        for mutation in ("bindings", "receipt", "assessed", "missing_receipts"):
            records = reflection_recording()
            summary = records[-1]
            if mutation == "bindings":
                summary["experiment_bindings"] = {}
            elif mutation == "receipt":
                summary["reflection_receipts"][0]["expected_outcome"]["position"] = [9, 9]
            elif mutation == "assessed":
                summary["reflection_receipts"][0]["assessed"] = False
            else:
                summary.pop("reflection_receipts")
            with self.subTest(mutation=mutation), self.assertRaisesRegex(ValueError, "reflection summary"):
                self.replay(records)

    def test_rejected_response_repair_preserves_memory_and_pending_receipt(self):
        records = reflection_recording()
        index = next(index for index, row in enumerate(records) if row["type"] == "request" and row["request_number"] == 5)
        rejected_request = copy.deepcopy(records[index])
        for row in records[index:]:
            if "request_number" in row:
                row["request_number"] += 1
        records[index:index] = [
            rejected_request,
            {"type": "completion", "request_number": 5, "finish_reason": "stop", "answer": None},
            {"type": "rejected_decision", "request_number": 5, "error": "Explicit fixture malformed reflection"},
        ]
        records[-1]["inference_requests"] += 1
        result = self.replay(records)
        self.assertEqual(result["reflected_steps"], 5)
        self.assertEqual(result["validated_comparisons"], 2)

    def test_setup_interaction_does_not_count_as_tested_family(self):
        records = reflection_recording(("north", "interact", "finish"))
        completion, decision = self.answer(records, 2)
        decision["action_role"] = "setup"
        completion["answer"] = json.dumps(decision)
        memory = self.memory(records, 2)
        memory["active_experiment"]["action_role"] = "setup"
        next(row for row in records if row["type"] == "step" and row["request_number"] == 2)["decision"] = decision
        request = next(row for row in records if row["type"] == "request" and row["request_number"] == 3)
        context = json.loads(request["messages"][1]["content"])
        context["active_experiment"] = memory["active_experiment"]
        ledger = {row["step"]: row for row in records if row["type"] == "step"}
        context["reflection_contract"]["experiments"] = experiment_context(memory["experiment_bindings"], ledger, {1})
        request["messages"][1]["content"] = json.dumps(context)
        self.replay(records)
        reviewed = review_session(self.path)
        self.assertTrue(reviewed["coverage"]["interaction"])
        self.assertEqual(reviewed["coverage"]["tested_experiment_count"], 1)
        self.assertFalse(reviewed["goal_checks"]["mechanics_coverage_observed"])

    def test_final_unassessed_action_is_preserved(self):
        records = reflection_recording(("north", "interact", "reset", "north", "interact"))
        result = self.replay(records)
        self.assertEqual(result["reflected_steps"], 4)
        reviewed = review_session(self.path)
        self.assertEqual(reviewed["learning"]["unassessed_final_step"], 5)
        self.assertFalse(reviewed["reflection_contract"]["recorded_reflections"][-1]["assessed"])
        self.assertEqual(reviewed["goal_checks"]["next_stage_candidate"], "exploration_validation")

    def test_interaction_focus_completion_does_not_complete_full_milestone(self):
        records = reflection_recording(
            ("interact", "reset", "interact", "finish"),
            updates={
                2: [{"id": "h-interact", "claim": "Recorded interact effect from the initial state.", "status": "uncertain", "evidence_steps": [1], "control": "interact", "comparison": None}],
                4: [{"id": "h-interact", "claim": "Recorded interact effect from the initial state.", "status": "supported", "evidence_steps": [1, 3], "control": "interact", "comparison": {"original_step": 1, "repeated_step": 3, "original_episode": 0, "repeated_episode": 1, "result": "same"}}],
            },
        )
        # The two interaction trials retain their actual experiment identity;
        # setup controls and finish need no extra tested family.
        records[0]["focus"] = "interaction-retention"
        self.replay(records)
        reviewed = review_session(self.path)
        self.assertTrue(reviewed["goal_checks"]["focused_question_observed"])
        self.assertEqual(reviewed["goal_checks"]["reset_retested_control_families"], {"movement": False, "interaction": True})
        self.assertEqual(reviewed["goal_checks"]["next_stage_candidate"], "exploration_validation")
        self.assertIn("both movement and interaction", reviewed["goal_checks"]["focus_note"])

    def test_unknown_or_unbound_reflection_contract_headers_are_rejected(self):
        for mutation in ("unknown", "boolean", "no_comparison", "wrong_schema"):
            records = reflection_recording()
            if mutation == "unknown":
                records[0]["reflection_contract"] = 5
            elif mutation == "boolean":
                records[0]["reflection_contract"] = True
            elif mutation == "no_comparison":
                records[0].pop("comparison_contract")
            else:
                records[0]["schema_version"] = 1
            with self.subTest(mutation=mutation), self.assertRaisesRegex(ValueError, "contract"):
                self.replay(records)

    def test_absent_contract_preserves_legacy_comparison_reviews(self):
        self.replay(strict_recording())
        reviewed = review_session(self.path)
        self.assertNotIn("reflection_contract", reviewed)
        self.assertNotIn("tested_experiment_count", reviewed["coverage"])


if __name__ == "__main__":
    unittest.main()
