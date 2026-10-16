"""Single-forecast fixtures keep model fields separate from rendered evidence."""

import copy
import json
from pathlib import Path
import tempfile
import unittest

from autoplay_qa.forecast import PREDICTION_SOURCE, experiment_context, prediction_record, reflection_receipt
from autoplay_qa.research import plan_context
from autoplay_qa.review import review_session
from autoplay_qa.session import replay
from tests.test_comparison_replay import note, pair
from tests.test_reflection_replay import reflection_recording


def plan(controls, mode="new", reason="", evidence=()):
    return {"mode": mode, "controls": list(controls), "revision_reason": reason, "evidence_steps": list(evidence)}


def single_forecast_recording(actions=("north", "interact", "reset", "north", "interact", "finish"), *, updates=None, plans=None, mismatch=False):
    """Transform explicit local game fixtures; this is never live model evidence."""
    rows = reflection_recording(actions, updates=updates, mismatch=mismatch)
    rows[0]["reflection_contract"] = 2
    rows[0]["prompt_version"] = "explicit-single-forecast-fixture"
    ledger, bindings, assessed, decisions = {}, {}, set(), {}
    active = pending = None
    for row in rows[1:]:
        kind = row["type"]
        number = row.get("request_number")
        if kind == "request":
            context = json.loads(row["messages"][1]["content"])
            context["active_experiment"] = ({key: copy.deepcopy(value) for key, value in active.items() if key != "plan"} if active is not None else None)
            context["reflection_contract"] = {
                "contract_version": 2, "pending_outcome": reflection_receipt(ledger, pending),
                "experiments": experiment_context(bindings, ledger, assessed),
            }
            context["research_context"] = plan_context(ledger, context["observation"])
            row["messages"][1]["content"] = json.dumps(context)
        elif kind == "completion":
            decision = json.loads(row["answer"])
            decision.pop("prediction")
            decision["plan"] = copy.deepcopy((plans or {}).get(number, plan([decision["action"]])))
            receipt = reflection_receipt(ledger, pending)
            if receipt is not None:
                decision["assessment"] = {
                    "step": pending, "outcome": decision["assessment"]["outcome"],
                    "prediction_verdict": receipt["prediction_verdict"],
                    "mismatched_facets": copy.deepcopy(receipt["mismatched_facets"]),
                }
            decisions[number] = decision
            row["answer"] = json.dumps(decision)
        elif kind == "memory":
            decision = decisions[number]
            receipt = reflection_receipt(ledger, pending)
            bindings[decision["experiment_id"]] = decision["experiment_control"]
            active = {
                "id": decision["experiment_id"], "goal": decision["experiment_goal"],
                "plan": copy.deepcopy(decision["plan"]), "control": decision["experiment_control"],
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
            decision = copy.deepcopy(decisions[number])
            row["decision"] = decision
            row.update(prediction_record(decision["action"], decision["expected_outcome"]))
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
                research_context=plan_context(ledger, row["final_observation"], finished=any(value["type"] == "finish" for value in rows)),
            )
    return rows


class SingleForecastReplayTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.path = Path(directory.name) / "explicit-single-forecast-fixture.jsonl"

    def write(self, records):
        self.path.write_text("\n".join(json.dumps(row) for row in records) + "\n")

    def replay(self, records):
        self.write(records)
        return replay(self.path)

    def answer(self, rows, request):
        row = next(row for row in rows if row["type"] == "completion" and row["request_number"] == request)
        return row, json.loads(row["answer"])

    def test_raw_model_decision_and_rendered_provenance_stay_separate(self):
        records = single_forecast_recording()
        result = self.replay(records)
        self.assertEqual(result["reflection_contract"], 2)
        self.assertEqual(result["validated_comparisons"], 2)
        self.assertEqual(result["reflected_steps"], 5)
        for row in records:
            if row["type"] == "completion":
                raw = json.loads(row["answer"])
                self.assertNotIn("prediction", raw)
                self.assertNotIn("prediction_source", raw)
                self.assertEqual(len(raw), 10)
            elif row["type"] in {"step", "reset", "finish"}:
                self.assertNotIn("prediction", row["decision"])
                self.assertNotIn("prediction_source", row["decision"])
                self.assertEqual(row["prediction_source"], PREDICTION_SOURCE)
                self.assertEqual(row["prediction"], prediction_record(row["decision"]["action"], row["decision"]["expected_outcome"])["prediction"])
        reviewed = review_session(self.path, include_evidence=True)
        self.assertEqual(reviewed["reflection_contract"]["version"], 2)
        self.assertTrue(reviewed["research_contract"]["execution_bound_controls_validated"])
        self.assertFalse(reviewed["research_contract"]["revision_reasons_verified"])
        citation = reviewed["hypothesis_evidence"]["notes"][0]["cited_steps"][0]
        self.assertNotIn("model_prediction", citation)
        self.assertEqual(citation["prediction_source"], PREDICTION_SOURCE)
        self.assertIn("Expected after north:", citation["rendered_prediction"])
        self.assertIn("model_expected_outcome", citation)
        self.assertIn("model_plan", citation)
        self.assertFalse(reviewed["learning_claims_verified"])

    def test_wrong_forecast_is_allowed_and_original_differing_facets_are_bound(self):
        records = single_forecast_recording(mismatch=True)
        self.replay(records)
        reviewed = review_session(self.path)
        receipt = reviewed["reflection_contract"]["recorded_reflections"][0]
        self.assertEqual(receipt["mismatched_facets"], ["position"])
        self.assertEqual(receipt["prediction_verdict"], "mismatched")
        self.assertEqual(receipt["expected_outcome"]["position"], [98, 99])
        _, answer = self.answer(records, 2)
        self.assertEqual(set(answer["assessment"]), {"step", "outcome", "prediction_verdict", "mismatched_facets"})

    def test_independently_written_prediction_fields_are_rejected(self):
        for field, value in (("prediction", "A conflicting gate-opening prediction."), ("prediction_source", "model_text")):
            records = single_forecast_recording()
            completion, decision = self.answer(records, 1)
            decision[field] = value
            completion["answer"] = json.dumps(decision)
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.replay(records)

    def test_executed_rows_require_exact_renderer_and_provenance_including_finish(self):
        for kind in ("step", "reset", "finish"):
            for field, bad_value in (("prediction", "Invented readable forecast"), ("prediction_source", "model_text")):
                records = single_forecast_recording()
                row = next(row for row in records if row["type"] == kind)
                row[field] = bad_value
                with self.subTest(kind=kind, field=field), self.assertRaisesRegex(ValueError, "rendering or provenance"):
                    self.replay(records)
            records = single_forecast_recording()
            next(row for row in records if row["type"] == kind).pop("prediction_source")
            with self.subTest(kind=kind, missing=True), self.assertRaisesRegex(ValueError, "rendering or provenance"):
                self.replay(records)

    def test_compact_assessment_cannot_use_new_prediction_or_wrong_facet_names(self):
        for field, bad_value in (("prediction_verdict", "matched"), ("mismatched_facets", []), ("mismatched_facets", ["gate_open"]), ("step", True)):
            records = single_forecast_recording(mismatch=True)
            completion, decision = self.answer(records, 2)
            decision["assessment"][field] = bad_value
            completion["answer"] = json.dumps(decision)
            with self.subTest(field=field, bad_value=bad_value), self.assertRaises(ValueError):
                self.replay(records)

    def test_request_and_summary_research_facts_are_required_and_reconstructed(self):
        for target, field in (("request", "pending_plan"), ("request", "current_state_action_attempts"), ("request", "session_finished"), ("summary", "session_finished"), ("request", "missing"), ("summary", "missing")):
            records = single_forecast_recording()
            row = records[-1] if target == "summary" else next(row for row in records if row["type"] == "request" and row["request_number"] == 2)
            value = json.loads(row["messages"][1]["content"]) if target == "request" else row
            if field == "missing":
                value.pop("research_context")
            elif field == "pending_plan":
                value["research_context"][field] = {"source_step": 1, "experiment_id": "movement", "controls": ["reset"]}
            elif field == "current_state_action_attempts":
                value["research_context"][field]["north"] = 99
            else:
                value["research_context"][field] = target != "summary"
            if target == "request":
                row["messages"][1]["content"] = json.dumps(value)
            with self.subTest(target=target, field=field), self.assertRaisesRegex(ValueError, "research"):
                self.replay(records)

    def test_continue_consumes_one_actual_control_and_finish_clears_tail(self):
        records = single_forecast_recording(
            ("north", "reset", "north", "finish"),
            updates={2: [note("north", [1])], 4: [note("north", [1, 3], pair(1, 3))]},
            plans={
                1: plan(["north", "reset", "north", "finish"]),
                2: plan(["reset", "north", "finish"], "continue"),
                3: plan(["north", "finish"], "continue"),
                4: plan(["finish"], "continue"),
            },
        )
        self.replay(records)
        requests = [json.loads(row["messages"][1]["content"]) for row in records if row["type"] == "request"]
        self.assertEqual(requests[1]["research_context"]["pending_plan"]["controls"], ["reset", "north", "finish"])
        self.assertEqual(requests[2]["research_context"]["pending_plan"]["controls"], ["north", "finish"])
        self.assertEqual(requests[3]["research_context"]["pending_plan"]["controls"], ["finish"])
        self.assertNotIn("plan", requests[1]["active_experiment"])
        self.assertTrue(records[-1]["research_context"]["session_finished"])
        self.assertIsNone(records[-1]["research_context"]["pending_plan"])
        reviewed = review_session(self.path)
        self.assertIsNone(reviewed["research_contract"]["progress"]["pending_plan"])

    def test_unexecuted_tail_remains_pending_at_budget_boundary(self):
        records = single_forecast_recording(("north",), updates={}, plans={1: plan(["north", "reset", "north"] )})
        self.replay(records)
        reviewed = review_session(self.path)
        self.assertEqual(reviewed["learning"]["unassessed_final_step"], 1)
        self.assertEqual(reviewed["research_contract"]["progress"]["pending_plan"]["controls"], ["reset", "north"])
        self.assertFalse(reviewed["research_contract"]["progress"]["session_finished"])

    def test_continue_cannot_silently_replace_or_drop_pending_controls(self):
        for mode, controls in (("continue", ["reset", "finish"]), ("new", ["reset", "north", "finish"])):
            records = single_forecast_recording(
                ("north", "reset", "north", "finish"),
                updates={2: [note("north", [1])], 4: [note("north", [1, 3], pair(1, 3))]},
                plans={
                    1: plan(["north", "reset", "north", "finish"]),
                    2: plan(["reset", "north", "finish"], "continue"),
                    3: plan(["north", "finish"], "continue"),
                    4: plan(["finish"], "continue"),
                },
            )
            completion, decision = self.answer(records, 2)
            decision["plan"] = plan(controls, mode)
            completion["answer"] = json.dumps(decision)
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                self.replay(records)

    def test_explicit_revision_uses_past_evidence_without_claiming_reason_truth(self):
        records = single_forecast_recording(plans={
            1: plan(["north", "reset", "north", "finish"]),
            2: plan(["interact"], "revise", "Inspect another chosen effect after the blocked trial.", [1]),
        })
        self.replay(records)
        reviewed = review_session(self.path)
        self.assertEqual(reviewed["research_contract"]["progress"]["plan_revisions"], 1)
        self.assertFalse(reviewed["research_contract"]["revision_reasons_verified"])
        for evidence in ([], [99], [True]):
            bad = copy.deepcopy(records)
            completion, decision = self.answer(bad, 2)
            decision["plan"]["evidence_steps"] = evidence
            completion["answer"] = json.dumps(decision)
            with self.subTest(evidence=evidence), self.assertRaises(ValueError):
                self.replay(bad)

    def test_canonical_raw_answer_cannot_be_changed_independently_of_execution(self):
        records = single_forecast_recording()
        completion, decision = self.answer(records, 1)
        decision["reason"] = "  " + decision["reason"]
        completion["answer"] = json.dumps(decision)
        with self.assertRaisesRegex(ValueError, "canonical model-written decision"):
            self.replay(records)

    def test_completed_reset_is_neutralized_and_remains_separate_from_trials(self):
        records = single_forecast_recording()
        self.replay(records)
        receipt = records[-1]["reflection_receipts"][2]
        self.assertEqual(receipt["control"], "reset")
        self.assertEqual((receipt["episode_before"], receipt["episode_after"]), (0, 1))
        self.assertEqual(receipt["observed_events"], ["Environment started."])
        reviewed = review_session(self.path)
        experiments = {item["id"]: item for item in reviewed["reflection_contract"]["experiments"]["experiments"]}
        self.assertEqual(experiments["movement"]["trial_steps"], [1, 4])
        self.assertEqual(experiments["movement"]["setup_steps"], [3])

    def test_memory_receipt_cannot_change_forecast_facets_or_renderer(self):
        for field, bad_value in (("mismatched_facets", ["key"]), ("prediction_source", "model_text"), ("prediction", "A second independent forecast")):
            records = single_forecast_recording()
            row = next(row for row in records if row["type"] == "memory" and row["request_number"] == 2)
            row["reflection_receipt"][field] = bad_value
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "memory receipt"):
                self.replay(records)

    def test_rejected_answer_repair_does_not_consume_a_plan_or_apply_learning(self):
        records = single_forecast_recording()
        index = next(index for index, row in enumerate(records) if row["type"] == "request" and row["request_number"] == 5)
        rejected = copy.deepcopy(records[index])
        for row in records[index:]:
            if "request_number" in row:
                row["request_number"] += 1
        records[index:index] = [
            rejected,
            {"type": "completion", "request_number": 5, "finish_reason": "stop", "answer": None},
            {"type": "rejected_decision", "request_number": 5, "error": "Explicit fixture conflicting prediction"},
        ]
        records[-1]["inference_requests"] += 1
        result = self.replay(records)
        self.assertEqual(result["reflected_steps"], 5)
        self.assertEqual(result["validated_comparisons"], 2)

    def test_compact_echoes_and_prose_still_do_not_verify_hypothesis_truth(self):
        records = single_forecast_recording()
        completion, decision = self.answer(records, 2)
        decision["assessment"]["outcome"] = "An incorrect freeform pickup assertion."
        completion["answer"] = json.dumps(decision)
        next(row for row in records if row["type"] == "memory" and row["request_number"] == 2)["assessment"]["outcome"] = decision["assessment"]["outcome"]
        next(row for row in records if row["type"] == "step" and row["request_number"] == 2)["decision"] = decision
        self.replay(records)
        reviewed = review_session(self.path, include_evidence=True)
        self.assertFalse(reviewed["reflection_contract"]["freeform_assessments_verified"])
        self.assertFalse(reviewed["hypothesis_evidence"]["claims_verified"])


if __name__ == "__main__":
    unittest.main()
