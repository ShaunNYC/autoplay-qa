"""Replayed actual state changes stay separate from fallible model forecasts."""

import copy
import json
from pathlib import Path
import tempfile
import unittest

from autoplay_qa.grounding import experiment_context, outcome_projection, prediction_record, reflection_receipt
from autoplay_qa.research import plan_context
from autoplay_qa.review import review_session
from autoplay_qa.session import replay
from tests.test_comparison_replay import note, pair, strict_recording


def grounded_recording(actions=("north", "interact", "reset", "north", "interact", "finish"), *, updates=None, forecasts=None):
    """Create explicit local fixtures from actual deterministic game execution."""
    rows = strict_recording(actions, updates=updates)
    rows[0]["reflection_contract"] = 3
    rows[0]["prompt_version"] = "explicit-grounding-fixture"
    outcomes, targets = {}, {}
    for row in rows:
        if row["type"] in {"step", "reset"}:
            outcomes[row["request_number"]] = row["after"]
        if row["type"] == "step":
            targets.setdefault(row["decision"]["experiment_id"], row["decision"]["action"])
    ledger, bindings, assessed, decisions = {}, {}, set(), {}
    pending = active = None
    for row in rows[1:]:
        kind, number = row["type"], row.get("request_number")
        if kind == "request":
            context = json.loads(row["messages"][1]["content"])
            context["active_experiment"] = ({key: copy.deepcopy(value) for key, value in active.items() if key != "plan"} if active is not None else None)
            context["reflection_contract"] = {
                "contract_version": 3, "pending_outcome": reflection_receipt(ledger, pending),
                "experiments": experiment_context(bindings, ledger, assessed),
            }
            context["research_context"] = plan_context(ledger, context["observation"])
            row["messages"][1]["content"] = json.dumps(context)
        elif kind == "completion":
            decision = json.loads(row["answer"])
            decision.pop("prediction")
            action = decision["action"]
            target = targets.get(decision["experiment_id"], "north")
            decision.update(
                expected_outcome=None if action == "finish" else outcome_projection(outcomes[number]),
                experiment_control=target,
                action_role="finish" if action == "finish" else "trial" if action == target else "setup",
                plan={"mode": "new", "controls": [action], "revision_reason": "", "evidence_steps": []},
            )
            if number in (forecasts or {}):
                decision["expected_outcome"].update(copy.deepcopy(forecasts[number]))
            receipt = reflection_receipt(ledger, pending)
            decision["assessment"] = ({
                "step": pending, "outcome": "Compare the recorded result with the original five-facet forecast.",
                "prediction_verdict": receipt["prediction_verdict"],
                "mismatched_facets": copy.deepcopy(receipt["mismatched_facets"]),
            } if receipt is not None else None)
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


class GroundingReplayTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.path = Path(directory.name) / "explicit-grounding-fixture.jsonl"

    def write(self, rows):
        self.path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")

    def replay(self, rows):
        self.write(rows)
        return replay(self.path)

    def test_grounded_receipts_separate_actual_changes_from_forecast_mismatches(self):
        rows = grounded_recording(forecasts={1: {"key": None}})
        result = self.replay(rows)
        self.assertEqual(result["reflection_contract"], 3)
        self.assertEqual(result["validated_comparisons"], 2)
        reviewed = review_session(self.path, include_evidence=True)
        receipt = reviewed["reflection_contract"]["recorded_reflections"][0]
        self.assertEqual(receipt["before_outcome"]["key"], [1, 3])
        self.assertEqual(receipt["observed_outcome"]["key"], [1, 3])
        self.assertIsNone(receipt["expected_outcome"]["key"])
        self.assertEqual(receipt["mismatched_facets"], ["key"])
        self.assertEqual(receipt["actual_changed_facets"], [])
        evidence = reviewed["hypothesis_evidence"]["notes"][0]["cited_steps"][0]
        self.assertEqual(evidence["actual_before_outcome"]["key"], [1, 3])
        self.assertEqual(evidence["actual_changed_facets"], [])
        self.assertNotIn("model_prediction", evidence)
        self.assertFalse(reviewed["hypothesis_evidence"]["claims_verified"])

    def test_actual_movement_and_reset_changes_bind_distinct_before_positions(self):
        rows = grounded_recording(
            ("east", "reset", "east", "finish"),
            updates={2: [note("east", [1])], 4: [note("east", [1, 3], pair(1, 3))]},
        )
        self.replay(rows)
        receipts = rows[-1]["reflection_receipts"]
        self.assertEqual(receipts[0]["before_outcome"]["position"], [1, 1])
        self.assertEqual(receipts[0]["observed_outcome"]["position"], [2, 1])
        self.assertEqual(receipts[0]["actual_changed_facets"], ["position"])
        self.assertEqual(receipts[1]["before_outcome"]["position"], [2, 1])
        self.assertEqual(receipts[1]["observed_outcome"]["position"], [1, 1])
        self.assertEqual(receipts[1]["actual_changed_facets"], ["position"])
        self.assertEqual(receipts[1]["observed_events"], ["Environment started."])
        self.assertEqual(receipts[2]["before_outcome"]["position"], [1, 1])

    def test_request_cannot_invent_before_state_or_observed_changes(self):
        for field, mutation in (("before_outcome", "position"), ("before_outcome", "boolean"), ("actual_changed_facets", "pickup")):
            rows = grounded_recording()
            request = next(row for row in rows if row["type"] == "request" and row["request_number"] == 2)
            context = json.loads(request["messages"][1]["content"])
            receipt = context["reflection_contract"]["pending_outcome"]
            if mutation == "position":
                receipt[field]["position"] = [9, 9]
            elif mutation == "boolean":
                receipt[field]["gate_open"] = 0
            else:
                receipt[field] = ["inventory", "key"]
            request["messages"][1]["content"] = json.dumps(context)
            with self.subTest(mutation=mutation), self.assertRaisesRegex(ValueError, "reflection request facts"):
                self.replay(rows)

    def test_memory_and_summary_cannot_turn_a_wrong_forecast_into_actual_changes(self):
        for kind in ("memory", "summary"):
            for field in ("before_outcome", "actual_changed_facets"):
                rows = grounded_recording(forecasts={1: {"position": [98, 99]}})
                if kind == "memory":
                    row = next(row for row in rows if row["type"] == "memory" and row["request_number"] == 2)
                    receipt = row["reflection_receipt"]
                else:
                    receipt = rows[-1]["reflection_receipts"][0]
                if field == "before_outcome":
                    receipt[field]["inventory"] = ["key"]
                else:
                    receipt[field] = ["position"]
                with self.subTest(kind=kind, field=field), self.assertRaisesRegex(ValueError, "memory receipt|summary receipts"):
                    self.replay(rows)

    def test_replay_rejects_divergence_in_the_actual_before_state(self):
        rows = grounded_recording()
        next(row for row in rows if row["type"] == "step")["before"]["player"]["position"] = [3, 1]
        with self.assertRaisesRegex(ValueError, "diverged before"):
            self.replay(rows)

    def test_final_unassessed_grounded_result_remains_explicit(self):
        rows = grounded_recording(("north",), updates={}, forecasts={1: {"key": None}})
        self.replay(rows)
        reviewed = review_session(self.path)
        self.assertEqual(reviewed["learning"]["unassessed_final_step"], 1)
        receipt = reviewed["reflection_contract"]["recorded_reflections"][0]
        self.assertFalse(receipt["assessed"])
        self.assertEqual(receipt["actual_changed_facets"], [])
        self.assertEqual(receipt["mismatched_facets"], ["key"])
        self.assertEqual(reviewed["goal_checks"]["next_stage_candidate"], "exploration_validation")


if __name__ == "__main__":
    unittest.main()
