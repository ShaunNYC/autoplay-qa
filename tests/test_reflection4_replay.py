"""Witnessed effects and scoped trial notes validate samples, never rule truth."""

import copy
import json
from pathlib import Path
import tempfile
import unittest

from autoplay_qa.measurements import comparison_receipts, evidence_measurements, experiment_context, outcome_projection, reflection_receipt
from autoplay_qa.research2 import plan_context, scope_progress
from autoplay_qa.review import review_session
from autoplay_qa.session import replay
from tests.test_grounding_replay import grounded_recording


def witnessed_note(note, ledger):
    value = copy.deepcopy(note)
    refs = value["evidence_steps"]
    if not refs:
        value["effect"] = None
        return value
    step = refs[0]
    facet = "gate_open" if value["control"] == "interact" else "position"
    row = ledger[step]
    value["effect"] = {
        "condition_step": step, "facet": facet,
        "before": outcome_projection(row["before"])[facet],
        "after": outcome_projection(row["after"])[facet],
    }
    return value


def reflection4_recording(*, retarget=False):
    """Extend explicit local game fixtures with chosen, sampled effect values."""
    rows = grounded_recording()
    rows[0]["reflection_contract"] = 4
    rows[0]["prompt_version"] = "explicit-reflection4-fixture"
    ledger, bindings, notes, assessed, decisions = {}, {}, {}, set(), {}
    pending = active = None
    for row in rows[1:]:
        kind, number = row["type"], row.get("request_number")
        if kind == "request":
            context = json.loads(row["messages"][1]["content"])
            context["hypotheses"] = copy.deepcopy(list(notes.values()))
            context["active_experiment"] = ({key: copy.deepcopy(value) for key, value in active.items() if key != "plan"} if active is not None else None)
            context["reflection_contract"] = {
                "contract_version": 4, "pending_outcome": reflection_receipt(ledger, pending),
                "experiments": experiment_context(bindings, ledger, assessed),
            }
            context["research_context"] = plan_context(ledger, context["observation"])
            context["experiment_progress"] = scope_progress(ledger, assessed, notes)
            row["messages"][1]["content"] = json.dumps(context)
        elif kind == "completion":
            decision = json.loads(row["answer"])
            if retarget and number == 1:
                decision["plan"]["controls"] = ["north", "interact"]
            elif retarget and number == 2:
                decision["plan"].update(mode="retarget", revision_reason="Inspect another chosen effect using the blocked trial as evidence.", evidence_steps=[1])
            decision["hypotheses"] = [witnessed_note(note, ledger) for note in decision["hypotheses"]]
            if decision["assessment"] is not None:
                decision["assessment"]["actual_changed_facets"] = reflection_receipt(ledger, pending)["actual_changed_facets"]
            decisions[number] = decision
            row["answer"] = json.dumps(decision)
        elif kind == "memory":
            decision = decisions[number]
            receipt = reflection_receipt(ledger, pending)
            bindings[decision["experiment_id"]] = decision["experiment_control"]
            notes.update({note["id"]: copy.deepcopy(note) for note in decision["hypotheses"]})
            active = {
                "id": decision["experiment_id"], "goal": decision["experiment_goal"],
                "plan": copy.deepcopy(decision["plan"]), "control": decision["experiment_control"],
                "action_role": decision["action_role"],
            }
            row.update(
                assessment=copy.deepcopy(decision["assessment"]), active_experiment=copy.deepcopy(active),
                experiment_bindings=copy.deepcopy(bindings), reflection_receipt=copy.deepcopy(receipt),
                hypotheses=copy.deepcopy(list(notes.values())),
                comparison_receipts=comparison_receipts(notes, ledger),
                evidence_measurements=evidence_measurements(notes, ledger),
            )
            if pending is not None:
                assessed.add(pending)
            pending = None
        elif kind in {"step", "reset", "finish"}:
            row["decision"] = copy.deepcopy(decisions[number])
            if kind != "finish":
                ledger[row["step"]] = row
                pending = row["step"]
        elif kind == "summary":
            row.update(
                active_experiment=copy.deepcopy(active), experiment_bindings=copy.deepcopy(bindings),
                hypotheses=copy.deepcopy(list(notes.values())),
                comparison_receipts=comparison_receipts(notes, ledger),
                reflection_receipts=[
                    {**reflection_receipt(ledger, step), "assessed": step in assessed}
                    for step in sorted(ledger)
                ],
                evidence_measurements=evidence_measurements(notes, ledger),
                experiment_progress=scope_progress(ledger, assessed, notes),
                research_context=plan_context(ledger, row["final_observation"], finished=True),
            )
    return rows


class Reflection4ReplayTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.path = Path(directory.name) / "explicit-reflection4-fixture.jsonl"

    def write(self, rows):
        self.path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")

    def replay(self, rows):
        self.write(rows)
        return replay(self.path)

    def answer(self, rows, request):
        row = next(row for row in rows if row["type"] == "completion" and row["request_number"] == request)
        return row, json.loads(row["answer"])

    def test_bound_effects_replay_and_evidence_marks_sampled_values(self):
        rows = reflection4_recording()
        result = self.replay(rows)
        self.assertEqual(result["reflection_contract"], 4)
        self.assertEqual(result["validated_comparisons"], 2)
        self.assertEqual(result["reflected_steps"], 5)
        reviewed = review_session(self.path, include_evidence=True)
        self.assertEqual(reviewed["reflection_contract"]["version"], 4)
        self.assertEqual(reviewed["research_contract"]["version"], 2)
        self.assertEqual(reviewed["experiment_progress"], rows[-1]["experiment_progress"])
        self.assertTrue(reviewed["hypothesis_evidence"]["effect_bindings_validated"])
        self.assertFalse(reviewed["hypothesis_evidence"]["claims_verified"])
        north = reviewed["hypothesis_evidence"]["notes"][0]
        self.assertEqual(north["model_effect"], {"condition_step": 1, "facet": "position", "before": [1, 1], "after": [1, 1]})
        self.assertEqual(north["evidence_measurements"][0]["source"], "actual_recorded_state")
        self.assertFalse(north["evidence_measurements"][0]["claims_verified"])
        self.assertEqual([measurement["step"] for measurement in north["evidence_measurements"][0]["measurements"]], [1, 4])
        self.assertEqual([measurement["episode"] for measurement in north["evidence_measurements"][0]["measurements"]], [0, 1])
        self.assertFalse(reviewed["learning_claims_verified"])

    def test_witness_before_after_and_facet_types_are_bound_to_actual_state(self):
        for mutation, value in (("before", [9, 9]), ("after", [2, 1]), ("condition_step", True), ("condition_step", 99), ("facet", "tick")):
            rows = reflection4_recording()
            completion, decision = self.answer(rows, 2)
            decision["hypotheses"][0]["effect"][mutation] = value
            completion["answer"] = json.dumps(decision)
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                self.replay(rows)

    def test_effect_cannot_invent_a_pickup_during_refused_interaction(self):
        rows = reflection4_recording()
        completion, decision = self.answer(rows, 3)
        decision["hypotheses"][0]["effect"] = {"condition_step": 2, "facet": "inventory", "before": [], "after": ["key"]}
        completion["answer"] = json.dumps(decision)
        with self.assertRaises(ValueError):
            self.replay(rows)

    def test_witness_must_be_a_cited_assessed_game_control_not_reset_or_other_control(self):
        for step in (2, 3, 99):
            rows = reflection4_recording()
            completion, decision = self.answer(rows, 5)
            decision["hypotheses"][0]["effect"]["condition_step"] = step
            completion["answer"] = json.dumps(decision)
            with self.subTest(step=step), self.assertRaises(ValueError):
                self.replay(rows)

    def test_five_field_assessment_separates_actual_changes_from_forecast_errors(self):
        rows = reflection4_recording()
        completion, decision = self.answer(rows, 2)
        self.assertEqual(len(decision["assessment"]), 5)
        self.assertEqual(decision["assessment"]["actual_changed_facets"], [])
        decision["assessment"]["actual_changed_facets"] = ["key"]
        completion["answer"] = json.dumps(decision)
        with self.assertRaises(ValueError):
            self.replay(rows)

    def test_first_assessed_trial_of_each_experiment_requires_its_note(self):
        rows = reflection4_recording()
        completion, decision = self.answer(rows, 3)
        decision["hypotheses"] = []
        completion["answer"] = json.dumps(decision)
        with self.assertRaisesRegex(ValueError, "experiment trial|first assessed|assessed experiment"):
            self.replay(rows)

    def test_memory_and_summary_measurements_are_required_and_canonical(self):
        for kind in ("memory", "summary"):
            for mutation in ("missing", "fake_value", "wrong_episode", "truth_flag"):
                rows = reflection4_recording()
                row = rows[-1] if kind == "summary" else next(row for row in rows if row["type"] == "memory" and row["request_number"] == 5)
                if mutation == "missing":
                    row.pop("evidence_measurements")
                elif mutation == "fake_value":
                    row["evidence_measurements"][0]["measurements"][0]["after"] = [9, 9]
                elif mutation == "wrong_episode":
                    row["evidence_measurements"][0]["measurements"][0]["episode"] = 10
                else:
                    row["evidence_measurements"][0]["claims_verified"] = True
                with self.subTest(kind=kind, mutation=mutation), self.assertRaisesRegex(ValueError, "measurements"):
                    self.replay(rows)

    def test_even_initial_empty_measurements_must_be_recorded(self):
        rows = reflection4_recording()
        next(row for row in rows if row["type"] == "memory" and row["request_number"] == 1).pop("evidence_measurements")
        with self.assertRaisesRegex(ValueError, "measurements"):
            self.replay(rows)

    def test_request_and_summary_scope_progress_is_reconstructed(self):
        for kind in ("request", "summary"):
            for mutation in ("missing", "fake_count"):
                rows = reflection4_recording()
                row = rows[-1] if kind == "summary" else next(row for row in rows if row["type"] == "request" and row["request_number"] == 3)
                value = row if kind == "summary" else json.loads(row["messages"][1]["content"])
                if mutation == "missing":
                    value.pop("experiment_progress")
                else:
                    value["experiment_progress"]["families"]["movement"]["trial_count"] = 99
                if kind == "request":
                    row["messages"][1]["content"] = json.dumps(value)
                with self.subTest(kind=kind, mutation=mutation), self.assertRaisesRegex(ValueError, "experiment progress"):
                    self.replay(rows)

    def test_original_executed_goal_is_stable_and_is_supplied(self):
        rows = reflection4_recording()
        request = next(row for row in rows if row["type"] == "request" and row["request_number"] == 2)
        context = json.loads(request["messages"][1]["content"])
        first = context["reflection_contract"]["experiments"]["experiments"][0]
        _, original = self.answer(rows, 1)
        self.assertEqual(first["goal"], original["experiment_goal"])
        completion, decision = self.answer(rows, 4)
        decision["experiment_goal"] = "Silently substitute another question under the same ID."
        completion["answer"] = json.dumps(decision)
        with self.assertRaises(ValueError):
            self.replay(rows)

    def test_evidence_linked_retarget_is_available_and_recorded_without_a_forced_route(self):
        rows = reflection4_recording(retarget=True)
        self.replay(rows)
        self.assertEqual(rows[-1]["research_context"]["contract_version"], 2)
        self.assertEqual(rows[-1]["research_context"]["scope_switches"], 1)
        request = next(row for row in rows if row["type"] == "request" and row["request_number"] == 2)
        context = json.loads(request["messages"][1]["content"])
        self.assertEqual(context["research_context"]["pending_plan"]["controls"], ["interact"])
        _, switched = self.answer(rows, 2)
        self.assertEqual(switched["action"], "interact")
        self.assertEqual(switched["plan"]["mode"], "retarget")
        self.assertEqual(switched["plan"]["evidence_steps"], [1])

    def test_rejected_response_does_not_apply_an_effect_note_or_consume_a_plan(self):
        rows = reflection4_recording(retarget=True)
        index = next(index for index, row in enumerate(rows) if row["type"] == "request" and row["request_number"] == 3)
        rejected = copy.deepcopy(rows[index])
        for row in rows[index:]:
            if "request_number" in row:
                row["request_number"] += 1
        rows[index:index] = [
            rejected,
            {"type": "completion", "request_number": 3, "finish_reason": "stop", "answer": None},
            {"type": "rejected_decision", "request_number": 3, "error": "Explicit fixture false effect"},
        ]
        rows[-1]["inference_requests"] += 1
        result = self.replay(rows)
        self.assertEqual(result["validated_comparisons"], 2)
        self.assertEqual(rows[-1]["research_context"]["scope_switches"], 1)

    def test_unassessed_new_experiment_at_a_budget_boundary_is_partial(self):
        full = reflection4_recording()
        end = next(index for index, row in enumerate(full) if row["type"] == "step" and row["step"] == 2)
        rows = copy.deepcopy(full[:end + 1])
        last_memory = next(row for row in reversed(rows) if row["type"] == "memory")
        ledger = {row["step"]: row for row in rows if row["type"] in {"step", "reset"}}
        notes = last_memory["hypotheses"]
        assessed = {1}
        summary = copy.deepcopy(full[-1])
        summary.update(
            status="action_limit", partial=True, steps=2, game_actions=2,
            resets=0, episodes=1, inference_requests=2,
            final_observation=rows[-1]["after"], unassessed_final_step=2,
            active_experiment=last_memory["active_experiment"],
            hypotheses=notes, experiment_bindings=last_memory["experiment_bindings"],
            comparison_receipts=comparison_receipts(notes, ledger),
            reflection_receipts=[{**reflection_receipt(ledger, step), "assessed": step in assessed} for step in sorted(ledger)],
            evidence_measurements=evidence_measurements(notes, ledger),
            experiment_progress=scope_progress(ledger, assessed, notes),
            research_context=plan_context(ledger, rows[-1]["after"], finished=False),
        )
        rows.append(summary)
        self.replay(rows)
        reviewed = review_session(self.path)
        self.assertEqual(reviewed["learning"]["unassessed_final_step"], 2)
        self.assertEqual(reviewed["experiment_progress"]["families"]["interaction"]["trial_count"], 1)
        self.assertEqual(reviewed["experiment_progress"]["families"]["interaction"]["assessed_trial_count"], 0)
        self.assertEqual(reviewed["goal_checks"]["next_stage_candidate"], "exploration_validation")

    def with_note_requirements(self, rows, *, nested=True, top=False):
        from autoplay_qa.exploration import trial_note_requirements

        ledger, assessed, notes = {}, set(), {}
        for row in rows:
            if row["type"] == "request":
                context = json.loads(row["messages"][1]["content"])
                required = trial_note_requirements(ledger, assessed, notes, context["pending_assessment_step"])
                if nested:
                    context.setdefault("response_contract", {}).setdefault("hypotheses", {})["required_trial_notes_after_valid_assessment"] = copy.deepcopy(required)
                if top:
                    context["required_trial_notes_after_valid_assessment"] = copy.deepcopy(required)
                row["messages"][1]["content"] = json.dumps(context)
            elif row["type"] == "memory":
                notes = {note["id"]: note for note in row["hypotheses"]}
                if row["assessment"] is not None:
                    assessed.add(row["assessment"]["step"])
            elif row["type"] in {"step", "reset"}:
                ledger[row["step"]] = row
        return rows

    def test_optional_conditional_note_requirement_does_not_advance_actual_progress(self):
        rows = self.with_note_requirements(reflection4_recording())
        self.replay(rows)
        request = next(row for row in rows if row["type"] == "request" and row["request_number"] == 2)
        context = json.loads(request["messages"][1]["content"])
        requirements = context["response_contract"]["hypotheses"]["required_trial_notes_after_valid_assessment"]
        self.assertEqual(requirements["pending_assessment_step"], 1)
        self.assertEqual(requirements["experiments"], [{"experiment_id": "movement", "control": "north", "evidence_trial_steps": [1]}])
        self.assertIn("If this response validly assesses", requirements["condition"])
        self.assertEqual(context["experiment_progress"]["families"]["movement"]["assessed_trial_count"], 0)
        self.assertEqual(context["experiment_progress"]["experiments"][0]["assessed_trial_steps"], [])

    def test_optional_top_level_conditional_requirement_is_also_validated(self):
        rows = self.with_note_requirements(reflection4_recording(), nested=False, top=True)
        self.replay(rows)
        request = next(row for row in rows if row["type"] == "request" and row["request_number"] == 2)
        context = json.loads(request["messages"][1]["content"])
        context["required_trial_notes_after_valid_assessment"]["experiments"][0]["control"] = "interact"
        request["messages"][1]["content"] = json.dumps(context)
        with self.assertRaisesRegex(ValueError, "conditional trial-note"):
            self.replay(rows)

    def test_conditional_note_requirements_reject_future_steps_or_false_accepted_assessments(self):
        for mutation in ("future_step", "boolean_pending", "hidden_requirement", "false_condition"):
            rows = self.with_note_requirements(reflection4_recording())
            request = next(row for row in rows if row["type"] == "request" and row["request_number"] == 2)
            context = json.loads(request["messages"][1]["content"])
            required = context["response_contract"]["hypotheses"]["required_trial_notes_after_valid_assessment"]
            if mutation == "future_step":
                required["experiments"][0]["evidence_trial_steps"] = [99]
            elif mutation == "boolean_pending":
                required["pending_assessment_step"] = True
            elif mutation == "hidden_requirement":
                required["experiments"] = []
            else:
                required["condition"] = "The pending assessment has already been accepted."
            request["messages"][1]["content"] = json.dumps(context)
            with self.subTest(mutation=mutation), self.assertRaisesRegex(ValueError, "conditional trial-note"):
                self.replay(rows)

    def test_both_present_requirement_copies_must_match_and_absence_stays_compatible(self):
        self.replay(reflection4_recording())
        rows = self.with_note_requirements(reflection4_recording(), nested=True, top=True)
        self.replay(rows)
        request = next(row for row in rows if row["type"] == "request" and row["request_number"] == 2)
        context = json.loads(request["messages"][1]["content"])
        context["required_trial_notes_after_valid_assessment"]["experiments"] = []
        request["messages"][1]["content"] = json.dumps(context)
        with self.assertRaisesRegex(ValueError, "conditional trial-note"):
            self.replay(rows)

    def test_correct_sampled_effect_does_not_verify_contradictory_claim_prose(self):
        rows = reflection4_recording()
        identifier = "h-north"
        for row in rows:
            if row["type"] == "completion":
                value = json.loads(row["answer"])
                for note in value["hypotheses"]:
                    if note["id"] == identifier:
                        note["claim"] = "A false universal claim that north always collects the key."
                row["answer"] = json.dumps(value)
            elif row["type"] in {"step", "reset", "finish"}:
                for note in row["decision"]["hypotheses"]:
                    if note["id"] == identifier:
                        note["claim"] = "A false universal claim that north always collects the key."
            elif row["type"] in {"memory", "summary"}:
                for note in row["hypotheses"]:
                    if note["id"] == identifier:
                        note["claim"] = "A false universal claim that north always collects the key."
            elif row["type"] == "request":
                value = json.loads(row["messages"][1]["content"])
                for note in value["hypotheses"]:
                    if note["id"] == identifier:
                        note["claim"] = "A false universal claim that north always collects the key."
                row["messages"][1]["content"] = json.dumps(value)
        self.replay(rows)
        reviewed = review_session(self.path, include_evidence=True)
        self.assertFalse(reviewed["hypothesis_evidence"]["claims_verified"])
        self.assertFalse(reviewed["hypothesis_evidence"]["notes"][0]["verified"])


if __name__ == "__main__":
    unittest.main()
