"""Typed comparison recordings bind receipts to execution, never claim truth."""

import copy
import json
from pathlib import Path
import tempfile
import unittest

from autoplay_qa.comparisons import comparison_context, comparison_receipts
from autoplay_qa.exploration import exploration_view
from autoplay_qa.game import KeyGateGame, evaluate_transition
from autoplay_qa.review import retention_progress, review_session
from autoplay_qa.session import replay
from tests.test_replay_provenance import recording as legacy_recording


def pair(original, repeated, original_episode=0, repeated_episode=1, result="same"):
    return {
        "original_step": original, "repeated_step": repeated,
        "original_episode": original_episode, "repeated_episode": repeated_episode,
        "result": result,
    }


def note(control, evidence=(), comparison=None, *, identifier=None):
    return {
        "id": identifier or "h-" + control,
        "claim": "Recorded " + control + " effect from the initial state.",
        "status": "supported" if comparison else "uncertain" if evidence else "untested",
        "evidence_steps": list(evidence), "control": control,
        "comparison": comparison,
    }


def strict_recording(actions=("north", "interact", "reset", "north", "interact", "finish"), *, updates=None):
    """Create explicit local fixtures with an actual deterministic game ledger."""
    if updates is None:
        updates = {
            1: [note("north")],
            2: [note("north", [1]), note("interact")],
            3: [note("interact", [2])],
            5: [note("north", [1, 4], pair(1, 4))],
            6: [note("interact", [2, 5], pair(2, 5))],
        }
    game = KeyGateGame(0)
    records = [{
        "type": "session", "schema_version": 2, "game_version": game.version,
        "seed": 0, "mode": "unit-test-fixtures", "task": "explore",
        "focus": "reset-retention", "comparison_contract": 1,
        "initial_observation": game.observe(),
    }]
    ledger, notes, assessed = {}, {}, set()
    episode = resets = steps = 0
    pending = active = None
    for request_number, action in enumerate(actions, 1):
        before = game.observe()
        context = {
            "episode": episode, "observation": exploration_view(before),
            "pending_assessment_step": pending, "active_experiment": active,
            "hypotheses": copy.deepcopy(list(notes.values())),
            "comparison_contract": comparison_context(ledger, assessed, pending),
        }
        records.append({
            "type": "request", "request_number": request_number,
            "episode": episode, "observation_tick": before["tick"],
            "messages": [
                {"role": "system", "content": "Explicit unit-test recording fixture"},
                {"role": "user", "content": json.dumps(context)},
            ],
        })
        assessment = None
        if pending is not None:
            previous = ledger[pending]
            previous_episode = previous.get("episode", previous.get("episode_before"))
            assessment = {
                "step": pending, "outcome": "Matched the recorded feedback.",
                "control": previous["decision"]["action"],
                "episode_before": previous_episode,
                "episode_after": previous.get("episode_after", previous_episode),
            }
            assessed.add(pending)
        changed = copy.deepcopy(updates.get(request_number, []))
        decision = {
            "action": action, "reason": "Explicit fixture chooses one control.",
            "plan": "Inspect the recorded result.",
            "experiment_id": "interaction" if action == "interact" or request_number == 6 else "movement",
            "experiment_goal": "Compare the chosen control under the recorded conditions.",
            "prediction": "The observable state may remain unchanged.",
            "assessment": assessment, "hypotheses": changed,
        }
        records.append({
            "type": "completion", "request_number": request_number,
            "completion_id": "explicit-fixture-response-" + str(request_number),
            "finish_reason": "stop", "answer": json.dumps(decision),
        })
        notes.update({item["id"]: copy.deepcopy(item) for item in changed})
        active = {"id": decision["experiment_id"], "goal": decision["experiment_goal"], "plan": decision["plan"]}
        records.append({
            "type": "memory", "request_number": request_number, "episode": episode,
            "active_experiment": copy.deepcopy(active), "assessment": copy.deepcopy(assessment),
            "hypotheses": copy.deepcopy(list(notes.values())),
            "comparison_receipts": comparison_receipts(notes, ledger),
        })
        pending = None
        if action == "finish":
            records.append({
                "type": "finish", "request_number": request_number, "episode": episode,
                "decision": decision, "observation": before,
            })
            break
        steps += 1
        if action == "reset":
            game = KeyGateGame(0)
            after = game.observe()
            row = {"type": "reset", "episode_before": episode, "episode_after": episode + 1}
            episode += 1
            resets += 1
        else:
            after = game.act(action)
            row = {"type": "step", "episode": episode, "evaluation": evaluate_transition(before, action, after)}
        row.update(step=steps, request_number=request_number, decision=decision, before=before, after=after)
        records.append(row)
        ledger[steps] = row
        pending = steps
    records.append({
        "type": "summary", "status": "finished" if actions[-1] == "finish" else "action_limit",
        "partial": actions[-1] != "finish", "steps": steps, "resets": resets,
        "game_actions": steps - resets, "episodes": episode + 1,
        "inference_requests": len(actions), "final_observation": game.observe(),
        "hypotheses": copy.deepcopy(list(notes.values())), "active_experiment": copy.deepcopy(active),
        "unassessed_final_step": pending,
        "comparison_receipts": comparison_receipts(notes, ledger),
    })
    return records


class ComparisonReplayTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.path = Path(directory.name) / "explicit-comparison-fixture.jsonl"

    def write(self, records):
        self.path.write_text("\n".join(json.dumps(record) for record in records) + "\n")

    def replay(self, records):
        self.write(records)
        return replay(self.path)

    def answer(self, records, request):
        completion = next(record for record in records if record["type"] == "completion" and record["request_number"] == request)
        return completion, json.loads(completion["answer"])

    def memory(self, records, request):
        return next(record for record in records if record["type"] == "memory" and record["request_number"] == request)

    def test_valid_typed_comparisons_replay_without_verifying_learning(self):
        result = self.replay(strict_recording())
        self.assertEqual(result["actions_replayed"], 5)
        self.assertEqual(result["resets_replayed"], 1)
        self.assertEqual(result["comparison_contract"], 1)
        self.assertTrue(result["comparison_bindings_validated"])
        self.assertEqual(result["validated_comparisons"], 2)
        self.assertFalse(result["learning_claims_verified"])
        review = review_session(self.path, include_evidence=True)
        self.assertEqual(review["comparison_contract"]["control_families"], {"movement": True, "interaction": True})
        self.assertEqual(review["goal_checks"]["reset_retested_control_families"], {"movement": True, "interaction": True})
        self.assertEqual(len(review["reset_learning"]["retest_pairs"]), 2)
        self.assertFalse(review["comparison_contract"]["learning_claims_verified"])
        evidence = review["hypothesis_evidence"]["notes"]
        self.assertEqual([item["declared_control"] for item in evidence], ["north", "interact"])
        self.assertEqual(evidence[0]["declared_comparison"], pair(1, 4))
        self.assertFalse(review["hypothesis_evidence"]["claims_verified"])
        progress = retention_progress(self.path)
        self.assertEqual(progress["families_observed"], {"movement": True, "interaction": True})

    def test_wrong_latest_control_or_episode_is_rejected(self):
        for field, value in (("control", "interact"), ("episode_before", 99), ("episode_after", True), ("step", True)):
            records = strict_recording()
            completion, value_dict = self.answer(records, 2)
            value_dict["assessment"][field] = value
            completion["answer"] = json.dumps(value_dict)
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.replay(records)

    def test_parser_outcome_guards_apply_during_strict_replay(self):
        for outcome in ("supported", "Pending execution", "Unassessed", 'hypotheses: schema placeholder'):
            records = strict_recording()
            completion, value = self.answer(records, 2)
            value["assessment"]["outcome"] = outcome
            completion["answer"] = json.dumps(value)
            with self.subTest(outcome=outcome), self.assertRaisesRegex(ValueError, "assessment.outcome"):
                self.replay(records)

    def test_existing_note_control_cannot_change(self):
        records = strict_recording()
        completion, value = self.answer(records, 5)
        value["hypotheses"][0]["control"] = "interact"
        completion["answer"] = json.dumps(value)
        with self.assertRaisesRegex(ValueError, "retain its declared control"):
            self.replay(records)

    def test_hypothesis_cannot_cite_a_reset_or_future_trial(self):
        for bad_step in (3, 99, True):
            records = strict_recording()
            completion, value = self.answer(records, 4)
            value["hypotheses"] = [note("north", [bad_step])]
            completion["answer"] = json.dumps(value)
            with self.subTest(step=bad_step), self.assertRaises(ValueError):
                self.replay(records)

    def test_bound_comparison_requires_actual_epochs_and_result(self):
        for field, wrong in (("original_episode", 1), ("repeated_episode", 0), ("repeated_episode", True), ("result", "different")):
            records = strict_recording()
            completion, value = self.answer(records, 5)
            value["hypotheses"][0]["comparison"][field] = wrong
            completion["answer"] = json.dumps(value)
            with self.subTest(field=field, value=wrong), self.assertRaises(ValueError):
                self.replay(records)

    def test_comparison_cannot_bind_trials_in_the_same_episode(self):
        records = strict_recording(
            ("north", "north", "reset", "north", "finish"),
            updates={1: [note("north")], 2: [note("north", [1])]},
        )
        completion, value = self.answer(records, 3)
        value["hypotheses"] = [note("north", [1, 2], pair(1, 2, repeated_episode=0))]
        completion["answer"] = json.dumps(value)
        with self.assertRaisesRegex(ValueError, "span a later episode"):
            self.replay(records)

    def test_comparison_cannot_refute_changed_starting_conditions(self):
        records = strict_recording(
            ("north", "east", "reset", "east", "north", "finish"),
            updates={1: [note("north")], 2: [note("north", [1])], 6: [note("north", [1, 5])]},
        )
        self.replay(records)
        completion, value = self.answer(records, 6)
        value["hypotheses"] = [note("north", [1, 5], pair(1, 5, result="different"))]
        completion["answer"] = json.dumps(value)
        with self.assertRaisesRegex(ValueError, "identical recorded starting conditions"):
            self.replay(records)

    def test_memory_snapshot_and_receipts_cannot_be_changed_independently(self):
        for field in ("hypotheses", "assessment", "active_experiment", "comparison_receipts"):
            records = strict_recording()
            memory = self.memory(records, 5)
            if field == "hypotheses":
                memory[field].pop()
            elif field == "assessment":
                memory[field]["outcome"] = "An invented memory reflection"
            elif field == "active_experiment":
                memory[field]["goal"] = "An invented memory experiment"
            else:
                memory[field][0]["observed_outcomes"]["repeated"]["player"]["inventory"] = ["key"]
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.replay(records)

    def test_even_empty_canonical_memory_receipts_are_required(self):
        records = strict_recording()
        self.memory(records, 1).pop("comparison_receipts")
        with self.assertRaisesRegex(ValueError, "memory receipts"):
            self.replay(records)

    def test_request_catalog_is_reconstructed_and_json_types_are_exact(self):
        for alteration in ("control", "boolean_version", "future_trial"):
            records = strict_recording()
            request = next(item for item in records if item["type"] == "request" and item["request_number"] == 2)
            context = json.loads(request["messages"][1]["content"])
            catalog = context["comparison_contract"]
            if alteration == "control":
                catalog["latest_control_receipt"]["control"] = "reset"
            elif alteration == "boolean_version":
                catalog["contract_version"] = True
            else:
                catalog["completed_trial_steps"].append(99)
            request["messages"][1]["content"] = json.dumps(context)
            with self.subTest(alteration=alteration), self.assertRaisesRegex(ValueError, "request catalog"):
                self.replay(records)

    def test_request_cannot_forget_retained_memory(self):
        records = strict_recording()
        request = next(item for item in records if item["type"] == "request" and item["request_number"] == 4)
        context = json.loads(request["messages"][1]["content"])
        context["hypotheses"] = []
        request["messages"][1]["content"] = json.dumps(context)
        with self.assertRaisesRegex(ValueError, "retained memory"):
            self.replay(records)

    def test_action_needs_memory_and_matches_its_completed_answer(self):
        for mutation in ("missing_memory", "changed_control", "truncated_completion"):
            records = strict_recording()
            if mutation == "missing_memory":
                records = [item for item in records if not (item["type"] == "memory" and item["request_number"] == 1)]
            elif mutation == "changed_control":
                next(item for item in records if item["type"] == "step")["decision"]["action"] = "south"
            else:
                next(item for item in records if item["type"] == "completion")["finish_reason"] = "length"
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                self.replay(records)

    def test_rejected_completion_can_be_repaired_without_applying_memory(self):
        records = strict_recording()
        index = next(index for index, item in enumerate(records) if item["type"] == "request" and item["request_number"] == 5)
        repair_context = copy.deepcopy(records[index])
        for row in records[index:]:
            if "request_number" in row:
                row["request_number"] += 1
        records[index:index] = [
            repair_context,
            {"type": "completion", "request_number": 5, "finish_reason": "length", "answer": None},
            {"type": "rejected_decision", "request_number": 5, "error": "Explicit fixture truncation"},
        ]
        records[-1]["inference_requests"] += 1
        result = self.replay(records)
        self.assertEqual(result["actions_replayed"], 5)
        self.assertEqual(result["validated_comparisons"], 2)

    def test_divergence_around_resets_and_summary_memory_is_rejected(self):
        for mutation in ("reset_after", "reset_episode", "step_epoch", "summary_memory", "summary_pending", "summary_receipt"):
            records = strict_recording()
            reset = next(item for item in records if item["type"] == "reset")
            if mutation == "reset_after":
                reset["after"]["player"]["position"] = [1, 2]
            elif mutation == "reset_episode":
                reset["episode_before"] = False
            elif mutation == "step_epoch":
                next(item for item in records if item["type"] == "step")["episode_after"] = 1
            elif mutation == "summary_memory":
                records[-1]["hypotheses"] = []
            elif mutation == "summary_pending":
                records[-1]["unassessed_final_step"] = 5
            else:
                records[-1]["comparison_receipts"][0]["verified"] = True
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                self.replay(records)

    def test_unsupported_explicit_contract_does_not_fall_back_to_legacy(self):
        for version in (None, 0, 2, True, 1.0, "1"):
            records = strict_recording()
            records[0]["comparison_contract"] = version
            with self.subTest(version=version), self.assertRaisesRegex(ValueError, "contract version"):
                self.replay(records)

    def test_declared_pair_without_pre_reset_note_is_not_retained_learning(self):
        records = strict_recording(
            ("north", "reset", "north", "finish"),
            updates={
                2: [note("north", [1], identifier="prior-note")],
                4: [note("north", [1, 3], pair(1, 3), identifier="created-after-retest")],
            },
        )
        self.replay(records)
        result = review_session(self.path)
        self.assertEqual(len(result["comparison_contract"]["validated_comparisons"]), 1)
        self.assertTrue(result["reset_learning"]["memory_supplied_after_reset"])
        self.assertFalse(result["reset_learning"]["behavioral_retest_observed"])
        self.assertFalse(result["goal_checks"]["reset_retention_observed"])

    def test_unbound_uncertain_pair_cannot_count_as_strict_retained_comparison(self):
        records = strict_recording(
            ("north", "reset", "north", "finish"),
            updates={2: [note("north", [1])], 4: [note("north", [1, 3])]},
        )
        self.replay(records)
        result = review_session(self.path)
        self.assertEqual(result["reset_learning"]["retest_pairs"], [])
        self.assertFalse(result["goal_checks"]["reset_retention_observed"])
        self.assertEqual(result["comparison_contract"]["validated_comparisons"], [])

    def test_changed_claim_cannot_bypass_retained_note_guard(self):
        records = strict_recording(
            ("north", "reset", "north", "finish"),
            updates={2: [note("north", [1])], 4: [{**note("north", [1, 3], pair(1, 3)), "claim": "A different new question"}]},
        )
        self.replay(records)
        result = review_session(self.path)
        self.assertEqual(len(result["comparison_contract"]["validated_comparisons"]), 1)
        self.assertEqual(result["reset_learning"]["retest_pairs"], [])

    def test_legacy_schema_one_and_two_keep_ordinary_review_shapes(self):
        for schema in (1, 2):
            records = legacy_recording(["north", "reset", "north", "finish"] if schema == 2 else ["north", "interact"], schema=schema)
            result = self.replay(records)
            self.assertNotIn("comparison_contract", result)
            review = review_session(self.path)
            self.assertNotIn("comparison_contract", review)
            evidence = review_session(self.path, include_evidence=True)
            for note_item in evidence["hypothesis_evidence"]["notes"]:
                self.assertNotIn("declared_control", note_item)
                self.assertNotIn("declared_comparison", note_item)


if __name__ == "__main__":
    unittest.main()
