"""Typed comparisons describe execution facts, never hypothesis truth."""

from copy import deepcopy
import json
import unittest

from autoplay_qa.comparisons import (
    COMPARISON_CONTRACT_VERSION,
    comparison_context,
    comparison_receipts,
    validate_assessment,
    validate_hypothesis,
)
from autoplay_qa.game import ACTIONS, KeyGateGame
from autoplay_qa.nebius import InvalidDecision


def records_for(controls):
    game, episode, records = KeyGateGame(0), 0, {}
    for step, control in enumerate(controls, 1):
        before = game.observe()
        if control == "reset":
            game = KeyGateGame(0)
            record = {"type": "reset", "episode_before": episode, "episode_after": episode + 1}
            episode += 1
            after = game.observe()
        else:
            record = {"type": "step", "episode": episode}
            after = game.act(control)
        records[step] = {**record, "step": step, "decision": {"action": control}, "before": before, "after": after}
    return records


def pair(original=1, repeated=3, original_episode=0, repeated_episode=1, result="same"):
    return {"original_step": original, "repeated_step": repeated, "original_episode": original_episode, "repeated_episode": repeated_episode, "result": result}


def hypothesis(control="north", status="supported", evidence=(1, 3), comparison="default"):
    return {
        "id": "effect", "claim": "This control leaves the starting state unchanged.",
        "status": status, "evidence_steps": list(evidence), "control": control,
        "comparison": pair() if comparison == "default" else comparison,
    }


def assessment(step=3, control="north", before=1, after=1):
    return {"step": step, "outcome": "The control was blocked, matching its prediction.", "control": control, "episode_before": before, "episode_after": after}


class ComparisonContractTests(unittest.TestCase):
    def setUp(self):
        self.records = records_for(["north", "reset", "north"])

    def valid(self, note=None, records=None, assessed=None, **kwargs):
        return validate_hypothesis(
            hypothesis() if note is None else note, self.records if records is None else records,
            {1, 3} if assessed is None else assessed,
            require_comparison=kwargs.pop("require_comparison", True), **kwargs,
        )

    def test_version_and_valid_blocked_pair(self):
        self.assertEqual(COMPARISON_CONTRACT_VERSION, 1)
        original = hypothesis()
        self.assertEqual(self.valid(original), original)
        self.assertEqual(set(self.valid()), {"id", "claim", "status", "evidence_steps", "control", "comparison"})

    def test_valid_refused_interaction_pair(self):
        records = records_for(["interact", "reset", "interact"])
        note = hypothesis("interact")
        self.assertEqual(self.valid(note, records), note)
        self.assertEqual(records[1]["after"]["events"], ["There is no gate within reach."])

    def test_assessment_is_bound_to_latest_actual_step_control_and_epochs(self):
        value = assessment()
        value["outcome"] = "  Observed the block; prediction matched.  "
        checked = validate_assessment(value, self.records, 3)
        self.assertEqual(checked["outcome"], "Observed the block; prediction matched.")
        self.assertEqual({field: checked[field] for field in ("step", "control", "episode_before", "episode_after")}, {"step": 3, "control": "north", "episode_before": 1, "episode_after": 1})
        self.assertEqual(value["outcome"], "  Observed the block; prediction matched.  ")
        for field, wrong in (("step", 1), ("control", "interact"), ("episode_before", 0), ("episode_after", 2), ("step", True), ("episode_before", True), ("episode_after", 1.0)):
            with self.subTest(field=field, wrong=wrong):
                candidate = {**assessment(), field: wrong}
                with self.assertRaises(InvalidDecision):
                    validate_assessment(candidate, self.records, 3)

    def test_assessment_handles_reset_and_null_without_fabricating_controls(self):
        records = records_for(["north", "reset"])
        value = assessment(2, "reset", 0, 1)
        self.assertEqual(validate_assessment(value, records, 2), value)
        self.assertIsNone(validate_assessment(None, {}, None))
        for value, records, pending in ((assessment(), {}, None), (None, self.records, 3), (assessment(1, "north", 0, 0), self.records, 1), (assessment(), self.records, True), (assessment(), self.records, 99)):
            with self.subTest(pending=pending):
                with self.assertRaises(InvalidDecision):
                    validate_assessment(value, records, pending)

    def test_exact_fields_and_trimmed_text_limits(self):
        for value in ({**assessment(), "plan": "unexecuted"}, {key: value for key, value in assessment().items() if key != "control"}):
            with self.assertRaises(InvalidDecision):
                validate_assessment(value, self.records, 3)
        for value in ({**hypothesis(), "plan": "unexecuted"}, {key: value for key, value in hypothesis().items() if key != "comparison"}):
            with self.assertRaises(InvalidDecision):
                self.valid(value)
        maximum = {**hypothesis(), "id": " " + "x" * 32 + " ", "claim": " " + "é" * 240 + " "}
        self.assertEqual(self.valid(maximum)["id"], "x" * 32)
        self.assertEqual(self.valid(maximum)["claim"], "é" * 240)
        self.assertEqual(len(validate_assessment({**assessment(), "outcome": " " + "é" * 360 + " "}, self.records, 3)["outcome"]), 360)
        for field, wrong in (("id", "x" * 33), ("claim", "é" * 241), ("id", "  "), ("claim", None), ("status", "verified"), ("status", [])):
            with self.subTest(field=field):
                with self.assertRaises(InvalidDecision):
                    self.valid({**hypothesis(), field: wrong})
        for outcome in ("x" * 361, "  ", None, True):
            with self.assertRaises(InvalidDecision):
                validate_assessment({**assessment(), "outcome": outcome}, self.records, 3)

    def test_evidence_must_be_actual_game_trials_of_declared_control(self):
        for refs in ([1, 2], [1, 4], [1, 0], [1, True], [1, 3.0], [1, "3"], [1, {}], [1, 1, 3], list(range(1, 10)), "1,3"):
            with self.subTest(refs=refs):
                with self.assertRaises(InvalidDecision):
                    self.valid({**hypothesis(), "evidence_steps": refs})
        with self.assertRaises(InvalidDecision):
            self.valid(hypothesis("interact"))
        for control in ("reset", "finish", "teleport", None, []):
            with self.subTest(control=control):
                with self.assertRaises(InvalidDecision):
                    self.valid(hypothesis(control, "untested", (), None))

    def test_control_is_stable_for_reused_hypothesis_id(self):
        previous = hypothesis("north", "uncertain", (1,), None)
        update = hypothesis("interact", "untested", (), None)
        with self.assertRaisesRegex(InvalidDecision, "retain its declared control"):
            self.valid(update, previous=previous)
        unchanged = hypothesis("north", "uncertain", (1,), None)
        self.assertEqual(self.valid(unchanged, previous=previous), unchanged)

    def test_uncertain_single_trial_and_untested_proposals_remain_valid(self):
        uncertain = hypothesis(status="uncertain", evidence=(1,), comparison=None)
        self.assertEqual(self.valid(uncertain, assessed=set()), uncertain)
        for control in ACTIONS:
            proposal = hypothesis(control, "untested", (), None)
            self.assertEqual(self.valid(proposal, assessed=set()), proposal)
        for status in ("supported", "contradicted"):
            note = hypothesis(status=status, evidence=(1,), comparison=None)
            with self.assertRaises(InvalidDecision):
                self.valid(note)
            self.assertEqual(self.valid(note, require_comparison=False), note)
            with self.assertRaises(InvalidDecision):
                self.valid(hypothesis(status=status, evidence=(), comparison=None), require_comparison=False)

    def test_comparison_requires_exact_pair_keys_integer_order_and_both_refs(self):
        invalid = [
            {**pair(), "control": "north"},
            {key: value for key, value in pair().items() if key != "result"},
            pair(original=3, repeated=1), pair(original=1, repeated=1),
            pair(original=True), pair(repeated=3.0), pair(original_episode=False),
            pair(repeated_episode="1"), pair(result="verified"), pair(result=[]),
        ]
        for comparison in invalid:
            with self.subTest(comparison=comparison):
                with self.assertRaises(InvalidDecision):
                    self.valid(hypothesis(comparison=comparison))
        with self.assertRaises(InvalidDecision):
            self.valid(hypothesis(evidence=(1,), comparison=pair()))

    def test_actual_episode_labels_and_cross_episode_requirement(self):
        for comparison in (pair(original_episode=1), pair(repeated_episode=2)):
            with self.assertRaises(InvalidDecision):
                self.valid(hypothesis(comparison=comparison))
        same_episode = deepcopy(self.records)
        same_episode[3]["episode"] = 0
        with self.assertRaises(InvalidDecision):
            self.valid(hypothesis(comparison=pair(repeated_episode=0)), same_episode)
        explicit = deepcopy(self.records)
        for step in (1, 3):
            episode = explicit[step].pop("episode")
            explicit[step].update(episode_before=episode, episode_after=episode)
        self.assertEqual(self.valid(records=explicit), hypothesis())

    def test_changed_position_inventory_objects_map_game_or_status_rejects_pair(self):
        changes = [
            lambda state: state["player"].update(position=[2, 1]),
            lambda state: state["player"].update(inventory=["key"]),
            lambda state: state["objects"].update(key=None),
            lambda state: state["objects"]["gate"].update(open=True),
            lambda state: state["map"].__setitem__(0, "changed visible map"),
            lambda state: state.update(game="another-game"),
            lambda state: state.update(status="won"),
        ]
        for change in changes:
            records = deepcopy(self.records)
            change(records[3]["before"])
            with self.subTest(before=records[3]["before"]["player"]):
                with self.assertRaisesRegex(InvalidDecision, "identical recorded starting conditions"):
                    self.valid(records=records)
                self.assertEqual(comparison_context(records, {1, 3}, None)["eligible_pairs"], [])

    def test_every_cited_trial_uses_the_pairs_starting_conditions(self):
        records = records_for(["north", "reset", "north", "east", "north"])
        with self.assertRaisesRegex(InvalidDecision, "identical recorded starting conditions"):
            self.valid(hypothesis(evidence=(1, 3, 5)), records)
        comparable = records_for(["north", "reset", "north", "north"])
        note = hypothesis(evidence=(1, 3, 4))
        self.assertEqual(self.valid(note, comparable), note)

    def test_both_trials_need_assessments_current_valid_assessment_can_count(self):
        for assessed in (set(), {1}, {3}, {True, 3}):
            with self.subTest(assessed=assessed):
                with self.assertRaisesRegex(InvalidDecision, "Both comparison trials"):
                    self.valid(assessed=assessed)
        checked = validate_assessment(assessment(), self.records, 3)
        self.assertEqual(self.valid(assessed={1, checked["step"]}), hypothesis())
        with self.assertRaises(InvalidDecision):
            self.valid(hypothesis(status="uncertain"), assessed={1})

    def test_observed_result_includes_after_state_and_feedback_ignoring_ticks(self):
        records = deepcopy(self.records)
        records[3]["after"]["tick"] = 100
        records[3]["before"]["tick"] = 99
        records[3]["before"]["events"] = ["An earlier event is not a starting condition."]
        self.assertEqual(self.valid(records=records), hypothesis())
        for change in (lambda state: state["events"].append("A different recorded effect."), lambda state: state["player"].update(position=[2, 1]), lambda state: state["objects"]["gate"].update(open=True)):
            changed = deepcopy(records)
            change(changed[3]["after"])
            with self.assertRaisesRegex(InvalidDecision, "disagrees with recorded"):
                self.valid(records=changed)
            different = hypothesis(comparison=pair(result="different"))
            self.assertEqual(self.valid(different, changed), different)
        with self.assertRaises(InvalidDecision):
            self.valid(hypothesis(comparison=pair(result="different")))

    def test_starting_conditions_distinguish_boolean_integer_and_float_values(self):
        for value in (True, 1.0):
            with self.subTest(value=value):
                records = deepcopy(self.records)
                records[3]["before"]["player"]["position"][0] = value
                with self.assertRaisesRegex(InvalidDecision, "identical recorded starting conditions"):
                    self.valid(records=records)
                self.assertEqual(comparison_context(records, {1, 3}, None)["eligible_pairs"], [])

    def test_observed_results_distinguish_boolean_and_numeric_values(self):
        records = deepcopy(self.records)
        records[3]["after"]["objects"]["gate"]["open"] = 0
        with self.assertRaisesRegex(InvalidDecision, "disagrees with recorded"):
            self.valid(records=records)
        different = hypothesis(comparison=pair(result="different"))
        self.assertEqual(self.valid(different, records), different)
        self.assertEqual(comparison_context(records, {1, 3}, None)["eligible_pairs"], [{**pair(result="different"), "control": "north"}])

    def test_exact_json_comparisons_ignore_object_key_order(self):
        records = deepcopy(self.records)
        for field in ("before", "after"):
            state = records[3][field]
            state["player"] = dict(reversed(list(state["player"].items())))
            state["objects"] = dict(reversed(list(state["objects"].items())))
        self.assertEqual(self.valid(records=records), hypothesis())
        self.assertEqual(comparison_context(records, {1, 3}, None)["eligible_pairs"], [{**pair(), "control": "north"}])

    def test_context_reports_actual_trials_pending_pair_and_latest_control(self):
        context = comparison_context(self.records, {1}, 3)
        self.assertEqual(context["contract_version"], 1)
        self.assertEqual([(trial["step"], trial["control"], trial["episode_before"], trial["episode_after"]) for trial in context["completed_trials"]], [(1, "north", 0, 0), (3, "north", 1, 1)])
        self.assertEqual(context["eligible_pairs"], [{**pair(), "control": "north"}])
        self.assertEqual(context["latest_control_receipt"]["control"], "north")
        self.assertEqual(context["latest_control_receipt"]["step"], 3)
        self.assertEqual(comparison_context(self.records, {1, 3}, None)["eligible_pairs"], context["eligible_pairs"])
        self.assertEqual(comparison_context(self.records, set(), 3)["eligible_pairs"], [])
        self.assertEqual(comparison_context(self.records, {1}, None)["eligible_pairs"], [])

    def test_first_reset_has_no_game_trials_or_impossible_comparison(self):
        context = comparison_context(records_for(["reset"]), set(), 1)
        self.assertEqual(context["completed_trials"], [])
        self.assertEqual(context["eligible_pairs"], [])
        self.assertEqual({field: context["latest_control_receipt"][field] for field in ("step", "control", "episode_before", "episode_after")}, {"step": 1, "control": "reset", "episode_before": 0, "episode_after": 1})
        self.assertEqual(context["latest_control_receipt"]["after"]["events"], ["Environment started."])
        self.assertEqual(comparison_context({}, set(), None)["latest_control_receipt"], None)

    def test_context_is_bounded_and_returns_copies_without_selecting_actions(self):
        records = records_for(["north"] * 30)
        context = comparison_context(records, set(), 30)
        self.assertEqual(len(context["completed_trials"]), 24)
        self.assertEqual([trial["step"] for trial in context["completed_trials"]], list(range(7, 31)))
        self.assertEqual(context["eligible_pairs"], [])
        self.assertEqual(context["completed_trial_steps"], list(range(1, 31)))
        self.assertEqual(set(context), {"contract_version", "completed_trial_steps", "completed_trials", "eligible_pairs", "latest_control_receipt"})
        context["completed_trials"][0]["before"]["player"]["position"][0] = 99
        self.assertEqual(records[7]["before"]["player"]["position"], [1, 1])

    def test_custom_budget_retains_all_trial_ids_and_older_eligible_baselines(self):
        records = records_for(["north", "reset"] + ["north"] * 25)
        context = comparison_context(records, {1}, 27)
        self.assertEqual(len(context["completed_trials"]), 24)
        self.assertNotIn(1, [trial["step"] for trial in context["completed_trials"]])
        self.assertEqual(context["completed_trial_steps"], [1, *range(3, 28)])
        expected = {**pair(repeated=27), "control": "north"}
        self.assertIn(expected, context["eligible_pairs"])
        note = hypothesis(evidence=(1, 27), comparison=pair(repeated=27))
        self.assertEqual(self.valid(note, records, assessed={1, 27}), note)

    def test_context_and_receipts_withhold_rules_strategy_hints_and_initial_events(self):
        hidden = "PRIVATE-STRATEGY-AND-PASSABILITY"
        records = deepcopy(self.records)
        for record in records.values():
            record["evaluation"] = {"contract_violations": [hidden]}
            record["decision"]["plan"] = hidden
            for state in (record["before"], record["after"]):
                state.update(goal=hidden, rules=[hidden], passability=hidden)
                state["player"]["strategy"] = hidden
                state["objects"]["developer_hints"] = hidden
                state["objects"]["gate"]["passable"] = hidden
                if state["tick"] == 0:
                    state["events"] = [hidden]
        note = {**hypothesis(), "claim": hidden}
        context = comparison_context(records, {1, 3}, None)
        receipts = comparison_receipts([note], records)
        for view in (context, receipts):
            self.assertNotIn(hidden, json.dumps(view))
        self.assertEqual(context["completed_trials"][0]["before"]["events"], ["Environment started."])
        self.assertEqual(receipts[0]["starting_conditions"]["player"], {"position": [1, 1], "inventory": []})
        self.assertEqual(set(receipts[0]["starting_conditions"]["objects"]["gate"]), {"position", "open"})

    def test_receipts_preserve_typed_facts_without_deciding_claim_truth(self):
        false_claim = {**hypothesis(), "claim": "Blocked north always collects the key."}
        self.assertEqual(self.valid(false_claim)["claim"], false_claim["claim"])
        receipts = comparison_receipts({"effect": false_claim}, self.records)
        self.assertEqual(len(receipts), 1)
        receipt = receipts[0]
        self.assertEqual(receipt["hypothesis_id"], "effect")
        self.assertEqual(receipt["control"], "north")
        self.assertEqual(receipt["comparison"], pair())
        self.assertFalse(receipt["verified"])
        self.assertEqual(receipt["starting_conditions"]["player"]["inventory"], [])
        self.assertEqual(receipt["observed_outcomes"]["original"]["player"]["inventory"], [])
        self.assertEqual(receipt["observed_outcomes"]["original"], receipt["observed_outcomes"]["repeated"])
        self.assertNotIn("claim", receipt)
        self.assertEqual(comparison_receipts([hypothesis(status="uncertain", evidence=(1,), comparison=None)], self.records), [])
        receipt["starting_conditions"]["player"]["position"][0] = 99
        self.assertEqual(self.records[1]["before"]["player"]["position"], [1, 1])

    def test_diagnostics_are_bounded_and_never_echo_invalid_model_content(self):
        hidden = "UNTRUSTED-CONTENT-MUST-NOT-BE-ECHOED"
        candidates = [
            {**hypothesis(), "id": hidden * 20},
            {**hypothesis(), "claim": hidden * 20},
            {**hypothesis(), "control": hidden},
            {**hypothesis(), "comparison": {**pair(), "result": hidden}},
            {**hypothesis(), "evidence_steps": [hidden]},
        ]
        for value in candidates:
            with self.assertRaises(InvalidDecision) as caught:
                self.valid(value)
            self.assertNotIn(hidden, str(caught.exception))
            self.assertLess(len(str(caught.exception)), 200)
        with self.assertRaises(InvalidDecision) as caught:
            validate_assessment({**assessment(), "outcome": hidden * 20}, self.records, 3)
        self.assertNotIn(hidden, str(caught.exception))


if __name__ == "__main__":
    unittest.main()
