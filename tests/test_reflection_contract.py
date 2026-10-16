"""Pure factual contract fixtures; forecasts here are never live learning."""

from copy import deepcopy
import json
import unittest

from autoplay_qa.game import ACTIONS, KeyGateGame
from autoplay_qa.nebius import InvalidDecision
from autoplay_qa.reflection import (
    REFLECTION_CONTRACT_VERSION, experiment_context, outcome_projection,
    prediction_schema, reflection_receipt, validate_assessment, validate_intent,
    trial_progress, validate_trial_comparisons,
)


def intent(action="north", *, control="north", role="trial", identifier="movement", expected=None):
    return {
        "action": action, "experiment_id": identifier,
        "experiment_control": control, "action_role": role,
        "expected_outcome": outcome_projection(KeyGateGame().observe()) if expected is None and action != "finish" else expected,
    }


def records_for(controls):
    """Fixture-only correct forecasts; actual feedback comes from the game."""
    game, episode, records = KeyGateGame(), 0, {}
    for step, control in enumerate(controls, 1):
        before = game.observe()
        if control == "reset":
            game = KeyGateGame()
            record = {"type": "reset", "episode_before": episode, "episode_after": episode + 1}
            episode += 1
            after = game.observe()
            decision = intent(control, role="setup", expected=outcome_projection(after))
        else:
            record = {"type": "step", "episode": episode}
            after = game.act(control)
            decision = intent(control, control=control, identifier="interaction" if control == "interact" else "movement", expected=outcome_projection(after))
        records[step] = {**record, "step": step, "decision": decision, "before": before, "after": after}
    return records


def assessment(records, step):
    receipt = reflection_receipt(records, step)
    return {**{field: value for field, value in receipt.items() if field != "expected_outcome"},
            "outcome": "Fixture prose: observed the actual feedback and compared the state forecast."}


def note(control="north", refs=(1, 3)):
    return {
        "id": "effect", "claim": "The sampled effect repeats under these conditions.",
        "status": "supported", "evidence_steps": list(refs), "control": control,
        "comparison": {"original_step": 1, "repeated_step": 3,
                       "original_episode": 0, "repeated_episode": 1, "result": "same"},
    }


class ReflectionContractTests(unittest.TestCase):
    def test_version_projection_and_schema_are_exact_five_facets(self):
        self.assertEqual(REFLECTION_CONTRACT_VERSION, 1)
        observed = outcome_projection(KeyGateGame().observe())
        self.assertEqual(set(observed), {"position", "inventory", "key", "gate_open", "status"})
        self.assertEqual(observed["position"], [1, 1])
        schema = prediction_schema()
        self.assertEqual(set(schema["properties"]), set(observed))
        self.assertEqual(set(schema["required"]), set(observed))
        self.assertFalse(schema["additionalProperties"])
        self.assertEqual(schema["properties"]["inventory"]["maxItems"], 8)
        self.assertEqual(schema["properties"]["inventory"]["items"]["maxLength"], 32)
        schema["properties"]["position"]["items"]["enum"] = [1]
        self.assertNotIn("enum", prediction_schema()["properties"]["position"]["items"])

    def test_wrong_future_outcomes_remain_valid_forecasts(self):
        value = intent()
        value["expected_outcome"].update(position=[-8, 99], inventory=["imagined object"], key=None, gate_open=True, status="won")
        checked = validate_intent(value, {})
        self.assertEqual(checked["expected_outcome"], value["expected_outcome"])
        self.assertEqual(set(checked), {"expected_outcome", "experiment_control", "action_role"})

    def test_forecast_types_are_exact_and_bounds_apply(self):
        invalid = [
            ("position", [True, 1]), ("position", [1.0, 1]), ("position", [1]),
            ("position", (1, 1)), ("position", "[1,1]"),
            ("key", [1, False]), ("key", [1, 1, 1]), ("gate_open", 0),
            ("gate_open", "false"), ("status", "unknown"), ("status", []),
            ("inventory", "key"), ("inventory", [False]), ("inventory", [" "]),
            ("inventory", ["x" * 33]), ("inventory", ["key"] * 9),
        ]
        for field, wrong in invalid:
            value = intent()
            value["expected_outcome"][field] = wrong
            with self.subTest(field=field, wrong=wrong), self.assertRaises(InvalidDecision):
                validate_intent(value, {})
        for forecast in (None, {}, {**intent()["expected_outcome"], "rules": []}):
            with self.subTest(forecast=forecast), self.assertRaises(InvalidDecision):
                validate_intent({**intent(), "expected_outcome": forecast}, {})
        maximum = intent()
        maximum["expected_outcome"]["inventory"] = ["é" * 32] * 8
        self.assertEqual(validate_intent(maximum, {})["expected_outcome"]["inventory"], ["é" * 32] * 8)
        with self.assertRaises(InvalidDecision):
            validate_intent({**maximum, "expected_outcome": {**maximum["expected_outcome"], "inventory": [" " + "é" * 32 + " "]}}, {})

    def test_inventory_forecast_preserves_literal_whitespace_and_mismatches_pickup(self):
        records = records_for(["south", "south"])
        records[2]["decision"]["expected_outcome"]["inventory"] = [" key "]
        checked = validate_intent(records[2]["decision"], {})
        self.assertEqual(checked["expected_outcome"]["inventory"], [" key "])
        receipt = reflection_receipt(records, 2)
        self.assertEqual(receipt["expected_outcome"]["inventory"], [" key "])
        self.assertEqual(receipt["observed_outcome"]["inventory"], ["key"])
        self.assertEqual(receipt["prediction_verdict"], "mismatched")

    def test_actual_inventory_projection_preserves_literal_whitespace(self):
        observation = KeyGateGame().observe()
        observation["player"]["inventory"] = [" key "]
        projected = outcome_projection(observation)
        self.assertEqual(projected["inventory"], [" key "])
        projected["inventory"][0] = "another item"
        self.assertEqual(observation["player"]["inventory"], [" key "])

    def test_experiment_ids_have_stable_controls_and_bindings_are_pure(self):
        bindings = {"movement": "north"}
        self.assertEqual(validate_intent(intent(), bindings)["experiment_control"], "north")
        self.assertEqual(bindings, {"movement": "north"})
        with self.assertRaisesRegex(InvalidDecision, "retain its declared"):
            validate_intent(intent("east", control="east"), bindings)
        for control in ("reset", "finish", None, [], True):
            with self.subTest(control=control), self.assertRaises(InvalidDecision):
                validate_intent(intent(control=control), {})
        for identifier in ("", " ", "x" * 33, True):
            with self.subTest(identifier=identifier), self.assertRaises(InvalidDecision):
                validate_intent(intent(identifier=identifier), {})
        value = intent(identifier="  movement  ")
        self.assertEqual(validate_intent(value, bindings)["experiment_control"], "north")

    def test_setup_controls_allow_navigation_and_reset(self):
        for action in (*ACTIONS, "reset"):
            with self.subTest(action=action):
                self.assertEqual(validate_intent(intent(action, role="setup"), {})["action_role"], "setup")
        for action in ACTIONS:
            self.assertEqual(validate_intent(intent(action, control=action), {})["action_role"], "trial")
        for value in (intent("east"), intent("reset"), intent(role="finish"), intent(role="unknown"), intent("teleport", role="setup")):
            with self.subTest(value=value), self.assertRaises(InvalidDecision):
                validate_intent(value, {})

    def test_finish_has_no_forecast_or_trial(self):
        value = intent("finish", role="finish")
        self.assertIsNone(validate_intent(value, {})["expected_outcome"])
        for value in (intent("finish"), intent("finish", role="setup"), intent("finish", role="finish", expected=outcome_projection(KeyGateGame().observe()))):
            with self.assertRaises(InvalidDecision):
                validate_intent(value, {})

    def test_completed_movement_receipt_and_assessment_are_bound_to_actual_state(self):
        records = records_for(["east"])
        receipt = reflection_receipt(records, 1)
        self.assertEqual(receipt["position_before"], [1, 1])
        self.assertEqual(receipt["observed_outcome"]["position"], [2, 1])
        self.assertEqual(receipt["prediction_verdict"], "matched")
        self.assertEqual(receipt["observed_events"], ["Moved east to [2, 1]."])
        value = assessment(records, 1)
        self.assertEqual(validate_assessment(value, records, 1), value)
        self.assertEqual(len(value), 9)

    def test_incorrect_pickup_forecast_is_mismatched_after_refusal(self):
        records = records_for(["interact"])
        records[1]["decision"]["expected_outcome"].update(inventory=["key"], key=None)
        receipt = reflection_receipt(records, 1)
        self.assertEqual(receipt["prediction_verdict"], "mismatched")
        self.assertEqual(receipt["observed_outcome"]["inventory"], [])
        self.assertEqual(receipt["observed_events"], ["There is no gate within reach."])
        value = assessment(records, 1)
        self.assertEqual(validate_assessment(value, records, 1)["prediction_verdict"], "mismatched")
        with self.assertRaisesRegex(InvalidDecision, "prediction_verdict"):
            validate_assessment({**value, "prediction_verdict": "matched"}, records, 1)

    def test_original_forecast_is_used_and_never_rewritten_from_after_state(self):
        records = records_for(["east"])
        records[1]["decision"]["expected_outcome"]["position"] = [1, 1]
        before = deepcopy(records)
        receipt = reflection_receipt(records, 1)
        self.assertEqual(receipt["expected_outcome"]["position"], [1, 1])
        self.assertEqual(receipt["observed_outcome"]["position"], [2, 1])
        self.assertEqual(receipt["prediction_verdict"], "mismatched")
        self.assertEqual(records, before)
        receipt["expected_outcome"]["position"][0] = 999
        self.assertEqual(records, before)

    def test_blocked_refused_and_reset_feedback_are_actual_and_reset_is_neutral(self):
        for control, feedback in (("north", ["Movement blocked by a wall or locked gate."]), ("interact", ["There is no gate within reach."])):
            records = records_for([control])
            receipt = reflection_receipt(records, 1)
            self.assertEqual(receipt["observed_events"], feedback)
            self.assertEqual(receipt["observed_outcome"]["position"], [1, 1])
        records = records_for(["east", "reset"])
        receipt = reflection_receipt(records, 2)
        self.assertEqual((receipt["control"], receipt["episode_before"], receipt["episode_after"]), ("reset", 0, 1))
        self.assertEqual(receipt["position_before"], [2, 1])
        self.assertEqual(receipt["observed_outcome"]["position"], [1, 1])
        self.assertEqual(receipt["observed_events"], ["Environment started."])
        self.assertEqual(validate_assessment(assessment(records, 2), records, 2)["control"], "reset")

    def test_actual_projection_handles_pickup_and_gate_changes_without_rules(self):
        game = KeyGateGame()
        before = game.observe()
        game.act("south")
        pickup = game.act("south")
        after_pickup = outcome_projection(pickup)
        self.assertEqual(after_pickup["inventory"], ["key"])
        self.assertIsNone(after_pickup["key"])
        game.act("east")
        game.act("east")
        opened = outcome_projection(game.act("interact"))
        self.assertTrue(opened["gate_open"])
        self.assertEqual(opened["inventory"], [])
        self.assertFalse(outcome_projection(before)["gate_open"])

    def test_wrong_before_position_events_or_observed_facets_are_rejected(self):
        records = records_for(["north"])
        original = assessment(records, 1)
        invalid = [
            {**original, "position_before": [1, 2]},
            {**original, "position_before": [True, 1]},
            {**original, "observed_events": ["Collected the key."]},
            {**original, "observed_events": "Movement blocked by a wall or locked gate."},
            {**original, "prediction_verdict": True},
            {**original, "observed_outcome": {**original["observed_outcome"], "gate_open": 0}},
            {**original, "observed_outcome": {**original["observed_outcome"], "position": [1.0, 1]}},
            {**original, "observed_outcome": {**original["observed_outcome"], "inventory": ["key"]}},
        ]
        for value in invalid:
            with self.subTest(value=value), self.assertRaises(InvalidDecision):
                validate_assessment(value, records, 1)
        self.assertEqual(validate_assessment(original, records, 1), original)

    def test_assessment_control_epochs_shape_and_pending_step_are_checked(self):
        records = records_for(["north", "reset", "north"])
        good = assessment(records, 3)
        invalid = [
            {**good, "control": "reset"}, {**good, "episode_before": 0},
            {**good, "episode_after": True}, {**good, "step": True},
            {**good, "expected_outcome": good["observed_outcome"]},
            {key: value for key, value in good.items() if key != "observed_events"},
        ]
        for value in invalid:
            with self.subTest(value=value), self.assertRaises(InvalidDecision):
                validate_assessment(value, records, 3)
        for pending in (1, True, 99):
            with self.subTest(pending=pending), self.assertRaises(InvalidDecision):
                validate_assessment(good, records, pending)
        with self.assertRaises(InvalidDecision):
            validate_assessment(None, records, 3)
        self.assertIsNone(validate_assessment(None, {}, None))
        with self.assertRaises(InvalidDecision):
            validate_assessment(good, records, None)

    def test_historical_receipts_work_but_assessments_require_latest(self):
        records = records_for(["north", "reset", "north"])
        self.assertEqual(reflection_receipt(records, 1)["step"], 1)
        self.assertEqual(reflection_receipt(records, 2)["control"], "reset")
        with self.assertRaises(InvalidDecision):
            validate_assessment(assessment(records, 1), records, 1)
        for pending in (0, 4, True, "1"):
            with self.subTest(pending=pending), self.assertRaises(InvalidDecision):
                reflection_receipt(records, pending)
        self.assertIsNone(reflection_receipt({}, None))

    def test_unverified_prose_is_not_replaced_with_a_verified_conclusion(self):
        records = records_for(["north"])
        value = assessment(records, 1)
        value["outcome"] = "This prose is intentionally not interpreted by the pure factual validator."
        self.assertEqual(validate_assessment(value, records, 1)["outcome"], value["outcome"])
        self.assertNotIn("verified", reflection_receipt(records, 1))

    def test_same_question_declared_trials_are_valid_comparisons(self):
        records = records_for(["north", "reset", "north"])
        self.assertIsNone(validate_trial_comparisons([note()], records))
        refused = records_for(["interact", "reset", "interact"])
        self.assertIsNone(validate_trial_comparisons([note("interact")], refused))

    def test_setup_relabeling_different_question_and_target_are_not_trials(self):
        for field, wrong in (("action_role", "setup"), ("experiment_control", "interact"), ("experiment_id", "renamed-question")):
            records = records_for(["north", "reset", "north"])
            records[3]["decision"][field] = wrong
            with self.subTest(field=field), self.assertRaises(InvalidDecision):
                validate_trial_comparisons([note()], records)

    def test_comparison_future_reset_wrong_control_and_wrong_results_are_rejected(self):
        records = records_for(["north", "reset", "north"])
        invalid = [
            {**note(), "comparison": {**note()["comparison"], "repeated_step": 4}},
            {**note(), "comparison": {**note()["comparison"], "repeated_step": 2}},
            {**note(), "control": "interact"},
            {**note(), "comparison": {**note()["comparison"], "result": "different"}},
            {**note(), "comparison": {**note()["comparison"], "original_step": True}},
        ]
        for value in invalid:
            with self.subTest(value=value), self.assertRaises(InvalidDecision):
                validate_trial_comparisons([value], records)
        self.assertIsNone(validate_trial_comparisons([{**note(), "status": "uncertain", "comparison": None, "evidence_steps": [1]}], records))

    def test_extra_evidence_uses_ordinary_checks_without_forcing_trial_role(self):
        records = records_for(["north", "reset", "north", "north"])
        records[4]["decision"]["action_role"] = "setup"
        self.assertIsNone(validate_trial_comparisons([note(refs=(1, 3, 4))], records))
        records[4]["before"]["player"]["position"] = [1, 2]
        with self.assertRaises(InvalidDecision):
            validate_trial_comparisons([note(refs=(1, 3, 4))], records)

    def test_experiment_context_is_past_factual_bookkeeping_in_order(self):
        records = records_for(["north", "interact", "reset", "north"])
        before = deepcopy(records)
        bindings = {"movement": "north", "interaction": "interact"}
        context = experiment_context(bindings, records, {1, 2, 3})
        self.assertEqual(context, {"contract_version": 1, "experiments": [
            {"id": "movement", "control": "north", "trial_steps": [1, 4], "setup_steps": [3], "assessed_steps": [1, 3]},
            {"id": "interaction", "control": "interact", "trial_steps": [2], "setup_steps": [], "assessed_steps": [2]},
        ]})
        self.assertEqual(records, before)
        self.assertEqual(bindings, {"movement": "north", "interaction": "interact"})
        context["experiments"][0]["trial_steps"].append(999)
        self.assertEqual(experiment_context(bindings, records, set())["experiments"][0]["trial_steps"], [1, 4])

    def test_experiment_context_rejects_unknown_binding_or_forged_metadata(self):
        records = records_for(["north"])
        for bindings in ({}, {"movement": "interact"}, {"movement": True}):
            with self.subTest(bindings=bindings), self.assertRaises(InvalidDecision):
                experiment_context(bindings, records, set())
        records[1]["step"] = True
        with self.assertRaises(InvalidDecision):
            experiment_context({"movement": "north"}, records, set())

    def test_projection_receipts_and_context_do_not_expose_rules_or_hints(self):
        records = records_for(["north", "reset", "north"])
        for record in records.values():
            record["evaluation"] = {"hidden_contract": "SECRET_EVALUATION"}
            for observation in (record["before"], record["after"]):
                observation.update(rules=["SECRET_RULES"], goal="SECRET_GOAL", neighbors={"north": {"passable": False}})
                observation["player"]["strategy"] = "SECRET_STRATEGY"
                observation["objects"]["gate"]["needs_key"] = "SECRET_PASSABILITY"
        text = json.dumps({
            "projection": outcome_projection(records[1]["before"]),
            "receipt": reflection_receipt(records, 3),
            "context": experiment_context({"movement": "north"}, records, {1, 2}),
        })
        for forbidden in ("SECRET_", "passable", "needs_key", "rules", "goal", "strategy", "evaluation"):
            self.assertNotIn(forbidden, text)

    def test_rejection_and_receipt_creation_leave_all_inputs_unchanged(self):
        records = records_for(["interact", "reset", "interact"])
        values = [assessment(records, 3), intent(), {"interaction": "interact", "movement": "north"}]
        originals = deepcopy([records, *values])
        bad = {**values[0], "position_before": [9, 9]}
        with self.assertRaises(InvalidDecision):
            validate_assessment(bad, records, 3)
        reflection_receipt(records, 2)
        validate_intent(values[1], values[2])
        self.assertEqual([records, *values], originals)

    def test_one_episode_interaction_loop_has_no_assessed_cross_episode_pairs(self):
        records = records_for(["interact", "interact", "interact"])
        progress = trial_progress({"interaction": "interact"}, records, {1, 2})
        self.assertEqual(progress, {
            "scope": "Completed trial-role actions grouped by declared experiment ID and control; starting conditions exclude ticks and feedback.",
            "claims_verified": False, "experiments": [{
                "id": "interaction", "control": "interact", "trial_count": 3,
                "assessed_trial_count": 2, "trial_episodes": [0],
                "same_start_repeated_trial_count": 2, "assessed_cross_episode_pair_count": 0,
            }],
        })

    def test_progress_excludes_all_setup_and_reset_actions(self):
        records = records_for(["north", "interact", "reset", "north"])
        records[1]["decision"]["action_role"] = "setup"
        records[2]["decision"]["action_role"] = "setup"
        bindings = {"movement": "north", "interaction": "interact"}
        progress = trial_progress(bindings, records, {1, 2, 3, 4})
        self.assertEqual(progress["experiments"], [
            {"id": "movement", "control": "north", "trial_count": 1,
             "assessed_trial_count": 1, "trial_episodes": [1],
             "same_start_repeated_trial_count": 0, "assessed_cross_episode_pair_count": 0},
            {"id": "interaction", "control": "interact", "trial_count": 0,
             "assessed_trial_count": 0, "trial_episodes": [],
             "same_start_repeated_trial_count": 0, "assessed_cross_episode_pair_count": 0},
        ])
        records[3]["decision"]["action_role"] = "trial"
        with self.assertRaises(InvalidDecision):
            trial_progress(bindings, records, {1, 2, 3, 4})

    def test_progress_counts_same_start_pair_after_reset_only_after_both_assessed(self):
        records = records_for(["interact", "reset", "interact"])
        bindings = {"interaction": "interact", "movement": "north"}
        for assessed, expected in (({1}, 0), ({3}, 0), ({1, 3}, 1)):
            with self.subTest(assessed=assessed):
                item = trial_progress(bindings, records, assessed)["experiments"][0]
                self.assertEqual(item["trial_count"], 2)
                self.assertEqual(item["trial_episodes"], [0, 1])
                self.assertEqual(item["same_start_repeated_trial_count"], 1)
                self.assertEqual(item["assessed_cross_episode_pair_count"], expected)
        records[3]["before"]["tick"] = 99
        records[3]["before"]["events"] = ["Prior feedback differs."]
        self.assertEqual(trial_progress(bindings, records, {1, 3})["experiments"][0]["assessed_cross_episode_pair_count"], 1)

    def test_progress_compares_actual_start_states_and_json_types(self):
        changes = [
            lambda before: before["player"].update(position=[1, 2]),
            lambda before: before["player"].update(inventory=["key"]),
            lambda before: before["objects"]["gate"].update(open=True),
            lambda before: before["objects"].update(key=None),
            lambda before: before["map"].__setitem__(0, "changed map"),
            lambda before: before["player"].update(position=[True, 1]),
        ]
        for change in changes:
            records = records_for(["north", "reset", "north"])
            change(records[3]["before"])
            with self.subTest(before=records[3]["before"]):
                item = trial_progress({"movement": "north"}, records, {1, 3})["experiments"][0]
                self.assertEqual(item["same_start_repeated_trial_count"], 0)
                self.assertEqual(item["assessed_cross_episode_pair_count"], 0)

    def test_progress_keeps_distinct_controls_and_experiment_ids_separate(self):
        records = records_for(["north", "reset", "interact"])
        items = trial_progress({"movement": "north", "interaction": "interact"}, records, {1, 3})["experiments"]
        self.assertTrue(all(item["trial_count"] == 1 and item["same_start_repeated_trial_count"] == 0 for item in items))
        records = records_for(["north", "reset", "north"])
        records[3]["decision"]["experiment_id"] = "other-north-question"
        items = trial_progress({"movement": "north", "other-north-question": "north"}, records, {1, 3})["experiments"]
        self.assertTrue(all(item["trial_count"] == 1 and item["assessed_cross_episode_pair_count"] == 0 for item in items))
        records[3]["decision"]["experiment_control"] = "interact"
        with self.assertRaises(InvalidDecision):
            trial_progress({"movement": "north", "other-north-question": "north"}, records, {1, 3})

    def test_progress_repeat_count_counts_trials_once_and_pair_count_counts_pairs(self):
        records = records_for(["north", "north", "reset", "north", "north"])
        item = trial_progress({"movement": "north"}, records, {1, 2, 4, 5})["experiments"][0]
        self.assertEqual(item["same_start_repeated_trial_count"], 3)
        self.assertEqual(item["assessed_cross_episode_pair_count"], 4)
        self.assertEqual(item["trial_episodes"], [0, 1])

    def test_progress_is_pure_has_no_policy_and_leaves_existing_context_unchanged(self):
        records = records_for(["north", "reset", "north"])
        bindings, assessed = {"movement": "north"}, {1, 3}
        original_context = experiment_context(bindings, records, assessed)
        for record in records.values():
            record["before"].update(goal="SECRET_GOAL", rules=["SECRET_RULES"], passable="SECRET_HINT")
        inputs = deepcopy([records, bindings, assessed])
        progress = trial_progress(bindings, records, assessed)
        text = json.dumps(progress)
        for prohibited in ("SECRET", "recommend", "next_action", "reset_steps", "hypotheses", "goal", "rules", "passable"):
            self.assertNotIn(prohibited, text)
        self.assertFalse(progress["claims_verified"])
        self.assertEqual(experiment_context(bindings, records, assessed), original_context)
        self.assertEqual([records, bindings, assessed], inputs)


if __name__ == "__main__":
    unittest.main()
