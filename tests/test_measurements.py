"""Assessed witness facts are separate from fallible forecasts and free prose."""

from copy import deepcopy
import json
import unittest

from autoplay_qa import forecast, grounding, measurements
from autoplay_qa.nebius import InvalidDecision
from tests.test_forecast_contract import compact_assessment
from tests.test_reflection_contract import note as comparison_note, records_for


def effect(step=1, facet="position", before=None, after=None):
    return {"condition_step": step, "facet": facet,
            "before": [1, 1] if before is None else before,
            "after": [1, 1] if after is None else after}


def note(control="north", refs=(1,), selected=None):
    return {
        "id": "effect-note", "claim": "Tentative model prose remains unverified.",
        "control": control, "status": "uncertain", "evidence_steps": list(refs),
        "comparison": None, "effect": selected if selected is not None else effect(),
    }


class MeasurementTests(unittest.TestCase):
    def test_version_and_existing_raw_fields_and_renderer_are_unchanged(self):
        self.assertEqual(measurements.REFLECTION_CONTRACT_VERSION, 4)
        self.assertEqual(grounding.REFLECTION_CONTRACT_VERSION, 3)
        self.assertEqual(measurements.DECISION_FIELDS, forecast.DECISION_FIELDS)
        self.assertIs(measurements.reflection_receipt, grounding.reflection_receipt)
        self.assertIs(measurements.render_prediction, forecast.render_prediction)

    def test_wait_forecast_error_cannot_become_observed_position_change(self):
        records = records_for(["east", "wait"])
        records[2]["decision"]["expected_outcome"]["position"] = [1, 1]
        selected = effect(2, before=[2, 1], after=[2, 1])
        learned = note("wait", [2], selected)
        self.assertEqual(measurements.validate_effect(selected, learned, records, {2}), selected)
        wrong = {**selected, "before": [1, 1]}
        with self.assertRaisesRegex(InvalidDecision, "witness state"):
            measurements.validate_effect(wrong, learned, records, {2})
        receipt = measurements.reflection_receipt(records, 2)
        self.assertEqual(receipt["mismatched_facets"], ["position"])
        self.assertEqual(receipt["actual_changed_facets"], [])

    def test_refused_interaction_has_literal_world_key_and_empty_inventory(self):
        records = records_for(["interact"])
        records[1]["decision"]["expected_outcome"]["key"] = None
        selected = effect(1, "key", [1, 3], [1, 3])
        learned = note("interact", [1], selected)
        self.assertEqual(measurements.validate_effect(selected, learned, records, {1}), selected)
        for wrong in ({**selected, "before": None}, {**selected, "after": None}):
            with self.assertRaises(InvalidDecision):
                measurements.validate_effect(wrong, learned, records, {1})
        rows = measurements.evidence_measurements([learned], records)[0]["measurements"]
        self.assertFalse(rows[0]["actual_changed"])
        inventory = {"condition_step": 1, "facet": "inventory", "before": [], "after": []}
        self.assertEqual(measurements.validate_effect(inventory, learned, records, {1}), inventory)

    def test_actual_pickup_and_gate_effects_are_selected_from_actual_rows(self):
        records = records_for(["south", "south", "east", "east", "interact"])
        pickup = {"condition_step": 2, "facet": "inventory", "before": [], "after": ["key"]}
        self.assertEqual(measurements.validate_effect(pickup, note("south", [2], pickup), records, {2}), pickup)
        gate = {"condition_step": 5, "facet": "gate_open", "before": False, "after": True}
        self.assertEqual(measurements.validate_effect(gate, note("interact", [5], gate), records, {5}), gate)

    def test_only_witness_values_are_bound_other_actual_measurements_can_differ(self):
        records = records_for(["north", "reset", "north"])
        records[3]["after"]["player"]["position"] = [2, 1]
        selected = effect()
        learned = note("north", [1, 3], selected)
        learned["status"] = "contradicted"
        self.assertEqual(measurements.validate_effect(selected, learned, records, {1, 3}), selected)
        receipt = measurements.evidence_measurements([learned], records)[0]
        self.assertEqual([row["after"] for row in receipt["measurements"]], [[1, 1], [2, 1]])
        self.assertEqual([row["actual_changed"] for row in receipt["measurements"]], [False, True])
        self.assertFalse(receipt["claims_verified"])
        self.assertEqual(receipt["source"], "actual_recorded_state")

    def test_witness_must_be_cited_assessed_and_match_actual_game_control(self):
        records = records_for(["north", "reset", "interact"])
        invalid = [
            (effect(1), note("north", [3]), {1, 3}),
            (effect(1), note("north", [1]), set()),
            (effect(2), note("north", [2]), {2}),
            (effect(3), note("north", [3]), {3}),
            (effect(4), note("north", [4]), {4}),
            (effect(True), note("north", [1]), {1}),
        ]
        for value, learned, assessed in invalid:
            with self.subTest(value=value, assessed=assessed), self.assertRaises(InvalidDecision):
                measurements.validate_effect(value, learned, records, assessed)
        with self.assertRaises(InvalidDecision):
            measurements.validate_effect(effect(), note(), records, None)

    def test_game_setup_can_be_an_assessed_effect_witness(self):
        records = records_for(["interact"])
        records[1]["decision"].update(experiment_id="movement", experiment_control="north", action_role="setup")
        selected = effect()
        learned = note("interact", [1], selected)
        self.assertEqual(measurements.validate_effect(selected, learned, records, {1}), selected)
        self.assertEqual(measurements.evidence_measurements([learned], records)[0]["measurements"][0]["action_role"], "setup")
        records = records_for(["north", "reset", "north"])
        records[3]["decision"]["action_role"] = "setup"
        learned = {**comparison_note(), "effect": effect()}
        with self.assertRaises(InvalidDecision):
            measurements.validate_trial_comparisons([learned], records)

    def test_absent_evidence_requires_null_effect_and_cited_evidence_requires_one(self):
        proposal = note("north", [], None)
        proposal["effect"] = None
        self.assertIsNone(measurements.validate_effect(None, proposal, {}, set()))
        with self.assertRaises(InvalidDecision):
            measurements.validate_effect(effect(), proposal, {}, set())
        with self.assertRaises(InvalidDecision):
            measurements.validate_effect(None, note(), records_for(["north"]), {1})
        self.assertEqual(measurements.evidence_measurements([proposal], {}), [])

    def test_exact_effect_keys_types_and_other_reference_metadata_are_checked(self):
        records = records_for(["north"])
        invalid = [
            {**effect(), "claim": "extra"}, {key: value for key, value in effect().items() if key != "after"},
            {**effect(), "facet": "tick"}, {**effect(), "before": [True, 1]},
            {**effect(), "before": [1.0, 1]}, {**effect(), "after": [1, 2]},
        ]
        for value in invalid:
            with self.subTest(value=value), self.assertRaises(InvalidDecision):
                measurements.validate_effect(value, note(), records, {1})
        for refs in ([1, 1], [True], [1, 2], [1, "2"]):
            with self.subTest(refs=refs), self.assertRaises(InvalidDecision):
                measurements.validate_effect(effect(), note(refs=refs), records, {1})

    def test_effect_schema_offers_assessed_and_pending_game_facts_only(self):
        records = records_for(["north", "reset", "interact"])
        schema = measurements.effect_schema(records, {1}, 3)
        variants = schema["anyOf"][1:]
        self.assertEqual(len(variants), 10)
        self.assertEqual({variant["properties"]["condition_step"]["enum"][0] for variant in variants}, {1, 3})
        self.assertTrue(all(set(variant["required"]) == measurements.EFFECT_FIELDS and not variant["additionalProperties"] for variant in variants))
        filtered = measurements.effect_schema(records, {1}, 3, control="interact")["anyOf"][1:]
        self.assertEqual(len(filtered), 5)
        self.assertTrue(all(variant["properties"]["condition_step"]["enum"] == [3] for variant in filtered))
        reset_only = records_for(["reset"])
        self.assertEqual(measurements.effect_schema(reset_only, set(), 1), {"type": "null"})
        self.assertEqual(measurements.effect_schema(records, {True}), {"type": "null"})
        for pending in (1, True, 4):
            with self.subTest(pending=pending), self.assertRaises(InvalidDecision):
                measurements.effect_schema(records, set(), pending)

    def test_assessment_cannot_equate_forecast_mismatch_with_actual_change(self):
        records = records_for(["east", "wait"])
        records[2]["decision"]["expected_outcome"]["position"] = [1, 1]
        value = {**compact_assessment(records, 2), "actual_changed_facets": []}
        self.assertEqual(measurements.validate_assessment(value, records, 2), value)
        self.assertEqual(len(value), 5)
        with self.assertRaises(InvalidDecision):
            measurements.validate_assessment({**value, "actual_changed_facets": ["position"]}, records, 2)
        with self.assertRaises(InvalidDecision):
            measurements.validate_assessment(compact_assessment(records, 2), records, 2)
        schema = measurements.assessment_schema(records, 2)
        self.assertEqual(schema["properties"]["actual_changed_facets"]["enum"], [[]])
        self.assertEqual(schema["properties"]["mismatched_facets"]["enum"], [["position"]])
        self.assertIsNone(measurements.validate_assessment(None, {}, None))
        self.assertEqual(measurements.assessment_schema({}, None), {"type": "null"})

    def test_comparison_wrappers_strip_only_effect_and_preserve_core_facts(self):
        records = records_for(["north", "reset", "north"])
        learned = {**comparison_note(), "effect": effect()}
        original = deepcopy(learned)
        receipts = measurements.comparison_receipts([learned], records)
        self.assertEqual(receipts[0]["control"], "north")
        self.assertIsNone(measurements.validate_trial_comparisons([learned], records))
        self.assertEqual(learned, original)
        with self.assertRaises(InvalidDecision):
            measurements.comparison_receipts([{**learned, "extra": True}], records)

    def test_measurement_receipts_include_each_actual_reference_without_claim_text(self):
        records = records_for(["north", "reset", "north"])
        learned = note("north", [1, 3], effect())
        receipt = measurements.evidence_measurements({"effect-note": learned}, records)[0]
        self.assertEqual(receipt, {
            "hypothesis_id": "effect-note", "control": "north", "facet": "position", "condition_step": 1,
            "measurements": [
                {"step": 1, "episode": 0, "action_role": "trial", "before": [1, 1], "after": [1, 1], "actual_changed": False},
                {"step": 3, "episode": 1, "action_role": "trial", "before": [1, 1], "after": [1, 1], "actual_changed": False},
            ], "source": "actual_recorded_state", "claims_verified": False,
        })
        self.assertNotIn("claim", receipt)

    def test_goal_context_uses_first_executed_goal_and_none_for_unexecuted_ids(self):
        records = records_for(["north"])
        records[1]["decision"]["experiment_goal"] = "  Compare a chosen movement effect.  "
        context = measurements.experiment_context({"movement": "north", "unexecuted": "interact"}, records, {1})
        self.assertEqual(context["experiments"][0]["goal"], "Compare a chosen movement effect.")
        self.assertIsNone(context["experiments"][1]["goal"])

    def test_receipts_and_schemas_are_detached_private_metadata_is_not_copied(self):
        records = records_for(["interact"])
        for observation in (records[1]["before"], records[1]["after"]):
            observation.update(goal="SECRET_GOAL", rules=["SECRET_RULE"], passable="SECRET_HINT")
            observation["player"]["private"] = "SECRET_PLAYER"
        learned = note("interact", [1], effect())
        originals = deepcopy([records, learned])
        checked = measurements.validate_effect(learned["effect"], learned, records, {1})
        checked["before"][0] = 99
        receipts = measurements.evidence_measurements([learned], records)
        receipts[0]["measurements"][0]["before"][0] = 99
        schema = measurements.effect_schema(records, {1})
        text = json.dumps([schema, measurements.evidence_measurements([learned], records)])
        for forbidden in ("SECRET", "rules", "goal", "passable", "private", "Tentative model prose"):
            self.assertNotIn(forbidden, text)
        self.assertEqual([records, learned], originals)


if __name__ == "__main__":
    unittest.main()
