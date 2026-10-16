"""Past observation differences are not differences in a model prediction."""

from copy import deepcopy
import json
import unittest

from autoplay_qa import forecast, grounding, reflection
from autoplay_qa.nebius import InvalidDecision
from tests.test_forecast_contract import compact_assessment
from tests.test_reflection_contract import records_for


class GroundingContractTests(unittest.TestCase):
    def test_version_three_only_enriches_receipts_and_preserves_raw_protocol(self):
        self.assertEqual(grounding.REFLECTION_CONTRACT_VERSION, 3)
        self.assertEqual(forecast.REFLECTION_CONTRACT_VERSION, 2)
        self.assertEqual(reflection.REFLECTION_CONTRACT_VERSION, 1)
        for name in (
            "ASSESSMENT_FIELDS", "DECISION_FIELDS", "FACET_ORDER", "PREDICTION_SOURCE",
            "assessment_schema", "experiment_context", "mismatched_facets", "outcome_projection",
            "prediction_record", "prediction_schema", "render_prediction", "trial_progress",
            "validate_assessment", "validate_intent", "validate_trial_comparisons",
        ):
            with self.subTest(name=name):
                self.assertIs(getattr(grounding, name), getattr(forecast, name))

    def test_interaction_forecast_key_mismatch_does_not_mean_world_key_changed(self):
        records = records_for(["interact"])
        records[1]["decision"]["expected_outcome"]["key"] = None
        receipt = grounding.reflection_receipt(records, 1)
        self.assertEqual(receipt["mismatched_facets"], ["key"])
        self.assertEqual(receipt["actual_changed_facets"], [])
        self.assertEqual(receipt["before_outcome"]["key"], [1, 3])
        self.assertEqual(receipt["observed_outcome"]["key"], [1, 3])
        self.assertEqual(receipt["before_outcome"]["inventory"], [])
        self.assertEqual(receipt["observed_outcome"]["inventory"], [])
        self.assertEqual(receipt["observed_events"], ["There is no gate within reach."])
        self.assertEqual(receipt["prediction_verdict"], "mismatched")

    def test_actual_movement_is_a_state_change_even_when_prediction_matches(self):
        records = records_for(["east"])
        receipt = grounding.reflection_receipt(records, 1)
        self.assertEqual(receipt["mismatched_facets"], [])
        self.assertEqual(receipt["actual_changed_facets"], ["position"])
        self.assertEqual(receipt["before_outcome"]["position"], [1, 1])
        self.assertEqual(receipt["observed_outcome"]["position"], [2, 1])

    def test_actual_pickup_changes_inventory_and_world_key_from_recorded_states(self):
        records = records_for(["south", "south"])
        receipt = grounding.reflection_receipt(records, 2)
        self.assertEqual(receipt["before_outcome"]["position"], [1, 2])
        self.assertEqual(receipt["before_outcome"]["inventory"], [])
        self.assertEqual(receipt["before_outcome"]["key"], [1, 3])
        self.assertEqual(receipt["observed_outcome"]["inventory"], ["key"])
        self.assertIsNone(receipt["observed_outcome"]["key"])
        self.assertEqual(receipt["actual_changed_facets"], ["position", "inventory", "key"])
        self.assertEqual(receipt["mismatched_facets"], [])

    def test_reset_actual_position_and_empty_inventory_do_not_invent_a_pickup(self):
        records = records_for(["east", "reset"])
        receipt = grounding.reflection_receipt(records, 2)
        self.assertEqual(receipt["control"], "reset")
        self.assertEqual((receipt["episode_before"], receipt["episode_after"]), (0, 1))
        self.assertEqual(receipt["before_outcome"]["position"], [2, 1])
        self.assertEqual(receipt["observed_outcome"]["position"], [1, 1])
        self.assertEqual(receipt["before_outcome"]["inventory"], [])
        self.assertEqual(receipt["observed_outcome"]["inventory"], [])
        self.assertEqual(receipt["before_outcome"]["key"], receipt["observed_outcome"]["key"])
        self.assertEqual(receipt["actual_changed_facets"], ["position"])
        self.assertEqual(receipt["observed_events"], ["Environment started."])

    def test_reset_restoration_and_inventory_loss_are_not_inferred_pickup_mechanics(self):
        records = records_for(["south", "south", "reset"])
        receipt = grounding.reflection_receipt(records, 3)
        self.assertEqual(receipt["before_outcome"]["inventory"], ["key"])
        self.assertIsNone(receipt["before_outcome"]["key"])
        self.assertEqual(receipt["observed_outcome"]["inventory"], [])
        self.assertEqual(receipt["observed_outcome"]["key"], [1, 3])
        self.assertEqual(receipt["actual_changed_facets"], ["position", "inventory", "key"])
        for prose in ("claim", "hypotheses", "mechanics", "pickup", "verified"):
            self.assertNotIn(prose, receipt)

    def test_blocked_refused_and_waited_feedback_can_occur_without_state_facet_changes(self):
        for control in ("north", "interact", "wait"):
            with self.subTest(control=control):
                records = records_for([control])
                receipt = grounding.reflection_receipt(records, 1)
                self.assertEqual(receipt["actual_changed_facets"], [])
                self.assertEqual(receipt["before_outcome"], receipt["observed_outcome"])
                self.assertEqual(receipt["observed_events"], records[1]["after"]["events"])

    def test_forecast_differences_and_actual_differences_can_overlap_or_disagree(self):
        records = records_for(["east"])
        records[1]["decision"]["expected_outcome"].update(position=[1, 1], gate_open=True)
        receipt = grounding.reflection_receipt(records, 1)
        self.assertEqual(receipt["mismatched_facets"], ["position", "gate_open"])
        self.assertEqual(receipt["actual_changed_facets"], ["position"])
        self.assertFalse(receipt["before_outcome"]["gate_open"])
        self.assertFalse(receipt["observed_outcome"]["gate_open"])

    def test_receipt_copies_do_not_alias_records_or_each_other(self):
        records = records_for(["north"])
        original = deepcopy(records)
        receipt = grounding.reflection_receipt(records, 1)
        receipt["before_outcome"]["position"][0] = 99
        receipt["before_outcome"]["inventory"].append("invented")
        receipt["actual_changed_facets"].append("invented")
        self.assertEqual(receipt["observed_outcome"]["position"], [1, 1])
        self.assertEqual(receipt["observed_outcome"]["inventory"], [])
        self.assertEqual(records, original)

    def test_private_metadata_and_initial_strategy_do_not_enter_before_projection(self):
        records = records_for(["north", "reset", "north"])
        for record in records.values():
            record["evaluation"] = {"private": "SECRET_CHECK"}
            for observation in (record["before"], record["after"]):
                observation.update(goal="SECRET_GOAL", rules=["SECRET_RULE"], neighbors={"north": {"passable": "SECRET_HINT"}})
                observation["player"]["private"] = "SECRET_PLAYER"
                observation["objects"]["gate"]["needs_key"] = "SECRET_PASSABILITY"
        receipt = grounding.reflection_receipt(records, 2)
        text = json.dumps(receipt)
        for forbidden in ("SECRET", "rules", "goal", "passable", "needs_key", "private", "evaluation", "Find the key", "unlock the gate"):
            self.assertNotIn(forbidden, text)
        self.assertEqual(set(receipt["before_outcome"]), {"position", "inventory", "key", "gate_open", "status"})

    def test_compact_assessment_still_refers_only_to_forecast_mismatches(self):
        records = records_for(["interact"])
        records[1]["decision"]["expected_outcome"]["key"] = None
        original = compact_assessment(records, 1)
        self.assertEqual(original["mismatched_facets"], ["key"])
        self.assertEqual(grounding.validate_assessment(original, records, 1), forecast.validate_assessment(original, records, 1))
        self.assertEqual(grounding.assessment_schema(records, 1), forecast.assessment_schema(records, 1))
        with self.assertRaises(InvalidDecision):
            grounding.validate_assessment({**original, "mismatched_facets": []}, records, 1)

    def test_historical_and_pending_receipts_validate_existing_steps(self):
        records = records_for(["north", "reset", "north"])
        self.assertEqual(grounding.reflection_receipt(records, 1)["actual_changed_facets"], [])
        self.assertEqual(grounding.reflection_receipt(records, 2)["actual_changed_facets"], [])
        self.assertIsNone(grounding.reflection_receipt(records, None))
        for step in (True, 0, 4, "1"):
            with self.subTest(step=step), self.assertRaises(InvalidDecision):
                grounding.reflection_receipt(records, step)

    def test_original_version_two_receipts_remain_byte_for_byte_unchanged(self):
        records = records_for(["interact", "reset", "interact"])
        records[1]["decision"]["expected_outcome"]["key"] = None
        prior = json.dumps([forecast.reflection_receipt(records, step) for step in sorted(records)], separators=(",", ":"))
        for step in sorted(records):
            receipt = grounding.reflection_receipt(records, step)
            self.assertEqual(set(receipt) - {"before_outcome", "actual_changed_facets"}, set(forecast.reflection_receipt(records, step)))
            self.assertEqual({key: value for key, value in receipt.items() if key not in {"before_outcome", "actual_changed_facets"}}, forecast.reflection_receipt(records, step))
        after = json.dumps([forecast.reflection_receipt(records, step) for step in sorted(records)], separators=(",", ":"))
        self.assertEqual(prior, after)


if __name__ == "__main__":
    unittest.main()
