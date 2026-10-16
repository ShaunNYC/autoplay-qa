"""Single-source forecast fixtures, never evidence of live model learning."""

from copy import deepcopy
import json
import unittest

from autoplay_qa import forecast, reflection
from autoplay_qa.game import ACTIONS, KeyGateGame
from autoplay_qa.nebius import InvalidDecision
from tests.test_reflection_contract import records_for


def compact_assessment(records, step):
    receipt = forecast.reflection_receipt(records, step)
    return {
        "step": step, "outcome": "Fixture: compared the sole prior forecast with recorded feedback.",
        "prediction_verdict": receipt["prediction_verdict"],
        "mismatched_facets": receipt["mismatched_facets"],
    }


class ForecastContractTests(unittest.TestCase):
    def test_version_and_raw_fields_remove_the_independent_prediction(self):
        self.assertEqual(forecast.REFLECTION_CONTRACT_VERSION, 2)
        self.assertEqual(reflection.REFLECTION_CONTRACT_VERSION, 1)
        self.assertEqual(len(forecast.DECISION_FIELDS), 10)
        self.assertNotIn("prediction", forecast.DECISION_FIELDS)
        self.assertNotIn("prediction_source", forecast.DECISION_FIELDS)
        self.assertEqual(forecast.ASSESSMENT_FIELDS, {"step", "outcome", "prediction_verdict", "mismatched_facets"})

    def test_readable_prediction_comes_only_from_literal_forecast_and_actual_control(self):
        expected = reflection.outcome_projection(KeyGateGame().observe())
        rendered = forecast.render_prediction("interact", expected)
        self.assertEqual(rendered, 'Expected after interact: position=[1,1]; inventory=[]; key=[1,3]; gate_open=false; status="playing".')
        record = forecast.prediction_record("interact", expected)
        self.assertEqual(record, {"prediction": rendered, "prediction_source": "rendered_from_expected_outcome"})
        for action in (*ACTIONS, "reset"):
            with self.subTest(action=action):
                self.assertTrue(forecast.render_prediction(action, expected).startswith(f"Expected after {action}: "))

    def test_direct_v21_conflict_renders_closed_gate_and_only_key_is_mismatched(self):
        records = records_for(["interact"])
        records[1]["decision"].update(
            prediction="The gate will open; this independent prose must not become a second forecast.",
            expected_outcome={**records[1]["decision"]["expected_outcome"], "key": None},
        )
        receipt = forecast.reflection_receipt(records, 1)
        self.assertEqual(receipt["mismatched_facets"], ["key"])
        self.assertEqual(receipt["prediction_verdict"], "mismatched")
        self.assertIn("gate_open=false", receipt["prediction"])
        self.assertIn("key=null", receipt["prediction"])
        self.assertNotIn("will open", receipt["prediction"])
        self.assertEqual(receipt["observed_events"], ["There is no gate within reach."])

    def test_thinking_v21_wrong_movement_forecast_stays_a_valid_interaction_forecast(self):
        records = records_for(["interact"])
        records[1]["decision"]["expected_outcome"].update(position=[1, 2], key=None)
        checked = forecast.validate_intent(records[1]["decision"], {})
        self.assertEqual(checked["expected_outcome"]["position"], [1, 2])
        receipt = forecast.reflection_receipt(records, 1)
        self.assertEqual(receipt["control"], "interact")
        self.assertEqual(receipt["mismatched_facets"], ["position", "key"])
        self.assertIn("Expected after interact: position=[1,2]", receipt["prediction"])
        self.assertNotIn("south", receipt["prediction"])

    def test_wrong_future_forecasts_are_not_corrected_before_execution(self):
        records = records_for(["north"])
        value = records[1]["decision"]
        value["expected_outcome"].update(position=[-10, 99], inventory=["imagined"], key=None, gate_open=True, status="won")
        prior = deepcopy(value)
        intent = forecast.validate_intent(value, {})
        self.assertEqual(intent["expected_outcome"], prior["expected_outcome"])
        rendered = forecast.render_prediction(value["action"], value["expected_outcome"])
        self.assertIn("position=[-10,99]", rendered)
        self.assertEqual(value, prior)
        receipt = forecast.reflection_receipt(records, 1)
        self.assertEqual(receipt["mismatched_facets"], list(forecast.FACET_ORDER))

    def test_literal_whitespace_unicode_and_order_are_preserved(self):
        observed = reflection.outcome_projection(KeyGateGame().observe())
        expected = {**observed, "inventory": [" key ", "é"]}
        self.assertIn('inventory=[" key ","é"]', forecast.render_prediction("interact", expected))
        self.assertEqual(forecast.mismatched_facets(expected, {**observed, "inventory": ["key", "é"]}), ["inventory"])
        self.assertEqual(forecast.mismatched_facets(expected, {**observed, "inventory": ["é", " key "]}), ["inventory"])

    def test_renderer_is_deterministic_with_different_dictionary_order_and_unbounded_derived_text(self):
        expected = reflection.outcome_projection(KeyGateGame().observe())
        reordered = dict(reversed(list(expected.items())))
        self.assertEqual(forecast.render_prediction("north", expected), forecast.render_prediction("north", reordered))
        expected["inventory"] = ["x" * 32] * 8
        rendered = forecast.render_prediction("wait", expected)
        self.assertGreater(len(rendered), 240)
        self.assertEqual(rendered.count("x" * 32), 8)
        self.assertTrue(rendered.endswith('status="playing".'))

    def test_finish_has_a_labeled_null_forecast_and_other_actions_require_one(self):
        self.assertEqual(forecast.prediction_record("finish", None), {
            "prediction": "Finish executes no game action; no state forecast.",
            "prediction_source": forecast.PREDICTION_SOURCE,
        })
        for action, expected in (("finish", reflection.outcome_projection(KeyGateGame().observe())), ("north", None), ("reset", None), ("teleport", None), ([], None)):
            with self.subTest(action=action), self.assertRaises(InvalidDecision):
                forecast.render_prediction(action, expected)

    def test_forecast_facet_types_and_bounds_are_still_authoritative(self):
        original = reflection.outcome_projection(KeyGateGame().observe())
        for facet, wrong in (("position", [True, 1]), ("position", [1.0, 1]), ("gate_open", 0), ("inventory", ["x" * 33]), ("key", "unknown"), ("status", "unknown")):
            with self.subTest(facet=facet), self.assertRaises(InvalidDecision):
                forecast.mismatched_facets({**original, facet: wrong}, original)
        self.assertEqual(forecast.mismatched_facets(original, original), [])

    def test_compact_assessment_has_four_fields_and_exact_original_forecast_verdict(self):
        records = records_for(["interact"])
        records[1]["decision"]["expected_outcome"]["gate_open"] = True
        value = compact_assessment(records, 1)
        self.assertEqual(value["mismatched_facets"], ["gate_open"])
        self.assertEqual(forecast.validate_assessment(value, records, 1), value)
        self.assertEqual(set(value), forecast.ASSESSMENT_FIELDS)
        self.assertNotIn("control", value)
        self.assertNotIn("observed_outcome", value)

    def test_wrong_verdict_differing_facets_duplicates_and_order_are_rejected(self):
        records = records_for(["interact"])
        records[1]["decision"]["expected_outcome"].update(position=[2, 1], key=None, gate_open=True)
        original = compact_assessment(records, 1)
        self.assertEqual(original["mismatched_facets"], ["position", "key", "gate_open"])
        invalid = [
            {**original, "prediction_verdict": "matched"},
            {**original, "prediction_verdict": True},
            {**original, "mismatched_facets": []},
            {**original, "mismatched_facets": ["key", "gate_open", "position"]},
            {**original, "mismatched_facets": ["position", "key", "gate_open", "key"]},
            {**original, "mismatched_facets": "position,key,gate_open"},
            {**original, "mismatched_facets": [True]},
        ]
        for value in invalid:
            with self.subTest(value=value), self.assertRaises(InvalidDecision):
                forecast.validate_assessment(value, records, 1)
        self.assertEqual(forecast.validate_assessment(original, records, 1), original)

    def test_assessment_null_exact_shape_step_and_text_bounds_are_checked(self):
        records = records_for(["north"])
        original = compact_assessment(records, 1)
        self.assertIsNone(forecast.validate_assessment(None, {}, None))
        for value in (None, {**original, "step": True}, {**original, "step": 2}, {**original, "control": "north"}, {**original, "outcome": " "}, {**original, "outcome": "x" * 361}, {key: item for key, item in original.items() if key != "mismatched_facets"}):
            with self.subTest(value=value), self.assertRaises(InvalidDecision):
                forecast.validate_assessment(value, records, 1)
        with self.assertRaises(InvalidDecision):
            forecast.validate_assessment(original, records, None)
        maximum = {**original, "outcome": " " + "é" * 360 + " "}
        self.assertEqual(len(forecast.validate_assessment(maximum, records, 1)["outcome"]), 360)

    def test_receipts_can_be_historical_but_assessments_must_be_latest(self):
        records = records_for(["north", "reset", "north"])
        self.assertEqual(forecast.reflection_receipt(records, 1)["control"], "north")
        self.assertEqual(forecast.reflection_receipt(records, 2)["observed_events"], ["Environment started."])
        self.assertEqual(forecast.reflection_receipt(records, 3)["episode_before"], 1)
        with self.assertRaises(InvalidDecision):
            forecast.validate_assessment(compact_assessment(records, 1), records, 1)
        for pending in (True, 99):
            with self.subTest(pending=pending), self.assertRaises(InvalidDecision):
                forecast.reflection_receipt(records, pending)
        self.assertIsNone(forecast.reflection_receipt(records, None))

    def test_schema_constrains_only_completed_reflection_and_leaves_forecasts_fallible(self):
        records = records_for(["interact"])
        records[1]["decision"]["expected_outcome"]["key"] = None
        schema = forecast.assessment_schema(records, 1)
        self.assertEqual(set(schema["properties"]), forecast.ASSESSMENT_FIELDS)
        self.assertEqual(schema["properties"]["prediction_verdict"]["enum"], ["mismatched"])
        self.assertEqual(schema["properties"]["mismatched_facets"]["enum"], [["key"]])
        self.assertFalse(schema["additionalProperties"])
        self.assertEqual(forecast.assessment_schema({}, None), {"type": "null"})
        future = forecast.prediction_schema()
        self.assertNotIn("enum", future["properties"]["position"]["items"])
        self.assertNotIn("enum", future["properties"]["gate_open"])

    def test_receipts_are_detached_and_no_model_learning_is_fabricated(self):
        records = records_for(["north"])
        prior = deepcopy(records)
        value = compact_assessment(records, 1)
        value["outcome"] = "Intentionally uninterpreted model prose."
        self.assertEqual(forecast.validate_assessment(value, records, 1)["outcome"], value["outcome"])
        receipt = forecast.reflection_receipt(records, 1)
        for forbidden in ("claim", "hypotheses", "verified", "learning_complete"):
            self.assertNotIn(forbidden, receipt)
        receipt["expected_outcome"]["position"][0] = 999
        receipt["mismatched_facets"].append("invented")
        self.assertEqual(records, prior)

    def test_private_metadata_and_developer_rules_are_absent(self):
        records = records_for(["north", "reset", "north"])
        for record in records.values():
            record["evaluation"] = {"private": "SECRET_CHECK"}
            for observation in (record["before"], record["after"]):
                observation.update(goal="SECRET_GOAL", rules=["SECRET_RULE"], neighbors={"north": {"passable": "SECRET_HINT"}})
                observation["player"]["private"] = "SECRET_PLAYER_HINT"
        text = json.dumps(forecast.reflection_receipt(records, 3))
        for forbidden in ("SECRET", "passable", "rules", "goal", "evaluation"):
            self.assertNotIn(forbidden, text)

    def test_version_one_helpers_and_contexts_remain_unchanged(self):
        self.assertIs(forecast.experiment_context, reflection.experiment_context)
        self.assertIs(forecast.trial_progress, reflection.trial_progress)
        self.assertIs(forecast.validate_trial_comparisons, reflection.validate_trial_comparisons)
        self.assertIs(forecast.validate_intent, reflection.validate_intent)
        records = records_for(["north"])
        receipt = reflection.reflection_receipt(records, 1)
        self.assertNotIn("prediction_source", receipt)
        self.assertNotIn("mismatched_facets", receipt)


if __name__ == "__main__":
    unittest.main()
