"""Comparison repair diagnostics preserve privacy and truthful phase guidance."""

import unittest

from autoplay_qa.comparison_diagnostics import (
    COMPARISON_FIELDS, COMPARISON_SHAPE_PREFIX, MAX_DIAGNOSTIC_CHARS,
    comparison_shape_diagnostic,
)


class ComparisonDiagnosticTests(unittest.TestCase):
    def diagnostic(self, value, count=0):
        result = comparison_shape_diagnostic(value, eligible_pair_count=count)
        self.assertIn(COMPARISON_SHAPE_PREFIX, result)
        phase = "no eligible completed assessed cross-episode pair exists" if count == 0 else "a recorded pair is eligible in the cited evidence"
        self.assertTrue(result.startswith("After the repaired response meets its assessment requirements, " + phase))
        self.assertLessEqual(len(result), MAX_DIAGNOSTIC_CHARS)
        return result

    def test_zero_pairs_requires_null_without_inventing_evidence_or_action(self):
        result = self.diagnostic({"original_step": 1})
        self.assertIn("no eligible completed assessed cross-episode pair exists", result)
        self.assertIn("in this note's cited evidence", result)
        self.assertIn("comparison must be JSON null", result)
        self.assertIn("for this update", result)
        self.assertLess(result.index("comparison must be JSON null"), result.index(COMPARISON_SHAPE_PREFIX))
        self.assertNotIn("copy one eligible", result)
        for prohibited in ("reset", "north", "interact", "supported", "claim", "finish"):
            self.assertNotIn(prohibited, result)

    def test_positive_pairs_refers_only_to_recorded_exact_fields(self):
        result = self.diagnostic({}, 1)
        self.assertIn("a recorded pair is eligible in the cited evidence", result)
        self.assertIn("comparison may be JSON null", result)
        self.assertIn("copy one eligible pair's exact five fields", result)
        self.assertLess(result.index("comparison may be JSON null"), result.index(COMPARISON_SHAPE_PREFIX))
        self.assertNotIn("no eligible", result)
        self.assertNotIn("comparison must be JSON null", result)
        self.assertNotIn("step 1", result)

    def test_known_and_missing_fields_are_canonical_and_unknown_keys_are_counted(self):
        result = self.diagnostic({"result": "SECRET_RESULT", "original_step": "SECRET_STEP", "SECRET_UNKNOWN_KEY": "SECRET_VALUE"}, 1)
        self.assertIn("Known fields: original_step, result.", result)
        self.assertIn("Missing fields: repeated_step, original_episode, repeated_episode.", result)
        self.assertIn("Unknown field count: 1.", result)
        self.assertNotIn("SECRET", result)

    def test_exact_five_fields_reports_no_missing_names_and_never_echoes_values(self):
        value = {field: "PRIVATE_MARKER" for field in COMPARISON_FIELDS}
        result = self.diagnostic(value, 1)
        self.assertIn("Known fields: " + ", ".join(COMPARISON_FIELDS), result)
        self.assertIn("Missing fields: none.", result)
        self.assertIn("Unknown field count: 0.", result)
        self.assertNotIn("PRIVATE_MARKER", result)

    def test_non_object_values_report_whitelisted_json_types_without_content(self):
        cases = [(None, "null"), (True, "boolean"), (1, "integer"), (1.0, "number"),
                 ("PRIVATE_PROSE", "string"), (["PRIVATE_ARRAY"], "array")]
        for value, label in cases:
            with self.subTest(label=label):
                result = self.diagnostic(value)
                self.assertIn(f"JSON type: {label}.", result)
                self.assertIn("Known fields: none.", result)
                self.assertIn("Unknown field count: 0.", result)
                self.assertNotIn("PRIVATE", result)

    def test_null_and_object_get_correct_phase_guidance(self):
        for value in (None, {}, "unexpected"):
            with self.subTest(value=value):
                self.assertIn("comparison must be JSON null", self.diagnostic(value, 0))
                self.assertIn("comparison may be JSON null or copy one eligible pair", self.diagnostic(value, 2))

    def test_non_string_keys_are_counted_without_string_conversion(self):
        value = {1: "PRIVATE_INT", None: "PRIVATE_NONE", ("PRIVATE_TUPLE",): "PRIVATE_VALUE", "original_step": 1}
        result = self.diagnostic(value)
        self.assertIn("Known fields: original_step.", result)
        self.assertIn("Unknown field count: 3.", result)
        self.assertNotIn("PRIVATE", result)
        self.assertNotIn("Tuple", result)

    def test_unknown_type_names_and_custom_representations_are_not_exposed(self):
        class PRIVATE_TYPE:
            def __str__(self):
                raise AssertionError("Private string representation must not be invoked.")

            def __repr__(self):
                raise AssertionError("Private representation must not be invoked.")

        for value in (PRIVATE_TYPE(), ("PRIVATE",), {"PRIVATE"}, b"PRIVATE"):
            with self.subTest(kind=type(value).__name__):
                result = self.diagnostic(value)
                self.assertIn("JSON type: non-JSON.", result)
                self.assertNotIn("PRIVATE", result)

    def test_non_string_key_equality_hooks_are_not_invoked(self):
        class PRIVATE_KEY:
            def __hash__(self):
                return hash("original_step")

            def __eq__(self, _other):
                raise AssertionError("Unknown key equality must not be invoked.")

        result = self.diagnostic({PRIVATE_KEY(): "PRIVATE_VALUE"})
        self.assertIn("Known fields: none.", result)
        self.assertIn("Unknown field count: 1.", result)
        self.assertNotIn("PRIVATE", result)

    def test_dict_subclass_hooks_are_not_invoked(self):
        class PRIVATE_DICT(dict):
            def __len__(self):
                raise AssertionError("Custom len must not be invoked.")

            def __contains__(self, _key):
                raise AssertionError("Custom membership must not be invoked.")

        result = self.diagnostic(PRIVATE_DICT(PRIVATE_NAME="PRIVATE_VALUE"))
        self.assertIn("JSON type: non-JSON.", result)
        self.assertNotIn("PRIVATE", result)

    def test_long_prose_keys_and_nested_values_do_not_expand_or_leak_diagnostic(self):
        value = {"PRIVATE_KEY" * 10000: {"nested": "PRIVATE_VALUE" * 10000}, "result": ["PRIVATE_PROSE" * 10000]}
        result = self.diagnostic(value, 1)
        self.assertNotIn("PRIVATE", result)
        self.assertIn("Unknown field count: 1.", result)
        self.assertEqual(result, self.diagnostic({"arbitrary": 0, "result": None}, 1))

    def test_large_eligible_counts_do_not_expand_output(self):
        self.assertEqual(self.diagnostic({}, 1), self.diagnostic({}, 10**10000))

    def test_conditional_guidance_and_all_canonical_field_subsets_remain_bounded(self):
        for bits in range(1 << len(COMPARISON_FIELDS)):
            value = {field: None for index, field in enumerate(COMPARISON_FIELDS) if bits & (1 << index)}
            value.update({index: "PRIVATE" for index in range(1000)})
            for count in (0, 1):
                with self.subTest(bits=bits, count=count):
                    result = self.diagnostic(value, count)
                    self.assertIn("After the repaired response meets its assessment requirements, ", result)
                    self.assertLessEqual(len(result), MAX_DIAGNOSTIC_CHARS)
                    self.assertNotIn("PRIVATE", result)

    def test_invalid_counts_reject_bool_negative_and_non_integer_without_echo(self):
        for value in (True, False, -1, 1.0, "PRIVATE_COUNT", None, [], {}):
            with self.subTest(value=value), self.assertRaises(ValueError) as raised:
                comparison_shape_diagnostic({}, eligible_pair_count=value)
            self.assertEqual(str(raised.exception), "eligible_pair_count must be a nonnegative integer, excluding bool.")

    def test_inputs_are_not_mutated(self):
        value = {"original_step": [1], "PRIVATE_UNKNOWN": [2]}
        self.diagnostic(value)
        self.assertEqual(value, {"original_step": [1], "PRIVATE_UNKNOWN": [2]})


if __name__ == "__main__":
    unittest.main()
