"""Local schema factoring preserves complete constraints and literal data."""

from copy import deepcopy
import json
from pathlib import Path
import unittest

from autoplay_qa.schema_refs import expand_schema, factor_schema


def fixture_schema():
    repeated = {
        "type": "object", "properties": {
            "claim": {"type": "string", "minLength": 1, "maxLength": 240},
            "values": {"type": "array", "minItems": 1, "maxItems": 8,
                       "items": {"type": "integer", "enum": [1, 2, 3]}},
            "control": {"type": "string", "enum": ["north", "south", "east", "west", "interact", "wait"]},
        }, "required": ["claim", "values", "control"], "additionalProperties": False,
    }
    return {
        "type": "object", "properties": {"first": deepcopy(repeated), "second": deepcopy(repeated)},
        "required": ["first", "second"], "additionalProperties": False,
        "anyOf": [{"properties": {"first": deepcopy(repeated)}}, {"properties": {"second": deepcopy(repeated)}}],
    }


class SchemaReferenceTests(unittest.TestCase):
    def test_repeated_complete_nodes_are_factored_with_exact_roundtrip(self):
        original = fixture_schema()
        compact = factor_schema(original)
        self.assertIn("$defs", compact)
        self.assertEqual(compact["properties"]["first"], compact["properties"]["second"])
        self.assertEqual(expand_schema(compact), original)
        self.assertLess(len(json.dumps(compact)), len(json.dumps(original)))

    def test_factor_and_expansion_are_pure_detached_and_deterministic(self):
        original = fixture_schema()
        saved = deepcopy(original)
        compact = factor_schema(original)
        self.assertEqual(factor_schema(original), compact)
        expanded = expand_schema(compact)
        expanded["properties"]["first"]["properties"]["claim"]["maxLength"] = 1
        self.assertEqual(original, saved)
        self.assertEqual(expand_schema(compact), saved)
        self.assertEqual(expanded["properties"]["second"]["properties"]["claim"]["maxLength"], 240)

    def test_nested_repeated_nodes_get_acyclic_internal_definitions(self):
        leaf = {"type": "string", "minLength": 1, "maxLength": 32, "description": "Bounded model-written text."}
        branch = {"type": "object", "properties": {"id": leaf, "other": deepcopy(leaf)}, "required": ["id", "other"], "additionalProperties": False}
        original = {"type": "array", "items": branch, "prefixItems": [deepcopy(branch)]}
        compact = factor_schema(original, min_size=30)
        self.assertGreaterEqual(len(compact["$defs"]), 2)
        self.assertEqual(expand_schema(compact), original)

    def test_literal_ref_like_values_in_enums_defaults_and_examples_are_not_schemas(self):
        literal = {"$ref": "https://must-not-be-fetched.example/data", "type": "literal-data"}
        original = {"type": "object", "enum": [literal], "default": literal,
                    "examples": [literal], "const": literal, "description": "A $ref in text is literal."}
        self.assertEqual(factor_schema(original, min_size=1), original)
        self.assertEqual(expand_schema(original), original)

    def test_schema_array_map_and_dependency_positions_roundtrip(self):
        repeated = fixture_schema()["properties"]["first"]
        original = {
            "allOf": [deepcopy(repeated)], "anyOf": [deepcopy(repeated)], "oneOf": [deepcopy(repeated)],
            "not": deepcopy(repeated), "if": deepcopy(repeated), "then": deepcopy(repeated), "else": deepcopy(repeated),
            "contains": deepcopy(repeated), "propertyNames": deepcopy(repeated),
            "patternProperties": {"^x": deepcopy(repeated)}, "dependentSchemas": {"x": deepcopy(repeated)},
            "dependencies": {"x": deepcopy(repeated), "y": ["x"]},
            "items": [deepcopy(repeated)], "additionalItems": False,
            "unevaluatedProperties": deepcopy(repeated), "unevaluatedItems": True,
        }
        self.assertEqual(expand_schema(factor_schema(original)), original)

    def test_existing_internal_refs_can_be_expanded_and_ref_siblings_intersect(self):
        original = {
            "$defs": {"value": {"type": "string", "maxLength": 20}},
            "type": "object", "properties": {
                "plain": {"$ref": "#/$defs/value"},
                "bounded": {"$ref": "#/$defs/value", "minLength": 2},
            },
        }
        expected = {"type": "object", "properties": {
            "plain": {"type": "string", "maxLength": 20},
            "bounded": {"allOf": [{"type": "string", "maxLength": 20}, {"minLength": 2}]},
        }}
        self.assertEqual(expand_schema(original), expected)
        self.assertEqual(expand_schema(factor_schema(original, min_size=1)), expected)

    def test_local_pointer_escapes_and_boolean_schema_targets(self):
        original = {"$defs": {"a/b~c": False}, "properties": {"blocked": {"$ref": "#/$defs/a~1b~0c"}}}
        self.assertEqual(expand_schema(original), {"properties": {"blocked": False}})
        self.assertEqual(factor_schema(True), True)
        self.assertEqual(expand_schema(False), False)

    def test_external_http_file_relative_and_anchor_refs_are_rejected(self):
        for reference in ("https://example.com/schema", "http://example.com/schema", "file:///tmp/schema.json", "relative.json", "#anchor", None, True):
            with self.subTest(reference=reference), self.assertRaises(ValueError):
                expand_schema({"$ref": reference})
        with self.assertRaises(ValueError):
            factor_schema({"$defs": {"unused": {"$ref": "https://example.com/hidden"}}, "type": "string"})
        with self.assertRaises(ValueError):
            factor_schema({"properties": {"nested": {"$id": "nested.json", "type": "string"}}})
        self.assertEqual(factor_schema({"$id": "https://example.com/document", "type": "string"}),
                         {"$id": "https://example.com/document", "type": "string"})

    def test_missing_invalid_and_non_schema_reference_targets_are_rejected(self):
        invalid = [
            {"$ref": "#/$defs/missing"},
            {"$defs": {"x": 1}, "$ref": "#/$defs/x"},
            {"$defs": {"x": {}}, "$ref": "#/$defs/x~2"},
            {"allOf": [True], "$ref": "#/allOf/01"},
        ]
        for schema in invalid:
            with self.subTest(schema=schema), self.assertRaises(ValueError):
                expand_schema(schema)

    def test_self_mutual_and_unused_cycles_are_rejected(self):
        invalid = [
            {"$ref": "#"},
            {"$defs": {"x": {"$ref": "#/$defs/x"}}, "$ref": "#/$defs/x"},
            {"$defs": {"x": {"$ref": "#/$defs/y"}, "y": {"$ref": "#/$defs/x"}}, "type": "string"},
        ]
        for schema in invalid:
            with self.subTest(schema=schema), self.assertRaisesRegex(ValueError, "Cyclic"):
                factor_schema(schema)
        for keyword in ("$dynamicRef", "$recursiveRef"):
            with self.assertRaises(ValueError):
                expand_schema({keyword: "#"})

    def test_invalid_json_schema_containers_and_factoring_bound_are_rejected(self):
        for schema in ([], {"properties": []}, {"anyOf": {}}, {"items": 1}, {"enum": [float("nan")]}):
            with self.subTest(schema=schema), self.assertRaises(ValueError):
                factor_schema(schema)
        for minimum in (0, -1, True, 1.0):
            with self.subTest(minimum=minimum), self.assertRaises(ValueError):
                factor_schema(fixture_schema(), min_size=minimum)

    def test_factoring_preserves_one_actual_v24_schema_and_reduces_size(self):
        path = Path(__file__).parents[1] / "runs/reset-retention-seed0-grounded-v24-direct-remaining.jsonl"
        if not path.exists():
            self.skipTest("Historical diagnostic recording is unavailable.")
        original = next(json.loads(line)["response_schema"] for line in path.read_text().splitlines()
                        if json.loads(line).get("type") == "request")
        compact = factor_schema(original)
        self.assertEqual(expand_schema(compact), original)
        self.assertLess(len(json.dumps(compact)), len(json.dumps(original)) / 2)


if __name__ == "__main__":
    unittest.main()
