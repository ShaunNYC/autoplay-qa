"""Factor repeated JSON-schema nodes into local references without weakening them.

Only schema positions are traversed: literal enum/default/example values remain
data. Expansion supports in-document JSON pointers, rejects external references
and cycles, and performs no network or filesystem reads. Decoder support for
these references still requires validation by the caller.
"""

from collections import Counter
from copy import deepcopy
import json
import re
from urllib.parse import unquote


_MAPS = {"properties", "patternProperties", "$defs", "definitions", "dependentSchemas"}
_LISTS = {"allOf", "anyOf", "oneOf", "prefixItems"}
_SINGLES = {"not", "if", "then", "else", "contains", "additionalProperties", "unevaluatedProperties", "unevaluatedItems", "propertyNames", "additionalItems", "contentSchema"}


def _json(value) -> str:
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError):
        raise ValueError("Schema must contain finite, serializable JSON values.") from None


def _map_children(node, transform):
    if type(node) is bool:
        return node
    if not isinstance(node, dict):
        raise ValueError("A schema node must be an object or boolean.")
    result = {}
    for field, value in node.items():
        if field in _MAPS:
            if not isinstance(value, dict):
                raise ValueError(f"Schema {field} must map names to schemas.")
            result[field] = {name: transform(child) for name, child in value.items()}
        elif field in _LISTS:
            if not isinstance(value, list):
                raise ValueError(f"Schema {field} must contain a list of schemas.")
            result[field] = [transform(child) for child in value]
        elif field in _SINGLES:
            result[field] = transform(value)
        elif field == "items":
            result[field] = [transform(child) for child in value] if isinstance(value, list) else transform(value)
        elif field == "dependencies":
            if not isinstance(value, dict):
                raise ValueError("Schema dependencies must be an object.")
            result[field] = {name: transform(child) if isinstance(child, (dict, bool)) else deepcopy(child) for name, child in value.items()}
        else:
            result[field] = deepcopy(value)
    return result


def expand_schema(schema):
    """Expand local references into fresh schemas; no external access is allowed.

    Definitions are removed after expansion. Reference siblings remain separate
    allOf constraints, preserving their intersection rather than overwriting the
    referenced schema. Generated references have no siblings, so fully expanded
    input schemas round-trip to the exact original structure.
    """
    _json(schema)

    def resolve(reference):
        if not isinstance(reference, str) or not reference.startswith("#"):
            raise ValueError("Only in-document schema references are supported; external references are forbidden.")
        fragment = unquote(reference[1:])
        if fragment and not fragment.startswith("/"):
            raise ValueError("Only local JSON-pointer schema references are supported.")
        target = schema
        for part in fragment.split("/")[1:]:
            if re.search(r"~(?![01])", part):
                raise ValueError("Schema reference contains an invalid JSON-pointer escape.")
            part = part.replace("~1", "/").replace("~0", "~")
            try:
                if isinstance(target, list):
                    if not part.isdigit() or len(part) > 1 and part[0] == "0":
                        raise ValueError("Invalid array index in schema reference.")
                    target = target[int(part)]
                elif isinstance(target, dict):
                    target = target[part]
                else:
                    raise ValueError("Schema reference does not identify a schema.")
            except (KeyError, IndexError):
                raise ValueError("Schema reference target does not exist.") from None
        if not isinstance(target, (dict, bool)):
            raise ValueError("Schema reference must identify an object or boolean schema.")
        return target

    def expand(node, stack=()):
        if type(node) is bool:
            return node
        if not isinstance(node, dict):
            raise ValueError("A schema node must be an object or boolean.")
        if "$id" in node and node is not schema:
            raise ValueError("Nested schema resource IDs are unsupported; references must stay in the original document.")
        if "$dynamicRef" in node or "$recursiveRef" in node:
            raise ValueError("Dynamic or recursive schema references are unsupported.")
        if "$ref" in node:
            reference = node["$ref"]
            if not isinstance(reference, str):
                raise ValueError("Schema reference must be a string.")
            if reference in stack:
                raise ValueError("Cyclic schema references are forbidden.")
            target = expand(resolve(reference), (*stack, reference))
            siblings = {key: value for key, value in node.items() if key not in {"$ref", "$defs", "definitions"}}
            if not siblings:
                return target
            return {"allOf": [target, _map_children(siblings, lambda child: expand(child, stack))]}
        # Check all definitions too, including unused ones, so external refs or
        # cycles cannot be concealed inside a definition skipped by expansion.
        expanded = _map_children(node, lambda child: expand(child, stack))
        expanded.pop("$defs", None)
        expanded.pop("definitions", None)
        return expanded

    return expand(schema)


def factor_schema(schema, *, min_size: int = 240):
    """Replace repeated complete schema nodes with internal $defs references."""
    if type(min_size) is not int or min_size < 1:
        raise ValueError("Schema factoring minimum size must be a positive integer.")
    original = expand_schema(schema)
    counts = Counter()

    def count(node):
        if isinstance(node, dict):
            encoded = _json(node)
            if len(encoded) >= min_size:
                counts[encoded] += 1
        _map_children(node, count)

    count(original)
    names, definitions = {}, {}

    def compact(node, *, root=False):
        if isinstance(node, dict):
            encoded = _json(node)
            if not root and counts[encoded] > 1:
                if encoded not in names:
                    name = f"s{len(names) + 1}"
                    names[encoded] = name
                    definitions[name] = None
                    definitions[name] = _map_children(node, compact)
                return {"$ref": f"#/$defs/{names[encoded]}"}
        return _map_children(node, compact)

    result = compact(original, root=True)
    if definitions:
        result["$defs"] = definitions
    return result
