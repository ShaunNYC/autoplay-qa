"""Bounded tentative investigations and factual public-field comparisons.

These helpers neither select controls nor know a game's rules. Selectors address
public dictionaries and list members by their visible ``id``; numeric array
indexes and transient execution telemetry are deliberately not conditions.
"""

import copy
import fnmatch
import hashlib
import json
import re

from .experience import public_observation


INVESTIGATION_INDEX_VERSION = "investigation-index-v1"
INVESTIGATION_FEEDBACK_VERSION = "investigation-feedback-v1"
INVESTIGATION_CONTEXT_VERSION = "sustained-investigation-v1"
MAX_INVESTIGATION_BYTES = 2_400
MAX_INDEX_BYTES = 4_000
MAX_VALUE_BYTES = 256
MAX_CONDITIONS_BYTES = 768
ROW_FIELDS = {
    "id", "status", "question", "explanations", "expected_result",
    "condition_facets", "conditions", "evidence", "counterexamples",
    "last_decision", "origin_state_id", "last_transition",
}
UPDATE_FIELDS = {"id", "decision", "question", "explanations", "expected_result", "condition_facets"}
_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,31}$")
_RUN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")
_STATE = re.compile(r"^[0-9a-f]{64}$")
_MISSING = object()


def encoded(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)


def byte_size(value):
    return len(encoded(value).encode())


def same_value(left, right):
    # bool/int and array order are meaningful, unlike Python's loose equality.
    return encoded(left) == encoded(right)


def clean_value(value, secrets=()):
    clean = public_observation({"player": {"value": value}}, secrets=secrets)["player"]["value"]
    if not same_value(clean, value):
        raise ValueError("Investigation metadata contains private or unusable data.")
    return copy.deepcopy(clean)


def text(value, limit, secrets=()):
    if not isinstance(value, str) or not value.strip() or len(value) > limit:
        raise ValueError("Investigation text is missing or exceeds its bound.")
    return clean_value(value.strip(), secrets)


def citation(value, *, secrets=()):
    if not isinstance(value, dict) or set(value) != {"run_id", "step"}:
        raise ValueError("Invalid investigation citation.")
    identifier, step = value["run_id"], value["step"]
    if (not isinstance(identifier, str) or not _RUN.fullmatch(identifier)
            or type(step) is not int or step < 1):
        raise ValueError("Invalid investigation citation.")
    clean_value(identifier, secrets)
    return {"run_id": identifier, "step": step}


def citations(values, maximum, *, secrets=()):
    if not isinstance(values, list) or len(values) > maximum:
        raise ValueError("Investigation citations exceed their bound.")
    result = [citation(item, secrets=secrets) for item in values]
    if len({(item["run_id"], item["step"]) for item in result}) != len(result):
        raise ValueError("Duplicate investigation citation.")
    return result


def selector(path):
    if not isinstance(path, str) or not 1 <= len(path) <= 96:
        raise ValueError("Invalid public field selector.")
    parts = path.split(".")
    if (len(parts) > 10 or any(not part or part.startswith("_") or "*" in part for part in parts)
            or parts[0] in {"tick", "events", "controls", "actions"}):
        raise ValueError("Selector is not an observed state condition.")
    # Private-name filtering is identical to the public observation boundary.
    nested = "value"
    for part in reversed(parts):
        nested = {part: nested}
    clean_value(nested)
    return path


def _transient(path, observation):
    capabilities = observation.get("_capabilities", {})
    if not isinstance(capabilities, dict):
        return False
    identity = capabilities.get("state_identity", {})
    identity = identity if isinstance(identity, dict) else {}
    declared = identity.get("transient_paths", identity.get("transient_fields", capabilities.get("transient_fields", [])))
    if not isinstance(declared, (list, tuple)):
        return False
    parts = path.split(".")
    for item in declared:
        if not isinstance(item, str):
            continue
        pattern = item.split(".")
        # Selecting a parent of telemetry also includes that telemetry.
        if all(fnmatch.fnmatchcase(a, b) for a, b in zip(parts, pattern)):
            return True
    return False


def public_field(observation, path, *, secrets=(), bounded=True):
    selector(path)
    if _transient(path, observation):
        raise ValueError("Declared transient telemetry is not a test condition.")
    value = public_observation(observation, secrets=secrets)
    for part in path.split("."):
        if isinstance(value, dict) and part in value:
            value = value[part]
        elif isinstance(value, list):
            matching = [item for item in value if isinstance(item, dict) and item.get("id") == part]
            if len(matching) != 1:
                raise ValueError("Public selector is missing or ambiguous.")
            value = matching[0]
        else:
            raise ValueError("Public selector is missing or ambiguous.")
    if bounded and byte_size(value) > MAX_VALUE_BYTES:
        raise ValueError("Selected public value exceeds its exact-copy bound.")
    return copy.deepcopy(value)


def copy_conditions(observation, facets, *, secrets=()):
    if not isinstance(facets, list) or not 1 <= len(facets) <= 4 or len(set(facets)) != len(facets):
        raise ValueError("Select one to four distinct public conditions.")
    result = {selector(path): public_field(observation, path, secrets=secrets) for path in facets}
    if byte_size(result) > MAX_CONDITIONS_BYTES:
        raise ValueError("Exact selected conditions exceed their bound.")
    return result


def applicability(row, observation, state_id, *, secrets=()):
    matching, mismatching, missing = [], [], []
    for path in row["condition_facets"]:
        try:
            current = public_field(observation, path, secrets=secrets)
        except (ValueError, TypeError):
            missing.append(path)
            continue
        (matching if same_value(current, row["conditions"][path]) else mismatching).append(path)
    return {
        "exact_origin_state_match": row["origin_state_id"] == state_id,
        "matching_facets": matching, "mismatching_facets": mismatching, "missing_facets": missing,
        "all_selected_match": not (mismatching or missing) if row["condition_facets"] else None,
    }


def validate_row(row, *, secrets=()):
    if not isinstance(row, dict) or set(row) != ROW_FIELDS:
        raise ValueError("Invalid investigation row fields.")
    clean_value(row, secrets)
    if not isinstance(row["id"], str) or not _ID.fullmatch(row["id"]):
        raise ValueError("Invalid investigation identifier.")
    if row["status"] not in {"active", "pending", "closed"} or row["last_decision"] not in {"continue", "revise", "close", "defer"}:
        raise ValueError("Invalid tentative investigation lifecycle.")
    if row["status"] != {"continue": "active", "revise": "active", "close": "closed", "defer": "pending"}[row["last_decision"]]:
        raise ValueError("Investigation status and decision differ.")
    text(row["question"], 200, secrets)
    text(row["expected_result"], 160, secrets)
    if not isinstance(row["explanations"], list) or not 1 <= len(row["explanations"]) <= 2:
        raise ValueError("Investigation needs one or two tentative explanations.")
    for item in row["explanations"]:
        text(item, 160, secrets)
    facets, conditions = row["condition_facets"], row["conditions"]
    if (not isinstance(facets, list) or not 1 <= len(facets) <= 4
            or any(not isinstance(path, str) for path in facets) or len(set(facets)) != len(facets)
            or not isinstance(conditions, dict) or set(conditions) != set(facets)):
        raise ValueError("Investigation selected conditions differ.")
    for path in facets:
        selector(path)
        if byte_size(conditions[path]) > MAX_VALUE_BYTES:
            raise ValueError("Investigation condition value exceeds its bound.")
    if byte_size(conditions) > MAX_CONDITIONS_BYTES:
        raise ValueError("Investigation conditions exceed their bound.")
    citations(row["evidence"], 6, secrets=secrets)
    citations(row["counterexamples"], 4, secrets=secrets)
    citation(row["last_transition"], secrets=secrets)
    if not isinstance(row["origin_state_id"], str) or not _STATE.fullmatch(row["origin_state_id"]):
        raise ValueError("Invalid investigation origin state.")
    if byte_size(row) > MAX_INVESTIGATION_BYTES:
        raise ValueError("Investigation row exceeds its bound.")
    return copy.deepcopy(row)


def validate_rows(rows, *, secrets=()):
    if not isinstance(rows, list) or len(rows) > 4:
        raise ValueError("Investigation working set exceeds its bound.")
    rows = [validate_row(row, secrets=secrets) for row in rows]
    if (len({row["id"] for row in rows}) != len(rows)
            or sum(row["status"] == "active" for row in rows) > 1
            or sum(row["status"] == "pending" for row in rows) > 3
            or sum(row["status"] == "closed" for row in rows) > 1):
        raise ValueError("Invalid bounded investigation queue.")
    return rows


def parse_update(value, observation, *, secrets=()):
    if value is None:
        return None
    if not isinstance(value, dict) or set(value) - UPDATE_FIELDS:
        raise ValueError("Invalid investigation update object.")
    clean_value(value, secrets)
    identifier = value.get("id")
    if not isinstance(identifier, str) or not _ID.fullmatch(identifier):
        raise ValueError("Invalid investigation update identifier.")
    if value.get("decision") not in {"continue", "revise", "close", "defer"}:
        raise ValueError("Invalid investigation decision.")
    result = {name: None for name in ("question", "explanations", "expected_result", "condition_facets")}
    result.update(id=identifier, decision=value["decision"])
    for name, limit in (("question", 200), ("expected_result", 160)):
        if value.get(name) is not None:
            result[name] = text(value[name], limit, secrets)
    if value.get("explanations") is not None:
        items = value["explanations"]
        if not isinstance(items, list) or not 1 <= len(items) <= 2:
            raise ValueError("Invalid competing explanations.")
        result["explanations"] = [text(item, 160, secrets) for item in items]
    if value.get("condition_facets") is not None:
        copy_conditions(observation, value["condition_facets"], secrets=secrets)
        result["condition_facets"] = copy.deepcopy(value["condition_facets"])
    return result


def parse_prediction(value, observation, *, secrets=()):
    if not isinstance(value, dict) or set(value) - {"path", "relation", "value_json"}:
        raise ValueError("Invalid public-field prediction.")
    clean_value(value, secrets)
    path, relation = value.get("path"), value.get("relation")
    public_field(observation, path, secrets=secrets)
    if relation not in {"equals", "changes", "unchanged"}:
        raise ValueError("Invalid prediction relation.")
    expected = value.get("value_json")
    if relation == "equals":
        if not isinstance(expected, str) or len(expected) > MAX_VALUE_BYTES:
            raise ValueError("Equality prediction needs bounded JSON text.")
        expected = json.loads(expected, parse_constant=lambda _: (_ for _ in ()).throw(ValueError("Nonfinite JSON.")))
        clean_value(expected, secrets)
        if byte_size(expected) > MAX_VALUE_BYTES:
            raise ValueError("Prediction value exceeds its bound.")
        expected = encoded(expected)
    elif expected is not None:
        raise ValueError("Change predictions use null value_json.")
    return {"path": path, "relation": relation, "value_json": expected}


def compare_prediction(prediction, before, after, *, secrets=()):
    result = {"path": prediction["path"], "relation": prediction["relation"]}
    result["expected"] = json.loads(prediction["value_json"]) if prediction["relation"] == "equals" else None
    result["before"] = public_field(before, prediction["path"], secrets=secrets)
    try:
        result["after"] = public_field(after, prediction["path"], secrets=secrets)
    except (ValueError, TypeError):
        result.update(after=None, status="not_comparable")
        return result
    changed = not same_value(result["before"], result["after"])
    matched = (same_value(result["expected"], result["after"]) if prediction["relation"] == "equals"
               else changed if prediction["relation"] == "changes" else not changed)
    result["status"] = "matched" if matched else "mismatch"
    return result


def condition_projection(observation, facets, *, secrets=()):
    """Pose plus selected public values, with missing fields explicitly marked."""
    view = public_observation(observation, secrets=secrets)
    player = view.get("player", {})
    pose = {name: copy.deepcopy(player[name]) for name in ("position", "facing", "heading", "rotation")
            if isinstance(player, dict) and name in player}
    selected = {}
    for path in facets:
        try:
            selected[path] = {"value": public_field(observation, path, secrets=secrets)}
        except (ValueError, TypeError):
            selected[path] = {"missing": True}
    return {"pose": pose, "selected": selected}


def projection_id(projection):
    return hashlib.sha256(encoded(projection).encode()).hexdigest()
