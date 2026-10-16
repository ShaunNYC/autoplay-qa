"""One model forecast, rendered explanations, and compact factual reflection.

Reflection contract 2 removes a separately model-written prediction sentence.
Its five-facet expected_outcome is still a fallible model forecast. Readable
prediction text is a deterministic rendering with explicit software provenance.
Compact assessments bind the prior step, verdict and differing facets; their
outcome prose and hypothesis claims remain unverified model text.
"""

from copy import deepcopy
import json

from .comparisons import validate_assessment as validate_control_assessment
from .game import ACTIONS
from .nebius import InvalidDecision
from .reflection import (
    experiment_context, outcome_projection, prediction_schema,
    reflection_receipt as _recorded_receipt, trial_progress, validate_intent,
    validate_trial_comparisons,
)


REFLECTION_CONTRACT_VERSION = 2
PREDICTION_SOURCE = "rendered_from_expected_outcome"
FACET_ORDER = ("position", "inventory", "key", "gate_open", "status")
ASSESSMENT_FIELDS = {"step", "outcome", "prediction_verdict", "mismatched_facets"}
DECISION_FIELDS = {
    "assessment", "hypotheses", "experiment_id", "experiment_goal", "plan",
    "action", "expected_outcome", "reason", "experiment_control", "action_role",
}


def _json(value) -> str:
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError):
        raise InvalidDecision("Forecast facts must contain finite JSON values.") from None


def _checked_outcome(value: dict) -> dict:
    """Reuse type checking only; the placeholder intent executes no control."""
    return validate_intent({
        "experiment_id": "forecast-types", "action": "wait", "experiment_control": "wait",
        "action_role": "trial", "expected_outcome": value,
    }, {})["expected_outcome"]


def render_prediction(action: str, expected_outcome: dict | None) -> str:
    """Render the sole forecast's literal facets, never a second prediction."""
    if not isinstance(action, str) or action not in {*ACTIONS, "reset", "finish"}:
        raise InvalidDecision("Prediction rendering requires a game control, reset, or finish.")
    if action == "finish":
        if expected_outcome is not None:
            raise InvalidDecision("finish has no expected_outcome to render.")
        return "Finish executes no game action; no state forecast."
    expected = _checked_outcome(expected_outcome)
    return f"Expected after {action}: " + "; ".join(f"{facet}={_json(expected[facet])}" for facet in FACET_ORDER) + "."


def prediction_record(action: str, expected_outcome: dict | None) -> dict:
    """Canonical execution fields label the renderer, not model-written prose."""
    return {"prediction": render_prediction(action, expected_outcome), "prediction_source": PREDICTION_SOURCE}


def mismatched_facets(expected_outcome: dict, observed_outcome: dict) -> list[str]:
    """Identify exact differing forecast facets without interpreting mechanics."""
    expected, observed = _checked_outcome(expected_outcome), _checked_outcome(observed_outcome)
    return [facet for facet in FACET_ORDER if _json(expected[facet]) != _json(observed[facet])]


def reflection_receipt(records: dict[int, dict], step: int | None) -> dict | None:
    """Enrich actual receipts with forecast differences and renderer provenance."""
    actual = _recorded_receipt(records, step)
    if actual is None:
        return None
    differences = mismatched_facets(actual["expected_outcome"], actual["observed_outcome"])
    return {
        **actual, "mismatched_facets": differences,
        **prediction_record(actual["control"], actual["expected_outcome"]),
    }


def validate_assessment(value, records: dict[int, dict], pending_step: int | None) -> dict | None:
    """Check the latest original forecast's verdict; prose remains unverified."""
    if pending_step is None:
        if value is not None:
            raise InvalidDecision("Assessment must be null when no completed action awaits assessment.")
        return None
    if not isinstance(value, dict) or set(value) != ASSESSMENT_FIELDS:
        raise InvalidDecision("Single-forecast assessment needs exactly step, outcome, prediction_verdict, and mismatched_facets.")
    actual = reflection_receipt(records, pending_step)
    # Reuse control checks against the full ledger to require the latest step.
    # Actual control/episode fields are canonical receipt facts, not model echoes.
    core = validate_control_assessment({
        "step": value["step"], "outcome": value["outcome"],
        **{field: actual[field] for field in ("control", "episode_before", "episode_after")},
    }, records, pending_step)
    if not isinstance(value["prediction_verdict"], str) or value["prediction_verdict"] != actual["prediction_verdict"]:
        raise InvalidDecision("assessment.prediction_verdict disagrees with the original expected_outcome and recorded outcome.")
    differences = value["mismatched_facets"]
    if not isinstance(differences, list) or _json(differences) != _json(actual["mismatched_facets"]):
        raise InvalidDecision("assessment.mismatched_facets must list the original forecast's actual differing facets in schema order.")
    return {
        "step": core["step"], "outcome": core["outcome"],
        "prediction_verdict": actual["prediction_verdict"],
        "mismatched_facets": deepcopy(actual["mismatched_facets"]),
    }


def assessment_schema(records: dict[int, dict], pending_step: int | None) -> dict:
    """Constrain prior factual reflection; never constrain a future forecast."""
    if pending_step is None:
        return {"type": "null"}
    actual = reflection_receipt(records, pending_step)
    return {
        "type": "object", "properties": {
            "step": {"type": "integer", "enum": [pending_step]},
            "outcome": {"type": "string", "minLength": 1, "maxLength": 360,
                        "description": "Model-written interpretation of the recorded feedback and the sole expected_outcome; prose remains unverified."},
            "prediction_verdict": {"type": "string", "enum": [actual["prediction_verdict"]]},
            "mismatched_facets": {
                "type": "array", "maxItems": len(FACET_ORDER),
                "items": {"type": "string", "enum": list(FACET_ORDER)},
                "enum": [actual["mismatched_facets"]],
                "description": "Exact differing facets of the ORIGINAL recorded forecast, ordered position, inventory, key, gate_open, status.",
            },
        },
        "required": ["step", "outcome", "prediction_verdict", "mismatched_facets"],
        "additionalProperties": False,
    }
