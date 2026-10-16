"""Actual state changes kept separate from differences in a model forecast.

Reflection contract 3 adds only past public before/after facts. A forecast can
misdescribe a key without a key appearing in the game: mismatched_facets compares
the forecast with the after-state, while actual_changed_facets compares actual
before and after states. Neither list interprets mechanics or verifies learning.
"""

from .forecast import (
    ASSESSMENT_FIELDS, DECISION_FIELDS, FACET_ORDER, PREDICTION_SOURCE,
    assessment_schema, experiment_context, mismatched_facets, outcome_projection,
    prediction_record, prediction_schema, reflection_receipt as _forecast_receipt,
    render_prediction, trial_progress, validate_assessment, validate_intent,
    validate_trial_comparisons,
)


REFLECTION_CONTRACT_VERSION = 3


def reflection_receipt(records: dict[int, dict], step: int | None) -> dict | None:
    """Add actual pre-action facets and observed changes to a completed receipt."""
    receipt = _forecast_receipt(records, step)
    if receipt is None:
        return None
    before = outcome_projection(records[step]["before"])
    return {
        **receipt, "before_outcome": before,
        "actual_changed_facets": mismatched_facets(before, receipt["observed_outcome"]),
    }
