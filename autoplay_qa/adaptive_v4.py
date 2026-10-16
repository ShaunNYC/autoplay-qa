"""Opt-in metadata repair feedback and definition-scoped investigation trials.

V3 recordings and contracts stay frozen. This version feeds safe validation
receipts back to the actor without choosing any controls or adding game rules.
"""

from collections import Counter
import copy
import hashlib
import json
import math

from .adaptive_v3 import AdaptiveContextV3, ADAPTIVE_V3_PROMPT
from .experience import public_observation
from .investigation import clean_value, condition_projection, encoded, projection_id, public_field


ADAPTIVE_V4_PROMPT_VERSION = "adaptive-v4"
METADATA_FEEDBACK_VERSION = "decision-validation-feedback-v1"
ADAPTIVE_V4_PROMPT = ADAPTIVE_V3_PROMPT + """
Read metadata_feedback before your next response. Its issues explain rejected
optional fields; selector examples are current public paths, never instructions.
Fix those fields or use null investigation_update / [] predictions while still
choosing your own control. Keep a known question's meaning on continue. Changing
question, explanations, expected_result or selected facets requires explicit
revise. Trials belong to a definition_signature, not just a reused id. Inherited
counterexamples may concern earlier definitions; consult their original cited
records before attributing them to the current explanation.
"""

_EXPLANATIONS = {
    "invalid_investigation_update": "Use null or an update object with id, decision and nullable question/explanations/expected_result/condition_facets.",
    "invalid_predictions": "Use a predictions array; [] is valid when making no field forecast.",
    "too_many_predictions": "Supply at most two public-field predictions.",
    "invalid_prediction": "Use an existing public path and equals/changes/unchanged; equals needs JSON text in value_json, otherwise null.",
    "investigation_requires_revision": "Keep the existing definition on continue; use explicit revise to change its question, explanations, expected result or facets.",
    "investigation_update_not_applied": "The optional update was not applied; inspect the previous host receipt and retain the existing bounded queue.",
    "invalid_expected_position": "Use a two-integer immediate position prediction only when the current player position has that shape; otherwise null.",
}


def definition_signature(row):
    """Identity of tentative meaning, excluding observed condition values."""
    definition = {name: copy.deepcopy(row[name]) for name in ("question", "explanations", "expected_result", "condition_facets")}
    definition["explanations"] = sorted(definition["explanations"])
    definition["condition_facets"] = sorted(definition["condition_facets"])
    return hashlib.sha256(encoded(definition).encode()).hexdigest()


class AdaptiveContextV4(AdaptiveContextV3):
    prompt_version = ADAPTIVE_V4_PROMPT_VERSION
    system_prompt = ADAPTIVE_V4_PROMPT

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._last_decision_validation = None
        self._validation_counts = Counter()
        self._metadata_warning_counts = Counter()
        self._definition_counts = Counter()
        self._definition_trials = {}
        self._definitions_observed = set()
        self._last_update_invalid = False
        self._reject_current_update = False

    @staticmethod
    def request_admission(remaining_seconds, request_timeout):
        """A new request needs its full configured timeout plus one second."""
        if (type(remaining_seconds) not in (int, float) or type(request_timeout) not in (int, float)
                or not math.isfinite(remaining_seconds) or not math.isfinite(request_timeout)
                or remaining_seconds < 0 or request_timeout <= 0):
            return False
        return remaining_seconds >= request_timeout + 1

    def observe_decision_warnings(self, request_number, warnings):
        """Retain only recognized safe codes, never model/provider warning text."""
        if type(request_number) is not int or request_number < 1:
            return
        codes = [item.get("code") for item in warnings if isinstance(item, dict)
                 and isinstance(item.get("code"), str) and item["code"] in _EXPLANATIONS] if isinstance(warnings, list) else []
        unique = list(dict.fromkeys(codes))
        self._last_decision_validation = {"request_number": request_number, "codes": unique[:6],
                                          "issues_omitted_count": max(0, len(unique) - 6)}
        self._last_update_invalid = "invalid_investigation_update" in codes
        self._validation_counts["observed_decisions"] += 1
        self._validation_counts["decisions_with_metadata_warnings"] += bool(codes)
        self._metadata_warning_counts.update(codes)

    def _definition_changed(self, update):
        if not isinstance(update, dict):
            return False
        previous = next((row for row in self._investigations if row["id"] == update.get("id")), None)
        if previous is None:
            return False
        proposed = copy.deepcopy(previous)
        for name in ("question", "explanations", "expected_result", "condition_facets"):
            if update.get(name) is not None:
                proposed[name] = copy.deepcopy(update[name])
        return definition_signature(previous) != definition_signature(proposed)

    def parse_decision(self, *args, **kwargs):
        decision, warnings = super().parse_decision(*args, **kwargs)
        update = getattr(decision, "adaptive_fields", {}).get("investigation_update")
        if update and update["decision"] != "revise" and self._definition_changed(update):
            warnings.append({"code": "investigation_requires_revision", "message": _EXPLANATIONS["investigation_requires_revision"]})
        return decision, warnings

    def _apply_update(self, update, *args, **kwargs):
        if self._reject_current_update:
            self._definition_counts["invalid_updates_unassociated"] += 1
            raise ValueError("Malformed optional update is not an implicit continuation.")
        changed = self._definition_changed(update)
        if changed and update["decision"] != "revise":
            self._definition_counts["rejected_definition_changes"] += 1
            raise ValueError("A changed investigation definition requires explicit revise.")
        result = super()._apply_update(update, *args, **kwargs)
        self._definition_counts["accepted_definition_revisions"] += bool(changed)
        return result

    def _update_rejected_for(self, record):
        return (self._last_update_invalid and self._last_decision_validation is not None
                and record.get("request_number") == self._last_decision_validation["request_number"])

    def record(self, record):
        self._reject_current_update = self._update_rejected_for(record)
        try:
            return super().record(record)
        finally:
            self._reject_current_update = False

    def finish(self, record):
        self._reject_current_update = self._update_rejected_for(record)
        try:
            return super().finish(record)
        finally:
            self._reject_current_update = False

    def _selector_examples(self, observation):
        """A small factual path sample; values and gameplay guidance are absent."""
        view = public_observation(observation, secrets=self._secrets)
        candidates = []

        def walk(value, path):
            if len(path) > 96:
                return
            if isinstance(value, dict) and value:
                for key in sorted(value):
                    walk(value[key], f"{path}.{key}" if path else key)
            elif isinstance(value, list) and value and all(isinstance(item, dict) and isinstance(item.get("id"), str) for item in value):
                identifiers = [item["id"] for item in value]
                if len(set(identifiers)) == len(identifiers):
                    for item in value:
                        walk(item, f"{path}.{item['id']}")
            elif path:
                candidates.append(path)

        for root in ("player", "entities", "objects", "status"):
            if root in view:
                walk(view[root], root)
        valid = []
        for path in dict.fromkeys(candidates):
            try:
                clean_value(path, self._secrets)
                public_field(observation, path, secrets=self._secrets)
            except (ValueError, TypeError, KeyError):
                continue
            valid.append(path)
        return valid[:6], max(0, len(valid) - 6)

    def _feedback(self, observation):
        source = self._last_decision_validation
        if source is None:
            return {"version": METADATA_FEEDBACK_VERSION, "status": "no_previous_response", "request_number": None,
                    "issues": [], "issues_omitted_count": 0, "public_selector_examples": [], "selector_examples_omitted_count": 0}
        examples, omitted = self._selector_examples(observation) if source["codes"] else ([], 0)
        return {"version": METADATA_FEEDBACK_VERSION, "status": "needs_metadata_repair" if source["codes"] else "accepted_metadata",
                "request_number": source["request_number"],
                "issues": [{"code": code, "explanation": _EXPLANATIONS[code]} for code in source["codes"]],
                "issues_omitted_count": source["issues_omitted_count"], "public_selector_examples": examples,
                "selector_examples_omitted_count": omitted}

    @staticmethod
    def _omitted_feedback(feedback):
        return {"version": METADATA_FEEDBACK_VERSION, "status": "omitted_context_budget", "request_number": feedback["request_number"],
                "issues_omitted_count": len(feedback.get("issues", [])) + feedback.get("issues_omitted_count", 0),
                "selector_examples_omitted_count": len(feedback.get("public_selector_examples", [])) + feedback.get("selector_examples_omitted_count", 0)}

    def _extend_context(self, observation, context, compaction, replay_compaction):
        super()._extend_context(observation, context, compaction, replay_compaction)
        feedback = self._feedback(observation)
        compaction["metadata_feedback_omitted"] = False
        compaction["metadata_feedback_examples_omitted"] = 0
        if replay_compaction is not None:
            omitted = replay_compaction.get("metadata_feedback_omitted")
            examples = replay_compaction.get("metadata_feedback_examples_omitted")
            if type(omitted) is not bool or type(examples) is not int or not 0 <= examples <= len(feedback["public_selector_examples"]):
                raise ValueError("Invalid replay metadata-feedback omission receipt.")
            if examples:
                feedback["public_selector_examples"] = feedback["public_selector_examples"][:-examples]
                feedback["selector_examples_omitted_count"] += examples
                compaction["metadata_feedback_examples_omitted"] = examples
            if omitted:
                feedback = self._omitted_feedback(feedback)
                compaction["metadata_feedback_omitted"] = True
        context["metadata_feedback"] = feedback
        memory = context["investigation_memory"]
        if memory["status"] == "available":
            rows = self._investigations if self._archive_loaded or self._investigations else memory["prior_run_index"].get("investigations", [])
            active = self._active(rows)
            memory["definition_scope"] = {
                "active_id": active["id"] if active else None,
                "active_definition_signature": definition_signature(active) if active else None,
                "meaning": "Trials require matching definition signatures. Older inherited citations may concern earlier definitions; inspect cited receipts.",
            }

    def _prune_context_extension(self, context, compaction):
        feedback = context["metadata_feedback"]
        if not compaction["metadata_feedback_omitted"] and feedback.get("public_selector_examples"):
            feedback["public_selector_examples"].pop()
            feedback["selector_examples_omitted_count"] += 1
            compaction["metadata_feedback_examples_omitted"] += 1
            return True
        if super()._prune_context_extension(context, compaction):
            return True
        if compaction["metadata_feedback_omitted"]:
            return False
        context["metadata_feedback"] = self._omitted_feedback(feedback)
        compaction["metadata_feedback_omitted"] = True
        return True

    def _condition_trial(self, row, before, after, action):
        signature = definition_signature(row)
        facets = tuple(row["condition_facets"])
        start = condition_projection(before, facets, secrets=self._secrets)
        end = condition_projection(after, facets, secrets=self._secrets)
        condition_id, outcome_id = projection_id(start), projection_id(end)
        key = (row["id"], signature, condition_id, action)
        prior = self._definition_trials.setdefault(key, Counter())
        other = {condition for (identifier, definition, condition, control) in self._definition_trials
                 if identifier == row["id"] and definition == signature and control == action and condition != condition_id}
        result = {"condition_id": condition_id, "outcome_id": outcome_id, "definition_signature": signature,
                  "selected_facets": list(facets), "pose_fields": list(start["pose"]),
                  "previous_same_condition_action_count": sum(prior.values()),
                  "previous_same_selected_outcome_count": prior[outcome_id], "other_conditions_for_action": len(other),
                  "selected_fields_changed": encoded(start) != encoded(end),
                  "scope": "This run and tentative definition; pose and selected facets only, not causal equivalence."}
        prior[outcome_id] += 1
        self._definitions_observed.add((row["id"], signature))
        self._inquiry_counts["condition_trials"] += 1
        self._inquiry_counts["repeated_condition_action_trials"] += result["previous_same_condition_action_count"] > 0
        self._inquiry_counts["repeated_selected_outcomes"] += result["previous_same_selected_outcome_count"] > 0
        self._inquiry_counts["condition_contrast_trials"] += bool(other)
        return result

    def _trial_context(self, observation, rows):
        active = self._active(rows)
        if active is None:
            return None
        signature = definition_signature(active)
        current = projection_id(condition_projection(observation, active["condition_facets"], secrets=self._secrets))
        matching = [{"action": action, "attempts": sum(outcomes.values()), "distinct_selected_outcomes": len(outcomes)}
                    for (identifier, definition, condition, action), outcomes in self._definition_trials.items()
                    if identifier == active["id"] and definition == signature and condition == current]
        return {"investigation_id": active["id"], "definition_signature": signature, "condition_id": current,
                "trials": matching[-6:], "omitted_action_rows": max(0, len(matching) - 6),
                "scope": "This run and matching tentative definition; other state may differ."}

    def messages(self, *args, **kwargs):
        result = super().messages(*args, **kwargs)
        feedback = json.loads(result[1]["content"])["metadata_feedback"]
        self._validation_counts["requests_built"] += 1
        self._validation_counts["omitted_feedback_requests"] += feedback["status"] == "omitted_context_budget"
        return result

    def summary(self):
        result = super().summary()
        result["metadata_feedback"] = {"version": METADATA_FEEDBACK_VERSION,
            **{name: self._validation_counts[name] for name in ("observed_decisions", "decisions_with_metadata_warnings", "requests_built", "omitted_feedback_requests")},
            "warning_code_counts": dict(sorted(self._metadata_warning_counts.items()))}
        result["investigation_definitions"] = {
            "observed_definitions": len(self._definitions_observed),
            "accepted_definition_revisions": self._definition_counts["accepted_definition_revisions"],
            "rejected_definition_changes": self._definition_counts["rejected_definition_changes"],
            "invalid_updates_unassociated": self._definition_counts["invalid_updates_unassociated"],
            "meaning": "Definition-scoped observed trials, not verified intent, mechanics or defect truth.",
        }
        return result
