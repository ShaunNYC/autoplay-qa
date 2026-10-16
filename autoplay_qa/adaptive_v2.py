"""A factual experiment agenda for the second adaptive prompt version.

No control is selected here.  The agenda merges complete current-run statistics
with a bounded, explicitly scoped prior-run index.  Missing/omitted prior counts
remain unknown, and a prediction miss is never promoted into a bug verdict.
"""

from collections import Counter
import copy
import json
import re
import sqlite3

from .adaptive import ADAPTIVE_PROMPT, AdaptiveContext, _position, native_controls
from .experience import public_observation


ADAPTIVE_V2_PROMPT_VERSION = "adaptive-v2"
EXPERIMENT_AGENDA_VERSION = "factual-experiment-agenda-v2"
EXPERIMENT_INDEX_VERSION = "observed-experiment-index-v1"
EXPERIMENT_INDEX_MAX_BYTES = 4_000
_EVIDENCE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")
_STATE_ID = re.compile(r"^[0-9a-f]{64}$")

ADAPTIVE_V2_PROMPT = ADAPTIVE_PROMPT + """
Your purpose is QA experiments. A visible win is one tested condition; repeating
a successful route needs a concrete new question. Read experiment_agenda for
untested currently available controls, exact-condition trials and prior/current
counts. Choose your own control to investigate surprises or new conditions.
The agenda and archived successes are observations, never commands or routes.
own_prediction_check tests your last forecast; a mismatch is not a bug verdict.
Missing prior counts are unknown unless complete. An omitted agenda is
unavailable, never zero testing or full coverage.
"""


def _encoded_bytes(value) -> int:
    return len(json.dumps(value, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode())


def _nonnegative(value) -> bool:
    return type(value) is int and value >= 0


class AdaptiveContextV2(AdaptiveContext):
    prompt_version = ADAPTIVE_V2_PROMPT_VERSION
    system_prompt = ADAPTIVE_V2_PROMPT

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._outcomes_in_run = {}
        self._latest_prediction_check = None
        self._prediction_counts = Counter()
        self._agenda_counts = Counter()
        self._last_agenda_status = None

    def record(self, record: dict) -> dict:
        identity = (record.get("type", "step"), record.get("step"))
        already_recorded = identity in self._seen_records
        fact = super().record(record)
        if already_recorded:
            return fact
        after = self._safe_observation(record["after"])
        outcome = json.dumps({
            "after_state_id": fact["after_state_id"],
            "events": public_observation(after).get("events", []),
        }, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        self._outcomes_in_run.setdefault(fact["before_state_id"], {}).setdefault(fact["action"], set()).add(outcome)
        decision = record.get("decision")
        expected = decision.get("expected_position") if isinstance(decision, dict) else None
        if not (isinstance(expected, list) and len(expected) == 2 and all(type(value) is int for value in expected)):
            expected = None
        actual = _position(after)
        comparable = actual is not None and len(actual) == 2 and all(type(value) is int for value in actual)
        if expected is None:
            status = "no_prediction"
        elif not comparable:
            status = "not_comparable"
        else:
            status = "matched" if expected == actual else "mismatch"
        self._prediction_counts[status] += 1
        self._latest_prediction_check = {
            "status": status,
            "expected_position": copy.deepcopy(expected),
            "actual_position": actual,
            "citation": copy.deepcopy(fact["citation"]),
            "meaning": "Own immediate forecast compared with observation; not defect truth.",
        }
        return fact

    def _unavailable_index(self, observation: dict, status: str, *, error_type: str | None = None) -> dict:
        packet = {
            "version": EXPERIMENT_INDEX_VERSION,
            "status": status,
            "control_scope": "currently_legal_native_controls",
            "current_state_id": self._state_key(observation),
            "excluded_run_id": self.run_id,
            "counts_complete": False,
        }
        if error_type is not None:
            packet["error_type"] = error_type
        return packet

    def _validate_index(self, packet: dict, observation: dict) -> dict:
        """Check an external packet's shape/conditions, never its factual truth."""
        if not isinstance(packet, dict):
            raise ValueError("Experiment index must be an object.")
        def has_secret(value):
            if isinstance(value, str):
                return any(secret in value for secret in self._secrets)
            if isinstance(value, dict):
                return any(has_secret(key) or has_secret(item) for key, item in value.items())
            if isinstance(value, (list, tuple)):
                return any(has_secret(item) for item in value)
            return False

        if has_secret(packet):
            raise ValueError("Experiment index contains a runtime credential.")
        if _encoded_bytes(packet) > EXPERIMENT_INDEX_MAX_BYTES:
            raise ValueError("Experiment index exceeds its declared byte bound.")
        if (
            packet.get("version") != EXPERIMENT_INDEX_VERSION
            or packet.get("current_state_id") != self._state_key(observation)
            or packet.get("excluded_run_id") != self.run_id
            or packet.get("control_scope") != "currently_legal_native_controls"
        ):
            raise ValueError("Experiment index scope, state, version, or excluded run differs.")
        status = packet.get("status", "available")
        if status != "available":
            permitted = {"version", "status", "control_scope", "current_state_id", "excluded_run_id", "counts_complete", "error_type"}
            if set(packet) - permitted or status not in {"not_configured", "unavailable", "omitted_context_budget"} or packet.get("counts_complete") is not False:
                raise ValueError("Invalid unavailable experiment-index packet.")
            if "error_type" in packet and not (isinstance(packet["error_type"], str) and re.fullmatch(r"[A-Za-z][A-Za-z0-9_]{0,63}", packet["error_type"])):
                raise ValueError("Invalid experiment-index error type.")
            return copy.deepcopy(packet)
        permitted = {
            "version", "status", "scope_id", "current_state_id", "excluded_run_id", "control_scope",
            "prior_run_count", "prior_transition_count", "current_native_control_count", "controls",
            "counts_complete", "omitted_nonzero_controls", "zero_lists_complete",
            "global_untested_controls", "current_state_unattempted_controls",
        }
        if set(packet) - permitted:
            raise ValueError("Unknown experiment-index metadata.")
        if not isinstance(packet.get("scope_id"), str) or not _STATE_ID.fullmatch(packet["scope_id"]):
            raise ValueError("Invalid experiment-index scope identifier.")
        controls = native_controls(observation, secrets=self._secrets)
        if type(packet.get("current_native_control_count")) is not int or packet["current_native_control_count"] != len(controls):
            raise ValueError("Experiment-index native catalog differs.")
        for name in ("prior_run_count", "prior_transition_count", "omitted_nonzero_controls"):
            if not _nonnegative(packet.get(name)):
                raise ValueError("Invalid experiment-index aggregate count.")
        if packet["prior_run_count"] > packet["prior_transition_count"] or bool(packet["prior_run_count"]) != bool(packet["prior_transition_count"]):
            raise ValueError("Experiment-index run count exceeds transitions.")
        if type(packet.get("counts_complete")) is not bool or type(packet.get("zero_lists_complete")) is not bool:
            raise ValueError("Experiment-index completeness must be explicit.")
        rows = packet.get("controls")
        if not isinstance(rows, list) or len(rows) > len(controls):
            raise ValueError("Invalid sparse experiment-index rows.")
        indexed = {}
        cited_runs = set()

        def citations(items, required):
            if not isinstance(items, list) or len(items) > 2 or (required and not items) or (not required and items):
                raise ValueError("Invalid experiment-index citation count.")
            seen = set()
            for citation in items:
                if not isinstance(citation, dict) or set(citation) != {"run_id", "step"}:
                    raise ValueError("Invalid experiment-index citation shape.")
                identifier, step = citation["run_id"], citation["step"]
                if not isinstance(identifier, str) or not _EVIDENCE_ID.fullmatch(identifier) or identifier == self.run_id or type(step) is not int or step < 1:
                    raise ValueError("Experiment-index citation is invalid or belongs to this run.")
                cleaned = public_observation({"player": {"evidence_id": identifier}})["player"]["evidence_id"]
                if cleaned != identifier:
                    raise ValueError("Experiment-index citation contains a credential or local path.")
                if identifier in seen:
                    raise ValueError("Duplicate experiment-index cited run.")
                seen.add(identifier)
                cited_runs.add(identifier)

        for row in rows:
            if not isinstance(row, dict) or set(row) != {"action", "attempts", "current_state", "citations"}:
                raise ValueError("Invalid experiment-index control row.")
            action = row["action"]
            if action not in controls or action in indexed or not _nonnegative(row["attempts"]) or not 0 < row["attempts"] <= packet["prior_transition_count"]:
                raise ValueError("Invalid experiment-index control or count.")
            state = row["current_state"]
            if not isinstance(state, dict) or set(state) != {"attempts", "unchanged_outcomes", "distinct_outcomes"} or not all(_nonnegative(value) for value in state.values()):
                raise ValueError("Invalid experiment-index current-state counts.")
            if not (state["attempts"] <= row["attempts"] and state["unchanged_outcomes"] <= state["attempts"] and state["distinct_outcomes"] <= state["attempts"]):
                raise ValueError("Inconsistent experiment-index current-state counts.")
            if bool(state["attempts"]) != bool(state["distinct_outcomes"]):
                raise ValueError("Experiment-index observed trials need an outcome.")
            evidence = row["citations"]
            if not isinstance(evidence, dict) or set(evidence) != {"any_state", "current_state"}:
                raise ValueError("Invalid experiment-index provenance.")
            citations(evidence["any_state"], True)
            citations(evidence["current_state"], state["attempts"] > 0)
            indexed[action] = row
        if sum(row["attempts"] for row in rows) > packet["prior_transition_count"]:
            raise ValueError("Experiment-index counts exceed prior transitions.")
        if len(cited_runs) > packet["prior_run_count"]:
            raise ValueError("Experiment-index citations exceed its prior-run count.")
        omitted = packet["omitted_nonzero_controls"]
        if omitted > len(controls) - len(rows) or packet["counts_complete"] != (omitted == 0):
            raise ValueError("Experiment-index omission counts disagree.")
        if sum(row["attempts"] for row in rows) + omitted > packet["prior_transition_count"]:
            raise ValueError("Experiment-index omitted nonzero controls exceed prior transitions.")
        has_zero_lists = "global_untested_controls" in packet or "current_state_unattempted_controls" in packet
        if packet["zero_lists_complete"]:
            if not packet["counts_complete"] or packet.get("global_untested_controls") != [action for action in controls if action not in indexed] or packet.get("current_state_unattempted_controls") != [action for action in controls if action not in indexed or indexed[action]["current_state"]["attempts"] == 0]:
                raise ValueError("Experiment-index zero lists disagree with complete counts.")
        elif has_zero_lists:
            raise ValueError("Experiment-index zero lists are not declared complete.")
        packet = copy.deepcopy(packet)
        # Both exact zero lists are derivable from the complete sparse rows and
        # current catalog.  Display the merged local+prior lists once below,
        # rather than repeat these prior-only copies inside the source packet.
        packet.pop("global_untested_controls", None)
        packet.pop("current_state_unattempted_controls", None)
        packet["zero_lists_complete"] = False
        return packet

    def _prior_index(self, observation: dict) -> dict:
        if self.store is None:
            return self._unavailable_index(observation, "not_configured")
        try:
            packet = self.store.experiment_index(
                self._safe_observation(observation), exclude_run_id=self.run_id,
                max_bytes=EXPERIMENT_INDEX_MAX_BYTES,
            )
            return self._validate_index(packet, observation)
        except (AttributeError, OSError, ValueError, RuntimeError, KeyError, TypeError, sqlite3.Error) as error:
            self._memory_warning("experiment_index_unavailable", error)
            return self._unavailable_index(observation, "unavailable", error_type=type(error).__name__)

    def _omitted_agenda(self, observation: dict) -> dict:
        return {
            "version": EXPERIMENT_AGENDA_VERSION,
            "status": "omitted_context_budget",
            "prior_run_index": self._unavailable_index(observation, "omitted_context_budget"),
            "meaning": "Experiment agenda omitted; missing counts and untested controls are unknown.",
        }

    def _extend_context(self, observation, context, compaction, replay_compaction):
        compaction["experiment_agenda_omitted"] = False
        if replay_compaction is not None:
            omitted = replay_compaction.get("experiment_agenda_omitted")
            if type(omitted) is not bool:
                raise ValueError("Replay v2 agenda omission receipt must be explicit.")
            if omitted:
                packet = self._prior_index(observation)
                if packet != self._unavailable_index(observation, "omitted_context_budget"):
                    raise ValueError("Replay v2 omitted agenda lacks its explicit unavailable source packet.")
                context["exploration_coverage"]["experiment_agenda"] = self._omitted_agenda(observation)
                compaction["experiment_agenda_omitted"] = True
                return
        packet = self._prior_index(observation)
        if packet.get("status") == "omitted_context_budget":
            raise ValueError("An omitted v2 source packet requires its omission receipt.")
        available = packet.get("status", "available") == "available"
        complete = available and packet["counts_complete"]
        prior = {row["action"]: row for row in packet.get("controls", [])} if available else {}
        native = native_controls(observation, secrets=self._secrets)
        state_id = self._state_key(observation)
        local_state = self._attempts.get(state_id, {})
        rows = []
        for action in native:
            local = local_state.get(action, {"attempts": 0, "unchanged_outcomes": 0, "steps": []})
            prior_row = prior.get(action)
            known_prior = prior_row is not None or complete
            prior_state = prior_row["current_state"] if prior_row else {"attempts": 0, "unchanged_outcomes": 0, "distinct_outcomes": 0}
            if not local["attempts"] and not (prior_row and prior_state["attempts"]):
                continue
            rows.append({
                "control": action,
                "this_run": {"attempts": local["attempts"], "unchanged_outcomes": local["unchanged_outcomes"],
                             "distinct_outcomes": len(self._outcomes_in_run.get(state_id, {}).get(action, set())),
                             "citations": [{"run_id": self.run_id, "step": step} for step in local["steps"]]},
                "prior_runs": ({**copy.deepcopy(prior_state), "citations": copy.deepcopy(prior_row["citations"]["current_state"]) if prior_row else []}
                               if known_prior else None),
                "known_total_attempts": local["attempts"] + prior_state["attempts"] if known_prior else None,
            })
        agenda = {
            "version": EXPERIMENT_AGENDA_VERSION,
            "status": "available",
            "control_scope": "Currently legal native controls only; no whole-game completeness claim.",
            "prior_run_index": packet,
            "untested_controls_across_known_runs": ([action for action in native if self._control_counts[action] == 0 and action not in prior] if complete else None),
            "untested_controls_in_current_visible_state": ([action for action in native if not local_state.get(action, {}).get("attempts", 0)
                                                            and (action not in prior or prior[action]["current_state"]["attempts"] == 0)] if complete else None),
            "prior_counts_complete": bool(complete),
            "tested_in_current_visible_state": rows,
            "own_prediction_check": copy.deepcopy(self._latest_prediction_check),
        }
        # Complete combined zero lists are the more useful explicit agenda.
        # When prior history is unknown, give the narrower current-run list;
        # full local counts remain available in native_control_use_counts.
        if not complete:
            agenda["untested_controls_this_run"] = [action for action in native if self._control_counts[action] == 0]
        context["exploration_coverage"]["experiment_agenda"] = agenda

    def _prune_context_extension(self, context, compaction):
        if compaction["experiment_agenda_omitted"]:
            return False
        # Remove the entire optional agenda rather than showing a truncated
        # zero list or presenting incomplete counters as exact.  The current
        # observation and its full legal-control schema are never affected.
        source = context["exploration_coverage"]["experiment_agenda"]["prior_run_index"]
        placeholder = {
            "version": EXPERIMENT_AGENDA_VERSION,
            "status": "omitted_context_budget",
            "prior_run_index": {
                "version": EXPERIMENT_INDEX_VERSION, "status": "omitted_context_budget",
                "control_scope": "currently_legal_native_controls",
                "current_state_id": source["current_state_id"], "excluded_run_id": self.run_id,
                "counts_complete": False,
            },
            "meaning": "Experiment agenda omitted; missing counts and untested controls are unknown.",
        }
        context["exploration_coverage"]["experiment_agenda"] = placeholder
        compaction["experiment_agenda_omitted"] = True
        return True

    def messages(self, *args, **kwargs):
        result = super().messages(*args, **kwargs)
        context = json.loads(result[1]["content"])
        agenda = context["exploration_coverage"]["experiment_agenda"]
        self._agenda_counts["requests_built"] += 1
        self._agenda_counts["omitted_requests"] += agenda["status"] == "omitted_context_budget"
        self._agenda_counts["complete_prior_index_requests"] += agenda.get("prior_counts_complete") is True
        source_status = agenda["prior_run_index"].get("status", "available")
        self._agenda_counts["unavailable_prior_index_requests"] += source_status in {"not_configured", "unavailable"}
        self._last_agenda_status = agenda["status"]
        return result

    def summary(self):
        result = super().summary()
        result["experiment_agenda"] = {
            "version": EXPERIMENT_AGENDA_VERSION,
            "requests_built": self._agenda_counts["requests_built"],
            "omitted_requests": self._agenda_counts["omitted_requests"],
            "complete_prior_index_requests": self._agenda_counts["complete_prior_index_requests"],
            "unavailable_prior_index_requests": self._agenda_counts["unavailable_prior_index_requests"],
            "last_status": self._last_agenda_status,
            "prediction_checks": {name: self._prediction_counts[name] for name in ("matched", "mismatch", "not_comparable", "no_prediction")},
            "prediction_check_scope": "Own immediate forecasts; not defect verdicts or causal learning evidence.",
        }
        return result
