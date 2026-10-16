"""Recompute adaptive contexts from recorded execution, without a live model.

Prior-run retrieval packets are retained inputs, not independently proven
history. Replay validates this run's facts and decisions and explicitly reports
that limitation; it never trusts a historical citation as a verified bug.
"""

from copy import deepcopy
import json
import math

from .adaptive import adaptive_input_bytes
from .adaptive_protocol import context_class
from .session import _LeanReplay, _same_json


class AdaptiveReplay(_LeanReplay):
    def __init__(self, header):
        super().__init__(header)
        config = header.get("adaptive_config", {})
        self.context = context_class(header["prompt_version"])(
            header["initial_observation"], run_id=config.get("run_id", "session"),
            max_prompt_bytes=config.get("max_prompt_bytes", 24_000))
        self.header = header
        self.prior_memory_packets = 0
        self._unused_context_pending = False
        self._admission_stopped = False

    def request(self, context, observation, record):
        from .lean import lean_schema

        allowed = self.context.allowed(observation)
        if self._admission_stopped:
            raise ValueError("Replay has inference after its terminal admission stop.")
        schema = self.context.schema(allowed)
        if not _same_json(record.get("response_schema"), schema):
            raise ValueError("Replay adaptive schema differs from the current native controls.")
        packet = context.get("retrieved_experiments")
        if not isinstance(packet, dict) or not isinstance(packet.get("facts"), list):
            raise ValueError("Replay adaptive retrieval packet is missing.")

        class RecordedPacket:
            def retrieve(self, *_args, **_kwargs):
                return deepcopy(packet)

            def experiment_index(self, *_args, **_kwargs):
                # Prior-run aggregates are recorded external inputs. V2 still
                # recomputes their combination with this run's actual controls;
                # replay alone cannot prove the external archive's provenance.
                try:
                    index = context["exploration_coverage"]["experiment_agenda"]["prior_run_index"]
                except (KeyError, TypeError):
                    raise ValueError("Replay adaptive experiment index is missing.") from None
                return deepcopy(index)

            def investigation_index(self, *_args, **_kwargs):
                try:
                    index = context["investigation_memory"]["prior_run_index"]
                except (KeyError, TypeError):
                    raise ValueError("Replay investigation archive packet is missing.") from None
                return deepcopy(index)

        # The retained retrieval packet is an input to this run. The context
        # builder still recomputes its working set, omissions and local facts.
        self.context.store = RecordedPacket()
        try:
            expected = self.context.messages(
                observation, context.get("episode"), self.notes, self.records,
                context.get("remaining_budget", {}), focus=self.header.get("focus", "qa"),
                feedback=context.get("validation_feedback", ""),
                replay_compaction=context.get("context_compaction"),
            )
        finally:
            self.context.store = None
        if not _same_json(record.get("messages"), expected):
            raise ValueError("Replay adaptive context differs from the current observation, local facts, or notes.")
        if adaptive_input_bytes(record["messages"], schema) > self.context.max_prompt_bytes:
            raise ValueError("Replay adaptive input exceeds its recorded budget.")
        self.prior_memory_packets += bool(packet["facts"])
        self.phase2 = True
        if self.context.prompt_version == "focused-v1":
            # Research-only focused responses carry no note array. Execution
            # keeps the established lean host facts and empty canonical notes.
            from .lean import LEAN_EVIDENCE_LIMIT
            self.parse_limits = (LEAN_EVIDENCE_LIMIT, schema["properties"]["reason"]["maxLength"])
        else:
            self.parse_limits = (schema["properties"]["notes"]["items"]["properties"]["evidence_steps"]["maxItems"],
                                 schema["properties"]["reason"]["maxLength"])
        self.allowed = allowed
        self.completion_record = None
        self.accepted_decision = None
        self.memory_recorded = False
        self.request_warnings = []
        self._unused_context_pending = record.get("type") == "unused_context"

    def memory(self, record, episode):
        super().memory(record, episode)
        observer = getattr(self.context, "observe_decision_warnings", None)
        if observer is not None:
            observer(record["request_number"], self.request_warnings)

    def admission_stop(self, record):
        admission = getattr(self.context, "request_admission", None)
        remaining, timeout = record.get("remaining_seconds"), record.get("request_timeout")
        configured = self.header.get("adaptive_config", {}).get("request_timeout", 45)
        if (admission is None or not self._unused_context_pending or self._admission_stopped
                or record.get("dispatched") is not False
                or any(type(value) not in (int, float) or not math.isfinite(value) for value in (remaining, timeout))
                or type(configured) not in (int, float) or not math.isfinite(configured)
                or timeout != configured or timeout <= 0 or remaining <= 0
                or record.get("required_seconds") != timeout + 1
                or admission(remaining, timeout)):
            raise ValueError("Replay request-admission stop receipt is invalid.")
        self._admission_stopped = True

    def executed(self, record):
        super().executed(record)
        if "experiment_fact_error" in record:
            self.fact_failures = getattr(self, "fact_failures", 0) + 1
            return
        recomputed = deepcopy(record)
        expected = self.context.record(recomputed)
        if not _same_json(record.get("experiment_fact"), expected):
            raise ValueError("Replay adaptive experiment fact differs from recorded execution.")
        if self.context.prompt_version in {"adaptive-v3", "adaptive-v4"} and not _same_json(
                record.get("investigation_fact"), recomputed.get("investigation_fact")):
            raise ValueError("Replay investigation facts differ from the executed observation and prediction.")
        if self.context.prompt_version == "focused-v1" and not _same_json(
                record.get("focused_fact"), recomputed.get("focused_fact")):
            raise ValueError("Replay focused facts differ from the executed observation and prediction.")

    def control(self, record):
        super().control(record)
        if record.get("type") == "finish" and hasattr(self.context, "finish"):
            expected = self.context.finish(deepcopy(record))
            fact_key = "focused_fact" if self.context.prompt_version == "focused-v1" else "investigation_fact"
            if not _same_json(record.get(fact_key), expected):
                raise ValueError("Replay final investigation update differs from its accepted answer.")

    def _parse_decision(self, *args, **kwargs):
        parser = getattr(self.context, "parse_decision", None)
        if parser is not None:
            return parser(*args, **kwargs)
        return super()._parse_decision(*args, **kwargs)

    def budget_failure(self, record, observation):
        from .adaptive import AdaptivePromptBudgetExceeded

        if (record.get("max_bytes") != self.context.max_prompt_bytes
                or type(record.get("required_bytes")) is not int
                or record["required_bytes"] <= self.context.max_prompt_bytes):
            raise ValueError("Replay adaptive budget failure receipt is invalid.")
        try:
            self.context.messages(observation, record.get("episode"), self.notes, self.records,
                                  record.get("remaining_budget", {}),
                                  focus=self.header.get("focus", "qa"), feedback=record.get("feedback", ""))
        except AdaptivePromptBudgetExceeded:
            return
        raise ValueError("Replay cannot reproduce the recorded essential-context budget failure.")

    def summary(self, record):
        super().summary(record)
        if record.get("request_admission_stopped", False) is not self._admission_stopped:
            raise ValueError("Replay request-admission summary differs from its receipt.")
        if self._admission_stopped and record.get("status") != "time_limit":
            raise ValueError("A request-admission stop requires a time-limit summary.")
        expected = self.context.summary()
        actual = record.get("adaptive")
        if not isinstance(actual, dict):
            raise ValueError("Replay adaptive summary is missing.")
        if getattr(self, "fact_failures", 0):
            if record.get("status") != "experiment_recording_error":
                raise ValueError("Missing adaptive facts require an explicit terminal recording failure.")
            return
        for key, value in expected.items():
            if key != "memory" and not _same_json(actual.get(key), value):
                raise ValueError(f"Replay adaptive metric {key} differs from recorded execution.")
        if not _same_json(record.get("coverage"), actual):
            raise ValueError("Replay adaptive coverage differs from its experiment metrics.")


def adaptive_review(records, replayed, *, include_evidence=False):
    from .review import _lean_learning, _recorded_hypothesis_evidence

    header = records[0]
    summary = records[-1] if records[-1].get("type") == "summary" else {}
    learning, retention, facts = _lean_learning(records)
    completions = [row for row in records if row.get("type") == "completion"]
    latencies = sorted(row["latency_ms"] for row in completions
                       if type(row.get("latency_ms")) in (int, float) and row["latency_ms"] >= 0)
    from statistics import median

    metrics = summary.get("adaptive", {})
    result = {
        "schema_version": 2, "mode": header["mode"], "task": "explore",
        "focus": header.get("focus"), "prompt_version": header["prompt_version"],
        "recording_complete": replayed["recording_complete"],
        "status": summary.get("status"), "partial": summary.get("partial", True),
        "execution_replayed": True,
        "requests": {"attempted": sum(row.get("type") == "request" for row in records),
                     "completed": len(completions),
                     "latency_ms": {"count": len(latencies), "median": median(latencies) if latencies else None,
                                    "max": max(latencies) if latencies else None}},
        "coverage": {**deepcopy(metrics), "game_actions": summary.get("game_actions"),
                     "resets": summary.get("resets"), "experiment_count": metrics.get("distinct_state_action_pairs", 0),
                     "movement": bool(metrics.get("position_changes")), "interaction": None},
        "learning": learning, "reset_learning": retention, "lean_facts": facts,
        "adaptive": deepcopy(metrics), "review_needed": True,
        "goal_checks": {"native_experiments_observed": bool(metrics.get("distinct_state_action_pairs")),
                        "learning_improvement_requires_controlled_comparison": True},
        "learning_claims_verified": False, "bugs_verified": False,
        "review_note": "Local execution and experiment facts were replayed. Prior-run retrieval packets are retained inputs; their source execution and memory causation need separate validation.",
    }
    if include_evidence:
        result["hypothesis_evidence"] = _recorded_hypothesis_evidence(records, replayed["recording_complete"])
    return result
