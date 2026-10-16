"""Inspectable JSONL evidence and deterministic replay for this instrumented game."""

from datetime import datetime, timezone
from dataclasses import asdict
import json
from pathlib import Path
import time
import uuid

from .game import KeyGateGame


def new_log_path(directory: Path = Path("runs")) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return directory / f"{stamp}-{uuid.uuid4().hex[:8]}.jsonl"


def redact_secrets(value, secrets: tuple[str, ...]):
    """Redact before JSON serialization, which may escape secret characters."""
    if isinstance(value, str):
        for secret in secrets:
            if secret:
                # Messages and legacy answers can themselves be JSON strings.
                for representation in (secret, json.dumps(secret)[1:-1]):
                    value = value.replace(representation, "[REDACTED]")
        return value
    if isinstance(value, dict):
        return {key: redact_secrets(item, secrets) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [redact_secrets(item, secrets) for item in value]
    return value


class SessionLog:
    def __init__(self, path: Path, secrets: tuple[str, ...] = ()):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.secrets = tuple(secret for secret in secrets if secret)
        self._file = path.open("x", encoding="utf-8")
        self.started = time.monotonic()

    def write(self, kind: str, **fields):
        record = {"type": kind, "timestamp": datetime.now(timezone.utc).isoformat(), "elapsed_ms": round((time.monotonic() - self.started) * 1000), **fields}
        # Defense against an upstream answer ever echoing the bearer key.
        self._file.write(json.dumps(redact_secrets(record, self.secrets), separators=(",", ":")) + "\n")
        self._file.flush()

    def close(self):
        self._file.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


def _same_json(value, expected) -> bool:
    """Unlike Python equality, preserve JSON distinctions such as true versus 1."""
    return json.dumps(value, sort_keys=True) == json.dumps(expected, sort_keys=True)


def _recording_game(header: dict) -> KeyGateGame:
    """Recreate private injected state from session metadata, never a request."""
    evaluation = header.get("evaluation")
    if evaluation is None:
        defect = None
    elif not isinstance(evaluation, dict):
        raise ValueError("Replay session evaluation metadata is malformed.")
    else:
        injected = evaluation.get("injected_defect")
        if injected is None:
            defect = None
        elif isinstance(injected, dict) and isinstance(injected.get("name"), str):
            defect = injected["name"]
        else:
            raise ValueError("Replay injected-defect metadata is malformed.")
    return KeyGateGame(header["seed"], defect)


class _ComparisonReplay:
    """Validate opt-in typed bookkeeping against replayed execution, not claims."""

    def __init__(self, header: dict):
        from . import comparisons

        self.contract = comparisons
        self.require_comparison = header.get("focus") in {"reset-retention", "interaction-retention"}
        self.reflection = None
        self.reflection_version = header.get("reflection_contract")
        if "reflection_contract" in header:
            if self.reflection_version == 4:
                from . import measurements as reflection
            elif self.reflection_version == 3:
                from . import grounding as reflection
            elif self.reflection_version == 2:
                from . import forecast as reflection
            else:
                from . import reflection

            self.reflection = reflection
        self.experiment_bindings = {}
        self.records = {}
        self.assessed_steps = set()
        self.pending_step = None
        self.notes = {}
        self.active_experiment = None
        self.completion_record = None
        self.accepted_decision = None
        self.memory_recorded = False
        self.finished = False

    def _validate_note(self, note: dict, assessed_steps: set[int], previous=None) -> dict:
        core = note
        if self.reflection_version == 4:
            fields = ("id", "claim", "status", "evidence_steps", "control", "comparison")
            core = {field: note[field] for field in fields}
            previous = {field: previous[field] for field in fields} if previous is not None else None
        normalized = self.contract.validate_hypothesis(
            core, self.records, assessed_steps,
            require_comparison=self.require_comparison, previous=previous,
        )
        if self.reflection_version == 4:
            normalized["effect"] = self.reflection.validate_effect(note["effect"], normalized, self.records, assessed_steps)
        return normalized

    def _comparison_receipts(self, notes):
        helper = self.reflection if self.reflection_version == 4 else self.contract
        return helper.comparison_receipts(notes, self.records)

    def _research_helper(self):
        if self.reflection_version == 4:
            from . import research2

            return research2
        from . import research

        return research

    def request(self, context: dict):
        expected = self.contract.comparison_context(self.records, self.assessed_steps, self.pending_step)
        if not _same_json(context.get("comparison_contract"), expected):
            raise ValueError("Replay comparison request catalog differs from completed execution.")
        if not _same_json(context.get("pending_assessment_step"), self.pending_step):
            raise ValueError("Replay comparison request pending assessment differs.")
        if not _same_json(context.get("hypotheses"), list(self.notes.values())):
            raise ValueError("Replay comparison request retained memory differs.")
        active = self.active_experiment
        if self.reflection_version in {2, 3, 4} and active is not None:
            active = {field: value for field, value in active.items() if field != "plan"}
        if not _same_json(context.get("active_experiment"), active):
            raise ValueError("Replay comparison request active experiment differs.")
        if self.reflection is not None:
            expected_reflection = {
                "contract_version": self.reflection_version,
                "pending_outcome": self.reflection.reflection_receipt(self.records, self.pending_step),
                "experiments": self.reflection.experiment_context(self.experiment_bindings, self.records, self.assessed_steps),
            }
            if not _same_json(context.get("reflection_contract"), expected_reflection):
                raise ValueError("Replay reflection request facts differ from completed execution.")
            progress = context.get("question_progress")
            if isinstance(progress, dict) and "declared_trial_progress" in progress:
                expected_progress = self.reflection.trial_progress(self.experiment_bindings, self.records, self.assessed_steps)
                if not _same_json(progress["declared_trial_progress"], expected_progress):
                    raise ValueError("Replay reflection declared trial progress differs from completed execution.")
            if self.reflection_version in {2, 3, 4}:
                expected_research = self._research_helper().plan_context(self.records, context["observation"])
                if "research_context" not in context or not _same_json(context["research_context"], expected_research):
                    raise ValueError("Replay research context differs from completed controls and the remaining plan.")
            if self.reflection_version == 4:
                progress = self._research_helper().scope_progress(self.records, self.assessed_steps, list(self.notes.values()))
                if "experiment_progress" not in context or not _same_json(context["experiment_progress"], progress):
                    raise ValueError("Replay experiment progress differs from completed trials and accepted effect notes.")
                requirement_field = "required_trial_notes_after_valid_assessment"
                supplied_requirements = []
                if requirement_field in context:
                    supplied_requirements.append(context[requirement_field])
                contract = context.get("response_contract")
                hypotheses_contract = contract.get("hypotheses") if isinstance(contract, dict) else None
                if isinstance(hypotheses_contract, dict) and requirement_field in hypotheses_contract:
                    supplied_requirements.append(hypotheses_contract[requirement_field])
                if supplied_requirements:
                    from .exploration import trial_note_requirements

                    expected_requirements = trial_note_requirements(self.records, self.assessed_steps, self.notes, self.pending_step)
                    if any(not _same_json(value, expected_requirements) for value in supplied_requirements):
                        raise ValueError("Replay conditional trial-note requirements differ from completed trials and accepted assessments.")
        self.completion_record = None
        self.accepted_decision = None
        self.memory_recorded = False

    def completion(self, record: dict):
        if self.completion_record is not None:
            raise ValueError("Replay comparison request has multiple completions.")
        self.completion_record = record

    def memory(self, record: dict, observation: dict, episode: int):
        from .exploration import ExplorationDecision

        if self.memory_recorded:
            raise ValueError("Replay comparison request has multiple memory updates.")
        completion = self.completion_record
        if completion is None or completion.get("finish_reason") != "stop":
            raise ValueError("Replay comparison memory requires a completed, untruncated answer.")
        if not isinstance(completion.get("answer"), str):
            raise ValueError("Replay comparison memory has no recorded accepted answer.") from None
        allowed = ["reset", "finish"] if observation["status"] == "won" else [*observation["actions"], "reset", "finish"]
        reflection_kwargs = {
            "reflection_records": self.records,
            "experiment_bindings": self.experiment_bindings,
        } if self.reflection is not None else {}
        if self.reflection_version in {2, 3, 4}:
            reflection_kwargs["reflection_version"] = self.reflection_version
        parsed = ExplorationDecision.parse(
            completion["answer"], allowed, set(self.records), self.pending_step,
            comparison_records=self.records, assessed_steps=self.assessed_steps,
            require_comparison=self.require_comparison, previous_hypotheses=self.notes,
            **reflection_kwargs,
        )
        # Match the controller's canonical answer and JSON tuple/list encoding.
        decision = json.loads(json.dumps(asdict(parsed)))
        if self.reflection_version in {2, 3, 4} and not _same_json(json.loads(completion["answer"]), decision):
            raise ValueError("Replay single-forecast answer differs from the canonical model-written decision.")
        assessment = (
            self.reflection.validate_assessment(decision["assessment"], self.records, self.pending_step)
            if self.reflection is not None else
            self.contract.validate_assessment(decision["assessment"], self.records, self.pending_step)
        )
        candidate_assessed = self.assessed_steps.copy()
        if assessment is not None:
            candidate_assessed.add(assessment["step"])
        updates = decision.get("hypotheses")
        if not isinstance(updates, list) or len(updates) > 8:
            raise ValueError("Replay comparison hypothesis updates are invalid.")
        candidate = self.notes.copy()
        updated_ids = set()
        for update in updates:
            normalized = self._validate_note(update, candidate_assessed, previous=self.notes.get(update["id"]))
            if normalized["id"] in updated_ids:
                raise ValueError("Replay comparison decision repeats a hypothesis ID.")
            updated_ids.add(normalized["id"])
            candidate[normalized["id"]] = normalized
        if len(candidate) > 8:
            raise ValueError("Replay comparison memory exceeds eight hypothesis IDs.")
        for identifier, note in candidate.items():
            self._validate_note(note, candidate_assessed, previous=self.notes.get(identifier))
        if self.reflection is not None:
            self.reflection.validate_trial_comparisons(list(candidate.values()), self.records)
        if self.reflection_version == 4 and self._research_helper().unnoted_trial_steps(self.records, candidate_assessed, list(candidate.values())):
            raise ValueError("Replay accepted an assessed experiment trial without its required witnessed effect note.")
        if any(item["type"] == "step" for item in self.records.values()) and not any(
            note["evidence_steps"] and note["status"] != "untested" for note in candidate.values()
        ):
            raise ValueError("Replay comparison memory has no first evidence-linked learned note.")
        active = {"id": decision["experiment_id"], "goal": decision["experiment_goal"], "plan": decision["plan"]}
        if self.reflection is not None:
            active.update(control=decision["experiment_control"], action_role=decision["action_role"])
        if type(record.get("episode")) is not int or record["episode"] != episode:
            raise ValueError("Replay comparison memory episode differs.")
        if not _same_json(record.get("assessment"), assessment) or not _same_json(record.get("active_experiment"), active):
            raise ValueError("Replay comparison memory differs from the accepted assessment or experiment.")
        if not _same_json(record.get("hypotheses"), list(candidate.values())):
            raise ValueError("Replay comparison memory differs from the full accepted hypothesis snapshot.")
        receipts = self._comparison_receipts(list(candidate.values()))
        if "comparison_receipts" not in record or not _same_json(record["comparison_receipts"], receipts):
            raise ValueError("Replay comparison memory receipts differ from canonical recorded trials.")
        if self.reflection is not None:
            candidate_bindings = {**self.experiment_bindings, decision["experiment_id"]: decision["experiment_control"]}
            if "experiment_bindings" not in record or not _same_json(record["experiment_bindings"], candidate_bindings):
                raise ValueError("Replay reflection memory experiment bindings differ from accepted decisions.")
            pending_receipt = self.reflection.reflection_receipt(self.records, self.pending_step)
            if "reflection_receipt" not in record or not _same_json(record["reflection_receipt"], pending_receipt):
                raise ValueError("Replay reflection memory receipt differs from recorded prediction and execution.")
            if self.reflection_version == 4:
                measured = self.reflection.evidence_measurements(list(candidate.values()), self.records)
                if "evidence_measurements" not in record or not _same_json(record["evidence_measurements"], measured):
                    raise ValueError("Replay evidence measurements differ from cited actual game states.")
            self.experiment_bindings = candidate_bindings
        self.notes = candidate
        self.assessed_steps = candidate_assessed
        self.active_experiment = active
        self.pending_step = None
        self.accepted_decision = decision
        self.memory_recorded = True

    def control(self, record: dict):
        if not self.memory_recorded or self.accepted_decision is None:
            raise ValueError("Replay comparison control has no accepted memory update.")
        if not _same_json(record.get("decision"), self.accepted_decision):
            raise ValueError("Replay comparison control differs from the accepted answer.")
        if self.reflection_version in {2, 3, 4}:
            derived = self.reflection.prediction_record(self.accepted_decision["action"], self.accepted_decision["expected_outcome"])
            if any(field not in record or not _same_json(record[field], value) for field, value in derived.items()):
                raise ValueError("Replay single-forecast rendering or provenance differs from the original expected_outcome.")
            if record.get("type") == "finish":
                self.finished = True

    def executed(self, record: dict, episode_before: int, episode_after: int):
        self.records[record["step"]] = {**record, "episode_before": episode_before, "episode_after": episode_after}
        self.pending_step = record["step"]

    def summary(self, record: dict):
        if not _same_json(record.get("hypotheses"), list(self.notes.values())) or not _same_json(record.get("active_experiment"), self.active_experiment):
            raise ValueError("Replay comparison summary memory differs from accepted learning.")
        if not _same_json(record.get("unassessed_final_step"), self.pending_step):
            raise ValueError("Replay comparison summary pending assessment differs.")
        if "comparison_receipts" in record and not _same_json(record["comparison_receipts"], self._comparison_receipts(list(self.notes.values()))):
            raise ValueError("Replay comparison summary receipts differ from canonical recorded trials.")
        if self.reflection is not None:
            expected = [
                {**self.reflection.reflection_receipt(self.records, step), "assessed": step in self.assessed_steps}
                for step in sorted(self.records)
            ]
            if "experiment_bindings" not in record or not _same_json(record["experiment_bindings"], self.experiment_bindings):
                raise ValueError("Replay reflection summary experiment bindings differ from accepted decisions.")
            if "reflection_receipts" not in record or not _same_json(record["reflection_receipts"], expected):
                raise ValueError("Replay reflection summary receipts differ from recorded prediction and execution.")
            if self.reflection_version in {2, 3, 4}:
                expected_research = self._research_helper().plan_context(self.records, record["final_observation"], finished=self.finished)
                if "research_context" not in record or not _same_json(record["research_context"], expected_research):
                    raise ValueError("Replay research summary differs from completed controls and the remaining plan.")
            if self.reflection_version == 4:
                measured = self.reflection.evidence_measurements(list(self.notes.values()), self.records)
                progress = self._research_helper().scope_progress(self.records, self.assessed_steps, list(self.notes.values()))
                if "evidence_measurements" not in record or not _same_json(record["evidence_measurements"], measured):
                    raise ValueError("Replay summary measurements differ from cited actual game states.")
                if "experiment_progress" not in record or not _same_json(record["experiment_progress"], progress):
                    raise ValueError("Replay summary experiment progress differs from completed trials and accepted effect notes.")


class _LeanReplay:
    """Recompute lean-v1 facts and bind accepted decisions to their controls."""

    def __init__(self, header: dict):
        self.history_size = header.get("limits", {}).get("history_size", 6)
        if type(self.history_size) is not int or self.history_size <= 0:
            raise ValueError("Replay lean history size is invalid.")
        self.notes = {}
        self.records = []
        self.completion_record = None
        self.accepted_decision = None
        self.memory_recorded = False
        self.request_warnings = []
        self.all_warnings = []
        self.allowed = []
        self.phase2 = None
        self.parse_limits = None

    def _note_dicts(self, notes) -> list[dict]:
        from .lean import note_dict

        values = [note_dict(note) for note in notes]
        if self.phase2 is False:
            for value in values:
                value.pop("kind", None)
        return values

    def _decision_dict(self, decision) -> dict:
        from .lean import decision_dict

        value = decision_dict(decision)
        if self.phase2 is False:
            for note in value["notes"]:
                note.pop("kind", None)
        return value

    def request(self, context: dict, observation: dict, record: dict):
        from .lean import (
            LEAN_NOTE_LIMIT,
            compact_software_facts,
            position_control_coverage,
            software_facts,
        )

        request_phase2 = "note_slots_remaining" in context
        if self.phase2 is None:
            self.phase2 = request_phase2
        elif self.phase2 != request_phase2:
            raise ValueError("Replay lean request contract changed within the session.")

        schema_properties = record.get("response_schema", {}).get("properties", {})
        note_properties = (
            schema_properties.get("notes", {}).get("items", {}).get("properties", {})
        )
        reason_limit = schema_properties.get("reason", {}).get("maxLength")
        claim_limit = note_properties.get("claim", {}).get("maxLength")
        evidence_limit = note_properties.get("evidence_steps", {}).get("maxItems")
        # Historical lean schemas used 240-character text and no evidence cap.
        # Read each recording's request contract instead of applying new defaults.
        if reason_limit is None and claim_limit is None:
            reason_limit = claim_limit = 240
        if (
            type(reason_limit) is not int
            or reason_limit <= 0
            or reason_limit != claim_limit
            or (evidence_limit is not None and (type(evidence_limit) is not int or evidence_limit <= 0))
        ):
            raise ValueError("Replay lean response schema limits are invalid.")
        request_parse_limits = (evidence_limit, reason_limit)
        if self.parse_limits is None:
            self.parse_limits = request_parse_limits
        elif self.parse_limits != request_parse_limits:
            raise ValueError("Replay lean response schema limits changed within the session.")

        compact = "context_compaction" in context
        include_non_qa_reset_anomalies = self.parse_limits[0] is not None
        if compact:
            expected = compact_software_facts(
                self.records,
                include_non_qa_reset_anomalies=include_non_qa_reset_anomalies,
            )
            checks = {
                "notes": self._note_dicts(self.notes.values()),
                "context_compaction": expected["context_compaction"],
                "factual_log": expected["factual_log"],
                "earlier_factual_summary": expected["earlier_factual_summary"],
                "cross_episode_pair_fields": expected["cross_episode_pair_fields"],
                "cross_episode_same_start_pairs": expected["cross_episode_same_start_pairs"],
                "retention_facts": expected["retention_facts"],
            }
        else:
            expected = software_facts(
                self.records,
                self.history_size,
                include_non_qa_reset_anomalies=include_non_qa_reset_anomalies,
            )
            checks = {
                "notes": self._note_dicts(self.notes.values()),
                "factual_log": expected["factual_log"],
                "cross_episode_same_start_pairs": expected["cross_episode_same_start_pairs"][-12:],
                "retention_facts": expected["retention_facts"],
            }
        if self.phase2:
            checks["note_slots_remaining"] = max(0, LEAN_NOTE_LIMIT - len(self.notes))
        if "anomalies" in context:
            checks["anomalies"] = expected.get("anomalies", [])
        if "anomaly_summary" in context:
            checks["anomaly_summary"] = expected.get("anomaly_summary", {
                "observed": 0, "unique": 0, "included": 0, "omitted": 0,
            })
        if "exploration_coverage" in context:
            checks["exploration_coverage"] = position_control_coverage(
                observation, self.records,
            )
        mismatched = [
            name for name, value in checks.items()
            if not _same_json(context.get(name), value)
        ]
        if mismatched:
            raise ValueError(
                "Replay lean request facts or retained notes differ from recorded execution: "
                + ", ".join(mismatched)
                + "."
            )
        self.allowed = context["observation"]["controls"]
        self.completion_record = None
        self.accepted_decision = None
        self.memory_recorded = False
        self.request_warnings = []

    def completion(self, record: dict):
        if self.completion_record is not None:
            raise ValueError("Replay lean request has multiple completions.")
        self.completion_record = record

    def warning(self, record: dict):
        warning = {name: record.get(name) for name in ("request_number", "code", "message")}
        if not isinstance(warning["code"], str) or not isinstance(warning["message"], str):
            raise ValueError("Replay lean decision warning is malformed.")
        self.request_warnings.append({name: warning[name] for name in ("code", "message")})
        self.all_warnings.append(warning)

    def memory(self, record: dict, episode: int):
        from .lean import LeanDecision, _merge_notes, _warning, software_facts

        if self.memory_recorded:
            raise ValueError("Replay lean request has multiple memory updates.")
        completion = self.completion_record
        if completion is None or completion.get("finish_reason") != "stop" or not isinstance(completion.get("answer"), str):
            raise ValueError("Replay lean memory requires a completed accepted answer.")
        parsed, normalized_warnings = self._parse_decision(
            completion["answer"],
            self.allowed,
            {item["step"] for item in self.records},
            evidence_limit=self.parse_limits[0],
            text_limit=self.parse_limits[1],
        )
        if any(warning not in self.request_warnings for warning in normalized_warnings):
            raise ValueError("Replay lean memory omitted a canonical normalization warning.")
        if not _same_json(json.loads(completion["answer"]), self._decision_dict(parsed)):
            raise ValueError("Replay lean accepted answer is not canonical.")
        if self.phase2:
            candidate, merge_warnings = _merge_notes(self.notes, parsed.notes)
        else:
            candidate = dict(self.notes)
            merge_warnings = []
            for index, note in enumerate(parsed.notes):
                if note.id not in candidate and len(candidate) >= 6:
                    merge_warnings.append(_warning(
                        "note_memory_full",
                        f"Dropped note[{index}]: session memory already contains six note IDs.",
                    ))
                    continue
                candidate[note.id] = note
        # Merge warnings are reproducible from canonical memory capacity. Other
        # parser warnings describe discarded raw fields and are recorded rows.
        for warning in merge_warnings:
            if warning not in self.request_warnings:
                raise ValueError("Replay lean memory omitted a canonical note-capacity warning.")
        facts = software_facts(
            self.records,
            include_non_qa_reset_anomalies=self.parse_limits[0] is not None,
        )
        expected_software = {
            "cross_episode_same_start_pairs": facts["cross_episode_same_start_pairs"],
            "retention_facts": facts["retention_facts"],
        }
        if (
            record.get("episode") != episode
            or not _same_json(record.get("hypotheses"), self._note_dicts(candidate.values()))
            or not _same_json(record.get("notes_warnings"), self.request_warnings)
            or not _same_json(record.get("software_facts"), expected_software)
        ):
            raise ValueError("Replay lean memory differs from accepted notes, warnings, or software facts.")
        self.notes = candidate
        self.accepted_decision = self._decision_dict(parsed)
        self.memory_recorded = True

    def _parse_decision(self, *args, **kwargs):
        from .lean import LeanDecision

        return LeanDecision.parse(*args, **kwargs)

    def control(self, record: dict):
        if not self.memory_recorded or self.accepted_decision is None:
            raise ValueError("Replay lean control has no accepted memory update.")
        if not _same_json(record.get("decision"), self.accepted_decision):
            raise ValueError("Replay lean control differs from its accepted answer.")

    def executed(self, record: dict):
        from .lean import step_fact

        expected = step_fact(
            record,
            self.records,
            include_non_qa_reset_anomalies=self.parse_limits[0] is not None,
        )
        if not _same_json(record.get("software_fact"), expected):
            raise ValueError("Replay lean step facts differ from replayed execution.")
        self.records.append(record)

    def summary(self, record: dict):
        from .lean import software_facts

        facts = software_facts(
            self.records,
            history_size=None,
            include_non_qa_reset_anomalies=self.parse_limits[0] is not None,
        )
        if any(not _same_json(record.get(name), value) for name, value in facts.items()):
            raise ValueError("Replay lean summary facts differ from replayed execution.")
        if not _same_json(record.get("hypotheses"), self._note_dicts(self.notes.values())):
            raise ValueError("Replay lean summary notes differ from accepted memory.")
        if not _same_json(record.get("parse_warnings"), self.all_warnings) or record.get("warning_count") != len(self.all_warnings):
            raise ValueError("Replay lean summary warning accounting differs.")


def replay(path: Path, *, game_factory=None) -> dict:
    if game_factory is None:
        return _replay(path)
    game = game_factory()
    try:
        return _replay(path, game=game)
    finally:
        close = getattr(game, "close", None)
        if callable(close):
            close()


def _replay(path: Path, *, game=None) -> dict:
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not records or records[0].get("type") != "session":
        raise ValueError("Replay requires a session header.")
    header = records[0]
    schema = header.get("schema_version")
    if schema not in {1, 2} or header.get("game_version") != (KeyGateGame.version if game is None else game.version):
        raise ValueError("Replay supports schema 1 or 2 and the current KeyGateGame version.")
    if schema == 2 and header.get("task") != "explore":
        raise ValueError("Schema 2 replay requires an exploration session.")
    strict = None
    if "comparison_contract" in header:
        if type(header["comparison_contract"]) is not int or header["comparison_contract"] != 1 or schema != 2:
            raise ValueError("Replay supports comparison contract version 1 on schema-2 exploration only.")
        strict = _ComparisonReplay(header)
    if "reflection_contract" in header:
        if type(header["reflection_contract"]) is not int or header["reflection_contract"] not in {1, 2, 3, 4} or schema != 2 or strict is None:
            raise ValueError("Replay supports reflection contract versions 1, 2, 3 or 4 with comparison contract 1 on schema-2 exploration only.")
    lean = None
    from .adaptive_protocol import ADAPTIVE_VERSIONS

    adaptive_mode = header.get("prompt_version") in ADAPTIVE_VERSIONS
    if str(header.get("prompt_version", "")).startswith("adaptive-") and not adaptive_mode:
        raise ValueError("Replay requires a supported versioned adaptive protocol.")
    if header.get("prompt_version") == "lean-v1" or adaptive_mode:
        if schema != 2 or strict is not None or "reflection_contract" in header:
            raise ValueError("Replay lean-v1 requires a schema-2 exploration session without legacy contracts.")
        if adaptive_mode:
            from .adaptive_replay import AdaptiveReplay
            lean = AdaptiveReplay(header)
        else:
            lean = _LeanReplay(header)
    game = _recording_game(header) if game is None else game
    if game.observe() != header["initial_observation"] or (strict is not None and not _same_json(game.observe(), header["initial_observation"])):
        raise ValueError("Replay initial state differs from the recording.")
    count = resets = episode = requests = 0
    finished = False
    request_control_executed = False
    for record in records[1:]:
        kind = record.get("type")
        if adaptive_mode and getattr(lean, "_admission_stopped", False) and kind != "summary":
            raise ValueError("Replay contains an event after terminal request admission stop.")
        if adaptive_mode and kind == "context_budget_failure":
            lean.budget_failure(record, game.observe())
            continue
        if adaptive_mode and kind == "request_admission_stop":
            lean.admission_stop(record)
            continue
        if adaptive_mode and kind == "unused_context":
            if record.get("dispatched") is not False or record.get("episode") != episode:
                raise ValueError("Replay unused adaptive context receipt is invalid.")
            lean.request(json.loads(record["messages"][1]["content"]), game.observe(), record)
            continue
        if kind == "request" and schema == 2:
            # Import here because exploration also uses SessionLog. Validate the
            # learner's actual refreshed view, including immediately after resets.
            from .exploration import exploration_view

            requests += 1
            if finished or type(record.get("request_number")) is not int or record["request_number"] != requests or record.get("episode") != episode or record.get("observation_tick") != game.observe()["tick"]:
                raise ValueError(f"Replay request accounting differs at request {requests}.")
            try:
                context = json.loads(record["messages"][1]["content"])
            except (KeyError, IndexError, TypeError, json.JSONDecodeError):
                raise ValueError(f"Replay request observation is missing at request {requests}.") from None
            if not isinstance(context, dict) or context.get("episode") != episode or (not adaptive_mode and context.get("observation") != exploration_view(game.observe())):
                raise ValueError(f"Replay request observation differs at request {requests}.")
            if strict is not None:
                if type(record.get("episode")) is not int or type(record.get("observation_tick")) is not int or type(context.get("episode")) is not int or not _same_json(context.get("observation"), exploration_view(game.observe())):
                    raise ValueError(f"Replay comparison request typed observation differs at request {requests}.")
                strict.request(context)
            if lean is not None:
                lean.request(context, game.observe(), record)
            request_control_executed = False
            continue
        if schema == 2 and kind in {"completion", "memory", "decision_warning", "rejected_decision", "step", "reset", "finish"}:
            if requests == 0 or type(record.get("request_number")) is not int or record["request_number"] != requests:
                raise ValueError(f"Replay {kind} request association differs at request {requests}.")
            if kind in {"step", "reset", "finish"}:
                if request_control_executed:
                    raise ValueError(f"Replay found multiple controls for request {requests}.")
                request_control_executed = True
            if strict is not None:
                if kind == "completion":
                    strict.completion(record)
                elif kind == "memory":
                    strict.memory(record, game.observe(), episode)
                elif kind in {"step", "reset", "finish"}:
                    strict.control(record)
                elif kind == "rejected_decision" and strict.memory_recorded:
                    raise ValueError("Replay comparison rejected an already accepted memory update.")
            if lean is not None:
                if kind == "completion":
                    lean.completion(record)
                elif kind == "decision_warning":
                    lean.warning(record)
                elif kind == "memory":
                    lean.memory(record, episode)
                elif kind in {"step", "reset", "finish"}:
                    lean.control(record)
                elif kind == "rejected_decision" and lean.memory_recorded:
                    raise ValueError("Replay lean session rejected an already accepted memory update.")
        if kind == "finish" and schema == 2:
            if finished or record.get("episode") != episode or record.get("observation") != game.observe() or record.get("decision", {}).get("action") != "finish":
                raise ValueError("Replay finish state differs from the recording.")
            if strict is not None and (type(record.get("episode")) is not int or not _same_json(record.get("observation"), game.observe())):
                raise ValueError("Replay comparison finish typed state differs.")
            finished = True
            continue
        if kind not in {"step", "reset"}:
            continue
        if finished:
            raise ValueError("Replay found an action after exploration finished.")
        if kind == "reset" and schema != 2:
            raise ValueError("Reset records require recording schema 2.")
        count += 1
        if schema == 2 and record.get("step") != count:
            raise ValueError(f"Replay global action number differs at action {count}.")
        if strict is not None and type(record.get("step")) is not int:
            raise ValueError(f"Replay comparison action number is not an integer at action {count}.")
        before = game.observe()
        if before != record["before"] or (strict is not None and not _same_json(before, record["before"])):
            raise ValueError(f"Replay diverged before action {count}.")
        action = record["decision"]["action"]
        episode_before = episode
        if strict is not None:
            fields = ("episode_before", "episode_after") if kind == "reset" else ("episode",)
            if any(type(record.get(field)) is not int for field in fields):
                raise ValueError(f"Replay comparison action episode is not an integer at action {count}.")
            if kind == "step" and any(field in record and (type(record[field]) is not int or record[field] != episode) for field in ("episode_before", "episode_after")):
                raise ValueError(f"Replay comparison game-control episode differs at action {count}.")
        if kind == "reset":
            if action != "reset" or record.get("episode_before") != episode or record.get("episode_after") != episode + 1:
                raise ValueError(f"Replay reset episode differs at action {count}.")
            game = game.reset()
            after = game.observe()
            episode += 1
            resets += 1
        else:
            if schema == 2 and record.get("episode") != episode:
                raise ValueError(f"Replay episode differs at action {count}.")
            after = game.act(action)
        if after != record["after"] or (strict is not None and not _same_json(after, record["after"])):
            raise ValueError(f"Replay diverged after action {count} ({action}).")
        game_evaluator = getattr(game, "evaluate_transition", None)
        if adaptive_mode and not callable(game_evaluator):
            replayed_evaluation = {"moved": before.get("player", {}).get("position") != after.get("player", {}).get("position"),
                                  "events": after.get("events", []), "contract_violations": []}
        else:
            replayed_evaluation = game_evaluator(before, action, after)
        if kind == "step" and (
            replayed_evaluation != record["evaluation"]
            or (strict is not None and not _same_json(replayed_evaluation, record["evaluation"]))
        ):
            raise ValueError(f"Replay evaluation differs at action {count}.")
        if strict is not None:
            strict.executed(record, episode_before, episode)
        if lean is not None:
            lean.executed(record)
    summary = records[-1] if records[-1].get("type") == "summary" else None
    if summary is not None:
        if summary.get("steps") != count or summary.get("final_observation") != game.observe():
            raise ValueError("Replay final state or action count differs from the session summary.")
        if schema == 2:
            if summary.get("resets") != resets or summary.get("game_actions") != count - resets or summary.get("episodes") != episode + 1:
                raise ValueError("Replay episode/reset counts differ from the session summary.")
            if summary.get("inference_requests") != requests:
                raise ValueError("Replay inference request count differs from the session summary.")
            if (summary.get("status") == "finished") != finished:
                raise ValueError("Replay finish control differs from the session summary.")
            if strict is not None:
                if any(type(summary.get(field)) is not int for field in ("steps", "resets", "game_actions", "episodes", "inference_requests")) or not _same_json(summary.get("final_observation"), game.observe()):
                    raise ValueError("Replay comparison summary typed state or counts differ.")
                strict.summary(summary)
            if lean is not None:
                lean.summary(summary)
    result = {"actions_replayed": count, "status": game.observe()["status"], "recorded_mode": header["mode"], "recording_complete": summary is not None}
    if schema == 2:
        result.update(task="explore", resets_replayed=resets, episodes=episode + 1, learning_claims_verified=False, exploration_status=summary.get("status") if summary else None)
    if strict is not None:
        result.update(comparison_contract=1, comparison_bindings_validated=True,
                      validated_comparisons=len(strict._comparison_receipts(list(strict.notes.values()))))
        if strict.reflection is not None:
            result.update(reflection_contract=strict.reflection_version, reflection_bindings_validated=True,
                          reflected_steps=len(strict.assessed_steps))
    if lean is not None:
        result.update(prompt_version=header.get("prompt_version"), lean_facts_validated=True)
    if adaptive_mode:
        result.update(adaptive_facts_validated=not bool(getattr(lean, "fact_failures", 0)), prior_run_memory_execution_verified=False)
    return result
