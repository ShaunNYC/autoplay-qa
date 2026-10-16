"""Offline evidence gates for the fixed six-session Mars UI experiment.

This module never calls a model, opens a browser, or changes an allowance.
Hashes bind the evidence; they do not establish that a reviewer is honest.
The review digest MUST come from a separately trusted independent review,
not from the actor, a recording, or a digest embedded in the review itself.
"""

from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import re
import time


MANIFEST_SCHEMA = "semantic-study-manifest-v1"
REVIEW_SCHEMA = "semantic-independent-review-v1"
ANALYSIS_SCHEMA = "semantic-study-analysis-v1"
REGRESSION_SCHEMA = "semantic-regression-case-v1"
TRACE_SCHEMA = "semantic-browser-trace-v1"
LIMITS = {"max_controls": 24, "max_requests": 6, "max_seconds": 300, "max_hypotheses": 2}
RESOURCE_LIMITS = {"max_input_bytes": 48_000, "max_total_input_bytes": 158_000, "request_timeout_seconds": 45}
MODEL_CONFIG = {"model": "nvidia/Nemotron-3-Ultra-550b-a55b", "enable_thinking": True,
                "max_tokens": 8192, "temperature": 1.0, "top_p": 0.95, "response_format": "json_schema"}


def _json(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def _digest(raw):
    return hashlib.sha256(raw).hexdigest()


def _hex(value):
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def _text(value):
    return isinstance(value, str) and bool(value.strip())


def _finite(value, maximum=None):
    return (type(value) in (int, float) and math.isfinite(value) and value >= 0
            and (maximum is None or value <= maximum))


def _time(value):
    _require(isinstance(value, str), "audit_timestamp_missing")
    try:
        timestamp = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        raise EvidenceError("audit_timestamp_invalid") from None
    _require(timestamp.tzinfo is not None, "audit_timestamp_timezone_missing")
    return timestamp


class EvidenceError(ValueError):
    """Missing, altered, or inconsistent evidence; never a favorable result."""


def _require(condition, message):
    if not condition:
        raise EvidenceError(message)


def _read(ref, base, *, lines=False):
    _require(isinstance(ref, dict) and _text(ref.get("path")) and _hex(ref.get("sha256")), "invalid_artifact_reference")
    path = Path(ref["path"])
    path = path if path.is_absolute() else base / path
    try:
        raw = path.read_bytes()
        _require(len(raw) <= 50_000_000, "artifact_too_large")
        _require(_digest(raw) == ref["sha256"], "artifact_hash_mismatch")
        value = [json.loads(line) for line in raw.splitlines() if line.strip()] if lines else json.loads(raw)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise EvidenceError("artifact_unreadable") from exc
    return value


def _read_source(ref, base):
    _require(isinstance(ref, dict) and _text(ref.get("path")) and _hex(ref.get("sha256")), "invalid_source_reference")
    path = Path(ref["path"])
    path = path if path.is_absolute() else base / path
    try:
        raw = path.read_bytes()
        _require(_digest(raw) == ref["sha256"], "source_hash_mismatch")
        lines = raw.decode("utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        raise EvidenceError("source_unreadable") from exc
    start, end = ref.get("line_start"), ref.get("line_end")
    _require(type(start) is int and type(end) is int and 1 <= start <= end <= len(lines)
             and _text(ref.get("explanation")), "source_citation_invalid")


def _index(rows, key, message):
    _require(isinstance(rows, list), message)
    out = {}
    for row in rows:
        _require(isinstance(row, dict) and _text(row.get(key)) and row[key] not in out, message)
        out[row[key]] = row
    return out


def _citations(refs, observations, *, before_event=None):
    _require(isinstance(refs, list) and bool(refs) and all(_text(ref) for ref in refs)
             and len(refs) == len(set(refs)), "evidence_citations_invalid")
    for ref in refs:
        _require(ref in observations, "evidence_citation_missing")
        if before_event is not None:
            _require(observations[ref][0] < before_event, "public_clue_not_prior")


def _reverse_changes(observation, changes):
    """Independently restore compact public deltas, checking every endpoint."""
    restored = deepcopy(observation)
    _require(isinstance(changes, list), "public_changes_invalid")
    for change in reversed(changes):
        _require(isinstance(change, dict) and isinstance(change.get("pointer"), str), "public_change_invalid")
        pointer = change["pointer"]
        _require((not pointer or pointer.startswith("/")) and not re.search(r"~(?![01])", pointer), "public_change_pointer_invalid")
        parts = [part.replace("~1", "/").replace("~0", "~") for part in pointer.split("/")[1:]] if pointer else []
        parent, key, current = None, None, restored
        for part in parts:
            parent = current
            if isinstance(parent, list):
                _require(bool(re.fullmatch(r"0|[1-9][0-9]*", part)), "public_change_index_invalid")
                key = int(part)
            else:
                _require(isinstance(parent, dict), "public_change_path_invalid")
                key = part
            try:
                current = parent[key]
            except (KeyError, IndexError):
                raise EvidenceError("public_change_path_missing") from None
        if "splice_kind" in change:
            expected_type = str if change["splice_kind"] == "string" else list if change["splice_kind"] == "list" else None
            start, count, deleted = change.get("start"), change.get("inserted_length"), change.get("deleted")
            _require(expected_type is not None and isinstance(current, expected_type) and isinstance(deleted, expected_type)
                     and type(start) is int and type(count) is int and 0 <= start <= len(current)
                     and 0 <= count <= len(current) - start, "public_splice_invalid")
            prior = current[:start] + deleted + current[start + count:]
        else:
            _require("before" in change and "after" in change and _json(current) == _json(change["after"]), "public_change_after_mismatch")
            prior = deepcopy(change["before"])
        if parent is None:
            restored = prior
        else:
            parent[key] = prior
    return restored


def _audit_packets(requests, responses, observations, controls, condition, summary):
    total, maximum, reviewed, last_page, page_number, page_ids = 0, 0, set(), 0, 0, []
    for number, (request_index, request) in requests.items():
        packet = request.get("packet")
        _require(isinstance(packet, dict) and packet.get("condition") == condition
                 and packet.get("model_role") == "nemotron-ultra" and packet.get("protocol") == "semantic-ui-v1", "actor_packet_identity_invalid")
        current_id, history = packet.get("current_evidence_id"), packet.get("evidence")
        _require(current_id in observations and isinstance(history, list) and history, "actor_packet_evidence_missing")
        _require(packet.get("current_observation") == observations[current_id][1]["observation"], "actor_current_observation_mismatch")
        ids = [item.get("evidence_id") for item in history if isinstance(item, dict)]
        _require(len(ids) == len(history) and all(isinstance(i, str) and re.fullmatch(r"e[0-9]+", i) for i in ids)
                 and len(set(ids)) == len(ids) and ids[-1] == current_id, "actor_evidence_ids_invalid")
        first, last = int(ids[0][1:]), int(ids[-1][1:])
        _require(ids == [f"e{i}" for i in range(first, last + 1)], "actor_evidence_page_gap")
        observed = deepcopy(packet["current_observation"])
        for item in reversed(history[1:]):
            event = observations[item["evidence_id"]][1]
            _require(item.get("action") == event.get("action") and item.get("changes") == event.get("changes")
                     and item.get("before") == event.get("before"), "actor_transition_differs_from_recording")
            observed = _reverse_changes(observed, item["changes"])
            _require(observed == observations[event["before"]][1]["observation"], "actor_history_reconstruction_mismatch")
        _require(observed == observations[ids[0]][1]["observation"], "actor_page_anchor_mismatch")
        if condition == "A":
            available = [identifier for identifier, (index, _) in observations.items() if index < request_index]
            _require(ids == available and "review_page" not in packet, "A_history_missing_or_future_evidence")
        else:
            page = packet.get("review_page")
            _require(isinstance(page, dict) and page.get("evidence_ids") == ids
                     and type(page.get("total")) is int and 1 <= page["total"] <= 6
                     and page.get("final") is (page.get("number") == page["total"])
                     and ((page.get("number") == page_number + 1 and first == last_page)
                          or (page.get("number") == page_number and page.get("final") is True and ids == page_ids)),
                     "B_review_page_invalid")
            _require(packet.get("reviewed_evidence") == sorted(reviewed, key=lambda value: int(value[1:])), "B_review_history_invalid")
            decision = responses[number][1]["response"]["decision"]
            _require(set(ids) <= set(decision.get("acknowledged_evidence", [])), "B_page_not_acknowledged")
            reviewed.update(ids)
            last_page = last
            page_number, page_ids = page["number"], ids
        schema = packet.get("response_schema")
        _require(isinstance(schema, dict), "actor_schema_missing")
        public = {key: value for key, value in packet.items() if key != "response_schema"}
        size = len(_json([{"role": "user", "content": _json(public)}]).encode()) + len(_json(schema).encode())
        _require(request.get("input_bytes") == size and size <= RESOURCE_LIMITS["max_input_bytes"], "input_size_mismatch_or_overflow")
        provider_size = responses[number][1]["response"].get("provider_input_bytes", size)
        _require(provider_size == size, "provider_input_size_mismatch")
        total += size
        maximum = max(maximum, size)
    _require(summary.get("max_input_bytes") == maximum and summary.get("total_input_bytes") == total
             and total <= RESOURCE_LIMITS["max_total_input_bytes"], "cumulative_input_mismatch_or_overflow")
    if condition == "B":
        _require(reviewed == set(observations) and summary.get("reviewed_evidence") == list(observations), "B_evidence_review_incomplete")


def _session(row, manifest, base):
    events = _read(row.get("recording"), base, lines=True)
    _require(events and all(isinstance(e, dict) for e in events), "recording_malformed")
    _require(events[0].get("type") == "semantic_session" and events[-1].get("type") == "semantic_summary", "recording_boundaries_invalid")
    header, summary = events[0], events[-1]
    _require(sum(e.get("type") == "semantic_session" for e in events) == 1
             and sum(e.get("type") == "semantic_summary" for e in events) == 1, "recording_boundaries_duplicated")
    for field in ("session_id", "condition", "instance_id", "actor_id", "source_sha256", "initial_state_sha256"):
        _require(_text(row.get(field)) and header.get(field) == row[field], "session_metadata_mismatch:" + field)
    for field in ("source_sha256", "initial_state_sha256"):
        _require(row[field] == manifest.get(field) and _hex(row[field]), "session_comparison_mismatch:" + field)
    _require(isinstance(header.get("limits"), dict) and all(header["limits"].get(k) == v for k, v in LIMITS.items()), "session_limits_mismatch")
    _require(all(header["limits"].get(k) == v for k, v in RESOURCE_LIMITS.items()), "session_resource_limits_mismatch")
    _require(header.get("model_config") == manifest.get("model_config")
             and header.get("model_config_sha256") == _digest(_json(manifest["model_config"]).encode())
             and header.get("browser_transport_sha256") == _digest(_json(manifest["browser_transport"]).encode()),
             "session_model_or_browser_configuration_mismatch")
    for field, artifact in (("baseline_policy_sha256", "baseline_policy"), ("known_registry_sha256", "known_registry")):
        _require(header.get(field) == manifest[artifact]["sha256"], "pre_session_freeze_binding_missing:" + field)
    _require(header.get("experience_shared") is False, "session_isolation_unproven")
    _require(summary.get("complete") is True and summary.get("usage_complete") is True
             and summary.get("unknown_usage") is False and summary.get("pending_outcome", "missing") is None,
             "session_incomplete_or_unknown")
    _require(_finite(summary.get("elapsed_seconds"), 300) and _finite(summary.get("cost_estimate_usd")), "session_accounting_missing")
    observations, controls, starts, requests, responses, decisions, hypotheses, assessments, claims = {}, {}, {}, {}, {}, {}, {}, [], {}
    for index, event in enumerate(events):
        kind = event.get("type")
        if kind in {"semantic_observation", "semantic_transition"}:
            identifier = event.get("evidence_id")
            _require(_text(identifier) and identifier not in observations and isinstance(event.get("observation"), dict), "observation_invalid")
            observations[identifier] = (index, event)
        if kind in {"semantic_control_started", "semantic_transition"}:
            number = event.get("control_number")
            target = starts if kind == "semantic_control_started" else controls
            _require(type(number) is int and 1 <= number <= 24 and number not in target, "control_number_invalid")
            target[number] = (index, event)
        if kind in {"semantic_request", "semantic_response"}:
            number = event.get("request_number")
            target = requests if kind == "semantic_request" else responses
            _require(type(number) is int and 1 <= number <= 6 and number not in target, "request_number_invalid")
            target[number] = (index, event)
        if kind == "semantic_decision":
            number = event.get("request_number")
            _require(type(number) is int and number not in decisions, "decision_number_invalid")
            decisions[number] = (index, event)
        if kind == "semantic_hypothesis":
            identifier = event.get("hypothesis_id")
            _require(_text(identifier) and identifier not in hypotheses, "hypothesis_identity_invalid")
            hypotheses[identifier] = (index, event)
        if kind == "semantic_assessment":
            assessments.append((index, event))
        if kind == "semantic_candidate":
            identifier = event.get("claim_id")
            _require(_text(identifier) and identifier not in claims, "claim_identity_invalid")
            claims[identifier] = (index, event)
    _require(set(controls) == set(range(1, len(controls) + 1)), "pending_or_noncontiguous_controls")
    _require(set(starts) == (set(controls) if row["condition"] == "A" else set()), "pending_or_noncontiguous_controls")
    _require(set(requests) == set(responses) == set(range(1, len(requests) + 1)), "missing_request_or_response")
    _require(set(decisions) == set(responses), "response_without_accepted_decision")
    _require(len(hypotheses) <= 2 and (row["condition"] == "A" or not hypotheses), "hypothesis_limit_or_B_intervention")
    _require(summary.get("controls") == len(controls) and summary.get("requests") == len(requests)
             and summary.get("responses") == len(responses), "summary_counts_mismatch")
    final_claims = summary.get("final_claims")
    payloads = [{k: v for k, v in event.items() if k not in {"type", "timestamp", "elapsed_ms"}} for _, event in claims.values()]
    _require(isinstance(final_claims, list) and final_claims == payloads, "claims_erased_or_changed")
    for number, (index, event) in controls.items():
        _require(event.get("model_selected") is (row["condition"] == "A"), "control_provenance_invalid")
        start_index = index
        if row["condition"] == "A":
            start_index, started = starts[number]
            _require(start_index < index and started.get("action") == event.get("action")
                     and started.get("model_selected") is True, "control_provenance_invalid")
            request = started.get("request_number")
            _require(request in responses and responses[request][0] < start_index, "model_control_without_prior_response")
            for field in ("request_number", "arm", "hypothesis_id"):
                _require(started.get(field) == event.get(field), "control_tags_mismatch")
        else:
            _require(event.get("arm") == "script" and event.get("request_number") is None and event.get("hypothesis_id") is None, "B_control_not_script")
        _require(event.get("before") in observations and observations[event["before"]][0] < start_index,
                 "control_before_observation_invalid")
    for number, (index, _) in responses.items():
        _require(requests[number][0] < index, "response_precedes_request")
        decision_index, accepted = decisions[number]
        raw = responses[number][1].get("response", {}).get("decision")
        _require(index < decision_index and accepted.get("decision") == raw and isinstance(raw, dict), "accepted_decision_differs_from_response")
        declared_actions = raw.get("actions")
        actual = [{"action": e["action"], "arm": e["arm"], "hypothesis_id": e.get("hypothesis_id")}
                  for i, e in controls.values() if e.get("request_number") == number and i > decision_index]
        _require(isinstance(declared_actions, list) and declared_actions == actual, "executed_controls_differ_from_model_choice")
        declarations = raw.get("hypotheses")
        _require(isinstance(declarations, list), "raw_hypotheses_malformed")
        for declaration in declarations:
            _require(isinstance(declaration, dict) and declaration.get("id") in hypotheses, "declared_hypothesis_missing")
            hypothesis_index, hypothesis = hypotheses[declaration["id"]]
            _require(hypothesis_index > decision_index and hypothesis.get("created_at_request") == number
                     and all(hypothesis.get(k) == v for k, v in declaration.items()), "hypothesis_differs_from_model_choice")
        _require(raw.get("finish") is (number == len(responses)), "completion_not_final_model_choice")
    final_decision = responses[len(responses)][1]["response"]["decision"] if responses else {}
    _require(not final_decision.get("actions") and f"e{len(controls)}" in final_decision.get("acknowledged_evidence", []), "final_observation_not_acknowledged")
    totals = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    cost = 0.0
    for _, event in responses.values():
        response = event.get("response")
        _require(response.get("finish_reason", "stop") == "stop", "raw_response_truncated_or_refused")
        if "raw_content" in response:
            _require(isinstance(response["raw_content"], str) and json.loads(response["raw_content"]) == response.get("decision"), "raw_content_decision_mismatch")
        usage = response.get("usage")
        _require(isinstance(usage, dict) and all(type(usage.get(k)) is int and usage[k] >= 0 for k in totals)
                 and usage["total_tokens"] == usage["prompt_tokens"] + usage["completion_tokens"], "raw_usage_unknown_or_inconsistent")
        for key in totals:
            totals[key] += usage[key]
        _require(_finite(response.get("estimated_cost_usd")), "raw_cost_unknown")
        cost += response["estimated_cost_usd"]
    _require(summary.get("usage") == totals and math.isclose(summary["cost_estimate_usd"], cost, abs_tol=1e-10), "summary_accounting_mismatch")
    for _, claim in claims.values():
        _citations(claim.get("evidence"), observations)
    _require("e0" in observations and _digest(_json(observations["e0"][1]["observation"]).encode()) == manifest["initial_state_sha256"], "initial_observation_hash_mismatch")
    # Reconstruct declarations from raw responses; an actor's final summary
    # cannot erase a mistaken earlier claim, including a rejected declaration.
    raw_claims, declared_ids = [], []
    for number, (_, event) in responses.items():
        decision = event.get("response", {}).get("decision")
        _require(isinstance(decision, dict) and isinstance(decision.get("suspected_bug"), list), "raw_response_claims_malformed")
        raw_claims.append({"request_number": number, "declarations": decision["suspected_bug"]})
        for declaration in decision["suspected_bug"]:
            _require(isinstance(declaration, dict) and _text(declaration.get("id")), "raw_claim_malformed")
            identifier = declaration["id"]
            _require(identifier in claims and identifier not in declared_ids, "raw_claim_inventory_incomplete")
            claim = claims[identifier][1]
            _require(claim.get("request_number") == number and all(claim.get(k) == v for k, v in declaration.items()), "raw_claim_changed")
            declared_ids.append(identifier)
    _require(summary.get("raw_claims") == raw_claims and declared_ids == list(claims), "raw_claim_inventory_incomplete")
    _audit_packets(requests, responses, observations, controls, row["condition"], summary)
    if row["condition"] == "B":
        trajectory = _read(row.get("baseline_trajectory"), base)
        _require(trajectory.get("policy_sha256") == manifest["baseline_policy"]["sha256"]
                 and trajectory.get("seed") == row.get("seed") and _finite(trajectory.get("elapsed_seconds"), 300), "B_trajectory_policy_mismatch")
        initial = next((event["observation"] for event in events if event.get("type") == "semantic_observation"), None)
        _require(trajectory.get("initial_observation") == initial and trajectory.get("transitions") ==
                 [{"action": event["action"], "observation": event["observation"]} for _, event in controls.values()], "B_raw_trajectory_mismatch")
    return {"row": row, "header": header, "summary": summary, "events": events,
            "observations": observations, "controls": controls, "hypotheses": hypotheses,
            "requests": requests, "responses": responses,
            "starts": starts, "assessments": assessments, "claims": claims}


def _source_manifest(ref, base, source_sha):
    source = _read(ref, base)
    _require(isinstance(source, dict) and source.get("format") == "mars-browser-source-v1"
             and source.get("sourceSha256") == source_sha and source.get("sourceUnmodified") is True
             and source.get("defectsInjected") is False, "original_source_manifest_invalid")
    path = Path(ref["path"])
    directory = (path if path.is_absolute() else base / path).parent.resolve()
    for field, digest_field in (("sourceFiles", "sourceSha256"), ("buildFiles", "buildSha256")):
        files = source.get(field)
        indexed = _index(files, "path", "source_file_inventory_invalid")
        _require(indexed, "source_file_inventory_empty")
        # The native preparer hashes JSON.stringify in recorded property order.
        serialized = json.dumps(files, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
        _require(_digest(serialized.encode()) == source.get(digest_field), "source_file_inventory_hash_mismatch")
        for item in indexed.values():
            candidate = (directory / item["path"]).resolve()
            _require(candidate.is_relative_to(directory) and _hex(item.get("sha256")), "source_file_reference_invalid")
            try:
                _require(_digest(candidate.read_bytes()) == item["sha256"], "source_or_build_bytes_changed")
            except OSError:
                raise EvidenceError("source_or_build_bytes_missing") from None


def _session_audits(review, sessions, manifest, base):
    """Validate original native replays and actual request-level cost rows."""
    _source_manifest(review.get("source_manifest"), base, manifest["source_sha256"])
    ledger = _read(review.get("request_ledger"), base)
    _require(isinstance(ledger, dict) and ledger.get("version") == "dashboard-cost-v1"
             and ledger.get("cap_usd") == 2.0, "request_ledger_cap_or_schema_invalid")
    ledger_rows = _index(ledger.get("requests"), "id", "request_ledger_rows_invalid")
    audits = _index(review.get("session_audits"), "session_id", "session_audits_missing")
    _require(set(audits) == set(sessions), "session_audits_incomplete")
    used, seen_instances = set(), {session["row"]["instance_id"] for session in sessions.values()}
    total_cost, reservation_times = 0.0, []
    for sid, audit in audits.items():
        session = sessions[sid]
        recording_sha = session["row"]["recording"]["sha256"]
        _require(audit.get("recording_sha256") == recording_sha and _text(audit.get("basis")), "session_audit_recording_mismatch")
        replay = _read(audit.get("native_replay"), base)
        _require(isinstance(replay, dict) and replay.get("schema") == "semantic-session-replay-v1"
                 and replay.get("session_id") == sid and replay.get("recording_sha256") == recording_sha
                 and replay.get("complete") is True and replay.get("pending_outcome", "missing") is None,
                 "original_native_replay_incomplete")
        instance, native = replay.get("instance_id"), replay.get("native_manifest")
        _require(_text(instance) and instance not in seen_instances and replay.get("fresh_instance") is True,
                 "original_native_replay_not_fresh")
        seen_instances.add(instance)
        _require(isinstance(native, dict) and native.get("backend") == "native-playwright-browser"
                 and native.get("sourceSha256") == manifest["source_sha256"] and native.get("instance_id") == instance
                 and native.get("browser_transport") == manifest["browser_transport"], "original_native_replay_provenance_mismatch")
        _require(replay.get("source_sha256") == manifest["source_sha256"]
                 and replay.get("initial_state_sha256") == manifest["initial_state_sha256"]
                 and replay.get("initial_observation") == session["observations"]["e0"][1]["observation"], "original_native_replay_initial_mismatch")
        expected = [{"number": number, "action": event["action"], "observation": event["observation"]}
                    for number, (_, event) in session["controls"].items()]
        _require(replay.get("controls") == expected, "original_native_replay_transition_mismatch")
        bindings = audit.get("request_bindings")
        _require(isinstance(bindings, list) and len(bindings) == len(session["requests"]), "request_audit_bindings_incomplete")
        for number, binding in enumerate(bindings, 1):
            _require(isinstance(binding, dict) and binding.get("request_number") == number
                     and _text(binding.get("ledger_id")) and binding["ledger_id"] in ledger_rows
                     and binding["ledger_id"] not in used, "request_audit_binding_invalid")
            identifier = binding["ledger_id"]
            used.add(identifier)
            cost_row = ledger_rows[identifier]
            request, response = session["requests"][number][1], session["responses"][number][1]
            payload = response["response"]
            _require(cost_row.get("status") == "completed" and cost_row.get("usage_known") is True
                     and cost_row.get("exceeded_reservation") is False and cost_row.get("usage") == payload["usage"],
                     "request_ledger_usage_unknown_or_mismatch")
            packet = request["packet"]
            public = {key: value for key, value in packet.items() if key != "response_schema"}
            prompt_bytes = len(_json([{"role": "user", "content": _json(public)}]).encode())
            schema_bytes = len(_json(packet["response_schema"]).encode())
            _require(cost_row.get("model") == MODEL_CONFIG["model"] and cost_row.get("prompt_bytes") == prompt_bytes
                     and cost_row.get("schema_bytes") == schema_bytes and cost_row.get("max_tokens") == 8192
                     and cost_row.get("input_usd_per_million") == 1.0 and cost_row.get("output_usd_per_million") == 3.0,
                     "request_reservation_configuration_mismatch")
            expected_reservation = (prompt_bytes + schema_bytes + 4096 + 8192 * 3) / 1_000_000
            usage = payload["usage"]
            expected_cost = (usage["prompt_tokens"] + usage["completion_tokens"] * 3) / 1_000_000
            for field, expected_value in (("reservation_usd", expected_reservation), ("known_cost_usd", expected_cost),
                                          ("accounted_cost_usd", expected_cost)):
                _require(_finite(cost_row.get(field)) and math.isclose(cost_row[field], expected_value, abs_tol=1e-10),
                         "request_reservation_or_cost_mismatch")
            _require(expected_cost <= expected_reservation and math.isclose(payload["estimated_cost_usd"], expected_cost, abs_tol=1e-10),
                     "request_cost_exceeds_reservation_or_response")
            reserved, settled = _time(cost_row.get("reserved_at")), _time(cost_row.get("settled_at"))
            _require(_time(request.get("timestamp")) <= reserved <= settled <= _time(response.get("timestamp")), "request_not_reserved_before_response")
            reservation_times.append(reserved)
            total_cost += expected_cost
    _require(used == set(ledger_rows) and total_cost <= 2.0, "unaccounted_request_or_budget_exceeded")
    _allowance_audit(review, manifest, base, total_cost, reservation_times)
    return {"native_replays": len(audits), "request_rows": len(used), "verified_known_cost_usd": total_cost}


def _allowance_audit(review, manifest, base, total_cost, reservation_times):
    snapshots = [_read(review.get(name), base) for name in ("allowance_before", "allowance_reserved", "allowance_after")]
    before, reserved, after = snapshots
    identifier = review.get("invocation_id")
    _require(_text(identifier) and reservation_times, "invocation_identity_or_requests_missing")
    for snapshot in snapshots:
        _require(isinstance(snapshot, dict) and snapshot.get("version") == "dashboard-allowance-v1"
                 and snapshot.get("cap_usd") == 10.0 and isinstance(snapshot.get("jobs"), dict)
                 and isinstance(snapshot.get("validation_studies"), dict)
                 and snapshot.get("supplemental_rounds", {}).get("adaptive-20261014", {}).get("cap_usd") == 15.0,
                 "shared_allowance_schema_or_original_grant_changed")
    _require(identifier not in before["jobs"] and set(reserved["jobs"]) == set(after["jobs"]) == set(before["jobs"]) | {identifier},
             "invocation_reused_or_extra_dispatch")
    study_id = manifest.get("study_id")
    _require(_text(study_id) and study_id not in before["validation_studies"], "single_use_study_reused")
    for snapshot in (reserved, after):
        _require(all(snapshot["jobs"].get(key) == row for key, row in before["jobs"].items()), "old_allowance_hold_or_job_changed")
        _require({k: v for k, v in snapshot.items() if k not in {"jobs", "validation_studies"}} ==
                 {k: v for k, v in before.items() if k not in {"jobs", "validation_studies"}}, "old_grant_or_round_changed")
        _require(snapshot["validation_studies"] == {**before["validation_studies"], study_id: {"cap_usd": 2.0, "stage_caps_usd": {"single": 2.0}}},
                 "old_study_changed_or_new_cap_invalid")
        row = snapshot["jobs"][identifier]
        _require(row.get("reservation_usd") == 2.0 and row.get("authorization_round") == "adaptive-20261014"
                 and row.get("validation_study") == study_id and row.get("study_stage") == "single", "invocation_hold_or_round_invalid")
        _require(all(_finite(job.get("accounted_usd")) for job in snapshot["jobs"].values()), "allowance_cost_invalid")
        occupied = sum(job["accounted_usd"] for job in snapshot["jobs"].values() if job.get("authorization_round") == "adaptive-20261014")
        _require(occupied <= 15.0, "fixed_round_cap_exceeded")
    held, final = reserved["jobs"][identifier], after["jobs"][identifier]
    _require(held.get("status") == "reserved" and held.get("usage_known") is False and held.get("accounted_usd") == 2.0
             and _time(held.get("reserved_at")) <= min(reservation_times), "full_invocation_hold_not_before_http")
    _require(final.get("status") == "completed" and final.get("usage_known") is True
             and _finite(final.get("accounted_usd")) and math.isclose(final["accounted_usd"], total_cost, abs_tol=1e-8),
             "invocation_unknown_or_settlement_mismatch")


def _value(observation, path):
    _require(isinstance(path, list) and path and all(isinstance(p, str) or type(p) is int for p in path), "assertion_path_invalid")
    value = observation
    try:
        for part in path:
            value = value[part]
    except (TypeError, KeyError, IndexError):
        return False, None
    return True, value


def evaluate_expectation(observation, expectation):
    """Evaluate a generic public-observation assertion; no defect-specific rule."""
    _require(isinstance(expectation, dict), "expectation_invalid")
    exists, actual = _value(observation, expectation.get("path"))
    op, expected = expectation.get("op"), expectation.get("value")
    _require(op in {"equals", "contains", "not_equals", "exists"}, "assertion_operator_invalid")
    if op == "exists":
        _require(type(expected) is bool, "exists_assertion_requires_boolean")
        return exists is expected
    if not exists:
        return False
    if op == "equals":
        return type(actual) is type(expected) and actual == expected
    if op == "not_equals":
        return not (type(actual) is type(expected) and actual == expected)
    return isinstance(actual, (str, list)) and isinstance(expected, str) and expected in actual


def _regression_definition(definition, sessions):
    _require(isinstance(definition, dict) and definition.get("schema") == REGRESSION_SCHEMA, "regression_schema_invalid")
    for variant in ("case", "normal_control"):
        plan = definition.get(variant)
        _require(isinstance(plan, dict) and isinstance(plan.get("controls"), list) and 1 <= len(plan["controls"]) <= 24,
                 "regression_controls_missing")
        _require(all(isinstance(action, dict) and action.get("kind", action.get("type")) in {"click", "input", "select", "wait"}
                     for action in plan["controls"]), "regression_non_generic_control")
        expectations = plan.get("expectations")
        _require(isinstance(expectations, list) and expectations, "regression_expectations_missing")
        for expectation in expectations:
            _require(isinstance(expectation, dict) and type(expectation.get("after_control")) is int
                     and 0 <= expectation["after_control"] <= len(plan["controls"]), "regression_assertion_index_invalid")
            basis = expectation.get("basis")
            _require(isinstance(basis, dict) and basis.get("session_id") in sessions and _text(basis.get("reason")), "expectation_public_basis_missing")
            session = sessions[basis["session_id"]]
            _citations(basis.get("evidence"), session["observations"])
            evaluate_expectation({}, expectation)
    origin = definition.get("origin")
    _require(isinstance(origin, dict) and origin.get("session_id") in sessions, "regression_origin_missing")
    original = sessions[origin["session_id"]]
    numbers = origin.get("controls")
    _require(isinstance(numbers, list) and numbers == list(range(1, len(numbers) + 1))
             and numbers and all(n in original["controls"] for n in numbers), "regression_origin_prefix_invalid")
    _require([original["controls"][n][1]["action"] for n in numbers] == definition["case"]["controls"], "regression_not_original_sequence")
    observed = {0: original["observations"]["e0"][1]["observation"]}
    observed.update({n: original["controls"][n][1]["observation"] for n in numbers})
    outcomes = [evaluate_expectation(observed[e["after_control"]], e) for e in definition["case"]["expectations"]]
    _require(any(value is False for value in outcomes), "regression_origin_does_not_fail")
    return outcomes


def _trace(ref, base, definition, definition_sha, variant, source_sha, initial_sha, seen_instances):
    trace = _read(ref, base)
    _require(isinstance(trace, dict) and trace.get("schema") == TRACE_SCHEMA and trace.get("backend") == "native-browser"
             and trace.get("complete") is True and trace.get("pending_outcome", "missing") is None, "native_trace_incomplete_or_stub")
    _require(trace.get("source_sha256") == source_sha and trace.get("initial_state_sha256") == initial_sha
             and trace.get("regression_sha256") == definition_sha and trace.get("variant") == variant, "trace_binding_invalid")
    instance = trace.get("instance_id")
    _require(_text(instance) and instance not in seen_instances and trace.get("fresh_instance") is True, "reproduction_not_fresh")
    seen_instances.add(instance)
    native_manifest = trace.get("native_manifest")
    _require(isinstance(native_manifest, dict) and native_manifest.get("sourceSha256") == source_sha
             and native_manifest.get("instance_id") == instance
             and native_manifest.get("backend") == "native-playwright-browser", "native_trace_manifest_missing")
    browser = native_manifest.get("browser_transport")
    _require(isinstance(browser, dict) and _text(browser.get("browser")) and isinstance(browser.get("clock"), dict)
             and browser.get("storage") == "fresh-browser-context", "native_browser_configuration_missing")
    _require(_finite(trace.get("elapsed_seconds")) and isinstance(trace.get("initial_observation"), dict), "trace_accounting_invalid")
    _require(_digest(_json(trace["initial_observation"]).encode()) == initial_sha, "trace_initial_state_hash_mismatch")
    controls, plan = trace.get("controls"), definition[variant]
    _require(isinstance(controls, list) and len(controls) == len(plan["controls"]), "trace_controls_incomplete")
    observations = {0: trace["initial_observation"]}
    for number, (actual, expected) in enumerate(zip(controls, plan["controls"]), 1):
        _require(isinstance(actual, dict) and actual.get("number") == number and actual.get("action") == expected
                 and isinstance(actual.get("observation"), dict), "trace_control_mismatch")
        observations[number] = actual["observation"]
    return [evaluate_expectation(observations[e["after_control"]], e) for e in plan["expectations"]]


def _prospective(contribution, session, claim):
    hypothesis_id = contribution.get("hypothesis_id")
    _require(hypothesis_id in session["hypotheses"], "prior_hypothesis_missing")
    index, hypothesis = session["hypotheses"][hypothesis_id]
    _require(claim.get("hypothesis_id") == hypothesis_id, "claim_hypothesis_mismatch")
    first = contribution.get("first_symptom_control")
    _require(type(first) is int and first in session["controls"], "first_symptom_missing")
    first_index = session["controls"][first][0]
    _require(session["controls"][first][1]["evidence_id"] in claim.get("evidence", []), "claim_does_not_cite_first_symptom")
    _require(index < session["starts"][first][0] and type(hypothesis.get("created_before_control")) is int
             and hypothesis["created_before_control"] <= first, "posthoc_hypothesis")
    _require(all(_text(hypothesis.get(field)) for field in ("statement", "falsifier", "predicted_difference", "intervention_plan", "control_plan")), "prospective_design_incomplete")
    clues = hypothesis.get("clues")
    _require(isinstance(clues, list) and clues and all(isinstance(clue, dict) for clue in clues), "public_clues_invalid")
    from .semantic_testing import _pointer
    for clue in clues:
        _citations([clue.get("evidence_id")], session["observations"], before_event=index)
        observed = session["observations"][clue["evidence_id"]][1]["observation"]
        value = _pointer(observed, clue.get("pointer"))
        rendered = value if isinstance(value, str) else _json(value)
        _require(_text(clue.get("quote")) and clue["quote"] in rendered, "public_clue_quote_missing")
    for arm in ("intervention", "control"):
        numbers = contribution.get(arm + "_controls")
        _require(isinstance(numbers, list) and numbers and all(type(n) is int and n in session["controls"] for n in numbers), "experiment_comparison_missing")
        for number in numbers:
            control_index, event = session["controls"][number]
            _require(session["starts"][number][0] > index and event.get("arm") == arm and event.get("hypothesis_id") == hypothesis_id
                     and event.get("model_selected") is True, "experiment_not_model_selected")
    critical = contribution.get("critical_controls")
    _require(isinstance(critical, list) and critical and all(type(n) is int and n in session["controls"] and n <= first for n in critical), "critical_controls_missing")
    _require(all(session["starts"][n][0] > index and session["controls"][n][1].get("model_selected") is True for n in critical), "critical_control_not_prospective_model_choice")
    _require(contribution.get("relevant_before_symptom") is True and contribution.get("expected_difference_supported") is True
             and _text(contribution.get("basis")), "independent_design_judgment_missing")
    completed = [e for i, e in session["assessments"] if i > first_index and e.get("hypothesis_id") == hypothesis_id
                 and e.get("status") in {"retain", "revise", "reject"} and _text(e.get("actual_result")) and _text(e.get("reason"))]
    _require(completed, "hypothesis_result_not_assessed")
    for assessment in completed:
        _citations(assessment.get("evidence"), session["observations"])


def assess_semantic_study(manifest_path, review_path, *, trusted_review_sha256=None):
    """Return FAIL on every missing gate; no incomplete outcome becomes a pass.

    ``trusted_review_sha256`` is an out-of-band trust input. The caller must
    obtain it from the independent reviewer and must not calculate it from an
    untrusted actor-supplied review merely to make this function pass.
    """
    result = {"schema": ANALYSIS_SCHEMA, "decision": "FAIL", "evidence_status": "incomplete_or_invalid",
              "errors": [], "qualifying_defects": [], "regression_ready_defects": [], "sessions": [],
              "scope": "Fixed six-session active semantic testing claim only; learning-route STOP remains unchanged.",
              "trust_limit": "External review pin and reviewer identity are trusted inputs; hashes establish integrity, not truth."}
    try:
        manifest_path, review_path = Path(manifest_path), Path(review_path)
        base = manifest_path.parent
        manifest_raw = manifest_path.read_bytes()
        manifest = json.loads(manifest_raw)
        _require(isinstance(manifest, dict) and manifest.get("schema") == MANIFEST_SCHEMA, "manifest_schema_invalid")
        _require(manifest.get("mode") == "live", "stub_evidence_cannot_support_competitive_claim")
        _require(manifest.get("frozen_before_sessions") is True and manifest.get("limits") == LIMITS, "study_freeze_or_limits_invalid")
        _require(manifest.get("resource_limits") == RESOURCE_LIMITS, "study_resource_limits_invalid")
        _require(isinstance(manifest.get("model_config"), dict)
                 and _json(manifest["model_config"]) == _json(MODEL_CONFIG)
                 and isinstance(manifest.get("browser_transport"), dict), "model_or_browser_configuration_missing")
        known = _read(manifest.get("known_registry"), base)
        policy = _read(manifest.get("baseline_policy"), base)
        _require(known.get("schema") == "semantic-known-defects-v1" and known.get("frozen_before_live") is True, "known_registry_not_frozen")
        known_ids = _index(known.get("issues"), "id", "known_registry_invalid")
        _require(len(known_ids) >= 3 and all(item.get("eligible_as_new") is False for item in known_ids.values()), "known_registry_incomplete")
        _require(known.get("integration_findings_complete") is True, "human_integration_findings_unaccounted")
        _require(policy.get("frozen_before_A") is True and policy.get("generic_controls_only") is True, "baseline_not_frozen")
        rows = _index(manifest.get("sessions"), "session_id", "session_manifest_invalid")
        _require(len(rows) == 6 and Counter(r.get("condition") for r in rows.values()) == {"A": 3, "B": 3}, "six_session_matrix_incomplete")
        _require(len({r.get("instance_id") for r in rows.values()}) == 6 and len({r.get("recording", {}).get("sha256") for r in rows.values()}) == 6, "sessions_reused")
        seeds = [r.get("seed") for r in rows.values() if r.get("condition") == "B"]
        _require(all(type(s) is int for s in seeds) and len(set(seeds)) == 3 and sorted(seeds) == sorted(policy.get("seeds", [])), "baseline_seeds_not_frozen")
        sessions = {identifier: _session(row, manifest, base) for identifier, row in rows.items()}
        result["sessions"] = [{"session_id": identifier, "condition": s["row"]["condition"], **{k: s["summary"][k] for k in ("controls", "requests", "responses", "cost_estimate_usd", "elapsed_seconds")}} for identifier, s in sessions.items()]
        review_raw = review_path.read_bytes()
        _require(_hex(trusted_review_sha256) and _digest(review_raw) == trusted_review_sha256, "independent_review_pin_missing_or_mismatch")
        review = json.loads(review_raw)
        _require(isinstance(review, dict) and review.get("schema") == REVIEW_SCHEMA
                 and review.get("manifest_sha256") == _digest(manifest_raw), "review_manifest_binding_invalid")
        reviewer = review.get("reviewer_id")
        _require(_text(reviewer) and reviewer in manifest.get("trusted_reviewer_ids", [])
                 and reviewer not in {s["row"]["actor_id"] for s in sessions.values()}
                 and review.get("review_method") == "independent-local-native-source", "reviewer_not_independent_or_untrusted")
        expected_hashes = {i: s["row"]["recording"]["sha256"] for i, s in sessions.items()}
        _require(review.get("recording_digests") == expected_hashes and review.get("known_registry_checked_sha256") == manifest["known_registry"]["sha256"], "review_evidence_binding_invalid")
        result["independent_session_audit"] = _session_audits(review, sessions, manifest, base)
        claims = {(identifier, claim_id): claim for identifier, session in sessions.items() for claim_id, (_, claim) in session["claims"].items()}
        judged = {}
        _require(isinstance(review.get("claim_reviews"), list), "all_claim_reviews_missing")
        for item in review["claim_reviews"]:
            _require(isinstance(item, dict) and _text(item.get("session_id")) and _text(item.get("claim_id")), "claim_review_invalid")
            key = (item["session_id"], item["claim_id"])
            _require(key in claims and key not in judged and _text(item.get("basis")), "claim_review_binding_invalid")
            _citations(item.get("evidence"), sessions[key[0]]["observations"])
            _require(set(item["evidence"]) & set(claims[key]["evidence"]), "review_does_not_cite_claim_evidence")
            judged[key] = item
        _require(set(judged) == set(claims), "claims_unreviewed_or_erased")
        bad = [list(key) for key, item in judged.items() if item.get("verdict") != "supported_defect"]
        if bad:
            result["errors"].append("unsupported_final_claims")
            result["unsupported_claims"] = bad
        defects = _index(review.get("defects"), "defect_id", "defect_review_invalid")
        _require(all(item.get("defect_id") in defects for item in judged.values() if item.get("verdict") == "supported_defect"), "supported_claim_without_truth_review")
        for defect_id, defect in defects.items():
            failures = []
            try:
                _require(defect.get("is_new") is True and defect_id not in known_ids and defect.get("known_issue_ids") == [], "defect_known_or_novelty_unreviewed")
                _require(_text(defect.get("novelty_basis")), "novelty_basis_missing")
                _require(defect.get("fixture_artifact_excluded") is True and _text(defect.get("fixture_artifact_basis")), "controlled_fixture_artifact_not_excluded")
                sources = defect.get("source_evidence")
                _require(isinstance(sources, list) and sources, "independent_source_confirmation_missing")
                for source in sources:
                    _read_source(source, base)
                definition_ref = defect.get("regression")
                definition = _read(definition_ref, base)
                _require(definition.get("defect_id") == defect_id and definition.get("initial_state_sha256") == manifest["initial_state_sha256"], "regression_defect_binding_invalid")
                original_outcomes = _regression_definition(definition, sessions)
                definition_sha = _digest(_json(definition).encode())
                instances = {s["row"]["instance_id"] for s in sessions.values()}
                reproductions = defect.get("fresh_reproductions")
                _require(isinstance(reproductions, list) and len(reproductions) == 3, "fresh_reproduction_not_three")
                outcomes = [_trace(ref, base, definition, definition_sha, "case", manifest["source_sha256"], manifest["initial_state_sha256"], instances) for ref in reproductions]
                _require(outcomes[0] == outcomes[1] == outcomes[2] == original_outcomes, "fresh_reproduction_not_three_equivalent_failures")
                patched_sha = defect.get("patched_source_sha256")
                _require(_hex(patched_sha) and patched_sha != manifest["source_sha256"], "patched_source_missing")
                _source_manifest(defect.get("patched_source_manifest"), base, patched_sha)
                _require(all(_trace(defect.get("patched_case"), base, definition, definition_sha, "case", patched_sha, manifest["initial_state_sha256"], instances)), "patched_case_still_fails")
                for field, source_sha in (("original_normal_control", manifest["source_sha256"]), ("patched_normal_control", patched_sha)):
                    _require(all(_trace(defect.get(field), base, definition, definition_sha, "normal_control", source_sha, manifest["initial_state_sha256"], instances)), "normal_control_fails")
                result["regression_ready_defects"].append(defect_id)
                contributing = set()
                contributions = defect.get("contributions")
                _require(isinstance(contributions, list), "contribution_reviews_missing")
                for item in contributions:
                    _require(isinstance(item, dict) and item.get("session_id") in sessions, "contribution_session_invalid")
                    sid, cid = item["session_id"], item.get("claim_id")
                    _require((sid, cid) in judged and judged[sid, cid].get("defect_id") == defect_id
                             and judged[sid, cid].get("verdict") == "supported_defect" and sessions[sid]["row"]["condition"] == "A", "contribution_claim_invalid")
                    _prospective(item, sessions[sid], claims[sid, cid])
                    contributing.add(sid)
                _require(len(contributing) >= 2, "new_defect_not_prospectively_discovered_in_two_A_sessions")
                exposure = _index(defect.get("baseline_exposure"), "session_id", "baseline_exposure_invalid")
                expected_b = {i for i, s in sessions.items() if s["row"]["condition"] == "B"}
                _require(set(exposure) == expected_b, "baseline_raw_evidence_review_incomplete")
                for sid, item in exposure.items():
                    _require(item.get("equivalent_evidence") is False, "baseline_triggered_equivalent_evidence")
                    _require(item.get("reviewed_evidence_ids") == list(sessions[sid]["observations"]) and _text(item.get("basis")), "baseline_full_trajectory_not_reviewed")
                result["qualifying_defects"].append(defect_id)
            except EvidenceError as exc:
                failures.append(str(exc))
            if failures:
                result.setdefault("defect_failures", {})[defect_id] = failures
        result["evidence_status"] = "complete_reviewed"
        if not result["qualifying_defects"]:
            result["errors"].append("no_defect_passes_all_fixed_gates")
        if not result["errors"]:
            result["decision"] = "PASS"
        result["manifest_sha256"] = _digest(manifest_raw)
        result["review_sha256"] = trusted_review_sha256
    except (EvidenceError, OSError, ValueError, TypeError, KeyError, AttributeError) as exc:
        result["errors"].append(str(exc) if isinstance(exc, EvidenceError) else "malformed_or_unreadable_evidence")
    return result


def export_semantic_regression(manifest_path, review_path, defect_id, output_path, *, trusted_review_sha256=None):
    """Export an independently reproduced and repair-checked generic case.

    Export is useful even when the experiment fails another gate. Its metadata
    always retains that FAIL, and never promotes a case into an experiment win.
    """
    assessment = assess_semantic_study(manifest_path, review_path, trusted_review_sha256=trusted_review_sha256)
    _require(defect_id in assessment["regression_ready_defects"], "regression_evidence_not_independently_confirmed")
    manifest_path = Path(manifest_path)
    review = json.loads(Path(review_path).read_bytes())
    defect = next(item for item in review["defects"] if item["defect_id"] == defect_id)
    definition = _read(defect["regression"], manifest_path.parent)
    exported = deepcopy(definition)
    exported["verification"] = {"study_decision": assessment["decision"], "manifest_sha256": assessment["manifest_sha256"],
                                "review_sha256": trusted_review_sha256, "fresh_reproductions": "3/3",
                                "patched_case_passes": True, "original_and_patched_normal_controls_pass": True}
    Path(output_path).write_text(_json(exported) + "\n", encoding="utf-8")
    return exported


def run_semantic_regression(adapter, definition, *, variant="case", source_sha256,
                            initial_state_sha256, instance_id, clock=time.monotonic):
    """Execute exported atomic controls on a caller-created fresh local adapter.

    There is no reset, route preparation, game-specific detection, inference,
    or retry here. The caller controls fresh native-instance construction.
    This runner returns observations and computed assertion results; independent
    review is still required before those results can support a study claim.
    """
    from .browser_adapter import BrowserAdapter, validate_action
    from tools.semantic_study import EditorScopeAdapter
    _require(isinstance(definition, dict) and definition.get("schema") == REGRESSION_SCHEMA
             and variant in {"case", "normal_control"}, "regression_definition_invalid")
    plan = definition.get(variant)
    _require(isinstance(plan, dict) and isinstance(plan.get("controls"), list)
             and 1 <= len(plan["controls"]) <= 24 and isinstance(plan.get("expectations"), list)
             and bool(plan["expectations"]), "regression_plan_invalid")
    actions = [validate_action(action) for action in plan["controls"]]
    _require(_hex(source_sha256) and _hex(initial_state_sha256) and _text(instance_id), "regression_instance_metadata_invalid")
    _require(definition.get("initial_state_sha256") == initial_state_sha256, "regression_definition_initial_state_mismatch")
    native = adapter.adapter if isinstance(adapter, EditorScopeAdapter) else adapter
    is_native = isinstance(native, BrowserAdapter)
    if is_native:
        _require(native.source_manifest.get("sourceSha256") == source_sha256
                 and native.source_manifest.get("instance_id") == instance_id, "regression_native_provenance_mismatch")
    started = clock()
    if is_native:
        native.timeout = min(native.timeout, 300)
    observation = adapter.observe()
    _require(_digest(_json(observation).encode()) == initial_state_sha256, "regression_initial_state_mismatch")
    canonical = {k: deepcopy(v) for k, v in definition.items() if k != "verification"}
    trace = {"schema": TRACE_SCHEMA, "backend": "native-browser" if is_native else "stub",
             "instance_id": instance_id, "fresh_instance": True, "source_sha256": source_sha256,
             "initial_state_sha256": initial_state_sha256, "regression_sha256": _digest(_json(canonical).encode()),
             "variant": variant, "initial_observation": observation, "controls": [], "complete": False,
             "pending_outcome": None, "expectation_results": []}
    if is_native:
        trace["native_manifest"] = deepcopy(native.source_manifest)
    observed = {0: observation}
    try:
        for number, action in enumerate(actions, 1):
            _require(clock() - started < 300, "regression_deadline")
            if is_native:
                native.timeout = min(native.timeout, 300 - (clock() - started))
            trace["pending_outcome"] = number
            observed[number] = adapter.act(deepcopy(action))
            trace["controls"].append({"number": number, "action": action, "observation": observed[number]})
            trace["pending_outcome"] = None
        _require(clock() - started < 300, "regression_deadline")
        for expectation in plan["expectations"]:
            _require(isinstance(expectation, dict) and type(expectation.get("after_control")) is int
                     and expectation["after_control"] in observed, "regression_assertion_index_invalid")
            trace["expectation_results"].append(evaluate_expectation(observed[expectation["after_control"]], expectation))
        trace["complete"] = True
    except Exception as exc:
        trace["failure"] = type(exc).__name__
    trace["elapsed_seconds"] = clock() - started
    return trace


def main(argv=None):
    parser = argparse.ArgumentParser(description="Offline fixed six-session semantic evidence assessment; never runs paid inference.")
    parser.add_argument("manifest")
    parser.add_argument("review")
    parser.add_argument("--trusted-review-sha256", required=True, help="Digest obtained independently from the reviewer, never actor-supplied.")
    parser.add_argument("--export-defect")
    parser.add_argument("--output")
    args = parser.parse_args(argv)
    if args.export_defect:
        if not args.output:
            parser.error("--export-defect requires --output")
        export_semantic_regression(args.manifest, args.review, args.export_defect, args.output,
                                   trusted_review_sha256=args.trusted_review_sha256)
    result = assess_semantic_study(args.manifest, args.review, trusted_review_sha256=args.trusted_review_sha256)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["decision"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
