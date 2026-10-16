"""Synthetic fixtures exercise gates only; none is an actual game discovery."""

from copy import deepcopy
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from autoplay_qa.semantic_analysis import (
    EvidenceError, LIMITS, RESOURCE_LIMITS, MODEL_CONFIG, MANIFEST_SCHEMA, REGRESSION_SCHEMA, REVIEW_SCHEMA,
    TRACE_SCHEMA, assess_semantic_study, evaluate_expectation,
    export_semantic_regression, run_semantic_regression,
)
from autoplay_qa.semantic_testing import run_semantic_session


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def visible(text="Save preserves text.", tick=0):
    return {"kind": "browser-visible-v1", "url": "/", "title": "Fixture",
            "text": text, "controls": [{"id": "save", "tag": "button", "role": "button",
                                         "name": "Save", "value": "", "disabled": False}], "tick": tick}


ACTION = {"kind": "click", "target": "save"}
WAIT = {"kind": "wait", "milliseconds": 0}


class Log:
    def __init__(self):
        self.events = []

    def write(self, kind, **fields):
        timestamp = datetime(2026, 1, 2, tzinfo=timezone.utc) + timedelta(milliseconds=len(self.events))
        self.events.append({"type": kind, "timestamp": timestamp.isoformat(), **deepcopy(fields)})

    def sync(self):
        pass


class Adapter:
    def __init__(self, broken=True):
        self.step = 0
        self.broken = broken

    def observe(self):
        return visible()

    def act(self, action):
        self.step += 1
        return visible("Broken" if self.broken and self.step == 2 else "Save preserves text.", self.step)


def decision(**fields):
    return {"hypotheses": [], "actions": [], "assessments": [], "suspected_bug": [],
            "acknowledged_evidence": ["e0"], "finish": False, **fields}


def model(replies):
    replies = iter(replies)
    return lambda packet: {"decision": next(replies), "usage": {"prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3},
                           "estimated_cost_usd": .000005}


class Fixture:
    """Explicitly forged 'live' fixture to test acceptance logic, never evidence."""
    def __init__(self, base):
        self.base = base
        self.initial = visible()
        source_path = base / "original.js"
        source_path.write_text("// Synthetic source basis for unit tests only.\n")
        (base / "built.js").write_text("// Synthetic built bytes.\n")
        self.source_files = [{"path": "original.js", "sha256": digest(source_path.read_bytes())}]
        self.source = digest(canonical(self.source_files).encode())
        patched_dir = base / "patched"
        patched_dir.mkdir(exist_ok=True)
        (patched_dir / "original.js").write_text("// Synthetic repaired bytes.\n")
        self.patched_files = [{"path": "original.js", "sha256": digest((patched_dir / "original.js").read_bytes())}]
        self.patched = digest(canonical(self.patched_files).encode())
        self.initial_sha = digest(canonical(self.initial).encode())
        known = {"schema": "semantic-known-defects-v1", "frozen_before_live": True,
                 "integration_findings_complete": True,
                 "issues": [{"id": f"known-{i}", "eligible_as_new": False} for i in range(3)]}
        self.manifest = {"schema": MANIFEST_SCHEMA, "study_id": "SYNTHETIC-UNIT-TEST",
                         "mode": "live", "frozen_before_sessions": True, "source_sha256": self.source,
                         "initial_state_sha256": self.initial_sha, "limits": LIMITS, "resource_limits": RESOURCE_LIMITS,
                         "model_config": MODEL_CONFIG,
                         "browser_transport": {"browser": "synthetic-unit-test", "clock": {"mode": "controlled-native-timers"}, "storage": "fresh-browser-context"},
                         "known_registry": self.write("known.json", known),
                         "baseline_policy": self.write("policy.json", {"frozen_before_A": True,
                             "generic_controls_only": True, "seeds": [101, 102, 103]}),
                         "trusted_reviewer_ids": ["independent-fixture-reviewer"], "sessions": []}
        self.recordings = {}
        for condition in ("A", "B"):
            for number in range(1, 4):
                sid = f"{condition}{number}"
                metadata = {"session_id": sid, "source_sha256": self.source,
                    "initial_state_sha256": self.initial_sha, "instance_id": sid, "actor_id": "actor-ultra",
                    "baseline_policy_sha256": self.manifest["baseline_policy"]["sha256"],
                    "known_registry_sha256": self.manifest["known_registry"]["sha256"], "experience_shared": False}
                metadata.update(model_config=self.manifest["model_config"],
                    model_config_sha256=digest(canonical(self.manifest["model_config"]).encode()),
                    browser_transport_sha256=digest(canonical(self.manifest["browser_transport"]).encode()))
                log = Log()
                row = {**metadata, "condition": condition}
                if condition == "A":
                    hypothesis = {"id": "h1", "clues": [{"evidence_id": "e0", "pointer": "/text", "quote": "Save preserves text."}],
                        "statement": "Saving twice may lose text.", "falsifier": "Both saves preserve text.",
                        "intervention_plan": "Save twice.", "control_plan": "Save once.",
                        "predicted_difference": "Text remains after one save but disappears after two."}
                    first = decision(hypotheses=[hypothesis], actions=[
                        {"action": ACTION, "arm": "control", "hypothesis_id": "h1"},
                        {"action": ACTION, "arm": "intervention", "hypothesis_id": "h1"}])
                    claims = [{"id": "c1", "title": "Saving loses text", "claim": "Second save loses text",
                               "evidence": ["e2"], "hypothesis_id": "h1"}] if number < 3 else []
                    final = decision(finish=True, acknowledged_evidence=["e2"], suspected_bug=claims,
                        assessments=[{"hypothesis_id": "h1", "status": "retain", "actual_result": "Second save lost text",
                                      "evidence": ["e1", "e2"], "reason": "The control preserved it."}])
                    summary = run_semantic_session(Adapter(), model([first, final]), log, condition="A", session_metadata=metadata)
                else:
                    trajectory = {"initial_observation": self.initial,
                        "transitions": [{"action": WAIT, "observation": visible(tick=1)}],
                        "elapsed_seconds": 0.01, "policy_sha256": self.manifest["baseline_policy"]["sha256"], "seed": 100 + number}
                    row["seed"] = 100 + number
                    row["baseline_trajectory"] = self.write(f"{sid}-trajectory.json", trajectory)
                    summary = run_semantic_session(None, model([decision(finish=True, acknowledged_evidence=["e0", "e1"])]), log,
                        condition="B", frozen_trajectory=trajectory, session_metadata=metadata)
                if not summary["complete"]:
                    raise AssertionError(log.events)
                self.recordings[sid] = log.events
                row["recording"] = self.write(f"{sid}.jsonl", log.events, lines=True)
                self.manifest["sessions"].append(row)
        basis = {"session_id": "A1", "evidence": ["e0"], "reason": "Visible save preservation promise."}
        self.definition = {"schema": REGRESSION_SCHEMA, "defect_id": "new-loss", "initial_state_sha256": self.initial_sha,
            "origin": {"session_id": "A1", "controls": [1, 2]},
            "case": {"controls": [ACTION, ACTION], "expectations": [{"after_control": 2, "path": ["text"],
                     "op": "equals", "value": "Save preserves text.", "basis": basis}]},
            "normal_control": {"controls": [ACTION], "expectations": [{"after_control": 1, "path": ["text"],
                     "op": "equals", "value": "Save preserves text.", "basis": basis}]}}
        source_path = base / "original.js"
        source_path.write_text("// Synthetic source basis for unit tests only.\n")
        self.defect = {"defect_id": "new-loss", "is_new": True, "known_issue_ids": [], "novelty_basis": "Distinct synthetic cause.",
            "fixture_artifact_excluded": True, "fixture_artifact_basis": "Synthetic example is unrelated to timing; unit-test claim only.",
            "source_evidence": [{"path": "original.js", "sha256": digest(source_path.read_bytes()), "line_start": 1,
                                 "line_end": 1, "explanation": "Synthetic source support."}],
            "regression": self.write("regression.json", self.definition), "patched_source_sha256": self.patched,
            "patched_source_manifest": self.write("patched/source-manifest.json", {
                "format": "mars-browser-source-v1", "sourceFiles": self.patched_files, "buildFiles": self.patched_files,
                "sourceSha256": self.patched, "buildSha256": self.patched, "sourceUnmodified": True, "defectsInjected": False}),
            "fresh_reproductions": [self.trace(f"repro-{i}", "case", self.source, broken=True) for i in range(3)],
            "patched_case": self.trace("patched-case", "case", self.patched),
            "original_normal_control": self.trace("original-normal", "normal_control", self.source),
            "patched_normal_control": self.trace("patched-normal", "normal_control", self.patched),
            "contributions": [{"session_id": sid, "claim_id": "c1", "hypothesis_id": "h1", "first_symptom_control": 2,
                               "critical_controls": [2], "control_controls": [1], "intervention_controls": [2],
                               "relevant_before_symptom": True, "expected_difference_supported": True,
                               "basis": "Prior hypothesis, control and selected intervention."} for sid in ("A1", "A2")],
            "baseline_exposure": [{"session_id": f"B{i}", "equivalent_evidence": False,
                                   "reviewed_evidence_ids": ["e0", "e1"], "basis": "Full original trace inspected."} for i in range(1, 4)]}
        self.review = {"schema": REVIEW_SCHEMA, "reviewer_id": "independent-fixture-reviewer",
                       "review_method": "independent-local-native-source",
                       "known_registry_checked_sha256": self.manifest["known_registry"]["sha256"],
                       "claim_reviews": [{"session_id": sid, "claim_id": "c1", "verdict": "supported_defect", "defect_id": "new-loss",
                                          "evidence": ["e2"], "basis": "Synthetic independent source/reproduction check."} for sid in ("A1", "A2")],
                       "defects": [self.defect]}
        self.save()

    def write(self, name, value, *, lines=False):
        raw = (("\n".join(canonical(e) for e in value) if lines else canonical(value)) + "\n").encode()
        (self.base / name).write_bytes(raw)
        return {"path": name, "sha256": digest(raw)}

    def trace(self, instance, variant, source, broken=False):
        plan = self.definition[variant]
        trace = {"schema": TRACE_SCHEMA, "backend": "native-browser", "fresh_instance": True, "instance_id": instance,
                 "source_sha256": source, "initial_state_sha256": self.initial_sha, "regression_sha256": digest(canonical(self.definition).encode()),
                 "native_manifest": {"sourceSha256": source, "instance_id": instance, "backend": "native-playwright-browser",
                     "browser_transport": {"browser": "synthetic-unit-test", "clock": {"mode": "controlled-native-timers"}, "storage": "fresh-browser-context"}},
                 "variant": variant, "initial_observation": self.initial, "complete": True, "pending_outcome": None,
                 "elapsed_seconds": .01, "controls": [{"number": n, "action": action,
                     "observation": visible("Broken" if broken and n == 2 else "Save preserves text.", n)}
                     for n, action in enumerate(plan["controls"], 1)]}
        return self.write(instance + ".json", trace)

    def save(self):
        for row in self.manifest["sessions"]:
            row["recording"] = self.write(row["recording"]["path"], self.recordings[row["session_id"]], lines=True)
        manifest_ref = self.write("manifest.json", self.manifest)
        self.review["manifest_sha256"] = manifest_ref["sha256"]
        self.review["recording_digests"] = {r["session_id"]: r["recording"]["sha256"] for r in self.manifest["sessions"]}
        if "session_audits" not in self.review:
            self.make_session_audits()
        for audit in self.review["session_audits"]:
            if audit["session_id"] not in self.review["recording_digests"]:
                continue
            audit["recording_sha256"] = self.review["recording_digests"][audit["session_id"]]
            ref = audit["native_replay"]
            replay = json.loads((self.base / ref["path"]).read_bytes())
            replay["recording_sha256"] = audit["recording_sha256"]
            audit["native_replay"] = self.write(ref["path"], replay)
        self.pin = self.write("review.json", self.review)["sha256"]

    def make_session_audits(self):
        built = [{"path": "built.js", "sha256": digest((self.base / "built.js").read_bytes())}]
        self.review["source_manifest"] = self.write("source-manifest.json", {
            "format": "mars-browser-source-v1", "sourceSha256": self.source,
            "sourceFiles": self.source_files, "buildFiles": built, "buildSha256": digest(canonical(built).encode()),
            "sourceUnmodified": True, "defectsInjected": False})
        ledger = {"version": "dashboard-cost-v1", "cap_usd": 2.0, "requests": []}
        audits = []
        for row in self.manifest["sessions"]:
            sid = row["session_id"]
            events = self.recordings[sid]
            instance = "replay-" + sid
            replay = {"schema": "semantic-session-replay-v1", "session_id": sid,
                "recording_sha256": row["recording"]["sha256"], "complete": True, "pending_outcome": None,
                "instance_id": instance, "fresh_instance": True, "source_sha256": self.source,
                "initial_state_sha256": self.initial_sha, "initial_observation": self.initial,
                "native_manifest": {"backend": "native-playwright-browser", "sourceSha256": self.source,
                                    "instance_id": instance, "browser_transport": self.manifest["browser_transport"]},
                "controls": [{"number": e["control_number"], "action": e["action"], "observation": e["observation"]}
                             for e in events if e["type"] == "semantic_transition"]}
            bindings = []
            requests = [e for e in events if e["type"] == "semantic_request"]
            responses = [e for e in events if e["type"] == "semantic_response"]
            for number, (request, response) in enumerate(zip(requests, responses), 1):
                packet = request["packet"]
                public = {key: value for key, value in packet.items() if key != "response_schema"}
                prompt_bytes = len(canonical([{"role": "user", "content": canonical(public)}]).encode())
                schema_bytes = len(canonical(packet["response_schema"]).encode())
                identifier = f"request-{sid}-{number}"
                usage = response["response"]["usage"]
                cost = response["response"]["estimated_cost_usd"]
                ledger["requests"].append({"id": identifier, "status": "completed", "usage_known": True,
                    "exceeded_reservation": False, "usage": usage, "model": MODEL_CONFIG["model"],
                    "prompt_bytes": prompt_bytes, "schema_bytes": schema_bytes, "max_tokens": 8192,
                    "input_usd_per_million": 1.0, "output_usd_per_million": 3.0,
                    "reservation_usd": (prompt_bytes + schema_bytes + 4096 + 8192 * 3) / 1_000_000,
                    "known_cost_usd": cost, "accounted_cost_usd": cost,
                    "reserved_at": request["timestamp"], "settled_at": response["timestamp"]})
                bindings.append({"request_number": number, "ledger_id": identifier})
            audits.append({"session_id": sid, "recording_sha256": row["recording"]["sha256"],
                           "native_replay": self.write(instance + ".json", replay), "request_bindings": bindings,
                           "basis": "Synthetic independent replay/source/request audit."})
        self.review["request_ledger"] = self.write("request-ledger.json", ledger)
        self.review["session_audits"] = audits
        before = {"version": "dashboard-allowance-v1", "cap_usd": 10.0,
                  "jobs": {"old-unknown": {"authorization_round": "adaptive-20261014", "accounted_usd": 1.0, "usage_known": False}},
                  "supplemental_rounds": {"adaptive-20261014": {"cap_usd": 15.0}},
                  "validation_studies": {"old-study": {"cap_usd": 1.0}}}
        reserved = deepcopy(before)
        reserved["validation_studies"][self.manifest["study_id"]] = {"cap_usd": 2.0, "stage_caps_usd": {"single": 2.0}}
        reserved["jobs"]["synthetic-invocation"] = {"authorization_round": "adaptive-20261014", "reservation_usd": 2.0,
            "accounted_usd": 2.0, "usage_known": False, "status": "reserved", "study_stage": "single",
            "validation_study": self.manifest["study_id"], "reserved_at": "2026-01-01T00:00:00+00:00"}
        after = deepcopy(reserved)
        after["jobs"]["synthetic-invocation"].update(usage_known=True, status="completed",
            accounted_usd=sum(r["accounted_cost_usd"] for r in ledger["requests"]))
        self.review["invocation_id"] = "synthetic-invocation"
        for name, snapshot in (("allowance_before", before), ("allowance_reserved", reserved), ("allowance_after", after)):
            self.review[name] = self.write(name + ".json", snapshot)

    def assess(self, pin=True):
        return assess_semantic_study(self.base / "manifest.json", self.base / "review.json",
                                     trusted_review_sha256=self.pin if pin else None)


class SemanticAnalysisTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.fixture = Fixture(Path(self.tmp.name))

    def assertFails(self, substring=None):
        self.fixture.save()
        result = self.fixture.assess()
        self.assertEqual(result["decision"], "FAIL", result)
        if substring:
            self.assertIn(substring, canonical(result))
        return result

    def test_complete_synthetic_gate_and_export(self):
        result = self.fixture.assess()
        self.assertEqual(result["decision"], "PASS", result)
        self.assertEqual(result["qualifying_defects"], ["new-loss"])
        output = self.fixture.base / "export.json"
        exported = export_semantic_regression(self.fixture.base / "manifest.json", self.fixture.base / "review.json", "new-loss", output,
                                               trusted_review_sha256=self.fixture.pin)
        self.assertEqual(exported["case"]["controls"], [ACTION, ACTION])
        self.assertEqual(exported["verification"]["study_decision"], "PASS")

    def test_partial_unknown_pending_and_missing_session_fail(self):
        for field, value in (("complete", False), ("unknown_usage", True), ("pending_outcome", 2),
                             ("usage_complete", False), ("elapsed_seconds", 301)):
            with self.subTest(field=field):
                old = self.fixture.recordings["A3"][-1][field]
                self.fixture.recordings["A3"][-1][field] = value
                self.assertFails()
                self.fixture.recordings["A3"][-1][field] = old
        self.fixture.manifest["sessions"].pop()
        self.assertFails("six_session_matrix_incomplete")

    def test_b_triggered_without_any_claim_still_fails(self):
        self.fixture.defect["baseline_exposure"][0]["equivalent_evidence"] = True
        result = self.assertFails("baseline_triggered_equivalent_evidence")
        self.assertEqual(result["regression_ready_defects"], ["new-loss"])
        output = self.fixture.base / "failed-study-case.json"
        exported = export_semantic_regression(self.fixture.base / "manifest.json", self.fixture.base / "review.json", "new-loss", output,
                                               trusted_review_sha256=self.fixture.pin)
        self.assertEqual(exported["verification"]["study_decision"], "FAIL")

    def test_missing_b_full_raw_exposure_review_fails(self):
        self.fixture.defect["baseline_exposure"][0]["reviewed_evidence_ids"] = ["e1"]
        self.assertFails("baseline_full_trajectory_not_reviewed")

    def test_known_or_integration_finding_is_not_new(self):
        self.fixture.defect["known_issue_ids"] = ["known-0"]
        self.assertFails("defect_known_or_novelty_unreviewed")

    def test_one_a_discovery_is_insufficient(self):
        self.fixture.defect["contributions"].pop()
        self.assertFails("new_defect_not_prospectively_discovered_in_two_A_sessions")

    def test_posthoc_hypothesis_fails(self):
        hypothesis = next(e for e in self.fixture.recordings["A1"] if e["type"] == "semantic_hypothesis")
        hypothesis["created_before_control"] = 3
        self.assertFails("posthoc_hypothesis")

    def test_model_selected_flag_does_not_replace_raw_decision(self):
        events = self.fixture.recordings["A1"]
        for event in events:
            if event["type"] in {"semantic_control_started", "semantic_transition"} and event["control_number"] == 2:
                event["action"] = WAIT
        self.assertFails("executed_controls_differ_from_model_choice")

    def test_all_claims_require_review_and_false_claims_cannot_be_dropped(self):
        self.fixture.review["claim_reviews"].pop()
        self.assertFails("claims_unreviewed_or_erased")

    def test_supported_success_plus_false_claim_fails_quality(self):
        self.fixture.review["claim_reviews"][0]["verdict"] = "unsupported"
        self.assertFails("unsupported_final_claims")

    def test_raw_claim_cannot_be_deleted_from_candidates_and_summary(self):
        events = self.fixture.recordings["A1"]
        events[:] = [e for e in events if e["type"] != "semantic_candidate"]
        events[-1]["final_claims"] = []
        self.assertFails("raw_claim_inventory_incomplete")

    def test_missing_pin_or_actor_reviewer_fails(self):
        self.assertEqual(self.fixture.assess(pin=False)["decision"], "FAIL")
        self.fixture.review["reviewer_id"] = "actor-ultra"
        self.fixture.manifest["trusted_reviewer_ids"] = ["actor-ultra"]
        self.assertFails("reviewer_not_independent_or_untrusted")

    def test_review_and_recording_tampering_fail(self):
        path = self.fixture.base / "review.json"
        path.write_bytes(path.read_bytes() + b" ")
        self.assertIn("independent_review_pin_missing_or_mismatch", self.fixture.assess()["errors"])
        self.fixture.save()
        path = self.fixture.base / "A1.jsonl"
        path.write_bytes(path.read_bytes() + b" ")
        self.assertIn("artifact_hash_mismatch", self.fixture.assess()["errors"])

    def test_stub_study_and_stub_native_witnesses_never_pass(self):
        self.fixture.manifest["mode"] = "stub"
        self.assertFails("stub_evidence_cannot_support_competitive_claim")
        self.fixture.manifest["mode"] = "live"
        ref = self.fixture.defect["fresh_reproductions"][0]
        trace = json.loads((self.fixture.base / ref["path"]).read_bytes())
        trace["backend"] = "stub"
        self.fixture.defect["fresh_reproductions"][0] = self.fixture.write(ref["path"], trace)
        self.assertFails("native_trace_incomplete_or_stub")

    def test_three_reproductions_must_be_fresh_and_equivalent(self):
        self.fixture.defect["fresh_reproductions"][1] = self.fixture.defect["fresh_reproductions"][0]
        self.assertFails("reproduction_not_fresh")

    def test_patched_case_must_pass_and_normal_control_must_survive(self):
        self.fixture.defect["patched_case"] = self.fixture.trace("bad-fix", "case", self.fixture.patched, broken=True)
        self.assertFails("patched_case_still_fails")

    def test_controlled_fixture_does_not_self_establish_real_bug(self):
        del self.fixture.defect["fixture_artifact_excluded"]
        self.assertFails("controlled_fixture_artifact_not_excluded")

    def test_actor_public_input_history_cannot_drop_an_observation(self):
        request = next(e for e in self.fixture.recordings["A1"] if e["type"] == "semantic_request" and e["request_number"] == 2)
        request["packet"]["evidence"].pop(1)
        self.assertFails("actor_evidence_page_gap")

    def test_cumulative_resource_cap_is_verified_from_original_packets(self):
        self.fixture.recordings["A1"][-1]["total_input_bytes"] = 1
        self.assertFails("cumulative_input_mismatch_or_overflow")

    def test_late_prediction_after_control_start_is_posthoc(self):
        events = self.fixture.recordings["A1"]
        hypothesis = next(e for e in events if e["type"] == "semantic_hypothesis")
        events.remove(hypothesis)
        start = next(i for i, e in enumerate(events) if e["type"] == "semantic_control_started" and e["control_number"] == 2)
        events.insert(start + 1, hypothesis)
        self.assertFails("posthoc_hypothesis")

    def test_independent_original_native_replay_is_mandatory(self):
        audit = self.fixture.review["session_audits"][0]
        ref = audit["native_replay"]
        replay = json.loads((self.fixture.base / ref["path"]).read_bytes())
        replay["controls"][-1]["observation"]["text"] = "fabricated replay"
        audit["native_replay"] = self.fixture.write(ref["path"], replay)
        self.assertFails("original_native_replay_transition_mismatch")

    def test_billing_audit_recomputes_reservation_and_rejects_unknown_or_extra(self):
        for mutation, expected in (("reservation", "request_reservation_or_cost_mismatch"),
                                   ("unknown", "request_ledger_usage_unknown_or_mismatch"),
                                   ("extra", "unaccounted_request_or_budget_exceeded"),
                                   ("late", "request_not_reserved_before_response")):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as directory:
                fixture = Fixture(Path(directory))
                ref = fixture.review["request_ledger"]
                ledger = json.loads((fixture.base / ref["path"]).read_bytes())
                if mutation == "reservation": ledger["requests"][0]["reservation_usd"] = 0
                elif mutation == "unknown": ledger["requests"][0]["usage_known"] = False
                elif mutation == "late": ledger["requests"][0]["reserved_at"] = "2030-01-01T00:00:00Z"
                else:
                    extra = deepcopy(ledger["requests"][0]); extra["id"] = "unexpected-request"
                    ledger["requests"].append(extra)
                fixture.review["request_ledger"] = fixture.write(ref["path"], ledger)
                fixture.save()
                self.assertIn(expected, fixture.assess()["errors"])

    def test_old_allowance_holds_cannot_be_released_or_full_hold_omitted(self):
        for mutation, expected in (("old", "old_allowance_hold_or_job_changed"),
                                   ("hold", "full_invocation_hold_not_before_http")):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as directory:
                fixture = Fixture(Path(directory))
                name = "allowance_after" if mutation == "old" else "allowance_reserved"
                ref = fixture.review[name]
                snapshot = json.loads((fixture.base / ref["path"]).read_bytes())
                if mutation == "old": snapshot["jobs"]["old-unknown"]["accounted_usd"] = 0
                else: snapshot["jobs"]["synthetic-invocation"]["accounted_usd"] = .1
                fixture.review[name] = fixture.write(ref["path"], snapshot)
                fixture.save()
                self.assertIn(expected, fixture.assess()["errors"])

    def test_source_and_patched_source_bytes_are_verified(self):
        (self.fixture.base / "patched/original.js").write_text("Changed after review.\n")
        self.assertFails("source_or_build_bytes_changed")

    def test_baseline_trajectory_hash_and_controls_are_bound(self):
        row = next(r for r in self.fixture.manifest["sessions"] if r["session_id"] == "B1")
        ref = row["baseline_trajectory"]
        trajectory = json.loads((self.fixture.base / ref["path"]).read_bytes())
        trajectory["transitions"][0]["action"] = ACTION
        row["baseline_trajectory"] = self.fixture.write(ref["path"], trajectory)
        self.assertFails("B_raw_trajectory_mismatch")

    def test_generic_runner_executes_no_model_and_retains_failure_evidence(self):
        with patch("urllib.request.OpenerDirector.open", side_effect=AssertionError("No inference")):
            trace = run_semantic_regression(Adapter(), self.fixture.definition,
                source_sha256=self.fixture.source, initial_state_sha256=self.fixture.initial_sha, instance_id="fresh-stub")
        self.assertTrue(trace["complete"])
        self.assertEqual(trace["expectation_results"], [False])
        self.assertEqual(trace["backend"], "stub")
        self.assertEqual(len(trace["controls"]), 2)

    def test_assertions_are_generic_and_fail_missing_paths(self):
        self.assertFalse(evaluate_expectation({}, {"path": ["text"], "op": "equals", "value": "x"}))
        self.assertFalse(evaluate_expectation({"a": True}, {"path": ["a"], "op": "equals", "value": 1}))
        self.assertTrue(evaluate_expectation({"text": "abc"}, {"path": ["text"], "op": "contains", "value": "bc"}))
        with self.assertRaises(EvidenceError):
            evaluate_expectation({}, {"path": [], "op": "eval", "value": "code"})


if __name__ == "__main__":
    unittest.main()
