"""Semantic experiment contracts use fake models and fake UI only."""

from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from autoplay_qa.semantic_testing import (
    SemanticContractError, SemanticLimits, public_observation,
    run_semantic_session, semantic_schema, validate_atomic_action, _diff,
)
from autoplay_qa.session import SessionLog


def observation(tick=0, value="1"):
    return {"kind": "browser-visible-v1", "url": "/", "title": "Editor",
            "text": "Edit a function, then run. Save restores your edited value.",
            "controls": [
                {"id": "edit", "tag": "input", "role": "textbox", "name": "Parameter", "value": value, "disabled": False},
                {"id": "run", "tag": "button", "role": "button", "name": "Run", "value": "", "disabled": False}],
            "tick": tick}


def decision(**changes):
    value = {"hypotheses": [], "actions": [], "assessments": [], "suspected_bug": [],
             "acknowledged_evidence": [], "finish": False}
    value.update(changes)
    return value


def hypothesis(name="h1"):
    return {"id": name, "clues": [{"evidence_id": "e0", "pointer": "/text", "quote": "Save restores your edited value."}],
            "statement": "An edited parameter should remain after save and restore.",
            "falsifier": "Restoring the file displays a different value.",
            "intervention_plan": "Edit and save, then restore the visible value.",
            "control_plan": "Save and restore the original value.",
            "predicted_difference": "Only the deliberately edited value differs; both restore their saved values."}


def action(arm="explore", name=None, value=None):
    control = {"kind": "click", "target": "run"} if value is None else {"kind": "input", "target": "edit", "value": value}
    return {"action": control, "arm": arm, "hypothesis_id": name}


def assessment(status="retain", refs=None, name="h1"):
    return {"hypothesis_id": name, "status": status, "actual_result": "The displayed value was recorded.",
            "evidence": refs or ["e1", "e2"], "reason": "Compare the public observations."}


def claim(name="c1", refs=None, hypothesis_id="h1"):
    return {"id": name, "title": "Tentative parameter issue", "claim": "The observed display might contradict the visible help.",
            "evidence": refs or ["e2"], "hypothesis_id": hypothesis_id}


def response(value, **extra):
    return {"decision": value, "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            "estimated_cost_usd": .0001, **extra}


class FakeLog:
    def __init__(self):
        self.rows = []
        self.syncs = 0

    def write(self, kind, **fields):
        self.rows.append({"type": kind, **deepcopy(fields)})

    def sync(self):
        self.syncs += 1


class FakeAdapter:
    def __init__(self, hook=None):
        self.current = observation()
        self.actions = []
        self.hook = hook
        self.timeout = 20.0

    def observe(self):
        return deepcopy(self.current)

    def act(self, value):
        if self.hook:
            self.hook(value)
        self.actions.append(deepcopy(value))
        self.current["tick"] += 1
        if value["kind"] == "input":
            self.current["controls"][0]["value"] = value["value"]
        return deepcopy(self.current)


class FakeModel:
    def __init__(self, values):
        self.values = iter(values)
        self.packets = []

    def __call__(self, packet):
        self.packets.append(deepcopy(packet))
        result = next(self.values)
        if isinstance(result, Exception):
            raise result
        return response(result)


class Clock:
    def __init__(self):
        self.now = 0

    def __call__(self):
        return self.now


def script(count=2):
    return {"initial_observation": observation(),
            "transitions": [{"action": {"kind": "click", "target": "run"}, "observation": observation(index)} for index in range(1, count + 1)],
            "elapsed_seconds": .25, "policy_sha256": "a" * 64, "seed": 771}


def reconstruct(packet):
    state = deepcopy(packet["current_observation"])
    reconstructed = {packet["current_evidence_id"]: deepcopy(state)}
    for row in reversed(packet["evidence"][1:]):
        for change in reversed(row["changes"]):
            parts = [part.replace("~1", "/").replace("~0", "~") for part in change["pointer"].split("/")[1:]]
            cursor = state
            for part in parts[:-1]:
                cursor = cursor[int(part)] if isinstance(cursor, list) else cursor[part]
            last = (int(parts[-1]) if isinstance(cursor, list) else parts[-1]) if parts else None
            value = cursor[last] if parts else state
            if "splice_kind" in change:
                start = change["start"]
                replacement = value[:start] + deepcopy(change["deleted"]) + value[start + change["inserted_length"]:]
            else:
                replacement = deepcopy(change["before"])
            if parts:
                cursor[last] = replacement
            else:
                state = replacement
        reconstructed[row["before"]] = deepcopy(state)
    return reconstructed


class SemanticTestingTests(unittest.TestCase):
    def setUp(self):
        self.network = patch("urllib.request.urlopen", side_effect=AssertionError("Offline tests must not dispatch HTTP."))
        self.network.start()
        self.addCleanup(self.network.stop)

    def run_session(self, answers, *, adapter=None, **kwargs):
        log = FakeLog()
        model = FakeModel(answers)
        adapter = adapter or FakeAdapter()
        summary = run_semantic_session(adapter, model, log, **kwargs)
        return summary, log, model, adapter

    def successful_answers(self, *, suspected=True):
        return [decision(hypotheses=[hypothesis()], actions=[action("control", "h1"), action("intervention", "h1", "2")]),
                decision(assessments=[assessment()], suspected_bug=[claim()] if suspected else [],
                         acknowledged_evidence=["e2"], finish=True)]

    def test_prospective_order_complete_but_not_self_confirmed(self):
        summary, log, model, adapter = self.run_session(self.successful_answers())
        self.assertTrue(summary["complete"])
        self.assertEqual((summary["controls"], summary["requests"], summary["responses"]), (2, 2, 2))
        self.assertFalse(summary["independently_confirmed"])
        self.assertEqual(summary["final_claims"][0]["attribution"], "prospective-structure")
        kinds = [row["type"] for row in log.rows]
        self.assertLess(kinds.index("semantic_response"), kinds.index("semantic_decision"))
        self.assertLess(kinds.index("semantic_hypothesis"), kinds.index("semantic_control_started"))
        self.assertLess(kinds.index("semantic_control_started"), kinds.index("semantic_transition"))
        self.assertGreater(log.syncs, 5)
        self.assertLessEqual(summary["max_input_bytes"], 48000)
        self.assertEqual(summary["pending_outcome"], None)

    def test_real_session_log_is_flushed_before_any_control(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "record.jsonl"
            def inspect(_action):
                rows = [json.loads(line) for line in path.read_text().splitlines()]
                self.assertEqual(rows[-1]["type"], "semantic_control_started")
                self.assertTrue(any(row["type"] == "semantic_hypothesis" for row in rows))
            with SessionLog(path) as log:
                summary = run_semantic_session(FakeAdapter(inspect), FakeModel(self.successful_answers()), log)
            self.assertTrue(summary["complete"])

    def test_posthoc_hypothesis_remains_ordinary_discovery(self):
        answers = [decision(actions=[action()]),
                   decision(hypotheses=[hypothesis()], suspected_bug=[claim(refs=["e1"])]),
                   decision(assessments=[assessment(refs=["e1"])], acknowledged_evidence=["e1"], finish=True)]
        summary, *_ = self.run_session(answers)
        self.assertTrue(summary["complete"])
        self.assertEqual(summary["final_claims"][0]["attribution"], "ordinary-discovery")

    def test_disproved_hypothesis_without_claim_is_not_false_report(self):
        answers = self.successful_answers(suspected=False)
        answers[1]["assessments"][0]["status"] = "reject"
        summary, *_ = self.run_session(answers)
        self.assertTrue(summary["complete"])
        self.assertEqual(summary["final_claims"], [])

    def test_candidate_does_not_require_detector_or_hypothesis(self):
        summary, *_ = self.run_session([decision(actions=[action()]),
            decision(suspected_bug=[claim(refs=["e1"], hypothesis_id=None)], acknowledged_evidence=["e1"], finish=True)])
        self.assertTrue(summary["complete"])
        self.assertEqual(summary["final_claims"][0]["attribution"], "ordinary-discovery")

    def test_initial_visible_issue_can_be_reported_as_an_ordinary_discovery(self):
        value = claim(refs=["e0"], hypothesis_id=None)
        summary, log, _, adapter = self.run_session([
            decision(suspected_bug=[value], acknowledged_evidence=["e0"], finish=True)])
        self.assertTrue(summary["complete"])
        self.assertEqual(adapter.actions, [])
        self.assertEqual(summary["final_claims"][0]["attribution"], "ordinary-discovery")
        self.assertEqual(summary["raw_claims"][0]["declarations"], [value])
        self.assertEqual(sum(row["type"] == "semantic_declared_claim" for row in log.rows), 1)

    def test_invalid_reference_preserves_raw_declaration_and_stops(self):
        value = decision(suspected_bug=[claim(refs=["e999"], hypothesis_id=None)], actions=[action()])
        summary, log, model, adapter = self.run_session([value])
        self.assertFalse(summary["complete"])
        self.assertEqual(adapter.actions, [])
        self.assertEqual(summary["raw_claims"][0]["declarations"], value["suspected_bug"])
        self.assertEqual(sum(row["type"] == "semantic_declared_claim" for row in log.rows), 1)
        self.assertTrue(summary["usage_complete"])

    def test_malformed_declarations_are_retained(self):
        value = decision(suspected_bug={"id": "bad", "claim": "must not disappear"})
        summary, log, *_ = self.run_session([value])
        self.assertFalse(summary["complete"])
        self.assertEqual(summary["raw_claims"][0]["declarations"], value["suspected_bug"])

    def test_unknown_usage_stops_before_actions_and_without_retry(self):
        model = lambda _packet: {"decision": decision(actions=[action()]), "usage": {}, "estimated_cost_usd": 0}
        adapter, log = FakeAdapter(), FakeLog()
        summary = run_semantic_session(adapter, model, log)
        self.assertTrue(summary["unknown_usage"])
        self.assertEqual(summary["requests"], 1)
        self.assertEqual(adapter.actions, [])
        self.assertEqual(summary["inference_retries"], 0)

    def test_provider_failure_has_unknown_usage_and_no_secret_exception_text(self):
        summary, log, *_ = self.run_session([RuntimeError("sensitive-token")])
        self.assertTrue(summary["unknown_usage"])
        self.assertEqual(summary["responses"], 0)
        self.assertNotIn("sensitive-token", json.dumps(log.rows))

    def test_private_observation_fields_fail_before_model(self):
        adapter = FakeAdapter()
        adapter.current["evaluation"] = {"known_issue": "secret-ground-truth"}
        summary, _, model, _ = self.run_session([], adapter=adapter)
        self.assertFalse(summary["complete"])
        self.assertEqual(model.packets, [])

    def test_private_host_metadata_never_enters_actor_packet(self):
        summary, log, model, _ = self.run_session([decision(acknowledged_evidence=["e0"], finish=True)],
            session_metadata={"session_id": "opaque", "known_registry_sha256": "private-truth-binding",
                              "source_sha256": "private-source-binding", "experience_shared": False})
        self.assertTrue(summary["complete"])
        self.assertNotIn("private-truth-binding", json.dumps(model.packets))
        self.assertNotIn("private-source-binding", json.dumps(model.packets))
        self.assertEqual(log.rows[0]["known_registry_sha256"], "private-truth-binding")

    def test_execution_configuration_bindings_are_header_only(self):
        settings = {"model": "private-model-identity", "thinking": True, "max_tokens": 8192,
                    "temperature": 1, "top_p": .95, "response_format": "json_schema"}
        summary, log, model, _ = self.run_session([decision(acknowledged_evidence=["e0"], finish=True)],
            session_metadata={"model_config_sha256": "private-config-hash", "browser_transport_sha256": "private-browser-hash",
                              "model_config": settings})
        self.assertTrue(summary["complete"])
        self.assertEqual(log.rows[0]["model_config"], settings)
        self.assertEqual(log.rows[0]["model_config_sha256"], "private-config-hash")
        self.assertEqual(log.rows[0]["browser_transport_sha256"], "private-browser-hash")
        for private_value in ("private-model-identity", "private-config-hash", "private-browser-hash"):
            self.assertNotIn(private_value, json.dumps(model.packets))

    def test_twenty_five_controls_rejected_before_any_execution(self):
        summary, _, _, adapter = self.run_session([decision(actions=[action()] * 25)])
        self.assertFalse(summary["complete"])
        self.assertEqual(adapter.actions, [])

    def test_last_request_is_report_only_and_cannot_leave_unacknowledged_control(self):
        answers = [decision()] * 5 + [decision(actions=[action()])]
        summary, _, model, adapter = self.run_session(answers)
        self.assertFalse(summary["complete"])
        self.assertEqual(summary["requests"], 6)
        self.assertTrue(model.packets[-1]["report_only"])
        self.assertEqual(adapter.actions, [])

    def test_finish_requires_latest_evidence(self):
        summary, *_ = self.run_session([decision(actions=[action()]), decision(acknowledged_evidence=["e0"], finish=True)])
        self.assertFalse(summary["complete"])
        self.assertEqual(summary["controls"], 1)

    def test_finish_requires_assessment_after_latest_hypothesis_control(self):
        answers = self.successful_answers()
        answers[1]["assessments"] = []
        summary, *_ = self.run_session(answers)
        self.assertFalse(summary["complete"])
        self.assertEqual(len(summary["final_claims"]), 1)

    def test_assessment_must_cite_latest_associated_control(self):
        answers = self.successful_answers()
        answers[1]["assessments"][0]["evidence"] = ["e1"]
        summary, *_ = self.run_session(answers)
        self.assertFalse(summary["complete"])

    def test_deadline_after_model_response_accounts_usage_but_does_not_act(self):
        clock = Clock()
        def model(_packet):
            clock.now = 301
            return response(decision(actions=[action()]))
        adapter = FakeAdapter()
        summary = run_semantic_session(adapter, model, FakeLog(), clock=clock)
        self.assertFalse(summary["complete"])
        self.assertFalse(summary["unknown_usage"])
        self.assertEqual(summary["usage"]["total_tokens"], 15)
        self.assertEqual(adapter.actions, [])

    def test_deadline_checked_between_atomic_controls(self):
        clock = Clock()
        def delay(_action):
            clock.now += 301
        summary, _, _, adapter = self.run_session([decision(actions=[action(), action()])], adapter=FakeAdapter(delay), clock=clock)
        self.assertFalse(summary["complete"])
        self.assertEqual(len(adapter.actions), 1)
        self.assertEqual(summary["pending_outcome"], None)

    def test_timeout_bound_is_forwarded_to_adapter(self):
        clock = Clock()
        adapter = FakeAdapter()
        def model(_packet):
            clock.now = 299.5
            return response(decision(actions=[action()]))
        summary = run_semantic_session(adapter, model, FakeLog(), clock=clock, limits=SemanticLimits(max_requests=2))
        self.assertLessEqual(adapter.timeout, .5)
        self.assertFalse(summary["complete"])

    def test_ambiguous_adapter_outcome_stops_with_pending_control(self):
        def fail(_action):
            raise RuntimeError("applied or not unknown")
        summary, *_ = self.run_session([decision(actions=[action(), action()])], adapter=FakeAdapter(fail))
        self.assertFalse(summary["complete"])
        self.assertEqual(summary["pending_outcome"], 1)
        self.assertEqual(summary["controls"], 1)

    def test_fabricated_public_clue_is_rejected(self):
        h = hypothesis()
        h["clues"][0]["quote"] = "The developer injected a bug"
        summary, _, _, adapter = self.run_session([decision(hypotheses=[h], actions=[action("control", "h1")])])
        self.assertFalse(summary["complete"])
        self.assertEqual(adapter.actions, [])

    def test_claims_cannot_be_deleted_by_later_empty_report(self):
        answers = [decision(actions=[action()]),
                   decision(suspected_bug=[claim(refs=["e1"], hypothesis_id=None)]),
                   decision(acknowledged_evidence=["e1"], finish=True)]
        summary, *_ = self.run_session(answers)
        self.assertTrue(summary["complete"])
        self.assertEqual(len(summary["final_claims"]), 1)
        self.assertEqual(len(summary["raw_claims"]), 3)

    def test_duplicate_claim_id_cannot_overwrite_false_claim(self):
        value = claim(refs=["e1"], hypothesis_id=None)
        answers = [decision(actions=[action()]), decision(suspected_bug=[value]),
                   decision(suspected_bug=[{**value, "claim": "withdrawn"}], acknowledged_evidence=["e1"], finish=True)]
        summary, *_ = self.run_session(answers)
        self.assertFalse(summary["complete"])
        self.assertEqual(summary["final_claims"][0]["claim"], value["claim"])
        self.assertEqual(len(summary["raw_claims"]), 3)

    def test_hypothesis_revision_cannot_bypass_two_proposal_cap(self):
        answers = [decision(hypotheses=[hypothesis("h1"), hypothesis("h2")]),
                   decision(hypotheses=[hypothesis("h3")])]
        summary, *_ = self.run_session(answers)
        self.assertFalse(summary["complete"])
        self.assertEqual(len(summary["hypotheses"]), 2)

    def test_rejected_hypothesis_cannot_drive_further_experiment(self):
        answers = [decision(hypotheses=[hypothesis()], actions=[action("control", "h1")]),
                   decision(assessments=[assessment("reject", ["e1"])], actions=[action("intervention", "h1")])]
        summary, _, _, adapter = self.run_session(answers)
        self.assertFalse(summary["complete"])
        self.assertEqual(len(adapter.actions), 1)

    def test_B_reviews_only_frozen_trajectory_without_adapter_access(self):
        class UnusableAdapter:
            def __getattr__(self, _name):
                raise AssertionError("B must not control browser")
        summary, _, model, _ = self.run_session([decision(acknowledged_evidence=["e0", "e1", "e2"], finish=True)],
            adapter=UnusableAdapter(), condition="B", frozen_trajectory=script())
        self.assertTrue(summary["complete"])
        self.assertEqual(summary["controls"], 2)
        self.assertGreaterEqual(summary["elapsed_seconds"], .25)
        self.assertTrue(model.packets[0]["report_only"])

    def test_B_cannot_execute_model_control(self):
        summary, _, _, adapter = self.run_session([decision(actions=[action()])], condition="B", frozen_trajectory=script())
        self.assertFalse(summary["complete"])
        self.assertEqual(adapter.actions, [])

    def test_B_overlong_script_cannot_bypass_control_limit(self):
        summary, _, model, _ = self.run_session([], condition="B", frozen_trajectory=script(25))
        self.assertFalse(summary["complete"])
        self.assertEqual(model.packets, [])

    def test_B_duration_includes_script_and_stops_before_dispatch(self):
        trajectory = script()
        trajectory["elapsed_seconds"] = 301
        summary, _, model, _ = self.run_session([], condition="B", frozen_trajectory=trajectory)
        self.assertFalse(summary["complete"])
        self.assertEqual(model.packets, [])

    def test_lossless_delta_packet_contains_every_intermediate_change(self):
        summary, _, model, _ = self.run_session([decision(actions=[action(value="2"), action(value="3")]),
            decision(acknowledged_evidence=["e2"], finish=True)])
        self.assertTrue(summary["complete"])
        packet = model.packets[-1]
        reconstructed = reconstruct(packet)
        self.assertEqual(reconstructed["e1"]["controls"][0]["value"], "2")
        self.assertEqual(reconstructed["e0"], observation())

    def test_full_observation_too_large_fails_without_truncation_or_call(self):
        adapter = FakeAdapter()
        adapter.current["text"] = "x" * 48001
        summary, _, model, _ = self.run_session([], adapter=adapter)
        self.assertFalse(summary["complete"])
        self.assertEqual(model.packets, [])
        self.assertGreater(summary["max_input_bytes"], 48000)

    def test_session_state_is_not_shared(self):
        first, *_ = self.run_session(self.successful_answers())
        second, _, model, _ = self.run_session([decision(acknowledged_evidence=["e0"], finish=True)])
        self.assertTrue(first["complete"] and second["complete"])
        self.assertEqual(model.packets[0]["hypotheses"], [])
        self.assertEqual(model.packets[0]["prior_claims"], [])

    def test_provider_raw_content_is_retained(self):
        log = FakeLog()
        raw = 'not json {"suspected_bug":'
        summary = run_semantic_session(FakeAdapter(), lambda _packet: response(raw, raw_content=raw, provider_input_bytes=10), log)
        self.assertFalse(summary["complete"])
        self.assertTrue(summary["usage_complete"])
        self.assertEqual(next(row for row in log.rows if row["type"] == "semantic_response")["response"]["raw_content"], raw)

    def test_truncated_response_records_known_usage_and_raw_claims_without_acting(self):
        log, adapter = FakeLog(), FakeAdapter()
        value = decision(suspected_bug=[claim(refs=["e1"], hypothesis_id=None)], actions=[action()])
        summary = run_semantic_session(adapter, lambda _packet: response(value, finish_reason="length", raw_content=json.dumps(value)), log)
        self.assertFalse(summary["complete"])
        self.assertFalse(summary["unknown_usage"])
        self.assertEqual(summary["usage"]["total_tokens"], 15)
        self.assertEqual(summary["raw_claims"][0]["declarations"], value["suspected_bug"])
        self.assertEqual(adapter.actions, [])

    def test_observation_accepts_public_file_dialog_options_and_checkbox(self):
        value = observation()
        value["controls"][0].update({"checked": False, "type": "text", "accept": "", "options": [{"value": "1", "label": "One", "disabled": False}]})
        value["downloads"] = [{"id": "download-1", "name": "save.json", "sha256": "a" * 64, "bytes": 100}]
        value["dialog"] = {"type": "alert", "message": "Saved"}
        self.assertEqual(public_observation(value), value)

    def test_frozen_limits_cannot_be_relaxed_or_given_boolean_values(self):
        for kwargs in ({"max_controls": 25}, {"max_requests": 7}, {"max_seconds": 301}, {"max_hypotheses": 3},
                       {"max_input_bytes": 48001}, {"max_total_input_bytes": 158001}, {"request_timeout_seconds": 46}, {"max_requests": True},
                       {"max_seconds": float("nan")}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                SemanticLimits(**kwargs)

    def test_action_rejects_macros_hidden_selectors_and_oversized_input(self):
        for value in ({"kind": "eval", "target": "window"}, {"kind": "click", "target": "run", "route": [1]},
                      {"kind": "wait", "milliseconds": True}, {"kind": "wait", "milliseconds": 2001},
                      {"kind": "input", "target": "edit", "value": "a" * 4001}):
            with self.subTest(value=value), self.assertRaises(SemanticContractError):
                validate_atomic_action(value)

    def test_schema_has_explicit_only_six_model_authored_fields(self):
        schema = semantic_schema()
        self.assertEqual(set(schema["properties"]), set(decision()))
        self.assertFalse(schema["additionalProperties"])
        self.assertEqual(schema["properties"]["actions"]["maxItems"], 24)

    def test_lossless_splices_reconstruct_unicode_list_reordering_and_insertions(self):
        pairs = [
            ("头\n😀 unchanged\nold paragraph\n尾", "头\nadded\n😀 unchanged\nnew paragraph\n尾"),
            ([{"id": "a", "value": "first"}, {"id": "b", "value": "second"}],
             [{"id": "b", "value": "revised"}, {"id": "c", "value": "third"}, {"id": "a", "value": "first"}]),
            ([1, 2, 3, 4], [0, 1, 4, 5]),
            ({"x/y": {"a~b": "😀 middle 😀"}}, {"x/y": {"a~b": "😀 replaced 😀"}}),
            ({"type": "one"}, ["new", "type"]),
        ]
        for before, after in pairs:
            with self.subTest(before=before):
                packet = {"current_observation": after, "current_evidence_id": "e1", "evidence": [
                    {"evidence_id": "e0"}, {"evidence_id": "e1", "before": "e0", "changes": _diff(before, after)}]}
                self.assertEqual(reconstruct(packet)["e0"], before)

    def test_B_pagination_delivers_every_snapshot_and_requires_each_acknowledgment(self):
        trajectory = script(24)
        originals = {"e0": trajectory["initial_observation"]}
        for index, row in enumerate(trajectory["transitions"], 1):
            row["observation"]["text"] = chr(64 + index) * 2400
            originals[f"e{index}"] = row["observation"]
        packets, log = [], FakeLog()
        def model(packet):
            packets.append(deepcopy(packet))
            return response(decision(acknowledged_evidence=packet["review_page"]["evidence_ids"],
                                     finish=packet["review_page"]["final"]))
        summary = run_semantic_session(None, model, log, condition="B", frozen_trajectory=trajectory)
        self.assertTrue(summary["complete"])
        self.assertGreater(len(packets), 1)
        self.assertLessEqual(len(packets), 6)
        recovered = {}
        for packet in packets:
            recovered.update(reconstruct(packet))
        self.assertEqual(recovered, originals)
        self.assertEqual(set(summary["reviewed_evidence"]), set(originals))
        self.assertLessEqual(summary["total_input_bytes"], 158000)
        self.assertLessEqual(summary["max_input_bytes"], 48000)

    def test_B_cannot_skip_review_of_intermediate_snapshot(self):
        summary, *_ = self.run_session([decision(acknowledged_evidence=["e0", "e2"], finish=True)],
                                      condition="B", frozen_trajectory=script())
        self.assertFalse(summary["complete"])

    def test_cumulative_input_budget_stops_before_extra_request(self):
        summary, _, model, _ = self.run_session([decision()] * 6,
                                               limits=SemanticLimits(max_total_input_bytes=16000))
        self.assertFalse(summary["complete"])
        self.assertEqual(summary["requests"], len(model.packets))
        self.assertLess(summary["requests"], 6)
        self.assertLessEqual(summary["total_input_bytes"], 16000)


if __name__ == "__main__":
    unittest.main()
