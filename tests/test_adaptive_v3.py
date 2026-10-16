"""Sustained factual investigations; all decisions and HTTP are offline stubs."""

import copy
import hashlib
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest

from autoplay_qa.adaptive import AdaptivePromptBudgetExceeded, adaptive_input_bytes, expand_observation
from autoplay_qa.adaptive_v3 import AdaptiveContextV3, ADAPTIVE_V3_PROMPT_VERSION
from autoplay_qa.experience import ExperienceStore, public_observation, state_key
from autoplay_qa.investigation import applicability, public_field
from autoplay_qa.lean import decision_dict
from autoplay_qa.nebius import InvalidDecision
from tests.test_adaptive_context import build, observation, record
from tests.test_adaptive_v2 import IndexStore


def question(identifier="q1", *, facets=None, decision="continue"):
    return {"id": identifier, "decision": decision,
            "question": "Which observed condition affects this control?",
            "explanations": ["The control changes visible state", "Another visible condition is required"],
            "expected_result": "The selected visible field changes after the probe",
            "condition_facets": facets or ["player.facing"]}


def forecast(path="player.position", relation="changes", value=None):
    return {"path": path, "relation": relation,
            "value_json": json.dumps(value, separators=(",", ":")) if relation == "equals" else None}


def decision(action="forward", *, update=None, predictions=()):
    return {"action": action, "reason": "Compare the current observed condition",
            "expected_position": None, "notes": [], "investigation_update": copy.deepcopy(update),
            "predictions": copy.deepcopy(list(predictions))}


def acknowledge(context, step, before, after, *, action="forward", update=None, predictions=()):
    build(context, before)
    parsed, warnings = context.parse_decision(json.dumps(decision(action, update=update, predictions=predictions)),
        context.allowed(before), {item for _, item in context._seen_records})
    item = record(step, before, action, after)
    item["decision"] = decision_dict(parsed)
    context.record(item)
    return item, warnings


class InquiryStore(IndexStore):
    def __init__(self, source=None, *, failure=None):
        super().__init__()
        self.source = copy.deepcopy(source)
        self.saved = []
        self.failure = failure

    def investigation_index(self, current, *, exclude_run_id, max_bytes):
        if self.failure:
            raise self.failure
        if self.source is None:
            return {"version": "investigation-index-v1", "status": "available", "scope_id": "a" * 64,
                    "current_state_id": state_key(current), "excluded_run_id": exclude_run_id,
                    "source_run_id": None, "revision": 0, "investigations": [], "applicability": {},
                    "complete": True, "omitted_count": 0, "evidence_facts": [], "omitted_fact_count": 0}
        packet = copy.deepcopy(self.source)
        packet.update(current_state_id=state_key(current), excluded_run_id=exclude_run_id)
        packet["applicability"] = {row["id"]: applicability(row, current, state_key(current)) for row in packet["investigations"]}
        return packet

    def save_investigations(self, run_id, revision, rows, *, parent_snapshot=None):
        if self.failure:
            raise self.failure
        self.saved.append((run_id, revision, copy.deepcopy(rows), copy.deepcopy(parent_snapshot)))
        return {"persisted": True}


class AdaptiveV3ContractTests(unittest.TestCase):
    def test_complete_native_schema_and_new_contract_without_legacy_conflict(self):
        current = observation(position=(0.5, 1.25, 2.75))
        context = AdaptiveContextV3(current, run_id="v3")
        messages, value = build(context, current)
        schema = context.schema(context.allowed(current))
        self.assertEqual(context.prompt_version, ADAPTIVE_V3_PROMPT_VERSION)
        self.assertEqual(set(schema["properties"]), {"action", "reason", "expected_position", "notes", "investigation_update", "predictions"})
        self.assertEqual(schema["properties"]["action"]["enum"], [*current["controls"], "reset", "finish"])
        self.assertEqual(set(value["response_contract"]["fields"]), set(schema["properties"]))
        self.assertEqual(expand_observation(value["observation"]), public_observation(current))
        self.assertNotIn("Return only action, reason, expected_position, notes.", messages[0]["content"])
        self.assertLessEqual(adaptive_input_bytes(messages, schema), 24_000)

    def test_bad_optional_update_and_one_bad_forecast_keep_legal_control_and_good_forecast(self):
        current = observation()
        context = AdaptiveContextV3(current)
        raw = decision(update={"id": "q", "decision": "invent_route"}, predictions=[{"path": "private.password"}, forecast()])
        parsed, warnings = context.parse_decision(json.dumps(raw), context.allowed(current), set())
        self.assertEqual(parsed.action, "forward")
        self.assertIsNone(parsed.adaptive_fields["investigation_update"])
        self.assertEqual(parsed.adaptive_fields["predictions"], [forecast()])
        self.assertEqual({item["code"] for item in warnings}, {"invalid_investigation_update", "invalid_prediction"})
        with self.assertRaises(InvalidDecision):
            context.parse_decision(json.dumps(decision("illegal")), context.allowed(current), set())

    def test_missing_extension_fields_are_small_canonical_noops(self):
        current = observation()
        context = AdaptiveContextV3(current)
        raw = decision()
        raw.pop("investigation_update")
        raw.pop("predictions")
        parsed, warnings = context.parse_decision(json.dumps(raw), context.allowed(current), set())
        self.assertEqual(warnings, [])
        self.assertEqual(parsed.adaptive_fields, {"investigation_update": None, "predictions": []})

    def test_private_transient_array_index_and_unbounded_forecasts_are_dropped(self):
        current = observation(entities=[{"id": "panel", "state": {"open": False, "large": "x" * 500}}])
        current["_capabilities"]["state_identity"] = {"transient_paths": ["entities.panel.state.execution"]}
        current["entities"][0]["state"]["execution"] = {"count": 1}
        context = AdaptiveContextV3(current)
        for path in ("tick", "events", "entities.0.state.open", "entities.panel.state.execution.count", "entities.panel.state.large", "player.secret"):
            with self.subTest(path=path):
                parsed, warnings = context.parse_decision(json.dumps(decision(predictions=[forecast(path)])), context.allowed(current), set())
                self.assertEqual(parsed.action, "forward")
                self.assertEqual(parsed.adaptive_fields["predictions"], [])
                self.assertEqual(warnings[0]["code"], "invalid_prediction")

    def test_generic_3d_and_ui_predictions_are_compared_without_two_integer_assumption(self):
        first = observation(position=(0.5, 1.25, 2.75), entities=[{"id": "panel", "state": {"open": False}}])
        after = observation(position=(1.0, 1.25, 2.75), entities=[{"id": "panel", "state": {"open": True}}], tick=1)
        context = AdaptiveContextV3(first)
        item, warnings = acknowledge(context, 1, first, after, predictions=[forecast("player.position", "equals", [1.0, 1.25, 2.75]), forecast("entities.panel.state.open", "equals", True)])
        self.assertEqual(warnings, [])
        self.assertEqual([check["status"] for check in item["investigation_fact"]["prediction_checks"]], ["matched", "matched"])
        ui = observation(controls=["click"], entities=[{"id": "panel", "state": {"open": False}}])
        ui["player"] = {"screen": "settings"}
        ui_after = copy.deepcopy(ui)
        ui_after["entities"][0]["state"]["open"] = True
        ui_context = AdaptiveContextV3(ui)
        ui_item, _ = acknowledge(ui_context, 1, ui, ui_after, action="click", predictions=[forecast("entities.panel.state.open", "changes")])
        self.assertEqual(ui_item["investigation_fact"]["prediction_checks"][0]["status"], "matched")
        self.assertEqual(ui_context.summary()["distinct_positions"], 0)

    def test_disappearing_forecast_field_is_uncomparable_not_a_bug(self):
        before = observation(entities=[{"id": "panel", "state": {"open": False}}])
        after = observation(tick=1)
        context = AdaptiveContextV3(before)
        item, _ = acknowledge(context, 1, before, after, predictions=[forecast("entities.panel.state.open", "changes")])
        check = item["investigation_fact"]["prediction_checks"][0]
        self.assertEqual(check["status"], "not_comparable")
        self.assertNotIn("verdict", item["investigation_fact"])


class AdaptiveV3InvestigationTests(unittest.TestCase):
    def test_null_update_sustains_one_question_and_counts_actual_citations_idempotently(self):
        first = observation()
        context = AdaptiveContextV3(first)
        step1, _ = acknowledge(context, 1, first, observation(tick=1), update=question(), predictions=[forecast()])
        step2, _ = acknowledge(context, 2, step1["after"], observation(tick=2), predictions=[forecast()])
        self.assertEqual(step2["investigation_fact"]["investigation_id"], "q1")
        self.assertEqual(context._investigations[0]["evidence"], [{"run_id": "session", "step": 1}, {"run_id": "session", "step": 2}])
        self.assertEqual(context._investigations[0]["conditions"], {"player.facing": "North"})
        prior = copy.deepcopy(context.summary()["investigation"])
        context.record(step2)
        self.assertEqual(context.summary()["investigation"], prior)

    def test_selected_conditions_ignore_unrelated_world_hash_novelty(self):
        first = observation(entities=[{"id": "clock", "state": {"phase": 0}}])
        second = observation(entities=[{"id": "clock", "state": {"phase": 1}}], tick=1)
        third = observation(entities=[{"id": "clock", "state": {"phase": 2}}], tick=2)
        context = AdaptiveContextV3(first)
        item1, _ = acknowledge(context, 1, first, second, update=question())
        item2, _ = acknowledge(context, 2, second, third)
        trial = item2["investigation_fact"]["condition_trial"]
        self.assertNotEqual(state_key(first), state_key(second))
        self.assertEqual(trial["previous_same_condition_action_count"], 1)
        self.assertEqual(trial["previous_same_selected_outcome_count"], 1)
        self.assertFalse(trial["selected_fields_changed"])
        self.assertEqual(item1["investigation_fact"]["condition_trial"]["condition_id"], trial["condition_id"])

    def test_same_control_under_changed_selected_condition_records_comparison_opportunity(self):
        first, turned = observation(), observation(facing="West", tick=1)
        context = AdaptiveContextV3(first)
        acknowledge(context, 1, first, turned, update=question())
        second, _ = acknowledge(context, 2, turned, observation(facing="West", tick=2))
        self.assertEqual(second["investigation_fact"]["condition_trial"]["other_conditions_for_action"], 1)
        self.assertEqual(context.summary()["investigation"]["condition_contrast_trials"], 1)

    def test_revision_copies_new_public_conditions_and_keeps_earlier_counterexamples(self):
        first = observation()
        context = AdaptiveContextV3(first)
        step1, _ = acknowledge(context, 1, first, observation(tick=1), update=question(), predictions=[forecast()])
        before2 = observation(facing="West", tick=2)
        update = {"id": "q1", "decision": "revise", "expected_result": "Compare after the visible heading changed", "condition_facets": ["player.facing"]}
        step2, _ = acknowledge(context, 2, before2, observation(facing="West", tick=3), update=update)
        row = context._investigations[0]
        self.assertEqual(row["conditions"], {"player.facing": "West"})
        self.assertEqual(row["origin_state_id"], state_key(before2))
        self.assertEqual(row["counterexamples"], [step1["investigation_fact"]["citation"]])
        self.assertEqual(step2["investigation_fact"]["update"]["decision"], "revise")

    def test_more_than_four_misses_keep_old_counterexamples_without_stopping_question(self):
        current = observation()
        context = AdaptiveContextV3(current)
        items = []
        for step in range(1, 7):
            after = observation(tick=step)
            item, _ = acknowledge(context, step, current, after, update=question() if step == 1 else None, predictions=[forecast()])
            items.append(item)
            current = after
        self.assertEqual(len(context._investigations[0]["counterexamples"]), 4)
        self.assertEqual(context._investigations[0]["last_transition"]["step"], 6)
        self.assertEqual(items[-1]["investigation_fact"]["update"]["counterexamples_omitted_count"], 1)
        self.assertEqual(context.summary()["investigation"]["forecast_mismatch"], 6)
        self.assertEqual(context.summary()["investigation"]["updates_dropped"], 0)

    def test_pending_capacity_discards_new_metadata_but_executes_control(self):
        current = observation()
        context = AdaptiveContextV3(current)
        for step in range(1, 6):
            item, _ = acknowledge(context, step, current, observation(tick=step), update=question(f"q{step}"))
            current = item["after"]
        summary = context.summary()["investigation"]
        self.assertEqual(context.summary()["executed_controls"], 5)
        self.assertEqual(summary["active_id"], "q4")
        self.assertEqual(summary["pending_ids"], ["q1", "q2", "q3"])
        self.assertEqual(item["investigation_fact"]["update"]["status"], "dropped")
        self.assertIsNone(item["investigation_fact"]["investigation_id"])
        self.assertIsNone(item["investigation_fact"]["condition_trial"])
        self.assertEqual(summary["acknowledged_controls_with_investigation"], 4)
        active = next(row for row in context._investigations if row["id"] == "q4")
        self.assertEqual(active["evidence"], [{"run_id": "session", "step": 4}])
        self.assertEqual(active["last_transition"]["step"], 4)

    def test_capacity_rejection_in_real_loop_does_not_fabricate_sustained_probe_and_replays(self):
        from tests.test_investigation_integration import InvestigationIntegrationTests
        from tests.test_lean_exploration import response
        from autoplay_qa.session import replay

        replies = [response(decision("east", update=question(f"q{step}", facets=["player.position"])))
                   for step in range(1, 6)]
        replies.append(response(decision("finish")))
        result, path, rows, http = InvestigationIntegrationTests.fixture(self, replies)
        executed = [row for row in rows if row["type"] == "step"]
        self.assertEqual((result["steps"], result["rejected_decisions"], http.call_count), (5, 0, 6))
        self.assertEqual([row["investigation_fact"]["investigation_id"] for row in executed], ["q1", "q2", "q3", "q4", None])
        self.assertEqual(executed[-1]["investigation_fact"]["update"]["status"], "dropped")
        self.assertIsNone(executed[-1]["investigation_fact"]["condition_trial"])
        self.assertEqual(result["adaptive"]["investigation"]["acknowledged_controls_with_investigation"], 4)
        self.assertTrue(replay(path)["adaptive_facts_validated"])

    def test_finish_can_close_existing_question_without_fake_transition_or_citation(self):
        current = observation()
        context = AdaptiveContextV3(current)
        item, _ = acknowledge(context, 1, current, observation(tick=1), update=question())
        evidence = copy.deepcopy(context._investigations[0]["evidence"])
        receipt = context.finish({"decision": decision("finish", update={"id": "q1", "decision": "close"}), "observation": item["after"]})
        self.assertIsNone(receipt["citation"])
        self.assertEqual(receipt["prediction_checks"], [])
        self.assertEqual(context._investigations[0]["evidence"], evidence)
        self.assertEqual(context.summary()["executed_controls"], 1)
        self.assertEqual(context.summary()["investigation"]["tentatively_closed_ids"], ["q1"])
        self.assertEqual(context.finish({"decision": decision("finish", update={"id": "q1", "decision": "close"}), "observation": item["after"]}), receipt)

    def test_finish_cannot_create_an_uncited_question(self):
        current = observation()
        context = AdaptiveContextV3(current)
        receipt = context.finish({"decision": decision("finish", update=question()), "observation": current})
        self.assertEqual(receipt["update"]["status"], "dropped")
        self.assertEqual(context._investigations, [])


class AdaptiveV3ArchiveTests(unittest.TestCase):
    def test_actual_persistence_recovery_conditional_applicability_and_frozen_no_writes(self):
        first = observation()
        after = observation(tick=1)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "experience.sqlite"
            with ExperienceStore(path, "game", "version", first) as store:
                context = AdaptiveContextV3(first, store=store, run_id="acquisition")
                acknowledge(context, 1, first, after, update=question(), predictions=[forecast()])
                self.assertEqual(context.summary()["memory"]["investigation_persistence"]["persisted_revisions"], 1)
                source = store.investigation_index(first, exclude_run_id="evaluation")
                self.assertEqual(source["source_run_id"], "acquisition")
                self.assertEqual(len(source["evidence_facts"]), 1)
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            with ExperienceStore(path, "game", "version", first, read_only=True) as store:
                context = AdaptiveContextV3(first, store=store, run_id="evaluation")
                _, value = build(context, first)
                source = value["investigation_memory"]["prior_run_index"]
                self.assertTrue(source["applicability"]["q1"]["all_selected_match"])
                self.assertEqual(context.summary()["investigation"]["restored_unfinished"], 1)
                second, _ = acknowledge(context, 1, first, after, predictions=[forecast()])
                self.assertEqual(second["investigation_fact"]["investigation_id"], "q1")
                self.assertEqual(store.summary()["transitions"], 1)
                _, changed = build(context, observation(facing="West"))
                self.assertFalse(changed["investigation_memory"]["prior_run_index"]["applicability"]["q1"]["all_selected_match"])
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), digest)

    def test_sqlite_failure_is_typed_warning_and_local_legal_action_still_learns(self):
        current = observation()
        store = InquiryStore(failure=sqlite3.OperationalError("locked /Users/private/project.sqlite sk-secret-private"))
        context = AdaptiveContextV3(current, store=store)
        item, _ = acknowledge(context, 1, current, observation(tick=1), update=question())
        self.assertEqual(item["investigation_fact"]["update"]["status"], "applied")
        self.assertEqual(context.summary()["executed_controls"], 1)
        self.assertEqual(context.summary()["memory"]["investigation_persistence"]["save_skipped_incomplete_source"], 1)
        serialized = json.dumps(context.summary())
        self.assertNotIn("/Users/private", serialized)
        self.assertNotIn("sk-secret-private", serialized)

    def test_sqlite_write_failure_keeps_both_executed_and_finish_updates_local(self):
        class FailingWrite(InquiryStore):
            def save_investigations(self, *_args, **_kwargs):
                raise sqlite3.OperationalError("locked /Users/private/memory.sqlite")
        current = observation()
        context = AdaptiveContextV3(current, store=FailingWrite())
        item, _ = acknowledge(context, 1, current, observation(tick=1), update=question())
        receipt = context.finish({"decision": decision("finish", update={"id": "q1", "decision": "close"}), "observation": item["after"]})
        self.assertEqual(receipt["update"]["status"], "applied")
        self.assertEqual(context.summary()["memory"]["investigation_persistence"]["save_failures"], 2)
        self.assertEqual(context.summary()["investigation"]["tentatively_closed_ids"], ["q1"])

    def test_credential_metadata_is_discarded_before_prompts_or_store(self):
        current = observation()
        store = InquiryStore()
        context = AdaptiveContextV3(current, store=store, secrets=("runtime-credential-123",))
        raw = decision(update=question())
        raw["investigation_update"]["question"] = "Use runtime-credential-123"
        parsed, warnings = context.parse_decision(json.dumps(raw), context.allowed(current), set())
        self.assertIsNone(parsed.adaptive_fields["investigation_update"])
        self.assertEqual(warnings[0]["code"], "invalid_investigation_update")
        self.assertEqual(store.saved, [])
        messages, _ = build(context, current)
        self.assertNotIn("runtime-credential-123", json.dumps(messages))

    def test_invalid_source_packet_never_becomes_unproven_complete_history(self):
        current = observation()
        valid = InquiryStore().investigation_index(current, exclude_run_id="evaluation", max_bytes=4000)
        mutations = [lambda p: p.update(revision=True), lambda p: p.update(scope_id="invalid"),
                     lambda p: p.update(source_run_id="evaluation", revision=1),
                     lambda p: p.update(complete=True, omitted_count=1),
                     lambda p: p.update(evidence_facts=[{"citation": {"run_id": "future", "step": 999}}]),
                     lambda p: p.update(reason="sk-abcdefghijklmnopqrstuv")]
        for mutate in mutations:
            with self.subTest(mutate=mutate):
                source = copy.deepcopy(valid)
                mutate(source)
                context = AdaptiveContextV3(current, store=InquiryStore(source), run_id="evaluation")
                _, value = build(context, current)
                packet = value["investigation_memory"]["prior_run_index"]
                self.assertEqual(packet["status"], "unavailable")
                self.assertFalse(packet["complete"])
                self.assertEqual(context._investigations, [])

    def test_saved_parent_lineage_preserves_negative_evidence_across_writable_runs(self):
        first, after = observation(), observation(tick=1)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "memory.sqlite"
            with ExperienceStore(path, "game", "version", first) as store:
                prior = AdaptiveContextV3(first, store=store, run_id="prior")
                acknowledge(prior, 1, first, after, update=question(), predictions=[forecast()])
                current = AdaptiveContextV3(first, store=store, run_id="current")
                acknowledge(current, 1, first, after, update={"id": "q1", "decision": "revise", "expected_result": "A different tentative result"})
                packet = store.investigation_index(first, exclude_run_id="next")
                self.assertEqual(packet["source_run_id"], "current")
                self.assertEqual(packet["investigations"][0]["counterexamples"], [{"run_id": "prior", "step": 1}])
                self.assertEqual(current.summary()["memory"]["investigation_persistence"]["save_failures"], 0)

    def test_unavailable_or_partial_parent_never_gets_replaced_by_a_locally_new_queue(self):
        current = observation()
        for partial in (False, True):
            with self.subTest(partial=partial):
                store = InquiryStore()
                packet = store.investigation_index(current, exclude_run_id="current", max_bytes=4000)
                if partial:
                    packet.update(source_run_id="prior", revision=1, complete=False, omitted_count=1)
                else:
                    packet.pop("scope_id")
                    packet.update(status="unavailable", complete=False, error_type="OperationalError")
                store.source = packet
                context = AdaptiveContextV3(current, store=store, run_id="current")
                item, _ = acknowledge(context, 1, current, observation(tick=1), update=question())
                self.assertEqual(item["investigation_fact"]["update"]["status"], "applied")
                self.assertEqual(store.saved, [])
                self.assertEqual(context.summary()["memory"]["investigation_persistence"]["save_skipped_incomplete_source"], 1)


class AdaptiveV3BudgetTests(unittest.TestCase):
    def test_minimum_budget_preserves_complete_current_state_and_schema(self):
        current = observation(controls=["tap"])
        context = AdaptiveContextV3(current, max_prompt_bytes=8_000)
        messages, value = build(context, current)
        self.assertLessEqual(adaptive_input_bytes(messages, context.schema(context.allowed(current))), 8_000)
        self.assertEqual(expand_observation(value["observation"]), public_observation(current))

    def test_essential_observation_overflow_fails_before_inference(self):
        current = observation(entities=[{"id": "large", "state": {"text": "x" * 30_000}}])
        context = AdaptiveContextV3(current)
        with self.assertRaises(AdaptivePromptBudgetExceeded):
            build(context, current)

    def test_omission_and_source_packets_reconstruct_deterministically(self):
        current = observation(controls=["tap"])
        context = AdaptiveContextV3(current, max_prompt_bytes=8_000, run_id="evaluation")
        messages, value = build(context, current)
        class Recorded(InquiryStore):
            def retrieve(self, *_args, **_kwargs):
                return copy.deepcopy(value["retrieved_experiments"])
            def experiment_index(self, *_args, **_kwargs):
                return copy.deepcopy(value["exploration_coverage"]["experiment_agenda"]["prior_run_index"])
            def investigation_index(self, *_args, **_kwargs):
                return copy.deepcopy(value["investigation_memory"]["prior_run_index"])
        replay = AdaptiveContextV3(current, max_prompt_bytes=8_000, run_id="evaluation", store=Recorded())
        reproduced = replay.messages(current, 0, {}, [], value["remaining_budget"], replay_compaction=value["context_compaction"])
        self.assertEqual(messages, reproduced)
        self.assertEqual(context.summary()["investigation"], replay.summary()["investigation"])

    def test_undelivered_prior_rows_never_become_hidden_state_or_get_overwritten(self):
        current = observation(controls=["tap"])
        prior = AdaptiveContextV3(current, run_id="prior")
        acknowledge(prior, 1, current, observation(controls=["tap"], tick=1), action="tap", update=question("prior-question"))
        row = copy.deepcopy(prior._investigations[0])
        source = {"version": "investigation-index-v1", "status": "available", "scope_id": "a" * 64,
                  "current_state_id": state_key(current), "excluded_run_id": "evaluation", "source_run_id": "prior", "revision": 1,
                  "investigations": [row], "applicability": {row["id"]: applicability(row, current, state_key(current))},
                  "complete": True, "omitted_count": 0, "evidence_facts": [], "omitted_fact_count": 1}
        store = InquiryStore(source)
        context = AdaptiveContextV3(current, store=store, run_id="evaluation", max_prompt_bytes=8_000)
        _, value = build(context, current)
        self.assertTrue(value["context_compaction"]["investigation_memory_omitted"])
        self.assertFalse(context._archive_loaded)
        self.assertEqual(context._investigations, [])
        acknowledge(context, 1, current, observation(controls=["tap"], tick=1), action="tap", update=question("local-question"))
        self.assertEqual(store.saved, [])
        self.assertEqual(context._investigations[0]["id"], "local-question")
        # A later truthful adapter observation can have a larger configured
        # working set; this white-box check exercises delayed import only.
        context.max_prompt_bytes = 24_000
        build(context, current)
        self.assertEqual(context._investigations[0]["id"], "local-question")
        self.assertTrue(context._parent_incomplete)


if __name__ == "__main__":
    unittest.main()
