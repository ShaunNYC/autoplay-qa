"""Behavioral checks for bounded native-control experimental contexts.

No inference calls are made.  The fixtures deliberately use relative controls,
UI-only state, and three-dimensional positions rather than one game's names.
"""

import copy
import json
import unittest

from autoplay_qa.adaptive import (
    ADAPTIVE_PROMPT_VERSION,
    AdaptiveContext,
    AdaptivePromptBudgetExceeded,
    adaptive_input_bytes,
    compact_observation,
    expand_observation,
    native_controls,
)
from autoplay_qa.experience import public_observation, state_key
from autoplay_qa.lean import LEAN_PROMPT_VERSION, LeanNote, lean_schema


def observation(*, position=(0, 0), facing="North", tick=0, events=(), controls=None, entities=None):
    return {
        "game": "native-fixture-v1",
        "tick": tick,
        "status": "playing",
        "coordinates": "Adapter coordinates: first coordinate horizontal, second vertical.",
        "player": {"position": list(position), "facing": facing, "inventory": ["blocks:3", "bombs:1"]},
        "entities": copy.deepcopy(entities or []),
        "events": list(events),
        "controls": list(controls if controls is not None else ("forward", "left", "right", "place", "remove", "bomb")),
        "_capabilities": {"reset": True},
    }


def record(step, before, action, after, *, kind="step"):
    return {
        "type": kind, "step": step, "episode": 0,
        "before": copy.deepcopy(before), "after": copy.deepcopy(after),
        "decision": {"action": action, "reason": "Test visible behavior", "expected_position": None, "notes": []},
        # This field must never be promoted into public experimental memory.
        "evaluation": {"injected_defect": "PRIVATE_TEST_LABEL", "developer_rule": "PRIVATE_TEST_RULE"},
    }


def context_value(messages):
    return json.loads(messages[1]["content"])


def build(context, current, *, records=(), notes=None, feedback=""):
    messages = context.messages(
        current, 0, notes or {}, list(records),
        {"actions": 24, "requests": 28, "seconds": 300},
        focus="qa", feedback=feedback,
    )
    return messages, context_value(messages)


class CapturingStore:
    def __init__(self, facts=(), *, fail_record=False, fail_retrieve=False):
        self.facts = list(facts)
        self.records = []
        self.queries = []
        self.fail_record = fail_record
        self.fail_retrieve = fail_retrieve

    def record(self, before, action, after, *, run_id, step):
        if self.fail_record:
            raise OSError("Private path /Users/private-project/experience.sqlite; PRIVATE_SECRET")
        self.records.append((copy.deepcopy(before), action, copy.deepcopy(after), run_id, step))

    def retrieve(self, current, *, limit, max_bytes):
        self.queries.append((copy.deepcopy(current), limit, max_bytes))
        if self.fail_retrieve:
            raise OSError("Private path /Users/private-project/experience.sqlite; PRIVATE_SECRET")
        return {
            "scope_id": "opaque-scope", "current_state_id": state_key(current),
            "facts": copy.deepcopy(self.facts), "omitted_count": 0,
        }

    def summary(self):
        return {"recorded": len(self.records)}


class NativeContextTests(unittest.TestCase):
    def test_catalog_schema_and_metrics_use_only_adapter_native_actions(self):
        current = observation()
        adaptive = AdaptiveContext(current)
        messages, value = build(adaptive, current)
        allowed = [*current["controls"], "reset", "finish"]
        self.assertEqual(adaptive.allowed(current), allowed)
        self.assertEqual(value["observation"]["controls"], allowed)
        self.assertEqual(adaptive.schema(allowed)["properties"]["action"]["enum"], allowed)
        self.assertEqual(set(value["exploration_coverage"]["native_control_use_counts"]), set(current["controls"]))
        self.assertEqual(set(adaptive.schema(allowed)["properties"]), {"action", "reason", "expected_position", "notes"})
        self.assertNotIn("neighbors", value["observation"])
        for old_control in ('"north"', '"south"', '"east"', '"west"', '"interact"', '"wait"'):
            self.assertNotIn(old_control, messages[1]["content"])
        self.assertEqual(value["prompt_version"], ADAPTIVE_PROMPT_VERSION)
        self.assertEqual(LEAN_PROMPT_VERSION, "lean-v1")

    def test_native_catalog_deduplicates_and_respects_host_meta_capabilities(self):
        current = observation(controls=["forward", "forward", "reset", "finish", "turn(45)"])
        current["_capabilities"]["reset"] = False
        adaptive = AdaptiveContext(current)
        self.assertEqual(native_controls(current), ["forward", "turn(45)"])
        self.assertEqual(adaptive.allowed(current), ["forward", "turn(45)", "finish"])
        current["status"] = "won"
        self.assertEqual(adaptive.allowed(current), ["finish"])

    def test_invalid_native_controls_are_rejected_without_game_assumptions(self):
        for catalog in ("forward", ["forward", 3], [""]):
            current = observation()
            current["controls"] = catalog
            with self.subTest(catalog=catalog), self.assertRaises(ValueError):
                AdaptiveContext(current)

    def test_native_catalog_can_change_with_visible_ui_state(self):
        first = observation(controls=["open"])
        second = observation(controls=["close", "confirm"])
        second["player"]["facing"] = "Panel"
        adaptive = AdaptiveContext(first)
        adaptive.record(record(1, first, "open", second))
        _, value = build(adaptive, second)
        self.assertEqual(value["observation"]["controls"], ["close", "confirm", "reset", "finish"])
        self.assertEqual(adaptive.summary()["native_control_use_counts"], {"open": 1, "close": 0, "confirm": 0})

    def test_private_evaluation_source_rules_and_credentials_never_enter_prompt(self):
        first = observation()
        first.update({"evaluation": {"label": "PRIVATE_LABEL"}, "developer_rules": ["PRIVATE_RULE"], "source_path": "/Users/private/source.ts", "api_key": "PRIVATE_SECRET"})
        first["entities"] = [{"id": "widget", "state": {"visible": True, "evaluation": "PRIVATE_LABEL", "developer_rule": "PRIVATE_RULE", "api_key": "PRIVATE_SECRET"}}]
        second = copy.deepcopy(first)
        second["tick"] = 1
        adaptive = AdaptiveContext(first)
        messages, _ = build(adaptive, second, records=[record(1, first, "forward", second)])
        text = json.dumps(messages)
        for secret in ("PRIVATE_LABEL", "PRIVATE_RULE", "PRIVATE_SECRET", "PRIVATE_TEST_LABEL", "PRIVATE_TEST_RULE", "/Users/private/"):
            self.assertNotIn(secret, text)


class GenericExperimentTests(unittest.TestCase):
    def test_turning_without_movement_is_a_state_change(self):
        first = observation(facing="North")
        second = observation(facing="West", tick=1, events=["Turned left"])
        adaptive = AdaptiveContext(first)
        fact = adaptive.record(record(1, first, "left", second))
        summary = adaptive.summary()
        self.assertTrue(fact["state_changed"])
        self.assertFalse(fact["position_changed"])
        self.assertEqual(summary["no_observable_state_change_actions"], 0)
        self.assertEqual(summary["distinct_observed_states"], 2)
        self.assertEqual(summary["distinct_positions"], 1)
        _, value = build(adaptive, second)
        self.assertEqual(value["observation"]["player"]["facing"], "West")

    def test_repeated_refusals_ignore_tick_and_feedback_but_keep_latest_error_visible(self):
        first = observation(entities=[{"id": "execution", "state": {"attempt": 0}}])
        first["_capabilities"]["state_identity"] = {"transient_paths": ["entities.execution"]}
        adaptive = AdaptiveContext(first)
        current = first
        records = []
        for step in range(1, 4):
            after = copy.deepcopy(current)
            after["tick"] = step
            after["events"] = [f"Boundary feedback {step}"]
            after["entities"][0]["state"] = {"attempt": step, "error": "outside boundary"}
            item = record(step, current, "forward", after)
            adaptive.record(item)
            records.append(item)
            current = after
        _, value = build(adaptive, current, records=records)
        summary = adaptive.summary()
        self.assertEqual(summary["executed_controls"], 3)
        self.assertEqual(summary["distinct_observed_states"], 1)
        self.assertEqual(summary["distinct_state_action_pairs"], 1)
        self.assertEqual(summary["repeated_state_action_attempts"], 2)
        self.assertEqual(summary["no_observable_state_change_actions"], 3)
        self.assertEqual(summary["repeated_unchanged_state_action_attempts"], 2)
        self.assertEqual(summary["maximum_unchanged_streak"], 3)
        self.assertEqual(value["observation"]["events"], ["Boundary feedback 3"])
        self.assertEqual(expand_observation(value["observation"])["entities"][0]["state"]["error"], "outside boundary")

    def test_inventory_and_world_changes_make_distinct_experiment_conditions(self):
        first = observation(controls=["use", "toggle"])
        unchanged = copy.deepcopy(first)
        unchanged["tick"] = 1
        changed = copy.deepcopy(unchanged)
        changed["tick"] = 2
        changed["player"]["inventory"].append("tool")
        changed["entities"] = [{"id": "door", "state": {"open": True}}]
        after = copy.deepcopy(changed)
        after["tick"] = 3
        adaptive = AdaptiveContext(first)
        adaptive.record(record(1, first, "use", unchanged))
        adaptive.record(record(2, unchanged, "toggle", changed))
        adaptive.record(record(3, changed, "use", after))
        summary = adaptive.summary()
        self.assertEqual(summary["distinct_state_action_pairs"], 3)
        self.assertEqual(summary["repeated_state_action_attempts"], 0)
        self.assertEqual(summary["no_observable_state_change_actions"], 2)
        self.assertEqual(summary["repeated_unchanged_state_action_attempts"], 0)

    def test_ui_without_position_tracks_screen_state_and_requires_null_prediction(self):
        first = {
            "game": "ui-fixture", "tick": 0, "status": "open",
            "entities": [{"id": "dialog", "kind": "dialog", "state": {"tab": "general", "enabled": True}}],
            "controls": ["select_advanced", "close"], "events": [],
            "_capabilities": {"reset": False},
        }
        second = copy.deepcopy(first)
        second["entities"][0]["state"]["tab"] = "advanced"
        second["tick"] = 1
        adaptive = AdaptiveContext(first)
        adaptive.record(record(1, first, "select_advanced", second))
        _, value = build(adaptive, second)
        self.assertEqual(adaptive.summary()["distinct_positions"], 0)
        self.assertEqual(adaptive.summary()["distinct_observed_states"], 2)
        self.assertIn("Use null", value["response_contract"]["expected_position"])
        self.assertEqual(value["observation"]["controls"], ["select_advanced", "close", "finish"])

    def test_three_dimensional_values_and_adapter_coordinates_are_not_rounded_or_reinterpreted(self):
        current = observation(position=(0.123456789, -2.987654321, 8.765432109), controls=["thrust", "yaw+"])
        current["coordinates"] = "[longitude, altitude, latitude] in adapter units"
        adaptive = AdaptiveContext(current)
        _, value = build(adaptive, current)
        self.assertEqual(value["observation"]["player"]["position"], current["player"]["position"])
        self.assertEqual(value["observation"]["coordinates"], current["coordinates"])
        self.assertIn("Use null", value["response_contract"]["expected_position"])
        self.assertNotIn("horizontal_axes", value["observation"])

    def test_reset_is_counted_separately_and_action_is_not_chosen_by_context(self):
        first, second = observation(position=(0, 0)), observation(position=(1, 0), tick=1)
        adaptive = AdaptiveContext(first)
        adaptive.record(record(1, first, "forward", second))
        fact = adaptive.record(record(2, second, "reset", first, kind="reset"))
        summary = adaptive.summary()
        self.assertEqual(fact["action"], "reset")
        self.assertEqual(summary["reset_use_count"], 1)
        self.assertEqual(summary["native_control_use_counts"]["forward"], 1)
        self.assertNotIn("reset", summary["native_control_use_counts"])
        self.assertEqual(summary["executed_controls"], 2)

    def test_reusing_a_record_is_idempotent_but_conflicting_evidence_fails(self):
        first = observation()
        second = observation(tick=1, facing="West")
        store = CapturingStore()
        adaptive = AdaptiveContext(first, store=store, run_id="run-opaque")
        item = record(1, first, "left", second)
        fact = adaptive.record(item)
        self.assertEqual(adaptive.record(item), fact)
        self.assertEqual(len(store.records), 1)
        self.assertEqual(adaptive.summary()["executed_controls"], 1)
        conflicting = record(1, first, "right", observation(facing="East", tick=1))
        with self.assertRaisesRegex(ValueError, "reused"):
            adaptive.record(conflicting)
        self.assertEqual(adaptive.summary()["executed_controls"], 1)


class MemoryWorkingSetTests(unittest.TestCase):
    def test_persistence_and_queries_use_completed_actual_transitions_and_bounded_retrieval(self):
        first = observation()
        second = observation(tick=1, facing="West")
        store = CapturingStore(facts=[{"action": "forward", "attempts": 2, "citations": [{"run_id": "earlier", "step": 9}]}])
        adaptive = AdaptiveContext(first, store=store, run_id="current-run")
        fact = adaptive.record(record(1, first, "left", second))
        _, value = build(adaptive, second)
        self.assertEqual(fact["citation"], {"run_id": "current-run", "step": 1})
        self.assertEqual(store.records[0][1], "left")
        self.assertEqual(public_observation(store.records[0][2]), public_observation(second))
        self.assertEqual(store.records[0][3:], ("current-run", 1))
        self.assertEqual(store.queries[0][1:], (8, 6000))
        self.assertEqual(value["retrieved_experiments"]["facts"][0]["citations"], [{"run_id": "earlier", "step": 9}])
        self.assertEqual(adaptive.summary()["memory"]["facts_delivered_across_requests"], 1)

    def test_persistence_failure_does_not_lose_executed_action_or_echo_private_path(self):
        first, second = observation(), observation(facing="West", tick=1)
        adaptive = AdaptiveContext(first, store=CapturingStore(fail_record=True, fail_retrieve=True))
        fact = adaptive.record(record(1, first, "left", second))
        _, value = build(adaptive, second)
        summary = adaptive.summary()
        self.assertEqual(fact["action"], "left")
        self.assertEqual(summary["executed_controls"], 1)
        self.assertTrue(value["retrieved_experiments"]["unavailable"])
        self.assertEqual(len(summary["memory"]["warnings"]), 2)
        self.assertNotIn("PRIVATE_SECRET", json.dumps(summary))
        self.assertNotIn("/Users/", json.dumps(summary))

    def test_known_runtime_secret_is_removed_before_model_context_and_persistence(self):
        secret = "opaque-runtime-secret-which-has-no-key-prefix"
        first = observation(events=[f"Game echoed {secret}"])
        first["player"]["visible_message"] = secret
        second = observation(facing="West", tick=1, events=[f"Error containing {secret}"])
        store = CapturingStore(facts=[{"outcome": {"message": secret}, "citations": [{"run_id": "earlier", "step": 1}]}])
        adaptive = AdaptiveContext(first, store=store, secrets=(secret,))
        fact = adaptive.record(record(1, first, "left", second))
        notes = {"n1": LeanNote("n1", f"Maybe {secret} matters", "uncertain", (1,))}
        messages, _ = build(adaptive, second, notes=notes, feedback=secret)
        for value in (messages, fact, store.records, adaptive.summary()):
            self.assertNotIn(secret, json.dumps(value))
        self.assertIn("[REDACTED]", json.dumps(messages))

    def test_current_feedback_survives_history_memory_and_note_pressure(self):
        first = observation()
        last = observation(facing="West", tick=1, events=["CURRENT CRITICAL FEEDBACK: refused at boundary"])
        notes = {f"n{i}": LeanNote(f"n{i}", "hypothesis " + "界" * 190, "uncertain", (1,)) for i in range(6)}
        facts = [{"action": "forward", "outcome": {"description": "old " + "z" * 900}, "citations": [{"run_id": "earlier", "step": i + 1}]} for i in range(80)]
        adaptive = AdaptiveContext(first, store=CapturingStore(facts=facts), max_prompt_bytes=8000)
        messages, value = build(adaptive, last, records=[record(1, first, "left", last)], notes=notes, feedback="The prior response had no legal action.")
        self.assertLessEqual(adaptive_input_bytes(messages, adaptive.schema(adaptive.allowed(last))), 8000)
        self.assertEqual(value["observation"]["events"], last["events"])
        self.assertEqual(value["observation"]["player"]["facing"], "West")
        self.assertIn("prior response", value["validation_feedback"])
        self.assertEqual(value["latest_experiment"]["action"], "left")
        self.assertTrue(value["context_compaction"]["latest_experiment_summary_retained"])
        self.assertGreater(value["context_compaction"]["notes_omitted"], 0)
        self.assertGreater(value["context_compaction"]["memory_omitted"], 0)
        self.assertEqual(value["note_slots_remaining"], 0)
        self.assertEqual(value["context_compaction"]["memory_included"], len(value["retrieved_experiments"]["facts"]))

    def test_thousand_steps_stay_bounded_and_keep_early_experience_in_aggregate(self):
        first = observation()
        adaptive = AdaptiveContext(first, run_id="long-run")
        records, current = [], first
        for step in range(1, 1001):
            after = observation(tick=step, events=["Observed boundary refusal"])
            item = record(step, current, "forward", after)
            records.append(item)
            current = after
        messages, value = build(adaptive, current, records=records)
        schema = adaptive.schema(adaptive.allowed(current))
        self.assertLessEqual(adaptive_input_bytes(messages, schema), 24000)
        self.assertGreaterEqual(value["context_compaction"]["history_omitted"], 992)
        self.assertLessEqual(len(value["factual_log"]), 8)
        self.assertEqual(value["latest_experiment"]["citation"]["step"], 1000)
        self.assertEqual(value["exploration_coverage"]["prior_attempts_in_current_state"][0]["attempts"], 1000)
        self.assertEqual(adaptive.summary()["repeated_unchanged_state_action_attempts"], 999)
        self.assertEqual(adaptive.summary()["distinct_observed_states"], 1)
        # The caller can ask again with the same records without double counting.
        build(adaptive, current, records=records)
        self.assertEqual(adaptive.summary()["executed_controls"], 1000)


class CurrentObservationAndBudgetTests(unittest.TestCase):
    def test_repeated_entity_keys_compact_losslessly_including_dynamic_state(self):
        entities = [{
            "id": f"cell-{i}", "kind": "cell", "position": [i % 10, i // 10],
            "state": {"builtHeight": i % 3, "targetHeight": 2, "visited": i % 2 == 0, "bomb": False, "rock": False},
        } for i in range(100)]
        entities += [{"id": "drone", "state": {"facing": "West", "variableV": 7}}, {"id": "execution", "state": {"error": "last failure"}}]
        current = observation(entities=entities)
        adaptive = AdaptiveContext(current)
        messages, value = build(adaptive, current)
        expected = public_observation(current)
        expected["controls"] = adaptive.allowed(current)
        self.assertEqual(expand_observation(value["observation"]), expected)
        self.assertLess(len(json.dumps(value["observation"])), len(json.dumps(expected)))
        self.assertLessEqual(adaptive_input_bytes(messages, adaptive.schema(adaptive.allowed(current))), 24000)
        self.assertEqual(len(expand_observation(value["observation"])["entities"]), 102)

    def test_compaction_preserves_missing_empty_null_order_and_reserved_keys(self):
        values = [{"id": i, "nested": {"a": None, "b": {}}, "items": [1, 2, 3]} for i in range(20)]
        values += [{"id": 21, "nested": {"b": {}}}, {"id": 22, "$object_groups": {"original": True}}, {"id": 23, "$literal_object": []}]
        compact = compact_observation(values)
        self.assertEqual(expand_observation(compact), values)
        self.assertEqual(expand_observation(compact_observation({"$object_groups": "literal"})), {"$object_groups": "literal"})

    def test_byte_ceiling_counts_unicode_escaping_and_actual_response_schema_envelope(self):
        current = observation(events=["反馈" * 100])
        adaptive = AdaptiveContext(current, max_prompt_bytes=8000)
        messages, _ = build(adaptive, current)
        schema = adaptive.schema(adaptive.allowed(current))
        envelope = {"messages": messages, "response_format": {"type": "json_schema", "json_schema": {"name": "game_decision", "strict": True, "schema": schema}}}
        expected = len(json.dumps(envelope, separators=(",", ":"), allow_nan=False).encode("utf-8"))
        self.assertEqual(adaptive_input_bytes(messages, schema), expected)
        self.assertLessEqual(expected, 8000)
        self.assertGreater(expected, len(json.dumps(messages, separators=(",", ":")).encode("utf-8")))

    def test_essential_current_observation_that_cannot_fit_fails_visibly_without_slicing(self):
        current = observation(events=["visible feedback " * 5000])
        original = copy.deepcopy(current)
        adaptive = AdaptiveContext(current)
        with self.assertRaises(AdaptivePromptBudgetExceeded) as raised:
            build(adaptive, current)
        self.assertGreater(raised.exception.required_bytes, 24000)
        self.assertIn("No inference request was made", str(raised.exception))
        self.assertEqual(current, original)
        self.assertEqual(adaptive.summary()["context_compaction"]["requests_built"], 0)
        self.assertEqual(adaptive.summary()["context_compaction"]["prompt_budget_failures"], 1)

    def test_large_action_schema_must_fit_and_never_drops_legal_controls(self):
        controls = [f"native-action-{i}-" + "x" * 200 for i in range(100)]
        current = observation(controls=controls)
        adaptive = AdaptiveContext(current)
        self.assertEqual(adaptive.allowed(current), [*controls, "reset", "finish"])
        self.assertEqual(adaptive.schema(adaptive.allowed(current))["properties"]["action"]["enum"], [*controls, "reset", "finish"])
        with self.assertRaises(AdaptivePromptBudgetExceeded):
            build(adaptive, current)

    def test_rejects_invalid_budgets_and_run_ids(self):
        for budget in (7999, 0, True, 24000.0):
            with self.subTest(budget=budget), self.assertRaises(ValueError):
                AdaptiveContext(observation(), max_prompt_bytes=budget)
        with self.assertRaises(ValueError):
            AdaptiveContext(observation(), run_id="")

    def test_four_field_contract_remains_legacy_compatible(self):
        current = observation(controls=["a", "b"])
        adaptive = AdaptiveContext(current)
        self.assertEqual(adaptive.schema(adaptive.allowed(current)), lean_schema(["a", "b", "reset", "finish"]))


if __name__ == "__main__":
    unittest.main()
