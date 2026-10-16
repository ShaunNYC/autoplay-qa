"""Factual v2 agendas, truthful missing data, prediction checks and v1 goldens."""

import copy
import hashlib
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest

from autoplay_qa.adaptive import AdaptiveContext, adaptive_input_bytes, expand_observation, native_controls
from autoplay_qa.adaptive_v2 import AdaptiveContextV2, ADAPTIVE_V2_PROMPT_VERSION, EXPERIMENT_INDEX_VERSION
from autoplay_qa.experience import ExperienceStore, public_observation, state_key
from tests.test_adaptive_context import CapturingStore, build, observation, record


def indexed_control(action="forward", *, attempts=2, state_attempts=2, unchanged=2, outcomes=1):
    return {
        "action": action, "attempts": attempts,
        "current_state": {"attempts": state_attempts, "unchanged_outcomes": unchanged, "distinct_outcomes": outcomes},
        "citations": {"any_state": [{"run_id": "prior-run", "step": 1}],
                      "current_state": [{"run_id": "prior-run", "step": 1}] if state_attempts else []},
    }


def index_packet(current, run_id, *, rows=None, complete=True, omitted=0, transitions=None):
    rows = copy.deepcopy(rows or [])
    transitions = sum(row["attempts"] for row in rows) + omitted if transitions is None else transitions
    return {
        "version": EXPERIMENT_INDEX_VERSION, "scope_id": "a" * 64,
        "current_state_id": state_key(current), "excluded_run_id": run_id,
        "control_scope": "currently_legal_native_controls",
        "prior_run_count": 1 if transitions else 0, "prior_transition_count": transitions,
        "current_native_control_count": len(native_controls(current)), "controls": rows,
        "counts_complete": complete, "omitted_nonzero_controls": omitted,
        "zero_lists_complete": False,
    }


class IndexStore(CapturingStore):
    def __init__(self, factory=None, *, sqlite_failure=False):
        super().__init__()
        self.factory = factory or (lambda current, run_id: index_packet(current, run_id))
        self.index_queries = []
        self.sqlite_failure = sqlite_failure

    def experiment_index(self, current, *, exclude_run_id, max_bytes):
        self.index_queries.append((copy.deepcopy(current), exclude_run_id, max_bytes))
        if self.sqlite_failure:
            raise sqlite3.OperationalError("database locked at /Users/private/experience.sqlite")
        return self.factory(current, exclude_run_id)


def agenda(value):
    return value["exploration_coverage"]["experiment_agenda"]


class AdaptiveV1GoldenTests(unittest.TestCase):
    def test_v1_initial_messages_are_byte_identical_to_committed_live_version(self):
        current = observation()
        messages, value = build(AdaptiveContext(current, run_id="golden-v1"), current)
        encoded = json.dumps(messages, separators=(",", ":")).encode()
        self.assertEqual(hashlib.sha256(encoded).hexdigest(), "6d8a6ea3623555014308a18a9ba93deb109841ec9174fe38a6439f45d0702e76")
        self.assertNotIn("experiment_agenda", value["exploration_coverage"])
        self.assertNotIn("experiment_agenda_omitted", value["context_compaction"])

    def test_v1_history_and_memory_messages_are_byte_identical(self):
        first = observation()
        second = observation(facing="West", tick=1, events=["Turned left"])
        store = CapturingStore(facts=[{"action": "forward", "attempts": 2, "citations": [{"run_id": "earlier", "step": 9}]}])
        context = AdaptiveContext(first, store=store, run_id="golden-v1")
        messages, _ = build(context, second, records=[record(1, first, "left", second)])
        encoded = json.dumps(messages, separators=(",", ":")).encode()
        self.assertEqual(hashlib.sha256(encoded).hexdigest(), "ec568d73c8f2f89796b6fa6875aeb2169e07dc9ad4c6dbb50997f8c0d42f765a")


class AdaptiveV2AgendaTests(unittest.TestCase):
    def test_version_contract_catalog_and_current_state_stay_complete(self):
        current = observation()
        context = AdaptiveContextV2(current, store=IndexStore(), run_id="current-run")
        messages, value = build(context, current)
        self.assertEqual(context.prompt_version, ADAPTIVE_V2_PROMPT_VERSION)
        self.assertEqual(value["prompt_version"], "adaptive-v2")
        self.assertEqual(context.summary()["prompt_version"], "adaptive-v2")
        self.assertEqual(expand_observation(value["observation"]), public_observation(current))
        self.assertEqual(context.schema(context.allowed(current))["properties"]["action"]["enum"], [*current["controls"], "reset", "finish"])
        self.assertEqual(set(context.schema(context.allowed(current))["properties"]), {"action", "reason", "expected_position", "notes"})
        self.assertLessEqual(adaptive_input_bytes(messages, context.schema(context.allowed(current))), 24000)
        self.assertIn("successful route needs a concrete new question", messages[0]["content"])

    def test_complete_sparse_prior_counts_produce_complete_cumulative_untested_lists(self):
        first = observation()
        after = observation(tick=1, events=["Refused"])
        source = IndexStore(lambda current, run_id: index_packet(current, run_id, rows=[indexed_control()]))
        context = AdaptiveContextV2(first, store=source, run_id="current-run")
        context.record(record(1, first, "forward", after))
        _, value = build(context, after)
        data = agenda(value)
        self.assertTrue(data["prior_counts_complete"])
        self.assertEqual(data["untested_controls_across_known_runs"], ["left", "right", "place", "remove", "bomb"])
        self.assertNotIn("untested_controls_this_run", data)
        row = data["tested_in_current_visible_state"][0]
        self.assertEqual(row["control"], "forward")
        self.assertEqual(row["this_run"]["attempts"], 1)
        self.assertEqual(row["prior_runs"]["attempts"], 2)
        self.assertEqual(row["known_total_attempts"], 3)
        self.assertEqual(row["this_run"]["distinct_outcomes"], 1)
        self.assertEqual(row["this_run"]["citations"], [{"run_id": "current-run", "step": 1}])
        self.assertEqual(row["prior_runs"]["citations"], [{"run_id": "prior-run", "step": 1}])
        self.assertEqual(source.index_queries[0][1:], ("current-run", 4000))

    def test_incomplete_index_missing_controls_remain_unknown(self):
        first = observation()
        after = observation(tick=1)
        source = IndexStore(lambda current, run_id: index_packet(current, run_id, rows=[indexed_control()], complete=False, omitted=1))
        context = AdaptiveContextV2(first, store=source, run_id="current-run")
        context.record(record(1, first, "right", after))
        _, value = build(context, after)
        data = agenda(value)
        self.assertFalse(data["prior_counts_complete"])
        self.assertIsNone(data["untested_controls_across_known_runs"])
        self.assertIsNone(data["untested_controls_in_current_visible_state"])
        self.assertEqual(value["exploration_coverage"]["native_control_use_counts"]["right"], 1)
        self.assertEqual(data["prior_run_index"]["controls"][0]["attempts"], 2)
        right = next(row for row in data["tested_in_current_visible_state"] if row["control"] == "right")
        self.assertIsNone(right["prior_runs"])
        self.assertIsNone(right["known_total_attempts"])

    def test_no_store_has_explicit_unknown_prior_history_instead_of_false_zeros(self):
        current = observation()
        _, value = build(AdaptiveContextV2(current), current)
        data = agenda(value)
        self.assertEqual(data["prior_run_index"]["status"], "not_configured")
        self.assertEqual(data["untested_controls_this_run"], current["controls"])
        self.assertIsNone(data["untested_controls_across_known_runs"])
        self.assertFalse(data["prior_counts_complete"])
        self.assertNotIn("controls", data["prior_run_index"])

    def test_writable_store_excludes_current_run_before_combining_exactly_once(self):
        first = observation()
        after1 = observation(tick=1, events=["Refused"])
        after2 = observation(tick=2, events=["Refused"])
        with tempfile.TemporaryDirectory() as directory:
            with ExperienceStore(Path(directory) / "memory.sqlite", "project", "build", first) as store:
                store.record(first, "forward", after1, run_id="prior-run", step=1)
                store.record(after1, "forward", after2, run_id="prior-run", step=2)
                context = AdaptiveContextV2(first, store=store, run_id="current-run")
                context.record(record(1, first, "forward", after1))
                _, value = build(context, after1)
                data = agenda(value)
                self.assertEqual(store.summary()["transitions"], 3)
                self.assertEqual(data["prior_run_index"]["prior_transition_count"], 2)
                self.assertEqual(value["exploration_coverage"]["native_control_use_counts"]["forward"], 1)
                self.assertEqual(data["prior_run_index"]["controls"][0]["attempts"], 2)
                self.assertEqual(data["tested_in_current_visible_state"][0]["known_total_attempts"], 3)
                self.assertEqual({citation["run_id"] for row in data["prior_run_index"]["controls"] for citation in row["citations"]["any_state"]}, {"prior-run"})

    def test_read_only_evaluation_reuses_training_without_database_writes(self):
        first, after = observation(), observation(tick=1, events=["Refused"])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "memory.sqlite"
            with ExperienceStore(path, "project", "build", first) as store:
                store.record(first, "forward", after, run_id="prior-run", step=1)
            before_hash = hashlib.sha256(path.read_bytes()).hexdigest()
            with ExperienceStore(path, "project", "build", first, read_only=True) as store:
                context = AdaptiveContextV2(first, store=store, run_id="evaluation-run")
                context.record(record(1, first, "forward", after))
                _, value = build(context, after)
                self.assertEqual(store.summary()["transitions"], 1)
                self.assertEqual(agenda(value)["tested_in_current_visible_state"][0]["known_total_attempts"], 2)
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), before_hash)

    def test_turned_heading_creates_new_conditions_not_a_failed_move(self):
        first, after = observation(), observation(facing="West", tick=1)
        context = AdaptiveContextV2(first, store=IndexStore(), run_id="current-run")
        context.record(record(1, first, "left", after))
        _, value = build(context, after)
        self.assertEqual(context.summary()["no_observable_state_change_actions"], 0)
        self.assertEqual(agenda(value)["tested_in_current_visible_state"], [])
        self.assertEqual(agenda(value)["untested_controls_in_current_visible_state"], first["controls"])
        self.assertNotIn("left", agenda(value)["untested_controls_across_known_runs"])

    def test_terminal_empty_native_catalog_does_not_claim_whole_game_coverage(self):
        current = observation()
        current["status"] = "won"
        context = AdaptiveContextV2(current, store=IndexStore(), run_id="current-run")
        _, value = build(context, current)
        self.assertEqual(value["observation"]["controls"], ["reset", "finish"])
        self.assertEqual(agenda(value)["untested_controls_across_known_runs"], [])
        self.assertIn("no whole-game completeness claim", agenda(value)["control_scope"])


class AdaptiveV2PredictionTests(unittest.TestCase):
    def test_prediction_mismatch_is_own_forecast_and_has_actual_step_provenance(self):
        first = observation()
        after = observation(tick=1, events=["Refused at boundary"])
        item = record(1, first, "forward", after)
        item["decision"]["expected_position"] = [0, -1]
        context = AdaptiveContextV2(first)
        context.record(item)
        _, value = build(context, after, records=[item])
        prediction = agenda(value)["own_prediction_check"]
        self.assertEqual(prediction["status"], "mismatch")
        self.assertEqual(prediction["expected_position"], [0, -1])
        self.assertEqual(prediction["actual_position"], [0, 0])
        self.assertEqual(prediction["citation"], {"run_id": "session", "step": 1})
        self.assertIn("not defect truth", prediction["meaning"])
        self.assertEqual(context.summary()["experiment_agenda"]["prediction_checks"]["mismatch"], 1)
        self.assertNotIn("verdict", prediction)

    def test_matching_and_noncomparable_forecasts_are_distinguished(self):
        first = observation()
        matched = record(1, first, "left", observation(tick=1, facing="West"))
        matched["decision"]["expected_position"] = [0, 0]
        context = AdaptiveContextV2(first)
        context.record(matched)
        self.assertEqual(context.summary()["experiment_agenda"]["prediction_checks"]["matched"], 1)
        world3d = observation(position=(0.5, 1.25, 2.75))
        next3d = observation(position=(1.0, 1.25, 2.75), tick=1)
        wrong_contract = record(1, world3d, "forward", next3d)
        wrong_contract["decision"]["expected_position"] = [1, 2]
        context3d = AdaptiveContextV2(world3d)
        context3d.record(wrong_contract)
        _, value = build(context3d, next3d)
        self.assertEqual(agenda(value)["own_prediction_check"]["status"], "not_comparable")
        self.assertEqual(agenda(value)["own_prediction_check"]["actual_position"], [1.0, 1.25, 2.75])


class AdaptiveV2PacketAndBudgetTests(unittest.TestCase):
    def test_invalid_external_packets_become_unknown_without_secret_or_path_leaks(self):
        first = observation(controls=["forward"])
        secret = "opaque-runtime-secret-for-v2"

        def mutate(kind):
            def factory(current, run_id):
                packet = index_packet(current, run_id, rows=[indexed_control()])
                if kind == "secret":
                    packet["controls"][0]["citations"]["any_state"][0]["run_id"] = secret
                elif kind == "private_metadata":
                    packet["source_path"] = "/Users/private/source.ts"
                elif kind == "current_run":
                    packet["controls"][0]["citations"]["current_state"][0]["run_id"] = run_id
                elif kind == "wrong_state":
                    packet["current_state_id"] = "b" * 64
                elif kind == "bool_catalog":
                    packet["current_native_control_count"] = True
                elif kind == "zero_runs":
                    packet["prior_run_count"] = 0
                elif kind == "credential_pattern":
                    packet["controls"][0]["citations"]["any_state"][0]["run_id"] = "sk-abcdefghijklmnopqrstuv"
                elif kind == "unknown_count":
                    packet["controls"][0]["current_state"]["attempts"] = 9
                return packet
            return factory

        for kind in ("secret", "private_metadata", "current_run", "wrong_state", "bool_catalog", "zero_runs", "credential_pattern", "unknown_count"):
            with self.subTest(kind=kind):
                context = AdaptiveContextV2(first, store=IndexStore(mutate(kind)), run_id="current-run", secrets=(secret,))
                messages, value = build(context, first)
                data = agenda(value)
                self.assertEqual(data["prior_run_index"]["status"], "unavailable")
                self.assertIsNone(data["untested_controls_across_known_runs"])
                text = json.dumps(messages)
                self.assertNotIn(secret, text)
                self.assertNotIn("/Users/private/", text)
                self.assertNotIn("sk-abcdefghijklmnopqrstuv", text)
                self.assertEqual(context.summary()["memory"]["warnings"][0]["code"], "experiment_index_unavailable")

    def test_sqlite_index_failure_retains_local_action_and_unknown_prior_counts(self):
        first, after = observation(), observation(tick=1)
        context = AdaptiveContextV2(first, store=IndexStore(sqlite_failure=True))
        context.record(record(1, first, "forward", after))
        _, value = build(context, after)
        self.assertEqual(context.summary()["executed_controls"], 1)
        self.assertEqual(agenda(value)["prior_run_index"]["status"], "unavailable")
        self.assertEqual(agenda(value)["prior_run_index"]["error_type"], "OperationalError")
        self.assertNotIn("/Users/private/", json.dumps(value))

    def test_eight_kib_budget_truthfully_omits_agenda_but_keeps_all_current_controls(self):
        first = observation(events=["Visible feedback " + "x" * 300])
        context = AdaptiveContextV2(first, store=IndexStore(), run_id="current-run", max_prompt_bytes=8000)
        messages, value = build(context, first)
        self.assertLessEqual(adaptive_input_bytes(messages, context.schema(context.allowed(first))), 8000)
        self.assertEqual(expand_observation(value["observation"]), public_observation(first))
        self.assertTrue(value["context_compaction"]["experiment_agenda_omitted"])
        self.assertEqual(agenda(value)["status"], "omitted_context_budget")
        self.assertFalse(agenda(value)["prior_run_index"]["counts_complete"])
        self.assertNotIn("untested_controls_across_known_runs", agenda(value))
        self.assertEqual(context.summary()["experiment_agenda"]["omitted_requests"], 1)

    def test_large_native_catalog_never_becomes_a_truncated_zero_list(self):
        first = observation(controls=[f"native-{index}-" + "x" * 40 for index in range(80)])
        context = AdaptiveContextV2(first, store=IndexStore(), max_prompt_bytes=24000)
        messages, value = build(context, first)
        self.assertLessEqual(adaptive_input_bytes(messages, context.schema(context.allowed(first))), 24000)
        self.assertEqual(value["observation"]["controls"], [*first["controls"], "reset", "finish"])
        data = agenda(value)
        if data["status"] == "available":
            self.assertEqual(data["untested_controls_across_known_runs"], first["controls"])
        else:
            self.assertEqual(data["status"], "omitted_context_budget")
            self.assertNotIn("untested_controls_across_known_runs", data)

    def test_retained_external_packets_reconstruct_exact_context_including_omission(self):
        for maximum in (8000, 24000):
            first = observation()
            source = IndexStore(lambda current, run_id: index_packet(current, run_id, rows=[indexed_control()]))
            original = AdaptiveContextV2(first, store=source, run_id="replayed-run", max_prompt_bytes=maximum)
            messages, value = build(original, first)

            class RecordedPacket:
                def retrieve(self, *args, **kwargs):
                    return copy.deepcopy(value["retrieved_experiments"])

                def experiment_index(self, *args, **kwargs):
                    return copy.deepcopy(agenda(value)["prior_run_index"])

            reproduced = AdaptiveContextV2(first, store=RecordedPacket(), run_id="replayed-run", max_prompt_bytes=maximum)
            actual = reproduced.messages(first, 0, {}, [], {"actions": 24, "requests": 28, "seconds": 300}, focus="qa",
                                         replay_compaction=value["context_compaction"])
            self.assertEqual(actual, messages)

    def test_replayed_cold_not_configured_packet_does_not_become_available_history(self):
        first = observation()
        original = AdaptiveContextV2(first, run_id="cold-run")
        messages, value = build(original, first)

        class RecordedPacket:
            def retrieve(self, *args, **kwargs):
                return copy.deepcopy(value["retrieved_experiments"])

            def experiment_index(self, *args, **kwargs):
                return copy.deepcopy(agenda(value)["prior_run_index"])

        replayed = AdaptiveContextV2(first, store=RecordedPacket(), run_id="cold-run")
        actual = replayed.messages(first, 0, {}, [], {"actions": 24, "requests": 28, "seconds": 300}, focus="qa",
                                   replay_compaction=value["context_compaction"])
        self.assertEqual(actual, messages)
        self.assertIsNone(agenda(json.loads(actual[1]["content"]))["untested_controls_across_known_runs"])


if __name__ == "__main__":
    unittest.main()
