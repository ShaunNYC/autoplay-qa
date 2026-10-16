"""Durable observed facts and bounded retrieval, with no inference calls."""

from copy import deepcopy
import json
import os
from pathlib import Path
import sqlite3
import tempfile
import unittest

from autoplay_qa.experience import (
    EvidenceConflict, ExperienceStore, MAX_FACT_BYTES, ScopeMismatch,
    experiment_fact, public_observation, state_key,
)


def observation(**overrides):
    value = {
        "game": "native-fixture-v2", "tick": 0, "status": "playing",
        "grid": ["@..", "..."], "coordinates": "Zero-based [x,y], y south",
        "legend": {"@": "player", ".": "floor"},
        "player": {"position": [0, 0], "facing": "North", "inventory": []},
        "entities": [{"id": "gate", "state": {"open": False}},
                     {"id": "execution", "state": {"attempts": 0, "error": None}}],
        "events": ["Ready"], "controls": ["forward", "left", "right", "place"],
        "_capabilities": {"reset": True, "state_identity": {"transient_paths": ["entities.execution"]}},
    }
    value.update(overrides)
    return value


def result(before, *, facing=None, position=None, event="No change"):
    value = deepcopy(before)
    value["tick"] += 1
    value["events"] = [event]
    if facing is not None:
        value["player"]["facing"] = facing
    if position is not None:
        value["player"]["position"] = position
    return value


class PublicExperienceTests(unittest.TestCase):
    def test_native_controls_and_feedback_no_keygate_hints(self):
        original = observation()
        original["legend"]["D"] = "native description"
        original["entities"][-1]["state"]["error"] = "Boundary rejected"
        view = public_observation(original)
        self.assertEqual(view["controls"], ["forward", "left", "right", "place", "reset", "finish"])
        self.assertNotIn("neighbors", view)
        self.assertEqual(view["legend"]["D"], "native description")
        self.assertEqual(view["events"], ["Ready"])
        self.assertEqual(view["entities"][-1]["state"]["error"], "Boundary rejected")
        view["player"]["facing"] = "South"
        self.assertEqual(original["player"]["facing"], "North")

    def test_meta_controls_follow_capabilities_and_native_action_priority(self):
        obs = observation(actions=["tap", "tap", "finish", "reset"], controls=["wrong"])
        obs["_capabilities"]["reset"] = False
        self.assertEqual(public_observation(obs)["controls"], ["tap", "finish"])
        obs["status"] = "won"
        self.assertEqual(public_observation(obs)["controls"], ["finish"])
        fallback = observation(actions=None, controls=["tap"])
        self.assertEqual(public_observation(fallback)["controls"], ["tap", "reset", "finish"])

    def test_recursive_private_metadata_and_echoed_secrets_removed(self):
        obs = observation()
        obs["rules"] = ["do not send"]
        obs["evaluation"] = {"injected_defect": "wall-clip"}
        obs["api_key"] = "opaque-sensitive-key-value"
        obs["player"]["private_path"] = "/Users/person/private/game"
        obs["entities"][0]["state"].update({
            "developer_rules": ["win by left"], "ground_truth": "bug",
            "nested": {"API-KEY": "another-sensitive-value", "_private": "hidden", "evaluation": {"label": "expected"}},
        })
        obs["events"] = ["opaque-sensitive-key-value another-sensitive-value", "Bearer abcd12345678", "/Users/person/secret/file.ts"]
        text = json.dumps(public_observation(obs))
        for secret in ("opaque-sensitive-key-value", "another-sensitive-value", "abcd12345678", "/Users/person", "developer_rules", "ground_truth", "expected", "wall-clip", "private_path"):
            self.assertNotIn(secret, text)
        self.assertIn("[REDACTED]", text)

    def test_legitimate_ui_labels_inventory_tokens_and_game_paths_preserved(self):
        obs = observation()
        obs["player"]["inventory"] = [{"token": "bronze-token", "label": "A door token"}]
        obs["entities"][0]["state"].update({"label": "Open", "labels": ["available"], "source": "toolbar", "path": [[0, 0], [1, 0]]})
        view = public_observation(obs)
        self.assertEqual(view["player"]["inventory"], obs["player"]["inventory"])
        for key in ("label", "labels", "source", "path"):
            self.assertEqual(view["entities"][0]["state"][key], obs["entities"][0]["state"][key])

    def test_unique_entity_reordering_does_not_invent_state_novelty(self):
        before = observation()
        before["entities"].append({"id": "lamp", "state": {"lit": False}})
        after = deepcopy(before)
        after["entities"].reverse()
        after["controls"].reverse()
        self.assertEqual(state_key(before), state_key(after))

    def test_state_identity_excludes_tick_feedback_and_declared_transients_only(self):
        before = observation()
        after = result(before)
        after["entities"][-1]["state"] = {"attempts": 99, "error": "Boundary"}
        self.assertEqual(state_key(before), state_key(after))
        self.assertNotEqual(public_observation(before), public_observation(after))
        for mutate in (
            lambda obs: obs["player"].update(facing="East"),
            lambda obs: obs["player"].update(inventory=["key"]),
            lambda obs: obs["entities"][0]["state"].update(open=True),
            lambda obs: obs["grid"].__setitem__(1, ".#."),
        ):
            changed = deepcopy(before)
            mutate(changed)
            self.assertNotEqual(state_key(before), state_key(changed))

    def test_wildcard_paths_select_dictionary_keys_and_entity_ids(self):
        before = observation(objects={"cell-1": {"visited": False, "bomb": False}})
        before["entities"].append({"id": "cell-1", "state": {"visited": False, "built": 0}})
        before["_capabilities"]["state_identity"]["transient_paths"] += ["objects.cell-*.visited", "entities.cell-*.state.visited"]
        after = result(before)
        after["objects"]["cell-1"]["visited"] = True
        after["entities"][-1]["state"]["visited"] = True
        self.assertEqual(state_key(before), state_key(after))
        after["objects"]["cell-1"]["bomb"] = True
        self.assertNotEqual(state_key(before), state_key(after))

    def test_json_booleans_and_numbers_are_different_states(self):
        before = observation()
        after = deepcopy(before)
        before["player"]["alive"] = True
        after["player"]["alive"] = 1
        self.assertNotEqual(state_key(before), state_key(after))

    def test_fact_retains_factual_outcome_without_bug_classification(self):
        before = observation()
        after = result(before, event="Boundary blocks forward")
        fact = experiment_fact(before, "forward", after, run_id="session-A", step=7)
        self.assertFalse(fact["state_changed"])
        self.assertFalse(fact["position_changed"])
        self.assertEqual(fact["changed_facets"], [])
        self.assertEqual(fact["citation"], {"run_id": "session-A", "step": 7})
        self.assertEqual(fact["outcome"]["events"], ["Boundary blocks forward"])
        self.assertNotIn("bug", json.dumps(fact))

    def test_turning_without_motion_is_a_real_state_change(self):
        before = observation()
        fact = experiment_fact(before, "right", result(before, facing="East"), run_id="session-B", step=1)
        self.assertTrue(fact["state_changed"])
        self.assertFalse(fact["position_changed"])
        self.assertIn("player", fact["changed_facets"])
        self.assertTrue(any(item["path"] == "player.facing" for item in fact["outcome"]["changes"]))

    def test_large_outcome_has_bounded_bytes_and_omission_receipts(self):
        before = observation()
        after = result(before)
        before["player"]["huge"] = {str(n): "旧" * 400 for n in range(100)}
        after["player"]["huge"] = {str(n): "新" * 400 for n in range(100)}
        after["events"] = ["巨" * 500] * 20
        fact = experiment_fact(before, "place", after, run_id="large", step=1)
        self.assertLessEqual(len(json.dumps(fact, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()), MAX_FACT_BYTES)
        self.assertGreater(fact["outcome"]["omitted_changes"], 0)

    def test_invalid_control_or_private_evidence_reference_rejected(self):
        before = observation()
        for action, run, step in (("north", "s", 1), ("forward", "/Users/me/run.jsonl", 1), ("forward", "s", 0), ("forward", "s", True)):
            with self.assertRaises(ValueError):
                experiment_fact(before, action, result(before), run_id=run, step=step)


class ExperienceStoreTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="experience-tests-")
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "facts.sqlite3"
        self.initial = observation()

    def store(self, *, path=None, project="project", version="v1", initial=None, **options):
        store = ExperienceStore(path or self.path, project, version, initial or self.initial, **options)
        self.addCleanup(store.close)
        return store

    def test_restart_preserves_exact_facts_and_disambiguates_run_step_citations(self):
        store = self.store()
        store.record(self.initial, "forward", result(self.initial), "run-one", 1)
        store.record(self.initial, "forward", result(self.initial), "run-two", 1)
        expected = store.summary()
        store.close()
        reopened = self.store()
        self.assertEqual(reopened.summary(), expected)
        memory = reopened.retrieve(self.initial)
        self.assertEqual(len(memory["facts"]), 1)
        fact = memory["facts"][0]
        self.assertEqual(fact["attempts"], 2)
        self.assertEqual(fact["unchanged_count"], 2)
        self.assertEqual({item["run_id"] for item in fact["citations"]}, {"run-one", "run-two"})
        self.assertEqual(reopened.summary()["repeated_unchanged_attempts"], 1)

    def test_counts_novelty_idempotency_and_conflicting_citations(self):
        store = self.store()
        first = store.record(self.initial, "forward", result(self.initial), "run", 1)
        self.assertEqual(first["same_state_action_prior_attempts"], 0)
        self.assertFalse(first["new_after_state"])
        second = store.record(self.initial, "forward", result(self.initial), "run", 2)
        self.assertEqual(second["prior_equivalent_unchanged_attempts"], 1)
        third = store.record(self.initial, "right", result(self.initial, facing="East"), "run", 3)
        self.assertTrue(third["new_after_state"])
        store.record(self.initial, "forward", result(self.initial), "run", 1)
        self.assertEqual(store.summary()["transitions"], 3)
        with self.assertRaises(EvidenceConflict):
            store.record(self.initial, "right", result(self.initial, facing="South"), "run", 1)

    def test_multiple_outcomes_kept_without_universal_rule_claim(self):
        store = self.store()
        store.record(self.initial, "forward", result(self.initial, event="No movement"), "one", 1)
        store.record(self.initial, "forward", result(self.initial, position=[1, 0], event="Moved"), "two", 1)
        facts = store.retrieve(self.initial)["facts"]
        self.assertEqual(len(facts), 2)
        self.assertTrue(all(fact["different_outcomes_for_action"] == 2 for fact in facts))
        self.assertEqual({fact["state_changed"] for fact in facts}, {True, False})
        self.assertNotIn("confirmed", json.dumps(facts))

    def test_long_feedback_outcomes_not_collapsed_by_display_truncation(self):
        store = self.store()
        for number, suffix in enumerate(("accepted", "rejected"), start=1):
            store.record(self.initial, "forward", result(self.initial, event="prefix" * 100 + suffix), "run", number)
        facts = store.retrieve(self.initial)["facts"]
        self.assertEqual(len(facts), 2)
        self.assertEqual(facts[0]["different_outcomes_for_action"], 2)

    def test_current_state_precedes_recent_unrelated_history(self):
        store = self.store()
        store.record(self.initial, "forward", result(self.initial), "run", 1)
        elsewhere = result(self.initial, position=[1, 0])
        store.record(elsewhere, "right", result(elsewhere, facing="East"), "run", 2)
        facts = store.retrieve(self.initial, limit=1)["facts"]
        self.assertEqual(len(facts), 1)
        self.assertTrue(facts[0]["matching_current_state"])
        self.assertEqual(facts[0]["action"], "forward")

    def test_project_version_scene_controls_and_coordinate_scopes_isolated(self):
        original = self.store()
        original.record(self.initial, "forward", result(self.initial), "run", 1)
        other_scene = deepcopy(self.initial)
        other_scene["grid"][1] = ".#."
        other_controls = observation(controls=["tap"])
        other_frame = observation(coordinates="[row,column]")
        for options in ({"project": "other"}, {"version": "v2"}, {"initial": other_scene}, {"initial": other_controls}, {"initial": other_frame}):
            separate = self.store(**options)
            self.assertNotEqual(separate.scope_id, original.scope_id)
            self.assertEqual(separate.summary()["transitions"], 0)
            self.assertEqual(separate.retrieve(options.get("initial", self.initial))["facts"], [])
        self.assertEqual(original.summary()["transitions"], 1)

    def test_runtime_scope_mismatch_rejected_without_adding_evidence(self):
        store = self.store()
        for changes in ({"game": "different"}, {"coordinates": "other"}, {"grid": ["@...."]}, {"controls": ["other"]}):
            foreign = observation(**changes)
            with self.assertRaises(ScopeMismatch):
                store.record(self.initial, "forward", foreign, "run", 1)
            with self.assertRaises(ScopeMismatch):
                store.retrieve(foreign)
        self.assertEqual(store.summary()["transitions"], 0)

    def test_world_changes_and_won_control_narrowing_are_allowed(self):
        store = self.store()
        after = result(self.initial, position=[1, 0])
        after["grid"] = [".@.", ".#."]
        after["entities"][0]["state"]["open"] = True
        after["status"] = "won"
        after["controls"] = []
        store.record(self.initial, "forward", after, "run", 1)
        self.assertEqual(store.summary()["transitions"], 1)
        self.assertEqual(store.retrieve(after)["current_state_id"], state_key(after))

    def test_feedback_durable_reviewer_only_and_append_revision(self):
        store = self.store()
        store.record(self.initial, "forward", result(self.initial), "run", 1)
        before_query = store.retrieve(self.initial)
        before_summary = store.summary()
        store.record_feedback("run", 1, "pending", comment="Needs a developer review")
        store.record_feedback("run", 1, "expected", comment="Ordinary boundary", reviewer="developer")
        self.assertEqual(store.retrieve(self.initial), before_query)
        self.assertEqual(store.summary(), before_summary)
        feedback = store.reviewer_feedback()
        self.assertEqual([item["status"] for item in feedback], ["expected", "pending"])
        self.assertEqual(feedback[0]["comment"], "Ordinary boundary")
        store.close()
        self.assertEqual(len(self.store().reviewer_feedback()), 2)

    def test_feedback_must_reference_local_executed_evidence(self):
        store = self.store()
        with self.assertRaises(ValueError):
            store.record_feedback("not-recorded", 1, "confirmed")
        with self.assertRaises(ValueError):
            store.record_feedback("not-recorded", 1, "model-guessed")

    def test_bounded_retrieval_and_bounded_unique_run_citations(self):
        store = self.store()
        for index in range(30):
            before = result(self.initial, position=[index, 0])
            store.record(before, "left", result(before, facing="West"), "run", index + 1)
        for index in range(12):
            store.record(self.initial, "forward", result(self.initial), f"run-{index}", 1)
        memory = store.retrieve(self.initial, limit=8, max_bytes=2_500)
        encoded = json.dumps(memory, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()
        self.assertLessEqual(len(encoded), 2_500)
        self.assertGreater(memory["omitted_count"], 0)
        self.assertGreaterEqual(len(memory["facts"]), 1)
        current = memory["facts"][0]
        self.assertTrue(current["matching_current_state"])
        self.assertLessEqual(len(current["citations"]), 4)
        self.assertEqual(len({cite["run_id"] for cite in current["citations"]}), len(current["citations"]))
        empty = store.retrieve(self.initial, limit=0, max_bytes=512)
        self.assertEqual(empty["facts"], [])
        self.assertEqual(empty["omitted_count"], 31)

    def test_snapshot_is_consistent_and_readonly_evaluation_does_not_learn(self):
        store = self.store()
        store.record(self.initial, "forward", result(self.initial), "train", 1)
        frozen_path = store.snapshot(Path(self.temp.name) / "frozen.sqlite3")
        frozen = self.store(path=frozen_path, read_only=True)
        initial_summary = frozen.summary()
        fact = frozen.record(self.initial, "right", result(self.initial, facing="East"), "test", 1)
        self.assertFalse(fact["persisted"])
        self.assertEqual(frozen.summary(), initial_summary)
        self.assertEqual(frozen.summary()["runs"], 1)
        with self.assertRaises(EvidenceConflict):
            frozen.record(self.initial, "right", result(self.initial, facing="East"), "train", 1)
        with self.assertRaises(ValueError):
            frozen.record_feedback("train", 1, "confirmed")
        with self.assertRaises(ScopeMismatch):
            self.store(path=frozen_path, project="another-project", read_only=True)
        store.record(self.initial, "right", result(self.initial, facing="East"), "later-train", 1)
        self.assertEqual(frozen.summary()["transitions"], 1)

    def test_path_project_identity_is_hashed_and_private_data_not_persisted(self):
        initial = observation()
        initial["entities"][0]["evaluation"] = "private label"
        initial["entities"][0]["state"]["api_key"] = "secret-value-12345"
        initial["events"] = ["secret-value-12345"]
        store = self.store(project="/Users/private/my-game", initial=initial)
        store.record(initial, "forward", result(initial), "run", 1)
        text = json.dumps(store.manifest) + json.dumps(store.retrieve(initial))
        self.assertNotIn("/Users", text)
        self.assertNotIn("private label", text)
        self.assertNotIn("secret-value", text)
        store.close()
        self.assertNotIn(b"secret-value-12345", self.path.read_bytes())

    def test_two_connections_see_committed_facts(self):
        first, second = self.store(), self.store()
        first.record(self.initial, "forward", result(self.initial), "first", 1)
        second.record(self.initial, "right", result(self.initial, facing="East"), "second", 1)
        self.assertEqual(first.summary()["transitions"], 2)
        self.assertEqual(second.summary()["runs"], 2)

    def test_optional_complete_catalog_supports_controls_unlocked_after_progress(self):
        initial = observation(controls=["open"])
        initial["_capabilities"]["control_catalog"] = ["open", "confirm", "close"]
        store = self.store(initial=initial)
        opened = result(initial)
        opened["controls"] = ["confirm", "close"]
        opened["entities"][0]["state"]["open"] = True
        store.record(initial, "open", opened, "run", 1)
        confirmed = result(opened, event="Confirmed")
        store.record(opened, "confirm", confirmed, "run", 2)
        self.assertEqual(store.summary()["control_counts"], {"confirm": 1, "open": 1})
        self.assertTrue(any(fact["action"] == "confirm" for fact in store.retrieve(opened)["facts"]))

    def test_complete_catalog_never_exposes_or_legalizes_unavailable_controls(self):
        from autoplay_qa.lean import lean_schema

        initial = observation(controls=["open"])
        initial["_capabilities"]["control_catalog"] = ["open", "confirm", "close"]
        store = self.store(initial=initial)
        view = public_observation(initial)
        self.assertEqual(view["controls"], ["open", "reset", "finish"])
        self.assertEqual(lean_schema(view["controls"])["properties"]["action"]["enum"], view["controls"])
        self.assertNotIn("control_catalog", json.dumps(view))
        with self.assertRaises(ValueError):
            experiment_fact(initial, "confirm", result(initial), run_id="run", step=1)
        with self.assertRaises(ValueError):
            store.record(initial, "confirm", result(initial), "run", 1)
        self.assertEqual(store.summary()["transitions"], 0)

    def test_complete_catalog_restart_and_order_are_stable(self):
        initial = observation(controls=["open"])
        initial["_capabilities"]["control_catalog"] = ["open", "confirm"]
        store = self.store(initial=initial)
        opened = result(initial)
        opened["controls"] = ["confirm"]
        store.record(initial, "open", opened, "run", 1)
        scope = store.scope_id
        store.close()
        reordered = deepcopy(initial)
        reordered["_capabilities"]["control_catalog"].reverse()
        reopened = self.store(initial=reordered)
        self.assertEqual(reopened.scope_id, scope)
        self.assertEqual(reopened.summary()["transitions"], 1)
        self.assertEqual(self.store(initial=opened).scope_id, scope)

    def test_complete_catalog_missing_changed_or_outside_controls_are_rejected(self):
        initial = observation(controls=["open"])
        initial["_capabilities"]["control_catalog"] = ["open", "confirm"]
        store = self.store(initial=initial)
        for kind in ("missing", "changed", "outside"):
            foreign = result(initial)
            if kind == "missing":
                del foreign["_capabilities"]["control_catalog"]
            elif kind == "changed":
                foreign["_capabilities"]["control_catalog"] = ["open", "cancel"]
            else:
                foreign["controls"] = ["outside"]
            with self.assertRaises(ScopeMismatch):
                store.record(initial, "open", foreign, "run", 1)
            with self.assertRaises(ScopeMismatch):
                store.retrieve(foreign)
        bad_initial = deepcopy(initial)
        bad_initial["controls"] = ["outside"]
        with self.assertRaises(ScopeMismatch):
            self.store(initial=bad_initial)

    def test_complete_catalog_invalid_declaration_fails_explicitly(self):
        for declaration in (None, "open", ["open", "open"], ["open", ""], ["open", 1], ["open", "reset"], ["open", "finish"]):
            initial = observation(controls=["open"])
            initial["_capabilities"]["control_catalog"] = declaration
            with self.assertRaises(ValueError):
                self.store(initial=initial)

    def test_undeclared_fixed_palette_scope_and_state_hashes_are_unchanged(self):
        store = self.store()
        self.assertEqual(store.scope_id, "7a3c6887e030ed0fbd1ddd9cadbd0891ed4f8ba07fb782c7c260447b63733eba")
        self.assertEqual(state_key(self.initial), "cf0556be0525355f8f8b1f84034fb938349ef52d87bd1656c694277af52e250e")
        self.assertNotIn("control_catalog", store.manifest)
        foreign = result(self.initial)
        foreign["controls"] = ["new-button"]
        with self.assertRaises(ScopeMismatch):
            store.record(self.initial, "forward", foreign, "run", 1)
        with self.assertRaises(ScopeMismatch):
            store.retrieve(foreign)

    def test_changed_complete_catalog_starts_an_isolated_scope(self):
        initial = observation(controls=["open"])
        initial["_capabilities"]["control_catalog"] = ["open", "confirm"]
        original = self.store(initial=initial)
        original.record(initial, "open", result(initial), "run", 1)
        changed = deepcopy(initial)
        changed["_capabilities"]["control_catalog"].append("cancel")
        other = self.store(initial=changed)
        self.assertNotEqual(other.scope_id, original.scope_id)
        self.assertEqual(other.summary()["transitions"], 0)

    def test_supplied_runtime_secrets_are_redacted_before_persistence(self):
        secret = "opaque-runtime-credential"
        initial = observation()
        initial["events"] = ["Runtime replied " + secret]
        store = self.store(initial=initial, secrets=(secret,))
        after = result(initial, event=secret)
        store.record(initial, "forward", after, "run", 1)
        self.assertNotIn(secret, json.dumps(store.retrieve(initial)))
        store.record_feedback("run", 1, "pending", comment=secret)
        self.assertNotIn(secret, json.dumps(store.reviewer_feedback()))
        store.close()
        self.assertNotIn(secret.encode(), self.path.read_bytes())

    @unittest.skipIf(os.name == "nt", "POSIX permission bits")
    def test_database_wal_sharedmemory_new_directories_and_snapshots_are_private(self):
        nested = Path(self.temp.name) / "new-parent" / "new-child" / "memory.sqlite3"
        store = self.store(path=nested)
        store.record(self.initial, "forward", result(self.initial), "run", 1)
        for directory in (nested.parent, nested.parent.parent):
            self.assertEqual(directory.stat().st_mode & 0o777, 0o700)
        for path in (nested, Path(str(nested) + "-wal"), Path(str(nested) + "-shm")):
            self.assertTrue(path.exists())
            self.assertEqual(path.stat().st_mode & 0o777, 0o600)
        snapshot = store.snapshot(Path(self.temp.name) / "frozen" / "snapshot.sqlite3")
        self.assertEqual(snapshot.stat().st_mode & 0o777, 0o600)
        self.assertEqual(snapshot.parent.stat().st_mode & 0o777, 0o700)


class ExperimentIndexTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="experiment-index-tests-")
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "experience.sqlite3"
        self.initial = observation()
        self.store = ExperienceStore(self.path, "project", "v1", self.initial)
        self.addCleanup(self.store.close)

    def index(self, observation=None, **options):
        return self.store.experiment_index(observation or self.initial,
                                           exclude_run_id="current", **options)

    def test_prior_counts_exact_state_outcomes_and_citations(self):
        self.store.record(self.initial, "forward", result(self.initial), "prior-one", 1)
        self.store.record(self.initial, "forward", result(self.initial), "prior-two", 1)
        self.store.record(self.initial, "forward", result(self.initial, position=[1, 0], event="Moved"), "prior-two", 2)
        elsewhere = result(self.initial, facing="East")
        self.store.record(elsewhere, "forward", result(elsewhere), "prior-two", 3)
        self.store.record(elsewhere, "place", result(elsewhere), "prior-two", 4)
        packet = self.index()
        self.assertTrue(packet["counts_complete"])
        self.assertEqual(packet["prior_run_count"], 2)
        self.assertEqual(packet["prior_transition_count"], 5)
        rows = {row["action"]: row for row in packet["controls"]}
        self.assertEqual(rows["forward"]["attempts"], 4)
        self.assertEqual(rows["forward"]["current_state"], {
            "attempts": 3, "unchanged_outcomes": 2, "distinct_outcomes": 2,
        })
        self.assertEqual(rows["place"]["current_state"]["attempts"], 0)
        self.assertEqual(rows["place"]["citations"]["current_state"], [])
        self.assertEqual({citation["run_id"] for citation in rows["forward"]["citations"]["any_state"]}, {"prior-one", "prior-two"})
        self.assertTrue(packet["zero_lists_complete"])
        self.assertEqual(packet["global_untested_controls"], ["left", "right"])
        self.assertEqual(packet["current_state_unattempted_controls"], ["left", "right", "place"])

    def test_current_run_is_excluded_everywhere_without_double_counting(self):
        self.store.record(self.initial, "forward", result(self.initial), "prior", 1)
        self.store.record(self.initial, "forward", result(self.initial), "current", 1)
        self.store.record(self.initial, "place", result(self.initial), "current", 2)
        packet = self.index()
        self.assertEqual(packet["excluded_run_id"], "current")
        self.assertEqual(packet["prior_run_count"], 1)
        self.assertEqual(packet["prior_transition_count"], 1)
        self.assertEqual(len(packet["controls"]), 1)
        self.assertEqual(packet["controls"][0]["attempts"], 1)
        for row in packet["controls"]:
            for citations in row["citations"].values():
                self.assertTrue(all(citation["run_id"] != "current" for citation in citations))
        self.assertIn("place", packet["global_untested_controls"])

    def test_empty_prior_runs_have_exact_zero_counts_with_current_availability_scope(self):
        self.store.record(self.initial, "forward", result(self.initial), "current", 1)
        packet = self.index()
        self.assertEqual(packet["control_scope"], "currently_legal_native_controls")
        self.assertTrue(packet["counts_complete"])
        self.assertEqual(packet["controls"], [])
        self.assertEqual(packet["global_untested_controls"], self.initial["controls"])
        self.assertEqual(packet["prior_run_count"], 0)

    def test_frozen_warm_store_is_read_only_and_index_does_not_learn(self):
        self.store.record(self.initial, "forward", result(self.initial), "train", 1)
        path = self.store.snapshot(Path(self.temp.name) / "frozen.sqlite3")
        frozen = ExperienceStore(path, "project", "v1", self.initial, read_only=True)
        self.addCleanup(frozen.close)
        before_bytes, before_summary = path.read_bytes(), frozen.summary()
        first = frozen.experiment_index(self.initial, exclude_run_id="current")
        frozen.record(self.initial, "place", result(self.initial), "current", 1)
        second = frozen.experiment_index(self.initial, exclude_run_id="current")
        self.assertEqual(first, second)
        self.assertEqual(before_summary, frozen.summary())
        self.assertEqual(before_bytes, path.read_bytes())

    def test_dynamic_catalog_only_currently_available_controls_are_reported(self):
        initial = observation(controls=["open"])
        initial["_capabilities"]["control_catalog"] = ["open", "confirm", "close"]
        memory = ExperienceStore(Path(self.temp.name) / "ui.sqlite3", "ui", "v1", initial)
        self.addCleanup(memory.close)
        opened = result(initial)
        opened["controls"] = ["confirm", "close"]
        memory.record(initial, "open", opened, "train", 1)
        memory.record(opened, "confirm", result(opened), "train", 2)
        initial_packet = memory.experiment_index(initial, exclude_run_id="current")
        self.assertEqual([row["action"] for row in initial_packet["controls"]], ["open"])
        self.assertEqual(initial_packet["current_native_control_count"], 1)
        self.assertNotIn("confirm", json.dumps(initial_packet))
        opened_packet = memory.experiment_index(opened, exclude_run_id="current")
        self.assertEqual([row["action"] for row in opened_packet["controls"]], ["confirm"])
        self.assertEqual(opened_packet["global_untested_controls"], ["close"])
        self.assertNotIn("open\"", json.dumps(opened_packet))

    def test_won_state_does_not_advertise_old_catalog_as_legal_actions(self):
        self.store.record(self.initial, "forward", result(self.initial), "train", 1)
        won = result(self.initial)
        won["status"] = "won"
        won["controls"] = []
        packet = self.index(won)
        self.assertEqual(packet["current_native_control_count"], 0)
        self.assertEqual(packet["controls"], [])
        self.assertEqual(packet["global_untested_controls"], [])
        self.assertEqual(packet["prior_transition_count"], 1)
        self.assertNotIn("forward", json.dumps(packet))

    def test_bounded_packet_omissions_are_not_reported_as_zero(self):
        names = ["操作" * 100 + str(index) for index in range(8)]
        initial = observation(controls=names)
        memory = ExperienceStore(Path(self.temp.name) / "long.sqlite3", "long", "v1", initial)
        self.addCleanup(memory.close)
        for index, action in enumerate(names):
            memory.record(initial, action, result(initial), "train", index + 1)
        packet = memory.experiment_index(initial, exclude_run_id="current", max_bytes=1_024)
        self.assertLessEqual(len(json.dumps(packet, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()), 1_024)
        self.assertFalse(packet["counts_complete"])
        self.assertEqual(packet["omitted_nonzero_controls"], len(names))
        self.assertFalse(packet["zero_lists_complete"])
        self.assertNotIn("global_untested_controls", packet)
        self.assertNotIn("current_state_unattempted_controls", packet)

    def test_long_zero_control_lists_are_omitted_atomically_without_false_empty_set(self):
        names = ["长" * 100 + str(index) for index in range(8)]
        initial = observation(controls=names)
        memory = ExperienceStore(Path(self.temp.name) / "long-zero.sqlite3", "long", "v1", initial)
        self.addCleanup(memory.close)
        packet = memory.experiment_index(initial, exclude_run_id="current", max_bytes=1_024)
        self.assertTrue(packet["counts_complete"])
        self.assertEqual(packet["omitted_nonzero_controls"], 0)
        self.assertFalse(packet["zero_lists_complete"])
        self.assertNotIn("global_untested_controls", packet)
        self.assertNotIn("current_state_unattempted_controls", packet)

    def test_partial_sparse_rows_have_exact_counts_and_citations_without_zero_claims(self):
        names = [f"button-{index}" for index in range(30)]
        initial = observation(controls=names)
        memory = ExperienceStore(Path(self.temp.name) / "partial.sqlite3", "partial", "v1", initial)
        self.addCleanup(memory.close)
        for index, action in enumerate(names):
            memory.record(initial, action, result(initial), "train", index + 1)
        packet = memory.experiment_index(initial, exclude_run_id="current", max_bytes=2_000)
        self.assertGreater(len(packet["controls"]), 0)
        self.assertLess(len(packet["controls"]), len(names))
        self.assertEqual(len(packet["controls"]) + packet["omitted_nonzero_controls"], len(names))
        self.assertFalse(packet["counts_complete"])
        self.assertNotIn("global_untested_controls", packet)
        for row in packet["controls"]:
            self.assertEqual(row["attempts"], 1)
            self.assertEqual(row["current_state"], {"attempts": 1, "unchanged_outcomes": 1, "distinct_outcomes": 1})
            self.assertTrue(row["citations"]["any_state"])
            self.assertTrue(row["citations"]["current_state"])

    def test_large_native_palette_queries_later_sql_chunks(self):
        names = [f"button-{index}" for index in range(450)]
        initial = observation(controls=names)
        memory = ExperienceStore(Path(self.temp.name) / "large-palette.sqlite3", "large", "v1", initial)
        self.addCleanup(memory.close)
        memory.record(initial, names[-1], result(initial), "train", 1)
        packet = memory.experiment_index(initial, exclude_run_id="current")
        self.assertEqual(packet["current_native_control_count"], 450)
        self.assertEqual(packet["controls"][0]["action"], names[-1])
        self.assertEqual(packet["controls"][0]["attempts"], 1)
        self.assertTrue(packet["counts_complete"])
        self.assertFalse(packet["zero_lists_complete"])

    def test_read_query_failure_does_not_return_fake_zeros_or_leave_transaction_open(self):
        self.store.record(self.initial, "forward", result(self.initial), "train", 1)
        database = self.store._db

        class FailingRead:
            def execute(self, sql, parameters=()):
                if "SUM(before_id=?)" in sql:
                    raise sqlite3.OperationalError("Offline injected read failure")
                return database.execute(sql, parameters)

        self.store._db = FailingRead()
        try:
            with self.assertRaises(sqlite3.OperationalError):
                self.index()
        finally:
            self.store._db = database
        self.assertFalse(database.in_transaction)
        self.assertEqual(self.index()["controls"][0]["attempts"], 1)

    def test_every_reported_nonzero_count_has_bounded_prior_citations(self):
        for index in range(8):
            self.store.record(self.initial, "forward", result(self.initial), f"train-{index}", 1)
        packet = self.index()
        row = packet["controls"][0]
        for citations in row["citations"].values():
            self.assertGreaterEqual(len(citations), 1)
            self.assertLessEqual(len(citations), 2)
            self.assertEqual(len(citations), len({citation["run_id"] for citation in citations}))

    def test_scope_and_game_version_isolation_and_errors(self):
        self.store.record(self.initial, "forward", result(self.initial), "train", 1)
        other = ExperienceStore(self.path, "project", "v2", self.initial)
        self.addCleanup(other.close)
        self.assertEqual(other.experiment_index(self.initial, exclude_run_id="current")["prior_run_count"], 0)
        with self.assertRaises(ScopeMismatch):
            self.index(observation(game="different"))
        for budget in (True, 0, 1_023):
            with self.assertRaises(ValueError):
                self.index(max_bytes=budget)
        for identifier in ("", "/Users/private/run", "contains space"):
            with self.assertRaises(ValueError):
                self.store.experiment_index(self.initial, exclude_run_id=identifier)
        self.store.close()
        with self.assertRaises(ValueError):
            self.index()
        with self.assertRaises(sqlite3.OperationalError):
            ExperienceStore(Path(self.temp.name) / "missing.sqlite3", "project", "v1", self.initial, read_only=True)

    def test_index_never_reads_reviewer_feedback_or_changes_old_retrieval(self):
        self.store.record(self.initial, "forward", result(self.initial), "train", 1)
        old_retrieval, old_summary, packet = self.store.retrieve(self.initial), self.store.summary(), self.index()
        self.store.record_feedback("train", 1, "confirmed", comment="PRIVATE_LABEL_SENTINEL")
        self.assertEqual(self.index(), packet)
        self.assertEqual(self.store.retrieve(self.initial), old_retrieval)
        self.assertEqual(self.store.summary(), old_summary)
        self.assertNotIn("PRIVATE_LABEL_SENTINEL", json.dumps(packet))

    def test_known_credentials_in_excluded_or_legacy_citation_ids_do_not_leak(self):
        secret = "opaqueSensitiveCredential123"
        self.store.record(self.initial, "forward", result(self.initial), secret, 1)
        reader = ExperienceStore(self.path, "project", "v1", self.initial, secrets=(secret,))
        self.addCleanup(reader.close)
        packet = reader.experiment_index(self.initial, exclude_run_id="current")
        self.assertNotIn(secret, json.dumps(packet))
        self.assertFalse(packet["counts_complete"])
        self.assertEqual(packet["omitted_nonzero_controls"], 1)
        self.assertNotIn("global_untested_controls", packet)
        with self.assertRaises(ValueError):
            reader.experiment_index(self.initial, exclude_run_id=secret)


if __name__ == "__main__":
    unittest.main()
