"""Offline tests for Phase-2 issue reporting and reproduction."""

import json
from pathlib import Path
import tempfile
import unittest

from autoplay_qa.game import KeyGateGame, defect_evaluation_metadata
from autoplay_qa.reporting import (
    evaluate_recording,
    evaluate_recordings,
    extract_issues,
    extract_observations,
    generate_report,
    load_recording,
    reproduce_issue,
    write_report,
)


class TinyGateGame:
    """Small deterministic adapter used to isolate report orchestration."""

    version = "tiny-gate-test"

    def __init__(self, seed=0, defect=None):
        self.seed = seed
        self.defect = defect
        self.tick = 0
        self.position = [1, 1]
        self.inventory = ["key"]
        self.gate_open = False
        self.events = ["Started."]

    def observe(self):
        row = ["#", "@", "D", "E", "#"]
        row[2] = "/" if self.gate_open else "D"
        return {
            "game": self.version,
            "tick": self.tick,
            "status": "playing",
            "map": ["#####", "".join(row), "#####"],
            "player": {
                "position": list(self.position),
                "inventory": list(self.inventory),
            },
            "objects": {
                "key": None,
                "gate": {"position": [2, 1], "open": self.gate_open},
                "exit": [3, 1],
            },
            "events": list(self.events),
            "actions": ["interact", "wait"],
        }

    def act(self, action):
        self.tick += 1
        if action == "wait":
            self.events = ["Waited."]
        elif action == "interact":
            self.gate_open = True
            if self.defect != "gate-keeps-key":
                self.inventory = []
            self.events = ["Unlocked the gate and used the key."]
        else:
            raise ValueError(action)
        return self.observe()

    def reset(self):
        retained_inventory = list(self.inventory)
        self.tick = 0
        self.position = [1, 1]
        self.inventory = (
            retained_inventory if self.defect == "reset-leak" else ["key"]
        )
        self.gate_open = False
        self.events = ["Started."]
        return self.observe()


def game_factory(seed, defect):
    return TinyGateGame(seed, defect)


class ToggleOutcomeGame:
    """Same visible start, different post-reset wait feedback."""

    version = "toggle-outcome-test"

    def __init__(self, seed=0, defect=None):
        self.seed = seed
        self.episode = 0
        self.tick = 0
        self.events = ["Started."]

    def observe(self):
        return {
            "game": self.version,
            "tick": self.tick,
            "status": "playing",
            "map": ["###", "#@#", "###"],
            "legend": {"#": "wall", "@": "player"},
            "player": {"position": [1, 1], "inventory": []},
            "objects": {
                "key": None,
                "gate": {"position": None, "open": False},
                "exit": None,
            },
            "events": list(self.events),
            "actions": ["wait"],
        }

    def act(self, action):
        if action != "wait":
            raise ValueError(action)
        self.tick += 1
        self.events = ["First outcome." if self.episode == 0 else "Second outcome."]
        return self.observe()

    def reset(self):
        self.episode += 1
        self.tick = 0
        self.events = ["Started."]
        return self.observe()


def toggle_factory(seed, defect):
    return ToggleOutcomeGame(seed, defect)


def transition(game, step, action, expected_position=None, anomalies=None):
    before = game.observe()
    after = game.act(action)
    fact = {
        "step": step,
        "action": action,
        "before": {
            "position": before["player"]["position"],
            "inventory": before["player"]["inventory"],
            "key": before["objects"]["key"],
            "gate_open": before["objects"]["gate"]["open"],
            "status": before["status"],
        },
        "after": {
            "position": after["player"]["position"],
            "inventory": after["player"]["inventory"],
            "key": after["objects"]["key"],
            "gate_open": after["objects"]["gate"]["open"],
            "status": after["status"],
        },
        "feedback": after["events"],
        "actual_changed_facets": [],
        "expected_position": expected_position,
        "expected_position_matched": (
            after["player"]["position"] == expected_position
            if expected_position is not None else None
        ),
    }
    if anomalies is not None:
        fact["anomalies"] = anomalies
    return {
        "type": "step",
        "step": step,
        "episode": 0,
        "before": before,
        "decision": {
            "action": action,
            "reason": "fixture",
            "expected_position": expected_position,
            "notes": [],
        },
        "after": after,
        "software_fact": fact,
    }


def raw_recording(game, actions, *, injected_defect=None):
    rows = [{
        "type": "session",
        "schema_version": 2,
        "game_version": game.version,
        "seed": game.seed,
        "initial_observation": game.observe(),
        **defect_evaluation_metadata(injected_defect),
    }]
    episode = 0
    for step, action in enumerate(actions, 1):
        before = game.observe()
        if action == "reset":
            after = game.reset()
            after = after if isinstance(after, dict) else game.observe()
            kind = "reset"
            episode_before, episode_after = episode, episode + 1
            episode += 1
        else:
            after = game.act(action)
            kind = "step"
            episode_before = episode_after = episode
        rows.append({
            "type": kind,
            "step": step,
            "episode": episode,
            "episode_before": episode_before,
            "episode_after": episode_after,
            "before": before,
            "decision": {
                "action": action,
                "reason": "fixture",
                "expected_position": None,
                "notes": [],
            },
            "after": after,
        })
    return rows


def gate_recording(*, injected=True, suspected=True):
    defect = "gate-keeps-key" if injected else None
    game = TinyGateGame(7, defect)
    initial = game.observe()
    first = transition(game, 1, "wait")
    second = transition(game, 2, "interact")
    note = {
        "id": "bug-gate-key",
        "kind": "suspected_bug" if suspected else "mechanic",
        "claim": "The gate opened but inventory still contains a key.",
        "status": "supported",
        "evidence_steps": [2],
    }
    return [
        {
            "type": "session",
            "schema_version": 2,
            "game_version": TinyGateGame.version,
            "seed": 7,
            "initial_observation": initial,
            "evaluation": {
                "injected_defect": ({
                    "name": "gate-keeps-key",
                    "description": "SECRET: injected key retention behavior",
                    "details": {"internal": True},
                } if injected else None),
            },
        },
        first,
        second,
        {"type": "memory", "hypotheses": [note]},
        {"type": "summary", "steps": 2, "hypotheses": [note]},
    ]


class ExtractionAndMarkdownTests(unittest.TestCase):
    def test_load_and_extract_latest_suspected_bug_note(self):
        records = gate_recording()
        records.insert(-1, {
            "type": "memory",
            "hypotheses": [{
                "id": "mechanic-only",
                "kind": "mechanic",
                "claim": "Waiting advances time.",
                "status": "supported",
                "evidence_steps": [1],
            }],
        })
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fixture.jsonl"
            path.write_text(
                "\n".join(json.dumps(row) for row in records) + "\n",
                encoding="utf-8",
            )
            loaded = load_recording(path)
            issues = extract_issues(path)

        self.assertEqual(loaded[0]["seed"], 7)
        self.assertEqual(len(issues), 2)
        self.assertEqual(issues[0]["source"], "suspected_bug")
        self.assertEqual(issues[0]["type"], "suspected_bug")
        self.assertEqual(issues[0]["evidence_steps"], [2])
        self.assertEqual(issues[0]["observed_signature"], "gate-keeps-key")
        self.assertEqual(issues[0]["expected"]["model_flag"], issues[0]["claim"])
        self.assertEqual(
            issues[0]["actual"]["feedback"],
            ["Unlocked the gate and used the key."],
        )

    def test_evicted_note_is_not_reported_from_an_older_memory_snapshot(self):
        records = gate_recording()
        replacement = {
            "id": "replacement",
            "kind": "mechanic",
            "claim": "The current retained note is not a bug flag.",
            "status": "uncertain",
            "evidence_steps": [1],
        }
        records[-2]["hypotheses"] = [replacement]
        records[-1]["hypotheses"] = [replacement]

        self.assertEqual(
            [issue["source"] for issue in extract_issues(records)],
            ["software_anomaly"],
        )

    def test_suspected_bug_without_executed_evidence_is_an_observation(self):
        game = TinyGateGame()
        records = raw_recording(game, ["wait"])
        note = {
            "id": "unsupported-bug",
            "kind": "suspected_bug",
            "claim": "Something may be wrong.",
            "status": "uncertain",
            "evidence_steps": [999],
        }
        records.append({"type": "summary", "hypotheses": [note]})

        self.assertEqual(extract_issues(records), [])
        observations = extract_observations(records)
        self.assertEqual(len(observations), 1)
        self.assertEqual(observations[0]["source"], "suspected_bug")
        self.assertEqual(observations[0]["evidence_steps"], [])

    def test_prediction_surprise_copies_are_one_non_counting_observation(self):
        game = TinyGateGame()
        initial = game.observe()
        anomaly = {
            "type": "prediction_surprise",
            "step": 1,
            "evidence_steps": [1],
            "expected_position": [9, 9],
            "actual_position": [1, 1],
        }
        step = transition(
            game, 1, "wait", expected_position=[9, 9], anomalies=[anomaly]
        )
        records = [
            {
                "type": "session",
                "game_version": TinyGateGame.version,
                "seed": 0,
                "initial_observation": initial,
                "evaluation": {"injected_defect": None},
            },
            step,
            {"type": "summary", "anomalies": [anomaly]},
        ]

        issues = extract_issues(records)
        observations = extract_observations(records)

        self.assertEqual(issues, [])
        self.assertEqual(len(observations), 1)
        self.assertEqual(observations[0]["source"], "software_anomaly")
        self.assertEqual(observations[0]["type"], "prediction_surprise")
        self.assertEqual(observations[0]["expected"], [9, 9])
        report = generate_report(records, game_factory=game_factory)
        self.assertNotIn("## Issue 1:", report)
        self.assertIn("## Appendix: Observations (not issues)", report)

    def test_unsubstantiated_map_anomaly_is_only_an_observation(self):
        game = TinyGateGame()
        initial = game.observe()
        anomaly = {
            "type": "map_object_inconsistency",
            "step": 1,
            "evidence_steps": [1],
            "action": "wait",
            "aspect": "player_position_glyph",
            "expected": {"detail": "player position is marked with @"},
            "actual": {"detail": "player glyph was missing"},
        }
        step = transition(game, 1, "wait", anomalies=[anomaly])
        records = [{
            "type": "session",
            "game_version": TinyGateGame.version,
            "seed": 0,
            "initial_observation": initial,
            "evaluation": {"injected_defect": None},
        }, step]

        self.assertEqual(extract_issues(records), [])
        observation = extract_observations(records)[0]

        self.assertEqual(observation["anomaly"]["aspect"], "player_position_glyph")
        self.assertEqual(observation["expected"], anomaly["expected"])
        self.assertEqual(observation["actual"], anomaly["actual"])

    def test_non_qa_recording_does_not_derive_prediction_anomalies(self):
        game = TinyGateGame()
        initial = game.observe()
        records = [{
            "type": "session",
            "game_version": TinyGateGame.version,
            "seed": 0,
            "initial_observation": initial,
            "evaluation": {"injected_defect": None},
        }, transition(game, 1, "wait", expected_position=[9, 9])]

        self.assertEqual(extract_issues(records), [])

    def test_report_hides_ground_truth_unless_requested_and_writes_file(self):
        records = gate_recording()

        report = generate_report(records, game_factory=game_factory)

        self.assertIn("Environment: game `tiny-gate-test`; seed `7`.", report)
        self.assertIn("## Issue 1:", report)
        self.assertIn("### Minimal reproduction", report)
        self.assertIn("1. `interact`", report)
        self.assertIn("Recording steps: 2.", report)
        self.assertNotIn("gate-keeps-key", report)
        self.assertNotIn("SECRET", report)

        revealed = generate_report(
            records, with_ground_truth=True, game_factory=game_factory
        )
        self.assertIn("gate-keeps-key", revealed)
        self.assertIn("SECRET", revealed)

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "report.md"
            written = write_report(
                records, output, game_factory=game_factory
            )
            self.assertEqual(written, output.resolve())
            self.assertEqual(output.read_text(encoding="utf-8"), report)


class ReproductionAndEvaluationTests(unittest.TestCase):
    def test_reproduce_verifies_and_drop_one_minimizes_irrelevant_prefix(self):
        records = gate_recording()
        for row in records:
            if isinstance(row.get("hypotheses"), list):
                row["hypotheses"][0]["evidence_steps"] = [1, 2]
        result = reproduce_issue(records, 1, game_factory=game_factory)

        self.assertTrue(result["reproduced"])
        self.assertEqual(result["exit_code"], 0)
        self.assertEqual(result["source"], "suspected_bug")
        self.assertEqual(result["type"], "suspected_bug")
        self.assertEqual(result["actions"], ["interact"])
        self.assertEqual(result["source_steps"], [2])
        self.assertEqual(result["original_action_count"], 2)
        self.assertEqual(result["minimized_action_count"], 1)

    def test_evaluate_hits_injected_defect_and_verifies_reproduction(self):
        result = evaluate_recording(
            gate_recording(), game_factory=game_factory
        )

        self.assertEqual(result["detection"], "hit")
        self.assertTrue(result["hit"])
        self.assertEqual(result["matched_issue_numbers"], [1, 2])
        self.assertEqual(result["false_positive_count"], 0)
        self.assertTrue(result["repro_verified"])
        self.assertTrue(result["issues"][0]["reproduced"])

    def test_raw_visible_contradictions_match_and_reproduce_without_model_notes(self):
        cases = {
            "gate-keeps-key": ["south", "south", "east", "east", "interact"],
            "wall-clip": ["east", "east", "east"],
            "phantom-block": ["east", "east", "south"],
            "exit-no-win": [
                "south", "south", "east", "east", "interact",
                "east", "east", "east", "east",
            ],
        }
        expected_aspects = {
            "gate-keeps-key": "key_use_feedback_inventory",
            "wall-clip": "entered_wall_labeled_tile",
            "phantom-block": "blocked_on_floor_labeled_tile",
            "exit-no-win": "exit_status_unchanged",
        }
        for defect, actions in cases.items():
            with self.subTest(defect=defect):
                records = raw_recording(
                    KeyGateGame(0, defect), actions, injected_defect=defect
                )
                issues = extract_issues(records)
                result = evaluate_recording(records)

                self.assertEqual(len(issues), 1)
                self.assertEqual(issues[0]["type"], "map_object_inconsistency")
                self.assertEqual(issues[0]["category"], "label_contradiction")
                self.assertIn(expected_aspects[defect], issues[0]["aspects"])
                self.assertEqual(result["detection"], "hit")
                self.assertEqual(result["false_positive_count"], 0)
                self.assertTrue(result["repro_verified"])

    def test_reset_sightings_deduplicate_and_any_occurrence_reproduces(self):
        records = raw_recording(
            TinyGateGame(3, "reset-leak"),
            ["interact", "reset", "reset"],
            injected_defect="reset-leak",
        )

        issues = extract_issues(records)
        result = evaluate_recording(records, game_factory=game_factory)

        self.assertEqual(len(issues), 1)
        self.assertEqual(issues[0]["type"], "reset_inconsistency")
        self.assertEqual(issues[0]["evidence_steps"], [2, 3])
        self.assertEqual(result["detection"], "hit")
        self.assertTrue(result["repro_verified"])

    def test_raw_cross_episode_issue_reproduces(self):
        records = raw_recording(ToggleOutcomeGame(), ["wait", "reset", "wait"])

        issues = extract_issues(records)
        self.assertEqual(len(issues), 1)
        self.assertEqual(issues[0]["type"], "cross_episode_nondeterminism")
        self.assertEqual(issues[0]["evidence_steps"], [1, 3])

        reproduced = reproduce_issue(records, 1, game_factory=toggle_factory)
        self.assertTrue(reproduced["reproduced"])
        self.assertEqual(reproduced["actions"], ["wait", "reset", "wait"])

    def test_bug_claim_text_alone_does_not_match_injected_defect(self):
        records = raw_recording(TinyGateGame(), ["wait"])
        records[0].update(defect_evaluation_metadata("gate-keeps-key"))
        note = {
            "id": "wrong-claim",
            "kind": "suspected_bug",
            "claim": "The gate kept the key.",
            "status": "supported",
            "evidence_steps": [1],
        }
        records.append({"type": "summary", "hypotheses": [note]})

        result = evaluate_recording(records, game_factory=game_factory)

        self.assertEqual(result["detection"], "miss")
        self.assertEqual(result["matched_issue_numbers"], [])
        self.assertEqual(result["false_positive_count"], 1)

    def test_unsubstantiated_cross_episode_anomaly_is_an_observation(self):
        game = TinyGateGame(7, "gate-keeps-key")
        initial = game.observe()
        first = transition(game, 1, "wait")
        second = transition(game, 2, "interact", anomalies=[{
            "type": "cross_episode_nondeterminism",
            "original_step": 1,
            "repeated_step": 2,
            "evidence_steps": [1, 2],
            "result": "different",
        }])
        records = [{
            "type": "session",
            "game_version": TinyGateGame.version,
            "seed": 7,
            "focus": "qa",
            "initial_observation": initial,
            "evaluation": {
                "injected_defect": {
                    "name": "gate-keeps-key",
                    "description": "fixture",
                },
            },
        }, first, second]

        issues = extract_issues(records)
        observations = extract_observations(records)

        # The cited actions are not the same and did not occur in separate
        # episodes, so the stored label alone cannot promote this to an issue.
        self.assertEqual(
            [issue["type"] for issue in issues],
            ["map_object_inconsistency"],
        )
        self.assertIn(
            "cross_episode_nondeterminism",
            [observation["type"] for observation in observations],
        )

    def test_clean_suspected_bug_is_a_false_positive(self):
        result = evaluate_recording(
            gate_recording(injected=False), game_factory=game_factory
        )

        self.assertEqual(result["detection"], "clean")
        self.assertIsNone(result["hit"])
        self.assertEqual(result["matched_issue_numbers"], [])
        self.assertEqual(result["false_positive_count"], 1)
        self.assertFalse(result["repro_verified"])

    def test_clean_prediction_surprise_is_not_a_false_positive(self):
        game = TinyGateGame()
        initial = game.observe()
        anomaly = {
            "type": "prediction_surprise",
            "step": 1,
            "evidence_steps": [1],
            "expected": {"position": [9, 9]},
            "actual": {"position": [1, 1], "facets": {}, "feedback": ["Waited."]},
        }
        records = [{
            "type": "session",
            "game_version": TinyGateGame.version,
            "seed": 0,
            "initial_observation": initial,
            "evaluation": {"injected_defect": None},
        }, transition(game, 1, "wait", expected_position=[9, 9], anomalies=[anomaly])]

        result = evaluate_recording(records, game_factory=game_factory)

        self.assertEqual(result["flagged_issue_count"], 0)
        self.assertEqual(result["issue_count"], 0)
        self.assertEqual(result["observation_count"], 1)
        self.assertEqual(result["false_positive_count"], 0)

    def test_multiple_recording_evaluation_has_aggregate_counts(self):
        result = evaluate_recordings(
            [gate_recording(), gate_recording(injected=False)],
            game_factory=game_factory,
        )

        self.assertEqual(result["totals"]["recordings"], 2)
        self.assertEqual(result["totals"]["defect_recordings"], 1)
        self.assertEqual(result["totals"]["hits"], 1)
        self.assertEqual(result["totals"]["misses"], 0)
        self.assertEqual(result["totals"]["false_positives"], 1)
        self.assertEqual(result["totals"]["repro_verified"], 1)

    def test_bad_issue_number_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "between 1 and 2"):
            reproduce_issue(gate_recording(), 3, game_factory=game_factory)


if __name__ == "__main__":
    unittest.main()
