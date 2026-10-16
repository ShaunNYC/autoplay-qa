"""Focused fixtures for the compact, software-grounded lean explorer."""

import copy
from dataclasses import asdict, is_dataclass
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from autoplay_qa.cli import begin_session, main
from autoplay_qa.config import Settings
from autoplay_qa.exploration import ExplorationLimits
from autoplay_qa.game import ACTIONS, DEFECTS, KeyGateGame
from autoplay_qa.lean import (
    DEFAULT_DIRECT_FALLBACK_MAX_TOKENS,
    LEAN_ANOMALY_LIMIT,
    LEAN_CONTEXT_HISTORY_SIZE,
    LEAN_CROSS_EPISODE_PAIR_LIMIT,
    LEAN_PROMPT,
    LEAN_PROMPT_VERSION,
    LeanDecision,
    lean_messages,
    lean_schema,
    run_lean_exploration,
)
from autoplay_qa.nebius import InvalidDecision, NebiusClient
from autoplay_qa.reporting import evaluate_recording, generate_report
from autoplay_qa.review import review_session
from autoplay_qa.session import SessionLog, replay


def note(identifier="h1", claim="North is blocked here", status="uncertain", evidence=()):
    return {
        "id": identifier,
        "claim": claim,
        "status": status,
        "evidence_steps": list(evidence),
    }


def decision(action="north", *, reason="Test one control", expected_position=None, notes=()):
    return {
        "action": action,
        "reason": reason,
        "expected_position": expected_position,
        "notes": list(notes),
    }


def response(value, finish="stop"):
    return {
        "id": "lean-unit-test-response",
        "model": "nvidia/unit-test-nemotron",
        "choices": [{"message": {"content": json.dumps(value)}, "finish_reason": finish}],
        "usage": {"prompt_tokens": 20, "completion_tokens": 10, "total_tokens": 30},
    }


def normalized(value):
    """Make parse results easy to inspect without prescribing tuple/list storage."""
    if is_dataclass(value):
        value = asdict(value)
    return json.loads(json.dumps(value))


class LeanSchemaAndParsingTests(unittest.TestCase):
    def test_schema_is_small_flat_and_contains_only_the_lean_fields(self):
        allowed = [*ACTIONS, "reset", "finish"]
        schema = lean_schema(allowed)
        encoded = json.dumps(schema, separators=(",", ":")).encode("utf-8")

        self.assertLess(len(encoded), 1_200)
        self.assertNotIn('"anyOf"', encoded.decode("utf-8"))
        self.assertEqual(
            set(schema["properties"]),
            {"action", "reason", "expected_position", "notes"},
        )
        self.assertEqual(schema["properties"]["action"]["enum"], allowed)
        self.assertEqual(schema["properties"]["reason"]["maxLength"], 200)
        self.assertEqual(schema["properties"]["notes"]["maxItems"], 6)
        note_schema = schema["properties"]["notes"]["items"]
        self.assertEqual(
            set(note_schema["properties"]),
            {"id", "claim", "status", "kind", "evidence_steps"},
        )
        self.assertEqual(note_schema["properties"]["claim"]["maxLength"], 200)
        self.assertEqual(note_schema["properties"]["evidence_steps"]["maxItems"], 6)
        self.assertEqual(
            note_schema["properties"]["status"]["enum"],
            ["uncertain", "supported", "contradicted"],
        )

    def test_only_a_missing_or_illegal_action_is_fatal(self):
        allowed = [*ACTIONS, "reset", "finish"]
        for value in ({}, {"action": "teleport"}, {"action": 1}, []):
            with self.subTest(value=value), self.assertRaises(InvalidDecision):
                LeanDecision.parse(json.dumps(value), allowed, set())

        sparse, sparse_warnings = LeanDecision.parse(
            json.dumps({"action": "north"}), allowed, set()
        )
        self.assertEqual(sparse.action, "north")
        self.assertGreaterEqual(len(sparse_warnings), 1)

        malformed, warnings = LeanDecision.parse(
            json.dumps({
                "action": "north",
                "reason": ["not text"],
                "expected_position": [1, True],
                "notes": "not a list",
                "legacy_bookkeeping": {"must": "be ignored"},
            }),
            allowed,
            set(),
        )
        value = normalized(malformed)
        self.assertEqual(value["action"], "north")
        self.assertIsNone(value["expected_position"])
        self.assertEqual(value["notes"], [])
        self.assertGreaterEqual(len(warnings), 3)

    def test_bad_notes_are_dropped_individually_and_bad_evidence_is_aggregated(self):
        parsed, warnings = LeanDecision.parse(
            json.dumps(decision(notes=[
                note(status="supported", evidence=[1, 3, 2, 4, -1, True, "3"]),
                {"claim": "missing id", "status": "uncertain", "evidence_steps": []},
                note("h2", claim=7),
                note("h3", status="verified"),
            ])),
            [*ACTIONS, "reset", "finish"],
            {1, 3},
        )
        value = normalized(parsed)

        self.assertEqual([item["id"] for item in value["notes"]], ["h1"])
        self.assertEqual(value["notes"][0]["evidence_steps"], [1, 3])
        # Nonexistent, future, negative, boolean, and noninteger evidence are
        # summarized once, while three malformed notes are discarded as units.
        self.assertEqual(len(warnings), 4)
        self.assertEqual(
            sum(item["code"] == "normalized_evidence_steps" for item in warnings),
            1,
        )

    def test_evidence_is_bounded_after_validation_with_one_aggregate_warning(self):
        parsed, warnings = LeanDecision.parse(
            json.dumps(decision(notes=[note(
                status="supported",
                evidence=[1, 1, 99, 2, "bad", 3, 4, 5, 6, 7, 8],
            )])),
            [*ACTIONS, "reset", "finish"],
            set(range(1, 9)),
        )
        value = normalized(parsed)

        self.assertEqual(value["notes"][0]["evidence_steps"], [1, 2, 3, 4, 5, 6])
        self.assertEqual(len(warnings), 1)
        self.assertEqual(warnings[0]["code"], "normalized_evidence_steps")

    def test_identical_parser_warnings_are_collapsed(self):
        repeated = {"code": "same_warning", "message": "Same warning."}
        with patch("autoplay_qa.lean._warning", return_value=repeated):
            _, warnings = LeanDecision.parse(
                json.dumps({
                    "action": "north",
                    "reason": [],
                    "expected_position": "invalid",
                    "notes": "invalid",
                    "unknown": True,
                }),
                [*ACTIONS, "reset", "finish"],
                set(),
            )

        self.assertEqual(warnings, [repeated])

    def test_parser_enforces_the_schema_text_limits(self):
        parsed, warnings = LeanDecision.parse(
            json.dumps(decision(
                reason="r" * 201,
                notes=[note(claim="c" * 201)],
            )),
            [*ACTIONS, "reset", "finish"],
            set(),
        )
        value = normalized(parsed)

        self.assertEqual(value["reason"], "r" * 200)
        self.assertEqual(value["notes"], [])
        self.assertEqual(
            {item["code"] for item in warnings},
            {"trimmed_reason", "invalid_note_claim"},
        )

    def test_messages_hide_rules_strategy_and_passability(self):
        game = KeyGateGame(0, "reset-leak")
        observation = game.observe()
        developer_rules = list(observation["rules"])
        observation["rules"] = [*developer_rules, "SECRET_DEVELOPER_RULE"]
        observation["goal"] = "SECRET_DEVELOPER_GOAL"
        messages = lean_messages(
            observation, 0, {}, [], {"actions": 24, "requests": 28, "seconds": 300},
            focus="qa",
        )
        serialized = json.dumps(messages)
        context = json.loads(messages[1]["content"])

        self.assertEqual(context["episode"], 0)
        self.assertEqual(context["factual_log"], [])
        self.assertEqual(context["cross_episode_same_start_pairs"], [])
        self.assertNotIn('"rules"', serialized)
        self.assertNotIn('"goal"', serialized)
        self.assertNotIn("passable", serialized.lower())
        self.assertNotIn("collect the key", serialized.lower())
        self.assertNotIn("SECRET_DEVELOPER", serialized)
        for rule in developer_rules:
            self.assertNotIn(rule, serialized)
        for name, description in DEFECTS.items():
            self.assertNotIn(name, serialized)
            self.assertNotIn(description, serialized)

        checklist = context["qa_checklist"].splitlines()
        self.assertLessEqual(len(checklist), 6)
        self.assertIn(
            "If something you observed contradicts what the visible map, labels, "
            "or an earlier identical attempt showed, record it as a suspected_bug note.",
            checklist,
        )
        joined = " ".join(checklist).lower()
        for phrase in (
            "walls, edges, and objects",
            "each control in different situations",
            "revisit objects after their state changes",
            "reset to check the start is restored and earlier results repeat",
            "anything inconsistent",
        ):
            self.assertIn(phrase, joined)
        self.assertIn(
            "Decide quickly; one short paragraph of thinking is enough.",
            " ".join(LEAN_PROMPT.split()),
        )


class LeanControllerTests(unittest.TestCase):
    def run_fixture(
        self,
        replies,
        *,
        limits=ExplorationLimits(),
        on_action=None,
        focus="mechanics",
        game=None,
        direct_fallback_max_tokens=DEFAULT_DIRECT_FALLBACK_MAX_TOKENS,
    ):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        path = Path(directory.name) / "lean.jsonl"
        game = game or KeyGateGame(0)
        client = NebiusClient(Settings("unit-test-key"), temperature=1.0, top_p=0.95)
        with SessionLog(path, ("unit-test-key",)) as log:
            begin_session(
                log,
                game,
                "unit-test-fixtures",
                schema_version=2,
                task="explore",
                focus=focus,
                prompt_version=LEAN_PROMPT_VERSION,
            )
            with patch.object(client, "_request", side_effect=replies) as transport:
                summary = run_lean_exploration(
                    game,
                    client,
                    log,
                    lambda: KeyGateGame(0),
                    limits,
                    on_action=on_action,
                    focus=focus,
                    direct_fallback_max_tokens=direct_fallback_max_tokens,
                )
        records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        return summary, records, path, transport

    def test_truncation_is_atomic_and_one_repair_can_recover(self):
        executed = []
        truncated = decision(
            "north",
            reason="This must never execute",
            expected_position=[1, 0],
            notes=[note("discarded")],
        )
        summary, records, path, transport = self.run_fixture([
            response(truncated, finish="length"),
            response(decision("east", expected_position=[2, 1])),
            response(decision(
                "finish",
                expected_position=None,
                notes=[note(status="supported", evidence=[1])],
            )),
        ], on_action=executed.append)

        self.assertEqual(summary["status"], "finished")
        self.assertEqual(summary["steps"], 1)
        self.assertEqual(summary["inference_requests"], 3)
        self.assertEqual(summary["rejected_decisions"], 1)
        self.assertEqual(summary["direct_fallbacks"], 1)
        self.assertEqual(transport.call_count, 3)
        self.assertEqual([item["decision"]["action"] for item in executed], ["east"])
        self.assertFalse(any(
            row["type"] in {"memory", "step", "reset", "finish"}
            and row.get("request_number") == 1
            for row in records
        ))
        first, repair = [row for row in records if row["type"] == "request"][:2]
        requests = [row for row in records if row["type"] == "request"]
        self.assertEqual(
            [row["decision_mode"] for row in requests],
            ["thinking", "direct_fallback", "thinking"],
        )
        bodies = [call.args[1] for call in transport.call_args_list]
        self.assertEqual(
            [body["chat_template_kwargs"]["enable_thinking"] for body in bodies],
            [True, False, True],
        )
        self.assertEqual(
            [body["max_tokens"] for body in bodies],
            [8192, DEFAULT_DIRECT_FALLBACK_MAX_TOKENS, 8192],
        )
        first_context = json.loads(first["messages"][1]["content"])
        repair_context = json.loads(repair["messages"][1]["content"])
        self.assertEqual(first_context["observation"], repair_context["observation"])
        self.assertEqual(repair_context["notes"], [])
        self.assertIn("validation_feedback", repair_context)
        completion = next(
            row for row in records
            if row["type"] == "completion" and row["request_number"] == 1
        )
        self.assertIsNone(completion["answer"])
        accepted_step = next(row for row in records if row["type"] == "step")
        self.assertEqual(accepted_step["decision_mode"], "direct_fallback")
        accepted_memory = next(
            row for row in records
            if row["type"] == "memory" and row["request_number"] == 2
        )
        self.assertEqual(accepted_memory["decision_mode"], "direct_fallback")
        self.assertTrue(replay(path)["lean_facts_validated"])

    def test_a_second_invalid_response_ends_the_single_repair_attempt(self):
        summary, records, _, transport = self.run_fixture([
            response(decision("north"), finish="length"),
            response(decision("teleport")),
            response(decision("east")),
        ])

        self.assertEqual(summary["status"], "invalid_decision")
        self.assertEqual(summary["steps"], 0)
        self.assertEqual(summary["inference_requests"], 2)
        self.assertEqual(summary["rejected_decisions"], 2)
        self.assertEqual(summary["direct_fallbacks"], 1)
        self.assertEqual(transport.call_count, 2)
        repair_body = transport.call_args_list[1].args[1]
        self.assertFalse(repair_body["chat_template_kwargs"]["enable_thinking"])
        self.assertEqual(repair_body["max_tokens"], DEFAULT_DIRECT_FALLBACK_MAX_TOKENS)
        self.assertFalse(any(row["type"] in {"memory", "step", "reset", "finish"} for row in records))

    def test_truncated_direct_repair_has_specific_status_and_safe_diagnostics(self):
        secret = "unit-test-key"

        def truncated_reply(label, head_character, tail_character):
            middle = f"MIDDLE_{label}_MUST_NOT_BE_RECORDED"
            content = (
                f"{label}:{secret}:"
                + head_character * 360
                + middle
                + tail_character * 360
                + f":{secret}:{label}"
            )
            value = response(decision("north"), finish="length")
            value["choices"][0]["message"]["content"] = (
                f"<think>PRIVATE_REASONING_{label}_{secret}</think>{content}"
            )
            return value, content, middle

        first_reply, first_content, first_middle = truncated_reply("FIRST", "H", "T")
        repair_reply, repair_content, repair_middle = truncated_reply("REPAIR", "A", "Z")
        summary, records, path, transport = self.run_fixture([
            first_reply,
            repair_reply,
            response(decision("east")),
        ])

        self.assertEqual(DEFAULT_DIRECT_FALLBACK_MAX_TOKENS, 1536)
        self.assertEqual(summary["status"], "model_output_limit")
        self.assertTrue(summary["partial"])
        self.assertEqual(summary["steps"], 0)
        self.assertEqual(summary["inference_requests"], 2)
        self.assertEqual(summary["completed_inference_responses"], 2)
        self.assertEqual(summary["rejected_decisions"], 2)
        self.assertEqual(summary["direct_fallbacks"], 1)
        self.assertEqual(transport.call_count, 2)
        bodies = [call.args[1] for call in transport.call_args_list]
        self.assertEqual(
            [(body["chat_template_kwargs"]["enable_thinking"], body["max_tokens"]) for body in bodies],
            [(True, 8192), (False, 1536)],
        )

        completion_rows = [row for row in records if row["type"] == "completion"]
        rejected_rows = [row for row in records if row["type"] == "rejected_decision"]
        self.assertEqual(len(completion_rows), 2)
        self.assertEqual(len(rejected_rows), 2)
        for request_number, content, middle in (
            (1, first_content, first_middle),
            (2, repair_content, repair_middle),
        ):
            sanitized = content.replace(secret, "[REDACTED]")
            expected = {
                "head": sanitized[:300],
                "tail": sanitized[-300:],
            }
            for row in (
                next(item for item in completion_rows if item["request_number"] == request_number),
                next(item for item in rejected_rows if item["request_number"] == request_number),
            ):
                self.assertEqual(row["content_diagnostic"], expected)
                serialized = json.dumps(row)
                self.assertNotIn(secret, serialized)
                self.assertNotIn("PRIVATE_REASONING", serialized)
                self.assertNotIn(middle, serialized)
                self.assertLessEqual(len(row["content_diagnostic"]["head"]), 300)
                self.assertLessEqual(len(row["content_diagnostic"]["tail"]), 300)

        self.assertEqual(records[-1]["type"], "summary")
        replayed = replay(path)
        self.assertTrue(replayed["recording_complete"])
        self.assertEqual(replayed["exploration_status"], "model_output_limit")
        reviewed = review_session(path)
        self.assertTrue(reviewed["recording_complete"])
        self.assertEqual(reviewed["status"], "model_output_limit")
        self.assertIn("AutoPlay QA bug report", generate_report(path))
        evaluated = evaluate_recording(path)
        self.assertEqual(evaluated["detection"], "clean")
        self.assertEqual(evaluated["false_positive_count"], 0)

    def test_unfinished_inline_reasoning_is_not_recorded_as_a_diagnostic(self):
        truncated = response(decision("north"), finish="length")
        truncated["choices"][0]["message"]["content"] = (
            "<think>PRIVATE_UNFINISHED_REASONING unit-test-key"
        )
        summary, records, _, _ = self.run_fixture([
            truncated,
            response(decision("finish")),
        ])

        self.assertEqual(summary["status"], "finished")
        first_completion = next(
            row for row in records
            if row["type"] == "completion" and row["request_number"] == 1
        )
        first_rejection = next(
            row for row in records
            if row["type"] == "rejected_decision" and row["request_number"] == 1
        )
        self.assertNotIn("content_diagnostic", first_completion)
        self.assertNotIn("content_diagnostic", first_rejection)
        self.assertNotIn("PRIVATE_UNFINISHED_REASONING", json.dumps(records))

    def test_time_pressure_uses_direct_mode_from_recent_thinking_latency(self):
        clock = [0.0]
        values = iter([
            response(decision("wait")),
            response(decision("wait")),
        ])

        def timed_transport(*_args):
            value = next(values)
            if clock[0] == 0:
                clock[0] = 40.0
            return value

        with patch("time.monotonic", side_effect=lambda: clock[0]):
            summary, records, path, transport = self.run_fixture(
                timed_transport,
                limits=ExplorationLimits(max_actions=2, max_requests=3, max_seconds=100),
            )

        self.assertEqual(summary["status"], "action_limit")
        self.assertEqual(summary["direct_fallbacks"], 1)
        request_rows = [row for row in records if row["type"] == "request"]
        self.assertEqual(
            [row["decision_mode"] for row in request_rows],
            ["thinking", "direct_fallback"],
        )
        self.assertEqual(request_rows[1]["decision_mode_reason"], "time_pressure")
        second_context = json.loads(request_rows[1]["messages"][1]["content"])
        self.assertEqual(second_context["remaining_budget"]["recent_request_median_seconds"], 40)
        bodies = [call.args[1] for call in transport.call_args_list]
        self.assertEqual(
            [(body["chat_template_kwargs"]["enable_thinking"], body["max_tokens"]) for body in bodies],
            [(True, 8192), (False, DEFAULT_DIRECT_FALLBACK_MAX_TOKENS)],
        )
        steps = [row for row in records if row["type"] == "step"]
        self.assertEqual(
            [row["decision_mode"] for row in steps],
            ["thinking", "direct_fallback"],
        )
        self.assertTrue(replay(path)["lean_facts_validated"])

    def test_twenty_four_step_prompt_context_is_compact_and_bounded(self):
        replies = []
        for index in range(12):
            # The first two surprises are duplicates; later predictions make
            # enough distinct surprises to exercise both deduplication and cap.
            expected = [1, 0] if index < 2 else [1, -index]
            replies.extend([
                response(decision("north", expected_position=expected)),
                response(decision("reset", expected_position=[1, 1])),
            ])
        summary, records, path, transport = self.run_fixture(
            replies,
            limits=ExplorationLimits(max_actions=24, max_requests=28, max_seconds=300),
            focus="qa",
        )

        self.assertEqual(summary["status"], "action_limit")
        self.assertEqual(summary["steps"], 24)
        self.assertEqual(summary["resets"], 12)
        self.assertEqual(summary["inference_requests"], 24)
        self.assertEqual(transport.call_count, 24)
        request_rows = [row for row in records if row["type"] == "request"]
        twentieth = request_rows[19]
        context = json.loads(twentieth["messages"][1]["content"])

        self.assertEqual(context["context_compaction"], {
            "version": 1,
            "factual_log_limit": LEAN_CONTEXT_HISTORY_SIZE,
            "cross_episode_pair_limit": LEAN_CROSS_EPISODE_PAIR_LIMIT,
            "anomaly_limit": LEAN_ANOMALY_LIMIT,
        })
        self.assertEqual(
            [fact["step"] for fact in context["factual_log"]],
            list(range(12, 20)),
        )
        self.assertTrue(all("anomalies" not in fact for fact in context["factual_log"]))
        self.assertEqual(context["earlier_factual_summary"], {
            "step_count": 11,
            "first_step": 1,
            "last_step": 11,
            "counts_by_action_outcome": [
                ["north", "unchanged", 6],
                ["reset", "unchanged", 5],
            ],
        })
        pairs = context["cross_episode_same_start_pairs"]
        self.assertEqual(len(pairs), LEAN_CROSS_EPISODE_PAIR_LIMIT)
        self.assertTrue(all(isinstance(pair, list) and len(pair) == 6 for pair in pairs))
        self.assertEqual(context["anomaly_summary"], {
            "observed": 10,
            "unique": 9,
            "included": LEAN_ANOMALY_LIMIT,
            "omitted": 1,
        })
        self.assertEqual(len(context["anomalies"]), LEAN_ANOMALY_LIMIT)
        self.assertLessEqual(len(context["anomalies"]), LEAN_ANOMALY_LIMIT)

        for row in request_rows:
            measured = len(json.dumps(
                row["messages"], separators=(",", ":")
            ).encode("utf-8"))
            self.assertEqual(row["prompt_bytes"], measured)
            supplied = json.loads(row["messages"][1]["content"])
            self.assertLessEqual(
                len(supplied["cross_episode_same_start_pairs"]),
                LEAN_CROSS_EPISODE_PAIR_LIMIT,
            )
            self.assertLessEqual(len(supplied["factual_log"]), LEAN_CONTEXT_HISTORY_SIZE)
            self.assertLessEqual(len(supplied["anomalies"]), LEAN_ANOMALY_LIMIT)
        late_sizes = [row["prompt_bytes"] for row in request_rows[16:]]
        self.assertLess(max(late_sizes) - min(late_sizes), 1_000)
        self.assertLess(max(late_sizes), 16_000)
        self.assertEqual(summary["request_prompt_bytes"]["first"], request_rows[0]["prompt_bytes"])
        self.assertEqual(summary["request_prompt_bytes"]["twentieth"], twentieth["prompt_bytes"])
        self.assertTrue(replay(path)["lean_facts_validated"])

    def test_note_problems_are_recorded_as_warnings_without_rejecting_action(self):
        summary, records, _, _ = self.run_fixture([
            response(decision("north", notes=[
                note(status="uncertain", evidence=[99]),
                {"claim": "missing id", "status": "uncertain", "evidence_steps": []},
            ])),
            response(decision("finish", notes=[note(status="supported", evidence=[1])])),
        ])

        self.assertEqual(summary["status"], "finished")
        self.assertEqual(summary["rejected_decisions"], 0)
        warning_rows = [row for row in records if row["type"] == "decision_warning"]
        self.assertGreaterEqual(len(warning_rows), 2)
        self.assertTrue(all(row["request_number"] == 1 for row in warning_rows))
        self.assertTrue(all(isinstance(row.get("code"), str) for row in warning_rows))
        self.assertTrue(all(isinstance(row.get("message"), str) for row in warning_rows))
        first_memory = next(
            row for row in records
            if row["type"] == "memory" and row["request_number"] == 1
        )
        self.assertEqual(first_memory["hypotheses"][0]["evidence_steps"], [])
        self.assertEqual(len(first_memory["notes_warnings"]), len(warning_rows))
        self.assertEqual(summary["warning_count"], len(summary["parse_warnings"]))
        self.assertGreaterEqual(summary["warning_count"], 2)

    def test_sparse_legal_action_records_and_replays(self):
        summary, records, path, _ = self.run_fixture([
            response({"action": "north"}),
            response(decision("finish")),
        ])

        self.assertEqual(summary["status"], "finished")
        self.assertEqual(summary["rejected_decisions"], 0)
        first_completion = next(
            row for row in records
            if row["type"] == "completion" and row["request_number"] == 1
        )
        canonical = json.loads(first_completion["answer"])
        self.assertEqual(canonical["action"], "north")
        self.assertEqual(canonical["reason"], "No valid reason supplied.")
        self.assertIsNone(canonical["expected_position"])
        self.assertEqual(canonical["notes"], [])
        self.assertTrue(any(
            row["type"] == "decision_warning"
            and row["request_number"] == 1
            and row["code"] == "invalid_reason"
            for row in records
        ))
        replayed = replay(path)
        self.assertEqual(replayed["actions_replayed"], 1)
        self.assertTrue(replayed["lean_facts_validated"])

    def test_factual_log_comparison_retention_replay_and_evidence_review(self):
        summary, records, path, _ = self.run_fixture([
            response(decision(
                "north",
                expected_position=[1, 0],
                notes=[note()],
            )),
            response(decision(
                "reset",
                expected_position=[1, 1],
                notes=[note(status="supported", evidence=[1])],
            )),
            response(decision(
                "north",
                expected_position=[1, 1],
                notes=[note(status="supported", evidence=[1])],
            )),
            response(decision(
                "finish",
                expected_position=None,
                notes=[note(status="supported", evidence=[1, 3])],
            )),
        ])

        self.assertEqual(summary["status"], "finished")
        self.assertEqual(summary["steps"], 3)
        self.assertEqual(summary["resets"], 1)
        self.assertIsNone(summary["unassessed_final_step"])
        requests = [row for row in records if row["type"] == "request"]
        after_first = json.loads(requests[1]["messages"][1]["content"])
        first_fact = after_first["factual_log"][0]
        self.assertEqual(first_fact["step"], 1)
        self.assertEqual(first_fact["action"], "north")
        self.assertEqual(first_fact["before"]["position"], [1, 1])
        self.assertEqual(first_fact["after"]["position"], [1, 1])
        self.assertEqual(first_fact["expected_position"], [1, 0])
        self.assertFalse(first_fact["expected_position_matched"])
        self.assertTrue(any("blocked" in item.lower() for item in first_fact["feedback"]))

        after_reset = json.loads(requests[2]["messages"][1]["content"])
        self.assertEqual(after_reset["episode"], 1)
        self.assertEqual(after_reset["notes"][0]["id"], "h1")
        final_context = json.loads(requests[3]["messages"][1]["content"])
        expected_pair = {
            "original_step": 1,
            "repeated_step": 3,
            "action": "north",
            "original_episode": 0,
            "repeated_episode": 1,
            "result": "same",
        }
        expected_pair_tuple = [1, 3, "north", 0, 1, "same"]
        self.assertEqual(
            final_context["cross_episode_pair_fields"],
            [
                "original_step", "repeated_step", "action",
                "original_episode", "repeated_episode", "result",
            ],
        )
        self.assertIn(expected_pair_tuple, final_context["cross_episode_same_start_pairs"])
        facts = final_context["retention_facts"]
        self.assertEqual(facts["pair_count"], 1)
        self.assertEqual(facts["same_outcome_pairs"], 1)
        north = next(item for item in facts["by_action"] if item["action"] == "north")
        self.assertEqual(north["pair_count"], 1)
        self.assertIn(expected_pair, summary["cross_episode_same_start_pairs"])
        self.assertEqual(summary["retention_facts"], facts)

        replayed = replay(path)
        self.assertEqual(replayed["actions_replayed"], 3)
        self.assertEqual(replayed["resets_replayed"], 1)
        self.assertEqual(replayed["exploration_status"], "finished")
        report = review_session(path, include_evidence=True)
        self.assertEqual(report["prompt_version"], LEAN_PROMPT_VERSION)
        self.assertIsNone(report["learning"]["unassessed_final_step"])
        self.assertEqual(
            report["lean_facts"]["cross_episode_same_start_pairs"],
            summary["cross_episode_same_start_pairs"],
        )
        self.assertEqual(report["lean_facts"]["retention_facts"], facts)
        self.assertTrue(report["lean_facts"]["facts_validated_by_replay"])
        self.assertFalse(report["lean_facts"]["claims_verified"])
        evidence = report["hypothesis_evidence"]
        self.assertTrue(evidence["available"])
        self.assertTrue(evidence["recording_complete"])
        self.assertTrue(evidence["software_facts_validated"])
        self.assertEqual(evidence["notes"][0]["id"], "h1")
        self.assertEqual(evidence["notes"][0]["evidence_steps"], [1, 3])
        receipts = evidence["notes"][0]["cited_steps"]
        self.assertEqual([item["control"] for item in receipts], ["north", "north"])
        self.assertEqual(
            [(item["episode_before"], item["episode_after"]) for item in receipts],
            [(0, 0), (1, 1)],
        )
        self.assertEqual(
            [item["model_expected_position"] for item in receipts],
            [[1, 0], [1, 1]],
        )
        self.assertEqual(
            [item["expected_position_matched"] for item in receipts],
            [False, True],
        )
        self.assertEqual(
            [item["actual_changed_facets"] for item in receipts],
            [[], []],
        )
        step_facts = {
            row["step"]: row["software_fact"]
            for row in records if row["type"] in {"step", "reset"}
        }
        self.assertEqual(
            [item["software_fact"] for item in receipts],
            [step_facts[1], step_facts[3]],
        )

        with patch(
            "autoplay_qa.cli.Settings.load",
            side_effect=AssertionError("Review must not load credentials"),
        ) as load, patch(
            "autoplay_qa.nebius.NebiusClient._request",
            side_effect=AssertionError("Review must not call inference"),
        ) as transport, patch("sys.stdout", new_callable=io.StringIO) as output:
            self.assertEqual(main(["review", str(path), "--evidence"]), 0)
        cli_report = json.loads(output.getvalue())
        self.assertEqual(cli_report["prompt_version"], LEAN_PROMPT_VERSION)
        self.assertEqual(cli_report["hypothesis_evidence"]["notes"][0]["id"], "h1")
        load.assert_not_called()
        transport.assert_not_called()

    def test_replay_rejects_tampered_step_and_request_software_facts(self):
        _, records, path, _ = self.run_fixture([
            response(decision("north", expected_position=[1, 0], notes=[note()])),
            response(decision(
                "finish",
                notes=[note(status="supported", evidence=[1])],
            )),
        ])

        def write_variant(name, changed):
            variant = path.parent / name
            variant.write_text(
                "\n".join(json.dumps(row) for row in changed) + "\n",
                encoding="utf-8",
            )
            return variant

        changed = copy.deepcopy(records)
        step = next(row for row in changed if row["type"] == "step")
        step["software_fact"]["expected_position_matched"] = True
        with self.assertRaisesRegex(ValueError, "lean step facts differ"):
            replay(write_variant("tampered-step.jsonl", changed))

        changed = copy.deepcopy(records)
        second_request = next(
            row for row in changed
            if row["type"] == "request" and row["request_number"] == 2
        )
        context = json.loads(second_request["messages"][1]["content"])
        context["factual_log"][0]["action"] = "south"
        second_request["messages"][1]["content"] = json.dumps(
            context, separators=(",", ":")
        )
        with self.assertRaisesRegex(ValueError, "lean request facts"):
            replay(write_variant("tampered-request.jsonl", changed))


class LeanCliTests(unittest.TestCase):
    def test_explore_mode_and_direct_fallback_cap_are_configurable(self):
        parsed = []

        def capture(args):
            parsed.append((args.mode, args.direct_fallback_max_tokens))
            return 0

        with patch("autoplay_qa.cli.explore", side_effect=capture) as handler:
            self.assertEqual(main(["explore"]), 0)
            self.assertEqual(main(["explore", "--mode", "legacy-v27"]), 0)
            self.assertEqual(main([
                "explore", "--direct-fallback-max-tokens", "640",
            ]), 0)

        self.assertEqual(parsed, [
            ("lean-v1", DEFAULT_DIRECT_FALLBACK_MAX_TOKENS),
            ("legacy-v27", DEFAULT_DIRECT_FALLBACK_MAX_TOKENS),
            ("lean-v1", 640),
        ])
        self.assertEqual(handler.call_count, 3)


if __name__ == "__main__":
    unittest.main()
