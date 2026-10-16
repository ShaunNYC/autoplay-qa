"""Offline evidence, judgement, cost, and provider-failure triage coverage."""

from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from autoplay_qa.config import Settings
from autoplay_qa.nebius import NebiusClient, NebiusError
from autoplay_qa.reporting import reproduce_issue
from autoplay_qa.session import SessionLog
from autoplay_qa.triage import (
    DEFAULT_TRIAGE_MODEL,
    DEFAULT_TRIAGE_INPUT_PRICE,
    DEFAULT_TRIAGE_OUTPUT_PRICE,
    TRIAGE_SCHEMA,
    TriageLimits,
    TriagePricing,
    build_evidence_packets,
    parse_assessment,
    render_triage_report,
    run_triage,
    triage_messages,
)
from tests.continuous_fixture import FakeContinuousGame
from tests.triage_fixture import (
    PRIVATE_SENTINELS,
    adapter_recording,
    assessment,
    blocked_note_recording,
    continuous_recording,
    grid_recording,
    provider_reply,
    write_recording,
)


def serialized(value):
    return json.dumps(value, separators=(",", ":"), ensure_ascii=False)


def nested_keys(value):
    if isinstance(value, dict):
        for key, item in value.items():
            yield key
            yield from nested_keys(item)
    elif isinstance(value, list):
        for item in value:
            yield from nested_keys(item)


class EvidencePacketTests(unittest.TestCase):
    def test_grid_packet_contains_executed_evidence_without_private_rules(self):
        records = grid_recording()
        original = deepcopy(records)
        packets = build_evidence_packets(records)

        self.assertEqual(len(packets), 1)
        packet = packets[0]
        self.assertEqual(packet["issue_number"], 1)
        self.assertEqual(packet["environment"]["seed"], 0)
        self.assertIn(5, packet["evidence_steps"])
        self.assertIn(5, packet["included_evidence_steps"])
        self.assertEqual(records, original)
        witness = next(row for row in packet["transitions"] if row["step"] == 5)
        self.assertEqual(witness["action"], "interact")
        self.assertIn("Unlocked", serialized(witness))
        self.assertIn("key", serialized(witness))
        self.assertIsNone(packet["reproduction"]["verified"])
        self.assertFalse(packet["reproduction"]["minimization_complete"])
        self.assertEqual(
            packet["reproduction"]["actions"],
            ["south", "south", "east", "east", "interact"],
        )
        prompt = serialized(triage_messages(packet))
        for sentinel in PRIVATE_SENTINELS:
            self.assertNotIn(sentinel, prompt)
        for private_key in (
            "rules", "evaluation", "injected_defect", "ground_truth",
            "unity_project", "reason", "neighbors", "passable",
        ):
            self.assertNotIn(private_key, set(nested_keys(packet)))

    def test_path_input_matches_in_memory_recording(self):
        records = grid_recording()
        with tempfile.TemporaryDirectory() as directory:
            path = write_recording(Path(directory) / "grid.jsonl", records)
            self.assertEqual(build_evidence_packets(path), build_evidence_packets(records))

    def test_nested_object_rules_are_not_sent_as_visible_evidence(self):
        records = grid_recording()
        for row in records[1:]:
            for field in ("before", "after"):
                row[field]["objects"]["gate"]["rules"] = [PRIVATE_SENTINELS[1]]
        prompt = serialized(triage_messages(build_evidence_packets(records)[0]))
        self.assertNotIn(PRIVATE_SENTINELS[1], prompt)

    def test_testing_capabilities_are_hidden_for_both_clean_and_defect_recordings(self):
        clean = adapter_recording(
            FakeContinuousGame(5),
            ("north", "north", "north", "north", "east", "north", "north"),
        )
        clean.append({
            "type": "memory",
            "hypotheses": [{
                "id": "movement-review",
                "kind": "suspected_bug",
                "status": "uncertain",
                "claim": "This movement may be unexpected.",
                "evidence_steps": [5],
            }],
        })
        visible_flags = {
            "grid": False, "legend": False,
            "continuous": True, "solids_visible": True,
        }
        capabilities = {
            **visible_flags,
            "defect_injection": True,
            "reset": True,
            "seed": True,
            "entities": True,
            "admin": {"active_test_mode": True},
            "private_harness": {"session_state": "HARNESS_STATE_MUST_STAY_LOCAL"},
        }
        for name, records in (("clean", clean), ("defect", continuous_recording())):
            with self.subTest(recording=name):
                snapshots = [records[0]["initial_observation"]]
                snapshots.extend(row[field] for row in records[1:]
                                 if row.get("type") == "step" for field in ("before", "after"))
                for snapshot in snapshots:
                    snapshot["_capabilities"] = deepcopy(capabilities)
                    for entity in snapshot["entities"]:
                        entity["state"]["defect_injection"] = True
                packets = build_evidence_packets(records)
                self.assertTrue(packets)
                for packet in packets:
                    context = json.loads(triage_messages(packet)[1]["content"])
                    included = [context["initial_observation"]]
                    included.extend(row[field] for row in context["transitions"]
                                    for field in ("before", "after"))
                    for snapshot in included:
                        self.assertEqual(snapshot["_capabilities"], visible_flags)
                    keys = set(nested_keys(context))
                    for testing_field in (
                        "defect_injection", "reset", "admin", "active_test_mode",
                        "private_harness", "session_state",
                    ):
                        self.assertNotIn(testing_field, keys)
                    self.assertNotIn("HARNESS_STATE_MUST_STAY_LOCAL", serialized(context))

    def test_long_wall_crossing_keeps_geometry_between_executed_endpoints(self):
        original = continuous_recording()
        row = deepcopy(original[1])
        wall = {
            "id": "crossed-wall",
            "kind": "wall",
            "position": [0.0, 1.0, 0.0],
            "size": [0.5, 2.0, 2.0],
            "state": {"solid": True},
        }
        row["decision"]["action"] = "east"
        row["before"]["player"]["position"] = [-10.0, 1.0, 0.0]
        row["after"]["player"]["position"] = [10.0, 1.0, 0.0]
        row["before"]["entities"] = [deepcopy(wall)]
        row["after"]["entities"] = [deepcopy(wall)]
        header = deepcopy(original[0])
        header["initial_observation"] = deepcopy(row["before"])
        packet = next(value for value in build_evidence_packets([header, row])
                      if value["candidate"]["type"] == "passed_through_solid")
        witness = packet["transitions"][0]
        self.assertIn("crossed-wall", serialized(witness["before"]["entities"]))
        self.assertIn("crossed-wall", serialized(witness["after"]["entities"]))

    def test_continuous_packet_retains_physics_and_public_geometry(self):
        packets = build_evidence_packets(continuous_recording())
        packet = next(
            value for value in packets
            if value["candidate"]["type"] == "sank_into_visible_floor"
        )
        witness = next(row for row in packet["transitions"] if row["step"] == 5)
        encoded = serialized(witness)
        self.assertIn("grounded", set(nested_keys(witness)))
        self.assertIn("velocity", set(nested_keys(witness)))
        self.assertIn("floor-2-4", encoded)
        self.assertIn("solid", set(nested_keys(witness)))
        self.assertIn("0.1", encoded)
        self.assertEqual(packet["reproduction"]["source_steps"], [1, 2, 3, 4, 5])

    def test_executed_witness_exposes_exact_continuous_player_change_before_geometry(self):
        records = continuous_recording()
        original = deepcopy(records)
        packet = next(value for value in build_evidence_packets(records)
                      if value["candidate"]["type"] == "sank_into_visible_floor")
        witness = next(value for value in packet["executed_witnesses"]
                       if value["step"] == 5)
        source = records[5]
        self.assertEqual(witness["player_before"], source["before"]["player"])
        self.assertEqual(witness["player_after"], source["after"]["player"])
        self.assertEqual(witness["position_delta"], [1.55, -0.95, 0.0])
        self.assertEqual(witness["player_after"]["position"], [-2.45, 0.1, 2.2])
        self.assertFalse(witness["player_after"]["grounded"])
        self.assertEqual(witness["player_after"]["velocity"], [0.0, 0.0, 0.0])
        self.assertEqual(witness["action"], source["decision"]["action"])
        self.assertEqual(witness["feedback"], source["after"]["events"])
        self.assertEqual(witness["status_after"], source["after"]["status"])
        self.assertNotIn("verdict", witness)
        self.assertEqual(records, original)
        context = json.loads(triage_messages(packet)[1]["content"])
        order = list(context)
        self.assertLess(order.index("executed_witnesses"), order.index("initial_observation"))
        self.assertLess(order.index("executed_witnesses"), order.index("transitions"))

    def test_entity_duplicate_object_views_are_removed_but_extra_object_data_is_kept(self):
        records = continuous_recording()
        records[5]["after"]["objects"]["door"]["interaction_radius"] = 1.8
        records[5]["after"]["objects"]["door"]["display_text"] = "The door is locked."
        records[5]["after"]["objects"]["mission"] = {"current_objective": "Reach the exit."}
        packet = next(value for value in build_evidence_packets(records)
                      if value["candidate"]["type"] == "sank_into_visible_floor")
        trace = next(value for value in packet["transitions"] if value["step"] == 5)
        self.assertNotIn("door", trace["before"]["objects"])
        self.assertIn("door", [value["id"] for value in trace["before"]["entities"]])
        self.assertEqual(trace["after"]["objects"]["door"], records[5]["after"]["objects"]["door"])
        self.assertEqual(trace["after"]["objects"]["mission"], {"current_objective": "Reach the exit."})
        self.assertNotIn("key", trace["after"]["objects"])
        self.assertIn("key", [value["id"] for value in trace["after"]["entities"]])
        self.assertGreater(trace["after"]["duplicate_objects_omitted"], 0)

    def test_initial_terrain_is_compacted_around_executed_samples_with_visible_omission_count(self):
        packet = next(value for value in build_evidence_packets(continuous_recording())
                      if value["candidate"]["type"] == "sank_into_visible_floor")
        initial = packet["initial_observation"]
        identifiers = [entity["id"] for entity in initial["entities"]]
        self.assertIn("floor-2-4", identifiers)
        self.assertNotIn("floor-6-0", identifiers)
        self.assertIn("exit", identifiers)
        self.assertGreater(initial["omitted_entity_count"], 0)

    def test_witness_ids_stay_aligned_when_packet_limit_removes_neighbor_transitions(self):
        records = continuous_recording()
        ordinary = next(value for value in build_evidence_packets(records)
                        if value["candidate"]["type"] == "sank_into_visible_floor")
        packet = next(value for value in build_evidence_packets(records, max_packet_bytes=9000)
                      if value["candidate"]["type"] == "sank_into_visible_floor")
        self.assertTrue(packet["usable"])
        self.assertLess(len(packet["transitions"]), len(ordinary["transitions"]))
        self.assertEqual([value["step"] for value in packet["transitions"]], [5])
        self.assertEqual([value["step"] for value in packet["executed_witnesses"]], [5])
        self.assertEqual(packet["included_evidence_steps"], [5])
        removed = next(value["step"] for value in ordinary["transitions"]
                       if value["step"] not in packet["included_evidence_steps"])
        normalized, warnings = parse_assessment(json.dumps(assessment(
            evidence_steps=(removed,),
        )), packet)
        self.assertEqual(normalized["verdict"], "inconclusive")
        self.assertTrue(warnings)

    def test_supplied_software_replay_is_separate_from_bug_judgement(self):
        records = continuous_recording()
        packets = build_evidence_packets(records)
        issue = next(
            value["issue_number"] for value in packets
            if value["candidate"]["type"] == "sank_into_visible_floor"
        )
        reproduction = reproduce_issue(
            records,
            issue,
            minimize=False,
            game_factory=lambda seed, defect: FakeContinuousGame(seed, defect),
        )
        self.assertTrue(reproduction["reproduced"])
        packet = next(value for value in build_evidence_packets(
            records, reproduction_results=[reproduction],
        ) if value["issue_number"] == issue)

        self.assertTrue(packet["reproduction"]["verified"])
        self.assertEqual(packet["reproduction"]["minimization_strategy"], "not_requested")
        self.assertFalse(packet["reproduction"]["minimization_complete"])
        self.assertNotIn("verdict", packet)
        normalized, _ = parse_assessment(json.dumps(assessment(
            verdict="inconclusive", severity="none", evidence_steps=(5,),
            rationale="Execution repeated, but player impact remains unclear.",
        )), packet)
        self.assertEqual(normalized["verdict"], "inconclusive")

    def test_failed_supplied_replay_cannot_mark_verified(self):
        packet = build_evidence_packets(grid_recording(), reproduction_results=[{
            "issue": 1,
            "reproduced": False,
            "actions": ["interact"],
            "source_steps": [5],
            "minimization_complete": False,
        }])[0]
        self.assertIsNot(packet["reproduction"]["verified"], True)

    def test_clean_run_has_no_candidates_but_grounded_suspected_note_is_reviewed(self):
        self.assertEqual(build_evidence_packets(grid_recording(defective=False)), [])
        packets = build_evidence_packets(blocked_note_recording())
        self.assertEqual(len(packets), 1)
        self.assertEqual(packets[0]["candidate"]["source"], "suspected_bug")
        normalized, _ = parse_assessment(json.dumps(assessment(
            verdict="not_bug", severity="none", evidence_steps=(1,),
            title="Movement was blocked by a visible wall",
            rationale="The map and feedback agree that north is a wall.",
        )), packets[0])
        self.assertEqual(normalized["verdict"], "not_bug")

    def test_unexecuted_suspected_note_does_not_create_issue(self):
        self.assertEqual(
            build_evidence_packets(blocked_note_recording(valid_reference=False)), [],
        )

    def test_negative_hybrid_witness_and_phase_step_are_preserved(self):
        records = continuous_recording()
        total = len(records) - 1
        for row in records[1:]:
            row["type"] = "sweep_step"
            row["source"] = "sweep"
            row["phase_step"] = row["step"]
            row["step"] = row["phase_step"] - total - 1
        packets = build_evidence_packets(records)
        packet = next(
            value for value in packets
            if value["candidate"]["type"] == "sank_into_visible_floor"
        )
        self.assertIn(-3, packet["included_evidence_steps"])
        witness = next(row for row in packet["transitions"] if row["step"] == -3)
        self.assertEqual(witness["phase_step"], 5)
        prominent = next(row for row in packet["executed_witnesses"] if row["step"] == -3)
        self.assertEqual(prominent["phase_step"], 5)
        self.assertEqual(prominent["position_delta"], [1.55, -0.95, 0.0])
        self.assertEqual(packet["reproduction"]["source_steps"], [-7, -6, -5, -4, -3])
        normalized, _ = parse_assessment(json.dumps(assessment(
            evidence_steps=(-3,),
        )), packet)
        self.assertEqual(normalized["evidence_steps"], [-3])

    def test_byte_bound_records_omissions_and_disallows_citing_omitted_step(self):
        records = continuous_recording()
        for row in records[1:]:
            for field in ("before", "after"):
                row[field]["events"] = ["Long visible feedback " * 300]
        packet = next(
            value for value in build_evidence_packets(records, max_packet_bytes=7000)
            if value["candidate"]["type"] == "sank_into_visible_floor"
        )
        self.assertLessEqual(len(serialized(packet).encode("utf-8")), 7000)
        self.assertTrue(packet["omissions"])
        self.assertTrue(packet["omitted_evidence_steps"])
        omitted = packet["omitted_evidence_steps"][0]
        normalized, warnings = parse_assessment(json.dumps(assessment(
            evidence_steps=(omitted,),
        )), packet)
        self.assertEqual(normalized["verdict"], "inconclusive")
        self.assertEqual(normalized["evidence_steps"], [])
        self.assertTrue(warnings)
        self.assertEqual(packet["executed_witnesses"], [])

    def test_unusable_fallback_is_bounded_even_with_many_merged_witness_ids(self):
        original = grid_recording()
        records = [original[0], *[
            {**original[-1], "step": step} for step in range(1, 1501)
        ]]
        issue = {
            "number": 1,
            "source": "software_anomaly",
            "type": "map_object_inconsistency",
            "title": "Many recorded sightings of one candidate",
            "expected": "Feedback and inventory agree.",
            "actual": "Feedback and inventory differ.",
            "evidence_steps": list(range(1, 1501)),
        }
        with patch("autoplay_qa.triage.extract_issues", return_value=[issue]):
            packet = build_evidence_packets(records, max_packet_bytes=4096)[0]
        self.assertLessEqual(len(serialized(packet).encode("utf-8")), 4096)
        self.assertFalse(packet["usable"])
        self.assertEqual(packet["included_evidence_steps"], [])
        self.assertTrue(packet["omissions"])


class AssessmentContractTests(unittest.TestCase):
    def setUp(self):
        self.packet = build_evidence_packets(grid_recording())[0]

    def test_schema_and_default_limits_are_separate_from_exploration(self):
        expected = set(assessment())
        self.assertEqual(set(TRIAGE_SCHEMA["required"]), expected)
        self.assertEqual(set(TRIAGE_SCHEMA["properties"]), expected)
        self.assertFalse(TRIAGE_SCHEMA["additionalProperties"])
        limits = TriageLimits()
        self.assertEqual(
            (limits.max_requests, limits.max_seconds, limits.timeout, limits.max_tokens, limits.max_usd),
            (8, 300, 45, 8192, 1.0),
        )
        self.assertEqual(DEFAULT_TRIAGE_MODEL, "nvidia/Nemotron-3-Ultra-550b-a55b")
        self.assertEqual((DEFAULT_TRIAGE_INPUT_PRICE, DEFAULT_TRIAGE_OUTPUT_PRICE), (1.0, 3.0))
        self.assertEqual(TriagePricing(), TriagePricing(1.0, 3.0))

    def test_valid_reference_survives_individual_invalid_and_boolean_refs(self):
        normalized, warnings = parse_assessment(json.dumps(assessment(
            evidence_steps=(5, 999, True, "5", 5, None),
        )), self.packet)
        self.assertEqual(normalized["verdict"], "bug")
        self.assertEqual(normalized["evidence_steps"], [5])
        self.assertGreaterEqual(len(warnings), 4)

    def test_unsupported_bug_and_nonbug_are_downgraded(self):
        for verdict in ("bug", "not_bug"):
            with self.subTest(verdict=verdict):
                normalized, warnings = parse_assessment(json.dumps(assessment(
                    verdict=verdict, evidence_steps=(999,),
                )), self.packet)
                self.assertEqual(normalized["verdict"], "inconclusive")
                self.assertEqual(normalized["severity"], "none")
                self.assertEqual(normalized["evidence_steps"], [])
                self.assertTrue(warnings)

    def test_nonbug_and_inconclusive_have_no_bug_severity(self):
        for verdict in ("not_bug", "inconclusive"):
            with self.subTest(verdict=verdict):
                normalized, warnings = parse_assessment(json.dumps(assessment(
                    verdict=verdict, severity="critical",
                )), self.packet)
                self.assertEqual(normalized["severity"], "none")
                self.assertTrue(warnings)

    def test_invalid_core_fields_cannot_be_treated_as_judgements(self):
        wrong_reference_container = assessment()
        wrong_reference_container["evidence_steps"] = "5"
        bad_values = [
            "not-json", "[]", "null", '```json\n{"verdict":"bug"}\n```',
            json.dumps({key: value for key, value in assessment().items() if key != "rationale"}),
            json.dumps(assessment(extra="invented")),
            json.dumps(assessment(verdict="confirmed")),
            json.dumps(assessment(severity="catastrophic")),
            json.dumps(assessment(title="   ")),
            json.dumps(assessment(player_impact=7)),
            json.dumps(assessment(rationale=None)),
            json.dumps(wrong_reference_container),
            json.dumps(assessment(limitations=[None])),
            json.dumps(assessment(severity="none")),
        ]
        for content in bad_values:
            with self.subTest(content=content):
                with self.assertRaises(ValueError):
                    parse_assessment(content, self.packet)


class TriageRunTests(unittest.TestCase):
    def setUp(self):
        self.packet = build_evidence_packets(grid_recording())[0]
        self.client = NebiusClient(
            Settings("offline-unit-test-key", model=DEFAULT_TRIAGE_MODEL),
            temperature=1.0,
            top_p=0.95,
        )

    def packets(self, count):
        values = [deepcopy(self.packet) for _ in range(count)]
        for index, value in enumerate(values, 1):
            value["issue_number"] = index
        return values

    def run_replies(self, replies, *, count=1, limits=None, pricing=None):
        with patch.object(self.client, "_request", side_effect=replies) as transport:
            result = run_triage(
                self.packets(count), self.client,
                limits=limits or TriageLimits(),
                pricing=pricing or TriagePricing(),
            )
        return result, transport

    def test_success_sends_ultra_schema_thinking_and_records_actual_cost(self):
        result, transport = self.run_replies([provider_reply()])
        body = transport.call_args.args[1]
        self.assertEqual(body["model"], DEFAULT_TRIAGE_MODEL)
        self.assertTrue(body["chat_template_kwargs"]["enable_thinking"])
        self.assertEqual(body["response_format"]["json_schema"]["schema"], TRIAGE_SCHEMA)
        self.assertEqual(body["max_tokens"], 8192)
        self.assertEqual(result["results"][0]["status"], "reviewed")
        self.assertEqual(result["results"][0]["assessment"]["verdict"], "bug")
        summary = result["summary"]
        self.assertEqual(summary["candidate_count"], 1)
        self.assertEqual(summary["requests"], 1)
        self.assertEqual(summary["final_pending_issues"], [])
        self.assertTrue(summary["usage_known"])
        self.assertAlmostEqual(summary["known_cost_usd"], 0.00025)
        self.assertAlmostEqual(summary["accounted_cost_usd"], 0.00025)
        self.assertGreater(summary["schema_bytes"], 0)
        self.assertEqual(len(summary["prompt_bytes"]), 1)
        self.assertGreater(summary["prompt_bytes"][0], 0)
        self.assertGreaterEqual(summary["elapsed_seconds"], 0)

    def test_failure_is_isolated_to_candidate_without_retry(self):
        result, transport = self.run_replies([
            NebiusError("Offline provider timeout."), provider_reply(),
        ], count=2)
        self.assertEqual(transport.call_count, 2)
        self.assertEqual(
            [row["status"] for row in result["results"]], ["failed", "reviewed"],
        )
        summary = result["summary"]
        self.assertEqual(summary["requests"], 2)
        self.assertFalse(summary["usage_known"])
        self.assertGreater(summary["accounted_cost_usd"], summary["known_cost_usd"])
        self.assertEqual(summary["final_pending_issues"], [1])

    def test_truncated_but_valid_json_is_failed_and_charged(self):
        result, transport = self.run_replies([provider_reply(finish_reason="length")])
        self.assertEqual(transport.call_count, 1)
        self.assertEqual(result["results"][0]["status"], "failed")
        self.assertIsNone(result["results"][0].get("assessment"))
        self.assertAlmostEqual(result["summary"]["known_cost_usd"], 0.00025)

    def test_unfinished_inline_thinking_is_not_persisted_as_an_answer(self):
        reply = provider_reply(finish_reason="length")
        private_trace = "PRIVATE_UNFINISHED_REASONING_SENTINEL"
        reply["choices"][0]["message"]["content"] = f"<think>{private_trace}"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "truncated.jsonl"
            with SessionLog(path) as log:
                with patch.object(self.client, "_request", return_value=reply):
                    result = run_triage([self.packet], self.client, log=log)
            audit_text = path.read_text(encoding="utf-8")
        self.assertNotIn(private_trace, audit_text)
        self.assertNotIn(private_trace, serialized(result))
        self.assertNotIn(private_trace, render_triage_report(result))
        completion = next(
            json.loads(line) for line in audit_text.splitlines()
            if json.loads(line)["type"] == "triage_completion"
        )
        self.assertIsNone(completion["content"])
        self.assertEqual(result["results"][0]["status"], "failed")
        self.assertAlmostEqual(result["summary"]["known_cost_usd"], 0.00025)

    def test_invalid_json_failure_does_not_consume_next_candidate(self):
        reply = provider_reply()
        reply["choices"][0]["message"]["content"] = "unfinished {"
        result, transport = self.run_replies([reply, provider_reply()], count=2)
        self.assertEqual(transport.call_count, 2)
        self.assertEqual([row["status"] for row in result["results"]], ["failed", "reviewed"])

    def test_interruption_preserves_failed_and_unattempted_issue_reservations(self):
        result, transport = self.run_replies([KeyboardInterrupt()], count=2)
        self.assertEqual(transport.call_count, 1)
        self.assertEqual(result["status"], "interrupted")
        self.assertEqual([row["status"] for row in result["results"]], ["failed", "pending"])
        self.assertEqual(result["summary"]["final_pending_issues"], [1, 2])
        self.assertFalse(result["summary"]["usage_known"])
        self.assertEqual(result["summary"]["known_cost_usd"], 0)
        self.assertGreater(result["summary"]["accounted_cost_usd"], 0)

    def test_unusable_byte_bounded_packet_never_reaches_provider(self):
        packet = deepcopy(self.packet)
        packet["usable"] = False
        packet["included_evidence_steps"] = []
        packet["transitions"] = []
        with patch.object(self.client, "_request") as transport:
            result = run_triage([packet], self.client)
        transport.assert_not_called()
        self.assertEqual(result["summary"]["requests"], 0)
        self.assertEqual(result["results"][0]["status"], "failed")
        self.assertEqual(result["summary"]["final_pending_issues"], [1])
        self.assertEqual(result["summary"]["accounted_cost_usd"], 0)

    def test_missing_and_invalid_usage_use_full_preallocated_reservation(self):
        for usage in (False, {"prompt_tokens": -1, "completion_tokens": 50}, {
            "prompt_tokens": True, "completion_tokens": 50,
        }):
            with self.subTest(usage=usage):
                result, _ = self.run_replies([provider_reply(usage=usage)])
                self.assertEqual(result["results"][0]["status"], "reviewed")
                self.assertFalse(result["summary"]["usage_known"])
                self.assertGreater(result["summary"]["accounted_cost_usd"], 0)
                self.assertEqual(result["summary"]["known_cost_usd"], 0)
                self.assertGreaterEqual(
                    result["summary"]["accounted_cost_usd"],
                    8192 * DEFAULT_TRIAGE_OUTPUT_PRICE / 1_000_000,
                )

    def test_unknown_usage_reservation_stops_next_request_before_spending(self):
        first, _ = self.run_replies([provider_reply(usage=False)])
        reservation = first["summary"]["accounted_cost_usd"]
        result, transport = self.run_replies(
            [provider_reply(usage=False), provider_reply()],
            count=2, limits=TriageLimits(max_usd=reservation * 1.5),
        )
        self.assertEqual(transport.call_count, 1)
        self.assertEqual(result["summary"]["requests"], 1)
        self.assertEqual(result["summary"]["final_pending_issues"], [2])
        self.assertEqual(result["results"][1]["status"], "pending")

    def test_tiny_budget_never_calls_provider(self):
        result, transport = self.run_replies(
            [provider_reply()], limits=TriageLimits(max_usd=1e-12),
        )
        transport.assert_not_called()
        self.assertEqual(result["summary"]["requests"], 0)
        self.assertEqual(result["summary"]["final_pending_issues"], [1])
        self.assertEqual(result["summary"]["accounted_cost_usd"], 0)

    def test_request_limit_reports_every_unreviewed_candidate_pending(self):
        result, transport = self.run_replies(
            [provider_reply()], count=3, limits=TriageLimits(max_requests=1),
        )
        self.assertEqual(transport.call_count, 1)
        self.assertEqual(result["summary"]["final_pending_issues"], [2, 3])
        self.assertEqual([row["status"] for row in result["results"]], ["reviewed", "pending", "pending"])

    def test_time_budget_caps_timeout_then_preserves_pending_candidates(self):
        clock = [0.0]

        def slow_reply(*_args):
            clock[0] += 10.0
            return provider_reply()

        with patch("autoplay_qa.triage.time.monotonic", side_effect=lambda: clock[0]):
            result, transport = self.run_replies(
                slow_reply, count=2,
                limits=TriageLimits(max_seconds=10, timeout=45),
            )
        self.assertEqual(transport.call_count, 1)
        timeout = transport.call_args.args[2]
        self.assertLessEqual(timeout, 10)
        self.assertGreater(timeout, 0)
        self.assertEqual(result["summary"]["final_pending_issues"], [2])

    def test_no_candidates_and_dry_run_need_no_client(self):
        with patch("autoplay_qa.nebius.NebiusClient._request") as transport:
            empty = run_triage([], None)
            dry_run = run_triage([self.packet], None)
        transport.assert_not_called()
        self.assertEqual(empty["summary"]["candidate_count"], 0)
        self.assertEqual(empty["summary"]["requests"], 0)
        self.assertEqual(dry_run["summary"]["requests"], 0)
        self.assertEqual(dry_run["summary"]["final_pending_issues"], [1])

    def test_secret_echo_is_redacted_from_result_report_and_audit(self):
        secret = 'offline-"key\\with-escapes'
        client = NebiusClient(Settings(secret, model=DEFAULT_TRIAGE_MODEL))
        echoed = f"Untrusted text {secret} and {json.dumps(secret)[1:-1]}"
        reply = provider_reply(assessment(rationale=echoed))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "triage.jsonl"
            with SessionLog(path, (secret,)) as log:
                with patch.object(client, "_request", return_value=reply):
                    result = run_triage([self.packet], client, log=log)
            contents = [serialized(result), render_triage_report(result), path.read_text()]
        for contents_part in contents:
            self.assertNotIn(secret, contents_part)
            self.assertNotIn(json.dumps(secret)[1:-1], contents_part)
            self.assertIn("[REDACTED]", contents_part)

    def test_error_echo_is_redacted_with_unknown_usage_and_no_retry(self):
        secret = 'offline-"error\\key'
        client = NebiusClient(Settings(secret, model=DEFAULT_TRIAGE_MODEL))
        with patch.object(client, "_request", side_effect=NebiusError(secret)) as transport:
            result = run_triage([self.packet], client)
        self.assertEqual(transport.call_count, 1)
        self.assertNotIn(secret, serialized(result))
        self.assertFalse(result["summary"]["usage_known"])

    def test_report_has_model_judgement_actual_steps_and_replay_limit(self):
        result, _ = self.run_replies([provider_reply()])
        report = render_triage_report(result)
        self.assertIn("Key retained", report)
        self.assertIn("high", report.lower())
        self.assertIn("interact", report)
        self.assertIn("5", report)
        self.assertIn("session", report.lower())
        self.assertIn("replay", report.lower())
        self.assertIn("judg", report.lower())
        self.assertNotIn("PRIVATE_", report)


if __name__ == "__main__":
    unittest.main()
