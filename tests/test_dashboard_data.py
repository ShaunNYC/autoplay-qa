"""Dashboard evidence stays public, replayable, and honest about its source.

These tests use adapter-generated recordings and stub HTTP only.  The public
dashboard must preserve the original software candidate even when Ultra
disagrees, and must never export developer metadata or raw provider thinking.
"""

from copy import deepcopy
import json
import unittest
from unittest.mock import patch

from autoplay_qa.config import Settings
from autoplay_qa.dashboard_data import public_observation, snapshot_from_records
from autoplay_qa.nebius import NebiusClient
from autoplay_qa.reporting import extract_issues
from autoplay_qa.triage import build_evidence_packets, run_triage
from tests.triage_fixture import (
    PRIVATE_SENTINELS,
    assessment,
    blocked_note_recording,
    continuous_recording,
    grid_recording,
    provider_reply,
)


CREATED_AT = "2026-10-14T16:00:00Z"
SECRET = "DASHBOARD_TEST_KEY_MUST_STAY_LOCAL"


def encoded(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def keys_in(value):
    if isinstance(value, dict):
        for key, item in value.items():
            yield key
            yield from keys_in(item)
    elif isinstance(value, list):
        for item in value:
            yield from keys_in(item)


def reviewed(records, value):
    """Run the real review parser with a fake transport, without an API key."""
    client = NebiusClient(Settings(SECRET))
    with patch.object(client, "_request", return_value=provider_reply(value)):
        return run_triage(build_evidence_packets(records), client)


def snapshot(records, **kwargs):
    return snapshot_from_records(
        records, run_id="fixture-recorded-run", created_at=CREATED_AT,
        secrets=(SECRET,), **kwargs,
    )


class DashboardObservationTests(unittest.TestCase):
    def test_visible_physics_geometry_and_feedback_survive_without_private_data(self):
        observation = deepcopy(continuous_recording()[5]["after"])
        observation.update({
            "evaluation": {"injected_defect": "HIDDEN_INJECTION"},
            "rules": ["HIDDEN_RULES"],
            "unity_project": "/private/developer/HIDDEN_PROJECT",
            "reasoning_content": "HIDDEN_THINKING",
            "api_key": SECRET,
        })
        observation["player"]["private_harness"] = "HIDDEN_PLAYER_HARNESS"
        observation["entities"][0]["state"].update({
            "defect_injection": True,
            "developer_rules": "HIDDEN_ENTITY_RULES",
            "developer_route": ["HIDDEN_ROUTE"],
            "victory_strategy": "HIDDEN_STRATEGY",
            "solution": "HIDDEN_SOLUTION",
            "private_evaluation": "HIDDEN_ENTITY_LABEL",
            "authorization": SECRET,
        })
        observation.setdefault("objects", {})["visible-panel"] = {
            "display_text": "The door is locked.",
            "nested": {"secret_config": "HIDDEN_NESTED_CONFIG"},
        }
        original = deepcopy(observation)

        public = public_observation(observation, secrets=(SECRET,))

        self.assertEqual(observation, original)
        self.assertEqual(public["player"]["position"], observation["player"]["position"])
        self.assertEqual(public["player"]["velocity"], observation["player"]["velocity"])
        self.assertEqual(public["player"]["grounded"], observation["player"]["grounded"])
        self.assertEqual(public["events"], observation["events"])
        self.assertTrue(public["entities"])
        self.assertTrue(any(row["state"].get("solid") is True for row in public["entities"]))
        self.assertIn("The door is locked.", encoded(public))
        for sentinel in (SECRET, "HIDDEN_", "/private/developer/"):
            self.assertNotIn(sentinel, encoded(public))
        for key in (
            "evaluation", "injected_defect", "rules", "unity_project",
            "reasoning_content", "api_key", "defect_injection", "authorization",
            "developer_route", "victory_strategy", "solution",
        ):
            self.assertNotIn(key, set(keys_in(public)))

    def test_secret_redaction_applies_to_visible_strings_and_nested_arrays(self):
        observation = {
            "events": ["Provider echoed " + SECRET],
            "player": {"position": [2, 3], "inventory": [SECRET]},
            "objects": {"sign": {"text": "token=" + SECRET}},
        }
        public = public_observation(observation, secrets=(SECRET,))
        self.assertNotIn(SECRET, encoded(public))
        self.assertEqual(public["player"]["position"], [2, 3])

    def test_bad_observation_shapes_produce_safe_public_values(self):
        for value in (None, [], "unexpected", 123):
            with self.subTest(value=value):
                self.assertEqual(public_observation(value), {})

    def test_public_object_details_beyond_duplicate_geometry_remain_visible(self):
        observation = deepcopy(continuous_recording()[5]["after"])
        observation["objects"]["door"]["display_text"] = "The door is locked."
        observation["objects"]["door"]["interaction_radius"] = 1.8

        public = public_observation(observation)

        self.assertEqual(public["objects"]["door"]["display_text"], "The door is locked.")
        self.assertEqual(public["objects"]["door"]["interaction_radius"], 1.8)
        self.assertIn("door", [row["id"] for row in public["entities"]])


class DashboardRecordingTests(unittest.TestCase):
    def test_recorded_import_is_honest_even_when_the_original_session_was_live(self):
        records = grid_recording()
        records[0].update({"mode": "live", "source": "live", "name": SECRET})
        records.append({"type": "request", "messages": [{"content": "RAW_PROMPT " + SECRET}]})
        records.append({
            "type": "completion", "content": "RAW_THINKING " + SECRET,
            "reasoning_content": "HIDDEN_THINKING",
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        })
        original = deepcopy(records)
        with patch("autoplay_qa.nebius.NebiusClient._request", side_effect=AssertionError("Import cannot infer")):
            public = snapshot(records)

        self.assertEqual(records, original)
        self.assertEqual(public["version"], "dashboard-v1")
        self.assertEqual(public["summary"]["source"], "recorded")
        self.assertEqual(public["summary"]["status"], "completed")
        self.assertEqual(public["summary"]["createdAt"], CREATED_AT)
        self.assertEqual(len(public["frames"]), 5)
        for text in (*PRIVATE_SENTINELS[:2], PRIVATE_SENTINELS[3], SECRET, "RAW_PROMPT", "RAW_THINKING", "HIDDEN_THINKING"):
            self.assertNotIn(text, encoded(public))

    def test_brief_exploration_reason_and_exact_executed_action_are_visible(self):
        records = grid_recording()
        records[1]["decision"]["reason"] = "Try the visible passage to the south."
        public = snapshot(records)
        frame = next(row for row in public["frames"] if row["step"] == 1)
        self.assertEqual(frame["action"], "south")
        self.assertEqual(frame["reason"], "Try the visible passage to the south.")
        self.assertEqual(frame["before"]["player"]["position"], records[1]["before"]["player"]["position"])
        self.assertEqual(frame["after"]["player"]["position"], records[1]["after"]["player"]["position"])

    def test_negative_sweep_ids_and_original_phase_steps_remain_cross_referencable(self):
        records = grid_recording()
        total = len(records) - 1
        for row in records[1:]:
            row["type"] = "sweep_step"
            row["source"] = "sweep"
            row["phase_step"] = row["step"]
            row["step"] = row["phase_step"] - total - 1
        first_llm = deepcopy(records[-1])
        first_llm.update({"type": "step", "source": "llm", "phase_step": 1, "step": 1})
        records.append(first_llm)

        public = snapshot(records)

        sweep = [row for row in public["frames"] if row["phase"] == "sweep"]
        self.assertEqual([row["step"] for row in sweep], [-5, -4, -3, -2, -1])
        self.assertEqual([row["phaseStep"] for row in sweep], [1, 2, 3, 4, 5])
        self.assertEqual(len({row["id"] for row in public["frames"]}), 6)
        llm = next(row for row in public["frames"] if row["step"] == 1)
        self.assertEqual(llm["phase"], "llm")
        self.assertEqual(llm["phaseStep"], 1)
        key_issue = next(row for row in public["issues"] if -1 in row["evidenceSteps"])
        self.assertIn(-1, key_issue["reproduction"]["sourceSteps"])
        self.assertEqual(key_issue["reproduction"]["actions"][-1], "interact")

    def test_existing_legacy_decision_plan_is_importable_without_new_contract_fields(self):
        records = grid_recording()
        records[0]["schema_version"] = 1
        records[0]["task"] = "run"
        for row in records[1:]:
            row["decision"] = {"action": row["decision"]["action"], "reason": "Legacy brief reason.", "plan": "Legacy plan."}
        public = snapshot(records)
        self.assertEqual(len(public["frames"]), 5)
        self.assertEqual(public["frames"][0]["reason"], "Legacy brief reason.")
        self.assertEqual(public["summary"]["controls"], 5)

    def test_reset_is_counted_as_a_control_with_its_actual_after_state(self):
        records = grid_recording()
        reset = deepcopy(records[-1])
        reset.update({"type": "reset", "step": 6, "phase_step": 6})
        reset["decision"]["action"] = "reset"
        reset["after"] = deepcopy(records[0]["initial_observation"])
        records.append(reset)

        public = snapshot(records)

        self.assertEqual(public["summary"]["controls"], 6)
        self.assertEqual(public["summary"]["resets"], 1)
        self.assertEqual(public["frames"][-1]["action"], "reset")
        self.assertEqual(public["frames"][-1]["after"]["player"]["position"], records[0]["initial_observation"]["player"]["position"])

    def test_live_marking_requires_the_explicit_runner_import_argument(self):
        public = snapshot(grid_recording(), source="live", status="running", phase="explore")
        self.assertEqual(public["summary"]["source"], "live")
        self.assertEqual(public["summary"]["status"], "running")
        self.assertEqual(public["summary"]["phase"], "explore")

    def test_public_limits_describe_the_preserved_bounded_exploration_and_review(self):
        public = snapshot(grid_recording())
        self.assertEqual(public["limits"]["modelControls"], 24)
        self.assertEqual(public["limits"]["explorationRequests"], 28)
        self.assertEqual(public["limits"]["triageRequests"], 8)
        self.assertEqual(public["limits"]["explorationSeconds"], 300)
        self.assertEqual(public["limits"]["triageSeconds"], 300)
        self.assertEqual(public["limits"]["timeout"], 45)
        self.assertEqual(public["limits"]["maxTokens"], 8192)

    def test_unknown_usage_stays_unknown_in_public_summary(self):
        records = grid_recording()
        records.append({"type": "request", "request_number": 1})
        records.append({"type": "inference_error", "request_number": 1, "usage_known": False})
        public = snapshot(records, cost={
            "accounted_cost_usd": 0.025, "known_cost_usd": 0,
            "usage_known": False, "unknown_usage_requests": 1, "requests": 1,
        }, budget_usd=1)
        self.assertFalse(public["summary"]["costKnown"])
        self.assertAlmostEqual(public["summary"]["estimatedCostUsd"], 0.025)
        self.assertEqual(public["summary"]["requests"], 1)
        self.assertEqual(public["summary"]["budgetUsd"], 1)


class DashboardAssessmentTests(unittest.TestCase):
    def test_not_bug_review_keeps_original_software_issue_and_executed_evidence(self):
        records = grid_recording()
        host_issue = extract_issues(records)[0]
        triage = reviewed(records, assessment(verdict="not_bug", severity="none"))
        public = snapshot(records, triage_result=triage)

        self.assertEqual(public["summary"]["issueCount"], 1)
        issue = public["issues"][0]
        self.assertEqual(issue["title"], host_issue["title"])
        self.assertEqual(issue["source"], host_issue["source"])
        self.assertEqual(issue["evidenceSteps"], host_issue["evidence_steps"])
        self.assertEqual(issue["assessment"]["verdict"], "not_bug")
        self.assertEqual(issue["assessment"]["severity"], "none")
        self.assertEqual(issue["assessment"]["status"], "reviewed")
        self.assertEqual(issue["reproduction"]["actions"], ["south", "south", "east", "east", "interact"])

    def test_key_consumption_disagreement_always_remains_for_human_review(self):
        records = grid_recording()
        for verdict, severity in (("not_bug", "none"), ("bug", "high"), ("inconclusive", "low")):
            with self.subTest(verdict=verdict):
                public = snapshot(records, triage_result=reviewed(records, assessment(verdict=verdict, severity=severity)))
                self.assertTrue(public["issues"][0]["needsHumanReview"])
                self.assertTrue(public["issues"][0].get("reviewNote"))

    def test_authored_normal_candidate_is_never_promoted_after_rejected_review(self):
        records = blocked_note_recording()
        triage = reviewed(records, assessment(
            verdict="not_bug", severity="none", evidence_steps=(1,),
            title="Ordinary blocked movement", player_impact="Movement matches the visible wall.",
        ))
        public = snapshot(records, triage_result=triage)
        self.assertEqual(len(public["issues"]), 1)
        self.assertEqual(public["issues"][0]["source"], "suspected_bug")
        self.assertEqual(public["issues"][0]["assessment"]["verdict"], "not_bug")

    def test_pending_review_never_claims_severity_or_verified_reproduction(self):
        records = grid_recording()
        public = snapshot(records, triage_result=run_triage(build_evidence_packets(records)))
        issue = public["issues"][0]
        self.assertEqual(issue["assessment"]["status"], "pending")
        self.assertIsNone(issue["assessment"].get("verdict"))
        self.assertIsNone(issue["assessment"].get("severity"))
        self.assertIsNone(issue["reproduction"]["verified"])
        self.assertFalse(issue["reproduction"]["minimal"])
        self.assertEqual(public["summary"]["pendingCount"], 1)

    def test_bad_review_reference_is_dropped_with_its_warning_kept_visible(self):
        records = grid_recording()
        triage = reviewed(records, assessment(evidence_steps=(5, 999)))
        self.assertTrue(triage["results"][0]["warnings"])
        public = snapshot(records, triage_result=triage)
        result = public["issues"][0]["assessment"]
        self.assertEqual(result["evidenceSteps"], [5])
        self.assertTrue(result["warnings"])
        self.assertGreaterEqual(public["summary"]["warningCount"], 1)


if __name__ == "__main__":
    unittest.main()
