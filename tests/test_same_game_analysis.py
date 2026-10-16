"""Offline fixtures test scoring/provenance; they are not model-performance data."""

from copy import deepcopy
import hashlib
import json
import unittest
from unittest.mock import patch

from autoplay_qa.same_game_analysis import (
    ARMS, CASE_MANIFEST, REVIEW_SCHEMA, SLOTS, analyze_fixed_suite, candidate_id,
    discordant_win_interval, exact_paired_test, make_review_bundle,
)


def fixture():
    rows = []
    for case, family, stratum in CASE_MANIFEST:
        for history in (1, 2):
            block = f"{case}h{history}"
            for slot in SLOTS:
                rows.append({"block_id": block, "case_id": case, "history_number": history,
                             "family_id": family, "stratum": stratum, "slot": slot,
                             "arm": slot if slot in ARMS else None, "game": "unity",
                             "acquisition": slot not in ARMS, "completed": True,
                             "status": "finished", "usage_known": True,
                             "unassessed_final_outcome": None, "steps": 2, "resets": 0,
                             "inference_requests": 3, "native_replay": {"verified": True},
                             "strict_replay": {"verified": True},
                             "recording_sha256": hashlib.sha256(f"{block}:{slot}".encode()).hexdigest(),
                             "candidates": []})
    return rows, {"schema": REVIEW_SCHEMA, "reviewer": "Offline fixture reviewer",
                  "candidate_reviews": [], "exposure_reviews": []}


def phase(rows, block, slot):
    return next(row for row in rows if (row["block_id"], row["slot"]) == (block, slot))


def add_report(row, reviews, *, verdict="supported_defect", family="target", reproduced=True):
    number = len(row["candidates"]) + 1
    identifier = candidate_id(row["recording_sha256"], number)
    row["candidates"].append({"candidate_id": identifier, "number": number,
                              "source": "software_anomaly", "type": "sank_into_visible_floor",
                              "evidence_steps": [2], "reproduced": reproduced})
    reviews["candidate_reviews"].append({"candidate_id": identifier,
                                         "recording_sha256": row["recording_sha256"],
                                         "verdict": verdict,
                                         "matched_family": row["family_id"] if family == "target" else family,
                                         "evidence_steps": [2],
                                         "source_basis": "Independent fixture source check and native replay.",
                                         "reproduction_verified": verdict == "supported_defect"})
    return reviews["candidate_reviews"][-1]


def outcomes(wins=0, losses=0):
    rows, reviews = fixture()
    blocks = [f"{case}h{history}" for case, _, stratum in CASE_MANIFEST
              if stratum == "positive" for history in (1, 2)]
    for block in blocks[:wins]:
        add_report(phase(rows, block, "eval3"), reviews)
    for block in blocks[wins:wins + losses]:
        add_report(phase(rows, block, "eval0"), reviews)
    return rows, reviews


class ExactPairedTests(unittest.TestCase):
    def test_two_sided_exact_values_and_no_optional_direction_switch(self):
        for wins, losses, expected in ((0, 0, 1), (3, 0, .25), (5, 0, .0625),
                                       (6, 0, .03125), (7, 1, .0703125), (8, 0, .0078125),
                                       (4, 4, 1), (0, 6, .03125)):
            with self.subTest(wins=wins, losses=losses):
                result = exact_paired_test(wins, losses)
                self.assertEqual(result["p_value"], expected)
                self.assertEqual(result["reject_equal_direction"], expected <= .05)

    def test_interval_is_direction_not_accuracy_difference(self):
        interval = discordant_win_interval(6, 0)
        self.assertAlmostEqual(interval["lower"], .025 ** (1 / 6), places=12)
        self.assertEqual(interval["upper"], 1)
        self.assertIn("not absolute accuracy", interval["scope"])
        inverse = discordant_win_interval(0, 6)
        self.assertAlmostEqual(inverse["upper"], 1 - interval["lower"], places=12)
        empty = discordant_win_interval(0, 0)
        self.assertEqual((empty["estimate"], empty["lower"], empty["upper"]), (None, 0, 1))

    def test_invalid_counts_and_confidence_rejected(self):
        for values in ((True, 1), (-1, 0), (1.5, 0)):
            with self.subTest(values=values), self.assertRaises(ValueError):
                exact_paired_test(*values)
        for confidence in (True, 0, 1, float("nan"), float("inf")):
            with self.subTest(confidence=confidence), self.assertRaises(ValueError):
                discordant_win_interval(1, 0, confidence)


class FixedSuiteAnalysisTests(unittest.TestCase):
    def test_complete_zero_hit_matrix_is_not_improvement_or_perfect_precision(self):
        rows, reviews = fixture()
        original = deepcopy((rows, reviews))
        result = analyze_fixed_suite(rows, reviews)
        self.assertEqual(result["status"], "no_supported_fixed_suite_effect")
        self.assertEqual(result["validation_errors"], [])
        self.assertEqual(result["primary"]["positive_pairs"], 8)
        self.assertEqual(result["false_reports"]["clean_pairs_observed"], 4)
        self.assertEqual(result["primary"]["hits"], {arm: 0 for arm in ARMS})
        self.assertTrue(all(value is None for value in result["false_reports"]["per_arm_report_precision"].values()))
        self.assertEqual((rows, reviews), original)

    def test_net_three_is_engineering_only_and_six_discordant_wins_support_fixed_suite(self):
        for wins, status, signal in ((3, "no_supported_fixed_suite_effect", True),
                                     (5, "no_supported_fixed_suite_effect", True),
                                     (6, "supported_fixed_suite_effect", True)):
            with self.subTest(wins=wins):
                rows, reviews = outcomes(wins)
                result = analyze_fixed_suite(rows, reviews)
                self.assertEqual(result["status"], status)
                self.assertEqual(result["primary"]["net_target_hits"], wins)
                self.assertEqual(result["secondary"]["engineering_net_at_least_three"], signal)
                self.assertEqual(result["primary"]["paired_detection_rate_difference"], wins / 8)

    def test_significant_degradation_is_not_supported_improvement(self):
        rows, reviews = outcomes(0, 6)
        result = analyze_fixed_suite(rows, reviews)
        self.assertTrue(result["primary"]["test"]["reject_equal_direction"])
        self.assertEqual(result["status"], "no_supported_fixed_suite_effect")
        self.assertEqual(len(result["regressions"]), 6)

    def test_all_target_matching_comes_from_review_not_detector_type(self):
        rows, reviews = fixture()
        add_report(phase(rows, "u01h1", "eval3"), reviews, family=None)
        result = analyze_fixed_suite(rows, reviews)
        self.assertEqual(result["primary"]["hits"]["eval3"], 0)
        self.assertEqual(result["paired_blocks"][0]["arms"]["eval3"]["supported_other_reports"], 1)

    def test_unsupported_reports_on_positive_or_clean_block_statistical_success(self):
        for block in ("u01h1", "u05h1"):
            with self.subTest(block=block):
                rows, reviews = outcomes(6)
                add_report(phase(rows, block, "eval3"), reviews, verdict="unsupported", family=None)
                result = analyze_fixed_suite(rows, reviews)
                self.assertEqual(result["status"], "no_supported_fixed_suite_effect")
                self.assertFalse(result["primary"]["no_added_unsupported_reports"])
                self.assertEqual(result["false_reports"]["all_evaluation_arms"]["eval3"], 1)

    def test_same_block_false_count_guard_and_secondary_harms_are_explicit(self):
        rows, reviews = outcomes(6)
        add_report(phase(rows, "u01h1", "eval0"), reviews, verdict="expected_behavior", family=None)
        add_report(phase(rows, "u01h1", "eval3"), reviews, verdict="unsupported", family=None)
        add_report(phase(rows, "u05h2", "eval1"), reviews, verdict="unsupported", family=None)
        result = analyze_fixed_suite(rows, reviews)
        self.assertEqual(result["status"], "supported_fixed_suite_effect")
        self.assertTrue(result["primary"]["no_added_unsupported_reports"])
        self.assertEqual(result["false_reports"]["paired_increases"], [{"block_id": "u05h2", "arm": "eval1", "additional_false_reports": 1}])
        self.assertIn("larger unsupported-report count", result["false_reports"]["meaning"])

    def test_supported_defect_on_authored_clean_is_contamination_not_false_positive(self):
        rows, reviews = outcomes(6)
        add_report(phase(rows, "u05h1", "eval3"), reviews, family=None)
        result = analyze_fixed_suite(rows, reviews)
        self.assertEqual(result["status"], "inconclusive")
        self.assertIn("authored_clean_control_contains_supported_defect", result["validation_errors"])
        self.assertEqual(result["false_reports"]["all_evaluation_arms"]["eval3"], 0)

    def test_missing_acquisition_or_eval_is_inconclusive_not_favorable_subset(self):
        for slot in ("acquire-2", "eval0"):
            with self.subTest(slot=slot):
                rows, reviews = outcomes(6)
                rows.remove(phase(rows, "u06h2", slot))
                result = analyze_fixed_suite(rows, reviews)
                self.assertEqual(result["status"], "inconclusive")
                self.assertIn("incomplete_predeclared_matrix", result["validation_errors"])
                self.assertFalse(result["primary"]["test"]["inference_valid_for_predeclared_matrix"])

    def test_missing_unreviewed_unreproduced_or_forged_review_never_scores_success(self):
        for mutation in ("missing", "unreviewed", "source", "native", "reproduced", "citation", "sha", "unknown"):
            with self.subTest(mutation=mutation):
                rows, reviews = outcomes(6)
                review = reviews["candidate_reviews"][0]
                row = phase(rows, "u01h1", "eval3")
                if mutation == "missing": reviews["candidate_reviews"].pop(0)
                elif mutation == "unreviewed": review["verdict"] = "unreviewed"
                elif mutation == "source": review["source_basis"] = ""
                elif mutation == "native": review["reproduction_verified"] = False
                elif mutation == "reproduced": row["candidates"][0]["reproduced"] = False
                elif mutation == "citation": review["evidence_steps"] = [999]
                elif mutation == "sha": review["recording_sha256"] = "f" * 64
                else: review["candidate_id"] = "f" * 24
                result = analyze_fixed_suite(rows, reviews)
                self.assertEqual(result["status"], "inconclusive")
                self.assertTrue(result["validation_errors"])

    def test_incomplete_usage_execution_or_replay_never_passes(self):
        for field, value in (("completed", False), ("usage_known", False),
                             ("native_replay", {"verified": False}), ("strict_replay", {"verified": False}),
                             ("unassessed_final_outcome", {"step": 1}), ("status", "inference_error")):
            with self.subTest(field=field):
                rows, reviews = outcomes(6)
                rows[0][field] = value
                self.assertEqual(analyze_fixed_suite(rows, reviews)["status"], "inconclusive")

    def test_acknowledged_contract_failure_requires_the_same_independent_truth_and_replay(self):
        rows, reviews = outcomes(6)
        phase(rows, "u01h1", "eval3")["status"] = "contract_failure"
        self.assertEqual(analyze_fixed_suite(rows, reviews)["status"], "supported_fixed_suite_effect")
        phase(rows, "u01h1", "eval3")["native_replay"]["verified"] = False
        self.assertEqual(analyze_fixed_suite(rows, reviews)["status"], "inconclusive")

    def test_no_recording_reuse_and_no_bool_manifest_or_budget_overflow(self):
        for field, value in (("history_number", True), ("steps", 25), ("inference_requests", 29),
                             ("recording_sha256", "reused")):
            with self.subTest(field=field):
                rows, reviews = fixture()
                rows[1][field] = rows[0]["recording_sha256"] if value == "reused" else value
                self.assertEqual(analyze_fixed_suite(rows, reviews)["status"], "inconclusive")

    def test_malformed_evidence_is_inconclusive_without_crashing(self):
        for field, value in (("candidates", None), ("candidates", [None]), ("block_id", []), ("recording_sha256", [])):
            with self.subTest(field=field):
                rows, reviews = fixture()
                rows[0][field] = value
                self.assertEqual(analyze_fixed_suite(rows, reviews)["status"], "inconclusive")
        rows, reviews = outcomes(6)
        reviews["candidate_reviews"][0]["candidate_id"] = []
        self.assertEqual(analyze_fixed_suite(rows, reviews)["status"], "inconclusive")

    def test_seen_unknown_and_explicitly_not_seen_are_not_operator_new_discoveries(self):
        rows, reviews = outcomes(6)
        add_report(phase(rows, "u01h1", "acquire-1"), reviews)
        for number in (1, 2, 3):
            row = phase(rows, "u01h2", f"acquire-{number}")
            reviews["exposure_reviews"].append({"recording_sha256": row["recording_sha256"],
                                                 "matched_family": row["family_id"], "observed": False,
                                                 "evidence_steps": [], "source_basis": "All original acquisition transitions independently inspected."})
        result = analyze_fixed_suite(rows, reviews)
        labels = {(r["block_id"], r["arm"]): r for r in result["acquisition_exposure"]}
        self.assertEqual(labels[("u01h1", "eval1")]["prior_target_exposure"], "confirmed_seen")
        self.assertEqual(labels[("u01h2", "eval3")]["prior_target_exposure"], "not_seen_in_reviewed_acquisition")
        self.assertEqual(labels[("u02h1", "eval3")]["prior_target_exposure"], "unknown")
        self.assertIn("never an operator-new", labels[("u01h2", "eval3")]["interpretation"])

    def test_source_exposure_cannot_deny_a_confirmed_acquisition_report(self):
        rows, reviews = fixture()
        row = phase(rows, "u01h1", "acquire-1")
        add_report(row, reviews)
        reviews["exposure_reviews"] = [{"recording_sha256": row["recording_sha256"], "matched_family": row["family_id"],
                                         "observed": False, "evidence_steps": [], "source_basis": "Conflicting fixture judgment"}]
        result = analyze_fixed_suite(rows, reviews)
        self.assertEqual(result["status"], "inconclusive")
        self.assertIn("acquisition_exposure_review_conflicts_with_supported_report", result["validation_errors"])

    def test_functions_never_dispatch_http(self):
        rows, reviews = outcomes(6)
        with patch("urllib.request.OpenerDirector.open", side_effect=AssertionError("No network")):
            self.assertEqual(analyze_fixed_suite(rows, reviews)["status"], "supported_fixed_suite_effect")


class BlindReviewBundleTests(unittest.TestCase):
    def bundle_fixture(self):
        before = {"game": "public-fixture", "player": {"position": [0, 1, 0]}, "actions": ["east"],
                  "events": [], "evaluation": {"defect": "PRIVATE_LABEL"}, "_capabilities": {"secret": "PRIVATE_KEY"}}
        after = deepcopy(before)
        after["player"]["position"] = [1, 1, 0]
        records = [{"type": "session", "initial_observation": before, "evaluation": {"profile": "floor-gap"}},
                   {"type": "step", "step": 1, "decision": {"action": "east", "reason": "eval3 history u01h1 PRIVATE_KEY"},
                    "before": before, "after": after, "evaluation": {"truth": "PRIVATE_LABEL"}}]
        original = ("\n".join(json.dumps(r) for r in records) + "\n").encode()
        digest = hashlib.sha256(original).hexdigest()
        row = {"block_id": "u01h1", "case_id": "u01", "arm": "eval3", "slot": "eval3", "family_id": "floor-gap",
               "recording_sha256": digest, "steps": 1,
               "candidates": [{"number": 1, "candidate_id": candidate_id(digest, 1), "type": "sank_into_visible_floor",
                               "source": "software_anomaly", "evidence_steps": [1], "reproduced": True}]}
        return row, original

    def test_public_packet_is_blind_and_private_map_retains_binding(self):
        row, original = self.bundle_fixture()
        saved = deepcopy(row)
        bundle = make_review_bundle([row], {row["recording_sha256"]: original}, secrets=("PRIVATE_KEY",))
        public = json.dumps(bundle["public"])
        for value in ("u01h1", '"eval3"', "floor-gap", "PRIVATE_KEY", "PRIVATE_LABEL", '"evaluation"', '"reason"', '"_capabilities"'):
            self.assertNotIn(value, public)
        case = bundle["public"]["cases"][0]
        self.assertEqual(case["actual_control_prefix"], [{"step": 1, "kind": "step", "action": "east"}])
        self.assertEqual(case["citations"], [{"recording_sha256": row["recording_sha256"], "step": 1}])
        self.assertEqual(bundle["private"]["mapping"][0]["arm"], "eval3")
        self.assertEqual(bundle["private"]["review_template"]["candidate_reviews"][0]["verdict"], "unreviewed")
        self.assertEqual(row, saved)

    def test_wrong_bytes_fabricated_step_duplicate_or_parsed_input_rejected(self):
        for mutation in ("bytes", "step", "duplicate", "parsed"):
            with self.subTest(mutation=mutation):
                row, original = self.bundle_fixture()
                rows = [row]
                mapping = {row["recording_sha256"]: original}
                if mutation == "bytes": mapping[row["recording_sha256"]] = original + b" "
                elif mutation == "step": row["candidates"][0]["evidence_steps"] = [2]
                elif mutation == "duplicate": rows.append(deepcopy(row))
                else: mapping[row["recording_sha256"]] = [json.loads(line) for line in original.splitlines()]
                with self.assertRaises(ValueError):
                    make_review_bundle(rows, mapping)

    def test_malformed_native_step_is_rejected_before_indexing(self):
        for malformed in (None, True, "1", 0):
            with self.subTest(step=malformed):
                row, original = self.bundle_fixture()
                records = [json.loads(line) for line in original.splitlines()]
                if malformed is None:
                    del records[1]["step"]
                else:
                    records[1]["step"] = malformed
                original = ("\n".join(json.dumps(record) for record in records) + "\n").encode()
                digest = hashlib.sha256(original).hexdigest()
                row["recording_sha256"] = digest
                row["candidates"][0]["candidate_id"] = candidate_id(digest, 1)
                with self.assertRaises(ValueError):
                    make_review_bundle([row], {digest: original})


if __name__ == "__main__":
    unittest.main()
