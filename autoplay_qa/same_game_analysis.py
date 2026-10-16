"""Pure, offline review packets and analysis of the predeclared V4 suite.

This module never runs a game, calls a model, reads a file, or writes a ledger.
Review judgments are trusted independent inputs, not truths inferred from a
detector's type or recurrence. Source files and actual native replay remain the
reviewer's and harness's responsibility.
"""

from __future__ import annotations

from collections import Counter
from copy import deepcopy
import hashlib
import json
import math

from .experience import public_observation


REVIEW_SCHEMA = "same-game-independent-review-v1"
ANALYSIS_SCHEMA = "same-game-fixed-suite-analysis-v1"
BUNDLE_SCHEMA = "same-game-blind-review-v1"
ARMS = ("eval0", "eval1", "eval3")
SLOTS = ("acquire-1", "acquire-2", "acquire-3", *ARMS)
CASE_MANIFEST = (
    ("u01", "floor-gap", "positive"),
    ("u02", "wall-tunnel", "positive"),
    ("u03", "door-ghost-collider", "positive"),
    ("u04", "respawn-leak", "positive"),
    ("u05", None, "clean"),
    ("u06", None, "clean"),
)
_REPORT_TYPES = {
    "cross_episode_nondeterminism", "reset_inconsistency",
    "map_object_inconsistency", "label_contradiction", "passed_through_solid",
    "blocked_no_visible_obstacle", "fell_through_visible_floor",
    "sank_into_visible_floor", "player_stuck_in_geometry", "suspected_bug",
}
_VERDICTS = {"supported_defect", "expected_behavior", "unsupported", "unreviewed"}


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)


def _hash(value):
    return hashlib.sha256(_json(value).encode()).hexdigest()


def _hex(value, length):
    return isinstance(value, str) and len(value) == length and all(c in "0123456789abcdef" for c in value)


def candidate_id(recording_sha256, number):
    if not _hex(recording_sha256, 64) or type(number) is not int or number < 1:
        raise ValueError("Candidate identity requires an original recording hash and positive issue number.")
    return hashlib.sha256(f"{recording_sha256}:{number}".encode()).hexdigest()[:24]


def _steps(value, maximum):
    return (isinstance(value, list) and bool(value)
            and all(type(step) is int and 1 <= step <= maximum for step in value)
            and len(value) == len(set(value)))


def exact_paired_test(warm_only, empty_only):
    """Two-sided exact McNemar/sign test, conditional on discordant pairs."""
    if any(type(n) is not int or n < 0 for n in (warm_only, empty_only)):
        raise ValueError("Discordant-pair counts must be nonnegative integers.")
    n = warm_only + empty_only
    tail = sum(math.comb(n, k) for k in range(min(warm_only, empty_only) + 1))
    probability = min(1.0, 2 * tail / (2 ** n)) if n else 1.0
    return {"method": "two_sided_exact_mcnemar", "alpha": 0.05,
            "warm_only": warm_only, "empty_only": empty_only,
            "discordant_pairs": n, "p_value": probability,
            "reject_equal_direction": probability <= 0.05}


def discordant_win_interval(warm_only, empty_only, confidence=0.95):
    """Exact Clopper-Pearson interval for direction among discordant pairs.

    It is not a confidence interval for absolute detection-rate gain, clean
    false-positive rate, or transfer to other bug families. Its binomial model
    assumes independent discordant outcomes with a common direction probability.
    """
    exact_paired_test(warm_only, empty_only)
    if type(confidence) not in (int, float) or not math.isfinite(confidence) or not 0 < confidence < 1:
        raise ValueError("Confidence must be finite and strictly between zero and one.")
    successes, n = warm_only, warm_only + empty_only
    alpha = 1 - confidence

    def tail(p, start, stop):
        return sum(math.comb(n, k) * p ** k * (1 - p) ** (n - k) for k in range(start, stop))

    def solve(function, increasing):
        low, high = 0.0, 1.0
        for _ in range(70):
            middle = (low + high) / 2
            if (function(middle) < alpha / 2) == increasing:
                low = middle
            else:
                high = middle
        return (low + high) / 2

    lower = solve(lambda p: tail(p, successes, n + 1), True) if successes else 0.0
    upper = solve(lambda p: tail(p, 0, successes + 1), False) if successes < n else 1.0
    return {"method": "clopper_pearson_discordant_direction", "confidence": confidence,
            "estimate": successes / n if n else None, "lower": lower, "upper": upper,
            "scope": "Discordant-pair direction only; binomial assumptions, not absolute accuracy gain or new-family generalization."}


def make_review_bundle(rows, recordings_by_sha256, *, secrets=()):
    """Build neutral candidate packets and a SEPARATE private routing map.

    Values in ``recordings_by_sha256`` are original JSONL bytes (or exact UTF-8
    text), so content hashes can be checked without reading the filesystem.
    Never give the private routing map to the blinded candidate reviewer.
    Physical trajectories can reveal behavior; structural blinding does not
    guarantee that an experienced reviewer cannot guess a condition.
    """
    if not isinstance(rows, list) or not isinstance(recordings_by_sha256, dict):
        raise ValueError("Review bundle requires session rows and original recording bytes.")
    packets, routing, template, seen = [], [], [], set()
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("candidates"), list):
            raise ValueError("Review bundle has an invalid session/candidate row.")
        digest = row.get("recording_sha256")
        if not _hex(digest, 64):
            raise ValueError("Review bundle has an invalid original hash.")
        original = recordings_by_sha256.get(digest)
        if isinstance(original, str):
            original = original.encode("utf-8")
        if not isinstance(original, bytes) or hashlib.sha256(original).hexdigest() != digest:
            raise ValueError("Original recording bytes do not match the declared hash.")
        records = [json.loads(line) for line in original.decode("utf-8").splitlines() if line.strip()]
        if not records or any(not isinstance(item, dict) for item in records) or records[0].get("type") != "session":
            raise ValueError("Original recording has no usable session header.")
        native = [item for item in records if item.get("type") in {"step", "reset"}]
        if any(type(item.get("step")) is not int or item["step"] < 1 for item in native):
            raise ValueError("Original recording has an invalid executed step.")
        controls = {item["step"]: item for item in native}
        if (len(controls) != len(native) or set(controls) != set(range(1, len(controls) + 1))
                or row.get("steps") != len(controls)):
            raise ValueError("Original recording repeats an executed step.")
        for issue in row.get("candidates", []):
            if not isinstance(issue, dict):
                raise ValueError("Review bundle has an invalid candidate.")
            identifier = candidate_id(digest, issue.get("number"))
            steps = issue.get("evidence_steps")
            if identifier != issue.get("candidate_id") or identifier in seen or not _steps(steps, len(controls)):
                raise ValueError("Candidate identity or original evidence steps are invalid.")
            if any(step not in controls for step in steps):
                raise ValueError("Candidate cites an unexecuted original step.")
            seen.add(identifier)
            maximum = max(steps)
            transitions = []
            for step in steps:
                item = controls[step]
                action = public_observation({"player": {"action": item["decision"]["action"]}}, secrets=secrets)["player"]["action"]
                transitions.append({"step": step, "action": action,
                                    "before": public_observation(item["before"], secrets=secrets),
                                    "after": public_observation(item["after"], secrets=secrets)})
            prefix = []
            for step, item in sorted(controls.items()):
                if step <= maximum:
                    action = public_observation({"player": {"action": item["decision"]["action"]}}, secrets=secrets)["player"]["action"]
                    prefix.append({"step": step, "kind": item["type"], "action": action})
            packets.append({"case_id": "review-" + identifier, "candidate_id": identifier,
                            "recording_sha256": digest,
                            "report_type": issue.get("type") if isinstance(issue.get("type"), str) and issue["type"] in _REPORT_TYPES else "other_report",
                            "citations": [{"recording_sha256": digest, "step": step} for step in steps],
                            "initial_observation": public_observation(records[0]["initial_observation"], secrets=secrets),
                            "actual_control_prefix": prefix, "evidence_transitions": transitions,
                            "report_prose_withheld": True,
                            "review_limit": "Inspect native replay and local source. Type/recurrence alone is not bug truth; request a separately masked claim if the public evidence does not specify the report."})
            routing.append({"candidate_id": identifier, "recording_sha256": digest,
                            "block_id": row.get("block_id"), "case_id": row.get("case_id"),
                            "slot": row.get("slot"), "arm": row.get("arm"), "family_id": row.get("family_id"),
                            "number": issue["number"], "reproduced": issue.get("reproduced")})
            template.append({"candidate_id": identifier, "recording_sha256": digest,
                             "verdict": "unreviewed", "matched_family": None,
                             "evidence_steps": list(steps), "source_basis": "", "reproduction_verified": False})
    packets.sort(key=lambda item: item["candidate_id"])
    routing.sort(key=lambda item: item["candidate_id"])
    template.sort(key=lambda item: item["candidate_id"])
    public = {"schema": BUNDLE_SCHEMA, "cases": packets,
              "scope": "Original public transitions and hashes; arm/history/profile labels, prompts, notes, memory and model prose withheld. No inference or automatic truth judgment."}
    return {"public": public,
            "private": {"mapping": routing, "public_bundle_sha256": _hash(public),
                        "review_template": {"schema": REVIEW_SCHEMA, "reviewer": "", "candidate_reviews": template, "exposure_reviews": []}}}


def analyze_fixed_suite(rows, review_document):
    """Score the full 72-session manifest; incomplete evidence stays inconclusive.

    Input flags/hashes are harness receipts, not independently executed here.
    The harness must additionally verify build/policy freeze, scoped history
    isolation, actual empty/frozen memory delivery and unchanged evaluation DBs.
    """
    errors, indexed, candidates, hashes = [], {}, {}, set()
    expected = {(f"{case}h{history}", slot): (case, family, stratum, history)
                for case, family, stratum in CASE_MANIFEST for history in (1, 2) for slot in SLOTS}
    if not isinstance(rows, list):
        rows = []
        errors.append("session_rows_not_a_list")
    for row in rows:
        if not isinstance(row, dict):
            errors.append("malformed_session_row")
            continue
        key = (row.get("block_id"), row.get("slot"))
        if not all(isinstance(part, str) for part in key) or key not in expected or key in indexed:
            errors.append("unknown_or_duplicate_phase")
            continue
        indexed[key] = row
        case, family, stratum, history = expected[key]
        if (type(row.get("history_number")) is not int or row.get("game") != "unity"
                or (row.get("case_id"), row.get("family_id"), row.get("stratum"), row.get("history_number")) != (case, family, stratum, history)):
            errors.append("phase_manifest_mismatch")
        if (row.get("arm") != (row["slot"] if row["slot"] in ARMS else None)
                or row.get("acquisition") is not (row["slot"] not in ARMS)):
            errors.append("phase_arm_mismatch")
        if (row.get("completed") is not True or row.get("usage_known") is not True
                or row.get("status") not in ("finished", "action_limit", "request_limit", "time_limit", "contract_failure")
                or "unassessed_final_outcome" not in row
                or row.get("unassessed_final_outcome") is not None
                or not isinstance(row.get("native_replay"), dict) or row["native_replay"].get("verified") is not True
                or not isinstance(row.get("strict_replay"), dict) or row["strict_replay"].get("verified") is not True):
            errors.append("incomplete_usage_execution_or_replay")
        if (not _hex(row.get("recording_sha256"), 64) or type(row.get("steps")) is not int or not 0 <= row["steps"] <= 24
                or type(row.get("resets")) is not int or not 0 <= row["resets"] <= row.get("steps", -1)
                or type(row.get("inference_requests")) is not int or not 0 <= row["inference_requests"] <= 28
                or not isinstance(row.get("candidates"), list)):
            errors.append("invalid_session_evidence_shape")
            continue
        if row["recording_sha256"] in hashes:
            errors.append("recording_reused_across_phases")
        hashes.add(row["recording_sha256"])
        for issue in row["candidates"]:
            if not isinstance(issue, dict):
                errors.append("invalid_candidate")
                continue
            try:
                identifier = candidate_id(row["recording_sha256"], issue.get("number"))
            except ValueError:
                errors.append("invalid_candidate_identity")
                continue
            if (issue.get("candidate_id") != identifier or identifier in candidates
                    or not _steps(issue.get("evidence_steps"), row["steps"])
                    or type(issue.get("reproduced")) is not bool):
                errors.append("invalid_candidate_binding")
                continue
            candidates[identifier] = (row, issue)
    if set(indexed) != set(expected):
        errors.append("incomplete_predeclared_matrix")
    reviews, exposure = {}, {}
    if (not isinstance(review_document, dict) or review_document.get("schema") != REVIEW_SCHEMA
            or not isinstance(review_document.get("reviewer"), str) or not review_document["reviewer"].strip()
            or not isinstance(review_document.get("candidate_reviews"), list)):
        errors.append("independent_review_missing_or_malformed")
        review_document = {"candidate_reviews": [], "exposure_reviews": []}
    for review in review_document["candidate_reviews"]:
        if (not isinstance(review, dict) or not _hex(review.get("candidate_id"), 24)
                or review["candidate_id"] not in candidates or review["candidate_id"] in reviews):
            errors.append("unknown_or_duplicate_candidate_review")
            continue
        identifier = review["candidate_id"]
        row, issue = candidates[identifier]
        if (review.get("recording_sha256") != row["recording_sha256"] or not isinstance(review.get("verdict"), str) or review["verdict"] not in _VERDICTS
                or not _steps(review.get("evidence_steps"), row["steps"])
                or not set(review.get("evidence_steps", [])) & set(issue["evidence_steps"])
                or type(review.get("reproduction_verified")) is not bool):
            errors.append("candidate_review_binding_invalid")
            continue
        if review["verdict"] == "unreviewed" or not isinstance(review.get("source_basis"), str) or not review["source_basis"].strip():
            errors.append("candidate_truth_unreviewed")
            continue
        family = review.get("matched_family")
        if family is not None and (not isinstance(family, str) or family not in {item[1] for item in CASE_MANIFEST if item[1] is not None}):
            errors.append("candidate_family_unknown")
            continue
        if review["verdict"] != "supported_defect" and family is not None:
            errors.append("unsupported_candidate_cannot_match_target")
            continue
        if review["verdict"] == "supported_defect" and (issue["reproduced"] is not True or review["reproduction_verified"] is not True):
            errors.append("supported_claim_lacks_independent_reproduction")
            continue
        reviews[identifier] = review
    if set(reviews) != set(candidates):
        errors.append("candidate_reviews_incomplete")
    # Exposure reviews are optional. Absence never establishes that a target
    # was unseen. Confirmed acquisition reports suffice to establish seen.
    exposure_rows = review_document.get("exposure_reviews", [])
    if not isinstance(exposure_rows, list):
        errors.append("exposure_reviews_malformed")
        exposure_rows = []
    acquisitions = {row.get("recording_sha256"): row for row in indexed.values()
                    if str(row.get("slot", "")).startswith("acquire-") and _hex(row.get("recording_sha256"), 64)}
    for item in exposure_rows:
        if (not isinstance(item, dict) or not _hex(item.get("recording_sha256"), 64)
                or item["recording_sha256"] not in acquisitions or item["recording_sha256"] in exposure):
            errors.append("exposure_review_binding_invalid")
            continue
        row = acquisitions[item["recording_sha256"]]
        if (row.get("stratum") != "positive" or item.get("matched_family") != row.get("family_id")
                or type(item.get("observed")) is not bool or not isinstance(item.get("source_basis"), str) or not item["source_basis"].strip()
                or (item["observed"] and not _steps(item.get("evidence_steps"), row["steps"]))
                or (not item["observed"] and item.get("evidence_steps") != [])):
            errors.append("exposure_review_invalid")
            continue
        exposure[item["recording_sha256"]] = item

    def scored(row):
        issues = row.get("candidates") if isinstance(row.get("candidates"), list) else []
        judged = [reviews[issue["candidate_id"]] for issue in issues
                  if isinstance(issue, dict) and isinstance(issue.get("candidate_id"), str) and issue["candidate_id"] in reviews]
        supported = [item for item in judged if item["verdict"] == "supported_defect"]
        return {"target_hit": bool(row.get("stratum") == "positive" and any(item.get("matched_family") == row.get("family_id") for item in supported)),
                "false_reports": sum(item["verdict"] in {"unsupported", "expected_behavior"} for item in judged),
                "supported_other_reports": sum(item.get("matched_family") != row.get("family_id") for item in supported),
                "supported_reports": len(supported), "candidate_count": len(issues)}

    phases = {key: scored(row) for key, row in indexed.items()}
    paired, exposure_labels, family_results = [], [], []
    false_increases, clean_supported = [], []
    for case, family, stratum in CASE_MANIFEST:
        family_pairs = []
        for history in (1, 2):
            block = f"{case}h{history}"
            if not all((block, arm) in phases for arm in ARMS):
                continue
            scores = {arm: phases[(block, arm)] for arm in ARMS}
            difference = int(scores["eval3"]["target_hit"]) - int(scores["eval0"]["target_hit"])
            pair = {"block_id": block, "case_id": case, "family_id": family, "stratum": stratum,
                    "arms": scores, "primary_difference": difference if stratum == "positive" else None,
                    "false_report_change_eval3": scores["eval3"]["false_reports"] - scores["eval0"]["false_reports"],
                    "false_report_change_eval1": scores["eval1"]["false_reports"] - scores["eval0"]["false_reports"]}
            paired.append(pair)
            if stratum == "positive":
                family_pairs.append(pair)
            for arm in ("eval1", "eval3"):
                if scores[arm]["false_reports"] > scores["eval0"]["false_reports"]:
                    false_increases.append({"block_id": block, "arm": arm, "additional_false_reports": scores[arm]["false_reports"] - scores["eval0"]["false_reports"]})
            if stratum == "clean" and any(scores[arm]["supported_reports"] for arm in ARMS):
                clean_supported.append(block)
                errors.append("authored_clean_control_contains_supported_defect")
            if stratum == "positive":
                for dose in (1, 3):
                    training = [indexed.get((block, f"acquire-{number}")) for number in range(1, dose + 1)]
                    seen = any(row is not None and phases[(block, row["slot"])]["target_hit"] for row in training)
                    explicit = [exposure.get(row.get("recording_sha256")) if _hex(row.get("recording_sha256"), 64) else None
                                for row in training if row is not None]
                    if any(item is not None and item["observed"] for item in explicit):
                        seen = True
                    known_absent = len(explicit) == dose and all(item is not None and not item["observed"] for item in explicit)
                    if seen and known_absent:
                        errors.append("acquisition_exposure_review_conflicts_with_supported_report")
                    exposure_labels.append({"block_id": block, "arm": f"eval{dose}",
                                            "prior_target_exposure": "confirmed_seen" if seen else "not_seen_in_reviewed_acquisition" if known_absent else "unknown",
                                            "evaluation_target_hit": scores[f"eval{dose}"]["target_hit"],
                                            "interpretation": "Previously seen target hits are retests. Not seen in this history is still a known benchmark target, never an operator-new discovery."})
        if stratum == "positive":
            family_results.append({"family_id": family, "paired_blocks": len(family_pairs),
                                   "eval0_hits": sum(p["arms"]["eval0"]["target_hit"] for p in family_pairs),
                                   "eval1_hits": sum(p["arms"]["eval1"]["target_hit"] for p in family_pairs),
                                   "eval3_hits": sum(p["arms"]["eval3"]["target_hit"] for p in family_pairs),
                                   "net_primary_gain": sum(p["primary_difference"] for p in family_pairs)})
    positives = [pair for pair in paired if pair["stratum"] == "positive"]
    wins = sum(pair["primary_difference"] == 1 for pair in positives)
    losses = sum(pair["primary_difference"] == -1 for pair in positives)
    test = exact_paired_test(wins, losses)
    hits = {arm: sum(pair["arms"][arm]["target_hit"] for pair in positives) for arm in ARMS}
    false_reports = {arm: sum(pair["arms"][arm]["false_reports"] for pair in paired) for arm in ARMS}
    try:
        review_hash = _hash(review_document)
    except (TypeError, ValueError, OverflowError, RecursionError):
        review_hash = None
        errors.append("review_document_not_json")
    primary_false_increase = any(item["arm"] == "eval3" for item in false_increases)
    complete = not errors
    supported = complete and wins > losses and test["reject_equal_direction"] and not primary_false_increase
    test["inference_valid_for_predeclared_matrix"] = complete
    status = "inconclusive" if errors else "supported_fixed_suite_effect" if supported else "no_supported_fixed_suite_effect"
    return {"schema": ANALYSIS_SCHEMA, "status": status,
            "validation_errors": sorted(set(errors)), "formal_phases_supplied": len(rows), "formal_phases_expected": 72,
            "independent_candidate_reviews": len(reviews), "original_candidates": len(candidates),
            "primary": {"comparison": "eval3_minus_eval0", "positive_pairs": len(positives), "expected_positive_pairs": 8,
                        "hits": hits, "net_target_hits": wins - losses,
                        "paired_detection_rate_difference": (wins - losses) / len(positives) if positives else None,
                        "test": test, "discordant_direction_interval": discordant_win_interval(wins, losses),
                        "no_added_unsupported_reports": not primary_false_increase,
                        "statistical_interpretation": "Exact paired direction analysis conditional on this frozen suite and acquired histories. Four families are not eight independent game types; no population or causal-mechanism generalization."},
            "secondary": {"eval1_trend": {"hits": hits, "observed_nondecreasing": hits["eval0"] <= hits["eval1"] <= hits["eval3"],
                                           "meaning": "Secondary descriptive dose trend; no monotonic-learning guarantee and no replacement primary test."},
                          "engineering_net_at_least_three": complete and wins - losses >= 3 and not primary_false_increase,
                          "engineering_signal_is_not_statistical_support": True},
            "false_reports": {"all_evaluation_arms": false_reports, "paired_increases": false_increases,
                              "per_arm_report_precision": {arm: (sum(p["arms"][arm]["supported_reports"] for p in paired)
                                  / sum(p["arms"][arm]["candidate_count"] for p in paired))
                                  if complete and sum(p["arms"][arm]["candidate_count"] for p in paired) else None for arm in ARMS},
                              "expected_clean_pairs": 4, "clean_pairs_observed": sum(pair["stratum"] == "clean" for pair in paired),
                              "clean_controls_with_supported_defect": clean_supported,
                              "meaning": "Unsupported/expected candidate reports are counted on positives and cleans; no candidate is not 100% precision. Added means a larger unsupported-report count in the same paired block."},
            "paired_blocks": paired, "family_results": family_results,
            "acquisition_exposure": exposure_labels,
            "regressions": [pair["block_id"] for pair in positives if pair["primary_difference"] < 0],
            "eval1_regressions": [pair["block_id"] for pair in positives if pair["arms"]["eval0"]["target_hit"] and not pair["arms"]["eval1"]["target_hit"]],
            "review_document_canonical_sha256": review_hash,
            "limitations": ["Independent source judgment and native reproduction are supplied evidence, not proved by detector recurrence or this pure scorer.",
                            "Harness must verify frozen policy/build, empty/frozen memory delivery, read-only evaluation hashes, scope isolation and no test-to-training leakage.",
                            "Two histories per case and four clean pairs cannot establish reliable population accuracy, false-positive noninferiority, general transfer or universal self-improvement.",
                            "No optional stopping, resampling, favorable-subset substitution or claim that a known-family retest is a new discovery.",
                            "A binomial direction interval is not an absolute detection-rate-gain interval; family clustering limits interpretation."]}
