"""Public, data-only dashboard views of local QA recordings.

This module never opens a game or makes an inference request. Raw recordings,
private harness metadata and prompts remain on the runner's machine.
"""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import re
from typing import Sequence

from .config import DEFAULT_MODEL
from .reporting import _actions_through, extract_issues, load_recording
from .triage import DEFAULT_TRIAGE_MODEL


VERSION = "dashboard-v1"
LIMITS = {
    "sweepControls": 400, "modelControls": 24, "explorationRequests": 28,
    "triageRequests": 8, "explorationSeconds": 300, "triageSeconds": 300,
    "maxTokens": 8192, "timeout": 45,
}
_PRIVATE = frozenset({
    "evaluation", "injecteddefect", "groundtruth", "defect", "defectinjection",
    "rules", "rule", "developerrules", "developerrule", "privaterules", "privaterule",
    "expectedverdicts", "expectedverdict", "label", "labels", "caseid", "kindofcase",
    "apikey", "keysecret", "token", "accesstoken", "authorization", "bearer",
    "credentials", "password", "secret", "env", "environmentvariables",
    "endpoint", "baseurl", "unityplayer", "playerpath", "sourcepath", "sourcerecording",
    "messages", "prompt", "rawprompt", "systemprompt", "responseschema",
    "thinking", "reasoningcontent", "privatereasoning", "rawresponse", "completion",
    "strategy", "victorystrategy", "solution", "route", "suppliedroute", "forcedresets", "passabilityhints",
})
_LOCAL_PATH = re.compile(r"(?:/(?:Users|home|private|tmp|var|Volumes|Applications)/[^\s\"'<>]+|[A-Za-z]:\\[^\s\"'<>]+)")
_TOKEN = re.compile(r"\b(?:sk-[A-Za-z0-9_-]{12,}|Bearer\s+[A-Za-z0-9._-]{8,})", re.IGNORECASE)
_KEY_NOTE = (
    "Human review required: key-use feedback and recorded inventory disagree. "
    "The AI report's initial-inventory and complete-playthrough assertions are "
    "not established by its cited transitions. Its verdict remains tentative."
)


def safe_text(value: object, secrets: tuple[str, ...] = ()) -> str:
    text = value if isinstance(value, str) else ""
    for secret in secrets:
        if secret:
            text = text.replace(secret, "[REDACTED]")
    text = _LOCAL_PATH.sub("[local path]", text)
    text = _TOKEN.sub("[REDACTED]", text)
    return "".join(c for c in text if c.isprintable() or c in "\n\t")[:16000]


def public_value(value: object, *, secrets: tuple[str, ...] = (), _depth: int = 0) -> object:
    """Sanitize public evidence recursively, including arbitrary object states."""
    if _depth > 12:
        return None
    if value is None or type(value) is bool or type(value) is int:
        return value
    if type(value) is float:
        return value if math.isfinite(value) else None
    if isinstance(value, str):
        return safe_text(value, secrets)
    if isinstance(value, (list, tuple)):
        return [public_value(item, secrets=secrets, _depth=_depth + 1) for item in value[:4096]]
    if isinstance(value, dict):
        result = {}
        for key, item in value.items():
            if not isinstance(key, str):
                continue
            normalized = re.sub(r"[^a-z0-9]", "", key.lower())
            if normalized in _PRIVATE or normalized.startswith(("private", "secret", "credential", "developer")):
                continue
            result[safe_text(key, secrets)] = public_value(item, secrets=secrets, _depth=_depth + 1)
        return result
    return None


def _vector(value: object) -> list | None:
    if isinstance(value, (list, tuple)) and 2 <= len(value) <= 3 and all(
        type(n) in (int, float) and math.isfinite(n) for n in value
    ):
        return list(value)
    return None


def _static(entity: dict) -> bool:
    kind = str(entity.get("kind", "")).lower()
    return entity.get("state", {}).get("solid") is True and any(
        part in kind for part in ("wall", "floor", "terrain", "ground", "boundary")
    )


def public_observation(value: object, *, compact: bool = False,
                       secrets: tuple[str, ...] = (), baseline: dict | None = None) -> dict:
    if not isinstance(value, dict):
        return {}
    result = {}
    for key in ("game", "status"):
        if isinstance(value.get(key), str):
            result[key] = safe_text(value[key], secrets)
    if type(value.get("tick")) is int:
        result["tick"] = value["tick"]
    player = value.get("player")
    if isinstance(player, dict):
        out = {}
        for key in ("position", "velocity"):
            vector = _vector(player.get(key))
            if vector is not None:
                out[key] = vector
        if isinstance(player.get("inventory"), list):
            out["inventory"] = public_value(player["inventory"], secrets=secrets)
        if type(player.get("grounded")) is bool:
            out["grounded"] = player["grounded"]
        result["player"] = out
    if not compact:
        rows = value.get("map", value.get("grid"))
        if isinstance(rows, list):
            result["map"] = [safe_text(row, secrets) if isinstance(row, str) else "".join(
                str(c) for c in row if isinstance(c, str)
            ) for row in rows if isinstance(row, (str, list))]
        if isinstance(value.get("legend"), dict):
            result["legend"] = public_value(value["legend"], secrets=secrets)
    initial_entities = {e.get("id"): e for e in (baseline or {}).get("entities", []) if isinstance(e, dict)}
    entities = []
    static_ids = set()
    for entity in value.get("entities", []) if isinstance(value.get("entities"), list) else []:
        if not isinstance(entity, dict) or not isinstance(entity.get("id"), str):
            continue
        if _static(entity):
            static_ids.add(entity["id"])
            if compact:
                continue
        position = _vector(entity.get("position"))
        # Collected objects can omit their position. Preserve their recorded
        # dynamic state and use the initial visible position only for drawing.
        if position is None:
            position = _vector(initial_entities.get(entity["id"], {}).get("position"))
        if position is None:
            continue
        clean = {"id": safe_text(entity["id"], secrets), "kind": safe_text(entity.get("kind", "object"), secrets),
                 "position": position, "state": public_value(entity.get("state", {}), secrets=secrets)}
        size = _vector(entity.get("size"))
        if size is not None:
            clean["size"] = size
        entities.append(clean)
    if entities or isinstance(value.get("entities"), list):
        result["entities"] = entities
    objects = value.get("objects")
    if isinstance(objects, dict):
        # Unity's objects repeat the entity geometry; dynamic entities retain
        # the same state. Grid object facets are small and remain inspectable.
        represented = {e.get("id"): e for e in value.get("entities", []) if isinstance(e, dict)}
        objects = {
            key: item for key, item in objects.items()
            if not isinstance(item, dict) or key not in represented or not all(
                (represented[key].get(field) == entry if field in {"id", "kind", "position", "size", "state"}
                 else represented[key].get("state", {}).get(field) == entry and field in represented[key].get("state", {}))
                for field, entry in item.items()
            )
        }
        result["objects"] = public_value(objects, secrets=secrets)
    if isinstance(value.get("events"), list):
        result["events"] = [safe_text(item, secrets) for item in value["events"] if isinstance(item, str)]
    return result


def _elapsed(record: dict) -> float:
    value = record.get("elapsed_ms", 0)
    return round(max(0, value / 1000), 3) if type(value) in (int, float) and math.isfinite(value) else 0.0


def _integers(value: object) -> list[int]:
    return [n for n in value if type(n) is int] if isinstance(value, list) else []


def _is_key_issue(issue: dict) -> bool:
    text = " ".join(str(issue.get(k, "")) for k in ("title", "type", "claim")).lower()
    return "key" in text and any(word in text for word in ("consum", "use", "inventory", "remain"))


def _issue_dto(issue: dict, result: dict | None, run_id: str, secrets: tuple[str, ...]) -> dict:
    number = issue.get("number", 1)
    packet = (result or {}).get("evidence_packet", {})
    reproduction = packet.get("reproduction", {})
    assessment = (result or {}).get("assessment")
    assessment = assessment if isinstance(assessment, dict) else {}
    review_status = (result or {}).get("status", "pending")
    if review_status not in {"reviewed", "pending", "failed"}:
        review_status = "failed"
    review = {
        "status": review_status,
        "evidenceSteps": _integers(assessment.get("evidence_steps")),
        "limitations": [safe_text(s, secrets) for s in assessment.get("limitations", []) if isinstance(s, str)],
        "warnings": [safe_text(w.get("message", ""), secrets) for w in (result or {}).get("warnings", []) if isinstance(w, dict)],
    }
    for old, new in (("verdict", "verdict"), ("severity", "severity"), ("title", "title"),
                     ("player_impact", "playerImpact"), ("rationale", "rationale")):
        if isinstance(assessment.get(old), str):
            review[new] = safe_text(assessment[old], secrets)
    key_issue = _is_key_issue(issue)
    human = key_issue or review_status != "reviewed" or assessment.get("verdict") != "bug"
    out = {
        "id": f"{run_id}-issue-{number}", "number": number,
        "title": safe_text(issue.get("title", "Recorded candidate"), secrets),
        "source": safe_text(issue.get("source", "software_anomaly"), secrets),
        "anomalyType": safe_text(issue.get("type", "unknown"), secrets),
        "evidenceSteps": _integers(issue.get("evidence_steps")),
        "expected": public_value(issue.get("expected"), secrets=secrets),
        "actual": public_value(issue.get("actual"), secrets=secrets),
        "reproduction": {
            "actions": [safe_text(a, secrets) for a in reproduction.get("actions", []) if isinstance(a, str)],
            "sourceSteps": _integers(reproduction.get("source_steps")),
            "verified": reproduction.get("verified") if type(reproduction.get("verified")) is bool else None,
            "minimal": reproduction.get("minimization_complete") is True and reproduction.get("minimization_strategy") not in {"not_requested", None},
        },
        "assessment": review, "needsHumanReview": human,
    }
    if key_issue:
        out["reviewNote"] = _KEY_NOTE
    elif human:
        out["reviewNote"] = "The original candidate is retained beside the tentative AI judgment for human review."
    return out


def snapshot_from_records(records: Sequence[dict], *, run_id: str, name: str | None = None,
                          source: str = "recorded", status: str = "completed", phase: str = "complete",
                          triage_result: dict | None = None, budget_usd: float | None = None,
                          cost: dict | None = None, created_at: str | None = None,
                          elapsed_seconds: float | None = None, secrets: tuple[str, ...] = (),
                          candidate_issues: Sequence[dict] | None = None, notes: Sequence[str] = ()) -> dict:
    """Construct a RunSnapshot from executed evidence, preserving step identity."""
    rows = list(records)
    header = next((row for row in rows if row.get("type") == "session"), {})
    initial = header.get("initial_observation", {})
    scene = public_observation(initial, secrets=secrets) or None
    game = "unity" if isinstance(initial, dict) and (initial.get("entities") or "physics" in str(initial.get("game", ""))) else "grid"
    frames = []
    timeline = []
    timeline.append({"id": "phase-sweep", "kind": "phase", "phase": "sweep", "label": "Public collision sweep", "elapsedSeconds": 0.0})
    for row in rows:
        kind = row.get("type")
        if kind in {"step", "reset", "sweep_step", "sweep_reset"} and type(row.get("step")) is int:
            decision = row.get("decision", {})
            action = row.get("action") or decision.get("action") or ("reset" if "reset" in kind else "wait")
            sweep = kind.startswith("sweep") or row.get("source") == "sweep"
            frame = {
                "id": f"{run_id}-{'sweep' if sweep else 'llm'}-{row['step']}", "step": row["step"],
                "phaseStep": row.get("phase_step", row["step"]), "phase": "sweep" if sweep else "llm",
                "action": safe_text(action, secrets), "reason": safe_text(decision.get("reason", row.get("reason", "Public coverage control" if sweep else "")), secrets),
                "before": public_observation(row.get("before"), compact=True, secrets=secrets, baseline=scene),
                "after": public_observation(row.get("after"), compact=True, secrets=secrets, baseline=scene),
                "feedback": [safe_text(s, secrets) for s in row.get("after", {}).get("events", []) if isinstance(s, str)],
                "elapsedSeconds": _elapsed(row),
            }
            frames.append(frame)
            timeline.append({"id": frame["id"], "kind": "action", "phase": "sweep" if sweep else "explore",
                             "label": f"{action} · {'sweep' if sweep else 'Lightning'} step {frame['phaseStep']}",
                             "elapsedSeconds": _elapsed(row)})
        elif kind in {"request", "triage_request"}:
            triage = kind == "triage_request"
            timeline.append({"id": f"{kind}-{row.get('request_number', len(timeline))}", "kind": "request",
                             "phase": "triage" if triage else "explore", "label": f"{'Ultra' if triage else 'Lightning'} request {row.get('request_number', '?')}",
                             "elapsedSeconds": _elapsed(row)})
        elif kind in {"decision_warning", "rejected_decision", "triage_error", "dashboard_warning"}:
            label = row.get("message", "Review request failed" if kind == "triage_error" else "A recorded decision needs review")
            timeline.append({"id": f"warning-{len(timeline)}", "kind": "warning", "phase": "triage" if kind == "triage_error" else "explore",
                             "label": safe_text(label, secrets), "elapsedSeconds": _elapsed(row)})
    issues = list(candidate_issues) if candidate_issues is not None else extract_issues(rows) if header else []
    assessments = {row.get("issue_number"): row for row in (triage_result or {}).get("results", [])}
    issue_views = [_issue_dto(issue, assessments.get(issue.get("number")), run_id, secrets) for issue in issues]
    for issue in issue_views:
        if not issue["reproduction"]["actions"] and issue["evidenceSteps"]:
            anchor = min(issue["evidenceSteps"]) if issue["anomalyType"] in {"passed_through_solid", "sank_into_visible_floor"} else max(issue["evidenceSteps"])
            controls = _actions_through(rows, anchor)
            issue["reproduction"].update(actions=[c["action"] for c in controls], sourceSteps=[c["source_step"] for c in controls])
    summaries = [row for row in rows if row.get("type") == "summary"]
    latest = summaries[-1] if summaries else {}
    warnings = sum(row.get("type") in {"decision_warning", "rejected_decision", "triage_error", "dashboard_warning"} for row in rows)
    warnings += sum(len(row.get("warnings", [])) for row in (triage_result or {}).get("results", []))
    pending = sum(issue["assessment"]["status"] != "reviewed" for issue in issue_views)
    if latest.get("unassessed_final_step") is not None:
        pending += 1
    request_count = sum(row.get("type") == "request" for row in rows)
    request_count += sum(type(row.get("request_number")) is int for row in (triage_result or {}).get("results", []))
    accounted = cost.get("accounted_cost_usd") if isinstance(cost, dict) else None
    cost_known = bool(cost and cost.get("usage_known") is True)
    if type(accounted) not in (int, float) or not math.isfinite(accounted):
        accounted = None
        cost_known = False
    duration = elapsed_seconds if elapsed_seconds is not None else max((_elapsed(row) for row in rows), default=0)
    if type(duration) not in (int, float) or not math.isfinite(duration):
        duration = 0
    summary = {
        "id": run_id, "name": safe_text(name or ("Unity physics session" if game == "unity" else "Grid exploration session"), secrets),
        "game": game, "source": source if source in {"recorded", "live"} else "recorded",
        "status": status, "phase": phase, "seed": header.get("seed", 0),
        "createdAt": created_at or header.get("timestamp") or datetime.now(timezone.utc).isoformat(),
        "durationSeconds": round(max(0, duration), 3), "controls": len(frames),
        "resets": sum(frame["action"] == "reset" for frame in frames),
        "requests": cost.get("requests", request_count) if isinstance(cost, dict) else request_count,
        "estimatedCostUsd": round(accounted, 8) if accounted is not None else None, "costKnown": cost_known,
        "issueCount": len(issue_views), "budgetUsd": budget_usd, "warningCount": warnings,
        "pendingCount": pending,
    }
    public_notes = [safe_text(note, secrets) for note in notes]
    if source == "recorded":
        public_notes.insert(0, "Saved recording replay. No game controls or model requests are executed by playback.")
        if duration == 0 and frames:
            public_notes.append("The original scripted recording did not capture wall-clock timing; its elapsed counter is unavailable.")
        elif any(row.get("type") == "sweep_summary" for row in rows):
            public_notes.append("Saved elapsed counters cover the recorded model session; the earlier collision sweep was written afterward and has no original per-control timing.")
    public_notes.append("Host signals and AI judgments remain separate. Execution evidence does not prove model claims or intent.")
    if latest.get("unassessed_final_step") is not None:
        public_notes.append(f"The final model action (step {latest['unassessed_final_step']}) has a recorded outcome without a subsequent model assessment.")
    if any(i.get("reviewNote") == _KEY_NOTE for i in issue_views):
        public_notes.append(_KEY_NOTE)
    if accounted is None:
        public_notes.append("Complete combined pricing data is unavailable for this saved session; no zero-cost claim is made.")
    return {
        "version": VERSION, "summary": summary, "scene": scene, "frames": frames, "issues": issue_views,
        "timeline": timeline[-1600:], "notes": public_notes, "limits": dict(LIMITS),
        "models": {"exploration": DEFAULT_MODEL, "triage": DEFAULT_TRIAGE_MODEL},
    }


def snapshot_markdown(snapshot: dict) -> str:
    s = snapshot["summary"]
    lines = [f"# {s['name']}", "", f"Source: {s['source']} · Status: {s['status']} · Seed: {s['seed']}",
             f"Controls: {s['controls']} · Resets: {s['resets']} · Requests: {s['requests']}", ""]
    for note in snapshot.get("notes", []):
        lines.extend([note, ""])
    for issue in snapshot.get("issues", []):
        a, r = issue["assessment"], issue["reproduction"]
        lines.extend([f"## {issue['number']}. {issue['title']}", "", f"Host signal: {issue['anomalyType']}",
                      f"Evidence steps: {', '.join(map(str, issue['evidenceSteps']))}",
                      f"AI: {a.get('verdict', a['status'])} · {a.get('severity', 'unrated')}", ""])
        for key in ("playerImpact", "rationale"):
            if a.get(key):
                lines.extend([a[key], ""])
        if issue.get("reviewNote"):
            lines.extend([issue["reviewNote"], ""])
        lines.extend(["Reproduction:", "", f"Fresh verification: {r['verified']} · Minimality verified: {r['minimal']}",
                      "Actions: " + " → ".join(r["actions"]),
                      "Original source steps: " + ", ".join(map(str, r["sourceSteps"])), ""])
        for limitation in a.get("limitations", []):
            lines.extend([f"Limitation: {limitation}", ""])
    return "\n".join(lines)


def export_demo(repository: Path) -> list[dict]:
    """Export saved live/benchmark evidence and final Step-1 judgments only."""
    repository = Path(repository)
    final = repository / "diagnostics/ultra-triage-final-2026-10-14"
    cases = json.loads((final / "cases.json").read_text(encoding="utf-8"))
    reviewed = json.loads((final / "result.json").read_text(encoding="utf-8"))
    by_source = {}
    for case, result in zip(cases, reviewed["results"]):
        by_source.setdefault(case["source_recording"], []).append(result)
    plan = [
        ("unity-floor", "Unity · Floor support", "runs/unity-live-floor-gap-seed0.jsonl"),
        ("unity-wall", "Unity · Wall overlap", "runs/unity-live-wall-tunnel-seed5.jsonl"),
        ("unity-clean", "Unity · Baseline session", "runs/unity-live-none-seed1.jsonl"),
        ("unity-door", "Unity · Open-door movement", "diagnostics/unity-benchmark/door-ghost-collider-seed0.jsonl"),
        ("unity-key", "Unity · Key use feedback", "diagnostics/unity-benchmark/key-not-consumed-seed0.jsonl"),
        ("unity-reset", "Unity · Reset comparison", "diagnostics/unity-benchmark/respawn-leak-seed0.jsonl"),
        ("grid-coverage", "Grid · Collision coverage", "runs/phase4-hybrid-wall-clip-seed0-live1.jsonl"),
    ]
    output = repository / "web/public/demo"
    output.mkdir(parents=True, exist_ok=True)
    manifests = []
    for run_id, name, recording in plan:
        records = load_recording(repository / recording)
        original_issues = extract_issues(records)
        issue_rows = deepcopy(original_issues)
        results = []
        for result in by_source.get(recording, []):
            result = deepcopy(result)
            packet = result.get("evidence_packet", {})
            candidate = packet.get("candidate", {})
            match = next((issue for issue in issue_rows if issue.get("type") == candidate.get("type") and issue.get("title") == candidate.get("title")), None)
            if match is None:
                # Authored normal candidates were reviewed in Step 1. Keep the
                # origin explicit rather than inventing a detector finding.
                match = {"number": len(issue_rows) + 1, "source": "authored_normal_candidate",
                         "type": candidate.get("type", "normal_behavior"), "title": candidate.get("title", "Normal behavior review"),
                         "expected": candidate.get("expected"), "actual": candidate.get("actual"),
                         "evidence_steps": packet.get("evidence_steps", [])}
                issue_rows.append(match)
            result["issue_number"] = match["number"]
            result["evidence_packet"]["issue_number"] = match["number"]
            results.append(result)
        triage = {"results": results} if results else None
        notes = []
        if recording.startswith("diagnostics/"):
            notes.append("The game actions in this saved session came from a scripted benchmark; Ultra reviewed the saved evidence separately.")
        elif results:
            notes.append("Lightning exploration was recorded earlier; these Ultra assessments are from the final Step-1 evidence review on October 14, 2026.")
        if run_id == "unity-clean":
            notes.append("The host detector raised no issues. Three explicitly authored normal candidates are retained to show the AI review's negative controls.")
        if run_id == "grid-coverage":
            notes.append("This grid session has host candidates and reproducible controls; no Ultra assessment has been claimed for it.")
        snapshot = snapshot_from_records(records, run_id=run_id, name=name, triage_result=triage,
                                         candidate_issues=issue_rows, notes=notes)
        # Supply exact recorded prefixes for candidates without a Step-1 packet.
        for issue in snapshot["issues"]:
            if not issue["reproduction"]["actions"] and issue["evidenceSteps"]:
                final_step = min(issue["evidenceSteps"]) if issue["anomalyType"] in {"passed_through_solid", "sank_into_visible_floor"} else max(issue["evidenceSteps"])
                relevant = [f for f in snapshot["frames"] if f["step"] <= final_step]
                issue["reproduction"].update(actions=[f["action"] for f in relevant], sourceSteps=[f["step"] for f in relevant])
        (output / f"{run_id}.json").write_text(json.dumps(snapshot, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n", encoding="utf-8")
        (output / f"{run_id}.md").write_text(snapshot_markdown(snapshot), encoding="utf-8")
        manifests.append(snapshot["summary"])
    manifest_path = repository / "web/lib/demo-manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps({"runs": manifests}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifests
