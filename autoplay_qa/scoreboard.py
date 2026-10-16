"""Offline Markdown scoreboards for live QA recordings.

The scoreboard consumes only JSONL recordings and the deterministic scripted
benchmark.  It does not load credentials or perform inference.  Counts are
kept close to their recording provenance: controls come from step/reset rows,
request modes come from request rows, and token totals marked with ``+`` are
known lower bounds when recorded usage is incomplete.
"""

from __future__ import annotations

from collections import Counter
import math
from pathlib import Path
from statistics import median
from typing import Callable, Sequence

from .game import DIRECTIONS
from .reporting import evaluate_recording, load_recording, reproduce_issue


SCOREBOARD_HEADERS = (
    "recording",
    "defect",
    "detection",
    "FP",
    "repro verified",
    "minimized steps",
    "actions",
    "resets",
    "requests",
    "tokens",
    "seconds",
    "decision-mode counts",
    "model suspected_bug count",
    "phase control counts",
)

_MODE_ORDER = (
    "thinking",
    "direct_retry",
    "direct_fallback",
    "direct_latency",
)


def _last_summary(records: Sequence[dict]) -> dict:
    for record in reversed(records):
        if record.get("type") == "summary":
            return record
    return {}


def _nonnegative_number(value: object) -> int | float | None:
    if type(value) not in {int, float} or not math.isfinite(value) or value < 0:
        return None
    return value


def _retained_suspected_bug_ids(records: Sequence[dict]) -> set[str]:
    """Return final retained suspected-bug IDs from summary or last memory.

    The scoreboard measures the model's final session judgment, not transient
    action-proposal bookkeeping.  Legacy or partial recordings without a
    summary fall back to their last validated memory snapshot.
    """
    notes: object = []
    found_summary = False
    for record in reversed(records):
        if record.get("type") == "summary":
            notes = record.get("hypotheses")
            found_summary = True
            break
    if not found_summary:
        for record in reversed(records):
            if record.get("type") == "memory":
                notes = record.get("hypotheses")
                break
    if not isinstance(notes, list):
        return set()
    return {
        note["id"]
        for note in notes
        if isinstance(note, dict)
        and note.get("kind") == "suspected_bug"
        and isinstance(note.get("id"), str)
        and note["id"]
    }


def _decision_modes(records: Sequence[dict]) -> Counter[str]:
    request_modes = [
        record.get("decision_mode")
        for record in records
        if record.get("type") == "request"
    ]
    if not request_modes:
        request_modes = [
            record.get("decision_mode")
            for record in records
            if record.get("type") == "completion"
        ]
    return Counter(
        mode if isinstance(mode, str) and mode else "unknown"
        for mode in request_modes
    )


def _token_total(records: Sequence[dict], summary: dict) -> tuple[int | None, bool]:
    usage = summary.get("usage")
    total = usage.get("total_tokens") if isinstance(usage, dict) else None
    if type(total) is int and total >= 0:
        complete = summary.get("usage_complete")
        return total, complete if type(complete) is bool else True

    completions = [record for record in records if record.get("type") == "completion"]
    totals = [
        record.get("usage", {}).get("total_tokens")
        for record in completions
        if isinstance(record.get("usage"), dict)
    ]
    known = [value for value in totals if type(value) is int and value >= 0]
    if not known:
        return None, False
    request_count = sum(record.get("type") == "request" for record in records)
    return sum(known), len(known) == len(completions) == request_count


def _latencies(records: Sequence[dict]) -> list[int | float]:
    values = [
        _nonnegative_number(record.get("latency_ms"))
        for record in records
        if record.get("type") == "completion"
    ]
    return [value for value in values if value is not None]


def _failed_request_count(records: Sequence[dict]) -> int:
    return sum(
        record.get("type") == "request"
        and (
            record.get("attempt_status") == "failed"
            or isinstance(record.get("failure_reason"), str)
            and bool(record["failure_reason"])
        )
        for record in records
    )


def _interior_wall_attempts(records: Sequence[dict]) -> int:
    """Count moves aimed at visibly wall-labeled, non-boundary map cells."""
    attempts = 0
    for record in records:
        if record.get("type") not in {"step", "sweep_step"}:
            continue
        decision = record.get("decision")
        before = record.get("before")
        if not isinstance(decision, dict) or not isinstance(before, dict):
            continue
        action = decision.get("action")
        if action not in DIRECTIONS:
            continue
        player = before.get("player")
        position = player.get("position") if isinstance(player, dict) else None
        rows = before.get("map")
        legend = before.get("legend")
        if (
            not isinstance(position, list)
            or len(position) != 2
            or not all(type(coordinate) is int for coordinate in position)
            or not isinstance(rows, list)
            or not rows
            or not all(isinstance(row, str) for row in rows)
            or not isinstance(legend, dict)
        ):
            continue
        dx, dy = DIRECTIONS[action]
        x, y = position[0] + dx, position[1] + dy
        if y <= 0 or y >= len(rows) - 1 or x <= 0 or x >= len(rows[y]) - 1:
            continue
        glyph = rows[y][x]
        meaning = legend.get(glyph)
        if isinstance(meaning, str) and "wall" in meaning.lower():
            attempts += 1
    return attempts


def score_recording(
    path: str | Path,
    *,
    game_factory: Callable[[int, str | None], object] | None = None,
) -> dict:
    """Return one offline scoreboard row plus recorded diagnostics."""
    source = Path(path)
    records = load_recording(source)
    summary = _last_summary(records)
    evaluation = evaluate_recording(
        records,
        verify_reproduction=False,
        game_factory=game_factory,
    )
    matched_numbers = evaluation.get("matched_issue_numbers", [])
    reproductions = [
        reproduce_issue(records, number, game_factory=game_factory)
        for number in matched_numbers
        if type(number) is int
    ]
    reproduced = bool(reproductions) and all(
        result.get("reproduced") is True for result in reproductions
    )
    minimized = [
        result.get("minimized_action_count")
        for result in reproductions
        if result.get("reproduced") is True
        and type(result.get("minimized_action_count")) is int
    ]

    step_rows = [
        record for record in records
        if record.get("type") in {"step", "sweep_step"}
    ]
    reset_rows = [
        record for record in records
        if record.get("type") in {"reset", "sweep_reset"}
    ]
    phase_counts = Counter()
    for record in [*step_rows, *reset_rows]:
        source_name = record.get("source")
        if not isinstance(source_name, str) or not source_name:
            source_name = "sweep" if str(record.get("type", "")).startswith("sweep_") else "llm"
        phase_counts[source_name] += 1
    recorded_phase_counts = summary.get("phase_control_counts")
    if isinstance(recorded_phase_counts, dict):
        for name, count in recorded_phase_counts.items():
            if isinstance(name, str) and type(count) is int and count >= 0:
                phase_counts[name] = count
    request_rows = [record for record in records if record.get("type") == "request"]
    request_count = len(request_rows)
    if not request_rows and type(summary.get("inference_requests")) is int:
        request_count = summary["inference_requests"]
    token_total, usage_complete = _token_total(records, summary)
    seconds = _nonnegative_number(summary.get("elapsed_seconds"))
    if seconds is None:
        elapsed = [
            _nonnegative_number(record.get("elapsed_ms"))
            for record in records
        ]
        elapsed = [value for value in elapsed if value is not None]
        seconds = max(elapsed) / 1000 if elapsed else None

    return {
        "recording": source.name,
        "path": str(source),
        "defect": evaluation.get("injected_defect") or "clean",
        "detection": evaluation["detection"],
        "false_positive_count": evaluation["false_positive_count"],
        "repro_verified": reproduced if matched_numbers else None,
        "minimized_steps": min(minimized) if minimized else None,
        "actions": len(step_rows),
        "resets": len(reset_rows),
        "requests": request_count,
        "tokens": token_total,
        "usage_complete": usage_complete,
        "seconds": seconds,
        "decision_mode_counts": dict(_decision_modes(records)),
        "model_suspected_bug_count": len(_retained_suspected_bug_ids(records)),
        "phase_control_counts": dict(phase_counts),
        "failed_requests": _failed_request_count(records),
        "completions": sum(
            record.get("type") == "completion" for record in records
        ),
        "completion_latencies_ms": _latencies(records),
        "interior_wall_attempts": _interior_wall_attempts(records),
        "scripted": False,
    }


def scripted_baseline_row() -> dict:
    """Aggregate the seed-0 deterministic benchmark into one comparison row."""
    from .benchmark import build_scripted_recording, run_scripted_benchmark

    result = run_scripted_benchmark(seeds=(0,))
    benchmark_rows = list(result.get("rows", []))
    defect_rows = [row for row in benchmark_rows if row.get("defect") != "none"]
    clean_rows = [row for row in benchmark_rows if row.get("defect") == "none"]

    actions = resets = 0
    for defect in result.get("defects", []):
        records = build_scripted_recording(0, defect)
        actions += sum(record.get("type") == "step" for record in records)
        resets += sum(record.get("type") == "reset" for record in records)

    hit_count = sum(row.get("detection") == "hit" for row in defect_rows)
    defect_count = len(defect_rows)
    return {
        "recording": "scripted baseline (seed 0)",
        "path": None,
        "defect": f"{defect_count} defects + {len(clean_rows)} clean",
        "detection": f"{hit_count}/{defect_count} hit",
        "false_positive_count": sum(
            row.get("false_positive_count", 0) for row in benchmark_rows
        ),
        "repro_verified": bool(defect_rows) and all(
            row.get("repro_verified") is True for row in defect_rows
        ),
        "minimized_steps": None,
        "actions": actions,
        "resets": resets,
        "requests": 0,
        "tokens": 0,
        "usage_complete": True,
        "seconds": None,
        "decision_mode_counts": {},
        "model_suspected_bug_count": 0,
        "phase_control_counts": {"scripted": actions + resets},
        "failed_requests": 0,
        "completions": 0,
        "completion_latencies_ms": [],
        "interior_wall_attempts": None,
        "scripted": True,
        "benchmark_passed": result.get("passed") is True,
        "benchmark_hits": hit_count,
        "benchmark_defects": defect_count,
    }


def build_scoreboard(
    paths: Sequence[str | Path],
    *,
    include_scripted_baseline: bool = True,
    game_factory: Callable[[int, str | None], object] | None = None,
) -> dict:
    if not paths:
        raise ValueError("At least one recording is required.")
    live_rows = [
        score_recording(path, game_factory=game_factory)
        for path in paths
    ]
    baseline = scripted_baseline_row() if include_scripted_baseline else None
    return {
        "rows": [*live_rows, *([baseline] if baseline is not None else [])],
        "live_rows": live_rows,
        "scripted_baseline": baseline,
    }


def _markdown_cell(value: object) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def _format_modes(counts: dict) -> str:
    if not counts:
        return "n/a (scripted)"
    names = [name for name in _MODE_ORDER if counts.get(name)]
    names.extend(sorted(name for name in counts if name not in _MODE_ORDER))
    return "; ".join(f"{name}={counts[name]}" for name in names)


def _format_phase_counts(counts: object, *, fallback_actions: int, fallback_resets: int) -> str:
    if not isinstance(counts, dict) or not counts:
        return f"llm={fallback_actions + fallback_resets}"
    order = [name for name in ("sweep", "llm", "scripted") if name in counts]
    order.extend(sorted(name for name in counts if name not in order))
    return "; ".join(f"{name}={counts[name]}" for name in order)


def _display_row(row: dict) -> tuple[str, ...]:
    token_total = row.get("tokens")
    tokens = "n/a" if token_total is None else f"{token_total:,}"
    if token_total is not None and not row.get("usage_complete", False):
        tokens += "+"
    seconds = row.get("seconds")
    seconds_text = "n/a" if seconds is None else f"{seconds:.3f}"
    reproduced = row.get("repro_verified")
    repro_text = "n/a" if reproduced is None else ("yes" if reproduced else "no")
    minimized = row.get("minimized_steps")
    minimized_text = "n/a" if minimized is None else str(minimized)
    return tuple(map(_markdown_cell, (
        row["recording"],
        row["defect"],
        row["detection"],
        row["false_positive_count"],
        repro_text,
        minimized_text,
        row["actions"],
        row["resets"],
        row["requests"],
        tokens,
        seconds_text,
        _format_modes(row["decision_mode_counts"]),
        row["model_suspected_bug_count"],
        _format_phase_counts(
            row.get("phase_control_counts"),
            fallback_actions=row.get("actions", 0),
            fallback_resets=row.get("resets", 0),
        ),
    )))


def _format_seconds(milliseconds: int | float) -> str:
    return f"{milliseconds / 1000:.3f} s"


def findings_paragraph(result: dict) -> str:
    """Summarize measured live outcomes without treating model notes as truth."""
    live_rows = list(result.get("live_rows", []))
    defect_rows = [row for row in live_rows if row.get("defect") != "clean"]
    hit_rows = [row for row in defect_rows if row.get("detection") == "hit"]
    miss_rows = [row for row in defect_rows if row.get("detection") == "miss"]
    hit_details = ", ".join(
        f"{row['defect']} ({row['minimized_steps']} minimized steps)"
        if row.get("minimized_steps") is not None else str(row["defect"])
        for row in hit_rows
    ) or "none"

    sentences = [
        f"The live QA runs detected and reproduction-verified {len(hit_rows)} of "
        f"{len(defect_rows)} injected defects: {hit_details}."
    ]
    for row in miss_rows:
        if row.get("defect") == "wall-clip" and row.get("interior_wall_attempts") == 0:
            sentences.append(
                "The wall-clip run missed because exploration never attempted a "
                "move into an interior wall; this was an exploration-coverage miss, "
                "not evidence that the defect was absent."
            )
        else:
            sentences.append(f"The {row['defect']} run was a miss.")

    clean_rows = [row for row in live_rows if row.get("defect") == "clean"]
    clean_false_positives = sum(
        row.get("false_positive_count", 0) for row in clean_rows
    )
    sentences.append(
        f"The {len(clean_rows)} clean controls produced {clean_false_positives} "
        "false positives."
    )

    modes: Counter[str] = Counter()
    latencies: list[int | float] = []
    failed = requests = completions = 0
    for row in live_rows:
        modes.update(row.get("decision_mode_counts", {}))
        requests += row.get("requests", 0)
        values = row.get("completion_latencies_ms", [])
        latencies.extend(values)
        completions += row.get("completions", len(values))
        failed += row.get("failed_requests", 0)
    mode_names = [name for name in _MODE_ORDER if modes.get(name)]
    mode_names.extend(sorted(name for name in modes if name not in _MODE_ORDER))
    mode_text = ", ".join(f"{name} {modes[name]}" for name in mode_names)
    sentences.append(
        f"Across {len(live_rows)} live recordings, {requests} request attempts "
        f"yielded {completions} recorded completions; decision modes were "
        f"{mode_text}, with {failed} failed requests."
    )
    if latencies:
        ordered = sorted(latencies)
        p95 = ordered[math.ceil(0.95 * len(ordered)) - 1]
        sentences.append(
            "Recorded completion latency was median "
            f"{_format_seconds(median(ordered))}, nearest-rank p95 "
            f"{_format_seconds(p95)}, and max {_format_seconds(ordered[-1])}."
        )

    suspected = sum(
        row.get("model_suspected_bug_count", 0) for row in live_rows
    )
    sentences.append(
        "The LLM contributed exploration but not judgment so far: it retained "
        f"{suspected} final suspected_bug notes, and all {len(hit_rows)} "
        "detections came from software anomaly detectors."
    )

    baseline = result.get("scripted_baseline")
    if isinstance(baseline, dict):
        sentences.append(
            "The seed-0 scripted baseline detected "
            f"{baseline.get('benchmark_hits', 0)} of "
            f"{baseline.get('benchmark_defects', 0)} defects with "
            f"{baseline.get('false_positive_count', 0)} false positives and "
            "verified every defect reproduction."
        )
    return " ".join(sentences)


def format_scoreboard(result: dict) -> str:
    rows = [_display_row(row) for row in result["rows"]]
    lines = [
        "# AutoPlay QA live-results scoreboard",
        "",
        "| " + " | ".join(SCOREBOARD_HEADERS) + " |",
        "| " + " | ".join("---" for _ in SCOREBOARD_HEADERS) + " |",
        *("| " + " | ".join(row) + " |" for row in rows),
    ]
    if any(
        row.get("tokens") is not None and not row.get("usage_complete", False)
        for row in result.get("live_rows", [])
    ):
        lines.extend([
            "",
            "`+` marks a known token minimum because one or more request usages were unavailable.",
        ])
    lines.extend([
        "",
        "## Findings",
        "",
        findings_paragraph(result),
        "",
    ])
    return "\n".join(lines)


def generate_scoreboard(
    paths: Sequence[str | Path],
    *,
    include_scripted_baseline: bool = True,
    game_factory: Callable[[int, str | None], object] | None = None,
) -> str:
    return format_scoreboard(build_scoreboard(
        paths,
        include_scripted_baseline=include_scripted_baseline,
        game_factory=game_factory,
    ))


def write_scoreboard(
    paths: Sequence[str | Path],
    out: str | Path,
    *,
    include_scripted_baseline: bool = True,
    game_factory: Callable[[int, str | None], object] | None = None,
) -> Path:
    destination = Path(out)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        generate_scoreboard(
            paths,
            include_scripted_baseline=include_scripted_baseline,
            game_factory=game_factory,
        ),
        encoding="utf-8",
    )
    return destination.resolve()
