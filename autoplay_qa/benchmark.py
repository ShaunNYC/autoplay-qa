"""Deterministic, inference-free coverage benchmark for the QA pipeline.

The scripted policy consumes only public observations.  It collects the key,
resets once, visits the visible map, interacts at reachable positions, probes
visible walls, and attempts the exit.  The resulting in-memory recording is
then passed through the same report, reproduce, and evaluate entry points used
for live recordings.
"""

from __future__ import annotations

from collections import deque
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
from typing import Callable, Iterable, Sequence

from .game import (
    DEFECT_CHOICES,
    DIRECTIONS,
    KeyGateGame,
    defect_evaluation_metadata,
)
from .reporting import (
    evaluate_recording,
    extract_issues,
    generate_report,
    reproduce_issue,
)
from .exploration import ExplorationLimits
from .hybrid import run_collision_sweep
from .lean import LEAN_PROMPT_VERSION, run_lean_exploration
from .nebius import Completion
from .session import SessionLog, replay
from .unity_launcher import (
    UNITY_DEFECT_CHOICES,
    unity_defect_evaluation_metadata,
)


SCRIPTED_SEEDS = (0, 1, 5)
_DIRECTION_ORDER = tuple(DIRECTIONS)
_REVERSE_DIRECTION = {
    "north": "south",
    "south": "north",
    "east": "west",
    "west": "east",
}


class _OfflineFinishClient:
    """One-decision stub used only by the deterministic hybrid benchmark."""

    def __init__(self):
        self.settings = SimpleNamespace(enable_thinking=False)
        self.timeout = 45
        self.max_tokens = 8192
        self.temperature = 1.0
        self.top_p = 0.95

    def complete(self, _messages, **_kwargs):
        return Completion(
            json.dumps({
                "action": "finish",
                "reason": "Collision coverage is complete; preserve the remaining LLM budget.",
                "expected_position": None,
                "notes": [],
            }),
            {
                "latency_ms": 0,
                "finish_reason": "stop",
                "usage": {
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                },
            },
        )


def _position(value: object) -> tuple[int, int] | None:
    if (
        isinstance(value, (list, tuple))
        and len(value) == 2
        and all(type(coordinate) is int for coordinate in value)
    ):
        return value[0], value[1]
    return None


class _Recording:
    """Record direct game controls in the Phase-2 JSONL-compatible shape."""

    def __init__(
        self,
        seed: int,
        defect: str,
        game_factory: Callable[[int, str | None], object] | None = None,
    ):
        self.game = (
            game_factory(seed, None if defect == "none" else defect)
            if game_factory is not None
            else KeyGateGame(seed, defect)
        )
        self.episode = 0
        self.action_records: list[dict] = []
        initial = deepcopy(self.game.observe())
        session_metadata = getattr(self.game, "session_metadata", {})
        session_metadata = (
            deepcopy(session_metadata)
            if isinstance(session_metadata, dict)
            else {}
        )
        evaluation_metadata = (
            defect_evaluation_metadata(defect)
            if defect in DEFECT_CHOICES
            else unity_defect_evaluation_metadata(defect)
        )
        self.records: list[dict] = [{
            "type": "session",
            "schema_version": 2,
            "game_version": self.game.version,
            "seed": seed,
            "mode": "scripted-benchmark",
            "task": "explore",
            "focus": "qa",
            "initial_observation": initial,
            **session_metadata,
            **evaluation_metadata,
        }]

    @property
    def observation(self) -> dict:
        return self.game.observe()

    def perform(self, action: str) -> dict:
        before = deepcopy(self.game.observe())
        step = len(self.action_records) + 1
        episode_before = self.episode
        if action == "reset":
            self.game.reset()
            self.episode += 1
            after = deepcopy(self.game.observe())
            record_type = "reset"
        else:
            after = deepcopy(self.game.act(action))
            record_type = "step"
        record = {
            "type": record_type,
            "step": step,
            "episode": self.episode,
            "episode_before": episode_before,
            "episode_after": self.episode,
            "before": before,
            "decision": {
                "action": action,
                "reason": "Deterministic scripted coverage probe.",
                # A scripted route is not a model prediction.  Leaving this
                # null also prevents expected blocking from becoming noise.
                "expected_position": None,
                "notes": [],
            },
            "after": after,
            "qa_mode": True,
        }
        if record_type == "step":
            record["evaluation"] = self.game.evaluate_transition(
                before, action, after
            )
        self.action_records.append(record)
        self.records.append(record)
        return record

    def finish(self) -> list[dict]:
        final_observation = deepcopy(self.game.observe())
        self.records.append({
            "type": "summary",
            "status": "won" if final_observation.get("status") == "won" else "finished",
            "steps": len(self.action_records),
            "game_actions": sum(
                record["decision"]["action"] != "reset"
                for record in self.action_records
            ),
            "resets": sum(
                record["decision"]["action"] == "reset"
                for record in self.action_records
            ),
            "inference_requests": 0,
            "hypotheses": [],
            "final_observation": final_observation,
        })
        close = getattr(self.game, "close", None)
        if callable(close):
            close()
        return self.records

    def add_sweep(self, result) -> None:
        """Append already-executed sweep controls to this recording."""
        for original in result.records:
            record = deepcopy(original)
            self.action_records.append(record)
            self.records.append(record)
        self.records.append({
            "type": "sweep_summary",
            "source": "sweep",
            "summary": deepcopy(result.summary),
        })


def _continuous_sweep_supported(game: object, observation: dict) -> bool:
    capabilities = getattr(game, "capabilities", None)
    if callable(capabilities):
        capabilities = capabilities()
    if not isinstance(capabilities, dict):
        capabilities = observation.get("_capabilities")
    return (
        isinstance(capabilities, dict)
        and capabilities.get("continuous") is True
        and capabilities.get("solids_visible") is True
    )


class _ScriptedProbe:
    """Small observation-driven navigator; it has no defect-specific branch."""

    def __init__(self, recording: _Recording):
        self.recording = recording
        observation = recording.observation
        self.legend = deepcopy(observation.get("legend", {}))
        self.terrain: dict[tuple[int, int], str] = {}
        for y, row in enumerate(observation["map"]):
            for x, glyph in enumerate(row):
                # The player overlays a traversable starting tile.
                self.terrain[(x, y)] = "." if glyph == "@" else glyph
        objects = observation["objects"]
        self.gate = _position(objects.get("gate", {}).get("position"))
        self.exit = _position(objects.get("exit"))
        self.blocked_edges: set[tuple[tuple[int, int], tuple[int, int]]] = set()
        self.interacted_positions: set[tuple[int, int]] = set()

    def _meaning(self, position: tuple[int, int]) -> str:
        glyph = self.terrain.get(position)
        meaning = self.legend.get(glyph, "")
        return meaning.lower() if isinstance(meaning, str) else ""

    def _is_wall(self, position: tuple[int, int]) -> bool:
        return "wall" in self._meaning(position)

    def _can_enter(
        self,
        position: tuple[int, int],
        *,
        current: tuple[int, int],
        include_exit: bool,
    ) -> bool:
        if position == current:
            return True
        if position not in self.terrain or self._is_wall(position):
            return False
        if position == self.exit and not include_exit:
            return False
        gate_state = self.recording.observation["objects"]["gate"]
        if position == self.gate and not gate_state["open"]:
            return False
        return True

    def _path_toward(
        self,
        target: tuple[int, int],
        *,
        include_exit: bool,
    ) -> list[str] | None:
        start = _position(self.recording.observation["player"]["position"])
        if start is None:
            return None
        if start == target:
            return []
        queue = deque([start])
        previous: dict[tuple[int, int], tuple[tuple[int, int], str]] = {}
        seen = {start}
        while queue:
            current = queue.popleft()
            for action in _DIRECTION_ORDER:
                dx, dy = DIRECTIONS[action]
                neighbor = current[0] + dx, current[1] + dy
                if neighbor in seen or (current, neighbor) in self.blocked_edges:
                    continue
                if not self._can_enter(
                    neighbor, current=start, include_exit=include_exit
                ):
                    continue
                previous[neighbor] = current, action
                if neighbor == target:
                    path: list[str] = []
                    cursor = neighbor
                    while cursor != start:
                        parent, path_action = previous[cursor]
                        path.append(path_action)
                        cursor = parent
                    path.reverse()
                    return path
                seen.add(neighbor)
                queue.append(neighbor)
        return None

    def navigate(
        self,
        target: tuple[int, int] | None,
        *,
        include_exit: bool = False,
    ) -> bool:
        if target is None:
            return False
        # Each failed attempt removes one directed edge from future searches.
        # The bound is defensive for malformed adapters, not a coverage limit.
        for _ in range(max(1, len(self.terrain) * len(_DIRECTION_ORDER))):
            current = _position(self.recording.observation["player"]["position"])
            if current == target:
                return True
            if self.recording.observation.get("status") == "won":
                return False
            path = self._path_toward(target, include_exit=include_exit)
            if not path:
                return False
            action = path[0]
            dx, dy = DIRECTIONS[action]
            attempted = current[0] + dx, current[1] + dy
            record = self.recording.perform(action)
            after = _position(record["after"]["player"]["position"])
            if after == current:
                self.blocked_edges.add((current, attempted))
        return False

    def interact(self) -> None:
        if self.recording.observation.get("status") == "won":
            return
        before_open = self.recording.observation["objects"]["gate"]["open"]
        self.recording.perform("interact")
        after_open = self.recording.observation["objects"]["gate"]["open"]
        if after_open != before_open:
            # A newly open gate can make an earlier route viable.
            self.blocked_edges.clear()

    def reset(self) -> None:
        self.recording.perform("reset")
        self.blocked_edges.clear()

    def collect_visible_key(self) -> None:
        key = _position(self.recording.observation["objects"].get("key"))
        if key is not None:
            self.navigate(key)

    def visit_and_interact_everywhere(self) -> None:
        candidates = {
            position
            for position in self.terrain
            if not self._is_wall(position) and position != self.exit
        }
        while (
            candidates - self.interacted_positions
            and self.recording.observation.get("status") != "won"
        ):
            current = _position(self.recording.observation["player"]["position"])
            routes = []
            for target in candidates - self.interacted_positions:
                path = self._path_toward(target, include_exit=False)
                if path is not None:
                    routes.append((len(path), target[1], target[0], target))
            if not routes:
                break
            target = min(routes)[-1]
            if not self.navigate(target):
                continue
            self.interacted_positions.add(target)
            self.interact()

    def probe_visible_walls(self) -> None:
        wall_targets = sorted(
            (position for position in self.terrain if self._is_wall(position)),
            key=lambda item: (item[1], item[0]),
        )
        for wall in wall_targets:
            if self.recording.observation.get("status") == "won":
                return
            sources: list[tuple[tuple[int, int], str]] = []
            for action in _DIRECTION_ORDER:
                dx, dy = DIRECTIONS[action]
                source = wall[0] - dx, wall[1] - dy
                if (
                    source in self.terrain
                    and not self._is_wall(source)
                    and source != self.exit
                ):
                    sources.append((source, action))
            for source, action in sources:
                if not self.navigate(source):
                    continue
                before = _position(self.recording.observation["player"]["position"])
                record = self.recording.perform(action)
                after = _position(record["after"]["player"]["position"])
                if after == wall:
                    # Leave an unexpectedly enterable wall immediately so the
                    # remainder of the generic scan stays navigable.
                    self.recording.perform(_REVERSE_DIRECTION[action])
                elif after != before:
                    # Unexpected movement still constitutes a completed probe.
                    pass
                break

    def ensure_gate_interaction(self) -> None:
        gate_state = self.recording.observation["objects"]["gate"]
        if gate_state["open"] or self.gate is None:
            return
        for action in _DIRECTION_ORDER:
            dx, dy = DIRECTIONS[action]
            adjacent = self.gate[0] - dx, self.gate[1] - dy
            if self.navigate(adjacent):
                self.interact()
                if self.recording.observation["objects"]["gate"]["open"]:
                    return

    def run(self) -> None:
        # Reset after a meaningful state change, then repeat from the visible
        # post-reset state.  This catches retained state without supplied rules.
        self.collect_visible_key()
        self.reset()
        self.collect_visible_key()

        self.visit_and_interact_everywhere()
        self.ensure_gate_interaction()
        self.probe_visible_walls()

        # Traversal may have opened the gate late; revisit newly reachable
        # positions before the final exit attempt.
        self.visit_and_interact_everywhere()
        self.ensure_gate_interaction()
        if self.exit is not None:
            self.navigate(self.exit, include_exit=True)


def build_scripted_recording(
    seed: int,
    defect: str = "none",
    *,
    game_factory: Callable[[int, str | None], object] | None = None,
    sweep_actions: int = 400,
) -> list[dict]:
    """Run one deterministic public-observation probe and return its records."""
    recording = _Recording(seed, defect, game_factory)
    if _continuous_sweep_supported(recording.game, recording.observation):
        recording.add_sweep(run_collision_sweep(
            recording.game,
            max_actions=sweep_actions,
        ))
        return recording.finish()
    _ScriptedProbe(recording).run()
    return recording.finish()


def build_hybrid_stub_recording(
    seed: int,
    defect: str,
    *,
    sweep_actions: int = 400,
) -> list[dict]:
    """Create a full sweep+stub-LLM recording without network access."""
    if type(sweep_actions) is not int or sweep_actions <= 0:
        raise ValueError("Sweep action budget must be a positive integer.")
    llm_game = KeyGateGame(seed, defect)
    sweep = run_collision_sweep(
        KeyGateGame(seed, defect),
        max_actions=sweep_actions,
    )
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "hybrid.jsonl"
        with SessionLog(path) as log:
            log.write(
                "session",
                schema_version=2,
                game_version=llm_game.version,
                seed=seed,
                mode="hybrid-benchmark",
                task="explore",
                focus="qa",
                prompt_version=LEAN_PROMPT_VERSION,
                initial_observation=llm_game.observe(),
                hybrid=True,
                sweep_action_budget=sweep_actions,
                limits={
                    "max_actions": 24,
                    "max_requests": 28,
                    "max_seconds": 300,
                    "history_size": 6,
                },
                **defect_evaluation_metadata(defect),
            )
            log.write("sweep_start", source="sweep", action_budget=sweep_actions)
            total = len(sweep.records)
            for index, original in enumerate(sweep.records, 1):
                record = deepcopy(original)
                record["step"] = index - total - 1
                record.setdefault("episode", 0)
                record.setdefault("episode_before", 0)
                record.setdefault("episode_after", 0)
                fact = record.get("software_fact")
                if isinstance(fact, dict):
                    for anomaly in fact.get("anomalies", []):
                        if isinstance(anomaly, dict):
                            anomaly["step"] = record["step"]
                            anomaly["evidence_steps"] = [record["step"]]
                kind = record.pop("type")
                log.write(kind, **record)
            log.write("sweep_summary", source="sweep", summary=sweep.summary)
            run_lean_exploration(
                llm_game,
                _OfflineFinishClient(),
                log,
                llm_game.reset,
                ExplorationLimits(24, 28, 300),
                focus="qa",
                sweep_summary=sweep.summary,
                sleep=lambda _seconds: None,
            )
        records = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        replay(path)
        return records


def _write_case_artifacts(
    record_dir: str | Path,
    *,
    defect: str,
    seed: int,
    records: Sequence[dict],
    report: str,
) -> dict[str, object]:
    """Persist the exact evaluated recording and its ground-truth report."""
    directory = Path(record_dir).expanduser()
    directory.mkdir(parents=True, exist_ok=True)
    stem = f"{defect}-seed{seed}"
    recording_path = directory / f"{stem}.jsonl"
    report_path = directory / f"{stem}.md"
    recording_path.write_text(
        "".join(
            json.dumps(record, separators=(",", ":"), ensure_ascii=False) + "\n"
            for record in records
        ),
        encoding="utf-8",
    )
    report_path.write_text(report, encoding="utf-8")
    return {
        "defect": defect,
        "seed": seed,
        "recording": str(recording_path.resolve()),
        "report": str(report_path.resolve()),
    }


def run_hybrid_benchmark(
    *,
    seeds: Sequence[int] = SCRIPTED_SEEDS,
    defects: Sequence[str] = ("none", "wall-clip"),
    sweep_actions: int = 400,
    record_dir: str | Path | None = None,
) -> dict:
    """Verify collision coverage plus a bounded stub-LLM phase offline."""
    rows: list[dict] = []
    artifacts: list[dict[str, object]] = []
    for defect in defects:
        for seed in seeds:
            records = build_hybrid_stub_recording(
                seed,
                defect,
                sweep_actions=sweep_actions,
            )
            evaluation = evaluate_recording(records)
            issues = extract_issues(records)
            reproductions = [
                reproduce_issue(records, issue["number"])
                for issue in issues
            ]
            if record_dir is not None:
                report = generate_report(
                    records,
                    with_ground_truth=True,
                    reproduction_results=reproductions,
                )
                artifacts.append(_write_case_artifacts(
                    record_dir,
                    defect=defect,
                    seed=seed,
                    records=records,
                    report=report,
                ))
            controls = [
                record for record in records
                if record.get("type") in {"sweep_step", "step", "reset"}
            ]
            phase_counts = {
                source: sum(record.get("source") == source for record in controls)
                for source in ("sweep", "llm")
            }
            rows.append({
                "defect": defect,
                "seed": seed,
                "detection": evaluation["detection"],
                "hit": evaluation["hit"],
                "false_positive_count": evaluation["false_positive_count"],
                "repro_verified": evaluation["repro_verified"],
                "issue_count": evaluation["issue_count"],
                "all_issue_reproductions_verified": all(
                    result["reproduced"] for result in reproductions
                ),
                "action_count": len(controls),
                "phase_control_counts": phase_counts,
                "inference_requests": sum(
                    record.get("type") == "request" for record in records
                ),
            })
    defect_rows = [row for row in rows if row["defect"] != "none"]
    clean_rows = [row for row in rows if row["defect"] == "none"]
    passed = (
        all(row["detection"] == "hit" and row["repro_verified"] for row in defect_rows)
        and all(row["false_positive_count"] == 0 for row in clean_rows)
        and all(row["all_issue_reproductions_verified"] for row in rows)
    )
    return {
        "mode": "hybrid",
        "seeds": list(seeds),
        "defects": list(defects),
        "sweep_actions": sweep_actions,
        "rows": rows,
        "passed": passed,
        "artifacts": artifacts,
    }


def run_scripted_benchmark(
    *,
    seeds: Sequence[int] = SCRIPTED_SEEDS,
    defects: Sequence[str] = DEFECT_CHOICES,
    game_factory: Callable[[int, str | None], object] | None = None,
    sweep_actions: int = 400,
    record_dir: str | Path | None = None,
) -> dict:
    """Exercise report, reproduce, and evaluate for every requested case."""
    if type(sweep_actions) is not int or sweep_actions <= 0:
        raise ValueError("Sweep action budget must be a positive integer.")
    rows: list[dict] = []
    artifacts: list[dict[str, object]] = []
    for defect in defects:
        for seed in seeds:
            records = build_scripted_recording(
                seed,
                defect,
                game_factory=game_factory,
                sweep_actions=sweep_actions,
            )
            issues = extract_issues(records)
            reproductions = [
                reproduce_issue(
                    records,
                    issue["number"],
                    minimize=False,
                    game_factory=game_factory,
                )
                for issue in issues
            ]
            # Generate the user-facing artifact as part of the benchmark even
            # though only its size is retained in the compact result.  Reuse
            # the explicitly verified plans above so a real Unity benchmark
            # launches one reproduction player per issue, not two.
            report = generate_report(
                records,
                with_ground_truth=record_dir is not None,
                minimize=False,
                game_factory=game_factory,
                reproduction_results=reproductions,
            )
            if record_dir is not None:
                artifacts.append(_write_case_artifacts(
                    record_dir,
                    defect=defect,
                    seed=seed,
                    records=records,
                    report=report,
                ))
            # Reproduction is exercised explicitly just above.  Avoid running
            # the reproducer a second time for every matrix cell;
            # that is especially expensive when each attempt launches Unity.
            evaluation = evaluate_recording(
                records,
                verify_reproduction=False,
                game_factory=game_factory,
            )
            reproduced_by_issue = {
                result["issue"]: result.get("reproduced") is True
                for result in reproductions
                if type(result.get("issue")) is int
            }
            matched_numbers = evaluation.get("matched_issue_numbers", [])
            matched_number_set = {
                number for number in matched_numbers if type(number) is int
            }
            matched_titles = [
                issue.get("title")
                for issue in issues
                if issue.get("number") in matched_number_set
                and isinstance(issue.get("title"), str)
            ]
            if matched_numbers:
                repro_verified = all(
                    reproduced_by_issue.get(number, False)
                    for number in matched_numbers
                )
            else:
                # Retain compatibility with small injected benchmark stubs
                # that predate matched_issue_numbers.
                repro_verified = evaluation.get("repro_verified") is True
            rows.append({
                "defect": defect,
                "seed": seed,
                "detection": evaluation["detection"],
                "hit": evaluation["hit"],
                "false_positive_count": evaluation["false_positive_count"],
                "repro_verified": repro_verified,
                "matched_title": (
                    "; ".join(dict.fromkeys(matched_titles))
                    if matched_titles else None
                ),
                "issue_count": evaluation["issue_count"],
                "all_issue_reproductions_verified": all(
                    result["reproduced"] for result in reproductions
                ),
                "action_count": sum(
                    record.get("type") in {
                        "step", "reset", "sweep_step", "sweep_reset",
                    }
                    for record in records
                ),
                "report_bytes": len(report.encode("utf-8")),
            })

    defect_rows = [row for row in rows if row["defect"] != "none"]
    clean_rows = [row for row in rows if row["defect"] == "none"]
    passed = (
        all(row["detection"] == "hit" for row in defect_rows)
        and all(row["repro_verified"] for row in defect_rows)
        and all(row["false_positive_count"] == 0 for row in clean_rows)
        and all(row["all_issue_reproductions_verified"] for row in rows)
    )
    return {
        "mode": "scripted",
        "seeds": list(seeds),
        "defects": list(defects),
        "sweep_actions": sweep_actions,
        "rows": rows,
        "passed": passed,
        "artifacts": artifacts,
    }


def format_benchmark_table(result: dict) -> str:
    """Render the compact defect-by-seed result requested for Phase 2.3."""
    rows: list[dict] = list(result["rows"])
    show_phases = any("phase_control_counts" in row for row in rows)
    show_titles = any("matched_title" in row for row in rows)
    headers = (
        "defect", "seed", "hit/miss", "FP", "repro verified",
        *(("matched title",) if show_titles else ()),
        *(("phase controls",) if show_phases else ()),
    )
    rendered = []
    for row in rows:
        reproduced = "n/a" if row["defect"] == "none" else (
            "yes" if row["repro_verified"] else "no"
        )
        values = [
            str(row["defect"]),
            str(row["seed"]),
            str(row["detection"]),
            str(row["false_positive_count"]),
            reproduced,
        ]
        if show_titles:
            values.append(str(row.get("matched_title") or "n/a"))
        if show_phases:
            counts = row.get("phase_control_counts", {})
            values.append("; ".join(
                f"{name}={counts.get(name, 0)}" for name in ("sweep", "llm")
            ))
        rendered.append(tuple(values))
    widths = [
        max(len(headers[index]), *(len(row[index]) for row in rendered))
        for index in range(len(headers))
    ]

    def line(values: Sequence[str]) -> str:
        return "  ".join(
            value.ljust(widths[index]) for index, value in enumerate(values)
        ).rstrip()

    separator = tuple("-" * width for width in widths)
    status = "PASS" if result["passed"] else "FAIL"
    label = "Hybrid" if result.get("mode") == "hybrid" else "Scripted"
    return "\n".join([
        line(headers),
        line(separator),
        *(line(row) for row in rendered),
        f"{label} benchmark: {status}",
    ])
