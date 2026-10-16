"""Minimal command-line entry points; AI mode always calls Nebius."""

import argparse
from copy import deepcopy
from dataclasses import asdict, replace
import json
from math import isfinite
from pathlib import Path
import sys
import uuid

from .config import ConfigurationError, Settings
from .comparisons import COMPARISON_CONTRACT_VERSION
from .exploration import EXPLORATION_FOCUSES, PROMPT_VERSION as LEGACY_PROMPT_VERSION, ExplorationLimits, exploration_view, run_exploration
from .game import (
    ACTIONS,
    DEFECT_CHOICES,
    KeyGateGame,
    defect_evaluation_metadata as grid_defect_evaluation_metadata,
    render,
)
from .lean import (
    DEFAULT_DIRECT_FALLBACK_MAX_TOKENS,
    LEAN_PROMPT_VERSION,
    run_lean_exploration,
)
from .hybrid import run_collision_sweep
from .nebius import DEFAULT_MAX_TOKENS, NebiusClient, NebiusError
from .measurements import REFLECTION_CONTRACT_VERSION
from .runner import RunLimits, run_agent
from .remote import RemoteGameAdapter
from .session import SessionLog, new_log_path, redact_secrets, replay
from .unity_launcher import (
    DEFAULT_STARTUP_TIMEOUT,
    UNITY_BENCHMARK_DEFECTS,
    UNITY_DEFECT_CHOICES,
    UnityPlayerGame,
    unity_defect_evaluation_metadata,
    unity_game_factory,
)


PREFLIGHT_MAX_TOKENS = 16
PREFLIGHT_LATENCY_LIMIT_MS = 15_000
PREFLIGHT_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {"ok": {"type": "boolean"}},
    "required": ["ok"],
    "additionalProperties": False,
}
PREFLIGHT_MESSAGES = [
    {
        "role": "system",
        "content": 'This is an inference availability probe. Reply only with {"ok":true}.',
    },
    {"role": "user", "content": "Respond now."},
]

HARNESS_DEFECT_CHOICES = tuple(dict.fromkeys(
    (*DEFECT_CHOICES, *UNITY_DEFECT_CHOICES)
))


def positive_integer(value: str) -> int:
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("Must be greater than zero.")
    return number


def positive_number(value: str) -> float:
    try:
        number = float(value)
    except ValueError:
        raise argparse.ArgumentTypeError("Must be a finite number greater than zero.") from None
    if not isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("Must be a finite number greater than zero.")
    return number


def nemotron_model(value: str) -> str:
    if (
        not value.lower().startswith("nvidia/")
        or "nemotron" not in value.lower()
        or any(character.isspace() for character in value)
    ):
        raise argparse.ArgumentTypeError("Use an exact NVIDIA Nemotron model ID from the Nebius catalog.")
    return value


def benchmark_seeds(value: str) -> tuple[int, ...]:
    parts = [part.strip() for part in value.split(",")]
    if not parts or any(not part for part in parts):
        raise argparse.ArgumentTypeError(
            "--seeds requires one or more comma-separated integers."
        )
    try:
        seeds = [int(part) for part in parts]
    except ValueError:
        raise argparse.ArgumentTypeError(
            "--seeds requires one or more comma-separated integers."
        ) from None
    return tuple(dict.fromkeys(seeds))


def benchmark_defects(value: str) -> tuple[str, ...]:
    defects = [part.strip() for part in value.split(",")]
    if not defects or any(not defect for defect in defects):
        raise argparse.ArgumentTypeError(
            "--defects requires one or more comma-separated defect names."
        )
    unknown = [
        defect for defect in defects if defect not in HARNESS_DEFECT_CHOICES
    ]
    if unknown:
        raise argparse.ArgumentTypeError(
            "Unknown benchmark defect(s): " + ", ".join(unknown) + "."
        )
    return tuple(dict.fromkeys(defects))


def exploration_temperature(value: str) -> float:
    try:
        number = float(value)
    except ValueError:
        raise argparse.ArgumentTypeError("Temperature must be a finite number between 0 and 2.") from None
    if not isfinite(number) or not 0 <= number <= 2:
        raise argparse.ArgumentTypeError("Temperature must be a finite number between 0 and 2.")
    return number


def begin_session(log: SessionLog, game, mode: str, *, schema_version: int = 1, **metadata):
    game_metadata = getattr(game, "session_metadata", {})
    if game_metadata is None:
        game_metadata = {}
    if not isinstance(game_metadata, dict):
        raise ValueError("Game session metadata must be an object.")
    conflicts = {
        key for key in game_metadata
        if key in metadata and metadata[key] != game_metadata[key]
    }
    if conflicts:
        raise ValueError(
            "Game session metadata conflicts with explicit fields: "
            + ", ".join(sorted(conflicts))
        )
    combined = {**game_metadata, **metadata}
    log.write("session", schema_version=schema_version, game_version=game.version, seed=game.seed, mode=mode, initial_observation=game.observe(), **combined)


def _defect_evaluation_metadata(defect: str | None) -> dict:
    name = "none" if defect is None else defect
    if name in DEFECT_CHOICES:
        return grid_defect_evaluation_metadata(name)
    return unity_defect_evaluation_metadata(name)


def _evaluation_metadata(args) -> dict:
    """Keep remote ground truth in the local header and off the wire."""
    if getattr(args, "game", None):
        ground_truth = getattr(args, "ground_truth", None)
        return _defect_evaluation_metadata(ground_truth) if ground_truth is not None else {}
    return _defect_evaluation_metadata(getattr(args, "defect", "none"))


def _validate_game_source(args) -> None:
    if getattr(args, "game", None) and getattr(args, "unity_player", None):
        raise ValueError("Choose either --game or --unity-player, not both.")


def _create_game(args):
    _validate_game_source(args)
    endpoint = getattr(args, "game", None)
    player = getattr(args, "unity_player", None)
    if player:
        return UnityPlayerGame(
            player,
            seed=args.seed,
            defect=getattr(args, "defect", "none"),
            timeout=getattr(args, "game_timeout", 5),
            startup_timeout=getattr(
                args, "unity_start_timeout", DEFAULT_STARTUP_TIMEOUT
            ),
        )
    if endpoint:
        if getattr(args, "defect", "none") != "none":
            raise ValueError(
                "--defect configures only the built-in game; configure the server "
                "and use --ground-truth separately for a remote game."
            )
        return RemoteGameAdapter(
            endpoint,
            timeout=getattr(args, "game_timeout", 5),
            seed=args.seed,
        )
    return KeyGateGame(args.seed, args.defect)


def _remote_game_factory(endpoint: str | None, timeout: float = 5.0):
    if not endpoint:
        return None

    def create(seed: int, _defect: str | None):
        return RemoteGameAdapter(endpoint, timeout=timeout, seed=seed)

    return create


def _recording_session_header(path: Path) -> dict:
    try:
        with Path(path).open("r", encoding="utf-8") as source:
            for line in source:
                if not line.strip():
                    continue
                header = json.loads(line)
                if not isinstance(header, dict) or header.get("type") != "session":
                    raise ValueError("Recording must begin with a session record.")
                return header
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not read recording session metadata: {exc}") from None
    raise ValueError("Recording is empty.")


def _recorded_unity_player(path: Path) -> Path | None:
    metadata = _recording_session_header(path).get("unity_player")
    if metadata is None:
        return None
    if not isinstance(metadata, dict):
        raise ValueError("Recording Unity player metadata is malformed.")
    player_path = metadata.get("path")
    version = metadata.get("version")
    if (
        not isinstance(player_path, str)
        or not player_path
        or not isinstance(version, str)
        or not version
    ):
        raise ValueError("Recording Unity player path/version metadata is malformed.")
    return Path(player_path)


def _recording_game_factory(args, paths: list[Path]):
    """Select an explicit endpoint/player or the player recorded in the header."""
    _validate_game_source(args)
    endpoint = getattr(args, "game", None)
    if endpoint:
        return _remote_game_factory(endpoint, args.game_timeout)

    explicit = getattr(args, "unity_player", None)
    if explicit is not None:
        player_path = explicit
    else:
        recorded = [_recorded_unity_player(path) for path in paths]
        player_paths = {
            str(path.expanduser().resolve())
            for path in recorded
            if path is not None
        }
        if player_paths and any(path is None for path in recorded):
            raise ValueError(
                "Cannot evaluate Unity and non-Unity recordings with one game factory."
            )
        if len(player_paths) > 1:
            raise ValueError(
                "Recordings name different Unity players; pass --unity-player "
                "to choose one explicitly."
            )
        player_path = Path(next(iter(player_paths))) if player_paths else None

    if player_path is None:
        return None
    return unity_game_factory(
        player_path,
        timeout=getattr(args, "game_timeout", 5),
        startup_timeout=getattr(
            args, "unity_start_timeout", DEFAULT_STARTUP_TIMEOUT
        ),
    )


def _close_game(game) -> None:
    close = getattr(game, "close", None)
    if callable(close):
        try:
            close()
        except (OSError, ValueError):
            # A final protocol close is best effort and must not replace the
            # completed session result with a teardown error.
            pass


def _write_sweep_phase(log: SessionLog, result) -> None:
    """Persist sweep evidence with non-colliding negative global step IDs."""
    total = len(result.records)
    log.write(
        "sweep_start",
        source="sweep",
        action_budget=result.summary.get("action_budget"),
    )
    for index, original in enumerate(result.records, 1):
        record = deepcopy(original)
        phase_step = record.get("phase_step", index)
        record["phase_step"] = phase_step
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
        kind = record.pop("type", "sweep_step")
        log.write(kind, **record)
    log.write("sweep_summary", source="sweep", summary=result.summary)


def human_play(args) -> int:
    game = _create_game(args)
    path = args.log or new_log_path()
    aliases = {"n": "north", "s": "south", "e": "east", "w": "west", "i": "interact", ".": "wait"}
    controls = list(getattr(game, "controls", ACTIONS))
    try:
        with SessionLog(path) as log:
            begin_session(log, game, "human", **_evaluation_metadata(args))
            if getattr(args, "game", None):
                print("Human-controlled remote game. Enter an advertised control, or q to quit.")
                print("Controls: " + ", ".join(controls))
            else:
                print("Human-controlled Key & Gate. n/s/e/w = move, i = interact, . = wait, q = quit.")
                print("@ player, K key, D gate, / open gate, E exit, # wall. Collect the key and reach E.")
            print(render(game.observe()), flush=True)
            try:
                while game.observe().get("status") != "won":
                    command = input("Action> ").strip().lower()
                    if command in {"q", "quit"}:
                        break
                    action = aliases.get(command, command)
                    if action not in controls:
                        print("Use one of: " + ", ".join(controls) + ", or q.")
                        continue
                    before = game.observe()
                    after = game.act(action)
                    evaluator = getattr(game, "evaluate_transition", None)
                    evaluation = evaluator(before, action, after) if callable(evaluator) else {"moved": False, "events": after.get("events", []), "contract_violations": []}
                    log.write("step", step=after.get("tick"), before=before, decision={"action": action, "reason": "Human input", "plan": ""}, after=after, evaluation=evaluation)
                    print(render(after), flush=True)
            except (EOFError, KeyboardInterrupt):
                print()
            final = game.observe()
            log.write("summary", status="won" if final.get("status") == "won" else "quit", steps=final.get("tick"), inference_requests=0, final_observation=final)
    finally:
        _close_game(game)
    print(f"Recorded human session: {path.resolve()}")
    return 0


def agent_run(args) -> int:
    settings = Settings.load(args.env_file)
    client = NebiusClient(settings, timeout=args.timeout, max_tokens=args.max_tokens)
    game = _create_game(args)
    path = args.log or new_log_path()
    limits = RunLimits(args.max_steps, args.max_requests, args.max_seconds)
    print(f"Model: {settings.model} | endpoint: {settings.base_url}")
    print(f"Thinking: {settings.enable_thinking} | output token cap per request: {args.max_tokens}")
    print(f"Limits: {limits.max_steps} actions, {limits.max_requests} inference requests, {limits.max_seconds}s.")
    print(f"Recording: {path.resolve()}")
    print(render(game.observe()), flush=True)

    def show_step(record: dict):
        decision = record["decision"]
        reason = decision["reason"].replace(settings.api_key, "[REDACTED]")
        reason = "".join(character for character in reason if character.isprintable())
        print(f"\nDecision {record['step']}: {decision['action']} — {reason}")
        print(render(record["after"]), flush=True)

    try:
        with SessionLog(path, secrets=(settings.api_key,)) as log:
            begin_session(
                log,
                game,
                "nebius",
                model=settings.model,
                endpoint=settings.base_url,
                response_format=settings.response_format,
                enable_thinking=settings.enable_thinking,
                max_tokens=args.max_tokens,
                limits=asdict(limits),
                **_evaluation_metadata(args),
            )
            result = run_agent(
                game,
                client,
                log,
                limits,
                on_step=show_step,
                evaluator=game.evaluate_transition,
            )
    finally:
        _close_game(game)
    print("\n" + json.dumps({key: value for key, value in result.items() if key != "final_observation"}, indent=2))
    print(f"Replay: python3 -m autoplay_qa replay '{path.resolve()}'")
    if result["status"] == "won":
        return 0
    if result["status"] == "interrupted":
        return 130
    return 3 if result["status"].endswith("_limit") else 2


def doctor(args) -> int:
    settings = Settings.load(args.env_file)
    client = NebiusClient(settings)
    print("Credential configured (value hidden). Checking the authenticated Nebius model catalog…", flush=True)
    models = client.list_models()
    nemotron_models = [model for model in models if model.lower().startswith("nvidia/") and "nemotron" in model.lower()]
    if args.list_models:
        print("Accessible NVIDIA Nemotron model IDs:")
        for model in nemotron_models:
            print("  " + model)
    if settings.model not in models:
        raise ConfigurationError(f"Configured model {settings.model!r} is absent from your live catalog. Run doctor --list-models and set NEBIUS_MODEL to an exact listed ID.")
    print(f"Catalog access works; configured model is listed: {settings.model}")
    if args.probe_inference:
        probe = probe_inference(client)
        print(
            "Inference probe works; "
            f"latency: {probe['latency_ms'] / 1000:.3f}s | "
            f"finish_reason: {probe['finish_reason']}"
        )
    else:
        print("This checks authentication and model listing only. Add --probe-inference to make one tiny direct-mode completion request.")
    return 0


def probe_inference(client: NebiusClient) -> dict:
    """Make one small direct-mode completion request and return safe diagnostics."""
    completion = client.complete(
        PREFLIGHT_MESSAGES,
        response_schema=PREFLIGHT_RESPONSE_SCHEMA,
        enable_thinking=False,
        max_tokens=PREFLIGHT_MAX_TOKENS,
    )
    latency_ms = completion.metadata.get("latency_ms")
    if type(latency_ms) is not int or latency_ms < 0:
        raise NebiusError("Nebius inference probe returned no valid latency measurement.")
    return {
        "latency_ms": latency_ms,
        "finish_reason": completion.metadata.get("finish_reason"),
    }


def exploration_exit_code(status: str) -> int:
    if status == "interrupted":
        return 130
    return 0 if status in {"finished", "action_limit", "request_limit", "time_limit"} else 2


def explore(args) -> int:
    from .adaptive_protocol import ADAPTIVE_VERSIONS, context_class

    adaptive_mode = args.mode in ADAPTIVE_VERSIONS
    if getattr(args, "memory_db", None) and (not adaptive_mode or not getattr(args, "project_id", None)):
        print("Error: --memory-db requires an adaptive mode and an explicit --project-id.", file=sys.stderr)
        return 2
    if getattr(args, "memory_read_only", False) and not getattr(args, "memory_db", None):
        print("Error: --memory-read-only requires --memory-db.", file=sys.stderr)
        return 2
    if args.preflight and args.mode != "lean-v1":
        print("Error: --preflight is supported only with --mode lean-v1.", file=sys.stderr)
        return 2
    if args.hybrid and (args.mode != "lean-v1" or args.focus != "qa"):
        print(
            "Error: --hybrid requires --mode lean-v1 and --focus qa.",
            file=sys.stderr,
        )
        return 2
    settings = Settings.load(args.env_file)
    # Preserve the Lightning sampling default while allowing bounded comparison
    # pilots. The working playthrough keeps its existing sampling settings.
    client = NebiusClient(settings, timeout=args.timeout, max_tokens=args.max_tokens, temperature=args.temperature, top_p=0.95)
    limits = ExplorationLimits(args.max_actions, args.max_requests, args.max_seconds)
    path = args.log or new_log_path()
    print(f"Exploration model: {settings.model} | thinking: {settings.enable_thinking}")
    print(f"Session question: {EXPLORATION_FOCUSES[args.focus]}")
    print(f"Limits: {limits.max_actions} actions including resets, {limits.max_requests} requests, {limits.max_seconds}s.")

    # Finish the sweep connection before opening the LLM connection. Besides
    # making the phase boundary explicit, this supports runtimes (including the
    # Unity skeleton) that intentionally serve one protocol client at a time.
    sweep = None
    if args.hybrid:
        print(
            f"Hybrid collision sweep: up to {args.sweep_actions} offline actions; "
            "the LLM limits above remain unchanged."
        )
        sweep_game = _create_game(args)
        try:
            sweep = run_collision_sweep(
                sweep_game,
                max_actions=args.sweep_actions,
            )
        finally:
            _close_game(sweep_game)
        print(
            "Sweep summary: "
            + json.dumps(sweep.summary, separators=(",", ":")),
            flush=True,
        )

    if args.preflight:
        try:
            probe = probe_inference(client)
        except NebiusError as exc:
            print(f"Preflight failed; exploration was not started: {exc}", file=sys.stderr)
            return 4
        latency_ms = probe["latency_ms"]
        print(
            "Inference preflight: "
            f"latency {latency_ms / 1000:.3f}s | "
            f"finish_reason {probe['finish_reason']}"
        )
        if latency_ms > PREFLIGHT_LATENCY_LIMIT_MS:
            print(
                "Preflight aborted exploration: provider latency exceeded 15 seconds; "
                "no recording was created.",
                file=sys.stderr,
            )
            return 4
    game = _create_game(args)
    print(f"Recording: {path.resolve()}")
    def show_observation(observation):
        if adaptive_mode:
            from .adaptive import compact_observation
            from .experience import public_observation
            return json.dumps(compact_observation(public_observation(observation, secrets=(settings.api_key,))), ensure_ascii=False)
        return render(exploration_view(observation))

    print(show_observation(game.observe()), flush=True)

    def safe(text: str) -> str:
        return text.replace(settings.api_key, "[REDACTED]")

    def show_legacy_action(record: dict):
        decision = record["decision"]
        forecast = json.dumps(decision["expected_outcome"], separators=(",", ":"))
        text = safe(f"Experiment {decision['experiment_id']}: {decision['experiment_goal']}\nAction {record['step']}: {decision['action']} ({decision['action_role']}, target {decision['experiment_control']}) — {decision['reason']}\nForecast (model state): {forecast}\nReadable forecast ({record['prediction_source']}): {record['prediction']}")
        print("\n" + "".join(character for character in text if character.isprintable() or character == "\n"))
        print(show_observation(record["after"]), flush=True)

    def show_lean_action(record: dict):
        decision = record["decision"]
        fact = record["software_fact"]
        text = safe(
            f"Action {record['step']}: {decision['action']} — {decision['reason']}\n"
            f"Expected position: {decision['expected_position']} | matched: {fact['expected_position_matched']}\n"
            f"Actual changed facets: {fact['actual_changed_facets']}"
        )
        print("\n" + "".join(character for character in text if character.isprintable() or character == "\n"))
        print(show_observation(record["after"]), flush=True)

    sweep_summary = sweep.summary if sweep is not None else None
    memory_store = None
    adaptive = None
    adaptive_run_id = "adaptive-" + uuid.uuid4().hex if adaptive_mode else None
    try:
        if adaptive_mode:
            from .experience import ExperienceStore

            if args.memory_db:
                memory_store = ExperienceStore(args.memory_db, project_id=args.project_id,
                                               game_version=game.version, initial_observation=game.observe(),
                                               read_only=args.memory_read_only, secrets=(settings.api_key,))
            adaptive = context_class(args.mode)(game.observe(), store=memory_store, run_id=adaptive_run_id,
                                                max_prompt_bytes=args.prompt_max_bytes, secrets=(settings.api_key,))
        with SessionLog(path, secrets=(settings.api_key,)) as log:
            common = dict(
                schema_version=2, task="explore", focus=args.focus,
                model=settings.model, endpoint=settings.base_url,
                response_format=settings.response_format,
                enable_thinking=settings.enable_thinking, max_tokens=args.max_tokens,
                temperature=client.temperature, top_p=client.top_p, limits=asdict(limits),
                **_evaluation_metadata(args),
            )
            if args.hybrid:
                common.update(hybrid=True, sweep_action_budget=args.sweep_actions)
            if args.mode == "legacy-v27":
                begin_session(
                    log, game, "nebius", prompt_version=LEGACY_PROMPT_VERSION,
                    comparison_contract=COMPARISON_CONTRACT_VERSION,
                    reflection_contract=REFLECTION_CONTRACT_VERSION, **common,
                )
                result = run_exploration(
                    game, client, log, game.reset, limits,
                    on_action=show_legacy_action, evaluator=game.evaluate_transition, focus=args.focus,
                    comparison_contract=True, reflection_contract=REFLECTION_CONTRACT_VERSION,
                )
            else:
                begin_session(
                    log, game, "nebius", prompt_version=args.mode if adaptive_mode else LEAN_PROMPT_VERSION,
                    direct_fallback_max_tokens=args.direct_fallback_max_tokens,
                    **({"adaptive_config": {"run_id": adaptive_run_id, "max_prompt_bytes": args.prompt_max_bytes,
                                            **({"request_timeout": args.timeout} if args.mode == "adaptive-v4" else {})}}
                       if adaptive_mode else {}),
                    **common,
                )
                if sweep is not None:
                    _write_sweep_phase(log, sweep)
                result = run_lean_exploration(
                    game, client, log, game.reset, limits,
                    on_action=show_lean_action, evaluator=game.evaluate_transition, focus=args.focus,
                    direct_fallback_max_tokens=args.direct_fallback_max_tokens,
                    sweep_summary=sweep_summary,
                    adaptive=adaptive,
                )
    finally:
        _close_game(game)
        if memory_store is not None:
            memory_store.close()
    # Everything below is computed from the recorded session, without a model call.
    concise = {key: result[key] for key in ("status", "partial", "steps", "game_actions", "resets", "inference_requests", "completed_inference_responses", "usage_complete", "rejected_decisions", "usage", "elapsed_seconds", "coverage", "hypotheses", "unassessed_final_step", "unassessed_final_outcome", "pilot_observations", "ready_for_live_review", "review_note", "learning_claims")}
    if "error" in result:
        concise["error"] = result["error"]
    if args.mode == "legacy-v27":
        concise["experiment_bindings"] = result["experiment_bindings"]
        concise["structured_predictions"] = {
            "matched": sum(item["prediction_verdict"] == "matched" for item in result["reflection_receipts"]),
            "mismatched": sum(item["prediction_verdict"] == "mismatched" for item in result["reflection_receipts"]),
            "assessed": sum(item["assessed"] for item in result["reflection_receipts"]),
            "scope": "Five public state facets from the sole model forecast; readable text is software-rendered. Assessment prose and hypothesis truth remain unverified.",
        }
        concise["research_context"] = result["research_context"]
        concise["evidence_measurements"] = result["evidence_measurements"]
        concise["experiment_progress"] = result["experiment_progress"]
    else:
        concise["warning_count"] = result["warning_count"]
        concise["direct_fallbacks"] = result["direct_fallbacks"]
        if "retries" in result:
            concise["retries"] = result["retries"]
        concise["request_prompt_bytes"] = result["request_prompt_bytes"]
        concise["cross_episode_same_start_pairs"] = result["cross_episode_same_start_pairs"]
        concise["retention_facts"] = result["retention_facts"]
        if "anomalies" in result:
            concise["anomalies"] = result["anomalies"]
        if args.hybrid:
            concise["sweep_coverage"] = result.get("sweep_coverage", sweep_summary)
    print("\nLearning summary:\n" + json.dumps(redact_secrets(concise, (settings.api_key,)), indent=2))
    from .review import review_session

    review = review_session(path, game_factory=(lambda: _create_game(args)) if adaptive_mode else None)
    latency = review["requests"]["latency_ms"]["median"]
    checks = {
        "execution_replayed": review["execution_replayed"],
        "experiments": review["coverage"]["experiment_count"],
        "movement": review["coverage"]["movement"], "interaction": review["coverage"]["interaction"],
        "evidence_linked_notes": review["learning"]["evidence_linked_hypotheses"],
        "memory_supplied_after_reset": review["reset_learning"]["memory_supplied_after_reset"],
        "assessed_retest_pairs": len(review["reset_learning"]["retest_pairs"]),
        "median_request_seconds": round(latency / 1000, 3) if latency is not None else None,
        "hypothesis_accuracy_requires_review": review["review_needed"],
        "milestone_observations": review["goal_checks"],
    }
    print("Recorded goal checks: " + json.dumps(checks))
    print(f"Replay: python3 -m autoplay_qa replay '{path.resolve()}'")
    print(f"Goal review: python3 -m autoplay_qa review '{path.resolve()}'")
    return exploration_exit_code(result["status"])


def review_recording(args) -> int:
    from .review import review_session

    report = review_session(args.path, include_evidence=True) if getattr(args, "evidence", False) else review_session(args.path)
    print(json.dumps(report, indent=2))
    return 0


def report_recording(args) -> int:
    from .reporting import generate_report, write_report

    factory = _recording_game_factory(args, [args.path])
    if args.out is None:
        print(generate_report(
            args.path,
            with_ground_truth=args.with_ground_truth,
            game_factory=factory,
        ), end="")
    else:
        destination = write_report(
            args.path,
            args.out,
            with_ground_truth=args.with_ground_truth,
            game_factory=factory,
        )
        print(f"Wrote bug report: {destination}")
    return 0


def _triage_source_counts(records: list[dict]) -> dict:
    phases = {
        "llm": {"game_actions": 0, "resets": 0, "controls": 0},
        "sweep": {"game_actions": 0, "resets": 0, "controls": 0},
    }
    for record in records:
        kind = record.get("type")
        if kind not in {"step", "reset", "sweep_step", "sweep_reset"}:
            continue
        phase = "sweep" if kind.startswith("sweep_") or record.get("source") == "sweep" else "llm"
        phases[phase]["resets" if kind in {"reset", "sweep_reset"} else "game_actions"] += 1
        phases[phase]["controls"] += 1
    return {
        "actions": sum(phase["controls"] for phase in phases.values()),
        "game_actions": sum(phase["game_actions"] for phase in phases.values()),
        "resets": sum(phase["resets"] for phase in phases.values()),
        "inference_requests": sum(record.get("type") == "request" for record in records),
        **phases,
    }


def triage_recording(args) -> int:
    from .reporting import load_recording, reproduce_issue
    from .triage import (
        DEFAULT_TRIAGE_INPUT_PRICE,
        DEFAULT_TRIAGE_MODEL,
        DEFAULT_TRIAGE_OUTPUT_PRICE,
        TriageLimits,
        TriagePricing,
        build_evidence_packets,
        render_triage_report,
        run_triage,
    )

    _validate_game_source(args)
    if (args.game or args.unity_player) and not args.verify_repro:
        raise ValueError("--game and --unity-player are used by triage only with --verify-repro.")
    if args.model != DEFAULT_TRIAGE_MODEL and (
        args.input_price is None or args.output_price is None
    ):
        raise ValueError(
            "An alternate --model requires both --input-price-per-million and "
            "--output-price-per-million in USD for bounded cost accounting."
        )
    limits = TriageLimits(
        max_requests=args.max_requests,
        max_seconds=args.max_seconds,
        timeout=args.timeout,
        max_tokens=args.max_tokens,
        max_usd=args.max_usd,
    )
    pricing = TriagePricing(
        input_usd_per_million=(
            DEFAULT_TRIAGE_INPUT_PRICE if args.input_price is None else args.input_price
        ),
        output_usd_per_million=(
            DEFAULT_TRIAGE_OUTPUT_PRICE if args.output_price is None else args.output_price
        ),
    )
    source = args.path.resolve()
    report_path = (args.out or args.path.with_suffix(".triage.md")).resolve()
    json_path = (args.json_out or args.path.with_suffix(".triage.json")).resolve()
    log_path = (args.log or new_log_path(Path("runs") / "triage")).resolve()
    destinations = (report_path, json_path, log_path)
    if source in destinations or len(set(destinations)) != len(destinations):
        raise ValueError("The recording, Markdown report, JSON result, and audit log must use different paths.")
    existing = [path for path in (source, *destinations) if path.exists()]
    if any(path.samefile(other) for index, path in enumerate(existing) for other in existing[index + 1:]):
        raise ValueError("The recording and triage artifacts must refer to different files.")
    if log_path.exists():
        raise ValueError("The triage audit log already exists; choose a new --log path.")
    for destination in destinations:
        if destination.is_dir():
            raise ValueError("A triage output path names a directory; choose a file path.")
        destination.parent.mkdir(parents=True, exist_ok=True)

    records = load_recording(args.path)
    packets = build_evidence_packets(records)
    if args.verify_repro and packets:
        factory = _recording_game_factory(args, [args.path])
        reproduction_results = {}
        for packet in packets:
            number = packet["issue_number"]
            reproduction_results[number] = reproduce_issue(
                records,
                number,
                minimize=False,
                game_factory=factory,
            )
        packets = build_evidence_packets(
            records,
            reproduction_results=reproduction_results,
        )

    settings = None
    client = None
    if packets and not args.dry_run:
        settings = replace(Settings.load(args.env_file), model=args.model)
        client = NebiusClient(settings, timeout=args.timeout, max_tokens=args.max_tokens, temperature=1.0, top_p=0.95)
    secrets = (settings.api_key,) if settings is not None else ()
    source_counts = _triage_source_counts(records)
    with SessionLog(log_path, secrets=secrets) as log:
        log.write(
            "triage_source",
            source_recording=str(source),
            source_counts=source_counts,
            model=args.model,
            dry_run=args.dry_run,
            verify_repro=args.verify_repro,
            limits=asdict(limits),
            pricing=asdict(pricing),
        )
        result = run_triage(packets, client, limits=limits, pricing=pricing, log=log)
        result["model"] = args.model
        result["source_counts"] = source_counts
        result = redact_secrets(result, secrets)
        log.write("triage_artifacts", report=str(report_path), json_result=str(json_path))
        json_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        report_path.write_text(render_triage_report(result), encoding="utf-8")

    summary = {key: value for key, value in result.items() if key != "results"}
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Triage report: {redact_secrets(str(report_path), secrets)}")
    print(f"Triage JSON: {redact_secrets(str(json_path), secrets)}")
    print(f"Triage audit: {redact_secrets(str(log_path), secrets)}")
    if result["status"] == "interrupted":
        return 130
    return 0 if result["status"] in {"completed", "dry_run", "no_candidates"} else 3


def reproduce_recording(args) -> int:
    from .reporting import reproduce_issue

    result = reproduce_issue(
        args.path,
        args.issue,
        game_factory=_recording_game_factory(args, [args.path]),
    )
    print(json.dumps(result, indent=2))
    return result["exit_code"]


def evaluate_recordings(args) -> int:
    from .reporting import evaluate_recordings as evaluate

    print(json.dumps(evaluate(
        args.paths,
        game_factory=_recording_game_factory(args, args.paths),
    ), indent=2))
    return 0


def scoreboard_recordings(args) -> int:
    from .scoreboard import generate_scoreboard, write_scoreboard

    include_baseline = not args.no_scripted_baseline
    factory = _recording_game_factory(args, args.paths)
    if args.out is None:
        print(generate_scoreboard(
            args.paths,
            include_scripted_baseline=include_baseline,
            game_factory=factory,
        ), end="")
    else:
        destination = write_scoreboard(
            args.paths,
            args.out,
            include_scripted_baseline=include_baseline,
            game_factory=factory,
        )
        print(f"Wrote QA scoreboard: {destination}")
    return 0


def benchmark(args) -> int:
    from .benchmark import (
        format_benchmark_table,
        run_hybrid_benchmark,
        run_scripted_benchmark,
    )

    if args.scripted == args.hybrid:
        raise ValueError("Choose exactly one benchmark mode: --scripted or --hybrid.")
    if args.hybrid:
        if args.game or args.unity_player:
            raise ValueError("The stub-LLM hybrid benchmark uses the built-in offline game.")
        defects = args.defects
        if defects is not None:
            unsupported = [
                defect for defect in defects
                if defect not in {"none", "wall-clip"}
            ]
            if unsupported:
                raise ValueError(
                    "The hybrid benchmark supports only: none, wall-clip."
                )
        options = {"sweep_actions": args.sweep_actions}
        if args.seeds is not None:
            options["seeds"] = args.seeds
        if defects is not None:
            options["defects"] = defects
        if args.record_dir is not None:
            options["record_dir"] = args.record_dir
        result = run_hybrid_benchmark(**options)
        print(format_benchmark_table(result))
        _print_benchmark_artifacts(result)
        return 0 if result["passed"] else 1
    _validate_game_source(args)
    common_options = {}
    if args.seeds is not None:
        common_options["seeds"] = args.seeds
    if args.record_dir is not None:
        common_options["record_dir"] = args.record_dir
    if args.game:
        defect = args.ground_truth
        if defect is None:
            raise ValueError(
                "Remote benchmarks require --ground-truth "
                "(use 'none' for a clean server)."
            )
        if args.defects is not None and args.defects != (defect,):
            raise ValueError(
                "A remote benchmark's --defects filter must contain only its "
                "--ground-truth value."
            )
        result = run_scripted_benchmark(
            defects=(defect,),
            game_factory=_remote_game_factory(args.game, args.game_timeout),
            sweep_actions=args.sweep_actions,
            **common_options,
        )
    elif args.unity_player:
        if args.defects is not None and args.ground_truth is not None:
            raise ValueError("Choose either --defects or --ground-truth, not both.")
        defects = (
            args.defects
            if args.defects is not None
            else (
                (args.ground_truth,)
                if args.ground_truth is not None
                else UNITY_BENCHMARK_DEFECTS
            )
        )
        unsupported = [
            defect for defect in defects if defect not in UNITY_DEFECT_CHOICES
        ]
        if unsupported:
            raise ValueError(
                "The Unity benchmark does not support: "
                + ", ".join(unsupported)
                + "."
            )
        result = run_scripted_benchmark(
            defects=defects,
            game_factory=unity_game_factory(
                args.unity_player,
                timeout=args.game_timeout,
                startup_timeout=args.unity_start_timeout,
            ),
            sweep_actions=args.sweep_actions,
            **common_options,
        )
    else:
        defects = args.defects
        if defects is not None:
            unsupported = [
                defect for defect in defects if defect not in DEFECT_CHOICES
            ]
            if unsupported:
                raise ValueError(
                    "The built-in benchmark does not support: "
                    + ", ".join(unsupported)
                    + "."
                )
            common_options["defects"] = defects
        if args.sweep_actions != 400:
            common_options["sweep_actions"] = args.sweep_actions
        result = run_scripted_benchmark(**common_options)
    print(format_benchmark_table(result))
    _print_benchmark_artifacts(result)
    return 0 if result["passed"] else 1


def _print_benchmark_artifacts(result: dict) -> None:
    for artifact in result.get("artifacts", []):
        recording = artifact.get("recording")
        report = artifact.get("report")
        if isinstance(recording, str):
            print(f"Recording: {recording}")
        if isinstance(report, str):
            print(f"Report: {report}")


def serve(args) -> int:
    from .server import serve_stdio, serve_tcp

    if args.stdio:
        if args.port is not None:
            raise ValueError("Choose either --stdio or --port, not both.")
        serve_stdio(seed=args.seed, defect=args.defect)
    else:
        if args.port is None:
            raise ValueError("serve requires --port P or --stdio.")
        serve_tcp(args.port, seed=args.seed, defect=args.defect)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="AutoPlay QA: a minimal game agent powered by NVIDIA Nemotron through Nebius.")
    commands = parser.add_subparsers(dest="command", required=True)

    def add_game_endpoint(command_parser, *, ground_truth: bool = False):
        command_parser.add_argument(
            "--game",
            metavar="tcp://127.0.0.1:PORT",
            help="Use a localhost protocol server instead of the built-in game.",
        )
        command_parser.add_argument(
            "--unity-player",
            type=Path,
            metavar="PATH",
            help=(
                "Launch and manage a headless Unity player instead of the "
                "built-in game. Report/reproduce/evaluate also reuse a path "
                "stored in the recording when this option is omitted."
            ),
        )
        command_parser.add_argument(
            "--game-timeout",
            type=positive_integer,
            default=5,
            help="Remote game request timeout in seconds (default: 5).",
        )
        command_parser.add_argument(
            "--unity-start-timeout",
            type=positive_integer,
            default=int(DEFAULT_STARTUP_TIMEOUT),
            help="Seconds to wait for a launched Unity player (default: 10).",
        )
        if ground_truth:
            command_parser.add_argument(
                "--ground-truth",
                choices=HARNESS_DEFECT_CHOICES,
                default=None,
                help=(
                    "Harness-only evaluation label for a remote game or a "
                    "single-case Unity benchmark; never sent over the protocol "
                    "or to the model."
                ),
            )

    play_parser = commands.add_parser("play", help="Play the interactive terminal game yourself; no inference.")
    play_parser.add_argument("--seed", type=int, default=0)
    play_parser.add_argument("--defect", choices=HARNESS_DEFECT_CHOICES, default="none")
    play_parser.add_argument("--log", type=Path)
    add_game_endpoint(play_parser, ground_truth=True)
    play_parser.set_defaults(handler=human_play)

    run_parser = commands.add_parser("run", help="Run observation-driven actions using real Nebius inference.")
    run_parser.add_argument("--env-file", type=Path, default=Path(".env"))
    run_parser.add_argument("--seed", type=int, default=0)
    run_parser.add_argument("--defect", choices=HARNESS_DEFECT_CHOICES, default="none")
    run_parser.add_argument("--max-steps", type=positive_integer, default=24)
    run_parser.add_argument("--max-requests", type=positive_integer, default=28)
    run_parser.add_argument("--max-seconds", type=positive_integer, default=300)
    run_parser.add_argument("--max-tokens", type=positive_integer, default=DEFAULT_MAX_TOKENS)
    run_parser.add_argument("--timeout", type=positive_integer, default=45)
    run_parser.add_argument("--log", type=Path)
    add_game_endpoint(run_parser, ground_truth=True)
    run_parser.set_defaults(handler=agent_run)

    explore_parser = commands.add_parser("explore", help="Let Nemotron choose experiments and learn mechanics from outcomes.")
    explore_parser.add_argument("--env-file", type=Path, default=Path(".env"))
    explore_parser.add_argument("--seed", type=int, default=0)
    explore_parser.add_argument("--defect", choices=HARNESS_DEFECT_CHOICES, default="none")
    explore_parser.add_argument("--focus", choices=EXPLORATION_FOCUSES, default="mechanics", help="Session question; the model still chooses every experiment and control.")
    explore_parser.add_argument("--mode", choices=("lean-v1", "adaptive-v1", "adaptive-v2", "adaptive-v3", "adaptive-v4", "legacy-v27"), default="lean-v1",
                                help="Exploration protocol (default: lean-v1; legacy-v27 is retained for compatibility experiments).")
    explore_parser.add_argument("--memory-db", type=Path, help="Private durable observed-transition database for adaptive exploration.")
    explore_parser.add_argument("--project-id", help="Explicit project namespace for persistent experience isolation.")
    explore_parser.add_argument("--memory-read-only", action="store_true", help="Use a frozen experience snapshot for evaluation.")
    explore_parser.add_argument("--prompt-max-bytes", type=positive_integer, default=24_000,
                                help="Adaptive full input byte cap including response schema (default: 24000).")
    explore_parser.add_argument("--max-actions", type=positive_integer, default=24)
    explore_parser.add_argument("--max-requests", type=positive_integer, default=28)
    explore_parser.add_argument("--max-seconds", type=positive_integer, default=300)
    explore_parser.add_argument("--max-tokens", type=positive_integer, default=DEFAULT_MAX_TOKENS)
    explore_parser.add_argument(
        "--direct-fallback-max-tokens", type=positive_integer,
        default=DEFAULT_DIRECT_FALLBACK_MAX_TOKENS,
        help="Lean-v1 output cap for thinking-off repair and time-pressure requests (default: 1536).",
    )
    explore_parser.add_argument("--timeout", type=positive_integer, default=45)
    explore_parser.add_argument("--temperature", type=exploration_temperature, default=1.0,
                                help="Exploration sampling temperature from 0 to 2 (default: 1.0).")
    explore_parser.add_argument(
        "--preflight", action="store_true",
        help="Before lean-v1 exploration, make one tiny direct-mode inference probe and abort if it errors or takes over 15 seconds.",
    )
    explore_parser.add_argument(
        "--hybrid",
        action="store_true",
        help="For QA focus, run an inference-free collision sweep before the unchanged LLM session.",
    )
    explore_parser.add_argument(
        "--sweep-actions",
        type=positive_integer,
        default=400,
        help="Maximum scripted collision-sweep controls in hybrid mode (default: 400).",
    )
    explore_parser.add_argument("--log", type=Path)
    add_game_endpoint(explore_parser, ground_truth=True)
    explore_parser.set_defaults(handler=explore)

    doctor_parser = commands.add_parser("doctor", help="Check credentials and the live model catalog; inference probing is opt-in.")
    doctor_parser.add_argument("--env-file", type=Path, default=Path(".env"))
    doctor_parser.add_argument("--list-models", action="store_true")
    doctor_parser.add_argument(
        "--probe-inference", action="store_true",
        help="Also make exactly one tiny direct-mode completion request and print its latency and finish reason.",
    )
    doctor_parser.set_defaults(handler=doctor)

    replay_parser = commands.add_parser("replay", help="Replay recorded game actions and check observations; no inference.")
    replay_parser.add_argument("path", type=Path)
    replay_parser.set_defaults(handler=lambda args: print(json.dumps(replay(args.path), indent=2)) or 0)
    review_parser = commands.add_parser("review", help="Review recorded goal progress, reset evidence, and latency; no inference.")
    review_parser.add_argument("path", type=Path)
    review_parser.add_argument("--evidence", action="store_true", help="Show hypotheses beside their actual cited controls and states; no inference.")
    review_parser.set_defaults(handler=review_recording)

    report_parser = commands.add_parser("report", help="Create offline Markdown bug reports from suspected notes and anomaly facts.")
    report_parser.add_argument("path", type=Path, metavar="RECORDING")
    report_parser.add_argument("--out", type=Path, metavar="FILE.md")
    report_parser.add_argument("--with-ground-truth", action="store_true", help="Include hidden injected-defect metadata in the report.")
    add_game_endpoint(report_parser)
    report_parser.set_defaults(handler=report_recording)

    from .triage import DEFAULT_TRIAGE_MODEL

    triage_parser = commands.add_parser("triage", help="Have Nemotron Ultra assess recorded candidate issues with bounded inference.")
    triage_parser.add_argument("path", type=Path, metavar="RECORDING")
    triage_parser.add_argument("--out", type=Path, metavar="FILE.md", help="Markdown destination (default: RECORDING.triage.md).")
    triage_parser.add_argument("--json-out", type=Path, metavar="FILE.json", help="Structured result destination (default: RECORDING.triage.json).")
    triage_parser.add_argument("--log", type=Path, metavar="FILE.jsonl", help="New audit log destination (default: a unique file in runs/triage).")
    triage_parser.add_argument("--model", type=nemotron_model, default=DEFAULT_TRIAGE_MODEL, help="Exact Nebius model ID (default: Nemotron 3 Ultra).")
    triage_parser.add_argument("--env-file", type=Path, default=Path(".env"))
    triage_parser.add_argument("--dry-run", action="store_true", help="Write evidence packets and pending reports without loading credentials or making inference requests.")
    triage_parser.add_argument("--verify-repro", action="store_true", help="Replay each issue's recorded prefix once before review; no minimization or inference for verification.")
    triage_parser.add_argument("--max-requests", type=positive_integer, default=8)
    triage_parser.add_argument("--max-seconds", type=positive_number, default=300)
    triage_parser.add_argument("--timeout", type=positive_number, default=45)
    triage_parser.add_argument("--max-tokens", type=positive_integer, default=DEFAULT_MAX_TOKENS)
    triage_parser.add_argument("--max-usd", type=positive_number, default=1.0, help="Conservative per-command inference estimate cap in USD (default: 1.00).")
    triage_parser.add_argument("--input-price-per-million", "--input-price", dest="input_price", type=positive_number, metavar="USD", help="Estimated USD per million input tokens; required for alternate models.")
    triage_parser.add_argument("--output-price-per-million", "--output-price", dest="output_price", type=positive_number, metavar="USD", help="Estimated USD per million output tokens; required for alternate models.")
    add_game_endpoint(triage_parser)
    triage_parser.set_defaults(handler=triage_recording)

    reproduce_parser = commands.add_parser("reproduce", help="Re-execute and minimize one recorded issue; no inference.")
    reproduce_parser.add_argument("path", type=Path, metavar="RECORDING")
    reproduce_parser.add_argument("--issue", type=positive_integer, required=True, metavar="N")
    add_game_endpoint(reproduce_parser)
    reproduce_parser.set_defaults(handler=reproduce_recording)

    evaluate_parser = commands.add_parser("evaluate", help="Compare recorded issues with hidden injected-defect metadata; no inference.")
    evaluate_parser.add_argument("paths", type=Path, nargs="+", metavar="RECORDING")
    add_game_endpoint(evaluate_parser)
    evaluate_parser.set_defaults(handler=evaluate_recordings)

    scoreboard_parser = commands.add_parser("scoreboard", help="Create an offline Markdown QA results table from recordings.")
    scoreboard_parser.add_argument("paths", type=Path, nargs="+", metavar="RECORDING")
    scoreboard_parser.add_argument("--out", "--output", dest="out", type=Path, metavar="FILE.md")
    scoreboard_parser.add_argument(
        "--no-scripted-baseline", action="store_true",
        help="Omit the aggregate seed-0 scripted benchmark comparison row.",
    )
    add_game_endpoint(scoreboard_parser)
    scoreboard_parser.set_defaults(handler=scoreboard_recordings)

    benchmark_parser = commands.add_parser("benchmark", help="Run deterministic offline QA pipeline coverage; no inference.")
    benchmark_parser.add_argument("--scripted", action="store_true", help="Probe seeds 0, 1, and 5 across the clean game and all injected defects.")
    benchmark_parser.add_argument(
        "--hybrid",
        action="store_true",
        help="Run the collision sweep plus a one-decision stub LLM on clean and wall-clip seeds 0, 1, and 5.",
    )
    benchmark_parser.add_argument(
        "--sweep-actions",
        type=positive_integer,
        default=400,
        help=(
            "Maximum scripted sweep controls for a continuous --scripted or "
            "stub --hybrid benchmark (default: 400)."
        ),
    )
    benchmark_parser.add_argument(
        "--defects",
        type=benchmark_defects,
        metavar="NAME[,NAME...]",
        help="Run only the listed injected-defect cases.",
    )
    benchmark_parser.add_argument(
        "--seeds",
        type=benchmark_seeds,
        metavar="N[,N...]",
        help="Run only the listed integer seeds.",
    )
    benchmark_parser.add_argument(
        "--record-dir",
        type=Path,
        metavar="DIR",
        help=(
            "Write each case's complete JSONL recording and ground-truth "
            "Markdown report to DIR."
        ),
    )
    add_game_endpoint(benchmark_parser, ground_truth=True)
    benchmark_parser.set_defaults(handler=benchmark)

    serve_parser = commands.add_parser("serve", help="Expose the reference KeyGate game over the local JSON-lines protocol.")
    serve_parser.add_argument("--port", type=positive_integer)
    serve_parser.add_argument("--stdio", action="store_true", help="Serve one protocol session on stdin/stdout instead of TCP.")
    serve_parser.add_argument("--seed", type=int, default=0)
    serve_parser.add_argument("--defect", choices=DEFECT_CHOICES, default="none")
    serve_parser.set_defaults(handler=serve)
    args = parser.parse_args(argv)
    try:
        return args.handler(args)
    except (ConfigurationError, NebiusError, OSError, ValueError, KeyError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
