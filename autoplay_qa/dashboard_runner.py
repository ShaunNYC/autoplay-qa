"""Bounded Mac runner for the hosted dashboard. Provider keys stay local.

The HTTPS site can request only a locally allowlisted game/profile/seed. A
durable receipt prevents a previously claimed job from being run twice, and
both request reservations and the session allowance survive process restarts.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import asdict, replace
from datetime import datetime, timezone
import fcntl
import json
import math
import os
from pathlib import Path
import re
import signal
import threading
import time
from typing import Callable
import urllib.error
import urllib.request
from urllib.parse import urlsplit
import uuid

from .cli import _close_game, _write_sweep_phase, begin_session
from .config import DEFAULT_MODEL, Settings
from .dashboard_data import snapshot_from_records, snapshot_markdown
from .exploration import ExplorationLimits
from .game import KeyGateGame, defect_evaluation_metadata
from .hybrid import SweepResult, run_collision_sweep
from .lean import LEAN_PROMPT_VERSION, run_lean_exploration
from .nebius import NebiusClient, NebiusError
from .reporting import extract_issues, load_recording, reproduce_issue
from .session import SessionLog, redact_secrets
from .triage import DEFAULT_TRIAGE_MODEL, TriageLimits, TriagePricing, build_evidence_packets, render_triage_report, run_triage
from .unity_launcher import UnityPlayerGame, unity_defect_evaluation_metadata


LIGHTNING_PRICING = TriagePricing(0.06, 0.24)
ULTRA_PRICING = TriagePricing(1.0, 3.0)
_JOB_FIELDS = frozenset({"id", "requestId", "leaseId", "createdAt", "game", "profile", "seed", "maxUsd", "cancelRequested"})
_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}\Z")
_PROFILES = frozenset({"clean", "wall", "floor", "key", "reset", "door"})
_GRID_PROFILES = {"clean": "none", "wall": "wall-clip", "floor": "phantom-block", "key": "gate-keeps-key", "reset": "reset-leak", "door": "exit-no-win"}
_UNITY_PROFILES = {"clean": "none", "wall": "wall-tunnel", "floor": "floor-gap", "key": "key-not-consumed", "reset": "respawn-leak", "door": "door-ghost-collider"}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _positive(value: object, label: str, *, maximum: float | None = None) -> float:
    if type(value) not in (int, float) or not math.isfinite(value) or value <= 0 or (maximum is not None and value > maximum):
        raise ValueError(f"{label} must be a finite positive number" + (f" no greater than {maximum}." if maximum else "."))
    return float(value)


def validate_job(value: object) -> dict:
    """No remotely supplied paths, shell commands, models or credentials."""
    if not isinstance(value, dict) or set(value) != _JOB_FIELDS:
        raise ValueError("Job must contain exactly the supported bounded fields.")
    for key in ("id", "requestId", "leaseId"):
        if not isinstance(value[key], str) or not _IDENTIFIER.fullmatch(value[key]):
            raise ValueError(f"Job {key} must be a safe identifier.")
    if not isinstance(value["game"], str) or not isinstance(value["profile"], str) or value["game"] not in {"grid", "unity"} or value["profile"] not in _PROFILES:
        raise ValueError("Unsupported game or profile.")
    if type(value["seed"]) is not int or value["seed"] not in {0, 1, 5}:
        raise ValueError("Job seed must be 0, 1 or 5.")
    _positive(value["maxUsd"], "Job maxUsd", maximum=10)
    if type(value["cancelRequested"]) is not bool:
        raise ValueError("Job cancelRequested must be boolean.")
    if not isinstance(value["createdAt"], str) or len(value["createdAt"]) > 64:
        raise ValueError("Job createdAt must be an ISO timestamp.")
    try:
        created = datetime.fromisoformat(value["createdAt"].replace("Z", "+00:00"))
        if created.tzinfo is None:
            raise ValueError
    except ValueError:
        raise ValueError("Job createdAt must include an ISO timezone.") from None
    return dict(value)


def _atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    temporary = path.with_name(f".{path.name}-{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("x", encoding="utf-8") as output:
            os.chmod(temporary, 0o600)
            json.dump(value, output, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)


class BudgetExceeded(NebiusError):
    """A local estimated reservation would exceed the already authorized cap."""


class Cancelled(KeyboardInterrupt):
    pass


class CostLedger:
    """Persistent combined Lightning/Ultra pre-HTTP cost reservations."""

    def __init__(self, path: Path, cap_usd: float):
        self.path = Path(path)
        self.cap_usd = _positive(cap_usd, "Cost cap")
        self._thread_lock = threading.RLock()
        self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        with self._locked():
            if self.path.exists():
                self._load()
            else:
                _atomic_json(self.path, {"version": "dashboard-cost-v1", "cap_usd": self.cap_usd, "requests": []})

    @contextmanager
    def _locked(self):
        with self._thread_lock:
            with self.path.with_suffix(self.path.suffix + ".lock").open("a+") as lock:
                os.chmod(lock.name, 0o600)
                fcntl.flock(lock, fcntl.LOCK_EX)
                try:
                    yield
                finally:
                    fcntl.flock(lock, fcntl.LOCK_UN)

    def _load(self) -> dict:
        state = json.loads(self.path.read_text(encoding="utf-8"))
        if state.get("version") != "dashboard-cost-v1" or state.get("cap_usd") != self.cap_usd or not isinstance(state.get("requests"), list):
            raise ValueError("Stored cost ledger is incompatible; its cap cannot be changed.")
        for request in state["requests"]:
            if type(request.get("accounted_cost_usd")) not in (int, float) or not math.isfinite(request["accounted_cost_usd"]) or request["accounted_cost_usd"] < 0:
                raise ValueError("Stored cost ledger is invalid.")
        return state

    def reserve_request(self, model: str, messages: list, schema: dict | None,
                        max_tokens: int, pricing: TriagePricing) -> str:
        if type(max_tokens) is not int or max_tokens <= 0:
            raise ValueError("Request output cap must be a positive integer.")
        prompt_bytes = len(json.dumps(messages, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8"))
        schema_bytes = len(json.dumps(schema or {}, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8"))
        reservation = pricing.cost(prompt_bytes + schema_bytes + 4096, max_tokens)
        with self._locked():
            state = self._load()
            current = sum(row["accounted_cost_usd"] for row in state["requests"])
            if any(row.get("exceeded_reservation") for row in state["requests"]) or current + reservation > self.cap_usd:
                raise BudgetExceeded("Combined estimated job budget reached; no further provider request was sent.")
            identifier = uuid.uuid4().hex
            state["requests"].append({
                "id": identifier, "model": model, "reserved_at": _now(), "status": "reserved",
                "prompt_bytes": prompt_bytes, "schema_bytes": schema_bytes, "max_tokens": max_tokens,
                "input_usd_per_million": pricing.input_usd_per_million,
                "output_usd_per_million": pricing.output_usd_per_million,
                "reservation_usd": reservation, "accounted_cost_usd": reservation,
                "known_cost_usd": None, "usage_known": False,
            })
            _atomic_json(self.path, state)
            return identifier

    def settle_request(self, identifier: str, usage: object) -> None:
        with self._locked():
            state = self._load()
            row = next((r for r in state["requests"] if r["id"] == identifier), None)
            if row is None:
                raise ValueError("Unknown request reservation.")
            if row["status"] != "reserved":
                return
            valid = isinstance(usage, dict) and all(type(usage.get(k)) is int and usage[k] >= 0 for k in ("prompt_tokens", "completion_tokens", "total_tokens"))
            valid = valid and usage["total_tokens"] == usage["prompt_tokens"] + usage["completion_tokens"]
            if valid:
                pricing = TriagePricing(row["input_usd_per_million"], row["output_usd_per_million"])
                cost = pricing.cost(usage["prompt_tokens"], usage["completion_tokens"])
                row.update(status="completed", usage_known=True, usage={k: usage[k] for k in ("prompt_tokens", "completion_tokens", "total_tokens")},
                           known_cost_usd=cost, accounted_cost_usd=cost, exceeded_reservation=cost > row["reservation_usd"])
            else:
                row.update(status="unknown", usage_known=False)
            row["settled_at"] = _now()
            _atomic_json(self.path, state)

    def summary(self) -> dict:
        with self._locked():
            rows = self._load()["requests"]
        return {
            "accounted_cost_usd": round(sum(r["accounted_cost_usd"] for r in rows), 8),
            "known_cost_usd": round(sum(r["known_cost_usd"] or 0 for r in rows), 8),
            "usage_known": all(r["usage_known"] for r in rows),
            "unknown_usage_requests": sum(not r["usage_known"] for r in rows),
            "requests": len(rows),
        }


class BudgetedClient:
    def __init__(self, client, ledger: CostLedger, pricing: TriagePricing,
                 cancel_check: Callable[[], bool] = lambda: False):
        self.client, self.ledger, self.pricing, self.cancel_check = client, ledger, pricing, cancel_check

    def __getattr__(self, name):
        return getattr(self.client, name)

    def complete(self, messages, timeout=None, *, response_schema=None, enable_thinking=None, max_tokens=None):
        if self.cancel_check():
            raise Cancelled("Job cancellation requested before provider dispatch.")
        identifier = self.ledger.reserve_request(self.settings.model, messages, response_schema,
                                                  self.max_tokens if max_tokens is None else max_tokens, self.pricing)
        try:
            completion = self.client.complete(messages, timeout=timeout, response_schema=response_schema,
                                              enable_thinking=enable_thinking, max_tokens=max_tokens)
        except BaseException:
            self.ledger.settle_request(identifier, None)
            raise
        self.ledger.settle_request(identifier, completion.metadata.get("usage"))
        return completion


class ReceiptStore:
    def __init__(self, state_dir: Path):
        self.state_dir = Path(state_dir)
        self.state_dir.mkdir(parents=True, exist_ok=True, mode=0o700)

    def start(self, job: dict) -> Path:
        job = validate_job(job)
        directory = self.state_dir / job["id"]
        directory.mkdir(mode=0o700)  # Exclusive: even an interrupted job cannot run twice.
        _atomic_json(directory / "receipt.json", {"jobId": job["id"], "leaseId": job["leaseId"],
                     "createdAt": job["createdAt"], "startedAt": _now(), "status": "running", "job": job})
        return directory

    def finish(self, job_id: str, status: str, **fields) -> dict:
        if not isinstance(job_id, str) or not _IDENTIFIER.fullmatch(job_id):
            raise ValueError("Unsafe job identifier.")
        if status not in {"completed", "failed", "cancelled", "interrupted"}:
            raise ValueError("Unsupported receipt outcome.")
        path = self.state_dir / job_id / "receipt.json"
        receipt = json.loads(path.read_text(encoding="utf-8"))
        receipt.update({"status": status, "completedAt": _now(), "deliveryPending": True, **fields})
        _atomic_json(path, receipt)
        return receipt


class TotalAllowance:
    """Reserve each entire job cap against the user's newly bounded allowance."""

    def __init__(self, path: Path, cap_usd: float = 10):
        self.path, self.cap_usd = Path(path), _positive(cap_usd, "Session allowance", maximum=10)
        self._thread_lock = threading.RLock()
        self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        with self._locked():
            if not self.path.exists():
                _atomic_json(self.path, {"version": "dashboard-allowance-v1", "cap_usd": self.cap_usd, "jobs": {}})
            self._load()

    @contextmanager
    def _locked(self):
        with self._thread_lock:
            with self.path.with_suffix(self.path.suffix + ".lock").open("a+") as lock:
                os.chmod(lock.name, 0o600)
                fcntl.flock(lock, fcntl.LOCK_EX)
                try:
                    yield
                finally:
                    fcntl.flock(lock, fcntl.LOCK_UN)

    def _load(self):
        state = json.loads(self.path.read_text(encoding="utf-8"))
        if state.get("version") != "dashboard-allowance-v1" or state.get("cap_usd") != self.cap_usd or not isinstance(state.get("jobs"), dict):
            raise ValueError("Existing total allowance cap cannot be changed.")
        if any(type(job.get("accounted_usd")) not in (int, float) or not math.isfinite(job["accounted_usd"]) or job["accounted_usd"] < 0 for job in state["jobs"].values()):
            raise ValueError("Stored session allowance is invalid.")
        return state

    def reserve(self, job_id: str, max_usd: float):
        amount = _positive(max_usd, "Job allowance", maximum=10)
        with self._locked():
            state = self._load()
            if job_id in state["jobs"]:
                raise ValueError("Job already reserved in the session allowance.")
            if sum(job["accounted_usd"] for job in state["jobs"].values()) + amount > self.cap_usd:
                raise BudgetExceeded("Remaining session allowance cannot reserve this entire job cap.")
            state["jobs"][job_id] = {"reservation_usd": amount, "accounted_usd": amount, "usage_known": False, "status": "reserved"}
            _atomic_json(self.path, state)

    def settle(self, job_id: str, summary: dict):
        with self._locked():
            state = self._load()
            job = state["jobs"][job_id]
            if job["status"] != "reserved":
                return
            if summary.get("usage_known") is True:
                amount = summary.get("accounted_cost_usd")
                if type(amount) not in (int, float) or not math.isfinite(amount) or amount < 0:
                    raise ValueError("Known job cost must be finite and nonnegative.")
                job.update(accounted_usd=amount, usage_known=True, status="completed")
            else:
                job.update(status="unknown")  # Unknown jobs retain their complete allowance reservation.
            _atomic_json(self.path, state)


class _EventLog(SessionLog):
    def __init__(self, path, secrets, callback):
        super().__init__(path, secrets=secrets)
        self.callback = callback

    def write(self, kind, **fields):
        super().write(kind, **fields)
        self.callback(redact_secrets({"type": kind, "timestamp": _now(),
                                     "elapsed_ms": round((time.monotonic() - self.started) * 1000), **fields}, self.secrets))


class _GuardedGame:
    def __init__(self, game, cancel_check):
        self.game, self.cancel_check = game, cancel_check

    def __getattr__(self, name):
        return getattr(self.game, name)

    def _check(self):
        if self.cancel_check():
            raise Cancelled("Job stopped at a safe game-control boundary.")

    def observe(self):
        # Observation is read-only and remains available for final summaries.
        return self.game.observe()

    def act(self, action):
        self._check()
        return self.game.act(action)

    def reset(self, *args, **kwargs):
        self._check()
        return _GuardedGame(self.game.reset(*args, **kwargs), self.cancel_check)


def execute_job(job: dict, *, state_dir: Path, settings: Settings,
                unity_player: Path | None = None, publish: Callable[[dict], None] = lambda snapshot: None,
                cancel_check: Callable[[], bool] = lambda: False, client_factory=NebiusClient,
                game_factory=None, allowance: TotalAllowance | None = None) -> dict:
    """Use existing QA APIs without changing the learner's controls or defaults."""
    job = validate_job(job)
    receipts = ReceiptStore(state_dir)
    directory = receipts.start(job)
    allowance = allowance or TotalAllowance(Path(state_dir) / "allowance.json")
    ledger = CostLedger(directory / "cost.json", job["maxUsd"])
    allowance_reserved = False
    records, provisional = [], []
    sweep_progress = None
    triage = None
    phase, status = "sweep", "running"
    started = time.monotonic()
    last_publish = 0.0
    metadata = (_UNITY_PROFILES if job["game"] == "unity" else _GRID_PROFILES)[job["profile"]]
    evaluate = unity_defect_evaluation_metadata(metadata) if job["game"] == "unity" else defect_evaluation_metadata(metadata)
    # Force locally fixed models and exploration defaults, regardless of .env.
    lightning_settings = replace(settings, model=DEFAULT_MODEL, enable_thinking=True, response_format="json_schema")
    ultra_settings = replace(settings, model=DEFAULT_TRIAGE_MODEL, enable_thinking=True, response_format="json_schema")
    notes = []

    def cancelled():
        return job["cancelRequested"] or cancel_check()

    def snapshot(final=False):
        rows = records + provisional
        view = snapshot_from_records(rows, run_id=job["id"], name=f"{'Unity' if job['game'] == 'unity' else 'Grid'} · Seed {job['seed']}",
                                     source="live", status=status, phase=phase, triage_result=triage,
                                     budget_usd=job["maxUsd"], cost=ledger.summary(), created_at=job["createdAt"],
                                     elapsed_seconds=time.monotonic() - started, secrets=(settings.api_key,),
                                     candidate_issues=[] if phase == "sweep" else None, notes=notes)
        view["summary"].update(game=job["game"], seed=job["seed"], budgetUsd=job["maxUsd"])
        if final:
            _atomic_json(directory / "snapshot.json", view)
            (directory / "report.md").write_text(snapshot_markdown(view), encoding="utf-8")
        return view

    def push(force=False):
        nonlocal last_publish
        if not force and time.monotonic() - last_publish < 1:
            return
        last_publish = time.monotonic()
        publish(snapshot())

    def on_record(record):
        nonlocal triage
        records.append(record)
        if record["type"] == "triage_result" and triage is not None:
            triage["results"] = [r for r in triage["results"] if r["issue_number"] != record["issue_number"]]
            triage["results"].append({key: item for key, item in record.items() if key not in {"type", "timestamp", "elapsed_ms"}})
        if record["type"] not in {"memory", "completion", "sweep_step", "sweep_reset"}:
            push()

    def on_sweep(record):
        if sweep_progress is not None:
            sweep_progress.write(record["type"], **{key: item for key, item in record.items() if key != "type"})
        transient = deepcopy(record)
        transient["step"] = transient["phase_step"] - 401
        transient["elapsed_ms"] = round((time.monotonic() - started) * 1000)
        provisional.append(transient)
        if len(provisional) % 20 == 0:
            push()

    def create():
        if cancelled():
            raise Cancelled()
        if game_factory is not None:
            game = game_factory(job["seed"], metadata)
        elif job["game"] == "unity":
            if unity_player is None:
                raise ValueError("The local Unity player is not configured.")
            game = UnityPlayerGame(unity_player, seed=job["seed"], defect=metadata)
        else:
            game = KeyGateGame(job["seed"], metadata)
        return _GuardedGame(game, cancelled)

    try:
        allowance.reserve(job["id"], job["maxUsd"])
        allowance_reserved = True
        if cancelled():
            raise Cancelled()
        with _EventLog(directory / "recording.jsonl", (settings.api_key,), on_record) as log:
            game = create()
            try:
                begin_session(log, game, "nebius", schema_version=2, task="explore", focus="qa",
                              prompt_version=LEAN_PROMPT_VERSION, direct_fallback_max_tokens=8192,
                              model=DEFAULT_MODEL, enable_thinking=True, max_tokens=8192, temperature=1.0, top_p=0.95,
                              response_format="json_schema", limits=asdict(ExplorationLimits()), hybrid=True,
                              sweep_action_budget=400, **evaluate)
                notes.append("During the sweep, displayed negative IDs are provisional; completion preserves the original final sweep step IDs.")
                with SessionLog(directory / "sweep-progress.jsonl", secrets=(settings.api_key,)) as progress:
                    sweep_progress = progress
                    begin_session(progress, game, "dashboard-sweep", schema_version=2, **evaluate)
                    try:
                        sweep = run_collision_sweep(game, max_actions=400, on_record=on_sweep)
                    except BaseException:
                        # Preserve every executed partial control with the same
                        # final negative mapping used by successful sweeps.
                        partial_rows = deepcopy(provisional)
                        provisional.clear()
                        for row in partial_rows:
                            row["step"] = row["phase_step"]
                            row.pop("elapsed_ms", None)
                        _write_sweep_phase(log, SweepResult(partial_rows, {"source": "sweep", "status": "interrupted", "actions": len(partial_rows), "action_budget": 400}))
                        raise
                    finally:
                        sweep_progress = None
            finally:
                _close_game(game)
            provisional.clear()
            notes = [n for n in notes if not n.startswith("During the sweep")]
            _write_sweep_phase(log, sweep)
            phase = "explore"
            push(True)
            game = create()
            try:
                lightning = BudgetedClient(client_factory(lightning_settings, timeout=45, max_tokens=8192, temperature=1.0, top_p=0.95),
                                           ledger, LIGHTNING_PRICING, cancelled)
                exploration = run_lean_exploration(game, lightning, log, game.reset, ExplorationLimits(),
                                                   focus="qa", evaluator=game.evaluate_transition,
                                                   on_action=lambda record: push(), sweep_summary=sweep.summary)
            finally:
                _close_game(game)
            if cancelled():
                raise Cancelled()
            if exploration["status"] == "interrupted":
                raise KeyboardInterrupt()
            if exploration["status"] in {"inference_error", "environment_error", "provider_unavailable", "invalid_decision", "model_output_limit"}:
                notes.append("Exploration ended with an incomplete provider or environment outcome; executed evidence is retained.")
                status = "failed"
            phase = "triage"
            push(True)
            # One original recorded prefix per issue, without expensive search.
            # At most eight issues and 3,392 controls (8 * (400+24)) are replayed.
            reproductions = []
            issues = extract_issues(records)
            for issue in issues[:8]:
                if cancelled():
                    raise Cancelled()
                def replay_factory(seed, defect):
                    return create()
                # The sweep and model exploration each start a fresh game.
                # Replay the original phase prefix, preserving original IDs,
                # rather than concatenating those distinct starts.
                anchor = min(issue["evidence_steps"]) if issue["type"] in {"passed_through_solid", "sank_into_visible_floor"} else max(issue["evidence_steps"])
                phase_records = [r for r in records if r["type"] == "session" or (
                    r["type"] in {"step", "reset", "sweep_step", "sweep_reset"} and type(r.get("step")) is int
                    and ((r["step"] < 0) == (anchor < 0))
                )]
                phase_issues = extract_issues(phase_records)
                match = next((candidate for candidate in phase_issues if candidate["type"] == issue["type"] and candidate["title"] == issue["title"]), None)
                if match is None:
                    continue
                reproduced = reproduce_issue(phase_records, match["number"], minimize=False, game_factory=replay_factory)
                reproduced["issue"] = issue["number"]
                reproduced["minimization_complete"] = False
                reproductions.append(reproduced)
            packets = build_evidence_packets(records, reproduction_results=reproductions)
            triage = {"results": [{"issue_number": p["issue_number"], "status": "pending", "assessment": None, "evidence_packet": p, "warnings": []} for p in packets]}
            push(True)
            ultra = BudgetedClient(client_factory(ultra_settings, timeout=45, max_tokens=8192, temperature=1.0, top_p=0.95),
                                   ledger, ULTRA_PRICING, cancelled)
            with _EventLog(directory / "triage-audit.jsonl", (settings.api_key,), on_record) as audit:
                triage = run_triage(packets, ultra, limits=TriageLimits(max_usd=min(1.0, job["maxUsd"])), pricing=ULTRA_PRICING, log=audit)
            _atomic_json(directory / "triage.json", triage)
            (directory / "triage.md").write_text(render_triage_report(triage), encoding="utf-8")
            if cancelled():
                raise Cancelled()
            if triage["status"] == "interrupted":
                raise KeyboardInterrupt()
            if triage["status"] in {"review_failures", "cost_limit", "time_limit", "request_limit"}:
                status = "failed"
                notes.append("The run ended with incomplete Ultra review. Unreviewed candidates remain pending beside their executed evidence.")
            if status != "failed":
                status = "completed"
            phase = "complete"
    except Cancelled:
        status, phase = "cancelled", "complete"
        notes.append("Cancelled at a safe boundary. Already dispatched provider requests retain known usage or conservative reservations.")
    except KeyboardInterrupt:
        status, phase = "interrupted", "complete"
        notes.append("Runner interrupted. This job's durable receipt prevents automatic replay or inference retries after restart.")
    except BaseException as exc:
        status, phase = "failed", "complete"
        # Never publish raw exceptions, provider content, local paths or keys.
        notes.append(f"Runner stopped with {type(exc).__name__}. Local execution evidence and cost reservations were retained.")
    finally:
        cost = ledger.summary()
        if allowance_reserved:
            allowance.settle(job["id"], cost)
        view = snapshot(True)
        receipts.finish(job["id"], status, cost=cost, controls=view["summary"]["controls"], resets=view["summary"]["resets"], requests=cost["requests"])
        publish(view)
    return view


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, request, fp, code, message, headers, newurl):
        return None


class DashboardTransport:
    def __init__(self, site_url: str, token: str, *, allow_local_http=False):
        parsed = urlsplit(site_url)
        local = parsed.hostname in {"127.0.0.1", "localhost", "::1"}
        if parsed.scheme != "https" and not (allow_local_http and parsed.scheme == "http" and local):
            raise ValueError("Dashboard URL must use HTTPS; loopback HTTP needs --allow-local-http.")
        if not parsed.hostname or parsed.username or parsed.password or parsed.query or parsed.fragment or parsed.path not in {"", "/"}:
            raise ValueError("Dashboard URL must be an origin without credentials or extra paths.")
        if not isinstance(token, str) or len(token) < 24 or any(c.isspace() for c in token):
            raise ValueError("DASHBOARD_RUNNER_TOKEN must be a strong local secret.")
        self.origin, self.token = site_url.rstrip("/"), token
        self.opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), _NoRedirect())

    def post(self, endpoint: str, body: dict) -> dict:
        if endpoint not in {"heartbeat", "claim", "update", "control"}:
            raise ValueError("Unsupported dashboard runner endpoint.")
        encoded = json.dumps(body, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8")
        if len(encoded) > 1_950_000:
            raise ValueError("Sanitized dashboard update exceeds its upload size bound.")
        request = urllib.request.Request(self.origin + "/api/runner/" + endpoint, data=encoded,
                                        headers={"Authorization": "Bearer " + self.token, "Content-Type": "application/json", "Accept": "application/json", "User-Agent": "AutoPlayQA-DashboardRunner/0.1"}, method="POST")
        try:
            with self.opener.open(request, timeout=15) as response:
                raw = response.read(2_000_001)
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError):
            raise RuntimeError("Dashboard HTTPS request failed; credentials and response bodies withheld.") from None
        if len(raw) > 2_000_000:
            raise RuntimeError("Dashboard response exceeded its size bound.")
        value = json.loads(raw)
        if not isinstance(value, dict):
            raise RuntimeError("Unexpected dashboard response shape.")
        return value


class _Bridge:
    """Background heartbeats, cancel polling and coalesced snapshot uploads."""

    def __init__(self, transport, runner_id, unity_ready, stop_event):
        self.transport, self.runner_id, self.unity_ready, self.stop_event = transport, runner_id, unity_ready, stop_event
        self.job = None
        self.latest = None
        self.revision = 0
        self.cancel = threading.Event()
        self.lock = threading.Lock()
        self.send_lock = threading.Lock()
        self.last_control_success = time.monotonic()
        self.upload_failure = False
        self.thread = threading.Thread(target=self._loop, daemon=True, name="dashboard-bridge")

    def heartbeat(self):
        return self.transport.post("heartbeat", {"runnerId": self.runner_id, "name": "Mac runner", "unityReady": self.unity_ready})

    def start(self):
        self.heartbeat()
        self.thread.start()

    def publish(self, snapshot):
        with self.lock:
            self.latest = deepcopy(snapshot)

    def begin(self, job, *, revision=0):
        if type(revision) is not int or revision < 0:
            raise ValueError("Runner update revision must be a nonnegative integer.")
        with self.lock:
            self.job, self.latest, self.revision = job, None, revision
        self.cancel.clear()
        self.last_control_success = time.monotonic()

    def cancelled(self):
        return self.cancel.is_set() or self.stop_event.is_set()

    def _send(self):
        with self.send_lock:
            return self._send_locked()

    def _send_locked(self):
        with self.lock:
            snapshot, job = self.latest, self.job
            if snapshot is None or job is None:
                return False
            self.revision += 1
            revision = self.revision
        value = self.transport.post("update", {"runnerId": self.runner_id, "jobId": job["id"], "leaseId": job["leaseId"], "revision": revision, "snapshot": snapshot})
        if value.get("cancelRequested") is True:
            self.cancel.set()
        with self.lock:
            if self.latest is snapshot:
                self.latest = None
        return True

    def flush(self):
        # A final upload is attempted before the next claim. Same snapshot may
        # be repeated with increasing revision, but inference is never rerun.
        for _ in range(3):
            try:
                with self.lock:
                    empty = self.latest is None
                if empty:
                    return True
                self._send()
                return True
            except (RuntimeError, ValueError):
                self.upload_failure = True
                self.stop_event.wait(1)
        return False

    def _loop(self):
        heartbeat_due, control_due = time.monotonic() + 10, time.monotonic()
        while not self.stop_event.wait(1):
            now = time.monotonic()
            job = None
            try:
                if now >= heartbeat_due:
                    self.heartbeat()
                    heartbeat_due = now + 10
                with self.lock:
                    job = self.job
                if job is not None and now >= control_due:
                    value = self.transport.post("control", {"runnerId": self.runner_id, "jobId": job["id"], "leaseId": job["leaseId"]})
                    self.last_control_success = now
                    control_due = now + 2
                    if value.get("cancelRequested") is True:
                        self.cancel.set()
                self._send()
            except (RuntimeError, ValueError):
                self.upload_failure = True
                if job is not None and now - self.last_control_success > 30:
                    self.cancel.set()


def recover_interrupted_receipts(state_dir: Path, publish_job: Callable[[dict, dict], None]) -> int:
    """Report saved leases as interrupted, with no adapter or provider calls."""
    recovered = 0
    store = ReceiptStore(state_dir)
    for directory in Path(state_dir).iterdir():
        if not directory.is_dir():
            continue
        receipt_path = directory / "receipt.json"
        if not receipt_path.is_file():
            continue
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        if receipt.get("status") != "running" and receipt.get("deliveryPending") is not True:
            continue
        job = validate_job(receipt.get("job"))
        if directory.name != job["id"] or receipt.get("leaseId") != job["leaseId"]:
            raise ValueError("Interrupted receipt identity is inconsistent.")
        rows = load_recording(directory / "recording.jsonl") if (directory / "recording.jsonl").is_file() else []
        # A hard process stop may leave a partial sweep only in its per-action
        # journal. Canonicalize those steps from the final retained count.
        if not any(row.get("type") in {"sweep_step", "sweep_reset"} for row in rows) and (directory / "sweep-progress.jsonl").is_file():
            progress = load_recording(directory / "sweep-progress.jsonl")
            if not rows and progress:
                rows = [progress[0]]
            actions = [deepcopy(row) for row in progress if row.get("type") in {"sweep_step", "sweep_reset"}]
            for index, row in enumerate(actions, 1):
                row["phase_step"] = row.get("phase_step", index)
                row["step"] = index - len(actions) - 1
                fact = row.get("software_fact")
                if isinstance(fact, dict):
                    for anomaly in fact.get("anomalies", []):
                        anomaly.update(step=row["step"], evidence_steps=[row["step"]])
            rows.extend(actions)
        cost = CostLedger(directory / "cost.json", job["maxUsd"]).summary() if (directory / "cost.json").is_file() else None
        triage = json.loads((directory / "triage.json").read_text()) if (directory / "triage.json").is_file() else None
        if triage is None and (directory / "triage-audit.jsonl").is_file():
            audit = [json.loads(line) for line in (directory / "triage-audit.jsonl").read_text().splitlines() if line.strip()]
            triage = {"results": [row for row in audit if row.get("type") == "triage_result"]}
        terminal_status = "interrupted" if receipt.get("status") == "running" else receipt["status"]
        snapshot = snapshot_from_records(rows, run_id=job["id"], source="live", status=terminal_status, phase="complete",
                                         created_at=job["createdAt"], cost=cost, triage_result=triage, budget_usd=job["maxUsd"],
                                         notes=["A previous runner process stopped. This existing receipt prevents any game or inference work from being restarted."])
        snapshot["summary"].update(game=job["game"], seed=job["seed"], budgetUsd=job["maxUsd"])
        _atomic_json(directory / "snapshot.json", snapshot)
        (directory / "report.md").write_text(snapshot_markdown(snapshot), encoding="utf-8")
        store.finish(job["id"], terminal_status, cost=cost)
        acknowledged = publish_job(job, snapshot)
        if acknowledged is not False:
            if isinstance(acknowledged, str) and acknowledged in {"completed", "failed", "cancelled", "interrupted"}:
                terminal_status = acknowledged
                snapshot["summary"]["status"] = terminal_status
                _atomic_json(directory / "snapshot.json", snapshot)
            store.finish(job["id"], terminal_status, cost=cost, deliveryPending=False)
        recovered += 1
    return recovered


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--once", action="store_true", help="Claim at most one job, then exit.")
    mode.add_argument("--serve", action="store_true", help="Heartbeat and claim bounded jobs until stopped (default).")
    parser.add_argument("--state-dir", type=Path, default=Path("runs/dashboard"))
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument("--unity-player", type=Path)
    parser.add_argument("--total-budget-usd", type=float, default=10)
    parser.add_argument("--allow-local-http", action="store_true", help="Explicit development-only loopback HTTP.")
    args = parser.parse_args(argv)
    stop = threading.Event()
    previous_handlers = {}
    bridge = None
    args.state_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    lock = (args.state_dir / "runner.lock").open("a+")
    try:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError("A runner already owns this state directory.") from None
        allowance = TotalAllowance(args.state_dir / "allowance.json", args.total_budget_usd)
        settings = Settings.load(args.env_file)
        transport = DashboardTransport(os.environ.get("DASHBOARD_SITE_URL", ""), os.environ.get("DASHBOARD_RUNNER_TOKEN", ""), allow_local_http=args.allow_local_http)
        identity_path = args.state_dir / "identity.json"
        if not identity_path.exists():
            _atomic_json(identity_path, {"runnerId": uuid.uuid4().hex})
        runner_id = json.loads(identity_path.read_text())["runnerId"]
        unity_ready = args.unity_player is not None and args.unity_player.is_file() and os.access(args.unity_player, os.X_OK)
        if args.unity_player is not None and not unity_ready:
            raise ValueError("Configured local Unity player must exist and be executable.")
        for signum in (signal.SIGINT, signal.SIGTERM):
            previous_handlers[signum] = signal.signal(signum, lambda signum, frame: stop.set())
        bridge = _Bridge(transport, runner_id, unity_ready, stop)
        bridge.start()
        print("Mac runner online. Provider credentials stay local; jobs share the persisted session allowance.", flush=True)
        def upload_recovered(job, snapshot):
            try:
                current = transport.post("control", {"runnerId": runner_id, "jobId": job["id"], "leaseId": job["leaseId"]})
            except (RuntimeError, ValueError):
                bridge.upload_failure = True
                return False
            if current.get("status") in {"completed", "failed", "cancelled", "interrupted"}:
                return current["status"]
            revision = current.get("revision")
            if type(revision) is not int or revision < 0:
                bridge.upload_failure = True
                return False
            bridge.begin(job, revision=revision)
            bridge.publish(snapshot)
            sent = bridge.flush()
            with bridge.lock:
                bridge.job = None
            return sent
        recovered = recover_interrupted_receipts(args.state_dir, upload_recovered)
        if recovered:
            print(f"Reported {recovered} saved interrupted job(s) without running them again.", flush=True)
        if bridge.upload_failure:
            return 1
        while not stop.is_set():
            result = transport.post("claim", {"runnerId": runner_id})
            candidate = result.get("job")
            if candidate is None:
                if args.once:
                    break
                stop.wait(3)
                continue
            job = validate_job(candidate)
            bridge.begin(job)
            try:
                view = execute_job(job, state_dir=args.state_dir, settings=settings, unity_player=args.unity_player,
                                   publish=bridge.publish, cancel_check=bridge.cancelled, allowance=allowance)
                delivered = bridge.flush()
                if delivered:
                    ReceiptStore(args.state_dir).finish(job["id"], view["summary"]["status"], deliveryPending=False)
                summary = view["summary"]
                print(f"Job {job['id']}: {summary['status']} · controls {summary['controls']} · resets {summary['resets']} · requests {summary['requests']} · estimated USD {summary['estimatedCostUsd']}", flush=True)
            except FileExistsError:
                print("Previously claimed job refused. Its receipt prevents another model run.", flush=True)
                # Upload already saved public evidence with a terminal outcome;
                # interrupted jobs are never silently re-executed.
                saved = args.state_dir / job["id"] / "snapshot.json"
                if saved.exists():
                    snapshot = json.loads(saved.read_text())
                    snapshot["summary"].update(status="interrupted", phase="complete")
                    bridge.publish(snapshot)
                    bridge.flush()
                else:
                    empty = snapshot_from_records([], run_id=job["id"], source="live", status="interrupted", phase="complete", created_at=job["createdAt"], budget_usd=job["maxUsd"], notes=["An existing local receipt prevented inference from being run again."])
                    empty["summary"]["game"] = job["game"]
                    empty["summary"]["seed"] = job["seed"]
                    bridge.publish(empty)
                    bridge.flush()
            with bridge.lock:
                bridge.job = None
            if args.once or bridge.upload_failure:
                break
        return 1 if bridge.upload_failure else 0
    except (ValueError, RuntimeError, OSError, NebiusError):
        print("Runner setup or transport failed. Check local configuration; secrets, paths and provider bodies are withheld.", flush=True)
        return 2
    finally:
        stop.set()
        if bridge is not None and bridge.thread.is_alive():
            bridge.thread.join(timeout=16)
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)
        fcntl.flock(lock, fcntl.LOCK_UN)
        lock.close()


if __name__ == "__main__":
    raise SystemExit(main())
