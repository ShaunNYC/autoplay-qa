"""Bounded, evidence-backed experience from public adapter transitions.

This is an external fact store, not model training.  It stores what executed,
never infers that an unchanged state is a bug, and does not choose controls.
Reviewer feedback is deliberately kept out of explorer queries.

An adapter may declare ``_capabilities.state_identity.transient_paths`` as
dotted paths (``entities.execution`` or ``entities.*.state.visited``).  Entity
lists can be addressed by their public ``id``; ``*`` components are supported.
Those fields remain visible feedback but do not define equivalent states.
"""

from __future__ import annotations

from contextlib import contextmanager
import fnmatch
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sqlite3
import threading
import time
from typing import Iterator


EXPERIENCE_VERSION = "observed-experience-v1"
FACT_VERSION = "observed-transition-v1"
EXPERIMENT_INDEX_VERSION = "observed-experiment-index-v1"
INVESTIGATION_STORE_VERSION = "tentative-investigation-store-v1"
MAX_FACT_BYTES = 3_000
MAX_CITATIONS = 4
_PUBLIC_KEYS = (
    "game", "tick", "status", "map", "grid", "coordinates", "coordinate_system",
    "units", "legend", "player", "objects", "entities", "events",
)
_PRIVATE_KEYS = frozenset({
    "evaluation", "injecteddefect", "groundtruth", "defect", "defectinjection",
    "developerrules", "developerrule", "privaterule", "privaterules", "rules", "rule",
    "strategy", "solution", "victorystrategy", "hints", "hint", "apikey",
    "authorization", "credentials", "credential", "password", "secret", "secrets",
    "apitoken", "accesstoken", "refreshtoken", "unityplayer", "endpoint", "baseurl",
    "localpath", "sourcepath", "sourcecode", "prompt", "thinking",
    "reasoningcontent", "rawresponse", "private",
})
_RUN_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")
_TOKEN_VALUE = re.compile(r"\b(?:Bearer\s+[A-Za-z0-9._~+/-]{8,}|(?:sk|nvapi)-[A-Za-z0-9_-]{12,})", re.I)
_LOCAL_PATH = re.compile(r"(?:/Users/|/home/|/private/|/var/folders/|[A-Za-z]:\\)[^\s\"<>]+")


def _json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def _bytes(value: object) -> int:
    return len(_json(value).encode("utf-8"))


def _digest(value: object) -> str:
    return hashlib.sha256(_json(value).encode("utf-8")).hexdigest()


def _private_key(key: object) -> bool:
    if not isinstance(key, str) or key.startswith("_"):
        return True
    folded = re.sub(r"[^a-z0-9]", "", key.lower())
    return folded in _PRIVATE_KEYS or folded.startswith(("private", "developer", "secret"))


def _secret_values(value: object) -> set[str]:
    """Avoid echoing a private string through an otherwise public feedback field."""
    values: set[str] = set()
    if isinstance(value, dict):
        for key, item in value.items():
            normalized = re.sub(r"[^a-z0-9]", "", str(key).lower())
            if normalized in {"apikey", "apitoken", "authorization", "credentials", "password", "secret", "accesstoken"}:
                if isinstance(item, str) and len(item) >= 6:
                    values.add(item)
            values.update(_secret_values(item))
    elif isinstance(value, (list, tuple)):
        for item in value:
            values.update(_secret_values(item))
    return values


def _public_value(value: object, secrets: tuple[str, ...] = ()) -> object:
    if isinstance(value, dict):
        return {
            key: _public_value(item, secrets) for key, item in value.items()
            if not _private_key(key)
        }
    if isinstance(value, (list, tuple)):
        return [_public_value(item, secrets) for item in value]
    if isinstance(value, str):
        for secret in secrets:
            if secret:
                value = value.replace(secret, "[REDACTED]")
        return _LOCAL_PATH.sub("[LOCAL_PATH]", _TOKEN_VALUE.sub("[REDACTED]", value))
    if value is None or type(value) in (bool, int):
        return value
    if type(value) is float and math.isfinite(value):
        return value
    return None


def public_observation(observation: dict, *, secrets: tuple[str, ...] = ()) -> dict:
    """Return the native public view, with no inferred neighbors or game rules.

    Only the adapter's controls are advertised.  Reset/finish are host meta
    controls, subject to reset capability and the established won-state rule.
    Private metadata is discarded recursively, including inside entities.
    """
    if not isinstance(observation, dict):
        raise TypeError("Observation must be a dictionary.")
    hidden = tuple(sorted(_secret_values(observation), key=len, reverse=True)) + tuple(secrets)
    view = _public_value({key: observation[key] for key in _PUBLIC_KEYS if key in observation}, hidden)
    controls = observation.get("actions")
    if not isinstance(controls, (list, tuple)):
        controls = observation.get("controls", [])
    controls = controls if isinstance(controls, (list, tuple)) else []
    native = []
    for action in controls:
        if isinstance(action, str) and action and action not in {"reset", "finish"}:
            sanitized = _public_value(action, hidden)
            if sanitized == action and action not in native:
                native.append(action)
    capabilities = observation.get("_capabilities")
    can_reset = not isinstance(capabilities, dict) or capabilities.get("reset") is not False
    meta = (["reset"] if can_reset else []) + ["finish"]
    view["controls"] = meta if observation.get("status") == "won" else native + meta
    return view


def _identity_paths(observation: dict) -> list[str]:
    capabilities = observation.get("_capabilities", {})
    if not isinstance(capabilities, dict):
        return []
    identity = capabilities.get("state_identity", {})
    if not isinstance(identity, dict):
        identity = {}
    paths = identity.get("transient_paths", identity.get("transient_fields", capabilities.get("transient_fields", [])))
    if not isinstance(paths, (list, tuple)):
        return []
    return sorted({path for path in paths if isinstance(path, str) and path and len(path) <= 256})


def _drop_path(value: object, components: list[str]) -> None:
    if not components:
        return
    component, *rest = components
    if isinstance(value, dict):
        for key in list(value):
            if fnmatch.fnmatchcase(key, component):
                if rest:
                    _drop_path(value[key], rest)
                else:
                    del value[key]
    elif isinstance(value, list):
        indexes = []
        for index, item in enumerate(value):
            identifier = str(item.get("id", index)) if isinstance(item, dict) else str(index)
            if fnmatch.fnmatchcase(identifier, component) or fnmatch.fnmatchcase(str(index), component):
                if rest:
                    _drop_path(item, rest)
                else:
                    indexes.append(index)
        for index in reversed(indexes):
            del value[index]


def _semantic_state(observation: dict) -> dict:
    view = public_observation(observation)
    view.pop("tick", None)
    view.pop("events", None)
    view["controls"] = sorted(view["controls"])
    for path in _identity_paths(observation):
        _drop_path(view, path.split("."))
    # The protocol describes entities by unique IDs, rather than list order.
    # Duplicate/missing IDs remain ordered because identity would be ambiguous.
    entities = view.get("entities")
    if isinstance(entities, list) and all(isinstance(item, dict) and isinstance(item.get("id"), str) for item in entities):
        identifiers = [item["id"] for item in entities]
        if len(set(identifiers)) == len(identifiers):
            view["entities"] = sorted(entities, key=lambda item: item["id"])
    return view


def state_key(observation: dict) -> str:
    """Exact semantic identity from public state, excluding declared transients.

    No movement/passability or expected behavior is inferred.  Heading,
    inventory and world changes count unless the adapter explicitly declares
    their paths transient.  Different feedback with equal state is still
    retained as a distinct recorded outcome.
    """
    return _digest(_semantic_state(observation))


def _brief(value: object, *, depth: int = 0) -> object:
    if isinstance(value, str):
        return value if len(value) <= 180 else value[:177] + "..."
    if isinstance(value, dict):
        if depth >= 3:
            return {"item_count": len(value), "content_id": _digest(value)}
        result = {key: _brief(item, depth=depth + 1) for key, item in list(value.items())[:10]}
        if len(value) > 10:
            result["omitted_fields"] = len(value) - 10
        return result
    if isinstance(value, list):
        if depth >= 3:
            return {"item_count": len(value), "content_id": _digest(value)}
        result = [_brief(item, depth=depth + 1) for item in value[:8]]
        if len(value) > 8:
            result.append({"omitted_items": len(value) - 8})
        return result
    return value


def _changes(before: object, after: object, prefix: str = "", *, maximum: int = 12) -> tuple[list[dict], int]:
    """Keep bounded leaf differences with an explicit omitted count."""
    differences: list[dict] = []
    count = 0

    def visit(left: object, right: object, path: str) -> None:
        nonlocal count
        if _json(left) == _json(right):
            return
        if isinstance(left, dict) and isinstance(right, dict):
            for key in sorted(set(left) | set(right)):
                visit(left.get(key), right.get(key), f"{path}.{key}".lstrip("."))
            return
        if isinstance(left, list) and isinstance(right, list) and len(left) == len(right):
            for index, (old, new) in enumerate(zip(left, right)):
                visit(old, new, f"{path}.{index}".lstrip("."))
            return
        count += 1
        if len(differences) < maximum:
            differences.append({"path": path, "before": _brief(left), "after": _brief(right)})

    visit(before, after, prefix)
    return differences, max(0, count - len(differences))


def _validate_evidence(run_id: str, step: int) -> None:
    if not isinstance(run_id, str) or not _RUN_ID.fullmatch(run_id):
        raise ValueError("run_id must be an opaque public evidence identifier, without a file path.")
    if type(step) is not int or step < 1:
        raise ValueError("Evidence step must be a positive integer.")


def experiment_fact(before: dict, action: str, after: dict, *, run_id: str, step: int) -> dict:
    """Produce a bounded observation fact, without labels or causal claims."""
    _validate_evidence(run_id, step)
    if not isinstance(action, str) or not action or action not in public_observation(before)["controls"]:
        raise ValueError("Executed action is not advertised by the before observation.")
    old, new = _semantic_state(before), _semantic_state(after)
    changes, omitted = _changes(old, new)
    old_player, new_player = old.get("player", {}), new.get("player", {})
    old_position = old_player.get("position") if isinstance(old_player, dict) else None
    new_position = new_player.get("position") if isinstance(new_player, dict) else None
    before_id, after_id = _digest(old), _digest(new)
    fact = {
        "version": FACT_VERSION,
        "before_state_id": before_id,
        "after_state_id": after_id,
        "action": action,
        "state_changed": before_id != after_id,
        "position_changed": (_json(old_position) != _json(new_position)) if old_position is not None and new_position is not None else None,
        "changed_facets": [key for key in sorted(set(old) | set(new)) if _json(old.get(key)) != _json(new.get(key))],
        "outcome": {
            "player_before": _brief(old_player), "player_after": _brief(new_player),
            "changes": changes, "omitted_changes": omitted,
            "events": _brief(public_observation(after).get("events", [])),
        },
        "citation": {"run_id": run_id, "step": step},
    }
    # Large public entities cannot turn one memory item into an unbounded prompt.
    while _bytes(fact) > MAX_FACT_BYTES and fact["outcome"]["changes"]:
        fact["outcome"]["changes"].pop()
        fact["outcome"]["omitted_changes"] += 1
    if _bytes(fact) > MAX_FACT_BYTES:
        for field in ("player_before", "player_after", "events"):
            value = fact["outcome"][field]
            if _bytes(value) > 300:
                fact["outcome"][field] = {"content_id": _digest(value), "details_omitted": True}
    return fact


class ScopeMismatch(ValueError):
    """Experience belongs to a different game/scene/control/frame scope."""


class EvidenceConflict(ValueError):
    """One evidence citation was reused for a different transition."""


def _grid_shape(observation: dict) -> object:
    rows = observation.get("map", observation.get("grid"))
    if isinstance(rows, list):
        return [len(row) if isinstance(row, (list, str)) else None for row in rows]
    return None


def _private_parent(path: Path) -> None:
    missing = []
    parent = path.parent
    while not parent.exists():
        missing.append(parent)
        parent = parent.parent
    for directory in reversed(missing):
        try:
            directory.mkdir(mode=0o700)
        except FileExistsError:
            continue
        directory.chmod(0o700)


def _private_database(path: Path, *, exclusive: bool = False) -> None:
    _private_parent(path)
    flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0)
    if exclusive:
        flags |= os.O_EXCL
    descriptor = os.open(path, flags, 0o600)
    try:
        os.fchmod(descriptor, 0o600)
    finally:
        os.close(descriptor)


def _declared_control_catalog(observation: dict, *, secrets: tuple[str, ...] = ()) -> list[str] | None:
    """Read optional host scope metadata; never promote it to legal controls."""
    capabilities = observation.get("_capabilities")
    if not isinstance(capabilities, dict) or "control_catalog" not in capabilities:
        return None
    catalog = capabilities["control_catalog"]
    if (not isinstance(catalog, (list, tuple))
            or not all(isinstance(action, str) and action and action not in {"reset", "finish"} for action in catalog)
            or len(set(catalog)) != len(catalog)
            or any(_public_value(action, secrets) != action for action in catalog)):
        raise ValueError("Declared control catalog must contain unique public native controls, without meta controls.")
    return sorted(catalog)


def _scope_manifest(project_id: str, game_version: str, observation: dict) -> dict:
    visible = public_observation(observation)
    # Initial map contents scope reuse.  They are not compared at every action:
    # player overlays, doors, terrain and scene objects can change legitimately.
    scene = {key: visible[key] for key in ("map", "grid") if key in visible}
    if not scene:
        scene = {"initial_world": {key: visible[key] for key in ("objects", "entities") if key in visible}}
        for path in _identity_paths(observation):
            _drop_path(scene["initial_world"], path.split("."))
    manifest = {
        "version": EXPERIENCE_VERSION,
        "project_id_hash": _digest(project_id),
        "game_version_hash": _digest(game_version),
        "game": visible.get("game"),
        "initial_scene_id": _digest(scene),
        "controls": sorted(visible["controls"]),
        "coordinates": {key: visible[key] for key in ("coordinates", "coordinate_system", "units") if key in visible},
        "grid_shape": _grid_shape(visible),
        "state_identity": _identity_paths(observation),
    }
    catalog = _declared_control_catalog(observation)
    if catalog is not None:
        meta = [action for action in visible["controls"] if action in {"reset", "finish"}]
        complete = sorted([*catalog, *meta])
        if not set(visible["controls"]).issubset(complete):
            raise ScopeMismatch("Current controls are outside the declared complete control catalog.")
        manifest["controls"] = complete
        manifest["control_catalog"] = catalog
    return manifest


class ExperienceStore:
    """Durable, scoped observed transitions with bounded relevance queries.

    ``read_only`` is useful for frozen-memory evaluation: ``record`` still
    returns its factual observation but deliberately never updates that store.
    Same run/step writes are idempotent; a conflicting citation is rejected.
    The SQLite transaction commits evidence and state counters together.
    """

    def __init__(self, path: str | Path, project_id: str, game_version: str,
                 initial_observation: dict, *, read_only: bool = False,
                 secrets: tuple[str, ...] = ()):
        if not isinstance(project_id, str) or not project_id or not isinstance(game_version, str) or not game_version:
            raise ValueError("Project and game version identifiers must be nonempty strings.")
        self.path = Path(path)
        self.read_only = read_only
        self._secrets = tuple(secret for secret in secrets if isinstance(secret, str) and secret)
        initial_observation = self._clean(initial_observation)
        self.manifest = _scope_manifest(project_id, game_version, initial_observation)
        self.scope_id = _digest(self.manifest)
        self._lock = threading.RLock()
        self._closed = False
        if read_only:
            self._db = sqlite3.connect(self.path.resolve().as_uri() + "?mode=ro", uri=True, timeout=30, isolation_level=None)
        else:
            _private_database(self.path)
            self._private_files()
            self._db = sqlite3.connect(self.path, timeout=30, isolation_level=None)
        self._db.row_factory = sqlite3.Row
        try:
            self._db.execute("PRAGMA foreign_keys=ON")
            if not read_only:
                self._db.execute("PRAGMA journal_mode=WAL")
                self._db.execute("PRAGMA synchronous=FULL")
                self._initialize()
                self._private_files()
            version = self._db.execute("SELECT value FROM experience_meta WHERE key='version'").fetchone()
            if version is None or version[0] != EXPERIENCE_VERSION:
                raise ValueError("Unsupported experience database version.")
            existing = self._db.execute("SELECT manifest FROM experience_scopes WHERE scope_id=?", (self.scope_id,)).fetchone()
            if existing is None:
                if read_only:
                    raise ScopeMismatch("No frozen experience exists for this exact initial scope.")
                with self._transaction():
                    self._db.execute("INSERT OR IGNORE INTO experience_scopes(scope_id,manifest) VALUES (?,?)", (self.scope_id, _json(self.manifest)))
                    self._insert_state(initial_observation)
            elif existing[0] != _json(self.manifest):
                raise ScopeMismatch("Experience scope manifest does not match.")
        except BaseException:
            self._db.close()
            self._closed = True
            raise

    def _initialize(self) -> None:
        self._db.executescript("""
            CREATE TABLE IF NOT EXISTS experience_meta(key TEXT PRIMARY KEY,value TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS experience_scopes(scope_id TEXT PRIMARY KEY,manifest TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS experience_states(
                scope_id TEXT NOT NULL,state_id TEXT NOT NULL,anchor TEXT NOT NULL,
                observations INTEGER NOT NULL DEFAULT 0,
                PRIMARY KEY(scope_id,state_id),
                FOREIGN KEY(scope_id) REFERENCES experience_scopes(scope_id));
            CREATE TABLE IF NOT EXISTS experience_transitions(
                seq INTEGER PRIMARY KEY AUTOINCREMENT,scope_id TEXT NOT NULL,
                run_id TEXT NOT NULL,step INTEGER NOT NULL,before_id TEXT NOT NULL,
                action TEXT NOT NULL,after_id TEXT NOT NULL,outcome_id TEXT NOT NULL,
                fact TEXT NOT NULL,
                UNIQUE(scope_id,run_id,step),
                FOREIGN KEY(scope_id) REFERENCES experience_scopes(scope_id));
            CREATE INDEX IF NOT EXISTS experience_query ON experience_transitions(scope_id,before_id,action,after_id);
            CREATE TABLE IF NOT EXISTS experience_feedback(
                id INTEGER PRIMARY KEY AUTOINCREMENT,scope_id TEXT NOT NULL,
                run_id TEXT NOT NULL,step INTEGER NOT NULL,status TEXT NOT NULL,
                comment TEXT NOT NULL,reviewer TEXT NOT NULL,created REAL NOT NULL,
                FOREIGN KEY(scope_id,run_id,step) REFERENCES experience_transitions(scope_id,run_id,step));
            CREATE TABLE IF NOT EXISTS experience_investigation_snapshots(
                seq INTEGER PRIMARY KEY AUTOINCREMENT,
                scope_id TEXT NOT NULL,run_id TEXT NOT NULL,revision INTEGER NOT NULL,
                payload TEXT NOT NULL,created REAL NOT NULL,
                UNIQUE(scope_id,run_id,revision),
                FOREIGN KEY(scope_id) REFERENCES experience_scopes(scope_id));
            CREATE INDEX IF NOT EXISTS experience_investigation_scope
                ON experience_investigation_snapshots(scope_id,seq);
        """)
        self._db.execute("INSERT OR IGNORE INTO experience_meta(key,value) VALUES ('version',?)", (EXPERIENCE_VERSION,))

    def _clean(self, observation: dict) -> dict:
        clean = public_observation(observation, secrets=self._secrets)
        # Identity declarations are trusted adapter metadata, never public text.
        # Keep only the metadata used by this module; credentials cannot be
        # copied into a stored fact through other private capability fields.
        capabilities = observation.get("_capabilities", {})
        clean["_capabilities"] = {
            "reset": not isinstance(capabilities, dict) or capabilities.get("reset") is not False,
            "state_identity": {"transient_paths": _identity_paths(observation)},
        }
        catalog = _declared_control_catalog(observation, secrets=self._secrets)
        if catalog is not None:
            clean["_capabilities"]["control_catalog"] = catalog
        return clean

    @contextmanager
    def _transaction(self) -> Iterator[None]:
        self._db.execute("BEGIN IMMEDIATE")
        try:
            yield
            self._db.execute("COMMIT")
        except BaseException:
            self._db.execute("ROLLBACK")
            raise
        finally:
            self._private_files()

    def _private_files(self) -> None:
        if not self.read_only:
            for path in (self.path, Path(str(self.path) + "-wal"), Path(str(self.path) + "-shm")):
                if path.exists():
                    path.chmod(0o600)

    def _check(self, observation: dict) -> None:
        if self._closed:
            raise ValueError("Experience store is closed.")
        visible = public_observation(observation)
        frame = {key: visible[key] for key in ("coordinates", "coordinate_system", "units") if key in visible}
        if (_json(visible.get("game")) != _json(self.manifest["game"])
                or _json(frame) != _json(self.manifest["coordinates"])
                or _grid_shape(visible) != self.manifest["grid_shape"]
                or _identity_paths(observation) != self.manifest["state_identity"]
                or _declared_control_catalog(observation) != self.manifest.get("control_catalog")
                or not set(visible["controls"]).issubset(self.manifest["controls"])):
            raise ScopeMismatch("Observation game, scene shape, controls, frame or state identity does not match this scope.")

    def _insert_state(self, observation: dict) -> None:
        view = public_observation(observation)
        anchor = {key: _brief(view[key]) for key in ("status", "player") if key in view}
        self._db.execute("""INSERT INTO experience_states(scope_id,state_id,anchor,observations)
            VALUES (?,?,?,1) ON CONFLICT(scope_id,state_id) DO UPDATE SET observations=observations+1""",
                         (self.scope_id, state_key(observation), _json(anchor)))

    def record(self, before: dict, action: str, after: dict, run_id: str, step: int) -> dict:
        before, after = self._clean(before), self._clean(after)
        self._check(before)
        self._check(after)
        fact = experiment_fact(before, action, after, run_id=run_id, step=step)
        before_id, after_id = fact["before_state_id"], fact["after_state_id"]
        # A bounded display excerpt must not collapse two different complete
        # feedback outcomes merely because their first 180 characters match.
        outcome_id = _digest({"after_state_id": after_id, "events": public_observation(after).get("events", [])})
        with self._lock:
            if self.read_only:
                existing = self._db.execute("SELECT fact FROM experience_transitions WHERE scope_id=? AND run_id=? AND step=?", (self.scope_id, run_id, step)).fetchone()
                if existing is not None and existing[0] != _json(fact):
                    raise EvidenceConflict("Evidence citation already identifies a different observed transition.")
                return {**fact, "persisted": False, "already_recorded": existing is not None}
            with self._transaction():
                # The uniqueness check belongs inside the write transaction:
                # independent runner connections may race after a restart.
                existing = self._db.execute("SELECT fact FROM experience_transitions WHERE scope_id=? AND run_id=? AND step=?", (self.scope_id, run_id, step)).fetchone()
                if existing is not None:
                    if existing[0] != _json(fact):
                        raise EvidenceConflict("Evidence citation already identifies a different observed transition.")
                    return {**fact, "persisted": True, "already_recorded": True}
                counts = self._db.execute("""SELECT COUNT(*) AS attempts,
                    COALESCE(SUM(before_id=after_id),0) AS unchanged FROM experience_transitions
                    WHERE scope_id=? AND before_id=? AND action=?""", (self.scope_id, before_id, action)).fetchone()
                new_state = self._db.execute("SELECT 1 FROM experience_states WHERE scope_id=? AND state_id=?", (self.scope_id, after_id)).fetchone() is None
                self._db.execute("""INSERT INTO experience_transitions
                    (scope_id,run_id,step,before_id,action,after_id,outcome_id,fact) VALUES (?,?,?,?,?,?,?,?)""",
                    (self.scope_id, run_id, step, before_id, action, after_id, outcome_id, _json(fact)))
                self._insert_state(before)
                self._insert_state(after)
            return {**fact, "persisted": True, "already_recorded": False,
                    "same_state_action_prior_attempts": counts["attempts"],
                    "prior_equivalent_unchanged_attempts": counts["unchanged"],
                    "new_after_state": new_state}

    def retrieve(self, observation: dict, limit: int = 8, max_bytes: int = 6_000) -> dict:
        """Query aggregated transitions, prioritizing the exact current state.

        Only a bounded number of result groups are materialized.  The full log
        and all reviewer labels stay in SQLite.  Citations use opaque run IDs,
        so same step numbers from different sessions remain distinguishable.
        """
        observation = self._clean(observation)
        self._check(observation)
        if type(limit) is not int or not 0 <= limit <= 64:
            raise ValueError("Retrieval limit must be between 0 and 64.")
        if type(max_bytes) is not int or max_bytes < 512:
            raise ValueError("Retrieval byte budget must be at least 512.")
        current = state_key(observation)
        with self._lock:
            groups = self._db.execute("""SELECT before_id,action,after_id,outcome_id,COUNT(*) AS attempts,
                SUM(before_id=after_id) AS unchanged_count,MAX(seq) AS latest_seq,MIN(seq) AS first_seq
                FROM experience_transitions WHERE scope_id=?
                GROUP BY before_id,action,after_id,outcome_id
                ORDER BY (before_id=?) DESC,latest_seq DESC LIMIT ?""", (self.scope_id, current, limit)).fetchall()
            total = self._db.execute("""SELECT COUNT(*) FROM
                (SELECT 1 FROM experience_transitions WHERE scope_id=? GROUP BY before_id,action,after_id,outcome_id)""", (self.scope_id,)).fetchone()[0]
            result = {"version": EXPERIENCE_VERSION, "scope_id": self.scope_id,
                      "current_state_id": current, "facts": [], "omitted_count": total,
                      "retained_fact_count": 0}
            for group in groups:
                row = self._db.execute("SELECT fact FROM experience_transitions WHERE seq=?", (group["latest_seq"],)).fetchone()
                fact = json.loads(row[0])
                fact.pop("citation", None)
                citations = self._db.execute("""SELECT run_id,MIN(step) AS step,MAX(seq) AS recent FROM experience_transitions
                    WHERE scope_id=? AND before_id=? AND action=? AND after_id=? AND outcome_id=?
                    GROUP BY run_id ORDER BY recent DESC LIMIT ?""",
                    (self.scope_id, group["before_id"], group["action"], group["after_id"], group["outcome_id"], MAX_CITATIONS)).fetchall()
                outcomes = self._db.execute("""SELECT COUNT(DISTINCT outcome_id) FROM experience_transitions
                    WHERE scope_id=? AND before_id=? AND action=?""", (self.scope_id, group["before_id"], group["action"])).fetchone()[0]
                fact.update(attempts=group["attempts"], unchanged_count=group["unchanged_count"],
                            matching_current_state=group["before_id"] == current,
                            different_outcomes_for_action=outcomes,
                            citations=[{"run_id": item["run_id"], "step": item["step"]} for item in citations])
                result["facts"].append(fact)
                result["retained_fact_count"] = len(result["facts"])
                result["omitted_count"] = total - len(result["facts"])
                if _bytes(result) > max_bytes:
                    result["facts"].pop()
                    result["retained_fact_count"] = len(result["facts"])
                    result["omitted_count"] = total - len(result["facts"])
                    # A large recent fact must not hide smaller relevant facts.
                    continue
            return result

    def summary(self) -> dict:
        if self._closed:
            raise ValueError("Experience store is closed.")
        with self._lock:
            row = self._db.execute("""SELECT COUNT(*) AS transitions,COUNT(DISTINCT run_id) AS runs,
                COALESCE(SUM(before_id=after_id),0) AS unchanged_transitions,
                COUNT(DISTINCT action) AS action_types FROM experience_transitions WHERE scope_id=?""", (self.scope_id,)).fetchone()
            states = self._db.execute("SELECT COUNT(*) FROM experience_states WHERE scope_id=?", (self.scope_id,)).fetchone()[0]
            pairs = self._db.execute("SELECT COUNT(*) FROM (SELECT 1 FROM experience_transitions WHERE scope_id=? GROUP BY before_id,action)", (self.scope_id,)).fetchone()[0]
            repeated = self._db.execute("""SELECT COALESCE(SUM(n-1),0) FROM
                (SELECT COUNT(*) AS n FROM experience_transitions WHERE scope_id=? AND before_id=after_id GROUP BY before_id,action)""", (self.scope_id,)).fetchone()[0]
            controls = self._db.execute("SELECT action,COUNT(*) AS n FROM experience_transitions WHERE scope_id=? GROUP BY action ORDER BY action", (self.scope_id,)).fetchall()
            return {"version": EXPERIENCE_VERSION, "scope_id": self.scope_id, **dict(row),
                    "distinct_states": states, "distinct_state_action_pairs": pairs,
                    "repeated_unchanged_attempts": repeated,
                    "control_counts": {item["action"]: item["n"] for item in controls},
                    "read_only": self.read_only}

    def experiment_index(self, observation: dict, *, exclude_run_id: str,
                         max_bytes: int = 4_000) -> dict:
        """Return prior-run experiment counts for currently legal native controls.

        This separate opt-in index never changes the legacy retrieval packet.
        All counts and citations exclude ``exclude_run_id`` so a caller can add
        its current-session observations once.  Sparse rows have exact counts;
        omitted rows are unknown unless ``counts_complete`` is true.  Complete
        zero-control lists are supplied only when both lists fit the budget.
        No reviewer table, hypotheses, hidden rules or action policy is read.
        """
        _validate_evidence(exclude_run_id, 1)
        if _public_value(exclude_run_id, self._secrets) != exclude_run_id:
            raise ValueError("Excluded evidence identifier must not contain a runtime credential.")
        if type(max_bytes) is not int or max_bytes < 1_024:
            raise ValueError("Experiment index byte budget must be at least 1024.")
        observation = self._clean(observation)
        self._check(observation)
        current = state_key(observation)
        controls = [action for action in public_observation(observation)["controls"]
                    if action not in {"reset", "finish"}]

        def encoded_size(value: object) -> int:
            # Match the escaped representation used by the inference client's
            # JSON envelope; this also bounds the smaller literal UTF-8 form.
            return len(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                  ensure_ascii=True, allow_nan=False).encode("utf-8"))

        def citations(action: str, *, exact_state: bool) -> list[dict]:
            condition = " AND before_id=?" if exact_state else ""
            parameters = [self.scope_id, exclude_run_id, action]
            if exact_state:
                parameters.append(current)
            # A few unsafe legacy identifiers must not expose credentials or
            # force an unbounded scan of evidence IDs.  Without a safe citation
            # the action row is omitted, never mislabeled as untested.
            rows = self._db.execute("""SELECT run_id,MIN(step) AS step,MAX(seq) AS recent
                FROM experience_transitions WHERE scope_id=? AND run_id<>? AND action=?"""
                + condition + " GROUP BY run_id ORDER BY recent DESC LIMIT 10", parameters).fetchall()
            safe = []
            for row in rows:
                identifier = row["run_id"]
                if (isinstance(identifier, str) and _RUN_ID.fullmatch(identifier)
                        and _public_value(identifier, self._secrets) == identifier
                        and type(row["step"]) is int and row["step"] > 0):
                    safe.append({"run_id": identifier, "step": row["step"]})
                    if len(safe) == 2:
                        break
            return safe

        with self._lock:
            # A single read transaction keeps aggregate counts and citations
            # consistent even if another runner appends experience concurrently.
            self._db.execute("BEGIN")
            try:
                scope_counts = self._db.execute("""SELECT COUNT(*) AS transitions,
                    COUNT(DISTINCT run_id) AS runs FROM experience_transitions
                    WHERE scope_id=? AND run_id<>?""", (self.scope_id, exclude_run_id)).fetchone()
                aggregates = []
                # Bound SQLite placeholders independently of a game's palette
                # size.  These are aggregates, never materialized history rows.
                for offset in range(0, len(controls), 400):
                    part = controls[offset:offset + 400]
                    placeholders = ",".join("?" for _ in part)
                    aggregates.extend(self._db.execute("""SELECT action,COUNT(*) AS attempts,
                        SUM(before_id=?) AS state_attempts,
                        SUM(before_id=? AND before_id=after_id) AS state_unchanged,
                        COUNT(DISTINCT CASE WHEN before_id=? THEN outcome_id END) AS state_outcomes,
                        MAX(seq) AS recent FROM experience_transitions
                        WHERE scope_id=? AND run_id<>? AND action IN (""" + placeholders + ") GROUP BY action",
                        [current, current, current, self.scope_id, exclude_run_id, *part]).fetchall())
                by_action = {row["action"]: row for row in aggregates}
                packet = {
                    "version": EXPERIMENT_INDEX_VERSION,
                    "scope_id": self.scope_id,
                    "current_state_id": current,
                    "excluded_run_id": exclude_run_id,
                    "control_scope": "currently_legal_native_controls",
                    "prior_run_count": scope_counts["runs"],
                    "prior_transition_count": scope_counts["transitions"],
                    "current_native_control_count": len(controls),
                    "controls": [], "counts_complete": False,
                    "omitted_nonzero_controls": len(aggregates),
                    "zero_lists_complete": False,
                }
                for row in sorted(aggregates, key=lambda item: (item["state_attempts"] > 0, item["recent"]), reverse=True):
                    any_citations = citations(row["action"], exact_state=False)
                    exact_citations = citations(row["action"], exact_state=True) if row["state_attempts"] else []
                    if not any_citations or (row["state_attempts"] and not exact_citations):
                        continue
                    entry = {
                        "action": row["action"], "attempts": row["attempts"],
                        "current_state": {"attempts": row["state_attempts"],
                                          "unchanged_outcomes": row["state_unchanged"],
                                          "distinct_outcomes": row["state_outcomes"]},
                        "citations": {"any_state": any_citations, "current_state": exact_citations},
                    }
                    packet["controls"].append(entry)
                    packet["omitted_nonzero_controls"] -= 1
                    if encoded_size(packet) > max_bytes:
                        packet["controls"].pop()
                        packet["omitted_nonzero_controls"] += 1
                packet["counts_complete"] = packet["omitted_nonzero_controls"] == 0
                if packet["counts_complete"]:
                    global_zero = [action for action in controls if action not in by_action]
                    state_zero = [action for action in controls
                                  if action not in by_action or by_action[action]["state_attempts"] == 0]
                    packet["global_untested_controls"] = global_zero
                    packet["current_state_unattempted_controls"] = state_zero
                    packet["zero_lists_complete"] = True
                    if encoded_size(packet) > max_bytes:
                        del packet["global_untested_controls"]
                        del packet["current_state_unattempted_controls"]
                        packet["zero_lists_complete"] = False
                if encoded_size(packet) > max_bytes:
                    raise ValueError("Experiment index metadata exceeds the configured byte budget.")
                return packet
            finally:
                self._db.execute("ROLLBACK")

    def record_feedback(self, run_id: str, step: int, status: str, *, comment: str = "", reviewer: str = "") -> int:
        """Append reviewer-only feedback.  Explorer retrieval never reads it."""
        _validate_evidence(run_id, step)
        if status not in {"pending", "confirmed", "expected"}:
            raise ValueError("Feedback status must be pending, confirmed or expected.")
        if self.read_only or self._closed:
            raise ValueError("Feedback requires an open writable store.")
        if not isinstance(comment, str) or not isinstance(reviewer, str):
            raise ValueError("Feedback comment and reviewer must be strings.")
        with self._lock, self._transaction():
            if self._db.execute("SELECT 1 FROM experience_transitions WHERE scope_id=? AND run_id=? AND step=?", (self.scope_id, run_id, step)).fetchone() is None:
                raise ValueError("Feedback must cite an executed transition in this scope.")
            cursor = self._db.execute("""INSERT INTO experience_feedback
                (scope_id,run_id,step,status,comment,reviewer,created) VALUES (?,?,?,?,?,?,?)""",
                (self.scope_id, run_id, step, status, _public_value(comment, self._secrets)[:2_000], _public_value(reviewer, self._secrets)[:128], time.time()))
            return cursor.lastrowid

    def _investigation_links(self, rows: list[dict]) -> list[dict]:
        """Validate scoped execution citations, without endorsing model claims."""
        from .investigation import citation

        references = []
        seen = set()
        for row in rows:
            if self._db.execute("SELECT 1 FROM experience_states WHERE scope_id=? AND state_id=?",
                                (self.scope_id, row["origin_state_id"])).fetchone() is None:
                raise ValueError("Investigation origin must be an observed state in this scope.")
            # Last feedback/counterexamples are most useful when a question is
            # resumed; this selects evidence, never a control or a hypothesis.
            for reference in [row["last_transition"], *reversed(row["counterexamples"]), *reversed(row["evidence"])]:
                reference = citation(reference, secrets=self._secrets)
                key = (reference["run_id"], reference["step"])
                if self._db.execute("SELECT 1 FROM experience_transitions WHERE scope_id=? AND run_id=? AND step=?",
                                    (self.scope_id, *key)).fetchone() is None:
                    raise ValueError("Investigation citation must identify an executed transition in this scope.")
                if key not in seen:
                    seen.add(key)
                    references.append(reference)
        return references

    def _investigation_payload(self, payload: str) -> dict:
        from .investigation import validate_rows

        value = json.loads(payload)
        if (not isinstance(value, dict) or set(value) != {"version", "parent_snapshot", "investigations"}
                or value["version"] != INVESTIGATION_STORE_VERSION):
            raise ValueError("Unsupported investigation archive payload.")
        value["investigations"] = validate_rows(value["investigations"], secrets=self._secrets)
        return value

    def save_investigations(self, run_id: str, revision: int, investigations: list[dict], *,
                            parent_snapshot: dict | None = None) -> dict:
        """Append a bounded snapshot of tentative model-authored questions.

        Snapshot references and origin states prove recorded execution only.
        The controller copies condition values from observations and replay
        checks those copies; this bounded archive cannot reconstruct full world
        snapshots independently.  Counterexamples survive same-ID revisions and
        explicit imported-parent lineage.  No row is promoted to a learned rule.
        """
        from .investigation import citation, validate_rows

        if self._closed:
            raise ValueError("Experience store is closed.")
        _validate_evidence(run_id, 1)
        if _public_value(run_id, self._secrets) != run_id:
            raise ValueError("Investigation source identifier contains private data.")
        if type(revision) is not int or revision < 1:
            raise ValueError("Investigation revision must be a positive integer.")
        result = {"version": INVESTIGATION_STORE_VERSION, "scope_id": self.scope_id,
                  "run_id": run_id, "revision": revision, "persisted": False,
                  "already_recorded": False, "counterexamples_retained": 0}
        # Evaluation outcomes deliberately do not enter frozen training memory.
        # Their newly executed citations need not exist in this read-only DB.
        if self.read_only:
            return result
        rows = validate_rows(investigations, secrets=self._secrets)
        if parent_snapshot is not None:
            if not isinstance(parent_snapshot, dict) or set(parent_snapshot) != {"run_id", "revision"}:
                raise ValueError("Invalid investigation parent snapshot reference.")
            citation({"run_id": parent_snapshot["run_id"], "step": parent_snapshot["revision"]}, secrets=self._secrets)
            if parent_snapshot["run_id"] == run_id:
                raise ValueError("Imported investigation parent must belong to another run.")
            parent_snapshot = dict(parent_snapshot)
        with self._lock, self._transaction():
            existing = self._db.execute("SELECT payload FROM experience_investigation_snapshots WHERE scope_id=? AND run_id=? AND revision=?",
                                        (self.scope_id, run_id, revision)).fetchone()
            latest = self._db.execute("SELECT MAX(revision) FROM experience_investigation_snapshots WHERE scope_id=? AND run_id=?",
                                      (self.scope_id, run_id)).fetchone()[0]
            if existing is None and latest is not None and revision <= latest:
                raise ValueError("Investigation revisions cannot move backwards.")
            predecessors = []
            if parent_snapshot is not None:
                parent = self._db.execute("SELECT payload FROM experience_investigation_snapshots WHERE scope_id=? AND run_id=? AND revision=?",
                                          (self.scope_id, parent_snapshot["run_id"], parent_snapshot["revision"])).fetchone()
                if parent is None:
                    raise ValueError("Investigation parent snapshot is missing from this scope.")
                predecessors.extend(self._investigation_payload(parent[0])["investigations"])
            previous = self._db.execute("SELECT payload FROM experience_investigation_snapshots WHERE scope_id=? AND run_id=? AND revision<? ORDER BY revision DESC LIMIT 1",
                                        (self.scope_id, run_id, revision)).fetchone()
            if previous is not None:
                predecessors.extend(self._investigation_payload(previous[0])["investigations"])
            for row in rows:
                accumulated = []
                for old in predecessors:
                    if old["id"] == row["id"]:
                        for reference in old["counterexamples"]:
                            if reference not in accumulated:
                                accumulated.append(reference)
                for reference in row["counterexamples"]:
                    if reference not in accumulated:
                        accumulated.append(reference)
                result["counterexamples_retained"] += sum(reference not in row["counterexamples"] for reference in accumulated)
                row["counterexamples"] = accumulated
            # Overflow rejects this revision instead of forgetting negative
            # evidence.  The caller can retain its previous valid working set.
            rows = validate_rows(rows, secrets=self._secrets)
            self._investigation_links(rows)
            payload = _json({"version": INVESTIGATION_STORE_VERSION,
                             "parent_snapshot": parent_snapshot, "investigations": rows})
            if existing is not None:
                if existing[0] != payload:
                    raise EvidenceConflict("Investigation revision already identifies a different snapshot.")
                return {**result, "persisted": True, "already_recorded": True}
            self._db.execute("""INSERT INTO experience_investigation_snapshots
                (scope_id,run_id,revision,payload,created) VALUES (?,?,?,?,?)""",
                (self.scope_id, run_id, revision, payload, time.time()))
            return {**result, "persisted": True}

    def investigation_index(self, observation: dict, *, exclude_run_id: str,
                            max_bytes: int = 4_000) -> dict:
        """Retrieve the latest prior-run tentative queue and cited fact anchors.

        A latest empty snapshot stays empty; older unfinished questions are not
        resurrected.  Complete rows and counterexamples are atomic under the
        byte budget.  Applicability compares selected public values only, never
        proves an explanation or that unobserved conditions are irrelevant.
        """
        from .investigation import INVESTIGATION_INDEX_VERSION, applicability, byte_size, citation

        _validate_evidence(exclude_run_id, 1)
        if _public_value(exclude_run_id, self._secrets) != exclude_run_id:
            raise ValueError("Excluded investigation identifier contains private data.")
        if type(max_bytes) is not int or max_bytes < 1_024:
            raise ValueError("Investigation index byte budget must be at least 1024.")
        observation = self._clean(observation)
        self._check(observation)
        current = state_key(observation)
        packet = {"version": INVESTIGATION_INDEX_VERSION, "status": "available",
                  "scope_id": self.scope_id, "current_state_id": current,
                  "excluded_run_id": exclude_run_id, "source_run_id": None,
                  "revision": 0, "investigations": [], "applicability": {},
                  "complete": True, "omitted_count": 0,
                  "evidence_facts": [], "omitted_fact_count": 0}
        with self._lock:
            self._db.execute("BEGIN")
            try:
                if self._db.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='experience_investigation_snapshots'").fetchone() is None:
                    return {**packet, "status": "unavailable", "complete": False,
                            "reason": "archive_schema_unavailable"}
                source = self._db.execute("SELECT run_id,revision,payload FROM experience_investigation_snapshots WHERE scope_id=? AND run_id<>? ORDER BY seq DESC LIMIT 1",
                                          (self.scope_id, exclude_run_id)).fetchone()
                if source is None:
                    return packet
                citation({"run_id": source["run_id"], "step": source["revision"]}, secrets=self._secrets)
                payload = self._investigation_payload(source["payload"])
                rows = sorted(payload["investigations"], key=lambda row: {"active": 0, "pending": 1, "closed": 2}[row["status"]])
                references = self._investigation_links(rows)
                packet.update(source_run_id=source["run_id"], revision=source["revision"],
                              investigations=rows,
                              applicability={row["id"]: applicability(row, observation, current, secrets=self._secrets) for row in rows},
                              omitted_fact_count=len(references))
                while packet["investigations"] and byte_size(packet) > max_bytes:
                    removed = packet["investigations"].pop()
                    del packet["applicability"][removed["id"]]
                    packet["omitted_count"] += 1
                    packet["complete"] = False
                retained = {(reference["run_id"], reference["step"])
                            for reference in self._investigation_links(packet["investigations"])}
                packet["omitted_fact_count"] = len(retained)
                for reference in references:
                    if len(packet["evidence_facts"]) >= 3:
                        break
                    if (reference["run_id"], reference["step"]) not in retained:
                        continue
                    stored = self._db.execute("SELECT fact FROM experience_transitions WHERE scope_id=? AND run_id=? AND step=?",
                                              (self.scope_id, reference["run_id"], reference["step"])).fetchone()
                    fact = _public_value(json.loads(stored[0]), self._secrets)
                    if not isinstance(fact, dict) or fact.get("version") != FACT_VERSION or _json(fact.get("citation")) != _json(reference):
                        raise ValueError("Investigation factual anchor does not match its executed citation.")
                    packet["evidence_facts"].append(fact)
                    packet["omitted_fact_count"] -= 1
                    if byte_size(packet) > max_bytes:
                        packet["evidence_facts"].pop()
                        packet["omitted_fact_count"] += 1
                if byte_size(packet) > max_bytes:
                    raise ValueError("Investigation provenance exceeds the configured byte budget.")
                return packet
            finally:
                self._db.execute("ROLLBACK")

    def reviewer_feedback(self, *, limit: int = 100) -> list[dict]:
        """Explicit reviewer API; do not include this in exploration context."""
        if self._closed:
            raise ValueError("Experience store is closed.")
        if type(limit) is not int or not 1 <= limit <= 1_000:
            raise ValueError("Feedback limit must be between 1 and 1000.")
        with self._lock:
            return [dict(row) for row in self._db.execute("""SELECT id,run_id,step,status,comment,reviewer,created
                FROM experience_feedback WHERE scope_id=? ORDER BY id DESC LIMIT ?""", (self.scope_id, limit))]

    def snapshot(self, destination: str | Path) -> Path:
        """Make a consistent frozen database, including committed WAL contents."""
        destination = Path(destination)
        if self._closed or destination.exists() or destination.resolve() == self.path.resolve():
            raise ValueError("Snapshot needs an open store and a new destination.")
        _private_database(destination, exclusive=True)
        with self._lock:
            target = sqlite3.connect(destination)
            try:
                self._db.backup(target)
            finally:
                target.close()
        return destination

    def close(self) -> None:
        if not self._closed:
            self._db.close()
            self._closed = True

    def __enter__(self) -> "ExperienceStore":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
