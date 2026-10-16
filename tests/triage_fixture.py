"""Real adapter recordings and provider replies for offline Ultra triage tests."""

from copy import deepcopy
import json
from pathlib import Path

from autoplay_qa.game import KeyGateGame
from tests.continuous_fixture import FakeContinuousGame


PRIVATE_SENTINELS = (
    "PRIVATE_EVALUATION_MARKER",
    "PRIVATE_RULE_MARKER",
    "PRIVATE_REASON_MARKER",
    "/private/developer/game-project",
)


def adapter_recording(game, actions, *, defect=None):
    records = [{
        "type": "session",
        "schema_version": 2,
        "task": "explore",
        "mode": "offline-triage-fixture",
        "game_version": game.version,
        "seed": game.seed,
        "initial_observation": deepcopy(game.observe()),
        "unity_project": PRIVATE_SENTINELS[3],
        "evaluation": {
            "injected_defect": {
                "name": defect,
                "description": PRIVATE_SENTINELS[0],
            } if defect else None,
        },
    }]
    for step, action in enumerate(actions, 1):
        before = deepcopy(game.observe())
        after = deepcopy(game.act(action))
        before.setdefault("rules", []).append(PRIVATE_SENTINELS[1])
        after.setdefault("rules", []).append(PRIVATE_SENTINELS[1])
        records.append({
            "type": "step",
            "step": step,
            "episode": 0,
            "before": before,
            "decision": {
                "action": action,
                "reason": PRIVATE_SENTINELS[2],
                "expected_position": None,
                "notes": [],
            },
            "after": after,
            "evaluation": {"private": PRIVATE_SENTINELS[0]},
        })
    return records


def grid_recording(*, defective=True, suspected=False):
    defect = "gate-keeps-key" if defective else None
    records = adapter_recording(
        KeyGateGame(0, defect),
        ("south", "south", "east", "east", "interact"),
        defect=defect,
    )
    if suspected:
        records.append({
            "type": "memory",
            "hypotheses": [{
                "id": "gate-inventory",
                "kind": "suspected_bug",
                "claim": "The key remains after feedback says it was used.",
                "status": "supported",
                "evidence_steps": [5],
            }],
        })
    return records


def continuous_recording():
    return adapter_recording(
        FakeContinuousGame(5, "floor-gap"),
        ("north", "north", "north", "north", "east", "north", "north"),
        defect="floor-gap",
    )


def blocked_note_recording(*, valid_reference=True):
    records = adapter_recording(KeyGateGame(0), ("north",))
    records.append({
        "type": "memory",
        "hypotheses": [{
            "id": "blocked-move",
            "kind": "suspected_bug",
            "claim": "Moving north did not move the player; that may be a bug.",
            "status": "uncertain",
            "evidence_steps": [1] if valid_reference else [999],
        }],
    })
    return records


def assessment(*, verdict="bug", severity="high", evidence_steps=(5,), **changes):
    value = {
        "verdict": verdict,
        "severity": severity,
        "title": "Key retained after the gate used it",
        "player_impact": "The player can keep a supposedly consumed resource.",
        "rationale": "The observed feedback and inventory contradict one another.",
        "evidence_steps": list(evidence_steps),
        "limitations": ["Only this recorded session has been inspected."],
    }
    value.update(changes)
    return value


def provider_reply(value=None, *, finish_reason="stop", usage=True):
    result = {
        "id": "offline-ultra-triage",
        "model": "nvidia/Nemotron-3-Ultra-550b-a55b",
        "choices": [{
            "message": {"content": json.dumps(value or assessment())},
            "finish_reason": finish_reason,
        }],
    }
    if usage is True:
        result["usage"] = {
            "prompt_tokens": 100,
            "completion_tokens": 50,
            "total_tokens": 150,
        }
    elif isinstance(usage, dict):
        result["usage"] = usage
    return result


def write_recording(path: Path, records):
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in records),
        encoding="utf-8",
    )
    return path
