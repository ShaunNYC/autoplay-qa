"""Portable helpers for local semantic regression execution.

This is a small extraction of the original encoding and editor-scope helpers,
kept under the existing module name for regression CLI compatibility. It does
not launch a study, load credentials, reserve a budget, or judge defect truth.
The extraction is packaging work, not evidence of a new testing effect.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re


def encoded(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def digest(value):
    return hashlib.sha256(encoded(value).encode()).hexdigest()


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def editor_control(control):
    """Scope restriction, not game logic: omit map/navigation/timed-mode UI."""
    name = control.get("name", "")
    return not re.search(r"^(Cell |Inspect matrix|Mission map$|Matrix values$|Matrix data |Open full matrix|Workspace$|Review$|Sources$|Robots|40-minute mock$|Package library|Start over|Reset session|Test helper across training maps$|Official practice guidance)", name, re.I)


class EditorScopeAdapter:
    def __init__(self, adapter):
        self.adapter = adapter

    def _scope(self, value):
        return {**value, "controls": [c for c in value.get("controls", []) if editor_control(c)]}

    def observe(self):
        return self._scope(self.adapter.observe())

    def set_timeout(self, remaining):
        self.adapter.timeout = min(self.adapter.timeout, remaining)

    def act(self, action):
        if action.get("kind") != "wait" and action.get("target") not in {c["id"] for c in self.observe()["controls"]}:
            raise ValueError("Control is outside the frozen editor workflow scope.")
        return self._scope(self.adapter.act(action))
