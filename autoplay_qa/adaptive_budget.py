"""Explicit supplemental authorization in the existing durable allowance.

The dashboard's original cap and unknown reservations remain intact. A named,
immutable authorization adds a bounded amount for a particular validation round;
it never resets an old allowance or grants the dashboard extra spending power.
"""

from copy import deepcopy
import math
from pathlib import Path
import re

from .dashboard_runner import BudgetExceeded, TotalAllowance, _atomic_json, _now


class SupplementalAllowance(TotalAllowance):
    def __init__(self, path: Path, *, round_id: str = "adaptive-20261014",
                 round_cap_usd: float = 15, authorize: bool = False):
        if not isinstance(round_id, str) or not re.fullmatch(r"[a-zA-Z0-9_-]{1,96}", round_id):
            raise ValueError("Invalid authorization round identifier.")
        if type(round_cap_usd) not in (int, float) or not math.isfinite(round_cap_usd) or not 0 < round_cap_usd <= 15:
            raise ValueError("Supplemental round must be bounded by the authorized $15.")
        self.round_id, self.round_cap_usd = round_id, float(round_cap_usd)
        if not Path(path).is_file():
            raise ValueError("The original durable allowance must exist; a missing ledger cannot reset authorization.")
        # Keep the old on-line runner compatible with the same file and cap.
        super().__init__(path, 10)
        with self._locked():
            state = self._load()
            rounds = state.setdefault("supplemental_rounds", {})
            if not isinstance(rounds, dict):
                raise ValueError("Invalid supplemental authorization records.")
            if any(not isinstance(row, dict) or type(row.get("cap_usd")) not in (int, float)
                   or not math.isfinite(row["cap_usd"]) or not 0 < row["cap_usd"] <= 15
                   for row in rounds.values()):
                raise ValueError("Invalid supplemental authorization records.")
            if round_id not in rounds:
                if not authorize:
                    raise ValueError("This round has no explicit supplemental authorization.")
                if sum(row.get("cap_usd", 0) for row in rounds.values()) + self.round_cap_usd > 15:
                    raise ValueError("The single supplemental authorization totals at most $15; changing its name cannot renew it.")
                rounds[round_id] = {"cap_usd": self.round_cap_usd, "authorized_at": _now()}
                _atomic_json(self.path, state)
            if rounds[round_id].get("cap_usd") != self.round_cap_usd:
                raise ValueError("An existing supplemental authorization cannot be changed.")

    def _rounds(self, state):
        rounds = state.get("supplemental_rounds", {})
        if not isinstance(rounds, dict) or any(
            not isinstance(row, dict) or type(row.get("cap_usd")) not in (int, float)
            or not math.isfinite(row["cap_usd"]) or not 0 < row["cap_usd"] <= 15
            for row in rounds.values()
        ):
            raise ValueError("Invalid supplemental authorization records.")
        if self.round_id not in rounds or rounds[self.round_id]["cap_usd"] != self.round_cap_usd:
            raise ValueError("Supplemental authorization is missing or incompatible.")
        if sum(row["cap_usd"] for row in rounds.values()) > 15:
            raise ValueError("Supplemental authorizations exceed the user's single $15 bound.")
        return rounds

    def reserve(self, job_id: str, max_usd: float):
        if not isinstance(job_id, str) or not re.fullmatch(r"[a-zA-Z0-9_-]{1,128}", job_id):
            raise ValueError("Invalid validation identifier.")
        if type(max_usd) not in (int, float) or not math.isfinite(max_usd) or not 0 < max_usd <= self.round_cap_usd:
            raise ValueError("Invalid validation cap.")
        with self._locked():
            state = self._load()
            rounds = self._rounds(state)
            jobs = state["jobs"]
            if job_id in jobs:
                raise ValueError("Validation already reserved; do not retry a previous invocation.")
            occupied = sum(job["accounted_usd"] for job in jobs.values())
            round_occupied = sum(job["accounted_usd"] for job in jobs.values()
                                 if job.get("authorization_round") == self.round_id)
            if round_occupied + max_usd > self.round_cap_usd:
                raise BudgetExceeded("Supplemental round allowance reached; no provider request was sent.")
            if occupied + max_usd > state["cap_usd"] + sum(row["cap_usd"] for row in rounds.values()):
                raise BudgetExceeded("Combined authorized allowance reached; no provider request was sent.")
            jobs[job_id] = {"reservation_usd": max_usd, "accounted_usd": max_usd,
                            "usage_known": False, "status": "reserved",
                            "authorization_round": self.round_id}
            _atomic_json(self.path, state)

    def settle(self, job_id: str, summary: dict):
        # TotalAllowance.settle holds unknown entire-invocation reservations.
        with self._locked():
            state = self._load()
            self._rounds(state)
            if state["jobs"].get(job_id, {}).get("authorization_round") != self.round_id:
                raise ValueError("Cannot settle a reservation from another authorization.")
        super().settle(job_id, summary)

    def summary(self):
        with self._locked():
            state = self._load()
            self._rounds(state)
            jobs = [deepcopy(row) for row in state["jobs"].values()
                    if row.get("authorization_round") == self.round_id]
        return {"round_id": self.round_id, "cap_usd": self.round_cap_usd,
                "occupied_usd": round(sum(row["accounted_usd"] for row in jobs), 8),
                "unknown_invocations": sum(not row["usage_known"] for row in jobs),
                "invocations": len(jobs)}
