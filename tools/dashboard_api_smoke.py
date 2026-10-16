"""Offline-provider integration checks against the loopback dashboard preview."""
import argparse
import json
from pathlib import Path
import urllib.error
import urllib.request
from urllib.parse import urlsplit
import uuid


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:5173")
    parser.add_argument("--secrets-file", type=Path, default=Path(".dashboard/secrets.json"))
    args = parser.parse_args()
    url = urlsplit(args.url)
    if url.scheme != "http" or url.hostname not in {"localhost", "127.0.0.1", "::1"}:
        raise ValueError("Smoke fixtures are allowed only in a loopback preview.")
    secrets = json.loads(args.secrets_file.read_text())
    checks = 0

    def call(method, path, data=None, role=None, expected=200, extra=None):
        nonlocal checks
        headers = {"Content-Type": "application/json", **(extra or {})}
        if role:
            headers["Authorization"] = "Bearer " + secrets[role]
        request = urllib.request.Request(args.url.rstrip("/") + path,
            data=None if data is None else json.dumps(data).encode(), headers=headers, method=method)
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                status, result = response.status, json.load(response)
        except urllib.error.HTTPError as response:
            status = response.code
            try:
                result = json.loads(response.read())
            except ValueError:
                result = {}
        assert status == expected, (method, path, status, result.get("error"))
        checks += 1
        return result

    public = call("GET", "/api/runs")
    assert public["canControl"] is False and len(public["runs"]) >= 7
    recorded = call("GET", "/api/runs/unity-key")
    assert recorded["summary"]["source"] == "recorded" and recorded["issues"][0]["needsHumanReview"]
    request = {"requestId": str(uuid.uuid4()), "game": "unity", "profile": "wall", "seed": 5, "maxUsd": 0.1}
    call("POST", "/api/runs", request, expected=403)
    call("POST", "/api/runs", request, expected=403, extra={"oai-authenticated-user-id": "local_seedy", "oai-authenticated-user-email": "seedy@sites.test"})
    call("POST", "/api/runs", request, role="ADMIN_TOKEN", expected=403, extra={"Origin": "https://different.example"})
    call("POST", "/api/runner/claim", {"runnerId": "smoke_runner"}, expected=403)
    call("POST", "/api/runner/heartbeat", {"runnerId": "smoke_runner", "name": "Mac runner", "unityReady": True}, role="RUNNER_TOKEN")
    job_id = call("POST", "/api/runs", request, role="ADMIN_TOKEN", expected=201)["id"]
    assert call("POST", "/api/runs", request, role="ADMIN_TOKEN")["id"] == job_id
    call("POST", "/api/runs", {**request, "seed": 0}, role="ADMIN_TOKEN", expected=409)
    call("POST", "/api/runs", {**request, "requestId": str(uuid.uuid4())}, role="ADMIN_TOKEN", expected=409)
    job = call("POST", "/api/runner/claim", {"runnerId": "smoke_runner"}, role="RUNNER_TOKEN")["job"]
    assert job["id"] == job_id and job["seed"] == 5
    lease = {"runnerId": "smoke_runner", "jobId": job_id, "leaseId": job["leaseId"]}
    call("POST", "/api/runner/control", {**lease, "leaseId": str(uuid.uuid4())}, role="RUNNER_TOKEN", expected=409)
    snapshot = call("GET", "/api/runs/" + job_id)
    snapshot["notes"] = ["Offline-provider HTTP integration fixture. No gameplay or model requests executed."]
    call("POST", "/api/runner/update", {**lease, "revision": 1, "snapshot": snapshot}, role="RUNNER_TOKEN")
    changed = json.loads(json.dumps(snapshot)); changed["summary"]["name"] = "Stale update must be ignored"
    call("POST", "/api/runner/update", {**lease, "revision": 1, "snapshot": changed}, role="RUNNER_TOKEN")
    assert call("GET", "/api/runs/" + job_id)["summary"]["name"] != changed["summary"]["name"]
    call("POST", "/api/runs/" + job_id + "/stop", {}, role="ADMIN_TOKEN")
    assert call("POST", "/api/runner/control", lease, role="RUNNER_TOKEN")["cancelRequested"]
    snapshot["summary"].update(status="cancelled", phase="complete")
    call("POST", "/api/runner/update", {**lease, "revision": 2, "snapshot": snapshot}, role="RUNNER_TOKEN")
    snapshot["summary"].update(status="running", phase="explore")
    call("POST", "/api/runner/update", {**lease, "revision": 3, "snapshot": snapshot}, role="RUNNER_TOKEN", expected=409)
    next_id = call("POST", "/api/runs", {**request, "requestId": str(uuid.uuid4())}, role="ADMIN_TOKEN", expected=201)["id"]
    assert call("POST", "/api/runs/" + next_id + "/stop", {}, role="ADMIN_TOKEN")["status"] == "cancelled"
    assert call("POST", "/api/runner/claim", {"runnerId": "smoke_runner"}, role="RUNNER_TOKEN")["job"] is None
    call("POST", "/api/runs", {**request, "command": "forbidden"}, role="ADMIN_TOKEN", expected=400)
    print(json.dumps({"checks": checks, "passed": True, "model_requests": 0}))


if __name__ == "__main__":
    main()
