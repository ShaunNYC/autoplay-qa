import assert from "node:assert/strict";
import { test } from "node:test";
import {
  ApiError, LIMITS, MODELS, MAX_SNAPSHOT_BYTES, matchesBearer, queuedSnapshot,
  reportMarkdown, sameOrigin, validateNewRun, validateRevision, validateSnapshot,
} from "../lib/api-core.ts";

const NEW_RUN = {
  requestId: "fixture-request-001", game: "grid", profile: "clean", seed: 0, maxUsd: 1,
};
const JOB = { id: "fixture-job-001", game: "grid", seed: 0, max_usd: 1 };
const NOW = "2026-10-14T16:00:00Z";

function snapshot() {
  const value = queuedSnapshot(JOB.id, NEW_RUN, NOW);
  Object.assign(value.summary, {
    status: "running", phase: "sweep", controls: 1, issueCount: 1, pendingCount: 1,
  });
  const before = { tick: 0, player: { position: [1, 1], inventory: [] }, events: [] };
  const after = { tick: 1, player: { position: [1, 0], inventory: [] }, events: ["Moved north."] };
  value.scene = { ...before, map: ["###", "#.#", "###"], legend: { "#": "wall", ".": "floor" } };
  value.frames = [{
    id: "fixture-job-001-sweep--1", step: -1, phaseStep: 1, phase: "sweep", action: "north",
    reason: "Public collision coverage", before, after, feedback: ["Moved north."], elapsedSeconds: 1,
  }];
  value.issues = [{
    id: "fixture-job-001-issue-1", number: 1, title: "Movement entered wall tile",
    source: "software_anomaly", anomalyType: "entered_wall_labeled_tile", evidenceSteps: [-1],
    expected: { solid: true }, actual: { position: [1, 0] },
    reproduction: { actions: ["north"], sourceSteps: [-1], verified: null, minimal: false },
    assessment: { status: "pending", evidenceSteps: [], limitations: ["This session only."], warnings: [] },
    needsHumanReview: true, reviewNote: "Awaiting a tentative AI assessment.",
  }];
  return value;
}

function apiFailure(action, status = 400) {
  assert.throws(action, error => error instanceof ApiError && error.status === status);
}

test("new jobs accept only the advertised games, profiles, seeds and authorized budget", () => {
  for (const game of ["grid", "unity"]) {
    for (const profile of ["clean", "wall", "floor", "key", "reset", "door"]) {
      for (const seed of [0, 1, 5]) {
        const job = { ...NEW_RUN, game, profile, seed };
        assert.deepEqual(validateNewRun(job), job);
      }
    }
  }
  assert.equal(validateNewRun({ ...NEW_RUN, maxUsd: 0.05 }).maxUsd, 0.05);
  assert.equal(validateNewRun({ ...NEW_RUN, maxUsd: 10 }).maxUsd, 10);
  apiFailure(() => validateNewRun({ ...NEW_RUN, maxUsd: 10.01 }));
  apiFailure(() => validateNewRun({ ...NEW_RUN, maxUsd: 2 }, 1));
});

test("a job cannot smuggle shell commands, paths, endpoints or engine settings", () => {
  for (const [key, value] of Object.entries({
    command: "curl https://example.invalid", path: "../../private", unityProject: "/private/project",
    endpoint: "https://example.invalid", api_key: "PRIVATE_KEY", maxRequests: 1000,
  })) apiFailure(() => validateNewRun({ ...NEW_RUN, [key]: value }));
});

test("JSON array and coerced values cannot pass the job allowlist", () => {
  for (const [key, values] of Object.entries({
    game: [["grid"], "Grid", null, {}], profile: [["clean"], null, "floor-gap"],
    seed: [true, "0", -1, 2, null], maxUsd: [true, "1", 0, NaN, Infinity, -1],
    requestId: ["../private", "", null],
  })) {
    for (const value of values) apiFailure(() => validateNewRun({ ...NEW_RUN, [key]: value }));
  }
  for (const value of [null, [], "job", 1]) apiFailure(() => validateNewRun(value));
});

test("runner and owner bearer authentication requires an exact configured long secret", () => {
  const secret = "TEST_BEARER_SECRET_1234567890_abcdefghi";
  assert.equal(matchesBearer(`Bearer ${secret}`, secret), true);
  for (const header of [null, secret, `bearer ${secret}`, `Bearer ${secret}x`, "Bearer wrong", `Bearer ${secret.slice(1)}`]) {
    assert.equal(matchesBearer(header, secret), false);
  }
  assert.equal(matchesBearer("Bearer short", "short"), false);
  assert.equal(matchesBearer(`Bearer ${secret}`, undefined), false);
});

test("owner browser writes reject a different origin while CLI bearer calls remain possible", () => {
  assert.equal(sameOrigin(new Request("https://demo.example/api/runs", { headers: { origin: "https://demo.example" } })), true);
  assert.equal(sameOrigin(new Request("https://demo.example/api/runs", { headers: { origin: "https://other.example" } })), false);
  assert.equal(sameOrigin(new Request("https://demo.example/api/runs", { headers: { origin: "null" } })), false);
  assert.equal(sameOrigin(new Request("https://demo.example/api/runs")), true);
});

test("progress revisions are positive safe integers and preserve replay identity", () => {
  for (const revision of [1, 2, Number.MAX_SAFE_INTEGER]) assert.equal(validateRevision(revision), revision);
  for (const revision of [0, -1, 1.5, Number.MAX_SAFE_INTEGER + 1, NaN, Infinity, "1", true, null]) {
    apiFailure(() => validateRevision(revision));
  }
  const value = validateSnapshot(snapshot(), JOB);
  assert.equal(value.frames[0].step, -1);
  assert.equal(value.frames[0].phaseStep, 1);
  assert.deepEqual(value.issues[0].reproduction.sourceSteps, [-1]);
});

test("progress identity, live source, budget and engine limits stay bound to the claimed job", () => {
  for (const [field, invalid] of Object.entries({ id: "wrong-job-id", game: "unity", seed: 5, source: "recorded", budgetUsd: 10 })) {
    const value = snapshot(); value.summary[field] = invalid;
    apiFailure(() => validateSnapshot(value, JOB));
  }
  for (const key of Object.keys(LIMITS)) {
    const value = snapshot(); value.limits[key] += 1;
    apiFailure(() => validateSnapshot(value, JOB));
  }
  for (const key of Object.keys(MODELS)) {
    const value = snapshot(); value.models[key] = "other/model";
    apiFailure(() => validateSnapshot(value, JOB));
  }
  for (const [key, invalid] of [["status", ["running"]], ["phase", ["sweep"]]]) {
    const value = snapshot(); value.summary[key] = invalid;
    apiFailure(() => validateSnapshot(value, JOB));
  }
});

test("out-of-budget control/request counts and contradictory displayed counts are rejected", () => {
  for (const [key, invalid] of Object.entries({ controls: 425, resets: 425, requests: 37, issueCount: 101, pendingCount: 101, warningCount: 10001 })) {
    const value = snapshot(); value.summary[key] = invalid;
    apiFailure(() => validateSnapshot(value, JOB));
  }
  const badControls = snapshot(); badControls.summary.controls = 3;
  apiFailure(() => validateSnapshot(badControls, JOB));
  const badResets = snapshot(); badResets.summary.resets = 2;
  apiFailure(() => validateSnapshot(badResets, JOB));
  const badIssues = snapshot(); badIssues.summary.issueCount = 2;
  apiFailure(() => validateSnapshot(badIssues, JOB));
});

test("nested hidden labels, rules, credentials and model thinking cannot be shared", () => {
  for (const key of [
    "api_key", "authorization", "token", "reasoning_content", "raw_response", "developer_rules",
    "evaluation", "injected_defect", "ground_truth", "expected_verdicts", "source_recording",
    "unity_project", "developer_route", "victory_strategy", "private_harness", "secret_config",
  ]) {
    const value = snapshot(); value.frames[0].before.objects = { hidden: { [key]: "PRIVATE_DATA" } };
    apiFailure(() => validateSnapshot(value, JOB));
  }
});

test("local paths and provider connection details are rejected in public text", () => {
  for (const secret of [
    "/Users/person/game", "/home/person/game", "/private/developer/project", "/tmp/private-file",
    "C:\\Users\\person\\game", "https://api.tokenfactory.nebius.com/v1/",
  ]) {
    const value = snapshot(); value.notes.push(`Accidental connection detail: ${secret}`);
    apiFailure(() => validateSnapshot(value, JOB));
  }
});

test("malformed nested frames and issues cannot reach the read-only UI", () => {
  const noAction = snapshot(); delete noAction.frames[0].action;
  apiFailure(() => validateSnapshot(noAction, JOB));
  const badPosition = snapshot(); badPosition.frames[0].after.player.position = [NaN, 1];
  apiFailure(() => validateSnapshot(badPosition, JOB));
  const noAssessment = snapshot(); delete noAssessment.issues[0].assessment;
  apiFailure(() => validateSnapshot(noAssessment, JOB));
  const inventedCitation = snapshot(); inventedCitation.issues[0].evidenceSteps = [999];
  apiFailure(() => validateSnapshot(inventedCitation, JOB));
  for (const key of ["title", "playerImpact", "rationale"]) {
    const value = snapshot(); value.issues[0].assessment[key] = { text: "unexpected object" };
    apiFailure(() => validateSnapshot(value, JOB));
  }
  const badNote = snapshot(); badNote.issues[0].reviewNote = ["unexpected array"];
  apiFailure(() => validateSnapshot(badNote, JOB));
  const badEntity = snapshot(); badEntity.scene.entities = [{ id: "wall", kind: "wall", position: "not a vector", state: {} }];
  apiFailure(() => validateSnapshot(badEntity, JOB));
});

test("unknown cost remains unknown and cost caps do not masquerade as an account balance", () => {
  const value = snapshot(); value.summary.estimatedCostUsd = null; value.summary.costKnown = false;
  assert.equal(validateSnapshot(value, JOB).summary.costKnown, false);
  const contradictory = snapshot(); contradictory.summary.estimatedCostUsd = null;
  apiFailure(() => validateSnapshot(contradictory, JOB));
  const report = reportMarkdown(value);
  assert.match(report, /unknown/);
  assert.match(report, /reservations retained/);
  assert.doesNotMatch(report, /balance/i);
});

test("the final unassessed model outcome can be pending without a host issue", () => {
  const value = snapshot();
  value.issues = [];
  value.summary.issueCount = 0;
  value.summary.pendingCount = 1;
  value.notes = ["The final model action has recorded feedback without a subsequent model assessment."];
  assert.equal(validateSnapshot(value, JOB).summary.pendingCount, 1);
});

test("bounded progress rejects deeply nested or oversized evidence without including it in errors", () => {
  const huge = snapshot(); huge.notes = ["x".repeat(MAX_SNAPSHOT_BYTES + 1)];
  apiFailure(() => validateSnapshot(huge, JOB), 413);
  const deep = snapshot(); let current = {};
  for (let i = 0; i < 30; i++) current = { nested: current };
  deep.frames[0].before.objects = current;
  apiFailure(() => validateSnapshot(deep, JOB));
});

test("downloaded reports retain original host signal, negative source steps and tentative judgment", () => {
  const value = snapshot();
  Object.assign(value.issues[0].assessment, { status: "reviewed", verdict: "not_bug", severity: "none", playerImpact: "No adverse effect." });
  const report = reportMarkdown(value);
  assert.match(report, /software_anomaly/);
  assert.match(report, /entered_wall_labeled_tile/);
  assert.match(report, /Evidence steps: -1/);
  assert.match(report, /source steps.*-1/i);
  assert.match(report, /not_bug/);
  assert.match(report, /not checked/);
  assert.match(report, /minimal: not proven/);
  assert.match(report, /AI judgments remain tentative/);
});
