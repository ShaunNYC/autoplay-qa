import type { NewRunRequest, RunSnapshot } from "./types";

export class ApiError extends Error {
  status: number;
  constructor(status: number, message: string) { super(message); this.status = status; }
}
export const MAX_SNAPSHOT_BYTES = 1_600_000;
export const TERMINAL = new Set(["completed", "failed", "cancelled", "interrupted"]);
export const LIMITS = { sweepControls: 400, modelControls: 24, explorationRequests: 28,
  triageRequests: 8, explorationSeconds: 300, triageSeconds: 300, maxTokens: 8192, timeout: 45 };
export const MODELS = { exploration: "nvidia/Nemotron-3_5-Lightning", triage: "nvidia/Nemotron-3-Ultra-550b-a55b" };

export function object(value: unknown): Record<string, unknown> {
  if (!value || typeof value !== "object" || Array.isArray(value)) throw new ApiError(400, "Expected a JSON object.");
  return value as Record<string, unknown>;
}
export function safeId(value: unknown): string {
  if (typeof value !== "string" || !/^[A-Za-z0-9_-]{8,100}$/.test(value)) throw new ApiError(400, "Invalid identifier.");
  return value;
}
export function validateNewRun(value: unknown, maxBudget = 10): NewRunRequest {
  const data = object(value);
  if (Object.keys(data).some(k => !["requestId", "game", "profile", "seed", "maxUsd"].includes(k))) throw new ApiError(400, "Unsupported test setting.");
  const requestId = safeId(data.requestId);
  if (typeof data.game !== "string" || !["grid", "unity"].includes(data.game)) throw new ApiError(400, "Choose Grid or Unity.");
  if (typeof data.profile !== "string" || !["clean", "wall", "floor", "key", "reset", "door"].includes(data.profile)) throw new ApiError(400, "Unknown test profile.");
  if (![0, 1, 5].includes(data.seed as number)) throw new ApiError(400, "Choose seed 0, 1, or 5.");
  if (typeof data.maxUsd !== "number" || !Number.isFinite(data.maxUsd) || data.maxUsd < 0.05 || data.maxUsd > Math.min(10, maxBudget)) throw new ApiError(400, "Budget must be between $0.05 and the configured limit.");
  return { requestId, game: data.game as NewRunRequest["game"], profile: data.profile as NewRunRequest["profile"], seed: data.seed as number, maxUsd: data.maxUsd };
}
export function matchesBearer(header: string | null, secret: string | undefined): boolean {
  if (!secret || secret.length < 32 || !header?.startsWith("Bearer ")) return false;
  const candidate = header.slice(7);
  if (candidate.length !== secret.length) return false;
  let difference = 0;
  for (let i = 0; i < secret.length; i++) difference |= candidate.charCodeAt(i) ^ secret.charCodeAt(i);
  return difference === 0;
}
export function sameOrigin(request: Request): boolean {
  const origin = request.headers.get("origin");
  return !origin || origin === new URL(request.url).origin;
}
export function validateRevision(value: unknown): number {
  if (typeof value !== "number" || !Number.isSafeInteger(value) || value < 1) throw new ApiError(400, "Invalid progress revision.");
  return value;
}

export function validateSnapshot(value: unknown, job: { id: string; game: string; seed: number; max_usd: number }): RunSnapshot {
  const data = object(value);
  if (new TextEncoder().encode(JSON.stringify(data)).length > MAX_SNAPSHOT_BYTES) throw new ApiError(413, "Progress is too large. Compact observations before upload.");
  const summary = object(data.summary);
  if (data.version !== "dashboard-v1" || summary.id !== job.id || summary.game !== job.game || summary.seed !== job.seed || summary.source !== "live") throw new ApiError(400, "Progress does not match the job.");
  if (typeof summary.status !== "string" || typeof summary.phase !== "string" || !["running", ...TERMINAL].includes(summary.status) || !["sweep", "explore", "triage", "complete"].includes(summary.phase)) throw new ApiError(400, "Invalid run state.");
  if (summary.budgetUsd !== job.max_usd) throw new ApiError(400, "Budget changed during the test.");
  for (const [key, max] of [["controls", 424], ["resets", 424], ["requests", 36], ["issueCount", 100], ["pendingCount", 100], ["warningCount", 10000]] as const) {
    if (!Number.isSafeInteger(summary[key]) || (summary[key] as number) < 0 || (summary[key] as number) > max) throw new ApiError(400, "Invalid run counts.");
  }
  if (typeof summary.durationSeconds !== "number" || !Number.isFinite(summary.durationSeconds) || summary.durationSeconds < 0) throw new ApiError(400, "Invalid timing.");
  if (summary.estimatedCostUsd !== null && (typeof summary.estimatedCostUsd !== "number" || !Number.isFinite(summary.estimatedCostUsd) || summary.estimatedCostUsd < 0)) throw new ApiError(400, "Invalid cost.");
  if (typeof summary.costKnown !== "boolean") throw new ApiError(400, "Missing cost status.");
  if (summary.costKnown && summary.estimatedCostUsd === null) throw new ApiError(400, "Known usage requires a cost estimate.");
  if (typeof summary.name !== "string" || summary.name.length > 300 || typeof summary.createdAt !== "string" || !Number.isFinite(Date.parse(summary.createdAt))) throw new ApiError(400, "Invalid run description.");
  for (const [key, max] of [["frames", 424], ["issues", 100], ["timeline", 2000], ["notes", 100]] as const) {
    if (!Array.isArray(data[key]) || (data[key] as unknown[]).length > max) throw new ApiError(400, "Progress exceeds the display limit.");
  }
  const limits = object(data.limits);
  const models = object(data.models);
  if (Object.entries(LIMITS).some(([k, v]) => limits[k] !== v) || Object.entries(MODELS).some(([k, v]) => models[k] !== v)) throw new ApiError(400, "Unexpected engine configuration.");
  const frames = data.frames as unknown[], issues = data.issues as unknown[];
  if (summary.controls !== frames.length || (summary.resets as number) > frames.length || summary.issueCount !== issues.length || (summary.pendingCount as number) > issues.length + 1) throw new ApiError(400, "Run counts disagree with recorded evidence.");
  function string(value: unknown, max = 16_000): void { if (typeof value !== "string" || value.length > max) throw new ApiError(400, "Invalid display text."); }
  function numbers(value: unknown): void { if (!Array.isArray(value) || value.length > 1000 || value.some(n => !Number.isSafeInteger(n))) throw new ApiError(400, "Invalid evidence steps."); }
  function strings(value: unknown, max = 1000): void { if (!Array.isArray(value) || value.length > max) throw new ApiError(400, "Invalid text list."); value.forEach(v => string(v)); }
  function observation(value: unknown): void {
    const obs = object(value);
    function vector(value: unknown): void { if (!Array.isArray(value) || value.length < 2 || value.length > 3 || value.some(n => typeof n !== "number" || !Number.isFinite(n))) throw new ApiError(400, "Invalid scene position."); }
    if (obs.player !== undefined) { const player = object(obs.player); for (const key of ["position", "velocity"]) if (player[key] !== undefined) vector(player[key]); if (player.inventory !== undefined && !Array.isArray(player.inventory)) throw new ApiError(400, "Invalid inventory."); if (player.grounded !== undefined && typeof player.grounded !== "boolean") throw new ApiError(400, "Invalid grounded state."); }
    if (obs.map !== undefined) strings(obs.map, 1000);
    if (obs.entities !== undefined && !Array.isArray(obs.entities)) throw new ApiError(400, "Invalid scene objects.");
    for (const value of (obs.entities ?? []) as unknown[]) { const entity = object(value); string(entity.id, 200); string(entity.kind, 200); vector(entity.position); if (entity.size !== undefined) vector(entity.size); object(entity.state); }
    for (const key of ["game", "status"]) if (obs[key] !== undefined) string(obs[key], 200);
    if (obs.events !== undefined) strings(obs.events);
  }
  if (data.scene !== null) observation(data.scene);
  const frameIds = new Set(); const frameSteps = new Set<number>();
  for (const value of frames) {
    const frame = object(value); string(frame.id, 100); if (frameIds.has(frame.id)) throw new ApiError(400, "Duplicate control identifier."); frameIds.add(frame.id);
    if (!Number.isSafeInteger(frame.step) || !Number.isSafeInteger(frame.phaseStep) || !["sweep", "llm"].includes(frame.phase as string)) throw new ApiError(400, "Invalid control step.");
    frameSteps.add(frame.step as number);
    string(frame.action, 1000); string(frame.reason); observation(frame.before); observation(frame.after); strings(frame.feedback);
    if (typeof frame.elapsedSeconds !== "number" || !Number.isFinite(frame.elapsedSeconds) || frame.elapsedSeconds < 0) throw new ApiError(400, "Invalid control timing.");
  }
  for (const value of issues) {
    const issue = object(value); string(issue.id, 100); string(issue.title); string(issue.source); string(issue.anomalyType); numbers(issue.evidenceSteps);
    if ((issue.evidenceSteps as number[]).some(step => !frameSteps.has(step))) throw new ApiError(400, "Issue refers to an unrecorded control.");
    if (!Number.isSafeInteger(issue.number) || typeof issue.needsHumanReview !== "boolean") throw new ApiError(400, "Invalid issue details.");
    const assessment = object(issue.assessment); if (!["reviewed", "pending", "failed"].includes(assessment.status as string)) throw new ApiError(400, "Invalid AI assessment.");
    for (const key of ["title", "playerImpact", "rationale"]) if (assessment[key] !== undefined) string(assessment[key]);
    if (issue.reviewNote !== undefined) string(issue.reviewNote);
    if (assessment.verdict !== undefined && !["bug", "not_bug", "inconclusive"].includes(assessment.verdict as string)) throw new ApiError(400, "Invalid AI verdict.");
    if (assessment.severity !== undefined && !["critical", "high", "medium", "low", "none"].includes(assessment.severity as string)) throw new ApiError(400, "Invalid severity.");
    numbers(assessment.evidenceSteps); strings(assessment.limitations); strings(assessment.warnings);
    if ((assessment.evidenceSteps as number[]).some(step => !frameSteps.has(step))) throw new ApiError(400, "AI assessment refers to an unrecorded control.");
    const reproduction = object(issue.reproduction); strings(reproduction.actions); numbers(reproduction.sourceSteps);
    if (![true, false, null].includes(reproduction.verified as boolean | null) || typeof reproduction.minimal !== "boolean") throw new ApiError(400, "Invalid replay receipt.");
  }
  for (const value of data.timeline as unknown[]) { const event = object(value); string(event.id, 100); string(event.label); if (!["phase", "action", "request", "warning", "issue", "complete"].includes(event.kind as string)) throw new ApiError(400, "Invalid timeline event."); }
  strings(data.notes, 100);
  // Runner uploads are already curated; reject accidental private fields before sharing.
  const privateKeys = new Set(["apikey", "authorization", "token", "accesstoken", "secret", "password", "prompt", "rawprompt", "systemprompt", "messages", "responseschema", "thinking", "reasoningcontent", "rawresponse", "rawtext", "completion", "developerrules", "scenario", "plantedbug", "unityplayerpath", "unityplayer", "unityproject", "sourcepath", "sourcerecording", "env", "environmentvariables", "evaluation", "injecteddefect", "groundtruth", "defect", "defectinjection", "rules", "rule", "expectedverdict", "expectedverdicts", "caseid", "kindofcase", "strategy", "victorystrategy", "solution", "route", "suppliedroute", "forcedresets", "passabilityhints", "endpoint", "baseurl"]);
  let nodes = 0;
  function inspect(item: unknown, depth = 0): void {
    if (++nodes > 180_000 || depth > 24) throw new ApiError(400, "Progress is too deeply nested.");
    if (typeof item === "string" && (/(?:\/(?:Users|home|private|tmp|var|Volumes|Applications)\/|[A-Z]:\\)/.test(item) || /(?:https?:\/\/)?api\.tokenfactory\.nebius\.com/.test(item) || /\b(?:sk-[A-Za-z0-9_-]{12,}|Bearer\s+[A-Za-z0-9._-]{8,})/i.test(item))) throw new ApiError(400, "Private connection details cannot be shared.");
    if (Array.isArray(item)) item.forEach(v => inspect(v, depth + 1));
    else if (item && typeof item === "object") for (const [k, v] of Object.entries(item)) { const normalized = k.toLowerCase().replace(/[^a-z0-9]/g, ""); if (privateKeys.has(normalized) || /^(private|secret|credential|developer)/.test(normalized)) throw new ApiError(400, "Private fields cannot be shared."); inspect(v, depth + 1); }
  }
  inspect(data);
  if (new TextEncoder().encode(JSON.stringify(data)).length > MAX_SNAPSHOT_BYTES) throw new ApiError(413, "Progress is too large. Compact observations before upload.");
  return data as unknown as RunSnapshot;
}

export function queuedSnapshot(id: string, run: NewRunRequest, now: string): RunSnapshot {
  return { version: "dashboard-v1", summary: { id, name: `${run.game === "unity" ? "Unity physics" : "Grid game"} · ${run.profile} · seed ${run.seed}`, game: run.game, source: "live", status: "queued", phase: "queued", seed: run.seed, createdAt: now, durationSeconds: 0, controls: 0, resets: 0, requests: 0, estimatedCostUsd: 0, costKnown: true, issueCount: 0, budgetUsd: run.maxUsd, warningCount: 0, pendingCount: 0 }, scene: null, frames: [], issues: [], timeline: [], notes: ["Waiting for the Mac runner."], limits: { ...LIMITS }, models: { ...MODELS } };
}

export function reportMarkdown(snapshot: RunSnapshot): string {
  const { summary: s } = snapshot;
  const lines = [`# ${s.name}`, "", `${s.source === "recorded" ? "Recorded session" : "Live test"} · ${s.status} · seed ${s.seed}`, "", `${s.controls} controls · ${s.resets} resets · ${s.requests} model requests · ${s.durationSeconds.toFixed(1)} seconds`, "", `Estimated token cost: ${s.estimatedCostUsd === null ? "unknown" : "$" + s.estimatedCostUsd.toFixed(6)}${s.costKnown ? "" : " (some usage unknown; reservations retained)"}`, "", "Replay receipts prove observed execution. AI judgments remain tentative."];
  for (const issue of snapshot.issues) {
    lines.push("", `## ${issue.number}. ${issue.title}`, "", `Original signal: ${issue.source} / ${issue.anomalyType}`, `Evidence steps: ${issue.evidenceSteps.join(", ")}`, "", `Expected: ${JSON.stringify(issue.expected)}`, `Observed: ${JSON.stringify(issue.actual)}`, "", `Ultra: ${issue.assessment.verdict ?? issue.assessment.status} · ${issue.assessment.severity ?? "unrated"}`, issue.assessment.playerImpact ?? "", issue.assessment.rationale ?? "", ...issue.assessment.limitations.map(l => `Limitation: ${l}`), "", `Replay verified: ${issue.reproduction.verified === null ? "not checked" : issue.reproduction.verified ? "yes" : "no"}; minimal: ${issue.reproduction.minimal ? "yes" : "not proven"}`, `Original source steps: ${issue.reproduction.sourceSteps.join(", ")}`, ...issue.reproduction.actions.map((action, i) => `${i + 1}. ${action}`));
    if (issue.needsHumanReview) lines.push("", `Human review: ${issue.reviewNote ?? "Required."}`);
  }
  lines.push("", ...snapshot.notes.map(n => `Note: ${n}`));
  return lines.filter(v => v !== undefined).join("\n");
}
