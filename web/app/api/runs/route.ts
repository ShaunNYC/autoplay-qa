import { api, body, canControl, database, expireDisconnected, json, manifest, requireOwner, runnerStatus, runtime, signInPath } from "@/lib/server";
import { ApiError, queuedSnapshot, validateNewRun } from "@/lib/api-core";

export async function GET(request: Request) {
  return api(async () => {
    await expireDisconnected();
    const rows = await database().prepare("SELECT summary FROM jobs ORDER BY created_at DESC LIMIT 50").all<{ summary: string }>();
    return json({ runs: [...rows.results.map(r => JSON.parse(r.summary)), ...manifest.runs], runner: await runnerStatus(), canControl: await canControl(request), signInPath });
  });
}
export async function POST(request: Request) {
  return api(async () => {
    await requireOwner(request);
    const run = validateNewRun(await body(request), Number(runtime.JOB_BUDGET_LIMIT_USD ?? 10));
    const existing = await database().prepare("SELECT id, game, profile, seed, max_usd FROM jobs WHERE request_id = ?").bind(run.requestId).first<{ id: string; game: string; profile: string; seed: number; max_usd: number }>();
    if (existing) {
      if (existing.game !== run.game || existing.profile !== run.profile || existing.seed !== run.seed || existing.max_usd !== run.maxUsd) throw new ApiError(409, "This request identifier belongs to different test settings.");
      return json({ id: existing.id });
    }
    await expireDisconnected();
    const runner = await runnerStatus();
    if (!runner.online) throw new ApiError(409, "Start the Mac runner before creating a test.");
    if (run.game === "unity" && !runner.unityReady) throw new ApiError(409, "The Mac runner has no Unity player configured.");
    const id = crypto.randomUUID(); const now = new Date().toISOString();
    const snapshot = queuedSnapshot(id, run, now);
    try {
      await database().prepare("INSERT INTO jobs (id, request_id, game, profile, seed, max_usd, created_at, updated_at, status, active_slot, cancel_requested, revision, summary, snapshot) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'queued', 1, 0, 0, ?, ?)").bind(id, run.requestId, run.game, run.profile, run.seed, run.maxUsd, now, now, JSON.stringify(snapshot.summary), JSON.stringify(snapshot)).run();
    } catch {
      const retry = await database().prepare("SELECT id, game, profile, seed, max_usd FROM jobs WHERE request_id = ?").bind(run.requestId).first<{ id: string; game: string; profile: string; seed: number; max_usd: number }>();
      if (retry && retry.game === run.game && retry.profile === run.profile && retry.seed === run.seed && retry.max_usd === run.maxUsd) return json({ id: retry.id });
      throw new ApiError(409, "A test is already queued or running. Wait or stop it first.");
    }
    return json({ id }, 201);
  });
}
