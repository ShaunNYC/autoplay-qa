import { api, body, database, expireDisconnected, json, requireRunner, runnerStatus } from "@/lib/server";
import type { JobRow } from "@/lib/server";
import { ApiError, object, safeId } from "@/lib/api-core";
import type { RunSnapshot } from "@/lib/types";
export async function POST(request: Request) {
  return api(async () => {
    requireRunner(request); const runnerId = safeId(object(await body(request)).runnerId);
    const presence = await database().prepare("SELECT runner_id FROM runner_presence WHERE slot = 1").first<{ runner_id: string }>();
    if (presence?.runner_id !== runnerId || !(await runnerStatus()).online) throw new ApiError(409, "Send a heartbeat before claiming a job.");
    await expireDisconnected();
    const leaseId = crypto.randomUUID(); const now = new Date().toISOString();
    const row = await database().prepare("UPDATE jobs SET status = 'running', runner_id = ?, lease_id = ?, updated_at = ? WHERE id = (SELECT id FROM jobs WHERE status = 'queued' AND cancel_requested = 0 ORDER BY created_at LIMIT 1) AND status = 'queued' RETURNING *").bind(runnerId, leaseId, now).first<JobRow>();
    if (!row) return json({ job: null });
    const snapshot = JSON.parse(row.snapshot) as RunSnapshot;
    snapshot.summary.status = "running"; snapshot.summary.phase = "sweep"; snapshot.notes = ["The Mac runner claimed this test."];
    await database().prepare("UPDATE jobs SET summary = ?, snapshot = ? WHERE id = ? AND lease_id = ? AND revision = 0").bind(JSON.stringify(snapshot.summary), JSON.stringify(snapshot), row.id, leaseId).run();
    return json({ job: { id: row.id, requestId: row.request_id, game: row.game, profile: row.profile, seed: row.seed, maxUsd: row.max_usd, createdAt: row.created_at, leaseId, cancelRequested: Boolean(row.cancel_requested) } });
  });
}
