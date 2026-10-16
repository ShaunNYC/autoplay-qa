import { api, body, database, json, leaseRow, MAX_SNAPSHOT_BYTES, requireRunner, TERMINAL } from "@/lib/server";
import { ApiError, object, safeId, validateRevision, validateSnapshot } from "@/lib/api-core";
export async function POST(request: Request) {
  return api(async () => {
    requireRunner(request); const data = object(await body(request, MAX_SNAPSHOT_BYTES + 20_000));
    const runnerId = safeId(data.runnerId), jobId = safeId(data.jobId), leaseId = safeId(data.leaseId);
    const revision = validateRevision(data.revision); const row = await leaseRow(runnerId, jobId, leaseId);
    if (revision <= row.revision) return json({ cancelRequested: Boolean(row.cancel_requested) || TERMINAL.has(row.status) });
    if (TERMINAL.has(row.status)) throw new ApiError(409, "This test has already ended.");
    const snapshot = validateSnapshot(data.snapshot, row);
    const terminal = TERMINAL.has(snapshot.summary.status);
    await database().prepare("UPDATE jobs SET status = ?, active_slot = ?, updated_at = ?, revision = ?, summary = ?, snapshot = ? WHERE id = ? AND runner_id = ? AND lease_id = ? AND revision < ? AND status = 'running'").bind(snapshot.summary.status, terminal ? null : 1, new Date().toISOString(), revision, JSON.stringify(snapshot.summary), JSON.stringify(snapshot), jobId, runnerId, leaseId, revision).run();
    const current = await leaseRow(runnerId, jobId, leaseId);
    return json({ cancelRequested: Boolean(current.cancel_requested) });
  });
}
