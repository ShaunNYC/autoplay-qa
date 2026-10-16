import { api, database, jobRow, json, requireOwner, TERMINAL } from "@/lib/server";
import { safeId } from "@/lib/api-core";
import type { RunSnapshot } from "@/lib/types";
export async function POST(request: Request, context: { params: Promise<{ id: string }> }) {
  return api(async () => {
    await requireOwner(request); const id = safeId((await context.params).id); const row = await jobRow(id);
    if (TERMINAL.has(row.status)) return json({ id, status: row.status });
    if (row.status === "queued") {
      const snapshot = JSON.parse(row.snapshot) as RunSnapshot;
      snapshot.summary.status = "cancelled"; snapshot.summary.phase = "complete";
      snapshot.notes = ["Cancelled before the runner claimed the job. No model requests were made."];
      const result = await database().prepare("UPDATE jobs SET cancel_requested = 1, status = 'cancelled', active_slot = NULL, summary = ?, snapshot = ? WHERE id = ? AND status = 'queued'").bind(JSON.stringify(snapshot.summary), JSON.stringify(snapshot), id).run();
      if (result.meta.changes) return json({ id, status: "cancelled" });
    }
    await database().prepare("UPDATE jobs SET cancel_requested = 1 WHERE id = ? AND status = 'running'").bind(id).run();
    return json({ id, cancelRequested: true });
  });
}
