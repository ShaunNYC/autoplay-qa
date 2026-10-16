import { api, body, json, leaseRow, requireRunner, TERMINAL } from "@/lib/server";
import { object, safeId } from "@/lib/api-core";
export async function POST(request: Request) {
  return api(async () => {
    requireRunner(request); const data = object(await body(request));
    const row = await leaseRow(safeId(data.runnerId), safeId(data.jobId), safeId(data.leaseId));
    return json({ cancelRequested: Boolean(row.cancel_requested) || TERMINAL.has(row.status), revision: row.revision, status: row.status });
  });
}
