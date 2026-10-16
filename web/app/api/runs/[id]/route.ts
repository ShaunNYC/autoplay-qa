import { api, json, readSnapshot } from "@/lib/server";
import { safeId } from "@/lib/api-core";
export async function GET(request: Request, context: { params: Promise<{ id: string }> }) {
  return api(async () => json(await readSnapshot(safeId((await context.params).id), request)));
}
