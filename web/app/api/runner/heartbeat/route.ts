import { api, body, database, json, requireRunner, runnerStatus } from "@/lib/server";
import { ApiError, object, safeId } from "@/lib/api-core";
export async function POST(request: Request) {
  return api(async () => {
    requireRunner(request); const data = object(await body(request)); const runnerId = safeId(data.runnerId);
    if (typeof data.unityReady !== "boolean") throw new ApiError(400, "Unity availability is required.");
    await database().prepare("INSERT INTO runner_presence (slot, runner_id, last_seen, unity_ready) VALUES (1, ?, ?, ?) ON CONFLICT(slot) DO UPDATE SET runner_id = excluded.runner_id, last_seen = excluded.last_seen, unity_ready = excluded.unity_ready").bind(runnerId, new Date().toISOString(), data.unityReady ? 1 : 0).run();
    return json(await runnerStatus());
  });
}
