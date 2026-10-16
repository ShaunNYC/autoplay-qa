import { api, readSnapshot } from "@/lib/server";
import { ApiError, reportMarkdown, safeId } from "@/lib/api-core";
export async function GET(request: Request, context: { params: Promise<{ id: string }> }) {
  return api(async () => {
    const id = safeId((await context.params).id); const format = new URL(request.url).searchParams.get("format") ?? "json";
    if (!["json", "markdown"].includes(format)) throw new ApiError(400, "Choose JSON or Markdown.");
    const snapshot = await readSnapshot(id, request);
    return new Response(format === "json" ? JSON.stringify(snapshot, null, 2) : reportMarkdown(snapshot), { headers: {
      "Content-Type": format === "json" ? "application/json; charset=utf-8" : "text/markdown; charset=utf-8",
      "Content-Disposition": `attachment; filename="${id}.${format === "json" ? "json" : "md"}"`,
      "Cache-Control": "no-store", "X-Content-Type-Options": "nosniff",
    } });
  });
}
