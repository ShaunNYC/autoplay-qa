import { env } from "cloudflare:workers";
import { getChatGPTUser, chatGPTSignInPath } from "@/app/chatgpt-auth";
import { ApiError, MAX_SNAPSHOT_BYTES, TERMINAL, matchesBearer, sameOrigin } from "./api-core";
import type { RunSnapshot, RunnerStatus } from "./types";
import manifest from "./demo-manifest.json";
import { recordings } from "./demo-recordings";

type DashboardEnv = { DB: D1Database; ASSETS?: Fetcher; ADMIN_TOKEN?: string; RUNNER_TOKEN?: string; OWNER_EMAIL?: string; JOB_BUDGET_LIMIT_USD?: string };
export const runtime = env as unknown as DashboardEnv;
export function database(): D1Database { if (!runtime.DB) throw new ApiError(503, "The task service is temporarily unavailable."); return runtime.DB; }
export function json(data: unknown, status = 200): Response {
  return Response.json(data, { status, headers: { "Cache-Control": "no-store", "X-Content-Type-Options": "nosniff" } });
}
export async function api(action: () => Promise<Response>): Promise<Response> {
  try { return await action(); } catch (e) { return json({ error: e instanceof ApiError ? e.message : "The task service could not complete the request." }, e instanceof ApiError ? e.status : 503); }
}
export async function body(request: Request, limit = 10_000): Promise<unknown> {
  const reader = request.body?.getReader(); if (!reader) throw new ApiError(400, "A JSON body is required.");
  const chunks: Uint8Array[] = []; let size = 0;
  for (;;) { const { done, value } = await reader.read(); if (done) break; size += value.length; if (size > limit) { await reader.cancel(); throw new ApiError(413, "Request is too large."); } chunks.push(value); }
  const bytes = new Uint8Array(size); let cursor = 0; for (const chunk of chunks) { bytes.set(chunk, cursor); cursor += chunk.length; }
  try { return JSON.parse(new TextDecoder().decode(bytes)); } catch { throw new ApiError(400, "Invalid JSON."); }
}
export async function canControl(request: Request): Promise<boolean> {
  if (matchesBearer(request.headers.get("authorization"), runtime.ADMIN_TOKEN)) return true;
  const ownerEmail = runtime.OWNER_EMAIL?.trim().toLowerCase();
  if (!ownerEmail) return false;
  const user = await getChatGPTUser();
  if (user?.email.toLowerCase() === ownerEmail) return true;
  // The starter strips caller-supplied identity and injects this user only on loopback in dev.
  return Boolean(import.meta.env.DEV && user?.userId === "local_seedy" && user.email === "seedy@sites.test");
}
export async function requireOwner(request: Request): Promise<void> {
  if (!sameOrigin(request)) throw new ApiError(403, "Use this site's test controls.");
  if (!await canControl(request)) throw new ApiError(403, "Only the site owner can start or stop tests.");
}
export function requireRunner(request: Request): void {
  if (!matchesBearer(request.headers.get("authorization"), runtime.RUNNER_TOKEN)) throw new ApiError(403, "Runner authentication required.");
}
export const signInPath = chatGPTSignInPath("/");
export async function runnerStatus(): Promise<RunnerStatus> {
  const row = await database().prepare("SELECT * FROM runner_presence WHERE slot = 1").first<{ runner_id: string; last_seen: string; unity_ready: number }>();
  return { online: Boolean(row && Date.now() - Date.parse(row.last_seen) < 90_000), lastSeen: row?.last_seen ?? null, name: "Mac runner", unityReady: Boolean(row?.unity_ready) };
}
export interface JobRow { id: string; request_id: string; game: string; profile: string; seed: number; max_usd: number; created_at: string; updated_at: string; status: string; runner_id: string | null; lease_id: string | null; cancel_requested: number; revision: number; summary: string; snapshot: string }
export async function jobRow(id: string): Promise<JobRow> {
  const row = await database().prepare("SELECT * FROM jobs WHERE id = ?").bind(id).first<JobRow>(); if (!row) throw new ApiError(404, "Test not found."); return row;
}
export async function leaseRow(runnerId: string, jobId: string, leaseId: string): Promise<JobRow> {
  const row = await jobRow(jobId);
  if (row.runner_id !== runnerId || row.lease_id !== leaseId) throw new ApiError(409, "This runner does not own the test lease.");
  return row;
}
export async function readSnapshot(id: string, _request: Request): Promise<RunSnapshot> {
  if (manifest.runs.some(r => r.id === id)) {
    if (!recordings[id]) throw new ApiError(404, "Recorded evidence not found.");
    return recordings[id];
  }
  return JSON.parse((await jobRow(id)).snapshot) as RunSnapshot;
}
export async function expireDisconnected(): Promise<void> {
  const cutoff = new Date(Date.now() - 15 * 60_000).toISOString();
  const rows = await database().prepare("SELECT * FROM jobs WHERE status = 'running' AND updated_at < ?").bind(cutoff).all<JobRow>();
  for (const row of rows.results) {
    const snapshot = JSON.parse(row.snapshot) as RunSnapshot;
    snapshot.summary.status = "interrupted"; snapshot.summary.phase = "complete"; snapshot.summary.costKnown = false; snapshot.summary.estimatedCostUsd = null;
    snapshot.notes.push("The runner disconnected. This job will not be retried; its local budget reservation remains held.");
    await database().prepare("UPDATE jobs SET status = 'interrupted', active_slot = NULL, cancel_requested = 1, summary = ?, snapshot = ? WHERE id = ? AND status = 'running' AND updated_at < ?").bind(JSON.stringify(snapshot.summary), JSON.stringify(snapshot), row.id, cutoff).run();
  }
}
export { manifest, MAX_SNAPSHOT_BYTES, TERMINAL };
