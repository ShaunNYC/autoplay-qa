import { integer, real, sqliteTable, text, uniqueIndex } from "drizzle-orm/sqlite-core";

export const jobs = sqliteTable("jobs", {
  id: text("id").primaryKey(),
  requestId: text("request_id").notNull().unique(),
  game: text("game").notNull(),
  profile: text("profile").notNull(),
  seed: integer("seed").notNull(),
  maxUsd: real("max_usd").notNull(),
  createdAt: text("created_at").notNull(),
  updatedAt: text("updated_at").notNull(),
  status: text("status").notNull(),
  activeSlot: integer("active_slot"),
  runnerId: text("runner_id"),
  leaseId: text("lease_id"),
  cancelRequested: integer("cancel_requested").notNull().default(0),
  revision: integer("revision").notNull().default(0),
  summary: text("summary").notNull(),
  snapshot: text("snapshot").notNull(),
}, (table) => [uniqueIndex("one_active_job").on(table.activeSlot)]);

export const runnerPresence = sqliteTable("runner_presence", {
  slot: integer("slot").primaryKey(),
  runnerId: text("runner_id").notNull(),
  lastSeen: text("last_seen").notNull(),
  unityReady: integer("unity_ready").notNull().default(0),
});
