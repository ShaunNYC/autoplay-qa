export type GameKind = "grid" | "unity";
export type RunStatus = "queued" | "running" | "completed" | "failed" | "cancelled" | "interrupted";
export type RunPhase = "queued" | "sweep" | "explore" | "triage" | "complete";
export type TestProfile = "clean" | "wall" | "floor" | "key" | "reset" | "door";
export type Verdict = "bug" | "not_bug" | "inconclusive";
export type Severity = "critical" | "high" | "medium" | "low" | "none";
export interface PlayerState {
  position?: number[]; inventory?: unknown[]; velocity?: number[]; grounded?: boolean;
}
export interface Entity {
  id: string; kind: string; position: number[]; size?: number[]; state: Record<string, unknown>;
}
export interface Observation {
  game?: string; tick?: number; status?: string; player?: PlayerState;
  map?: string[]; legend?: Record<string, string>; entities?: Entity[];
  objects?: Record<string, unknown>; events?: string[];
}
export interface Frame {
  id: string; step: number; phaseStep: number; phase: "sweep" | "llm";
  action: string; reason: string; before: Observation; after: Observation;
  feedback: string[]; elapsedSeconds: number;
}
export interface Assessment {
  status: "reviewed" | "pending" | "failed";
  verdict?: Verdict; severity?: Severity; title?: string;
  playerImpact?: string; rationale?: string; evidenceSteps: number[];
  limitations: string[]; warnings: string[];
}
export interface Issue {
  id: string; number: number; title: string; source: string; anomalyType: string;
  evidenceSteps: number[]; expected: unknown; actual: unknown;
  reproduction: { actions: string[]; sourceSteps: number[]; verified: boolean | null; minimal: boolean };
  assessment: Assessment;
  needsHumanReview: boolean; reviewNote?: string;
}
export interface RunSummary {
  id: string; name: string; game: GameKind; source: "recorded" | "live";
  status: RunStatus; phase: RunPhase; seed: number; createdAt: string;
  durationSeconds: number; controls: number; resets: number; requests: number;
  estimatedCostUsd: number | null; costKnown: boolean; issueCount: number;
  budgetUsd: number | null; warningCount: number; pendingCount: number;
}
export interface TimelineEvent {
  id: string; kind: "phase" | "action" | "request" | "warning" | "issue" | "complete";
  phase: RunPhase; label: string; detail?: string; elapsedSeconds: number;
}
export interface RunSnapshot {
  version: "dashboard-v1"; summary: RunSummary; scene: Observation | null;
  frames: Frame[]; issues: Issue[]; timeline: TimelineEvent[]; notes: string[];
  limits: { sweepControls: number; modelControls: number; explorationRequests: number; triageRequests: number; explorationSeconds: number; triageSeconds: number; maxTokens: number; timeout: number };
  models: { exploration: string; triage: string };
}
export interface RunnerStatus { online: boolean; lastSeen: string | null; name: string; unityReady: boolean }
export interface RunsResponse { runs: RunSummary[]; runner: RunnerStatus; canControl: boolean; signInPath: string }
export interface NewRunRequest { requestId: string; game: GameKind; profile: TestProfile; seed: number; maxUsd: number }
export interface Job extends NewRunRequest { id: string; createdAt: string; leaseId: string; cancelRequested: boolean }
