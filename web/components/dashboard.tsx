"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { Activity, ArrowRight, ArrowUpRight, Box, Bug, Check, CheckCircle2, ChevronLeft, ChevronRight, Circle, Copy, Download, FileJson, FileText, Flag, Gamepad2, Grid2X2, Info, ListFilter, Loader2, LockKeyhole, Pause, Play, Plus, Radio, RefreshCw, Scan, Search, ShieldCheck, SkipBack, SkipForward, Sparkles, Square, Target, TriangleAlert, X, Zap } from "lucide-react";
import type { Frame, GameKind, Issue, NewRunRequest, Observation, RunPhase, RunSnapshot, RunsResponse, RunSummary, Severity, TestProfile } from "@/lib/types";
import SceneMap from "./scene-map";
import { ProjectIntro, ProjectMethod, CommunityDirections, AboutAuthor } from "./project-intro";

const ACTIVE = new Set(["queued", "running"]);
const PHASES: { value: RunPhase; title: string; subtitle: string; icon: typeof Scan }[] = [
  { value: "sweep", title: "Map the space", subtitle: "Spatial sweep", icon: Scan },
  { value: "explore", title: "Explore & investigate", subtitle: "Nemotron Lightning", icon: Zap },
  { value: "triage", title: "Review the evidence", subtitle: "Nemotron Ultra", icon: Sparkles },
];
const PROFILES: { value: TestProfile; title: string; detail: string }[] = [
  { value: "clean", title: "Baseline", detail: "Unmodified control session" },
  { value: "wall", title: "Collision regression", detail: "Wall contact test environment" },
  { value: "floor", title: "Floor support", detail: "Grounding test environment" },
  { value: "key", title: "Key interaction", detail: "Item behavior test environment" },
  { value: "reset", title: "Reset consistency", detail: "Session reset test environment" },
  { value: "door", title: "Door interaction", detail: "Door behavior test environment" },
];

function duration(seconds: number) {
  if (!Number.isFinite(seconds)) return "—";
  const value = Math.max(0, Math.round(seconds));
  return value < 60 ? `${value}s` : `${Math.floor(value / 60)}m ${String(value % 60).padStart(2, "0")}s`;
}
function money(value: number | null) { return value === null ? "Unknown" : `$${value.toFixed(value < .01 && value > 0 ? 4 : 2)}`; }
function date(value: string, compact = false) {
  const parsed = new Date(value);
  return Number.isNaN(parsed.getTime()) ? "Date unavailable" : parsed.toLocaleString("en-US", compact ? { month: "short", day: "numeric" } : { month: "short", day: "numeric", year: "numeric", hour: "numeric", minute: "2-digit" });
}
function gameName(game: GameKind) { return game === "unity" ? "Unity PhysicsKeyDoor" : "Grid KeyDoor"; }
function formatPosition(position: number[] | undefined) { return position?.length ? position.map((n) => Number.isInteger(n) ? n : n.toFixed(2)).join(", ") : "Unavailable"; }
function textValue(value: unknown): string {
  if (value === null || value === undefined) return "Not recorded";
  if (typeof value === "string") return value;
  return JSON.stringify(value, null, 2);
}
function verdict(issue: Issue) {
  if (issue.assessment.status === "pending") return "Awaiting review";
  if (issue.assessment.status === "failed") return "Review incomplete";
  if (issue.assessment.verdict === "bug") return "AI: likely bug";
  if (issue.assessment.verdict === "not_bug") return "AI: not a bug";
  return "AI: more evidence needed";
}
function verdictTone(issue: Issue) {
  if (issue.assessment.status !== "reviewed") return "pending";
  if (issue.needsHumanReview || issue.assessment.verdict === "inconclusive") return "review";
  return issue.assessment.verdict === "bug" ? "bug" : "clear";
}
function safeError(value: unknown) { return value instanceof Error ? value.message : "Something went wrong. Please try again."; }
async function fetchJson<T>(url: string, init?: RequestInit): Promise<T> {
  const response = await fetch(url, { cache: "no-store", ...init });
  const body: unknown = await response.json().catch(() => null);
  if (!response.ok) {
    const error = (body && typeof body === "object" && "error" in body ? body.error : null) as string | { message?: string } | null;
    throw new Error(typeof error === "string" ? error : error?.message ?? `Request could not be completed (${response.status}).`);
  }
  if (!body) throw new Error("The server returned an empty response.");
  return body as T;
}

function SeverityBadge({ severity }: { severity?: Severity }) {
  if (!severity || severity === "none") return null;
  return <span className={`severity severity-${severity}`}><i />{severity}</span>;
}
function Status({ run }: { run: RunSummary }) {
  const live = ACTIVE.has(run.status);
  return <span className={`run-status ${live ? "status-running" : run.status === "completed" ? "status-complete" : "status-other"}`}><i />{run.status === "running" ? "In progress" : run.status.charAt(0).toUpperCase() + run.status.slice(1)}</span>;
}

export default function Dashboard() {
  const [catalog, setCatalog] = useState<RunsResponse | null>(null);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [run, setRun] = useState<RunSnapshot | null>(null);
  const [frameIndex, setFrameIndex] = useState(0);
  const [selectedIssueId, setSelectedIssueId] = useState<string | null>(null);
  const [playing, setPlaying] = useState(false);
  const [playbackEnd, setPlaybackEnd] = useState<number | null>(null);
  const [speed, setSpeed] = useState(1);
  const [following, setFollowing] = useState(false);
  const followingRef = useRef(false);
  const [query, setQuery] = useState("");
  const [gameFilter, setGameFilter] = useState<"all" | GameKind>("all");
  const [activeTab, setActiveTab] = useState<"findings" | "activity">("findings");
  const [showNewTest, setShowNewTest] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [detailError, setDetailError] = useState<string | null>(null);
  const [refresh, setRefresh] = useState(0);
  const [stopping, setStopping] = useState(false);
  const firstLoad = useRef(true);
  const previousRun = useRef<string | null>(null);
  const searchRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key !== "/" || event.metaKey || event.ctrlKey || event.altKey) return;
      const target = event.target as HTMLElement | null;
      if (target?.isContentEditable || target?.tagName === "INPUT" || target?.tagName === "SELECT" || target?.tagName === "TEXTAREA") return;
      event.preventDefault();
      searchRef.current?.focus();
    };
    document.addEventListener("keydown", onKeyDown);
    return () => document.removeEventListener("keydown", onKeyDown);
  }, []);

  useEffect(() => {
    let closed = false;
    const update = async () => {
      try {
        const data = await fetchJson<RunsResponse>("/api/runs");
        if (closed) return;
        setCatalog(data);
        setError(null);
        setSelectedId((current) => current ?? data.runs.find((item) => item.game === "unity" && item.issueCount > 0)?.id ?? data.runs[0]?.id ?? null);
      } catch (cause) { if (!closed) setError(safeError(cause)); }
    };
    void update();
    const timer = setInterval(update, 5000);
    return () => { closed = true; clearInterval(timer); };
  }, [refresh]);

  useEffect(() => {
    if (!selectedId) return;
    let closed = false;
    let timer: ReturnType<typeof setTimeout> | undefined;
    firstLoad.current = previousRun.current !== selectedId;
    previousRun.current = selectedId;
    const update = async () => {
      try {
        const data = await fetchJson<RunSnapshot>(`/api/runs/${encodeURIComponent(selectedId)}`);
        if (closed) return;
        setRun(data);
        setDetailError(null);
        if (firstLoad.current) {
          firstLoad.current = false;
          const issue = data.issues.find((item) => item.assessment.verdict === "bug") ?? data.issues[0];
          setSelectedIssueId(issue?.id ?? null);
          const live = ACTIVE.has(data.summary.status);
          followingRef.current = live;
          setFollowing(live);
          const evidence = data.frames.findIndex((item) => issue?.evidenceSteps.includes(item.step));
          setFrameIndex(live ? Math.max(0, data.frames.length - 1) : evidence >= 0 ? evidence : 0);
        } else {
          setFrameIndex((current) => followingRef.current ? Math.max(0, data.frames.length - 1) : Math.min(current, Math.max(0, data.frames.length - 1)));
          setSelectedIssueId((current) => current ?? data.issues[0]?.id ?? null);
        }
        if (ACTIVE.has(data.summary.status)) timer = setTimeout(update, 2000);
      } catch (cause) {
        if (!closed) { setDetailError(safeError(cause)); timer = setTimeout(update, 10000); }
      }
    };
    void update();
    return () => { closed = true; if (timer) clearTimeout(timer); };
  }, [selectedId, refresh]);

  useEffect(() => {
    if (!playing || !run?.frames.length) return;
    const timer = setInterval(() => {
      setFrameIndex((current) => {
        if (current >= (playbackEnd ?? run.frames.length - 1)) { setPlaying(false); return current; }
        return current + 1;
      });
    }, 600 / speed);
    return () => clearInterval(timer);
  }, [playing, speed, run?.frames.length, playbackEnd]);

  const seek = useCallback((index: number) => {
    followingRef.current = false;
    setFollowing(false);
    setPlaying(false);
    setPlaybackEnd(null);
    setFrameIndex(Math.max(0, Math.min(index, (run?.frames.length ?? 1) - 1)));
  }, [run?.frames.length]);

  const openRun = (id: string) => {
    if (id === selectedId) return;
    setRun(null);
    setPlaying(false);
    setPlaybackEnd(null);
    setDetailError(null);
    setActiveTab("findings");
    setSelectedId(id);
  };

  const selectIssue = (issue: Issue) => {
    setSelectedIssueId(issue.id);
    setActiveTab("findings");
    const index = run?.frames.findIndex((item) => issue.evidenceSteps.includes(item.step)) ?? -1;
    if (index >= 0) seek(index);
  };
  const jumpToStep = (step: number) => {
    const index = run?.frames.findIndex((item) => item.step === step) ?? -1;
    if (index >= 0) seek(index);
  };
  const followLive = () => {
    followingRef.current = true;
    setFollowing(true);
    setPlaying(false);
    setFrameIndex(Math.max(0, (run?.frames.length ?? 1) - 1));
  };
  const stop = async () => {
    if (!run || !catalog?.canControl || stopping) return;
    setStopping(true);
    try {
      await fetchJson(`/api/runs/${encodeURIComponent(run.summary.id)}/stop`, { method: "POST" });
      setRefresh((n) => n + 1);
    } catch (cause) { setDetailError(safeError(cause)); }
    finally { setStopping(false); }
  };

  const isLive = !!run && ACTIVE.has(run.summary.status);
  const selectedIssue = run?.issues.find((item) => item.id === selectedIssueId);
  const frame = run?.frames[frameIndex];
  const filteredRuns = (catalog?.runs ?? []).filter((item) => (gameFilter === "all" || item.game === gameFilter) && `${item.name} ${gameName(item.game)} ${item.id}`.toLowerCase().includes(query.toLowerCase()));
  const bugCount = run?.issues.filter((item) => item.assessment.verdict === "bug").length ?? 0;
  const softwareCount = run?.issues.filter((item) => item.source === "software_anomaly").length ?? 0;
  const authoredCount = run?.issues.filter((item) => item.source === "authored_normal_candidate").length ?? 0;
  const newTestAvailable = catalog?.canControl && catalog.runner.online;

  return (
    <main className="dashboard-shell" id="main-content">
      <a className="skip-link" href="#evidence-workspace">Skip to evidence workspace</a>
      <ProjectIntro />
      <ProjectMethod />
      <section className="evidence-workspace" id="evidence-workspace" aria-labelledby="workspace-title">
        <div className="workspace-intro"><div><h2 id="workspace-title">Inspect the play.<br /><span>Question the verdict.</span></h2></div><p>Explore recorded Grid and Unity sessions. Follow the controls, inspect the feedback and read the model’s cited review.<span>Playback uses saved observations and makes no model requests.</span></p></div>
      <div className="app-body">
        <aside className="sidebar" aria-label="Test sessions">
          <div className="workspace-access"><span className="access-label"><LockKeyhole size={13} />{catalog?.canControl ? "Owner workspace" : "Public · read only"}</span><button className="button button-primary" disabled={!newTestAvailable} onClick={() => setShowNewTest(true)} title={!catalog?.canControl ? "Only the workspace owner can start real tests." : !catalog.runner.online ? "Connect the Mac runner to start a test." : "Start a bounded game test"}><Plus size={16} />New test</button>{!catalog?.canControl && catalog?.signInPath && <a href={catalog.signInPath} className="sign-in">Owner sign in<ArrowUpRight size={13} /></a>}</div>
          <div className="sidebar-heading"><span>TEST SESSIONS</span><button className="icon-button" aria-label="Refresh test sessions" onClick={() => setRefresh((n) => n + 1)}><RefreshCw size={14} /></button></div>
          <label className="search-box"><Search size={14} /><input ref={searchRef} value={query} onChange={(event) => setQuery(event.target.value)} placeholder="Find a session…" aria-label="Find a session" /><span>/</span></label>
          <div className="game-filters" aria-label="Filter sessions by game">{(["all", "unity", "grid"] as const).map((kind) => <button key={kind} className={gameFilter === kind ? "active" : ""} onClick={() => setGameFilter(kind)}>{kind === "all" ? "All sessions" : kind === "unity" ? "Unity" : "Grid"}</button>)}</div>
          <div className="run-list">
            {!catalog && !error && <div className="sidebar-empty"><Loader2 size={18} className="spin" /><span>Loading saved sessions…</span></div>}
            {catalog && !filteredRuns.length && <div className="sidebar-empty"><Search size={19} /><span>{query ? "No matching sessions" : "No sessions yet"}</span></div>}
            {filteredRuns.map((item) => <button className={`run-item ${selectedId === item.id ? "selected" : ""}`} key={item.id} onClick={() => openRun(item.id)}>
              <div className="run-item-top"><span className="game-icon">{item.game === "unity" ? <Box size={15} /> : <Grid2X2 size={15} />}</span><span>{item.game === "unity" ? "UNITY" : "GRID"}</span><span className="run-item-date">{date(item.createdAt, true)}</span></div>
              <strong>{item.name}</strong>
              <div className="run-item-meta"><span className={`small-status ${ACTIVE.has(item.status) ? "live" : ""}`}><i />{ACTIVE.has(item.status) ? "Live" : "Saved recording"}</span><span className={`issue-count ${item.issueCount ? "has-issues" : ""}`}>{item.issueCount ? <Flag size={10} /> : <Check size={11} />}{item.issueCount} candidates</span></div>
              {selectedId === item.id && <span className="selected-arrow"><ChevronRight size={13} /></span>}
            </button>)}
          </div>
          <div className="sidebar-bottom">
            <div className="runner-card"><span className={`runner-dot ${catalog?.runner.online ? "online" : ""}`} /><div><strong>{catalog?.runner.online ? "Mac runner connected" : "Mac runner offline"}</strong><span>{catalog?.runner.online ? catalog.runner.unityReady ? "Grid + Unity available" : "Grid available" : "Saved evidence stays available"}</span></div><span className="runner-device"><Activity size={15} /></span></div>
            <div className="powered-card"><span className="nebius-symbol"><Zap size={17} fill="currentColor" /></span><div><strong>Powered by Nemotron</strong><span>Lightning explores. Ultra reviews.</span><span className="nebius-credit">Inference on Nebius Token Factory</span></div></div>
            <div className="sidebar-footnote"><ShieldCheck size={12} />Every finding has a trail.</div>
          </div>
        </aside>

        <div className="main-content">
          {error && <div className="error-banner" role="alert"><TriangleAlert size={16} /><span>{error}</span><button onClick={() => setRefresh((n) => n + 1)}>Try again</button></div>}
          {detailError && <div className="error-banner" role="alert"><TriangleAlert size={16} /><span>{detailError}</span><button onClick={() => setRefresh((n) => n + 1)}>Reload</button></div>}
          {!run ? <WorkspaceEmpty loading={!!selectedId && !detailError} hasSessions={!!catalog?.runs.length} canControl={!!newTestAvailable} onNewTest={() => setShowNewTest(true)} /> : <>
            <div className="page-breadcrumb"><span>Workspace</span><ChevronRight size={12} /><span>{run.summary.game === "unity" ? "Unity" : "Grid"}</span><ChevronRight size={12} /><strong>Session overview</strong></div>
            <section className="session-heading">
              <div><div className="eyebrow"><span className={`mode-label ${isLive ? "mode-live" : ""}`}><i />{isLive ? "LIVE TEST" : "RECORDING PLAYBACK"}</span><span>SEED {run.summary.seed}</span></div><h3>{run.summary.name}</h3><p>{gameName(run.summary.game)}<span>·</span>{date(run.summary.createdAt)}<span>·</span>{duration(run.summary.durationSeconds)}</p></div>
              <div className="session-actions"><Status run={run.summary} /><div className="download-menu"><a className="button button-secondary" href={`/api/runs/${encodeURIComponent(run.summary.id)}/download?format=markdown`} download><Download size={14} />Report</a><a className="icon-button download-json" href={`/api/runs/${encodeURIComponent(run.summary.id)}/download?format=json`} download title="Download complete JSON evidence" aria-label="Download complete JSON evidence"><FileJson size={16} /></a></div>{isLive && catalog?.canControl && <button className="button button-stop" onClick={stop} disabled={stopping}>{stopping ? <Loader2 size={13} className="spin" /> : <Square size={12} />}Stop test</button>}</div>
            </section>

            <div className="session-metrics">
              <Metric icon={Gamepad2} label="Executed controls" value={run.summary.controls.toLocaleString()} detail={`${run.summary.resets} resets included`} />
              <Metric icon={Bug} label="AI: likely bugs" value={String(bugCount)} detail={`${softwareCount} software signals${authoredCount ? ` · ${authoredCount} authored review controls` : ""} · ${run.summary.pendingCount} pending`} accent={bugCount > 0} />
              <Metric icon={Sparkles} label="Model requests" value={String(run.summary.requests)} detail="Lightning + Ultra" />
              <Metric icon={ArrowUpRight} label="Estimated spend" value={money(run.summary.estimatedCostUsd)} detail={!run.summary.costKnown ? "Unknown usage · estimate incomplete" : run.summary.budgetUsd !== null ? `of ${money(run.summary.budgetUsd)} configured cap` : "Recorded provider usage"} />
            </div>

            <div className="workflow-bar"><span className="workflow-label">THE TEST LOOP</span><div className="workflow-phases">{PHASES.map((phase, i) => {
              const active = run.summary.phase === phase.value && isLive;
              const performed = phase.value === "sweep" ? run.frames.some((frame) => frame.phase === "sweep") : phase.value === "explore" ? run.frames.some((frame) => frame.phase === "llm") || run.timeline.some((event) => event.kind === "request" && event.phase === "explore") : run.issues.some((issue) => issue.assessment.status === "reviewed") || run.timeline.some((event) => event.kind === "request" && event.phase === "triage");
              const done = performed && !active;
              const notExecuted = !performed && !isLive;
              const Icon = phase.icon;
              return <div key={phase.value} className={`workflow-stage ${active ? "active" : ""} ${done ? "done" : ""} ${notExecuted ? "not-executed" : ""}`}><span className="stage-icon">{done ? <Check size={13} /> : active ? <Loader2 size={13} className="spin" /> : <Icon size={14} />}</span><div><strong>{phase.title}</strong><small>{notExecuted ? "Not executed in this session" : phase.subtitle}</small></div>{i < PHASES.length - 1 && <ArrowRight className="phase-arrow" size={14} />}</div>;
            })}</div></div>

            <div className="workspace-grid">
              <section className="panel viewer-panel">
                <div className="panel-heading"><div><span className="panel-kicker">PLAYTEST VIEWER</span><h2>Follow the evidence</h2></div><span className={`viewer-mode ${isLive && following ? "live" : ""}`}>{isLive && following ? <Radio size={12} /> : <Play size={11} />}{isLive && following ? "Following live" : "Saved observations"}</span></div>
                <SceneMap run={run} index={frameIndex} evidenceSteps={selectedIssue?.evidenceSteps ?? []} />
                <Playback run={run} frameIndex={frameIndex} frame={frame} playing={playing} speed={speed} following={following} isLive={isLive} onSeek={seek} onPlay={() => { followingRef.current = false; setFollowing(false); setPlaybackEnd(null); if (frameIndex >= run.frames.length - 1) setFrameIndex(0); setPlaying((value) => !value); }} onSpeed={setSpeed} onFollow={followLive} issue={selectedIssue} />
                <ActionCard frame={frame} />
              </section>

              <section className="panel findings-panel">
                <div className="findings-tabs"><button className={activeTab === "findings" ? "active" : ""} onClick={() => setActiveTab("findings")}><Flag size={14} />Findings<span>{run.issues.length}</span></button><button className={activeTab === "activity" ? "active" : ""} onClick={() => setActiveTab("activity")}><Activity size={14} />Activity</button><span className="tabs-spacer" /><ListFilter size={15} className="muted" /></div>
                {activeTab === "activity" ? <ActivityList run={run} /> : <>
                  <div className="findings-caption">Ultra reviews each candidate against recorded game feedback.</div>
                  <div className="finding-list">
                    {!run.issues.length ? <div className="no-findings"><span><ShieldCheck size={28} /></span><h3>{isLive ? "Watching for surprises" : "No signals captured"}</h3><p>{isLive ? "New candidates will appear here as the game responds." : "This session recorded no software anomaly candidates. This is evidence from one run, not a guarantee of correctness."}</p></div> : run.issues.map((issue) => <button key={issue.id} className={`finding-item ${selectedIssueId === issue.id ? "selected" : ""}`} onClick={() => selectIssue(issue)}><div className="finding-index">{String(issue.number).padStart(2, "0")}</div><div className="finding-content"><div className="finding-title">{issue.title}</div><div className="finding-meta"><span className={`verdict-label verdict-${verdictTone(issue)}`}><i />{verdict(issue)}</span><SeverityBadge severity={issue.assessment.severity} />{issue.needsHumanReview && <span className="human-review-label">Human review</span>}</div></div><ChevronRight size={14} /></button>)}
                  </div>
                  {selectedIssue && <div className="finding-preview"><div className="preview-heading"><Sparkles size={14} /><span>NEMOTRON ULTRA REVIEW</span><span className={`review-dot verdict-${verdictTone(selectedIssue)}`} /></div><h3>{selectedIssue.assessment.title ?? selectedIssue.title}</h3><p>{selectedIssue.assessment.playerImpact ?? (selectedIssue.assessment.status === "pending" ? "The candidate is waiting for an evidence review." : "A complete player-impact assessment is not available.")}</p>{selectedIssue.needsHumanReview && <div className="human-review-notice"><TriangleAlert size={14} /><div><strong>Needs a developer’s judgment</strong><span>{selectedIssue.reviewNote ?? "The software signal and model judgment need to be reconciled. Both are retained below."}</span></div></div>}<div className="preview-evidence"><span>Referenced evidence</span><div>{selectedIssue.assessment.evidenceSteps.length ? selectedIssue.assessment.evidenceSteps.map((step) => <button key={step} onClick={() => jumpToStep(step)} title={`Show original source step ${step}`}>Step {step}<ArrowUpRight size={10} /></button>) : <span className="muted">No validated citations</span>}</div></div><a href="#issue-report" className="text-link">Read complete report<ArrowRight size={13} /></a></div>}
                </>}
                <div className="findings-footer"><Info size={13} /><span>AI judgments are tentative and scoped to this session.</span></div>
              </section>
            </div>

            {selectedIssue && <IssueReport issue={selectedIssue} run={run} onStep={jumpToStep} onReproduce={() => {
              const steps = selectedIssue.reproduction.sourceSteps;
              const start = run.frames.findIndex((item) => steps.includes(item.step));
              const end = run.frames.reduce((last, item, index) => steps.includes(item.step) ? index : last, start);
              if (start >= 0) { seek(start); setPlaybackEnd(end); setPlaying(true); document.querySelector(".viewer-panel")?.scrollIntoView({ behavior: "smooth", block: "start" }); }
            }} />}
            <RunDetails run={run} />
          </>}
        </div>
      </div>
      </section>
      <CommunityDirections />
      <AboutAuthor />
      {showNewTest && catalog && <NewTestDialog catalog={catalog} onClose={() => setShowNewTest(false)} onCreated={(id) => { openRun(id); setRefresh((n) => n + 1); setShowNewTest(false); }} />}
    </main>
  );
}

function Metric({ icon: Icon, label, value, detail, accent }: { icon: typeof Scan; label: string; value: string; detail: string; accent?: boolean }) {
  return <div className={`metric ${accent ? "metric-accent" : ""}`}><div className="metric-label">{label}<Icon size={14} /></div><strong>{value}</strong><span>{detail}</span></div>;
}

function WorkspaceEmpty({ loading, hasSessions, canControl, onNewTest }: { loading: boolean; hasSessions: boolean; canControl: boolean; onNewTest: () => void }) {
  return <div className="workspace-empty"><span className="empty-illustration"><Scan size={40} strokeWidth={1} /><Gamepad2 size={24} /></span><span className="eyebrow">AUTOPLAY QA WORKSPACE</span><h3>{loading ? "Opening the evidence" : hasSessions ? "Select a test session" : "Every game has a story to test."}</h3><p>{loading ? "Loading the recorded controls, observations and AI review." : hasSessions ? "Choose a session to inspect what happened, what the AI concluded, and how to reproduce it." : "Start a bounded playtest and turn surprising game behavior into reproducible developer evidence."}</p>{loading ? <Loader2 size={22} className="spin" /> : canControl && <button className="button button-primary" onClick={onNewTest}><Plus size={15} />Start your first test</button>}<div className="empty-workflow"><span><Scan size={17} />Capture</span><ArrowRight size={14} /><span><Zap size={17} />Explore</span><ArrowRight size={14} /><span><Sparkles size={17} />Review</span></div></div>;
}

function Playback({ run, frameIndex, frame, playing, speed, following, isLive, onSeek, onPlay, onSpeed, onFollow, issue }: { run: RunSnapshot; frameIndex: number; frame?: Frame; playing: boolean; speed: number; following: boolean; isLive: boolean; onSeek: (index: number) => void; onPlay: () => void; onSpeed: (speed: number) => void; onFollow: () => void; issue?: Issue }) {
  const count = run.frames.length;
  return <div className="playback"><div className="timeline-top"><span>{frame ? <><strong>Control {frameIndex + 1}</strong><span> / {count}</span></> : "Waiting for controls"}</span><span>{frame?.phase === "sweep" ? "Spatial sweep" : frame ? "Lightning exploration" : run.summary.phase === "queued" ? "Queued" : "Ready"}{frame && <span className="original-step"> · source step {frame.step}</span>}</span></div><div className="timeline-slider"><input aria-label="Recorded control timeline" type="range" min="0" max={Math.max(0, count - 1)} value={frameIndex} disabled={!count} onChange={(event) => onSeek(Number(event.target.value))} style={{ "--progress": `${count > 1 ? frameIndex / (count - 1) * 100 : 0}%` } as React.CSSProperties} />{run.issues.flatMap((item) => item.evidenceSteps.map((step) => {
    const index = run.frames.findIndex((f) => f.step === step);
    return index < 0 || count < 2 ? null : <button key={`${item.id}-${step}`} className={`timeline-marker ${issue?.id === item.id ? "selected" : ""}`} style={{ left: `${index / (count - 1) * 100}%` }} onClick={() => onSeek(index)} aria-label={`Jump to evidence source step ${step}`} title={`Evidence · step ${step}`} />;
  }))}</div><div className="playback-buttons"><div className="transport"><button className="icon-button" aria-label="Go to first control" onClick={() => onSeek(0)} disabled={!count}><SkipBack size={14} /></button><button className="icon-button" aria-label="Previous control" onClick={() => onSeek(frameIndex - 1)} disabled={frameIndex <= 0}><ChevronLeft size={16} /></button><button className="play-button" aria-label={playing ? "Pause recording playback" : "Play recorded observations"} onClick={onPlay} disabled={!count}>{playing ? <Pause size={15} fill="currentColor" /> : <Play size={15} fill="currentColor" />}</button><button className="icon-button" aria-label="Next control" onClick={() => onSeek(frameIndex + 1)} disabled={!count || frameIndex >= count - 1}><ChevronRight size={16} /></button><button className="icon-button" aria-label="Go to latest control" onClick={() => onSeek(count - 1)} disabled={!count}><SkipForward size={14} /></button><label className="speed-control"><select value={speed} onChange={(event) => onSpeed(Number(event.target.value))} aria-label="Playback speed"><option value={.5}>0.5×</option><option value={1}>1×</option><option value={2}>2×</option><option value={4}>4×</option></select></label></div>{isLive ? <button className={`follow-live ${following ? "active" : ""}`} onClick={onFollow}><Radio size={12} />{following ? "Following live" : "Follow live"}</button> : <span className="playback-footnote"><LockKeyhole size={11} />Recording only</span>}</div></div>;
}

function StateRow({ label, observation }: { label: string; observation?: Observation }) {
  return <div className="state-row"><span className="state-label">{label}</span><code>[{formatPosition(observation?.player?.position)}]</code>{observation?.player?.grounded !== undefined && <span className={`grounded-state ${observation.player.grounded ? "grounded" : "airborne"}`}>{observation.player.grounded ? "Grounded" : "Airborne"}</span>}<span className="state-inventory">{observation?.player?.inventory === undefined ? "Items unavailable" : observation.player.inventory.length ? observation.player.inventory.map(String).join(", ") : "No items"}</span></div>;
}
function ActionCard({ frame }: { frame?: Frame }) {
  return <div className="action-card"><div className="action-heading"><span className="action-chip"><Gamepad2 size={13} />{frame?.action ?? "Awaiting feedback"}</span><span>{frame ? `${frame.phase === "sweep" ? "Sweep" : "Lightning"} · control ${frame.phaseStep}` : "No action executed"}</span></div><p className="action-reason">{frame?.reason || (frame?.phase === "sweep" ? "Systematic spatial coverage; the observation records the result of this control." : "The next executed control and its reason will appear here.")}</p><div className="state-comparison"><StateRow label="Before" observation={frame?.before} /><StateRow label="After" observation={frame?.after} /></div><div className="game-feedback"><span><Activity size={12} />GAME FEEDBACK</span><p>{frame?.feedback.length ? frame.feedback.join(" · ") : frame ? "No text feedback returned for this control." : "Waiting for the game response."}</p></div></div>;
}

function ActivityList({ run }: { run: RunSnapshot }) {
  return <div className="activity-list">{!run.timeline.length && <p className="muted">No activity recorded yet.</p>}{[...run.timeline].reverse().slice(0, 60).map((event) => <div className={`activity-event activity-${event.kind}`} key={event.id}><span className="activity-event-icon">{event.kind === "issue" ? <Flag size={12} /> : event.kind === "warning" ? <TriangleAlert size={12} /> : event.kind === "complete" ? <Check size={12} /> : event.kind === "request" ? <Sparkles size={12} /> : <Circle size={9} />}</span><div><strong>{event.label}</strong>{event.detail && <p>{event.detail}</p>}<span>{duration(event.elapsedSeconds)} into session</span></div></div>)}</div>;
}

function IssueReport({ issue, run, onStep, onReproduce }: { issue: Issue; run: RunSnapshot; onStep: (step: number) => void; onReproduce: () => void }) {
  const [copied, setCopied] = useState(false);
  const controls = issue.reproduction.actions;
  const chunks: { action: string; count: number; start: number }[] = [];
  controls.forEach((action, i) => {
    const last = chunks[chunks.length - 1];
    if (last?.action === action) last.count++;
    else chunks.push({ action, count: 1, start: i });
  });
  const copy = async () => {
    try { await navigator.clipboard.writeText(controls.join("\n")); setCopied(true); setTimeout(() => setCopied(false), 1800); }
    catch { setCopied(false); }
  };
  const firstEvidence = run.frames.find((item) => issue.evidenceSteps.includes(item.step));
  return <section className="panel report-panel" id="issue-report"><div className="report-heading"><div className="report-title"><span className="report-index">{String(issue.number).padStart(2, "0")}</span><div><span className="panel-kicker">EVIDENCE REPORT</span><h2>{issue.title}</h2></div></div><div className="report-badges"><SeverityBadge severity={issue.assessment.severity} /><span className={`verdict-label verdict-${verdictTone(issue)}`}><i />{verdict(issue)}</span></div></div>{issue.needsHumanReview && <div className="human-review-notice report-review-note"><TriangleAlert size={15} /><div><strong>Human review required</strong><span>{issue.reviewNote ?? "The host evidence and AI judgment need to be reconciled. Read both before using this report."}</span></div></div>}<div className="report-columns"><div className="host-evidence"><div className="section-label"><Target size={14} />{issue.source === "authored_normal_candidate" ? "Authored review control" : "Software signal"}<span>{issue.source === "authored_normal_candidate" ? "AUTHORED" : "OBSERVED"}</span></div><h3>{issue.title}</h3><p className="signal-type">{issue.anomalyType.replaceAll("_", " ")}<span> · Source: </span><code>{issue.source}</code></p><div className="evidence-links"><span>Original steps</span>{issue.evidenceSteps.map((step) => <button key={step} onClick={() => onStep(step)}>#{step}<ArrowUpRight size={11} /></button>)}</div>{firstEvidence && <div className="report-state"><StateRow label="Before" observation={firstEvidence.before} /><StateRow label="After" observation={firstEvidence.after} /></div>}<details className="raw-evidence"><summary>Expected and observed facets<span><Plus size={12} /></span></summary><div><h4>Candidate expectation</h4><pre>{textValue(issue.expected)}</pre><h4>Recorded result</h4><pre>{textValue(issue.actual)}</pre></div></details></div><div className="ai-evidence"><div className="section-label"><Sparkles size={14} />Nemotron Ultra judgment<span>TENTATIVE</span></div>{issue.assessment.title && <h3 className="ai-review-title">{issue.assessment.title}</h3>}<h4>What the player experiences</h4><p>{issue.assessment.playerImpact ?? "The review did not produce a player-impact assessment."}</p><h4>Why this judgment</h4><p>{issue.assessment.rationale ?? (issue.assessment.status === "pending" ? "The evidence review is pending." : "No complete rationale was returned. The software evidence remains available.")}</p>{issue.assessment.evidenceSteps.length > 0 && <div className="evidence-links"><span>Validated citations</span>{issue.assessment.evidenceSteps.map((step) => <button key={step} onClick={() => onStep(step)}>#{step}<ArrowUpRight size={11} /></button>)}</div>}{issue.assessment.limitations.length > 0 && <div className="limitations"><h4><Info size={12} />Limits of this review</h4><ul>{issue.assessment.limitations.map((limitation, i) => <li key={i}>{limitation}</li>)}</ul></div>}{issue.assessment.warnings.length > 0 && <div className="review-warning"><TriangleAlert size={12} /><span>{issue.assessment.warnings.join(" · ")}</span></div>}</div></div><div className="reproduction"><div className="reproduction-heading"><div><h3><Gamepad2 size={15} />Reproduction controls</h3><span>{controls.length} controls from the recorded {issue.reproduction.minimal ? "minimized sequence" : "sequence"} · seed {run.summary.seed}</span></div><div>{issue.reproduction.verified === true ? <span className="replay-verification"><CheckCircle2 size={12} />Execution verified</span> : <span className="replay-verification unverified"><Info size={12} />{issue.reproduction.verified === false ? "Verification failed" : "Not replay-verified"}</span>}<button className="icon-button" onClick={copy} disabled={!controls.length} aria-label="Copy reproduction controls" title="Copy complete controls, one per line">{copied ? <Check size={14} /> : <Copy size={14} />}</button><button className="button button-secondary button-small" onClick={onReproduce} disabled={!controls.length}><Play size={11} />View sequence</button></div></div><div className="control-sequence">{chunks.length ? chunks.map((chunk, index) => <span key={index}><button onClick={() => { const step = issue.reproduction.sourceSteps[chunk.start]; if (step !== undefined) onStep(step); }} title={issue.reproduction.sourceSteps[chunk.start] !== undefined ? `Original step ${issue.reproduction.sourceSteps[chunk.start]}` : "Recorded control"}>{chunk.action}{chunk.count > 1 && <b>×{chunk.count}</b>}</button>{index < chunks.length - 1 && <ChevronRight size={10} />}</span>) : <p className="muted">No usable reproduction sequence was captured.</p>}</div><p className="reproduction-note">Viewing a sequence plays saved observations. Replay verification proves execution; it does not prove the model’s judgment.</p></div></section>;
}

function RunDetails({ run }: { run: RunSnapshot }) {
  return <details className="run-details panel"><summary><span><FileText size={15} />Session details & limitations</span><span>{run.summary.warningCount} warnings · {run.summary.pendingCount} pending<ChevronRight size={14} /></span></summary><div className="run-details-body"><div><h3>Model configuration</h3><dl><dt>Exploration</dt><dd>{run.models.exploration}</dd><dt>Evidence review</dt><dd>{run.models.triage}</dd><dt>Output allowance</dt><dd>{run.limits.maxTokens.toLocaleString()} tokens / request</dd><dt>Request timeout</dt><dd>{run.limits.timeout}s</dd><dt>Exploration limits</dt><dd>{run.limits.modelControls} controls · {run.limits.explorationRequests} requests · {run.limits.explorationSeconds}s</dd><dt>Review limits</dt><dd>{run.limits.triageRequests} requests · {run.limits.triageSeconds}s</dd></dl></div><div><h3>Scope & recording notes</h3><p>Software signals and AI assessments are preserved separately. Negative step IDs refer to the spatial sweep; original phase control numbers remain unchanged.</p><p>Estimated costs use configured prices, and do not represent the provider account balance. Unknown usage and failures may retain their budget reservation.</p>{run.notes.map((note, i) => <p className="recording-note" key={i}><Info size={12} />{note}</p>)}<code className="run-id">Session ID: {run.summary.id}</code></div></div></details>;
}

function NewTestDialog({ catalog, onClose, onCreated }: { catalog: RunsResponse; onClose: () => void; onCreated: (id: string) => void }) {
  const dialogRef = useRef<HTMLDialogElement>(null);
  const requestId = useRef<string | null>(null);
  const [game, setGame] = useState<GameKind>(catalog.runner.unityReady ? "unity" : "grid");
  const [profile, setProfile] = useState<TestProfile>("clean");
  const [seed, setSeed] = useState(0);
  const [budget, setBudget] = useState(1);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const profiles = PROFILES.map((item) => game === "grid" && item.value === "floor" ? { ...item, title: "Blocked tile movement", detail: "Tile movement test environment" } : game === "grid" && item.value === "door" ? { ...item, title: "Exit response", detail: "Exit feedback test environment" } : item);
  useEffect(() => { dialogRef.current?.showModal(); }, []);
  const submit = async (event: React.FormEvent) => {
    event.preventDefault();
    if (!catalog.canControl || !catalog.runner.online || submitting || !Number.isFinite(budget) || budget < .05 || budget > 10) return;
    setSubmitting(true);
    setError(null);
    requestId.current ??= crypto.randomUUID();
    const payload: NewRunRequest = { requestId: requestId.current, game, profile, seed, maxUsd: budget };
    try { const result = await fetchJson<{ id: string }>("/api/runs", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload) }); onCreated(result.id); }
    catch (cause) { setError(safeError(cause)); }
    finally { setSubmitting(false); }
  };
  return <dialog className="test-dialog" ref={dialogRef} onCancel={(event) => { if (submitting) event.preventDefault(); else onClose(); }} aria-labelledby="new-test-title"><div className="dialog-heading"><span className="dialog-icon"><Gamepad2 size={21} /></span><button className="icon-button" aria-label="Close new test dialog" onClick={onClose} disabled={submitting}><X size={18} /></button></div><span className="panel-kicker">NEW PLAYTEST</span><h2 id="new-test-title">Give your game a fresh pair of eyes.</h2><p className="dialog-intro">The Mac runner executes real controls. Lightning explores the feedback, then Ultra reviews any captured signals.</p><form onSubmit={submit}><fieldset disabled={submitting}><label className="form-label">Game environment</label><div className="game-options"><button className={game === "unity" ? "selected" : ""} type="button" disabled={!catalog.runner.unityReady} onClick={() => { setGame("unity"); requestId.current = null; }}><Box size={22} /><div><strong>Unity PhysicsKeyDoor</strong><span>{catalog.runner.unityReady ? "3D physics · real Unity game" : "Unity runner is not ready"}</span></div>{game === "unity" && <Check size={16} />}</button><button className={game === "grid" ? "selected" : ""} type="button" onClick={() => { setGame("grid"); requestId.current = null; }}><Grid2X2 size={22} /><div><strong>Grid KeyDoor</strong><span>Terminal game · observed tile map</span></div>{game === "grid" && <Check size={16} />}</button></div><div className="form-row"><label className="form-label">Test environment<select value={profile} onChange={(event) => { setProfile(event.target.value as TestProfile); requestId.current = null; }}>{profiles.map((item) => <option value={item.value} key={item.value}>{item.title}</option>)}</select><small>{profiles.find((item) => item.value === profile)?.detail}</small></label><label className="form-label">Seed<select value={seed} onChange={(event) => { setSeed(Number(event.target.value)); requestId.current = null; }}><option value={0}>0</option><option value={1}>1</option><option value={5}>5</option></select><small>Repeatable starting environment</small></label></div><label className="form-label budget-label">Estimated spend cap<span className="budget-field"><span>$</span><input type="number" min="0.05" max="10" step="0.01" value={budget} onChange={(event) => { setBudget(Number(event.target.value)); requestId.current = null; }} required aria-label="Maximum estimated spend in US dollars" /><span>USD</span></span><small>$0.05–$10.00 per test. Configured prices set the estimate; unknown usage keeps its reservation. The runner also enforces its remaining shared allowance.</small></label><div className="test-boundaries"><ShieldCheck size={17} /><div><strong>A bounded test, with an evidence trail</strong><span>24 model controls · 28 exploration requests · up to 8 review requests · 45s request timeout. You can stop the test while it runs.</span></div></div>{error && <div className="error-banner" role="alert"><TriangleAlert size={14} /><span>{error}</span></div>}<div className="dialog-footer"><span><i className="runner-dot online" />{catalog.runner.name || "Mac runner"}</span><button type="submit" className="button button-primary" disabled={!catalog.canControl || !catalog.runner.online || !Number.isFinite(budget) || budget < .05 || budget > 10}>{submitting ? <Loader2 size={14} className="spin" /> : <Play size={13} fill="currentColor" />}{submitting ? "Starting test…" : `Start test · up to ${money(budget)}`}</button></div></fieldset></form></dialog>;
}
