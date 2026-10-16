"use client";

import { useMemo } from "react";
import type { Entity, Observation, RunSnapshot } from "@/lib/types";

interface Props {
  run: RunSnapshot;
  index: number;
  evidenceSteps: number[];
}

function point(position: number[] | undefined): [number, number] | null {
  if (!position || position.length < 2 || !position.every(Number.isFinite)) return null;
  return [position[0], position.length > 2 ? position[2] : position[1]];
}

function entityColor(entity: Entity) {
  const kind = entity.kind.toLowerCase();
  if (kind.includes("door")) return entity.state?.open ? "#b8d5bc" : "#dbae6a";
  if (kind.includes("exit") || kind.includes("goal")) return "#b6d8c5";
  if (kind.includes("floor")) return "#edeae2";
  if (kind.includes("key")) return "#ddb45e";
  return "#71828b";
}

function observedEntities(scene: Observation | null, current: Observation | undefined): Entity[] {
  const all = new Map<string, Entity>();
  for (const entity of scene?.entities ?? []) all.set(entity.id, entity);
  for (const entity of current?.entities ?? []) {
    const old = all.get(entity.id);
    all.set(entity.id, { ...old, ...entity, position: entity.position ?? old?.position ?? [] });
  }
  return [...all.values()].filter((entity) => entity.state?.present !== false && point(entity.position));
}

export default function SceneMap({ run, index, evidenceSteps }: Props) {
  const frame = run.frames[index];
  const observation = frame?.after ?? run.scene;
  const grid = observation?.map ?? run.scene?.map;
  const isGridGame = run.summary.game === "grid";
  const isGrid = isGridGame && !!grid?.length;
  const entities = useMemo(() => observedEntities(run.scene, observation ?? undefined), [run.scene, observation]);
  const player = useMemo(() => point(observation?.player?.position), [observation]);
  const paths = useMemo(() => {
    const segments: [number, number][][] = [];
    let segment: [number, number][] = [];
    const first = point(run.scene?.player?.position);
    if (first) segment.push(first);
    for (const item of run.frames.slice(0, index + 1)) {
      if (item.action === "reset") {
        if (segment.length) segments.push(segment);
        segment = [];
      }
      const next = point(item.after.player?.position);
      if (next) segment.push(next);
    }
    if (segment.length) segments.push(segment);
    return segments;
  }, [run.scene, run.frames, index]);

  const bounds = useMemo(() => {
    if (isGrid) return { minX: -0.7, maxX: Math.max(...grid!.map((row) => row.length)) - 0.3, minY: -0.7, maxY: grid!.length - 0.3 };
    const xs: number[] = [], ys: number[] = [];
    for (const entity of entities) {
      const p = point(entity.position);
      if (!p) continue;
      const sizeX = entity.size?.[0] ?? 0.5;
      const sizeY = entity.size?.[2] ?? entity.size?.[1] ?? 0.5;
      xs.push(p[0] - sizeX / 2, p[0] + sizeX / 2);
      ys.push(p[1] - sizeY / 2, p[1] + sizeY / 2);
    }
    if (player) { xs.push(player[0]); ys.push(player[1]); }
    if (!xs.length) return { minX: -5, maxX: 5, minY: -5, maxY: 5 };
    return { minX: Math.min(...xs) - 0.8, maxX: Math.max(...xs) + 0.8, minY: Math.min(...ys) - 0.8, maxY: Math.max(...ys) + 0.8 };
  }, [entities, isGrid, grid, player]);

  const width = bounds.maxX - bounds.minX;
  const height = bounds.maxY - bounds.minY;
  const sx = (x: number) => 56 + (x - bounds.minX) / width * 488;
  const sy = (y: number) => isGrid ? 30 + (y - bounds.minY) / height * 300 : 330 - (y - bounds.minY) / height * 300;
  const unitX = 488 / width;
  const unitY = 300 / height;
  const issuePoints = run.frames.filter((item) => evidenceSteps.includes(item.step)).map((item) => ({ point: point(item.after.player?.position), step: item.step })).filter((item) => item.point);

  return (
    <div className="scene-canvas">
      <div className="canvas-caption"><span>{isGridGame ? isGrid ? "GRID MAP" : "GRID · AWAITING OBSERVATION" : "UNITY · X/Z TOP VIEW"}</span><span>{observation ? "Observed game state" : "No game state recorded yet"}</span></div>
      <svg className="game-map" viewBox="0 0 600 370" role="img" aria-label={!observation ? `Waiting for the first ${isGridGame ? "Grid" : "Unity"} game observation` : isGridGame ? "Recorded grid map showing player position, path and selected evidence" : "Top view of recorded Unity geometry, player path and selected evidence"}>
        <defs>
          <pattern id="map-dots" width="16" height="16" patternUnits="userSpaceOnUse"><circle cx="1" cy="1" r="0.65" fill="#c5cdc8" /></pattern>
          <filter id="player-shadow"><feDropShadow dx="0" dy="2" stdDeviation="3" floodColor="#153a3c" floodOpacity=".18" /></filter>
        </defs>
        <rect x="0" y="0" width="600" height="370" fill="url(#map-dots)" />
        {isGrid ? grid!.flatMap((row, y) => [...row].map((tile, x) => {
          const legend = (observation?.legend ?? run.scene?.legend)?.[tile]?.toLowerCase() ?? "";
          const wall = tile === "#" || legend.includes("wall");
          const door = /[dD]/.test(tile) || legend.includes("door");
          const key = /[kK]/.test(tile) || legend.includes("key");
          const exit = /[eE]/.test(tile) || legend.includes("exit");
          return <g key={`${x}-${y}`}><rect x={sx(x) - unitX * .44} y={sy(y) - unitY * .44} width={unitX * .88} height={unitY * .88} rx="3" fill={wall ? "#71828b" : door ? "#dbae6a" : exit ? "#b6d8c5" : "#eeebe4"} stroke={wall ? "#63737d" : "#dedfd5"} strokeWidth=".7" />{key && <path d={`M${sx(x)},${sy(y) - 5} l5,5 l-5,5 l-5,-5z`} fill="#b58d37" />}</g>;
        })) : [...entities].sort((a, b) => Number(!a.kind.includes("floor")) - Number(!b.kind.includes("floor"))).map((entity) => {
          const p = point(entity.position)!;
          const sizeX = (entity.size?.[0] ?? .6) * unitX;
          const sizeY = (entity.size?.[2] ?? .6) * unitY;
          const key = entity.kind.toLowerCase().includes("key");
          return <g key={entity.id}><title>{entity.kind}: {entity.id}</title>{key ? <g transform={`translate(${sx(p[0])},${sy(p[1])})`}><circle r="9" fill="#fbefd0" stroke="#d6b665" /><path d="M-2,-3 a3,3 0 1,0 0,6 a3,3 0 1,0 0,-6 M1,0 h5 m-1,0 v3" fill="none" stroke="#aa7e23" strokeWidth="1.5" /></g> : <rect x={sx(p[0]) - sizeX / 2} y={sy(p[1]) - sizeY / 2} width={Math.max(4, sizeX)} height={Math.max(4, sizeY)} rx={entity.kind.includes("floor") ? 2 : 3} fill={entityColor(entity)} stroke={entity.kind.includes("floor") ? "#d6d8ce" : "#607579"} strokeWidth=".6" opacity={entity.state?.open === true ? .6 : 1} />}</g>;
        })}
        {paths.map((path, i) => path.length > 1 && <polyline key={i} points={path.map((p) => `${sx(p[0])},${sy(p[1])}`).join(" ")} fill="none" stroke="#528e8b" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" opacity={i === paths.length - 1 ? .78 : .2} />)}
        {issuePoints.map((item) => <g key={item.step} transform={`translate(${sx(item.point![0])},${sy(item.point![1])})`}><circle r="15" fill="#f6d8ca" fillOpacity=".8" stroke="#c96c4d" strokeWidth="1.3" strokeDasharray="3 3" /><circle r="3" fill="#bc654c" /></g>)}
        {player && <g transform={`translate(${sx(player[0])},${sy(player[1])})`} filter="url(#player-shadow)"><circle r="15" fill="#244f50" fillOpacity=".12" /><circle r="8.5" fill="#173f45" stroke="#fff" strokeWidth="2.8" /><circle r="2.2" fill="#abdcb2" /></g>}
        {!player && !entities.length && !isGrid && <text x="300" y="180" textAnchor="middle" fill="#637675" fontSize="14">Waiting for the first observation</text>}

      </svg>
      <div className="map-legend"><span><i className="legend-player" />Player</span><span><i className="legend-path" />Executed path</span><span><i className="legend-wall" />Wall</span><span><i className="legend-door" />Door</span><span><i className="legend-exit" />Exit</span><span><i className="legend-evidence" />Evidence</span></div>
    </div>
  );
}
