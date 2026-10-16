/** Test-only stdio connection to the unchanged Outpost: Mars engine. */
import { createHash, randomUUID } from "node:crypto";
import { mkdir, readFile, writeFile, rename, unlink } from "node:fs/promises";
import { createRequire } from "node:module";
import path from "node:path";
import { pathToFileURL } from "node:url";
import readline from "node:readline";

export const PROTOCOL = "autoplay-qa/1";
const MAX_LINE_BYTES = 1024 * 1024;
const digest = (bytes) => createHash("sha256").update(bytes).digest("hex");

/** Bundle the original module locally, recording both source and bundle hashes. */
export async function loadEngine({ projectDir, cacheDir }) {
  const project = path.resolve(projectDir);
  const cache = path.resolve(cacheDir);
  const source = path.join(project, "src", "mars", "engine.ts");
  const sourceBytes = await readFile(source);
  const sourceSha256 = digest(sourceBytes);
  await mkdir(cache, { recursive: true });
  const bundlePath = path.join(cache, `engine-${sourceSha256}.mjs`);
  const recordPath = `${bundlePath}.json`;
  let bundleSha256;
  try {
    const record = JSON.parse(await readFile(recordPath, "utf8"));
    const bundle = await readFile(bundlePath);
    if (record.sourceSha256 !== sourceSha256 || record.bundleSha256 !== digest(bundle)) {
      throw new Error("Invalid engine cache.");
    }
    bundleSha256 = record.bundleSha256;
  } catch {
    const require = createRequire(path.join(project, "package.json"));
    let esbuild;
    try {
      esbuild = require("esbuild");
    } catch {
      const fallback = createRequire(new URL("../web/package.json", import.meta.url));
      esbuild = fallback("esbuild");
    }
    const built = await esbuild.build({
      entryPoints: [source],
      bundle: true,
      platform: "node",
      format: "esm",
      target: "node22",
      write: false,
      logLevel: "silent",
      sourcemap: false,
    });
    if (digest(await readFile(source)) !== sourceSha256) {
      throw new Error("Original engine changed while bundling; no game was started.");
    }
    const bundle = built.outputFiles[0].contents;
    bundleSha256 = digest(bundle);
    const temporaryPath = `${bundlePath}.${randomUUID()}.tmp`;
    try {
      await writeFile(temporaryPath, bundle, { flag: "wx", mode: 0o600 });
      await rename(temporaryPath, bundlePath);
      await writeFile(recordPath, JSON.stringify({ sourceSha256, bundleSha256 }) + "\n", { mode: 0o600 });
    } finally {
      await unlink(temporaryPath).catch(() => {});
    }
  }
  const engine = await import(pathToFileURL(bundlePath).href);
  return {
    engine,
    manifest: {
      source: "src/mars/engine.ts",
      sourceSha256,
      bundleSha256,
      rulesVersion: engine.MARS_RULES,
      trainingMaps: engine.TRAINING_MAPS.map((map) => map.id),
      missionMaps: engine.MISSION_MAPS.map((map) => map.id),
    },
  };
}

function seedValue(value) {
  if (!Number.isSafeInteger(value)) throw new Error("Game seed must be an integer.");
  return value;
}

/** Each control executes real blocks using the previous real DroneState. */
export function createGame(engine, { mapId = null, seed = 0, version = "mars-engine-v1" } = {}) {
  seedValue(seed);
  if (mapId !== null && !engine.ALL_MAPS.some((map) => map.id === mapId)) {
    throw new Error("Unknown Mars map.");
  }
  let selectedSeed = seed;
  let map;
  let state;
  let tick;
  let events;
  let last;
  let controlBlocks;

  function selectMap() {
    const index = ((selectedSeed % engine.ALL_MAPS.length) + engine.ALL_MAPS.length) % engine.ALL_MAPS.length;
    map = engine.ALL_MAPS.find((candidate) => candidate.id === mapId) ?? engine.ALL_MAPS[index];
  }

  function catalog() {
    const blocks = new Map(engine.ACTIONS.map((action) => [action, { id: "control", kind: action }]));
    for (let y = 0; y < map.height; y++) {
      for (let x = 0; x < map.width; x++) {
        blocks.set(`moveTo(${x},${y})`, { id: "control", kind: "moveTo", x, y });
      }
    }
    for (const [kind, values] of [["set", [0, 1]], ["add", [1, -1]]]) {
      for (const value of values) blocks.set(`${kind}(${value})`, { id: "control", kind, value });
    }
    for (const [count, kind] of [[0, "forward"], [2, "forward"], [2, "place"], [2, "bomb"]]) {
      blocks.set(`repeat(${count},${kind})`, {
        id: "control", kind: "repeat", value: count, body: [{ id: "body", kind }],
      });
    }
    return blocks;
  }

  function reset(nextSeed = selectedSeed) {
    selectedSeed = seedValue(nextSeed);
    selectMap();
    state = engine.initialState(map);
    tick = 0;
    controlBlocks = catalog();
    const ready = engine.execute(map, [], [], state);
    events = ready.frames.map((frame) => frame.message);
    last = { actions: 0, instructions: 0, success: ready.success, error: null };
    return observe();
  }

  function observe() {
    const grid = Array.from({ length: map.height }, (_, y) =>
      Array.from({ length: map.width }, (_, x) => {
        if (state.x === x && state.y === y) return "@";
        return map.rocks.includes(y * map.width + x) ? "#" : ".";
      }).join(""),
    );
    const entities = map.target.map((target, index) => ({
      id: `cell-${index % map.width}-${Math.floor(index / map.width)}`,
      kind: "matrix-cell",
      position: [index % map.width, Math.floor(index / map.width)],
      state: {
        builtHeight: state.tiles[index],
        targetHeight: target,
        visited: state.visited?.[index] === true,
        bomb: state.bombs?.[index] === true,
        rock: map.rocks.includes(index),
      },
    }));
    entities.push({
      id: "drone", kind: "drone", position: [state.x, state.y],
      state: { facing: engine.DIRECTIONS[state.facing], variableV: state.v, blockSupply: state.supply, bombSupply: state.bombSupply },
    });
    entities.push({
      id: "mission", kind: "mission-display", position: null,
      state: {
        name: map.name,
        objective: map.objective ?? "construction",
        columns: map.width,
        rows: map.height,
        complete: engine.isSolved(map, state),
        bombedCells: engine.bombedCells(map, state),
        visitedCells: engine.visitedCells(map, state),
        correctCells: engine.correctCells(map, state),
        totalCells: engine.traversableCells(map),
      },
    });
    entities.push({ id: "execution", kind: "execution-display", position: null, state: structuredClone(last) });
    return {
      tick,
      status: last.success ? "won" : "playing",
      coordinates: "Zero-based [x,y]; x increases east, y increases south. North, East, South, West are headings.",
      player: {
        position: [state.x, state.y],
        inventory: [`blocks:${state.supply}`, `bombs:${state.bombSupply}`],
        facing: engine.DIRECTIONS[state.facing],
        variableV: state.v,
      },
      grid,
      legend: { "#": "rock wall", ".": "floor cell", "@": "player drone" },
      entities,
      events: [...events],
      controls: [...controlBlocks.keys()],
    };
  }

  function executeProgram(program, functions = []) {
    const result = engine.execute(map, structuredClone(program), structuredClone(functions), state);
    state = structuredClone(result.frames.at(-1).state);
    tick++;
    // Keep actual engine feedback, including a rejection after partial execution.
    events = result.frames.slice(1).map((frame) => frame.message);
    last = { actions: result.actions, instructions: result.instructions, success: result.success, error: result.error ?? null };
    return { observation: observe(), execution: structuredClone(result) };
  }

  function act(control) {
    if (typeof control !== "string" || !controlBlocks.has(control)) throw new Error("Unknown Mars control.");
    return executeProgram([controlBlocks.get(control)]).observation;
  }

  reset(seed);
  return {
    observe,
    act,
    reset,
    executeProgram,
    get mapId() { return map.id; },
    get seed() { return selectedSeed; },
    get controls() { return [...controlBlocks.keys()]; },
    get version() { return version; },
  };
}

function argumentsOf(argv) {
  const result = {};
  for (let index = 0; index < argv.length; index += 2) {
    const name = argv[index], value = argv[index + 1];
    if (!new Set(["--project", "--cache", "--map", "--seed"]).has(name) || value === undefined) {
      throw new Error("Invalid bridge arguments.");
    }
    result[name.slice(2)] = value;
  }
  if (!result.project || !result.cache) throw new Error("Project and cache are required.");
  return result;
}

async function serve() {
  const config = argumentsOf(process.argv.slice(2));
  const { engine, manifest } = await loadEngine({ projectDir: config.project, cacheDir: config.cache });
  const version = `mars-rules${manifest.rulesVersion}-${manifest.sourceSha256.slice(0, 12)}`;
  const game = createGame(engine, { mapId: config.map ?? null, seed: config.seed === undefined ? 0 : Number(config.seed), version });
  const capabilities = { grid: true, legend: true, entities: true, reset: true, seed: true, defect_injection: false };
  const lines = readline.createInterface({ input: process.stdin, crlfDelay: Infinity });
  for await (const line of lines) {
    let request;
    let response;
    try {
      if (Buffer.byteLength(line, "utf8") > MAX_LINE_BYTES) throw new Error("Request exceeds protocol size limit.");
      request = JSON.parse(line);
      if (!request || Array.isArray(request) || typeof request.id !== "string" || !request.id || typeof request.type !== "string") {
        throw new Error("Invalid request envelope.");
      }
      response = { id: request.id, type: request.type, ok: true };
      if (request.type === "hello") {
        if (request.protocol !== PROTOCOL) throw new Error("Unsupported protocol version.");
        Object.assign(response, { protocol: PROTOCOL, game: { id: "outpost-mars", version }, capabilities, controls: game.controls });
      } else if (request.type === "sourceManifest") {
        response.manifest = { ...manifest, mapId: game.mapId };
      } else if (request.type === "listmaps") {
        response.maps = engine.ALL_MAPS.map(({ id, name, width, height, objective }) => ({ id, name, width, height, objective: objective ?? "construction" }));
      } else if (request.type === "observe") response.observation = game.observe();
      else if (request.type === "act") response.observation = game.act(request.control);
      else if (request.type === "reset") response.observation = game.reset(request.seed ?? game.seed);
      else if (request.type === "close") {
        process.stdout.write(JSON.stringify(response) + "\n");
        lines.close();
        return;
      } else throw new Error("Unknown request type.");
    } catch (error) {
      response = { id: request?.id ?? null, type: "error", ok: false, error: { code: "invalid_request", message: error instanceof Error ? error.message : "Game request failed.", retryable: false } };
    }
    process.stdout.write(JSON.stringify(response) + "\n");
  }
}

if (process.argv[1] && path.resolve(process.argv[1]) === path.resolve(new URL(import.meta.url).pathname)) {
  serve().catch(() => {
    // A source path or esbuild diagnostic must never become model-visible feedback.
    process.stderr.write("Mars bridge could not load the local game engine.\n");
    process.exitCode = 1;
  });
}
