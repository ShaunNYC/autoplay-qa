#!/usr/bin/env node
/**
 * Offline oracle for the original Mars engine, independent of the QA bridge.
 * All supplied programs are deterministic test fixtures, never learner prompts.
 * The project is read only. The temporary bundle is removed after execution.
 */
import { createHash } from "node:crypto";
import { mkdtemp, readFile, rm } from "node:fs/promises";
import { createRequire } from "node:module";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import { pathToFileURL } from "node:url";

const projectFlag = process.argv.indexOf("--project-dir");
if (projectFlag < 0 || !process.argv[projectFlag + 1]) {
  process.stderr.write("Usage: node tools/mars_conformance.mjs --project-dir PROJECT < requests.json\n");
  process.exit(2);
}
const projectDir = resolve(process.argv[projectFlag + 1]);
const relativeSource = "src/mars/engine.ts";
const source = join(projectDir, relativeSource);
const before = await readFile(source);
const esbuild = createRequire(join(projectDir, "package.json"))("esbuild");
const temporaryDir = await mkdtemp(join(tmpdir(), "mars-engine-oracle-"));
try {
  const bundle = join(temporaryDir, "engine.mjs");
  await esbuild.build({
    entryPoints: [source],
    outfile: bundle,
    bundle: true,
    platform: "node",
    format: "esm",
    logLevel: "silent",
    sourcemap: false,
  });
  const engine = await import(pathToFileURL(bundle).href);
  let input = "";
  for await (const chunk of process.stdin) input += chunk;
  const requests = input.trim() ? JSON.parse(input) : [];
  if (!Array.isArray(requests)) throw new Error("Expected a list of offline execution fixtures.");
  const executions = requests.map(({ mapId, program, functions = [], start }) => {
    const map = engine.ALL_MAPS.find((candidate) => candidate.id === mapId);
    if (!map) throw new Error("Unknown original map ID.");
    const originals = JSON.stringify({ map, program, functions, start });
    const execution = engine.execute(map, program, functions, start);
    if (JSON.stringify({ map, program, functions, start }) !== originals) {
      throw new Error("Original engine mutated supplied map, program, functions or start state.");
    }
    return {
      mapId,
      initial: start ?? engine.initialState(map),
      execution,
      final: execution.frames.at(-1).state,
      visited: engine.visitedCells(map, execution.frames.at(-1).state),
      bombed: engine.bombedCells(map, execution.frames.at(-1).state),
    };
  });
  if (!before.equals(await readFile(source))) throw new Error("Original source changed during the offline check.");
  process.stdout.write(JSON.stringify({
    source: {
      relativePath: relativeSource,
      sha256: createHash("sha256").update(before).digest("hex"),
      rulesVersion: engine.MARS_RULES,
    },
    maps: engine.ALL_MAPS,
    executions,
  }));
} finally {
  await rm(temporaryDir, { recursive: true, force: true });
}
