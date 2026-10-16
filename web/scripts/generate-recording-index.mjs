import { readFileSync, writeFileSync } from "node:fs";
const manifest = JSON.parse(readFileSync(new URL("../lib/demo-manifest.json", import.meta.url), "utf8"));
const lines = ['import type { RunSnapshot } from "./types";'];
manifest.runs.forEach((run, i) => {
  if (!/^[a-zA-Z0-9_-]{8,100}$/.test(run.id)) throw new Error("Unsafe recorded run ID");
  lines.push(`import recording${i} from "../public/demo/${run.id}.json";`);
});
lines.push("export const recordings: Record<string, RunSnapshot> = {");
manifest.runs.forEach((run, i) => lines.push(`  ${JSON.stringify(run.id)}: recording${i} as unknown as RunSnapshot,`));
lines.push("};", "");
writeFileSync(new URL("../lib/demo-recordings.ts", import.meta.url), lines.join("\n"));
