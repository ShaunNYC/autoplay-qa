import type { RunSnapshot } from "./types";
import recording0 from "../public/demo/unity-floor.json";
import recording1 from "../public/demo/unity-wall.json";
import recording2 from "../public/demo/unity-clean.json";
import recording3 from "../public/demo/unity-door.json";
import recording4 from "../public/demo/unity-key.json";
import recording5 from "../public/demo/unity-reset.json";
import recording6 from "../public/demo/grid-coverage.json";
export const recordings: Record<string, RunSnapshot> = {
  "unity-floor": recording0 as unknown as RunSnapshot,
  "unity-wall": recording1 as unknown as RunSnapshot,
  "unity-clean": recording2 as unknown as RunSnapshot,
  "unity-door": recording3 as unknown as RunSnapshot,
  "unity-key": recording4 as unknown as RunSnapshot,
  "unity-reset": recording5 as unknown as RunSnapshot,
  "grid-coverage": recording6 as unknown as RunSnapshot,
};
