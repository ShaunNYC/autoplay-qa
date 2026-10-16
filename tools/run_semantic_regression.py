"""Run an exported semantic regression on a fresh local browser, no inference."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from autoplay_qa.browser_adapter import BrowserAdapter, LocalBrowserSite
from autoplay_qa.dashboard_runner import _atomic_json
from autoplay_qa.semantic_analysis import run_semantic_regression
from tools import semantic_study as study


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case", type=Path)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--variant", choices=("case", "normal_control"), default="case")
    parser.add_argument("--node", default="node")
    parser.add_argument("--playwright-module")
    parser.add_argument("--executable")
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error("Refusing to overwrite prior regression evidence.")
    source = json.loads((args.prepared / "source-manifest.json").read_bytes())
    for item in source["buildFiles"]:
        if study.file_sha(args.prepared / item["path"]) != item["sha256"]:
            parser.error("The prepared build no longer matches its source manifest.")
    definition = json.loads(args.case.read_bytes())
    with LocalBrowserSite(args.prepared / "dist") as site:
        with BrowserAdapter(site.url + "/#/mars", node=args.node, playwright_module=args.playwright_module,
                            executable_path=args.executable, source_manifest=source) as native:
            adapter = study.EditorScopeAdapter(native)
            trace = run_semantic_regression(adapter, definition, variant=args.variant,
                        source_sha256=source["sourceSha256"], initial_state_sha256=study.digest(adapter.observe()),
                        instance_id=native.source_manifest["instance_id"])
    _atomic_json(args.output, trace)
    passed = trace["complete"] and all(trace["expectation_results"])
    print(json.dumps({"completed": trace["complete"], "assertions_passed": passed,
                      "controls": len(trace["controls"]), "trace": str(args.output), "model_calls": 0}))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
