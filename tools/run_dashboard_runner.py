"""Load local bridge settings and start the bounded Mac dashboard runner."""
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from autoplay_qa.dashboard_runner import main  # noqa: E402


def run():
    os.chdir(ROOT)
    bridge_file = ROOT / ".env.dashboard"
    if bridge_file.is_file():
        for line in bridge_file.read_text().splitlines():
            key, separator, value = line.strip().partition("=")
            if separator and key in {"DASHBOARD_SITE_URL", "DASHBOARD_RUNNER_TOKEN"}:
                os.environ.setdefault(key, value.strip().strip("\"'"))
    args = sys.argv[1:]
    config_file = ROOT / ".dashboard" / "runner-config.json"
    if "--unity-player" not in args and config_file.is_file():
        config = json.loads(config_file.read_text())
        if isinstance(config.get("unity_player"), str):
            args = [*args, "--unity-player", config["unity_player"]]
    return main(args)


if __name__ == "__main__":
    raise SystemExit(run())
