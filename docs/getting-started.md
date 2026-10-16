# Getting started

## Requirements

- Python 3.11+ for the core, CLI, and Python tests. Run commands from the cloned
  repository root; the core has no third-party Python runtime dependencies.
- Node.js 22.13+ and npm for the optional web application.
- Your own Unity installation/project for the Unity bridge.
- A compatible original Mars project, its build dependencies, Node, Playwright,
  and a Chromium-based browser for Mars browser work.

The Grid examples below need only Python. They make no provider calls.

## Record a local play session

```sh
python3 -m autoplay_qa play --seed 0 --log runs/my-play.jsonl
```

Use `n`, `s`, `e`, `w`, `i` (interact), `.` (wait), and `q` (finish). Choose a
fresh output filename for each session. Then inspect that ordinary Grid record:

```sh
python3 -m autoplay_qa replay runs/my-play.jsonl
python3 -m autoplay_qa review runs/my-play.jsonl --evidence
python3 -m autoplay_qa report runs/my-play.jsonl --out runs/my-play.md
```

An uneventful playthrough can produce a report with no candidates. It still
demonstrates the observation/action recording path.

## Run the independent scripted benchmark

```sh
python3 -m autoplay_qa benchmark --scripted --seeds 0 --record-dir runs/benchmark
```

The command executes controlled reference fixtures and prints its own result
table. The output directory contains case artifacts and Markdown reports with
fixture labels. Use those results and reports for this workflow; benchmark
JSONL is not the ordinary session `replay` quickstart format.

The hybrid integration check also uses a fake model, not a provider:

```sh
python3 -m autoplay_qa benchmark --hybrid --seeds 0
```

See [evidence formats](evidence-formats.md) for the distinction between these
outputs, ordinary session recordings, browser traces, and web demo JSON.

## Optional Nebius configuration

These instructions are for a contributor who chooses to use their own account
and permits the relevant public game state and evidence to be sent to Nebius.
They are not required for offline use.

Copy `.env.example` to `.env` and fill in your own `NEBIUS_API_KEY`. Keep that
file untracked. The default API base is
`https://api.tokenfactory.nebius.com/v1/`; environment variables override `.env`.

The default exploration model is `nvidia/Nemotron-3_5-Lightning`; Ultra review
uses `nvidia/Nemotron-3-Ultra-550b-a55b`. Check your account's available model IDs
before planning a paid run. `doctor --list-models` makes an authenticated catalog
request; `doctor --probe-inference` additionally makes one paid inference probe.

An explicitly bounded exploration example is:

```sh
python3 -m autoplay_qa explore --focus qa \
  --max-actions 6 --max-requests 8 --max-seconds 120 \
  --timeout 30 --max-tokens 8192 --log runs/my-ai-session.jsonl
```

This command makes real inference requests. Its limits bound actions, requests,
time, and output tokens; the direct `explore` CLI does **not** expose a dollar
cap. Estimate cost using current provider pricing before running it. Default
`lean-v1` has bounded repair/retry behavior inside the request cap; adaptive
protocols stop on provider failures. All attempts remain part of the record.

For candidate review, first prepare evidence offline:

```sh
python3 -m autoplay_qa triage runs/my-ai-session.jsonl --dry-run
```

The [triage guide](triage.md) explains the separate live review command and its
configured-price cost cap. A configured estimate is not an account balance or
a provider billing guarantee.

## Tests and optional interfaces

```sh
python3 -m unittest discover -s tests
```

Implementation tests use fake or stub HTTP. Some native integration work needs
its corresponding runtime and source. The public checkout does not include
private game projects or prior operator execution archives.

Tests requiring optional runtimes or unavailable historical fixtures report
explicit skips. Run the available tests from the repository root and include
any skips when describing your environment.

Continue with [web setup](../web/README.md),
[runtime integrations](integrations.md), or [research modules](research-roadmap.md).
