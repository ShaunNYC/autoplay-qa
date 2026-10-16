# CLI guide

Use AutoPlay QA from a terminal to record a game session, inspect its evidence,
run Nemotron exploration, and reproduce a reported issue. Start with the local
walkthrough below; it needs no account, API key, or game download.

[Install](#install) · [First session](#your-first-session) ·
[Command reference](#command-reference) · [AI testing](#run-an-ai-test) ·
[Your own game](#connect-your-own-game) · [Troubleshooting](#troubleshooting)

## Install

The core CLI requires **Python 3.11 or newer** and has no third-party Python
runtime dependencies. Git is needed for cloning; alternatively, download and
extract the repository ZIP. Node and Unity are only needed for their respective
optional integrations.

These examples use a macOS/Linux shell:

```sh
git clone https://github.com/ShaunNYC/autoplay-qa.git
cd autoplay-qa
python3 --version
python3 -m venv .venv
. .venv/bin/activate
python -m pip install .
autoplay-qa --help
```

This installs the checked-out project, including the `autoplay-qa` command, in
your virtual environment. The build may download setuptools; it makes no model
requests. Activate the environment again in each new terminal. After pulling
updates, repeat `python -m pip install .` to update the installed copy.
Contributors editing Python code can use `python -m pip install -e .` instead.

You can also **skip installation** and run directly from the repository root:

```sh
python3 -m autoplay_qa --help
```

Both entry points expose the same commands. Replace `autoplay-qa` with
`python3 -m autoplay_qa` throughout this guide if using the source directly.
Installed commands can run from another working directory; relative paths such
as `.env` and `runs/` resolve from that directory.

## Your first session

### 1. Record a few controls

```sh
autoplay-qa play --seed 0 --log runs/first-session.jsonl
```

Type `e` and press Enter, then `i` and Enter. Finish with `q` and Enter.

| Input | Control |
| --- | --- |
| `n`, `s`, `e`, `w` | Move north, south, east, west |
| `i` | Interact |
| `.` | Wait |
| `q` | Finish the recording |

The command saves the observations and executed controls to the JSONL file.
Choose a **new log filename** for each session: existing session logs are
protected from overwrite. Omitting `--log` generates a unique filename in
`runs/`.

### 2. Replay and inspect the evidence

```sh
autoplay-qa replay runs/first-session.jsonl
autoplay-qa review runs/first-session.jsonl --evidence
autoplay-qa report runs/first-session.jsonl --out runs/first-session.md
```

- `replay` re-executes the ordinary Grid recording and compares observations.
- `review` prints deterministic session statistics and cited evidence.
- `report` writes a readable Markdown report of candidate issues.

All three commands work without an API key. Open `runs/first-session.md` in a
text editor. A report with no candidates is a valid result for this short
walkthrough; a completed session does not certify the whole game as bug-free.

### 3. Prepare a candidate-review packet

```sh
autoplay-qa triage runs/first-session.jsonl --dry-run \
  --out runs/first-preview.md \
  --json-out runs/first-preview.json \
  --log runs/first-preview.audit.jsonl
```

`--dry-run` prepares any available candidate evidence without loading
credentials or calling a model. With this uneventful first session, expect
`no_candidates`. The files still show the result of the preparation step.

## Command reference

Get the complete flags for any command with `autoplay-qa COMMAND --help`.
Place command-specific options **after** the command name.

| Command | Use | Model / network behavior |
| --- | --- | --- |
| `play` | Record your own controls | No model requests |
| `run` | Run the original observation/action AI baseline | Real Nebius inference |
| `explore` | Let Nemotron choose exploratory tests | Real Nebius inference |
| `doctor` | Check credentials and available models | Authenticated Nebius catalog request; optional inference probe |
| `replay` | Check recorded Grid controls against observations | No model requests |
| `review` | Inspect progress, resets, latency, and evidence | No model requests |
| `report` | Generate a Markdown candidate report | No model requests; can execute local reproduction |
| `triage` | Ask Ultra to interpret candidate evidence | Real inference when candidates exist; `--dry-run` is offline |
| `reproduce` | Re-execute and minimize one issue | No model requests; executes game controls |
| `evaluate` | Compare recordings with reference-fixture labels | No model requests; can execute local reproduction |
| `scoreboard` | Summarize recordings in a Markdown table | No model requests; can execute local reproduction and a scripted baseline |
| `benchmark` | Check the pipeline on controlled fixtures | No model requests, including `--hybrid` |
| `serve` | Expose the reference Grid through the local game protocol | Local server; no model requests |

`review` and `triage` serve different purposes: the former computes facts from
recordings; the latter adds model judgment. Native verification commands can
launch or control a local game, so run them against a test instance.

## Run an AI test

The commands in this section use **your Nebius account** and can incur charges.
Game observations, available controls, and relevant recorded evidence are sent
to Nebius for inference. Configure them for a game you are permitted to test
and share with the provider.

### Configure Nebius

From the repository root, create `.env` if you do not already have one:

```sh
cp .env.example .env
```

If `.env` already exists, edit it instead of copying over it. Add your own key;
the repository defaults are:

```dotenv
NEBIUS_API_KEY=your-key-here
NEBIUS_BASE_URL=https://api.tokenfactory.nebius.com/v1/
NEBIUS_MODEL=nvidia/Nemotron-3_5-Lightning
NEBIUS_RESPONSE_FORMAT=json_schema
NEBIUS_ENABLE_THINKING=true
```

Keep `.env` untracked. Exported environment variables override values in the
file. Use `--env-file /path/to/your.env` after `run`, `explore`, `doctor`, or
`triage` to choose a different file. Configuration accepts Nebius endpoints and
NVIDIA Nemotron model IDs; it is not a generic provider-switching interface.

To check your account's catalog:

```sh
autoplay-qa doctor --list-models
```

This makes an authenticated network request, but no inference request.
Adding `--probe-inference` makes one real inference probe. Model availability
and pricing depend on your account; check them before starting a run.

### Start bounded exploration

```sh
autoplay-qa explore --focus qa \
  --max-actions 6 --max-requests 8 --max-seconds 120 \
  --timeout 30 --max-tokens 8192 \
  --log runs/ai-session.jsonl
```

Nemotron chooses controls from the game's public interface. The default mode
is `lean-v1`; `--focus qa` sets the session question to testing. After the run,
use `review` and `report` from the first walkthrough on `runs/ai-session.jsonl`
when it is an ordinary Grid recording.

| Limit | Default | Meaning |
| --- | --- | --- |
| `--max-actions` | 24 | Maximum model-controlled actions, including resets |
| `--max-requests` | 28 | Maximum session model requests |
| `--max-seconds` | 300 | Session time budget in seconds |
| `--timeout` | 45 | Per-request timeout in seconds |
| `--max-tokens` | 8192 | Requested maximum output tokens per response |

The direct `explore` and `run` commands have **no `--max-usd` option**. These
limits bound activity, not a dollar amount. Estimate costs using current prices
and configure your own provider account limits. Default `lean-v1` can make
bounded repair/retry requests within its request cap; adaptive protocols stop
on provider failures. Failed attempts remain part of the record.

For the original baseline, use `run` with **`--max-steps`**, not `--max-actions`:

```sh
autoplay-qa run --max-steps 6 --max-requests 8 --max-seconds 120 \
  --timeout 30 --max-tokens 8192 --log runs/baseline-session.jsonl
```

Advanced options are described in [adaptive exploration](adaptive.md). In
particular, `explore --preflight` adds a real model probe outside the session
request/time budget, and `explore --hybrid` adds a scripted sweep outside the
model action budget. Neither is needed for the examples above.

### Review candidates with Nemotron Ultra

First inspect an offline packet, using fresh filenames:

```sh
autoplay-qa triage runs/ai-session.jsonl --dry-run \
  --out runs/ai-preview.md --json-out runs/ai-preview.json \
  --log runs/ai-preview.audit.jsonl
```

When you choose to send those candidates for real review:

```sh
autoplay-qa triage runs/ai-session.jsonl \
  --max-requests 4 --max-seconds 180 --timeout 30 \
  --max-tokens 8192 --max-usd 0.50 \
  --out runs/ai-reviewed.md --json-out runs/ai-reviewed.json \
  --log runs/ai-reviewed.audit.jsonl
```

Triage defaults to `nvidia/Nemotron-3-Ultra-550b-a55b`, independently of
`NEBIUS_MODEL`. Its default limits are 8 requests, 300 seconds, a 45-second
request timeout, 8,192 output tokens, and a $1 configured-price estimate cap.
`--max-usd` uses configured prices; it is not a provider balance or billing
guarantee. An alternate `--model` requires both `--input-price-per-million` and
`--output-price-per-million` estimates. Triage does not automatically retry
provider failures. No candidates means no inference requests.

The output includes Markdown, structured JSON, and a separate audit JSONL.
Model verdicts are candidate assessments for developers to check against the
citations. See [candidate review](triage.md) for verdicts, evidence selection,
and `--verify-repro`.

## Reproduce and compare results

If a report lists issue 1, reproduce it with:

```sh
autoplay-qa reproduce runs/ai-session.jsonl --issue 1
```

This re-executes the recorded controls, attempts to minimize the reproduction,
and prints JSON. Use the issue number from that recording's report; the short
first-session example is not expected to have an issue. Exit code 0 means the
issue recurred in this check; 1 means it did not.

For a Markdown comparison of available ordinary recordings:

```sh
autoplay-qa scoreboard runs/first-session.jsonl runs/ai-session.jsonl \
  --no-scripted-baseline --out runs/comparison.md
```

Without `--no-scripted-baseline`, `scoreboard` also runs a seed-0 scripted
baseline. `evaluate RECORDING [RECORDING ...]` compares issues with hidden
labels in controlled reference fixtures. Such labels are useful for checking
the pipeline; they do not establish the truth of issues in an arbitrary game.

## Run offline pipeline checks

```sh
autoplay-qa benchmark --scripted --seeds 0 --record-dir runs/benchmark-first
autoplay-qa benchmark --hybrid --seeds 0
```

Choose exactly one of `--scripted` or `--hybrid`. Both are offline;
`benchmark --hybrid` uses a **fake model**, while `explore --hybrid` uses real
inference. The scripted benchmark exercises authored reference defects and a
clean control, then prints its results. With `--record-dir`, it also writes
case JSONL and Markdown reports. Use a new directory to preserve prior results.

Use the benchmark's own table and reports. Its JSONL is not the ordinary
Grid-session format accepted by the `replay`/`review` walkthrough. Browser
traces, triage audit logs, and downloadable web demo JSON also have their own
consumers; see [evidence formats](evidence-formats.md).

## Connect your own game

A game must expose the [AutoPlay QA protocol](protocol.md) or have a compatible
adapter. You do not upload a package to the CLI. The
[Unity package guide](../unity/AutoPlayQA/README.md) explains how to integrate
the bridge and build a compatible player.

For a local protocol example, open two terminals with the CLI available.
Start the reference server in the first:

```sh
autoplay-qa serve --port 8765 --seed 0
```

Connect and record in the second:

```sh
autoplay-qa play --game tcp://127.0.0.1:8765 \
  --seed 0 --log runs/protocol-session.jsonl
```

Finish play with `q`; stop the server with Ctrl+C. `serve` is a loopback game
protocol server, not an HTTP website. It requires either `--port` or `--stdio`.

For a compatible Unity build, replace the example path with your executable:

```sh
autoplay-qa play --unity-player /path/to/compatible-player \
  --seed 0 --log runs/unity-session.jsonl
```

| Runtime option | Purpose |
| --- | --- |
| `--game tcp://127.0.0.1:PORT` | Connect to an already running local game server |
| `--unity-player PATH` | Launch and manage a compatible headless Unity player |
| `--game-timeout SECONDS` | Protocol request timeout; default 5 |
| `--unity-start-timeout SECONDS` | Player startup timeout; default 10 |

Choose either `--game` or `--unity-player`. These options are available on
`play`, `run`, `explore`, `report`, `reproduce`, `evaluate`, `scoreboard`, and
`benchmark`. Triage accepts them with `--verify-repro` for local verification
before review. Recorded Unity paths may be reused automatically by reporting
and reproduction; override them with your own build path on another machine.

CLI `replay` and `review` do **not** take these runtime options; their direct
workflow is for ordinary current Grid recordings. Use the adapter-specific
tools in [runtime integrations](integrations.md) for other replay workflows.
Mars/browser setup is documented there separately.

## Output files and automation

| Output | Behavior |
| --- | --- |
| `play` / `run` / `explore` JSONL | New file only; auto-generated unique path if `--log` is omitted |
| `report` / `scoreboard` | Markdown to stdout, or `--out FILE.md` (can overwrite) |
| `triage` Markdown / JSON | Defaults beside the input as `.triage.md` / `.triage.json` (can overwrite) |
| `triage` audit log | New file only; defaults to a unique path under `runs/triage/` |
| `benchmark --record-dir` | Deterministic case filenames; a reused directory can overwrite results |

Keep the input, reports, and audit log at distinct paths. Inspect recordings
before sharing them: native recordings can contain local paths and game data.
Use the [test-data guide](contributing-data.md) to contribute a small reviewed
bundle with permissions, expected behavior, and reproducible evidence.

Exit status depends on the command. A successful `explore` can end at a normal
action/request/time limit; it does not mean a bug was found. `run` returns 3 on
a normal run limit. `triage` returns 0 for completion, dry-run, or no candidates,
and 3 for other recorded non-success outcomes. `benchmark` returns 1 if a case
fails; `reproduce` returns 1 if recurrence fails. Configuration and argument
errors generally return 2. Interrupted `run`, `explore`, and `triage` sessions
return 130; an aborted `explore --preflight` returns 4. Inspect the printed
status and artifacts alongside the process exit code in CI.

## Troubleshooting

| Symptom | What to check |
| --- | --- |
| `autoplay-qa: command not found` | Activate `.venv`, or run `python3 -m autoplay_qa` from the repository root. |
| `No module named autoplay_qa` | Use the environment where you installed the project, or change to the checkout root. |
| Installation fails on the Python version | Run `python3 --version`; select Python 3.11+. |
| Log already exists | Choose a fresh `--log` filename; preserve the earlier evidence. |
| `unrecognized arguments` | Put options after the subcommand and inspect its `--help`. `run` uses `--max-steps`; `explore` uses `--max-actions`. |
| Key missing or authentication fails | Check the selected `.env`, working directory, and any overriding exported variables. Never paste your key into an issue. |
| Model unavailable | Check `doctor --list-models` with your own credentials and choose a supported Nemotron ID. |
| Timeout or request limit | Inspect the saved terminal status and request record before choosing another run; an unanswered request may still incur a charge. |
| `no_candidates` | The recording contains no reportable candidates; no Ultra request is needed. |
| Replay fails on a downloaded file | Check the [format guide](evidence-formats.md); web JSON, benchmark logs, browser traces, and audit logs are not ordinary Grid sessions. |
| Game connection refused | Start the compatible server, check its local port, or supply the correct Unity executable. |

For dashboard setup, use the [user guide](user-guide.md). For an overview of
the evidence workflow and available extensions, visit the
[documentation index](README.md).
