# Your first AutoPlay QA session

Choose the route that matches what you want to do:

| Your goal | Start here | What you need |
| --- | --- | --- |
| See how it works | [Explore saved evidence](#1-explore-saved-evidence) | A browser |
| Try recording and replay | [Run the local walkthrough](#2-record-and-replay-locally) | Python 3.11+ |
| Start an AI test from the dashboard | [Set up New test](#3-start-a-new-test-from-your-dashboard) | Your local service, runner, and Nebius account |
| Test your own game | [Connect your game](#4-connect-your-own-game) | A compatible game adapter |

## 1. Explore saved evidence

Open the [public demo](https://autoplay-qa-lab.foomusta.chatgpt.site).
Select a saved session, such as **Unity · Floor support**, then:

1. Use the playback controls and timeline to follow the recorded actions.
2. Inspect the before/after state beside a selected action.
3. Select a finding and follow the review's step references to its evidence.
4. Open the full report or download the JSON/Markdown evidence.

You need no account or API key for this route. Playback displays saved
observations; it does not start a new game or model request. The hosted
interface may differ slightly from the current repository's local interface.

## 2. Record and replay locally

Download the repository ZIP or clone the repository, open a terminal, and
change into the folder containing `README.md` and `autoplay_qa/`.
Check Python, then record your first session:

```sh
python3 --version
python3 -m autoplay_qa play --seed 0 --log runs/first-session.jsonl
```

Type `e` and press Enter, then `i` and Enter to try an interaction. Use
`n`, `s`, `e`, `w` to move, `i` to interact, and `.` to wait. Finish with `q`.
The terminal should return to your shell and save `runs/first-session.jsonl`.
Use a new filename for a later session.

```sh
python3 -m autoplay_qa replay runs/first-session.jsonl
python3 -m autoplay_qa review runs/first-session.jsonl --evidence
python3 -m autoplay_qa report runs/first-session.jsonl --out runs/first-session.md
```

You now have a saved action record, replay/review summaries, and a readable
Markdown report. A session can have no suspected issues and still be useful.
These commands need no API key and make no inference requests.

For scripted examples and optional direct AI exploration, continue with
[Getting started](getting-started.md).

## 3. Start a New test from your dashboard

**New test runs a game already configured on your local runner.** Game files
stay on your machine; the button is not a game-package upload form. The web
form selects the bundled Grid or Unity PhysicsKeyDoor environment. You need
owner access to your own dashboard and an online runner.

The following is a setup recipe for a new contributor's local installation.
Use your own provider account and budget. Starting the runner lets it execute
queued jobs, and starting a test can incur inference charges.

### Prepare the local service

Follow [web setup](../web/README.md) to install dependencies, build the app,
and initialize its local database. In the ignored `web/.dev.vars` file, set:

```text
OWNER_EMAIL=your-owner-email
RUNNER_TOKEN=your-own-random-runner-secret
```

Replace the example values. Start the development web server with
`npm run dev -- --hostname 127.0.0.1` from `web/`. Use its printed URL;
the example below assumes port 5173. Open **Owner sign in**. The portable
development profile provides a local mock sign-in; a hosted installation needs
its own compatible authentication and actual owner identity.

### Connect your runner

From the repository root, configure your own `NEBIUS_API_KEY` in `.env` as
described in [Nebius setup](getting-started.md#optional-nebius-configuration).
Create the ignored `.env.dashboard` file:

```text
DASHBOARD_SITE_URL=http://127.0.0.1:5173
DASHBOARD_RUNNER_TOKEN=the-same-runner-secret-as-the-web-service
```

Use the same runner secret on both sides. The Nebius key stays in the local
runner's `.env`; it does not belong in the web app. For a new Grid-only local
installation, start the runner from the repository root:

```sh
python3 tools/run_dashboard_runner.py --serve --allow-local-http --total-budget-usd 1
```

The terminal should print **Mac runner online**, and the dashboard should show
the runner as connected. The $1 is a shared configured-price estimate for this
installation, not a provider billing guarantee. Keep the same `runs/dashboard`
directory and allowance when restarting. Do not delete ledgers to reset usage.
For a hosted service, use its HTTPS URL and omit `--allow-local-http`.

### Create and inspect the test

1. Click **New test** on your own dashboard.
2. Choose **Grid KeyDoor**, **Baseline** (the normal environment), and seed **0**.
3. Choose a small per-test budget within your runner's remaining allowance.
4. Start the test and follow the session as results arrive.
5. Inspect any findings, their cited steps, and the downloadable report.

Use the session's **Stop** control to request cancellation. An in-flight
request may finish before the runner stops. Keep the local evidence and usage
records if a test is interrupted.

For the Unity sample, follow the [Unity package guide](../unity/AutoPlayQA/README.md)
and build the PhysicsKeyDoor player. Restart the same runner with
`--unity-player '/absolute/path/to/your/player-executable'` added to its command.
Unity becomes selectable when the configured executable is available.

### If the button is disabled

| What you see | What to check |
| --- | --- |
| Public / read only | Sign in as the owner of your own installation. Public demo visitors cannot start tests. |
| Runner offline | Keep the runner terminal open; check the service URL and matching runner secret. |
| Unity unavailable | Supply the actual executable path of your compatible sample build. |
| A test is already queued/running | Wait for it to finish, or use its Stop control. |
| Budget or allowance error | Inspect existing usage and retained reservations before planning another run. |

## 4. Connect your own game

Start with a local connection before adding dashboard controls:

1. For Unity, install the package, implement `IAutoPlayQAGame`, and attach the
   server as shown in the [Unity guide](../unity/AutoPlayQA/README.md).
2. Expose visible game state and ordinary controls. Enter Play Mode with the
   local server listening at its configured address.
3. Try a human-controlled connection without a model:

```sh
python3 -m autoplay_qa play --game tcp://127.0.0.1:8765 --seed 0 --log runs/my-game.jsonl
```

Use the controls advertised by your adapter. Check that an action changes the
game and that the returned feedback matches what happened. For other engines,
follow [runtime integrations](integrations.md) and the [protocol](protocol.md).

Adding a custom game to **New test** also requires extending the runner's
allowlisted game/profile configuration and the web form; selecting an arbitrary
executable is not a complete integration. Start with the CLI connection above
while developing your adapter.

## Share a useful case

Once you have a recording worth sharing, follow the
[test-data contribution guide](contributing-data.md). Contribute the permitted
evidence and reproduction instructions; you do not need to upload an entire
game. Check [evidence formats](evidence-formats.md) when choosing a reader:
downloaded web JSON and native session JSONL use different tools.
