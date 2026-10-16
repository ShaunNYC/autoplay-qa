# Tool index

Run tools from the repository root and inspect each command's `--help` before
use. The core quickstart is in [getting started](../docs/getting-started.md).

| Tool group | Purpose and prerequisites |
| --- | --- |
| `unity_smoke.py` | Offline protocol check against your running Unity server |
| `mars_bridge.mjs`, `mars_conformance.mjs` | Native Mars interpreter integration; requires your compatible original game source and dependencies |
| `browser_bridge.mjs`, `prepare_mars_browser.mjs` | Visible-browser integration and isolated Mars preparation; requires Node, compatible source, and browser dependencies |
| `run_semantic_regression.py` | Local execution of a reviewed case against a prepared compatible browser build |
| `run_dashboard_runner.py` | Optional authenticated runner; requires your service, runtime, credentials, and explicit model-use authorization |
| `dashboard_api_smoke.py` | Loopback HTTP checks with fake progress against a configured local web service |
| `semantic_study.py` | Pure JSON/hash helpers and an editor-control scope wrapper used by the local regression tool |

The public repository includes research interfaces in `autoplay_qa/` so
contributors can inspect and adapt the methods. Private campaign orchestration,
original operator archives, and deployment identities are not included.
Experimental modules retain some specific protocol constants for compatibility;
they are not a generic ready-to-run model benchmark or active authorization.

Use fake/stub HTTP for development. Model commands require your own explicit
limits, provider account, and data permission. Native tools need the actual
runtime they describe; synthetic responses are not a replacement for native
reproduction evidence.
