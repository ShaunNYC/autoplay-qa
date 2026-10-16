# AutoPlay QA web viewer

The web application presents saved game observations, candidate reports, and
downloadable evidence. It uses React, Vinext, Cloudflare Workers, and D1. The
repository includes sanitized demo files under `public/demo/`.

Playback draws recorded state and geometry. It is **not native gameplay video**,
does not execute game controls, and makes no model requests. Demo JSON is a
viewer format, **not Python session JSONL**; it cannot be passed to the ordinary
`autoplay_qa replay` command. See [evidence formats](../docs/evidence-formats.md).

## Local setup

Use Node.js **22.13.0 or newer** and npm. From the repository root:

```sh
cd web
npm run install:ci
npm run build
```

`install:ci` runs a locked installation. A clean clone uses the portable
execution profile. The build indexes the checked-in demo JSON automatically;
it does not need private source recordings or the maintainer export utility.

The catalog uses the `DB` D1 binding even for saved sessions. The local build
requires binding metadata in `.openai/hosting.json` with `"d1": "DB"` and
`"r2": null`; it does not need another deployment's project identity. After
building, initialize a fresh local database:

```sh
node --import ./scripts/sites-env.mjs ./node_modules/wrangler/bin/wrangler.js \
  d1 execute DB --local --config dist/server/wrangler.json \
  --persist-to .wrangler/state --file drizzle/0000_workable_namora.sql
npm run dev -- --hostname 127.0.0.1
```

Apply migrations once, in order. Use the server's printed URL, normally port
5173. To preview the built Worker, stop the development server and run
`npm start`. These commands stay local and do not publish a site.

## Optional task service

For a step-by-step owner and runner walkthrough, see
[Start a New test](../docs/user-guide.md#3-start-a-new-test-from-your-dashboard).

The code also contains owner job controls and runner endpoints. Reading saved
evidence does not require a provider key or live runner. Starting a real job
requires your own authentication, server configuration, compatible game runtime,
local Python runner, provider account, and explicit execution/data permission.

Owner writes use `OWNER_EMAIL` identity or `ADMIN_TOKEN`; runner endpoints use
a separate `RUNNER_TOKEN`. Local values belong in ignored `.dev.vars`. The
provider key belongs with the local runner, not the web application.

The portable development profile has a loopback-only mock sign-in at
`/signin-with-chatgpt?return_to=/`. Production identity in the original hosting
integration is supplied by Sites. Deployments elsewhere must supply their own
compatible identity and authorization integration. A build or sign-in does not
automatically start inference, create a production deployment, or grant access
to a project maintainer's runner.

## Implementation and checks

| Area | Location |
| --- | --- |
| Page and metadata | `app/page.tsx`, `app/layout.tsx` |
| Evidence interface | `components/dashboard.tsx` |
| Observed geometry | `components/scene-map.tsx` |
| Visual styles | `app/globals.css` |
| Public/owner and runner endpoints | `app/api/runs/`, `app/api/runner/` |
| Validation, access, types | `lib/api-core.ts`, `lib/server.ts`, `lib/types.ts` |
| Demo files and generated index | `public/demo/`, `lib/demo-manifest.json` |
| Database | `db/schema.ts`, `drizzle/` |

After dependency installation, run from `web/`:

```sh
node --experimental-strip-types --test tests/*.test.mjs
npx tsc --noEmit
npm run build
```

Use fake progress and fake HTTP for implementation checks. Preserve third-party
notices in `build/sites-vite-plugin.LICENSE` and
`vendor/shadcn-tailwind-4.13.0.LICENSE.md`. Keep local dependencies, databases,
runtime profiles, logs, credentials, and build output out of source commits.
