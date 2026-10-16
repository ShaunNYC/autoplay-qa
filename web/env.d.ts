declare namespace Cloudflare {
  interface Env { DB: D1Database; ASSETS: Fetcher; CONNECTORS?: import("./lib/connector-contract.mjs").ConnectorBinding; ADMIN_TOKEN?: string; RUNNER_TOKEN?: string; OWNER_EMAIL?: string; JOB_BUDGET_LIMIT_USD?: string }
}
interface ImportMeta { env: { DEV: boolean } }
