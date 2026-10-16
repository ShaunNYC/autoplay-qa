# Contributing to AutoPlay QA

We welcome work that connects new games, improves evidence inspection, or helps
agents choose more useful tests. The [research roadmap](docs/research-roadmap.md)
describes the existing extension points and longer-term vision.

## Propose a concrete change

Explain the developer problem, the resulting behavior, and how you checked it.
Include a focused offline example when practical. For an adapter, document the
runtime prerequisites, public observation shape, controls, and execution
acknowledgments. For a candidate issue, provide the executed sequence and enough
public evidence for another person to investigate it.

Preserve the `GameAdapter.observe()` / `act()` boundary and existing recording
contracts. Add incompatible protocols as opt-in versions. Keep scripted actions,
stub responses, real model decisions, and human findings clearly labeled.

## Verify locally

Use Python 3.11+ from the repository root:

```sh
python3 -m unittest discover -s tests
python3 -m autoplay_qa benchmark --scripted --seeds 0
```

For web changes, follow [web setup](web/README.md), then run in `web/`:

```sh
node --experimental-strip-types --test tests/*.test.mjs
npx tsc --noEmit
npm run build
```

Use fake HTTP for implementation tests. Exercise the relevant recording format
and native adapter when changing execution or replay. State which checks you
ran and which runtime prerequisites were unavailable. See
[evidence formats](docs/evidence-formats.md): web JSON, ordinary session JSONL,
benchmark outputs, and semantic traces have different consumers.

## Research contributions

Define the question and success criteria before collecting outcomes. Keep
comparison conditions, resource limits, and baseline policies explicit. Preserve
predictions before interventions, original claims, and contradictory evidence.
Use independently supported defect judgments and fresh runtime reproduction
when making discovery claims.

Memory storage, retrieval, or a model acknowledgment does not by itself show a
learning benefit. Compare outcomes under matched conditions. Cross-game
transfer and self-evolving agents remain research goals.

## Model access and data

Use your own credentials and authorize your own bounded model runs and data
transmission. A repository checkout does not grant access to another person's
credits, accounts, game source, or deployment. Public example commands are not
permission for an automated contributor to dispatch inference.

Do not commit keys, raw provider bodies containing private data, model thinking,
private game source, local databases, personal paths, or runtime secrets. Inspect
artifacts before sharing them. Keep evaluation-only labels outside actor and
reviewer inputs, and obtain permission to publish contributed game assets.

## License

Contribute material you have the right to share under the repository's MIT
license. Preserve third-party license notices and clearly identify separately
licensed dependencies or assets.
