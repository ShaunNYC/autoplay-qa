# Evidence formats and entry points

Choose the consumer for the artifact you have. A shared `.json` or `.jsonl`
extension does not make recordings interchangeable.

| Artifact | Producer | How to inspect it |
| --- | --- | --- |
| Ordinary Grid session JSONL | `play`, `run`, or supported `explore` modes | `replay`, `review`, `report`; live model use is not needed for inspection |
| Native integration session | A runtime-specific adapter/runner | The compatible adapter/build and its replay/reproduction integration |
| Scripted benchmark outputs | `benchmark --scripted --record-dir DIR` | The benchmark's result table and generated Markdown reports; use the benchmark entry point for another execution |
| Hybrid stub benchmark outputs | `benchmark --hybrid` | Its benchmark summary and artifacts, labeled as synthetic model integration |
| Web demo JSON/Markdown | A sanitized dashboard export | The web viewer or a JSON/Markdown reader |
| Semantic browser trace | Browser/semantic tooling | The semantic adapter, analysis, and regression tools with the matching source/build |
| Experience SQLite database | `ExperienceStore` | Its scoped retrieval/snapshot APIs, with source recordings for detailed evidence |

## Ordinary Grid sessions

For a session you recorded with the standard `play` command:

```sh
python3 -m autoplay_qa replay runs/my-play.jsonl
python3 -m autoplay_qa review runs/my-play.jsonl --evidence
python3 -m autoplay_qa report runs/my-play.jsonl
```

The quickstart is specifically for ordinary built-in Grid sessions. Do not
route benchmark artifacts, web JSON, or semantic traces through that command
based only on their filename extension. Native game recordings need the
corresponding adapter and runtime; the public checkout does not contain every
original game build.

## Scripted benchmark workflow

The benchmark owns the controlled fixture configuration and its result table.
Its case JSONL includes benchmark metadata and is not covered by the ordinary
session replay quickstart. Read the generated Markdown reports and use the
benchmark's dedicated entry point to execute the cases. A controlled fixture
is an example of the pipeline, not an unknown autonomous AI finding.

## Web playback

Files under `web/public/demo/` contain selected sanitized observations, report
data, and presentation metadata. The viewer draws those observations and moves
through them when a visitor presses playback controls. This is not a native
Unity video stream and does not rerun the game or call a model.

Downloading viewer JSON does not yield a complete Python session JSONL. It may
include references to evidence or reproduction checks, but the visualization
alone does not independently reproduce the native game.

## Claims and verification

An execution acknowledgment establishes that a control completed. Replay can
check that a compatible runtime returns the recorded observable result.
Reproduction can establish recurrence. A model verdict adds interpretation.
An independently confirmed defect additionally needs evidence about intended
behavior and the issue's cause or effect.

Preserve these distinctions, source labels, missing evidence, and original
claims when preparing a report. Do not turn a synthetic test trace or model
summary into a native observation.
