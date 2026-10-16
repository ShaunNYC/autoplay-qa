# AutoPlay QA

**Turn AI playtesting into evidence developers can inspect and reproduce.**

AutoPlay QA is an open source workbench for connecting AI testers to game
runtimes. NVIDIA Nemotron selects actions and reviews candidate evidence through
Nebius Token Factory. The host records what actually executes, preserves the
resulting feedback, and provides tools for developer review and reproduction.

Our vision is a self-evolving testing partner: an agent that uses experience to
ask better questions and create useful regression cases. This is a community
research goal. The framework contributes the runtime interfaces, evidence
contracts, and experimental tools on which to build it.

[Getting started](docs/getting-started.md) · [Capabilities](docs/capabilities.md)
· [Research roadmap](docs/research-roadmap.md) · [Contributing](CONTRIBUTING.md)
· [MIT license](LICENSE)

[Public evidence demo](https://autoplay-qa-lab.foomusta.chatgpt.site) ·
Created by [Shaun Wang](https://github.com/ShaunNYC)

## More than an AI playthrough

| Developer need | What AutoPlay QA adds |
| --- | --- |
| Know what happened | Public observations, accepted controls, execution acknowledgments, and before/after state |
| Inspect a candidate | Cited transitions, tentative model assessments, and reproduction evidence |
| Repeat a behavior | Saved controls and runtime-specific replay/reproduction tools |
| Combine testing strategies | Scripted coverage and model exploration with distinct source labels |
| Connect another game | A small observe/act boundary, local protocol, Unity package, and browser adapter |
| Bound the work | Action, request, and time limits; estimated cost accounting in supported runners and review workflows |

Nemotron contributes contextual action selection and evidence review. Software
execution records and native reproduction remain separate from model judgments.
A recurring behavior is evidence to inspect; whether it violates the intended
design still needs independent review.

## Start without an API key

Use Python **3.11 or newer** from the repository root. The Python core uses the
standard library and does not require a package installation.

```sh
python3 -m autoplay_qa --help
python3 -m autoplay_qa play --seed 0 --log runs/my-play.jsonl
```

Play with `n`, `s`, `e`, `w`, `i` (interact), and `.` (wait); press `q` to finish.
Inspect that ordinary Grid session:

```sh
python3 -m autoplay_qa replay runs/my-play.jsonl
python3 -m autoplay_qa review runs/my-play.jsonl --evidence
```

For the independent scripted benchmark entry point:

```sh
python3 -m autoplay_qa benchmark --scripted --seeds 0 --record-dir runs/benchmark
```

Read its printed results and generated Markdown reports. Benchmark artifacts
have a separate workflow; do not pass them to the ordinary session `replay`
command. These examples make no model requests. See
[evidence formats](docs/evidence-formats.md) before choosing an inspection tool.

## What is included

- **Game testing core:** a terminal Grid reference game, bounded exploration,
  scripted coverage, candidate reports, and reproduction tools.
- **Nemotron integration:** Lightning for default exploration and Ultra for
  structured, evidence-citing candidate review.
- **Runtime adapters:** a local protocol and Unity package, plus Mars interpreter
  and visible-browser adapters. Mars requires separately supplied game source.
- **Web evidence viewer:** saved state playback, reports, and JSON/Markdown
  downloads. Bundled demo JSON is a sanitized viewer format, not native video
  or the Python session JSONL format.
- **Research modules:** bounded context, observed experience, experiment agendas,
  persistent investigations, focused facts, prospective hypotheses, controlled
  comparisons, and regression-export tooling.

The research modules provide mechanisms to study. Reliable learning gains,
cross-game learned-mechanics transfer, and autonomous discovery of previously
unknown defects are not established capabilities. See
[capability status](docs/capabilities.md) and the
[community roadmap](docs/research-roadmap.md).

## Optional model use

To run your own model session, configure your own Nebius credentials and choose
explicit limits. Model commands send public game observations and selected
evidence to the provider and may incur charges. The
[setup guide](docs/getting-started.md#optional-nebius-configuration) separates
offline commands from model calls and describes their limits.

The default model identifiers are `nvidia/Nemotron-3_5-Lightning` for exploration
and `nvidia/Nemotron-3-Ultra-550b-a55b` for review. Availability is determined by
your provider account. No provider credential is needed to inspect the bundled
web evidence.

## Develop and contribute

```sh
python3 -m unittest discover -s tests
```

Implementation tests use fake or stubbed model responses. Web development needs
Node.js 22.13+; Unity and browser integrations have additional prerequisites.
Checks that require optional runtimes or unavailable historical fixtures report
explicit skips in the public checkout.
Use the [documentation index](docs/README.md) to choose the appropriate guide.

Contributions are welcome in adapters, evidence inspection, agent experiments,
and evaluation. Keep model claims, recorded observations, and independently
confirmed defects distinguishable as the project grows.
