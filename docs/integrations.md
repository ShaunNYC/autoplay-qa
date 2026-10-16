# Runtime integrations

## Built-in Grid

The terminal Grid game ships with the Python source. `play` and the scripted
benchmark work without an API key. The reference game can also expose the local
protocol:

```sh
python3 -m autoplay_qa serve --port 8765 --seed 0
```

In another terminal:

```sh
python3 -m autoplay_qa play --game tcp://127.0.0.1:8765 --seed 0
```

See the [protocol specification](protocol.md) for framing, observation fields,
acknowledgments, and error behavior. The protocol is intended for loopback use.

## Unity

Install `unity/AutoPlayQA` into your own Unity project using Package Manager's
local package option. Implement `IAutoPlayQAGame`, or start with the bundled
sample adapters. The package contains a small grid example and a PhysicsKeyDoor
sample built from primitives.

The package declares Unity 2021.3 as its minimum package version. Native physics
and build compatibility depend on the Unity version and target you use; the
sample provides a macOS build entry point. The repository does not include a
complete Unity project or prebuilt player. Follow the
[package guide](../unity/AutoPlayQA/README.md) for setup.

Once a compatible server is running, the Python host can connect using `--game`.
Tools that launch a standalone player take `--unity-player`; provide your own
executable path rather than a path from another machine.

## Mars interpreter

Mars is a separate game project. Its source and assets are not included here.
The interpreter adapter expects the compatible TypeScript engine layout and
exports used by `tools/mars_bridge.mjs`, including `src/mars/engine.ts` and the
original state/mission definitions. Supply a project you have permission to use
and its build dependencies. A different game requires adapting the bridge.

The interpreter adapter calls the actual engine through a Node stdio bridge.
It exposes a fixed native control palette and records public execution feedback.
It does not provide arbitrary editor authoring or browser rendering.

## Mars browser editor

The separate browser adapter exposes visible click, input, select, and wait
operations. It requires a compatible Mars application, Node, Playwright, and a
Chromium-based executable. The caller prepares an isolated build and supplies
explicit browser/module paths. The public release does not download those
dependencies or include a private game copy.

Browser observations, source/build identity, and replay configuration belong
to the evidence. Matching a controlled browser trace is scoped to that
environment. Generic semantic interfaces are in `autoplay_qa/semantic_testing.py`
and `semantic_analysis.py`. The public release contains these core interfaces
and local utilities, rather than private study orchestration or archives. See
[semantic testing](semantic-testing.md).

## Add another runtime

Expose public state and ordinary controls through the smallest useful adapter.
Keep hidden rules, routes, private labels, and credentials outside observations.
Record completed controls and feedback, and represent unknown execution
explicitly rather than silently retrying a mutation. Version the game identity
when observation or action semantics change.
