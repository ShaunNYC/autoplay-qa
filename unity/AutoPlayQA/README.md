# AutoPlay QA Unity bridge

This UPM package exposes a game through the `autoplay-qa/1` local JSON-lines
protocol. It has no third-party runtime dependencies. The repository supplies
the package and samples; create your own Unity project and build.

## Install

Use Package Manager's **Add package from disk** and select this directory's
`package.json`. Alternatively, add a local-file dependency in the consuming
project's `Packages/manifest.json`, using the relative path from that project:

```json
"com.autoplayqa.bridge": "file:../../AutoPlay-QA/unity/AutoPlayQA"
```

The manifest declares Unity 2021.3 as its minimum package version. Check the
samples against your Unity version and build target.

1. Implement `IAutoPlayQAGame` on a `MonoBehaviour`, or use
   `SampleGridGameAdapter` for an initial check.
2. Add `AutoPlayQAServer` to a GameObject and assign the adapter.
3. Set a game identity/version and only the capabilities it supplies.
4. Enter Play Mode. The default endpoint is `tcp://127.0.0.1:8765`.

Enable **Run In Background** for unattended use. The server binds IPv4 loopback;
the protocol is not an authenticated public network service.

## Adapter interface

```csharp
string[] Controls { get; }
void Reset(int seed);
AutoPlayQAObservation Observe();
AutoPlayQAObservation Act(string control);
```

These members run on Unity's main thread. Socket work happens in a background
thread and dispatches through `Update`. Observations expose public player,
inventory, entity, feedback, status, and available-control fields. Optional grid
and legend data require the corresponding capability.

Give the actor visible feedback and ordinary controls. Do not put hidden
evaluation labels, injected-defect identity, routes, or developer rules in the
protocol. The full contract is in [protocol.md](../../docs/protocol.md).

## Samples

`SampleGridGameAdapter` demonstrates seeded reset, a small control set, inventory,
feedback, and terminal state. `Runtime/Samples/PhysicsKeyDoor` builds a physics
scene from primitives. Each protocol action advances a bounded number of fixed
simulation steps and exposes position, velocity, grounded state, inventory, and
visible geometry.

The editor assembly provides:

- `AutoPlayQA.Editor.PhysicsKeyDoorBuild.Validate`: compile/setup validation.
- `AutoPlayQA.Editor.PhysicsKeyDoorBuild.BuildMacOS`: generate the bootstrap scene,
  put it first in build settings, and build `Builds/AutoPlayQA-Physics.app` in the
  consuming project.

The build method changes the consuming project's generated scene/build settings;
use a project intended for this sample. Other targets need an appropriate build
entry point. No prebuilt player is included.

The player accepts `-autoplayqa-port` and `-autoplayqa-seed`. Optional process-only
controlled fixtures use `-autoplayqa-defect` with `floor-gap`,
`door-ghost-collider`, `key-not-consumed`, `respawn-leak`, or `wall-tunnel`. Omit
that option for the normal sample. Fixture identity is excluded from protocol
messages and is not an AI discovery.

## Connect from Python

With the Unity server running, use the repository-root CLI:

```sh
python3 -m autoplay_qa play --game tcp://127.0.0.1:8765 --seed 0
python3 tools/unity_smoke.py --host 127.0.0.1 --port 8765 --seed 0 --actions wait,jump
```

These commands make no inference calls. Use your own build and matching adapter
for native reproduction; physics observations are scoped to the tested runtime
and are not a promise of bitwise equality across platforms.
