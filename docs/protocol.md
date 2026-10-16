# AutoPlay QA remote game protocol

Protocol identifier: `autoplay-qa/1`

This protocol lets AutoPlay QA drive a game without depending on that game's
runtime. It preserves the host-side `GameAdapter.observe()` / `act()` boundary:
the remote adapter turns each acknowledged wire response into the same generic
observation shape used by the rest of the host.

Version 1 is a request/response JSON-lines protocol. The required transport is
localhost TCP; stdio uses the same messages and framing.

## Transport and framing

- Each message is exactly one UTF-8 JSON object followed by `\n`. Receivers may
  also accept `\r\n`. Embedded newlines must be JSON-escaped.
- A line is limited to 1 MiB, measured as received bytes before the newline. A
  receiver closes the connection if an overlong line cannot be answered safely.
- TCP servers bind to a loopback address by default. Version 1 has no
  authentication, authorization, or encryption and must not be exposed on an
  untrusted interface.
- For stdio, the client writes requests to the server process's stdin and reads
  responses from stdout. The server reserves stdout for protocol messages and
  sends diagnostics to stderr.
- Requests are processed serially. A version-1 client waits for the response to
  one request before sending the next.
- Every request has a nonempty string `id`, unique for that connection. Every
  ordinary response echoes it. An error that occurs before an ID can be parsed
  uses `"id":null`.
- JSON numbers must be finite. Implementations must not emit `NaN`, infinity,
  comments, a byte-order mark, or trailing non-whitespace after the object.

The client must send `hello` first. A connection then accepts zero or more
`reset`, `observe`, and `act` requests, followed by at most one `close`. A
successful `close` response is the final message on the connection.

## Common envelopes

A request contains an ID and a message type:

```json
{"id":"1","type":"observe"}
```

A successful response echoes both and sets `ok` to true:

```json
{"id":"1","type":"observe","ok":true,"observation":{}}
```

An error uses the common error envelope instead of a command-specific payload:

```json
{
  "id":"1",
  "type":"error",
  "ok":false,
  "error":{
    "code":"illegal_control",
    "message":"Control is not advertised by this game.",
    "retryable":false
  }
}
```

`message` is safe diagnostic text, not a stack trace. It must not contain
credentials, filesystem secrets, an injected-defect name, or other hidden game
state.

## Commands

### `hello`

The client declares the one protocol version it implements:

```json
{"id":"hello-1","type":"hello","protocol":"autoplay-qa/1"}
```

The server returns its game identity, complete control catalog, and boolean
capabilities:

```json
{
  "id":"hello-1",
  "type":"hello",
  "ok":true,
  "protocol":"autoplay-qa/1",
  "game":{"id":"key-gate","version":"key-gate-v1"},
  "controls":["north","south","east","west","interact","wait"],
  "capabilities":{
    "grid":true,
    "legend":true,
    "entities":true,
    "reset":true,
    "seed":true,
    "defect_injection":true,
    "continuous":false,
    "solids_visible":false
  }
}
```

`game.id` identifies the game family. `game.version` changes when observation
or action semantics become replay-incompatible. Control strings are opaque,
case-sensitive identifiers and are unique within the list.

Every present capability value is boolean. Omission means false/not advertised;
this allows implementations to preserve older hello encodings when a new
capability is unsupported. A server may add game-specific boolean capabilities,
and a client ignores names it does not know. The two continuous-world
capabilities are additive in `autoplay-qa/1`, so older clients remain
compatible:

- `continuous:true` means positions and motion are continuous rather than grid
  cells or turns. Each successful `act` advances a bounded game-defined time
  slice and returns the state after that slice; the game does not free-run
  between requests merely because this capability is true.
- `solids_visible:true` means collision-relevant visible level geometry is
  represented by `entities` with positions and axis-aligned sizes. It is the
  continuous-world analogue of a visible grid and legend, not a computed
  passability map. It describes what the player can see, not a guarantee that
  the corresponding physics colliders behave correctly.

`defect_injection:true` means only that the server can be configured out of
band for controlled tests. It does not say whether a defect is active or which
one.

### `reset`

```json
{"id":"reset-1","type":"reset","seed":5}
```

For a seed-capable server, `seed` is a required JSON integer. The server starts
a fresh episode using that seed and returns the first observation:

```json
{"id":"reset-1","type":"reset","ok":true,"observation":{"status":"playing"}}
```

The command requires `capabilities.reset`. Supplying `seed` also requires
`capabilities.seed`; a reset-capable server without seed selection resets its
current configuration and rejects a supplied seed rather than silently
ignoring it. Reset success means the returned observation is already current;
an additional `observe` is not required.

### `observe`

```json
{"id":"observe-1","type":"observe"}
```

The response contains the current observation without advancing the game:

```json
{"id":"observe-1","type":"observe","ok":true,"observation":{"status":"playing"}}
```

Repeated observations with no intervening mutation must describe the same game
state. Transient presentation events may be repeated; they must not be
fabricated or consumed merely by observing.

### `act`

```json
{"id":"act-1","type":"act","control":"north"}
```

`control` must exactly match a value advertised by `hello`. A successful
response acknowledges that the control completed and includes the resulting
observation:

```json
{"id":"act-1","type":"act","ok":true,"observation":{"status":"playing"}}
```

Success is an execution acknowledgement, not proof that the control had its
expected effect. Blocked movement and no-op interactions are therefore normal
successful responses whose events explain the visible outcome.

### `close`

```json
{"id":"close-1","type":"close"}
```

```json
{"id":"close-1","type":"close","ok":true}
```

`close` releases connection-local resources but does not imply a game reset.
The server sends the response before closing the stream.

## Observation schema

An observation is intentionally game-agnostic. Its core fields are:

```json
{
  "status":"playing",
  "player":{"position":[1,1],"inventory":[]},
  "entities":[
    {"id":"key","kind":"key","position":[3,1],"state":{"collected":false}},
    {"id":"gate","kind":"gate","position":[4,3],"state":{"open":false}}
  ],
  "events":["Game started."],
  "controls":["north","south","east","west","interact","wait"],
  "grid":["#####","#@K.#","#####"],
  "legend":{"#":"wall","@":"player","K":"key",".":"floor"}
}
```

- `status` is a nonempty string such as `playing`, `won`, `lost`, or a
  game-specific terminal status.
- `player.position` is an array of finite coordinates. A 2D grid game uses
  `[x,y]`; other games may use three coordinates. `player.inventory` is an
  array of stable item identifiers. Games may add visible player state such as
  a finite `velocity` coordinate array or boolean `grounded` value.
- `entities` is an array. Each entity has a stable string `id`, a string `kind`,
  and a JSON object `state`. `position` is a coordinate array when the entity
  has a visible location and may be omitted otherwise. An entity may include a
  finite, non-negative `size` array with the same dimensionality as `position`;
  it gives the full axis-aligned dimensions centered at `position`. The same
  visible entity keeps its ID across observations in an episode.
- `events` is an array of player-visible feedback strings caused by the most
  recent lifecycle operation or control. An operation with no feedback uses an
  empty array.
- `controls` is the subset of the `hello` control catalog that is legal in the
  current observation. It is an empty array in a terminal state. A client must
  validate an action against this current list as well as the catalog.
- `grid` and `legend` are present only when the corresponding capabilities are
  true. `grid` is an array of equal-length strings indexed by `[y][x]`;
  `legend` maps each one-character glyph used by the grid to a visible meaning.

A continuous 3D game can therefore report an observation without `grid` or
`legend`:

```json
{
  "status":"playing",
  "player":{
    "position":[0.0,1.0,0.0],
    "velocity":[0.0,0.0,0.0],
    "grounded":true,
    "inventory":[]
  },
  "entities":[
    {
      "id":"floor-0",
      "kind":"floor",
      "position":[0.0,-0.25,0.0],
      "size":[4.0,0.5,4.0],
      "state":{}
    },
    {
      "id":"door",
      "kind":"door",
      "position":[2.0,1.0,0.0],
      "size":[0.25,2.0,2.0],
      "state":{"open":false}
    }
  ],
  "events":[],
  "controls":["north","south","east","west","jump","interact","wait"]
}
```

Coordinate axes, control durations, and the meaning of positive directions are
game semantics and stay stable for a given `game.version`. For a deterministic
step-driven physics game, the server may simulate a fixed number of fixed-time
steps per control and then pause. The response still acknowledges completed
execution, and `observe` still does not advance simulation. Protocol version 1
does not by itself promise bitwise-identical floating-point results across
different runtime, engine, architecture, or build versions.

Games may add visible, JSON-serializable fields such as `tick`, `goal`, or an
`extensions` object. Extensions must be namespaced by their owner and must not
contain developer rules, passability for untried controls, routes, credentials,
or evaluation-only state. The reference server's compatibility extension is
sanitized on the wire; the Python adapter restores any legacy host-only fields
locally so existing recordings remain byte-for-byte comparable. Clients retain
unknown visible fields when recording but do not infer a capability from their
presence.

Detectors must honor the handshake. Grid/legend collision detectors run only
when the required capabilities and well-formed fields are present. Entity,
inventory, event-feedback, reset-consistency, and repeated-transition checks
use only the generic fields they actually receive; absence of an optional
capability is not itself a defect. Continuous-world checks likewise use
`continuous` and `solids_visible` only when advertised; visible entity bounds
may be compared with observed movement outcomes, but they are not treated as
collider ground truth.

## Errors

Version 1 defines these stable server error codes:

| code | meaning |
| --- | --- |
| `invalid_json` | The line is not valid JSON. |
| `invalid_utf8` | The line is not valid UTF-8. |
| `invalid_message` | The decoded JSON value is not an object. |
| `message_too_large` | The line exceeds 1 MiB or is not newline-terminated within that bound. |
| `invalid_request` | The request ID or type is absent, empty, or has the wrong type. |
| `unsupported_protocol` | The server does not implement the requested protocol. |
| `hello_required` | A non-hello command arrived before negotiation. |
| `already_initialized` | A second `hello` arrived on an initialized connection. |
| `duplicate_id` | The request ID was already used on this connection. |
| `invalid_control` | `act.control` is missing or is not a string. |
| `illegal_control` | The control is unknown or unavailable in the current state. |
| `invalid_seed` | A required reset seed is absent or is not an integer. |
| `unknown_type` | The request type is not a version-1 command. |
| `timeout` | The game runtime did not complete the operation within its bound. |
| `internal_error` | The server failed without exposing implementation details. |

Malformed input is a protocol error, not a game event. The server should keep
the connection usable after a recoverable request error. It closes after an
unrecoverable framing error or when it cannot know whether its state and the
client's state still agree.

## Timeouts and retry safety

Clients use bounded connect and response timeouts. Servers also bound dispatch
to a game runtime or Unity main thread. Exact durations are deployment settings,
not wire-protocol constants.

An `observe` that times out is safe to retry with a new request ID. An `act` or
`reset` is not automatically retried: the caller may not know whether the first
mutation executed. A server emits a `timeout` response only when it can prevent
that timed-out mutation from executing later. Otherwise it closes the stream,
and the client marks the state unknown until it reconnects, performs `hello`,
and explicitly establishes state with `observe` or `reset`.

## Ground-truth separation

Injected-defect ground truth never crosses this protocol. In particular, it is
forbidden in observations, events, errors, game IDs or versions, and extension
fields. The `defect_injection` capability advertises test-fixture support only.

The reference server's `--defect` option is process configuration. Evaluation
receives the expected defect separately from the harness (for example through
an explicit ground-truth option) and may store it only under recording
evaluation keys. Model prompt builders consume observations and sweep facts,
not evaluation keys or server process metadata.

## Reference TCP session

The reference game can be served on loopback with:

```sh
python3 -m autoplay_qa serve --port 8765 --seed 5 --defect wall-clip
```

All game-driving commands may select it with
`--game tcp://127.0.0.1:8765`. Stdio transports exchange the same request lines
over a child process's stdin/stdout; logs remain on stderr. Neither transport
changes recording, prompt, or detector semantics.
