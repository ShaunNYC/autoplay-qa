using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.IO;
using System.Net;
using System.Net.Sockets;
using System.Text;
using System.Threading;
using UnityEngine;

namespace AutoPlayQA
{
    /// <summary>
    /// Localhost-only AutoPlay QA protocol server. Network I/O runs on a
    /// background thread; game adapter calls are dispatched by Update.
    /// </summary>
    [DisallowMultipleComponent]
    public sealed class AutoPlayQAServer : MonoBehaviour
    {
        [Header("Game adapter")]
        [Tooltip("A MonoBehaviour on this GameObject or another object that implements IAutoPlayQAGame.")]
        [SerializeField]
        private MonoBehaviour gameComponent;

        [SerializeField]
        private AutoPlayQAGameInfo game = new AutoPlayQAGameInfo();

        [SerializeField]
        private AutoPlayQACapabilities capabilities = new AutoPlayQACapabilities();

        [Header("Local server")]
        [SerializeField]
        private bool startOnEnable = true;

        [SerializeField]
        [Range(0, 65535)]
        private int port = 8765;

        [Tooltip("Maximum time a network request waits to be handled by Unity's main thread.")]
        [SerializeField]
        [Min(0.1f)]
        private float mainThreadTimeoutSeconds = 5f;

        [Tooltip("Bounds main-thread work when a client sends a burst of requests.")]
        [SerializeField]
        [Min(1)]
        private int maxRequestsPerFrame = 32;

        private readonly ConcurrentQueue<PendingRequest> pendingRequests =
            new ConcurrentQueue<PendingRequest>();
        private readonly object clientLock = new object();
        private readonly object backgroundErrorLock = new object();

        private IAutoPlayQAGame gameAdapter;
        private TcpListener listener;
        private TcpClient activeClient;
        private Thread networkThread;
        private volatile bool running;
        private int boundPort;
        private int requestTimeoutMilliseconds;
        private string backgroundError;

        public bool IsRunning
        {
            get { return running; }
        }

        /// <summary>The actual port, useful when the configured port is zero.</summary>
        public int BoundPort
        {
            get { return boundPort; }
        }

        /// <summary>
        /// Configure a server created at runtime before its GameObject is
        /// enabled. Inspector-authored integrations can continue using the
        /// serialized fields instead.
        /// </summary>
        public void Configure(
            MonoBehaviour adapter,
            AutoPlayQAGameInfo gameInfo,
            AutoPlayQACapabilities advertisedCapabilities,
            int listenPort)
        {
            if (running)
            {
                throw new InvalidOperationException(
                    "A running AutoPlay QA server cannot be reconfigured.");
            }

            if (adapter == null || !(adapter is IAutoPlayQAGame))
            {
                throw new ArgumentException(
                    "Adapter must be a MonoBehaviour implementing IAutoPlayQAGame.",
                    "adapter");
            }

            if (gameInfo == null)
            {
                throw new ArgumentNullException("gameInfo");
            }

            if (advertisedCapabilities == null)
            {
                throw new ArgumentNullException("advertisedCapabilities");
            }

            if (listenPort < 0 || listenPort > 65535)
            {
                throw new ArgumentOutOfRangeException("listenPort");
            }

            gameComponent = adapter;
            game = gameInfo;
            capabilities = advertisedCapabilities;
            port = listenPort;
        }

        private void OnEnable()
        {
            if (startOnEnable)
            {
                StartServer();
            }
        }

        private void OnDisable()
        {
            StopServer();
        }

        private void OnApplicationQuit()
        {
            StopServer();
        }

        private void Update()
        {
            FlushBackgroundError();

            var limit = Math.Max(1, maxRequestsPerFrame);
            for (var count = 0; count < limit; count++)
            {
                PendingRequest pending;
                if (!pendingRequests.TryDequeue(out pending))
                {
                    break;
                }

                if (!pending.TryBegin())
                {
                    continue;
                }

                ExecuteOnMainThread(pending);
            }
        }

        public void StartServer()
        {
            if (running)
            {
                return;
            }

            gameAdapter = ResolveGameAdapter();
            if (gameAdapter == null)
            {
                Debug.LogError(
                    "AutoPlay QA server needs a MonoBehaviour that implements IAutoPlayQAGame.",
                    this);
                return;
            }

            if (port < 0 || port > 65535)
            {
                Debug.LogError("AutoPlay QA port must be between 0 and 65535.", this);
                return;
            }

            requestTimeoutMilliseconds = Math.Max(
                100,
                (int)Math.Min(int.MaxValue, mainThreadTimeoutSeconds * 1000.0));

            try
            {
                // Loopback is intentional: the bridge never listens on a LAN or
                // public interface.
                listener = new TcpListener(IPAddress.Loopback, port);
                listener.Start();
                boundPort = ((IPEndPoint)listener.LocalEndpoint).Port;
                running = true;
                networkThread = new Thread(AcceptLoop);
                networkThread.IsBackground = true;
                networkThread.Name = "AutoPlay QA localhost server";
                networkThread.Start();
                Debug.Log(
                    "AutoPlay QA listening on tcp://127.0.0.1:" + boundPort + 
                    " using " + AutoPlayQAProtocol.Version + ".",
                    this);
            }
            catch (Exception exception)
            {
                running = false;
                if (listener != null)
                {
                    listener.Stop();
                    listener = null;
                }

                Debug.LogError("AutoPlay QA could not start: " + exception.Message, this);
            }
        }

        public void StopServer()
        {
            if (!running && listener == null)
            {
                return;
            }

            running = false;
            var stoppedListener = listener;
            listener = null;
            if (stoppedListener != null)
            {
                try
                {
                    stoppedListener.Stop();
                }
                catch (SocketException)
                {
                    // Stopping an already-closed listener is harmless.
                }
            }

            lock (clientLock)
            {
                if (activeClient != null)
                {
                    try
                    {
                        activeClient.Close();
                    }
                    catch (SocketException)
                    {
                        // Closing is best effort during teardown.
                    }

                    activeClient = null;
                }
            }

            PendingRequest pending;
            while (pendingRequests.TryDequeue(out pending))
            {
                pending.TryCompleteWithoutExecution(
                    AutoPlayQAProtocolJson.Error(
                        pending.Request.Id,
                        "server_stopping",
                        "The Unity server is stopping.",
                        false));
            }

            var stoppedThread = networkThread;
            networkThread = null;
            if (stoppedThread != null && stoppedThread != Thread.CurrentThread)
            {
                stoppedThread.Join(1000);
            }

            boundPort = 0;
        }

        private IAutoPlayQAGame ResolveGameAdapter()
        {
            if (gameComponent != null)
            {
                return gameComponent as IAutoPlayQAGame;
            }

            var behaviours = GetComponents<MonoBehaviour>();
            for (var index = 0; index < behaviours.Length; index++)
            {
                var candidate = behaviours[index] as IAutoPlayQAGame;
                if (candidate != null)
                {
                    gameComponent = behaviours[index];
                    return candidate;
                }
            }

            return null;
        }

        private void AcceptLoop()
        {
            while (running)
            {
                TcpClient client = null;
                try
                {
                    var currentListener = listener;
                    if (currentListener == null)
                    {
                        return;
                    }

                    client = currentListener.AcceptTcpClient();
                    client.NoDelay = true;
                    client.SendTimeout = 5000;
                    lock (clientLock)
                    {
                        activeClient = client;
                    }

                    HandleClient(client);
                }
                catch (SocketException exception)
                {
                    if (running)
                    {
                        SetBackgroundError("AutoPlay QA socket error: " + exception.Message);
                    }
                }
                catch (IOException exception)
                {
                    if (running)
                    {
                        SetBackgroundError("AutoPlay QA connection error: " + exception.Message);
                    }
                }
                catch (Exception exception)
                {
                    if (running)
                    {
                        SetBackgroundError("AutoPlay QA server error: " + exception.Message);
                    }
                }
                finally
                {
                    lock (clientLock)
                    {
                        if (ReferenceEquals(activeClient, client))
                        {
                            activeClient = null;
                        }
                    }

                    if (client != null)
                    {
                        client.Close();
                    }
                }
            }
        }

        private void HandleClient(TcpClient client)
        {
            var utf8 = new UTF8Encoding(false, true);
            using (var stream = client.GetStream())
            using (var reader = new StreamReader(stream, utf8, false, 4096, true))
            using (var writer = new StreamWriter(stream, new UTF8Encoding(false), 4096, true))
            {
                writer.NewLine = "\n";
                writer.AutoFlush = true;
                var helloComplete = false;
                var requestIds = new HashSet<string>(StringComparer.Ordinal);

                while (running && client.Connected)
                {
                    string line;
                    bool tooLong;
                    try
                    {
                        if (!TryReadBoundedLine(reader, out line, out tooLong))
                        {
                            return;
                        }
                    }
                    catch (DecoderFallbackException)
                    {
                        writer.WriteLine(
                            AutoPlayQAProtocolJson.Error(
                                null,
                                "invalid_utf8",
                                "Message is not valid UTF-8.",
                                false));
                        return;
                    }

                    if (tooLong || Encoding.UTF8.GetByteCount(line) > AutoPlayQAProtocol.MaxLineBytes)
                    {
                        writer.WriteLine(
                            AutoPlayQAProtocolJson.Error(
                                null,
                                "message_too_large",
                                "Message exceeds 1 MiB or is missing its newline terminator.",
                                false));
                        continue;
                    }

                    AutoPlayQARequest request;
                    string errorCode;
                    string errorMessage;
                    if (!AutoPlayQAProtocolJson.TryParseRequest(
                        line,
                        out request,
                        out errorCode,
                        out errorMessage))
                    {
                        writer.WriteLine(
                            AutoPlayQAProtocolJson.Error(
                                request == null ? null : request.Id,
                                errorCode,
                                errorMessage,
                                false));
                        continue;
                    }

                    if (!requestIds.Add(request.Id))
                    {
                        writer.WriteLine(
                            AutoPlayQAProtocolJson.Error(
                                request.Id,
                                "duplicate_id",
                                "Request id was already used on this connection.",
                                false));
                        continue;
                    }

                    if (!helloComplete && request.Type != "hello")
                    {
                        writer.WriteLine(
                            AutoPlayQAProtocolJson.Error(
                                request.Id,
                                "hello_required",
                                "Send a compatible hello request first.",
                                false));
                        continue;
                    }

                    if (!IsKnownRequestType(request.Type))
                    {
                        writer.WriteLine(
                            AutoPlayQAProtocolJson.Error(
                                request.Id,
                                "unknown_type",
                                "Unknown request type.",
                                false));
                        continue;
                    }

                    if (request.Type == "hello" && helloComplete)
                    {
                        writer.WriteLine(
                            AutoPlayQAProtocolJson.Error(
                                request.Id,
                                "already_initialized",
                                "This connection has already completed hello.",
                                false));
                        continue;
                    }

                    var pending = new PendingRequest(request);
                    pendingRequests.Enqueue(pending);
                    if (!pending.Wait(requestTimeoutMilliseconds))
                    {
                        if (pending.TryCancelBeforeExecution())
                        {
                            writer.WriteLine(
                                AutoPlayQAProtocolJson.Error(
                                    request.Id,
                                    "timeout",
                                    "Unity did not handle the request before the server timeout.",
                                    false));
                            pending.ReleaseWaiter();
                            continue;
                        }

                        if (!pending.Wait(0))
                        {
                            // The main thread started a possibly mutating call.
                            // A timeout reply would falsely imply it was
                            // canceled, and a late reply would desynchronize
                            // framing, so close the connection as state-unknown.
                            pending.ReleaseWaiter();
                            return;
                        }
                    }

                    writer.WriteLine(pending.Response);
                    if (pending.HelloAccepted)
                    {
                        helloComplete = true;
                    }

                    var closeAfterResponse = pending.CloseAfterResponse;
                    pending.ReleaseWaiter();
                    if (closeAfterResponse)
                    {
                        return;
                    }
                }
            }
        }

        private static bool TryReadBoundedLine(
            TextReader reader,
            out string line,
            out bool tooLong)
        {
            var builder = new StringBuilder(Math.Min(4096, AutoPlayQAProtocol.MaxLineBytes));
            tooLong = false;
            while (true)
            {
                var value = reader.Read();
                if (value < 0)
                {
                    line = builder.ToString();
                    if (builder.Length == 0 && !tooLong)
                    {
                        return false;
                    }

                    // A partial final line is a framing error even when its
                    // payload would otherwise fit.
                    tooLong = true;
                    return true;
                }

                var character = (char)value;
                if (character == '\n')
                {
                    line = builder.ToString();
                    return true;
                }

                if (character == '\r')
                {
                    continue;
                }

                if (builder.Length < AutoPlayQAProtocol.MaxLineBytes)
                {
                    builder.Append(character);
                }
                else
                {
                    tooLong = true;
                }
            }
        }

        private static bool IsKnownRequestType(string type)
        {
            return type == "hello" || type == "reset" || type == "observe" ||
                type == "act" || type == "close";
        }

        private void ExecuteOnMainThread(PendingRequest pending)
        {
            try
            {
                var request = pending.Request;
                switch (request.Type)
                {
                    case "hello":
                        ExecuteHello(pending);
                        break;
                    case "reset":
                        ExecuteReset(pending);
                        break;
                    case "observe":
                        ExecuteObserve(pending);
                        break;
                    case "act":
                        ExecuteAct(pending);
                        break;
                    case "close":
                        pending.CloseAfterResponse = true;
                        pending.Complete(AutoPlayQAProtocolJson.Close(request.Id));
                        break;
                    default:
                        pending.Complete(
                            AutoPlayQAProtocolJson.Error(
                                request.Id,
                                "unknown_type",
                                "Unknown request type.",
                                false));
                        break;
                }
            }
            catch (Exception exception)
            {
                Debug.LogException(exception, this);
                pending.Complete(
                    AutoPlayQAProtocolJson.Error(
                        pending.Request.Id,
                        "internal_error",
                        "The Unity game adapter could not complete the request.",
                        false));
            }
        }

        private void ExecuteHello(PendingRequest pending)
        {
            var request = pending.Request;
            if (request.Protocol != AutoPlayQAProtocol.Version)
            {
                pending.Complete(
                    AutoPlayQAProtocolJson.Error(
                        request.Id,
                        "unsupported_protocol",
                        "The server supports protocol " + AutoPlayQAProtocol.Version + ".",
                        false));
                return;
            }

            var controls = gameAdapter.Controls;
            string validationError;
            if (!TryValidateConfiguration(controls, out validationError))
            {
                pending.Complete(
                    AutoPlayQAProtocolJson.Error(
                        request.Id,
                        "internal_error",
                        validationError,
                        false));
                return;
            }

            pending.HelloAccepted = true;
            pending.Complete(AutoPlayQAProtocolJson.Hello(request.Id, game, controls, capabilities));
        }

        private void ExecuteReset(PendingRequest pending)
        {
            var request = pending.Request;
            if (capabilities == null || !capabilities.Reset)
            {
                pending.Complete(
                    AutoPlayQAProtocolJson.Error(
                        request.Id,
                        "invalid_request",
                        "This game does not advertise reset support.",
                        false));
                return;
            }

            if (!request.HasSeed)
            {
                pending.Complete(
                    AutoPlayQAProtocolJson.Error(
                        request.Id,
                        "invalid_seed",
                        "Reset requires an integer 'seed'.",
                        false));
                return;
            }

            gameAdapter.Reset(request.Seed);
            var observation = RequireObservation(gameAdapter.Observe());
            pending.Complete(
                AutoPlayQAProtocolJson.Observation(
                    request.Id,
                    "reset",
                    observation,
                    capabilities));
        }

        private void ExecuteObserve(PendingRequest pending)
        {
            var observation = RequireObservation(gameAdapter.Observe());
            pending.Complete(
                AutoPlayQAProtocolJson.Observation(
                    pending.Request.Id,
                    "observe",
                    observation,
                    capabilities));
        }

        private void ExecuteAct(PendingRequest pending)
        {
            var request = pending.Request;
            if (string.IsNullOrEmpty(request.Control))
            {
                pending.Complete(
                    AutoPlayQAProtocolJson.Error(
                        request.Id,
                        "invalid_control",
                        "Act requires a non-empty string 'control'.",
                        false));
                return;
            }

            var advertisedControls = gameAdapter.Controls;
            var current = RequireObservation(gameAdapter.Observe());
            if (!Contains(advertisedControls, request.Control) ||
                !Contains(current.Controls, request.Control))
            {
                pending.Complete(
                    AutoPlayQAProtocolJson.Error(
                        request.Id,
                        "illegal_control",
                        "The requested control is not legal in the current state.",
                        false));
                return;
            }

            var observation = RequireObservation(gameAdapter.Act(request.Control));
            pending.Complete(
                AutoPlayQAProtocolJson.Observation(
                    request.Id,
                    "act",
                    observation,
                    capabilities));
        }

        private bool TryValidateConfiguration(string[] controls, out string error)
        {
            if (game == null || string.IsNullOrEmpty(game.Id) || string.IsNullOrEmpty(game.Version))
            {
                error = "Game id and version must be non-empty.";
                return false;
            }

            if (capabilities == null)
            {
                error = "Capabilities must be configured.";
                return false;
            }

            if (capabilities.Reset && !capabilities.Seed)
            {
                error = "IAutoPlayQAGame.Reset requires the seed capability when reset is advertised.";
                return false;
            }

            if (controls == null || controls.Length == 0)
            {
                error = "The game adapter must advertise at least one control.";
                return false;
            }

            for (var first = 0; first < controls.Length; first++)
            {
                if (string.IsNullOrEmpty(controls[first]))
                {
                    error = "Advertised controls must be non-empty strings.";
                    return false;
                }

                for (var second = first + 1; second < controls.Length; second++)
                {
                    if (controls[first] == controls[second])
                    {
                        error = "Advertised controls must be unique.";
                        return false;
                    }
                }
            }

            error = null;
            return true;
        }

        private AutoPlayQAObservation RequireObservation(AutoPlayQAObservation observation)
        {
            if (observation == null)
            {
                throw new InvalidOperationException("The game adapter returned a null observation.");
            }

            if (string.IsNullOrEmpty(observation.Status))
            {
                throw new InvalidOperationException("The game adapter returned an observation without status.");
            }

            if (observation.Player == null || observation.Player.Inventory == null)
            {
                throw new InvalidOperationException("The game adapter returned an invalid player.");
            }

            RequireFinitePosition(observation.Player.Position, "player");
            RequireFiniteVector(observation.Player.Velocity, "player velocity");
            if (observation.Player.Position != null && observation.Player.Velocity != null &&
                observation.Player.Position.Length != observation.Player.Velocity.Length)
            {
                throw new InvalidOperationException(
                    "Player velocity must have the same dimensionality as position.");
            }

            if (capabilities != null && capabilities.Continuous &&
                (observation.Player.Velocity == null || !observation.Player.Grounded.HasValue))
            {
                throw new InvalidOperationException(
                    "Continuous capability requires player velocity and grounded state.");
            }

            RequireStrings(observation.Player.Inventory, "player inventory");
            RequireStrings(observation.Events, "events");
            RequireStrings(observation.Controls, "current controls");

            var catalog = gameAdapter.Controls;
            for (var index = 0; index < observation.Controls.Length; index++)
            {
                if (!Contains(catalog, observation.Controls[index]))
                {
                    throw new InvalidOperationException(
                        "An observation control is absent from the hello catalog.");
                }
            }

            if (observation.Entities == null)
            {
                throw new InvalidOperationException("Observation entities must be an array.");
            }

            var entityIds = new HashSet<string>(StringComparer.Ordinal);
            for (var index = 0; index < observation.Entities.Length; index++)
            {
                var entity = observation.Entities[index];
                if (entity == null || string.IsNullOrEmpty(entity.Id) ||
                    string.IsNullOrEmpty(entity.Kind) || entity.State == null)
                {
                    throw new InvalidOperationException("An observation entity is invalid.");
                }

                if (!entityIds.Add(entity.Id))
                {
                    throw new InvalidOperationException("Observation entity ids must be unique.");
                }

                RequireFinitePosition(entity.Position, "entity");
                RequireFiniteVector(entity.Size, "entity size");
                if (entity.Position != null && entity.Size != null &&
                    entity.Position.Length != entity.Size.Length)
                {
                    throw new InvalidOperationException(
                        "Observation entity size must have the same dimensionality as position.");
                }

                if (entity.Size != null)
                {
                    if (entity.Size.Length == 0)
                    {
                        throw new InvalidOperationException(
                            "Observation entity size must contain at least one dimension.");
                    }

                    for (var sizeIndex = 0; sizeIndex < entity.Size.Length; sizeIndex++)
                    {
                        if (entity.Size[sizeIndex] < 0.0)
                        {
                            throw new InvalidOperationException(
                                "Observation entity size dimensions must be non-negative.");
                        }
                    }
                }

                var stateKeys = new HashSet<string>(StringComparer.Ordinal);
                for (var stateIndex = 0; stateIndex < entity.State.Length; stateIndex++)
                {
                    var state = entity.State[stateIndex];
                    if (state == null || string.IsNullOrEmpty(state.Key) ||
                        !stateKeys.Add(state.Key))
                    {
                        throw new InvalidOperationException("Entity state keys must be non-empty and unique.");
                    }

                    if (state.Kind == AutoPlayQAValueKind.Number &&
                        (double.IsNaN(state.NumberValue) || double.IsInfinity(state.NumberValue)))
                    {
                        throw new InvalidOperationException("Entity numeric state must be finite.");
                    }
                }
            }

            if (capabilities != null && capabilities.Grid)
            {
                if (observation.Grid == null)
                {
                    throw new InvalidOperationException("Grid capability requires an observation grid.");
                }

                RequireStrings(observation.Grid, "grid");
                if (observation.Grid.Length > 0)
                {
                    var width = observation.Grid[0].Length;
                    for (var row = 1; row < observation.Grid.Length; row++)
                    {
                        if (observation.Grid[row].Length != width)
                        {
                            throw new InvalidOperationException("Observation grid rows must have equal length.");
                        }
                    }
                }
            }

            if (capabilities != null && capabilities.Legend && observation.Legend == null)
            {
                throw new InvalidOperationException("Legend capability requires an observation legend.");
            }

            if (observation.Legend != null)
            {
                var glyphs = new HashSet<string>(StringComparer.Ordinal);
                for (var index = 0; index < observation.Legend.Length; index++)
                {
                    var entry = observation.Legend[index];
                    if (entry == null || string.IsNullOrEmpty(entry.Glyph) ||
                        entry.Glyph.Length != 1 ||
                        entry.Meaning == null || !glyphs.Add(entry.Glyph))
                    {
                        throw new InvalidOperationException(
                            "Legend glyphs must be non-empty, unique, and have string meanings.");
                    }
                }
            }

            return observation;
        }

        private static void RequireStrings(string[] values, string field)
        {
            if (values == null)
            {
                throw new InvalidOperationException("Observation " + field + " must be an array.");
            }

            for (var index = 0; index < values.Length; index++)
            {
                if (values[index] == null)
                {
                    throw new InvalidOperationException(
                        "Observation " + field + " must contain only strings.");
                }
            }
        }

        private static void RequireFinitePosition(double[] position, string field)
        {
            if (position == null)
            {
                return;
            }

            for (var index = 0; index < position.Length; index++)
            {
                if (double.IsNaN(position[index]) || double.IsInfinity(position[index]))
                {
                    throw new InvalidOperationException(
                        "Observation " + field + " position must contain finite coordinates.");
                }
            }
        }

        private static void RequireFiniteVector(double[] values, string field)
        {
            if (values == null)
            {
                return;
            }

            for (var index = 0; index < values.Length; index++)
            {
                if (double.IsNaN(values[index]) || double.IsInfinity(values[index]))
                {
                    throw new InvalidOperationException(
                        "Observation " + field + " must contain finite coordinates.");
                }
            }
        }

        private static bool Contains(string[] values, string expected)
        {
            if (values == null)
            {
                return false;
            }

            for (var index = 0; index < values.Length; index++)
            {
                if (values[index] == expected)
                {
                    return true;
                }
            }

            return false;
        }

        private void SetBackgroundError(string message)
        {
            lock (backgroundErrorLock)
            {
                backgroundError = message;
            }
        }

        private void FlushBackgroundError()
        {
            string message;
            lock (backgroundErrorLock)
            {
                message = backgroundError;
                backgroundError = null;
            }

            if (!string.IsNullOrEmpty(message))
            {
                Debug.LogWarning(message, this);
            }
        }

        private sealed class PendingRequest
        {
            // 0 queued, 1 executing, 2 canceled before execution, 3 complete.
            private readonly ManualResetEventSlim completed = new ManualResetEventSlim(false);
            private int state;
            private int waiterReleased;
            private int completionSignaled;
            private int disposed;

            public PendingRequest(AutoPlayQARequest request)
            {
                Request = request;
            }

            public AutoPlayQARequest Request { get; private set; }
            public string Response { get; private set; }
            public bool HelloAccepted { get; set; }
            public bool CloseAfterResponse { get; set; }

            public bool TryBegin()
            {
                return Interlocked.CompareExchange(ref state, 1, 0) == 0;
            }

            public bool Wait(int milliseconds)
            {
                return completed.Wait(milliseconds);
            }

            public void Complete(string response)
            {
                Response = response;
                Volatile.Write(ref state, 3);
                completed.Set();
                Volatile.Write(ref completionSignaled, 1);
                DisposeIfFinished();
            }

            public bool TryCompleteWithoutExecution(string response)
            {
                if (Interlocked.CompareExchange(ref state, 3, 0) != 0)
                {
                    return false;
                }

                Response = response;
                completed.Set();
                Volatile.Write(ref completionSignaled, 1);
                DisposeIfFinished();
                return true;
            }

            public bool TryCancelBeforeExecution()
            {
                return Interlocked.CompareExchange(ref state, 2, 0) == 0;
            }

            public void ReleaseWaiter()
            {
                Volatile.Write(ref waiterReleased, 1);
                DisposeIfFinished();
            }

            private void DisposeIfFinished()
            {
                var currentState = Volatile.Read(ref state);
                var safeToDispose = currentState == 2 ||
                    (currentState == 3 && Volatile.Read(ref completionSignaled) == 1);
                if (safeToDispose &&
                    Volatile.Read(ref waiterReleased) == 1 &&
                    Interlocked.CompareExchange(ref disposed, 1, 0) == 0)
                {
                    completed.Dispose();
                }
            }
        }
    }
}
