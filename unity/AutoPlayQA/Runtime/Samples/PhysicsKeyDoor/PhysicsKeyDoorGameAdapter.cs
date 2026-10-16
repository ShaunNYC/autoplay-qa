using System;
using System.Collections.Generic;
using UnityEngine;

namespace AutoPlayQA.Samples.PhysicsKeyDoor
{
    /// <summary>
    /// A deterministic, step-driven real-time physics sample. The scene is
    /// assembled entirely from Unity primitives at runtime.
    /// </summary>
    [DisallowMultipleComponent]
    public sealed class PhysicsKeyDoorGameAdapter : MonoBehaviour, IAutoPlayQAGame
    {
        private const float FixedStepSeconds = 0.02f;
        private const int LongActionSteps = 20;
        private const int ShortActionSteps = 2;
        private const float MoveSpeed = 4f;
        private const float JumpSpeed = 5f;
        private const float InteractionDistance = 1.5f;
        private const int EnvironmentLayer = 0;
        private const int PlayerLayer = 2;

        private static readonly string[] AllControls =
        {
            "north", "south", "east", "west", "jump", "interact", "wait"
        };

        private static readonly Vector3 SpawnPosition = new Vector3(-4f, 1.05f, -4f);
        private static readonly Vector3 KeyPosition = new Vector3(-4f, 0.5f, 2f);
        private static readonly Vector3 KeySize = new Vector3(0.7f, 0.7f, 0.7f);
        private static readonly Vector3 DoorPosition = new Vector3(0f, 1.25f, 2f);
        private static readonly Vector3 DoorSize = new Vector3(0.5f, 2.5f, 2f);
        private static readonly Vector3 ExitPosition = new Vector3(4f, 0.1f, 2f);
        private static readonly Vector3 ExitSize = new Vector3(1.5f, 0.2f, 1.5f);

        private readonly List<string> events = new List<string>();
        private readonly List<LevelPart> floorParts = new List<LevelPart>();
        private readonly List<LevelPart> wallParts = new List<LevelPart>();
        private readonly List<LevelPart> floorGapCandidates = new List<LevelPart>();

        private PhysicsKeyDoorLaunchOptions launchOptions;
        private Rigidbody playerBody;
        private Collider keyCollider;
        private Collider doorCollider;
        private Collider exitCollider;
        private Collider killPlaneCollider;
        private Renderer keyRenderer;
        private Renderer doorRenderer;

        private bool initialized;
        private bool keyPresent;
        private bool hasKey;
        private bool doorOpen;
        private bool won;
        private bool pendingKey;
        private bool pendingExit;
        private bool pendingFall;
        private int tick;
        private int episodeSeed;

        private float previousFixedDeltaTime;
        private Vector3 previousGravity;
        private bool physicsSettingsCaptured;
#if UNITY_2022_2_OR_NEWER
        private SimulationMode previousSimulationMode;
#else
        private bool previousAutoSimulation;
#endif

        public string[] Controls
        {
            get { return (string[])AllControls.Clone(); }
        }

        internal void ConfigureLaunch(PhysicsKeyDoorLaunchOptions options)
        {
            if (initialized)
            {
                throw new InvalidOperationException(
                    "PhysicsKeyDoor launch options must be configured before Awake.");
            }

            launchOptions = options;
        }

        private void Awake()
        {
            if (launchOptions == null)
            {
                launchOptions = new PhysicsKeyDoorLaunchOptions();
            }

            CapturePhysicsSettings();
            BuildLevel();
            ResetEpisode(launchOptions.Seed, false);
            initialized = true;
        }

        private void OnDestroy()
        {
            RestorePhysicsSettings();
        }

        public void Reset(int seed)
        {
            EnsureInitialized();
            var preserveProgress =
                launchOptions.Defect == PhysicsKeyDoorDefect.RespawnLeak;
            ResetEpisode(seed, preserveProgress);
        }

        public AutoPlayQAObservation Observe()
        {
            EnsureInitialized();
            var entities = new List<AutoPlayQAEntity>(
                floorParts.Count + wallParts.Count + 3);

            for (var index = 0; index < floorParts.Count; index++)
            {
                entities.Add(CreateSolidObservation(floorParts[index]));
            }

            for (var index = 0; index < wallParts.Count; index++)
            {
                entities.Add(CreateSolidObservation(wallParts[index]));
            }

            entities.Add(new AutoPlayQAEntity
            {
                Id = "door",
                Kind = "door",
                Position = ToArray(DoorPosition),
                Size = ToArray(DoorSize),
                State = new[]
                {
                    AutoPlayQAStateEntry.Boolean("open", doorOpen),
                    AutoPlayQAStateEntry.Boolean("solid", !doorOpen)
                }
            });
            entities.Add(new AutoPlayQAEntity
            {
                Id = "key",
                Kind = "key",
                Position = keyPresent ? ToArray(KeyPosition) : null,
                Size = ToArray(KeySize),
                State = new[]
                {
                    AutoPlayQAStateEntry.Boolean("present", keyPresent)
                }
            });
            entities.Add(new AutoPlayQAEntity
            {
                Id = "exit",
                Kind = "exit",
                Position = ToArray(ExitPosition),
                Size = ToArray(ExitSize),
                State = new[]
                {
                    AutoPlayQAStateEntry.Boolean("reached", won)
                }
            });

            return new AutoPlayQAObservation
            {
                Tick = tick,
                Status = won ? "won" : "playing",
                Player = new AutoPlayQAPlayer
                {
                    Position = ToArray(playerBody.position),
                    Velocity = ToArray(GetPlayerVelocity()),
                    Grounded = IsGrounded(),
                    Inventory = hasKey ? new[] { "key" } : new string[0]
                },
                Entities = entities.ToArray(),
                Events = events.ToArray(),
                Controls = won ? new string[0] : Controls
            };
        }

        public AutoPlayQAObservation Act(string control)
        {
            EnsureInitialized();
            if (won)
            {
                throw new InvalidOperationException("PhysicsKeyDoor is already won.");
            }

            if (!ContainsControl(control))
            {
                throw new ArgumentException("Unknown PhysicsKeyDoor control.", "control");
            }

            tick++;
            events.Clear();
            switch (control)
            {
                case "north":
                    SimulateMovement(Vector3.forward);
                    break;
                case "south":
                    SimulateMovement(Vector3.back);
                    break;
                case "east":
                    SimulateMovement(Vector3.right);
                    break;
                case "west":
                    SimulateMovement(Vector3.left);
                    break;
                case "jump":
                    Jump();
                    break;
                case "interact":
                    Interact();
                    SimulateSteps(Vector3.zero, ShortActionSteps);
                    break;
                case "wait":
                    SimulateSteps(Vector3.zero, LongActionSteps);
                    break;
            }

            return Observe();
        }

        internal void NotifyTrigger(Collider other)
        {
            if (other == keyCollider && keyPresent)
            {
                pendingKey = true;
            }
            else if (other == exitCollider && !won)
            {
                pendingExit = true;
            }
            else if (other == killPlaneCollider)
            {
                pendingFall = true;
            }
        }

        private void CapturePhysicsSettings()
        {
            previousFixedDeltaTime = Time.fixedDeltaTime;
            previousGravity = Physics.gravity;
#if UNITY_2022_2_OR_NEWER
            previousSimulationMode = Physics.simulationMode;
            Physics.simulationMode = SimulationMode.Script;
#else
            previousAutoSimulation = Physics.autoSimulation;
            Physics.autoSimulation = false;
#endif
            Time.fixedDeltaTime = FixedStepSeconds;
            Physics.gravity = new Vector3(0f, -9.81f, 0f);
            physicsSettingsCaptured = true;
        }

        private void RestorePhysicsSettings()
        {
            if (!physicsSettingsCaptured)
            {
                return;
            }

            Time.fixedDeltaTime = previousFixedDeltaTime;
            Physics.gravity = previousGravity;
#if UNITY_2022_2_OR_NEWER
            Physics.simulationMode = previousSimulationMode;
#else
            Physics.autoSimulation = previousAutoSimulation;
#endif
            physicsSettingsCaptured = false;
        }

        private void BuildLevel()
        {
            var world = new GameObject("PhysicsKeyDoor Level");
            world.transform.SetParent(transform, false);

            for (var zIndex = 0; zIndex < 7; zIndex++)
            {
                for (var xIndex = 0; xIndex < 7; xIndex++)
                {
                    var position = new Vector3(
                        -6f + (xIndex * 2f),
                        -0.125f,
                        -6f + (zIndex * 2f));
                    var part = CreatePart(
                        world.transform,
                        "floor-" + xIndex + "-" + zIndex,
                        "floor",
                        position,
                        new Vector3(2f, 0.25f, 2f),
                        new Color(0.68f, 0.68f, 0.7f));
                    floorParts.Add(part);
                    if (IsFloorGapCandidate(xIndex, zIndex))
                    {
                        floorGapCandidates.Add(part);
                    }
                }
            }

            wallParts.Add(CreatePart(
                world.transform,
                "wall-perimeter-north",
                "wall",
                new Vector3(0f, 1.25f, 7.25f),
                new Vector3(14.5f, 2.5f, 0.5f),
                new Color(0.3f, 0.32f, 0.36f)));
            wallParts.Add(CreatePart(
                world.transform,
                "wall-perimeter-south",
                "wall",
                new Vector3(0f, 1.25f, -7.25f),
                new Vector3(14.5f, 2.5f, 0.5f),
                new Color(0.3f, 0.32f, 0.36f)));
            wallParts.Add(CreatePart(
                world.transform,
                "wall-perimeter-east",
                "wall",
                new Vector3(7.25f, 1.25f, 0f),
                new Vector3(0.5f, 2.5f, 14.5f),
                new Color(0.3f, 0.32f, 0.36f)));
            wallParts.Add(CreatePart(
                world.transform,
                "wall-perimeter-west",
                "wall",
                new Vector3(-7.25f, 1.25f, 0f),
                new Vector3(0.5f, 2.5f, 14.5f),
                new Color(0.3f, 0.32f, 0.36f)));

            var interiorZ = new[] { -6f, -4f, -2f, 0f, 4f, 6f };
            for (var index = 0; index < interiorZ.Length; index++)
            {
                wallParts.Add(CreatePart(
                    world.transform,
                    "wall-interior-" + index,
                    "wall",
                    new Vector3(0f, 1.25f, interiorZ[index]),
                    new Vector3(0.5f, 2.5f, 2f),
                    new Color(0.36f, 0.38f, 0.42f)));
            }

            var door = CreatePrimitive(
                world.transform,
                "door",
                PrimitiveType.Cube,
                DoorPosition,
                DoorSize,
                new Color(0.55f, 0.28f, 0.08f));
            doorCollider = door.GetComponent<Collider>();
            doorRenderer = door.GetComponent<Renderer>();

            var key = CreatePrimitive(
                world.transform,
                "key",
                PrimitiveType.Sphere,
                KeyPosition,
                KeySize,
                new Color(1f, 0.78f, 0.05f));
            keyCollider = key.GetComponent<Collider>();
            keyCollider.isTrigger = true;
            keyRenderer = key.GetComponent<Renderer>();

            var exit = CreatePrimitive(
                world.transform,
                "exit",
                PrimitiveType.Cube,
                ExitPosition,
                ExitSize,
                new Color(0.12f, 0.8f, 0.3f));
            exitCollider = exit.GetComponent<Collider>();
            exitCollider.isTrigger = true;

            var killPlane = CreatePrimitive(
                world.transform,
                "kill-plane",
                PrimitiveType.Cube,
                new Vector3(0f, -5.5f, 0f),
                new Vector3(18f, 1f, 18f),
                Color.black);
            killPlaneCollider = killPlane.GetComponent<Collider>();
            killPlaneCollider.isTrigger = true;
            killPlane.GetComponent<Renderer>().enabled = false;

            var player = CreatePrimitive(
                world.transform,
                "player",
                PrimitiveType.Capsule,
                SpawnPosition,
                Vector3.one,
                new Color(0.1f, 0.42f, 0.95f));
            player.layer = PlayerLayer;
            playerBody = player.AddComponent<Rigidbody>();
            playerBody.mass = 1f;
            playerBody.useGravity = true;
            playerBody.constraints = RigidbodyConstraints.FreezeRotation;
            playerBody.collisionDetectionMode = CollisionDetectionMode.ContinuousDynamic;
            playerBody.interpolation = RigidbodyInterpolation.None;
            playerBody.sleepThreshold = 0f;
            var relay = player.AddComponent<PhysicsKeyDoorTriggerRelay>();
            relay.Configure(this);

            CreatePresentation(world.transform);
            Physics.SyncTransforms();
        }

        private static LevelPart CreatePart(
            Transform parent,
            string id,
            string kind,
            Vector3 position,
            Vector3 size,
            Color color)
        {
            var gameObject = CreatePrimitive(
                parent,
                id,
                PrimitiveType.Cube,
                position,
                size,
                color);
            return new LevelPart
            {
                Id = id,
                Kind = kind,
                Position = position,
                Size = size,
                Collider = gameObject.GetComponent<Collider>()
            };
        }

        private static GameObject CreatePrimitive(
            Transform parent,
            string name,
            PrimitiveType type,
            Vector3 position,
            Vector3 size,
            Color color)
        {
            var gameObject = GameObject.CreatePrimitive(type);
            gameObject.name = name;
            gameObject.layer = EnvironmentLayer;
            gameObject.transform.SetParent(parent, false);
            gameObject.transform.position = position;
            gameObject.transform.localScale = size;
            var renderer = gameObject.GetComponent<Renderer>();
            if (renderer != null)
            {
                renderer.material.color = color;
            }

            return gameObject;
        }

        private static void CreatePresentation(Transform parent)
        {
            var cameraObject = new GameObject("PhysicsKeyDoor Camera");
            cameraObject.transform.SetParent(parent, false);
            cameraObject.transform.position = new Vector3(0f, 14f, -14f);
            cameraObject.transform.LookAt(new Vector3(0f, 0f, 0f));
            var camera = cameraObject.AddComponent<Camera>();
            camera.clearFlags = CameraClearFlags.SolidColor;
            camera.backgroundColor = new Color(0.08f, 0.1f, 0.14f);

            var lightObject = new GameObject("PhysicsKeyDoor Light");
            lightObject.transform.SetParent(parent, false);
            lightObject.transform.rotation = Quaternion.Euler(50f, -30f, 0f);
            var light = lightObject.AddComponent<Light>();
            light.type = LightType.Directional;
            light.intensity = 1.1f;
        }

        private void ResetEpisode(int seed, bool preserveProgress)
        {
            episodeSeed = seed;
            tick = 0;
            won = false;
            pendingKey = false;
            pendingExit = false;
            pendingFall = false;
            events.Clear();

            ApplySeededGeometryDefect();
            if (!preserveProgress)
            {
                RestoreLevelProgress();
            }

            ApplyKeyPresentation();
            ApplyDoorPresentation();
            RespawnPlayer();
            events.Add("Game started.");
            Physics.SyncTransforms();
        }

        private void ApplySeededGeometryDefect()
        {
            for (var index = 0; index < floorParts.Count; index++)
            {
                floorParts[index].Collider.enabled = true;
            }

            for (var index = 0; index < wallParts.Count; index++)
            {
                wallParts[index].Collider.enabled = true;
            }

            if (launchOptions.Defect == PhysicsKeyDoorDefect.FloorGap)
            {
                var selected = SelectIndex(episodeSeed, floorGapCandidates.Count, 0x76A9u);
                floorGapCandidates[selected].Collider.enabled = false;
            }
            else if (launchOptions.Defect == PhysicsKeyDoorDefect.WallTunnel)
            {
                // Only interior wall segments participate so the missing
                // collider is reachable from either room half.
                var interiorOffset = 4;
                var count = wallParts.Count - interiorOffset;
                var selected = SelectIndex(episodeSeed, count, 0x51F2u);
                wallParts[interiorOffset + selected].Collider.enabled = false;
            }
        }

        private void ApplyKeyPresentation()
        {
            keyCollider.enabled = keyPresent;
            keyRenderer.enabled = keyPresent;
        }

        private void ApplyDoorPresentation()
        {
            doorRenderer.enabled = !doorOpen;
            doorCollider.enabled = !doorOpen ||
                launchOptions.Defect == PhysicsKeyDoorDefect.DoorGhostCollider;
        }

        private void RespawnPlayer()
        {
            playerBody.position = SpawnPosition;
            playerBody.rotation = Quaternion.identity;
            SetPlayerVelocity(Vector3.zero);
            playerBody.angularVelocity = Vector3.zero;
            playerBody.WakeUp();
            Physics.SyncTransforms();
        }

        private void SimulateMovement(Vector3 direction)
        {
            SimulateSteps(direction * MoveSpeed, LongActionSteps);
        }

        private void Jump()
        {
            if (!IsGrounded())
            {
                events.Add("You cannot jump while airborne.");
                SimulateSteps(Vector3.zero, LongActionSteps);
                return;
            }

            var velocity = GetPlayerVelocity();
            velocity.y = JumpSpeed;
            SetPlayerVelocity(velocity);
            events.Add("Jumped.");
            SimulateSteps(Vector3.zero, LongActionSteps);
        }

        private void Interact()
        {
            if (doorOpen)
            {
                events.Add("The door is already open.");
                return;
            }

            if (Vector3.Distance(playerBody.position, DoorPosition) > InteractionDistance)
            {
                events.Add("There is nothing to interact with here.");
                return;
            }

            if (!hasKey)
            {
                events.Add("The door is locked; you need the key.");
                return;
            }

            doorOpen = true;
            if (launchOptions.Defect != PhysicsKeyDoorDefect.KeyNotConsumed)
            {
                hasKey = false;
            }

            ApplyDoorPresentation();
            Physics.SyncTransforms();
            events.Add("Unlocked the door and used the key.");
        }

        private void SimulateSteps(Vector3 horizontalVelocity, int steps)
        {
            Physics.SyncTransforms();
            for (var step = 0; step < steps; step++)
            {
                var velocity = GetPlayerVelocity();
                velocity.x = horizontalVelocity.x;
                velocity.z = horizontalVelocity.z;
                SetPlayerVelocity(velocity);
                playerBody.WakeUp();
                Physics.Simulate(FixedStepSeconds);

                if (playerBody.position.y < -7f)
                {
                    pendingFall = true;
                }

                if (ProcessPendingTriggers() || won)
                {
                    break;
                }
            }

            var stoppedVelocity = GetPlayerVelocity();
            stoppedVelocity.x = 0f;
            stoppedVelocity.z = 0f;
            SetPlayerVelocity(stoppedVelocity);
        }

        private bool ProcessPendingTriggers()
        {
            if (pendingFall)
            {
                pendingFall = false;
                pendingKey = false;
                pendingExit = false;
                if (launchOptions.Defect != PhysicsKeyDoorDefect.RespawnLeak)
                {
                    RestoreLevelProgress();
                    ApplyKeyPresentation();
                    ApplyDoorPresentation();
                }

                RespawnPlayer();
                events.Add("You fell.");
                return true;
            }

            if (pendingKey && keyPresent)
            {
                pendingKey = false;
                keyPresent = false;
                hasKey = true;
                ApplyKeyPresentation();
                events.Add("Picked up the key.");
            }

            if (pendingExit && !won)
            {
                pendingExit = false;
                won = true;
                events.Add("Reached the exit!");
            }

            return false;
        }

        private void RestoreLevelProgress()
        {
            keyPresent = true;
            hasKey = false;
            doorOpen = false;
            won = false;
        }

        private bool IsGrounded()
        {
            return Physics.Raycast(
                playerBody.position,
                Vector3.down,
                1.15f,
                1 << EnvironmentLayer,
                QueryTriggerInteraction.Ignore);
        }

        private Vector3 GetPlayerVelocity()
        {
#if UNITY_6000_0_OR_NEWER
            return playerBody.linearVelocity;
#else
            return playerBody.velocity;
#endif
        }

        private void SetPlayerVelocity(Vector3 velocity)
        {
#if UNITY_6000_0_OR_NEWER
            playerBody.linearVelocity = velocity;
#else
            playerBody.velocity = velocity;
#endif
        }

        private static AutoPlayQAEntity CreateSolidObservation(LevelPart part)
        {
            return new AutoPlayQAEntity
            {
                Id = part.Id,
                Kind = part.Kind,
                Position = ToArray(part.Position),
                Size = ToArray(part.Size),
                State = new[]
                {
                    AutoPlayQAStateEntry.Boolean("solid", true)
                }
            };
        }

        private static bool IsFloorGapCandidate(int xIndex, int zIndex)
        {
            return (xIndex == 1 && zIndex == 2) ||
                (xIndex == 1 && zIndex == 3) ||
                (xIndex == 2 && zIndex == 1) ||
                (xIndex == 2 && zIndex == 4) ||
                (xIndex == 4 && zIndex == 1) ||
                (xIndex == 4 && zIndex == 3) ||
                (xIndex == 5 && zIndex == 2) ||
                (xIndex == 5 && zIndex == 5);
        }

        private static int SelectIndex(int seed, int count, uint salt)
        {
            var value = unchecked((uint)seed) ^ salt;
            value ^= value >> 16;
            value *= 0x7FEB352Du;
            value ^= value >> 15;
            value *= 0x846CA68Bu;
            value ^= value >> 16;
            return (int)(value % (uint)count);
        }

        private static bool ContainsControl(string control)
        {
            for (var index = 0; index < AllControls.Length; index++)
            {
                if (AllControls[index] == control)
                {
                    return true;
                }
            }

            return false;
        }

        private static double[] ToArray(Vector3 value)
        {
            return new[]
            {
                Round2(value.x),
                Round2(value.y),
                Round2(value.z)
            };
        }

        private static double Round2(float value)
        {
            var rounded = Math.Round(value, 2, MidpointRounding.AwayFromZero);
            return rounded == 0.0 ? 0.0 : rounded;
        }

        private void EnsureInitialized()
        {
            if (!initialized)
            {
                throw new InvalidOperationException("PhysicsKeyDoor is not initialized.");
            }
        }

        private sealed class LevelPart
        {
            public string Id;
            public string Kind;
            public Vector3 Position;
            public Vector3 Size;
            public Collider Collider;
        }
    }

    internal sealed class PhysicsKeyDoorTriggerRelay : MonoBehaviour
    {
        private PhysicsKeyDoorGameAdapter owner;

        public void Configure(PhysicsKeyDoorGameAdapter adapter)
        {
            owner = adapter;
        }

        private void OnTriggerEnter(Collider other)
        {
            if (owner != null)
            {
                owner.NotifyTrigger(other);
            }
        }
    }
}
