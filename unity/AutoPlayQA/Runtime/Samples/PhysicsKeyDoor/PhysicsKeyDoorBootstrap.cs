using System;
using System.Globalization;
using UnityEngine;

namespace AutoPlayQA.Samples.PhysicsKeyDoor
{
    /// <summary>
    /// Scene entry point for the self-contained PhysicsKeyDoor sample. Launch
    /// configuration is intentionally process-only; injected-defect state is
    /// never serialized into the scene or exposed through the protocol.
    /// </summary>
    [DisallowMultipleComponent]
    public sealed class PhysicsKeyDoorBootstrap : MonoBehaviour
    {
        private GameObject runtimeRoot;

        private void Awake()
        {
            PhysicsKeyDoorLaunchOptions options;
            string error;
            if (!PhysicsKeyDoorLaunchOptions.TryParse(
                Environment.GetCommandLineArgs(),
                out options,
                out error))
            {
                Debug.LogError("AutoPlay QA PhysicsKeyDoor could not start: " + error, this);
#if !UNITY_EDITOR
                Application.Quit(2);
#endif
                enabled = false;
                return;
            }

            Application.runInBackground = true;

            runtimeRoot = new GameObject("AutoPlay QA PhysicsKeyDoor Runtime");
            runtimeRoot.SetActive(false);

            var adapter = runtimeRoot.AddComponent<PhysicsKeyDoorGameAdapter>();
            adapter.ConfigureLaunch(options);

            var server = runtimeRoot.AddComponent<AutoPlayQAServer>();
            server.Configure(
                adapter,
                new AutoPlayQAGameInfo
                {
                    Id = "physics-key-door",
                    Version = "physics-key-door-v1"
                },
                new AutoPlayQACapabilities
                {
                    Grid = false,
                    Legend = false,
                    Entities = true,
                    Reset = true,
                    Seed = true,
                    DefectInjection = true,
                    Continuous = true,
                    SolidsVisible = true
                },
                options.Port);

            runtimeRoot.SetActive(true);
        }

        private void OnDestroy()
        {
            if (runtimeRoot != null)
            {
                Destroy(runtimeRoot);
                runtimeRoot = null;
            }
        }
    }

    internal enum PhysicsKeyDoorDefect
    {
        None,
        FloorGap,
        DoorGhostCollider,
        KeyNotConsumed,
        RespawnLeak,
        WallTunnel
    }

    internal sealed class PhysicsKeyDoorLaunchOptions
    {
        public int Seed;
        public int Port = 8765;
        public PhysicsKeyDoorDefect Defect;

        public static bool TryParse(
            string[] arguments,
            out PhysicsKeyDoorLaunchOptions options,
            out string error)
        {
            options = new PhysicsKeyDoorLaunchOptions();
            error = null;
            var hasSeed = false;
            var hasPort = false;
            var hasDefect = false;

            for (var index = 0; index < arguments.Length; index++)
            {
                string value;
                if (TryReadOption(arguments, ref index, "-autoplayqa-seed", out value))
                {
                    if (hasSeed || !int.TryParse(
                        value,
                        NumberStyles.Integer,
                        CultureInfo.InvariantCulture,
                        out options.Seed))
                    {
                        error = "-autoplayqa-seed requires one 32-bit integer.";
                        return false;
                    }

                    hasSeed = true;
                    continue;
                }

                if (TryReadOption(arguments, ref index, "-autoplayqa-port", out value))
                {
                    if (hasPort || !int.TryParse(
                        value,
                        NumberStyles.Integer,
                        CultureInfo.InvariantCulture,
                        out options.Port) || options.Port < 0 || options.Port > 65535)
                    {
                        error = "-autoplayqa-port requires one integer from 0 through 65535.";
                        return false;
                    }

                    hasPort = true;
                    continue;
                }

                if (TryReadOption(arguments, ref index, "-autoplayqa-defect", out value))
                {
                    if (hasDefect || !TryParseDefect(value, out options.Defect))
                    {
                        error = "-autoplayqa-defect requires one supported defect name.";
                        return false;
                    }

                    hasDefect = true;
                }
            }

            return true;
        }

        private static bool TryReadOption(
            string[] arguments,
            ref int index,
            string name,
            out string value)
        {
            var argument = arguments[index];
            if (argument == name)
            {
                if (index + 1 >= arguments.Length)
                {
                    value = null;
                    return true;
                }

                index++;
                value = arguments[index];
                return true;
            }

            var prefix = name + "=";
            if (argument.StartsWith(prefix, StringComparison.Ordinal))
            {
                value = argument.Substring(prefix.Length);
                return true;
            }

            value = null;
            return false;
        }

        private static bool TryParseDefect(string value, out PhysicsKeyDoorDefect defect)
        {
            switch (value)
            {
                case "floor-gap":
                    defect = PhysicsKeyDoorDefect.FloorGap;
                    return true;
                case "door-ghost-collider":
                    defect = PhysicsKeyDoorDefect.DoorGhostCollider;
                    return true;
                case "key-not-consumed":
                    defect = PhysicsKeyDoorDefect.KeyNotConsumed;
                    return true;
                case "respawn-leak":
                    defect = PhysicsKeyDoorDefect.RespawnLeak;
                    return true;
                case "wall-tunnel":
                    defect = PhysicsKeyDoorDefect.WallTunnel;
                    return true;
                case "none":
                    defect = PhysicsKeyDoorDefect.None;
                    return true;
                default:
                    defect = PhysicsKeyDoorDefect.None;
                    return false;
            }
        }
    }
}
