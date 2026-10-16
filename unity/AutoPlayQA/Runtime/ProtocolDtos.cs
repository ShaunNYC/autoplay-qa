using System;

namespace AutoPlayQA
{
    public static class AutoPlayQAProtocol
    {
        public const string Version = "autoplay-qa/1";
        public const int MaxLineBytes = 1024 * 1024;
    }

    [Serializable]
    public sealed class AutoPlayQACapabilities
    {
        public bool Grid = true;
        public bool Legend = true;
        public bool Entities = true;
        public bool Reset = true;
        public bool Seed = true;
        public bool DefectInjection;

        // Optional version-1 extensions. They remain false by default so
        // existing adapters retain their original handshake shape.
        public bool Continuous;
        public bool SolidsVisible;
    }

    [Serializable]
    public sealed class AutoPlayQAGameInfo
    {
        public string Id = "unity-game";
        public string Version = "1";
    }

    [Serializable]
    public sealed class AutoPlayQAObservation
    {
        public int Tick;
        public string Status = "playing";
        public AutoPlayQAPlayer Player = new AutoPlayQAPlayer();
        public AutoPlayQAEntity[] Entities = new AutoPlayQAEntity[0];
        public string[] Events = new string[0];
        public string[] Controls = new string[0];

        // Grid and Legend are omitted from the wire response when the
        // corresponding capability is false or the field is null.
        public string[] Grid;
        public AutoPlayQALegendEntry[] Legend;
    }

    [Serializable]
    public sealed class AutoPlayQAPlayer
    {
        // Null means that this game does not expose a player position.
        public double[] Position;

        // Optional continuous-world state. Nullable Grounded lets legacy
        // adapters omit the field instead of changing false into a claim.
        public double[] Velocity;
        public bool? Grounded;
        public string[] Inventory = new string[0];
    }

    [Serializable]
    public sealed class AutoPlayQAEntity
    {
        public string Id;
        public string Kind;

        // Null means that this entity has no currently visible position.
        public double[] Position;

        // Optional full axis-aligned dimensions centered at Position.
        public double[] Size;
        public AutoPlayQAStateEntry[] State = new AutoPlayQAStateEntry[0];
    }

    [Serializable]
    public sealed class AutoPlayQALegendEntry
    {
        public string Glyph;
        public string Meaning;

        public AutoPlayQALegendEntry()
        {
        }

        public AutoPlayQALegendEntry(string glyph, string meaning)
        {
            Glyph = glyph;
            Meaning = meaning;
        }
    }

    public enum AutoPlayQAValueKind
    {
        Null,
        String,
        Boolean,
        Number
    }

    /// <summary>
    /// One property in an entity's JSON state object. The tagged representation
    /// avoids dictionaries and third-party JSON dependencies in Unity.
    /// </summary>
    [Serializable]
    public sealed class AutoPlayQAStateEntry
    {
        public string Key;
        public AutoPlayQAValueKind Kind;
        public string StringValue;
        public bool BooleanValue;
        public double NumberValue;

        public static AutoPlayQAStateEntry Null(string key)
        {
            return new AutoPlayQAStateEntry { Key = key, Kind = AutoPlayQAValueKind.Null };
        }

        public static AutoPlayQAStateEntry String(string key, string value)
        {
            return new AutoPlayQAStateEntry
            {
                Key = key,
                Kind = value == null ? AutoPlayQAValueKind.Null : AutoPlayQAValueKind.String,
                StringValue = value
            };
        }

        public static AutoPlayQAStateEntry Boolean(string key, bool value)
        {
            return new AutoPlayQAStateEntry
            {
                Key = key,
                Kind = AutoPlayQAValueKind.Boolean,
                BooleanValue = value
            };
        }

        public static AutoPlayQAStateEntry Number(string key, double value)
        {
            return new AutoPlayQAStateEntry
            {
                Key = key,
                Kind = AutoPlayQAValueKind.Number,
                NumberValue = value
            };
        }
    }
}
