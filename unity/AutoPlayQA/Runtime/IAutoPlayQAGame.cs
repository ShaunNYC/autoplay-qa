namespace AutoPlayQA
{
    /// <summary>
    /// The stable boundary between a Unity game and the AutoPlay QA transport.
    /// Implementations are called only from Unity's main thread.
    /// </summary>
    public interface IAutoPlayQAGame
    {
        /// <summary>All controls understood by this adapter.</summary>
        string[] Controls { get; }

        /// <summary>Start a fresh deterministic episode for the supplied seed.</summary>
        void Reset(int seed);

        /// <summary>Return the current externally visible game state.</summary>
        AutoPlayQAObservation Observe();

        /// <summary>Apply one control and return the resulting game state.</summary>
        AutoPlayQAObservation Act(string control);
    }
}
