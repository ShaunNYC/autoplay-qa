using System;
using System.Collections.Generic;
using UnityEngine;

namespace AutoPlayQA
{
    /// <summary>
    /// Small deterministic example adapter. Collect the token and walk onto the
    /// exit. It is intentionally self-contained so it can be copied or replaced.
    /// </summary>
    [DisallowMultipleComponent]
    public sealed class SampleGridGameAdapter : MonoBehaviour, IAutoPlayQAGame
    {
        private static readonly string[] AllControls =
        {
            "north", "south", "east", "west", "wait"
        };

        private static readonly string[] BaseGrid =
        {
            "#######",
            "#.....#",
            "#.#.#.#",
            "#....E#",
            "#######"
        };

        private static readonly double[][] TokenPositions =
        {
            new[] { 3.0, 1.0 },
            new[] { 1.0, 3.0 }
        };

        [SerializeField]
        private int initialSeed;

        private readonly List<string> events = new List<string>();
        private double[] playerPosition;
        private double[] tokenPosition;
        private bool tokenPresent;
        private bool hasToken;
        private bool won;
        private bool initialized;
        private int tick;

        public string[] Controls
        {
            get { return (string[])AllControls.Clone(); }
        }

        private void Awake()
        {
            Reset(initialSeed);
        }

        public void Reset(int seed)
        {
            playerPosition = new[] { 1.0, 1.0 };
            var selected = TokenPositions[seed & 1];
            tokenPosition = new[] { selected[0], selected[1] };
            tokenPresent = true;
            hasToken = false;
            won = false;
            tick = 0;
            events.Clear();
            events.Add("Game started. Collect the token and reach the exit.");
            initialized = true;
        }

        public AutoPlayQAObservation Observe()
        {
            EnsureInitialized();
            var token = new AutoPlayQAEntity
            {
                Id = "token",
                Kind = "collectible",
                Position = tokenPresent ? CopyPosition(tokenPosition) : null,
                State = new[]
                {
                    AutoPlayQAStateEntry.Boolean("present", tokenPresent)
                }
            };
            var exit = new AutoPlayQAEntity
            {
                Id = "exit",
                Kind = "goal",
                Position = new[] { 5.0, 3.0 },
                State = new[]
                {
                    AutoPlayQAStateEntry.String("requires", "token")
                }
            };

            return new AutoPlayQAObservation
            {
                Tick = tick,
                Status = won ? "won" : "playing",
                Player = new AutoPlayQAPlayer
                {
                    Position = CopyPosition(playerPosition),
                    Inventory = hasToken ? new[] { "token" } : new string[0]
                },
                Entities = new[] { token, exit },
                Events = events.ToArray(),
                Controls = won ? new string[0] : Controls,
                Grid = RenderGrid(),
                Legend = new[]
                {
                    new AutoPlayQALegendEntry("#", "wall"),
                    new AutoPlayQALegendEntry(".", "floor"),
                    new AutoPlayQALegendEntry("@", "player"),
                    new AutoPlayQALegendEntry("T", "token"),
                    new AutoPlayQALegendEntry("E", "exit")
                }
            };
        }

        public AutoPlayQAObservation Act(string control)
        {
            EnsureInitialized();
            if (won)
            {
                throw new InvalidOperationException("The sample game is already won.");
            }

            if (!ContainsControl(control))
            {
                throw new ArgumentException("Unknown sample-game control.", "control");
            }

            tick++;
            events.Clear();
            if (control == "wait")
            {
                events.Add("Waited one tick.");
                return Observe();
            }

            var deltaX = 0;
            var deltaY = 0;
            switch (control)
            {
                case "north":
                    deltaY = -1;
                    break;
                case "south":
                    deltaY = 1;
                    break;
                case "east":
                    deltaX = 1;
                    break;
                case "west":
                    deltaX = -1;
                    break;
            }

            var targetX = (int)playerPosition[0] + deltaX;
            var targetY = (int)playerPosition[1] + deltaY;
            if (BaseGrid[targetY][targetX] == '#')
            {
                events.Add("Movement blocked by a wall.");
                return Observe();
            }

            playerPosition[0] = targetX;
            playerPosition[1] = targetY;
            events.Add("Moved " + control + ".");
            if (tokenPresent && PositionsEqual(playerPosition, tokenPosition))
            {
                tokenPresent = false;
                hasToken = true;
                events.Add("Collected the token.");
            }

            if (targetX == 5 && targetY == 3)
            {
                if (hasToken)
                {
                    won = true;
                    events.Add("Reached the exit. You won!");
                }
                else
                {
                    events.Add("The exit requires the token.");
                }
            }

            return Observe();
        }

        private string[] RenderGrid()
        {
            var rows = new string[BaseGrid.Length];
            for (var row = 0; row < BaseGrid.Length; row++)
            {
                var cells = BaseGrid[row].ToCharArray();
                if (tokenPresent && tokenPosition[1] == row)
                {
                    cells[(int)tokenPosition[0]] = 'T';
                }

                if (playerPosition[1] == row)
                {
                    cells[(int)playerPosition[0]] = '@';
                }

                rows[row] = new string(cells);
            }

            return rows;
        }

        private void EnsureInitialized()
        {
            if (!initialized)
            {
                Reset(initialSeed);
            }
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

        private static double[] CopyPosition(double[] position)
        {
            return new[] { position[0], position[1] };
        }

        private static bool PositionsEqual(double[] left, double[] right)
        {
            return left[0] == right[0] && left[1] == right[1];
        }
    }
}
