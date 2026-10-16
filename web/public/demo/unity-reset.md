# Unity · Reset comparison

Source: recorded · Status: completed · Seed: 0
Controls: 336 · Resets: 1 · Requests: 1

Saved recording replay. No game controls or model requests are executed by playback.

The game actions in this saved session came from a scripted benchmark; Ultra reviewed the saved evidence separately.

The original scripted recording did not capture wall-clock timing; its elapsed counter is unavailable.

Host signals and AI judgments remain separate. Execution evidence does not prove model claims or intent.

Complete combined pricing data is unavailable for this saved session; no zero-cost claim is made.

## 1. Reset did not restore the initial state

Host signal: reset_inconsistency
Evidence steps: 336
AI: bug · high

The key-door puzzle is broken at game start. The door begins open (solid: false, open: true) and the key is already missing (present: false), making the puzzle trivially solvable or impossible depending on exit reachability. The game definition (expected) specifies door closed + key present at tick 0.

The 'expected' observation defines the canonical initial state: door at [0,1.25,2] closed (open: false, solid: true) and key at [-4,0.5,2] present: true. Both the 'actual' observation at tick 0 and the 'initial_observation' show door open (open: true, solid: false) and key missing (present: false). The 336-action replay ends with a reset that returns to the same broken initial state (door open, key missing). The puzzle is unsolvable as designed because the key-door logic is bypassed at initialization.

Reproduction:

Fresh verification: True · Minimality verified: False
Actions: north → south → south → north → east → west → west → south → east → south → east → west → west → north → north → east → west → west → east → east → east → north → south → south → east → east → west → west → north → north → north → south → east → west → west → south → south → south → south → west → east → east → east → north → south → south → east → east → west → west → west → north → north → north → north → south → east → west → west → east → east → east → north → south → east → east → west → west → north → north → south → east → west → west → north → south → east → west → west → east → east → east → north → south → east → east → west → west → north → north → east → west → west → north → south → east → west → west → east → east → east → interact → north → south → east → west → west → north → north → south → south → north → east → west → west → north → south → east → west → west → south → east → east → east → east → interact → north → north → west → south → south → east → south → east → west → north → north → north → south → east → east → west → west → north → north → east → west → west → north → east → west → west → south → south → south → east → east → east → east → east → interact → north → south → south → north → east → west → west → west → north → north → north → north → east → east → east → west → south → south → south → east → east → east → south → interact → south → north → east → west → west → west → east → north → north → interact → north → east → west → west → west → south → south → east → south → south → south → east → west → west → west → east → east → north → north → interact → south → north → east → north → north → west → interact → north → south → east → west → west → north → north → north → south → north → east → west → west → west → east → south → south → south → south → south → south → south → north → south → south → north → east → west → west → west → east → east → north → south → north → east → north → north → interact → south → north → east → west → east → north → north → north → south → east → west → north → north → north → south → north → east → south → south → south → south → south → south → south → south → west → west → west → south → east → west → west → west → east → east → north → south → north → east → north → north → east → west → east → north → north → east → north → north → north → north → east → west → south → west → west → south → south → south → south → south → south → south → east → south → east → east → north → east → west → south → east → south → east → reset
Original source steps: 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255, 256, 257, 258, 259, 260, 261, 262, 263, 264, 265, 266, 267, 268, 269, 270, 271, 272, 273, 274, 275, 276, 277, 278, 279, 280, 281, 282, 283, 284, 285, 286, 287, 288, 289, 290, 291, 292, 293, 294, 295, 296, 297, 298, 299, 300, 301, 302, 303, 304, 305, 306, 307, 308, 309, 310, 311, 312, 313, 314, 315, 316, 317, 318, 319, 320, 321, 322, 323, 324, 325, 326, 327, 328, 329, 330, 331, 332, 333, 334, 335, 336

Limitation: Full 336-step replay not manually inspected step-by-step; analysis focuses on initialization mismatch and reset behavior.
