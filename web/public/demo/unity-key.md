# Unity · Key use feedback

Source: recorded · Status: completed · Seed: 0
Controls: 336 · Resets: 1 · Requests: 1

Saved recording replay. No game controls or model requests are executed by playback.

The game actions in this saved session came from a scripted benchmark; Ultra reviewed the saved evidence separately.

The original scripted recording did not capture wall-clock timing; its elapsed counter is unavailable.

Host signals and AI judgments remain separate. Execution evidence does not prove model claims or intent.

Human review required: key-use feedback and recorded inventory disagree. The AI report's initial-inventory and complete-playthrough assertions are not established by its cited transitions. Its verdict remains tentative.

Complete combined pricing data is unavailable for this saved session; no zero-cost claim is made.

## 1. Key remained in inventory after feedback said it was used

Host signal: map_object_inconsistency
Evidence steps: 126
AI: not_bug · none

none

The replay shows a complete playthrough of the physics-key-door-v1 episode. The player starts with the key already in inventory at position [-4, 1.05, -4], navigates through the level, and successfully uses the 'interact' action at step 126 to unlock the door (event: 'Unlocked the door and used the key'). The door transitions from closed (solid, closed) to open (open: true, solid: false). The key entity shows present: false throughout (already collected). Player inventory retains the key after interaction. The player then proceeds north toward the exit at [4, 0.1, 2]. No bugs, crashes, or logic errors are observed in the replay data.

Human review required: key-use feedback and recorded inventory disagree. The AI report's initial-inventory and complete-playthrough assertions are not established by its cited transitions. Its verdict remains tentative.

Reproduction:

Fresh verification: True · Minimality verified: False
Actions: north → south → south → north → east → west → west → south → east → south → east → west → west → north → north → east → west → west → east → east → east → north → south → south → east → east → west → west → north → north → north → south → east → west → west → south → south → south → south → west → east → east → east → north → south → south → east → east → west → west → west → north → north → north → north → south → east → west → west → east → east → east → north → south → east → east → west → west → north → north → south → east → west → west → north → south → east → west → west → east → east → east → north → south → east → east → west → west → north → north → east → west → west → north → south → east → west → west → east → east → east → interact → north → south → east → west → west → north → north → south → south → north → east → west → west → north → south → east → west → west → south → east → east → east → east → interact
Original source steps: 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126

Limitation: Replay covers episode 0 only; later episodes not shown.

Limitation: Player inventory still shows key after 'used the key' event; unclear if key should be consumed.

Limitation: Exit reached status not shown as true in final state.
