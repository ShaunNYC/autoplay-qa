# Unity · Open-door movement

Source: recorded · Status: completed · Seed: 0
Controls: 178 · Resets: 1 · Requests: 1

Saved recording replay. No game controls or model requests are executed by playback.

The game actions in this saved session came from a scripted benchmark; Ultra reviewed the saved evidence separately.

The original scripted recording did not capture wall-clock timing; its elapsed counter is unavailable.

Host signals and AI judgments remain separate. Execution evidence does not prove model claims or intent.

Complete combined pricing data is unavailable for this saved session; no zero-cost claim is made.

## 1. Movement blocked with no visible obstacle

Host signal: blocked_no_visible_obstacle
Evidence steps: 160
AI: bug · high

Player at [-0.75, 1, 1.85] cannot move east through an open doorway (door at x=0, z=1–3, solid=false), blocking progression toward the exit at x=4. No visible solid entity intersects the player's forward path in the provided entity lists.

Step 159 moves player east 1.1 units to [-0.75, 1, 1.85]. Step 160 attempts another east move from that position but yields zero displacement. The step-160 before-state lists the door at [0, 1.25, 2] size [0.5, 2.5, 2] with state open=true, solid=false, spanning z≈1–3 — the player's z=1.85 lies within this opening. Adjacent walls (wall-interior-3 at z=0, wall-interior-4 at z=4) are outside the player's z. Floors are at y=-0.13, below the player. No feedback or solid obstacle appears in the included entities. The reset at step 161 confirms the session can restart but does not resolve the blockage.

Reproduction:

Fresh verification: True · Minimality verified: False
Actions: north → south → south → north → east → west → west → south → east → south → east → west → west → north → north → east → west → west → east → east → east → north → south → south → east → east → west → west → north → north → north → south → east → west → west → south → south → south → south → west → east → east → east → north → south → south → east → east → west → west → west → north → north → north → north → south → east → west → west → east → east → east → north → south → east → east → west → west → north → north → south → east → west → west → north → south → east → west → west → east → east → east → north → south → east → east → west → west → north → north → east → west → west → north → south → east → west → west → east → east → east → interact → north → south → east → west → west → north → north → south → south → north → east → west → west → north → south → east → west → west → south → east → east → east → east → interact → north → north → west → south → south → east → south → east → west → north → north → north → south → east → east → west → west → north → north → east → west → west → north → east → west → west → south → south → south → east → east → east → east → east
Original source steps: 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160

Limitation: Player collision geometry and exact movement step size are unknown.

Limitation: Forty-eight entities are omitted from the step-160 view; however, the initial observation shows no other entity at the door location.

Limitation: No feedback message explains the blockage.

Limitation: Only three consecutive steps are provided; earlier approach path is not shown.
