# Unity · Baseline session

Source: recorded · Status: completed · Seed: 1
Controls: 360 · Resets: 1 · Requests: 27

Saved recording replay. No game controls or model requests are executed by playback.

Lightning exploration was recorded earlier; these Ultra assessments are from the final Step-1 evidence review on October 14, 2026.

The host detector raised no issues. Three explicitly authored normal candidates are retained to show the AI review's negative controls.

Saved elapsed counters cover the recorded model session; the earlier collision sweep was written afterward and has no original per-control timing.

Host signals and AI judgments remain separate. Execution evidence does not prove model claims or intent.

Human review required: key-use feedback and recorded inventory disagree. The AI report's initial-inventory and complete-playthrough assertions are not established by its cited transitions. Its verdict remains tentative.

Complete combined pricing data is unavailable for this saved session; no zero-cost claim is made.

## 1. The movement did not succeed; this may be a movement bug

Host signal: suspected_bug
Evidence steps: -311
AI: not_bug · none

None. The player attempted to move east at step -311 from position [-0.75, 1, -4.94] but the position did not change. The geometry shows a solid interior wall (wall-interior-1) at position [0, 1.25, -4] with size [0.5, 2.5, 2], spanning x ∈ [-0.25, 0.25] and z ∈ [-5, -3]. The player at z = -4.94 is within the wall's z-span. At x = -0.75 with a typical player half-width of ~0.5, the player's right edge reaches x = -0.25, exactly contacting the wall's left face. The prior east move at step -312 moved the player from x = -1.85 to x = -0.75 (Δx = +1.1), bringing the player into contact with the wall. The subsequent east input at step -311 correctly produced zero displacement because the solid wall blocks further eastward motion. The subsequent west move at step -310 succeeded (Δx = -1.55), confirming movement away from the wall works. This is expected collision behavior, not a movement defect.

The zero-displacement east move at step -311 is explained by collision with the solid interior wall at x ∈ [-0.25, 0.25], z ∈ [-5, -3]. The player reached the wall's left face at step -312 and could not penetrate it. Westward movement at step -310 succeeded normally. No defect is evidenced.

The original candidate is retained beside the tentative AI judgment for human review.

Reproduction:

Fresh verification: None · Minimality verified: False
Actions: north → south → south → north → east → west → west → south → east → south → east → west → west → north → north → east → west → west → east → east → east → north → south → south → east → east
Original source steps: -336, -335, -334, -333, -332, -331, -330, -329, -328, -327, -326, -325, -324, -323, -322, -321, -320, -319, -318, -317, -316, -315, -314, -313, -312, -311

Limitation: Player collision radius/half-width is not explicitly provided in the evidence; inference from wall contact at x = -0.75 assumes a typical 0.5 half-width.

Limitation: Only three sweep steps (-312, -311, -310) are included; earlier approach to the wall is not visible.

Limitation: No player feedback messages (e.g., 'blocked') are present in the evidence steps.

## 2. Interacting had no effect; this may be a broken interaction

Host signal: suspected_bug
Evidence steps: -235
AI: not_bug · none

The player received appropriate feedback ('There is nothing to interact with here') when pressing interact while ~1.85 units away from the closed door. No progression block or incorrect behavior observed; the player simply moved away afterward.

At step -235 the player is at [-1.85, 1, 2.8] with the closed door at [0, 1.25, 2] (west face at x=-0.25). The ~1.85m horizontal gap likely exceeds the interaction range, so the 'nothing to interact with' feedback is correct behavior. No state change occurred, and the player subsequently moved north away from the door. Without evidence of the intended interaction range or a closer attempt, this is expected feedback, not a defect.

The original candidate is retained beside the tentative AI judgment for human review.

Reproduction:

Fresh verification: None · Minimality verified: False
Actions: north → south → south → north → east → west → west → south → east → south → east → west → west → north → north → east → west → west → east → east → east → north → south → south → east → east → west → west → north → north → north → south → east → west → west → south → south → south → south → west → east → east → east → north → south → south → east → east → west → west → west → north → north → north → north → south → east → west → west → east → east → east → north → south → east → east → west → west → north → north → south → east → west → west → north → south → east → west → west → east → east → east → north → south → east → east → west → west → north → north → east → west → west → north → south → east → west → west → east → east → east → interact
Original source steps: -336, -335, -334, -333, -332, -331, -330, -329, -328, -327, -326, -325, -324, -323, -322, -321, -320, -319, -318, -317, -316, -315, -314, -313, -312, -311, -310, -309, -308, -307, -306, -305, -304, -303, -302, -301, -300, -299, -298, -297, -296, -295, -294, -293, -292, -291, -290, -289, -288, -287, -286, -285, -284, -283, -282, -281, -280, -279, -278, -277, -276, -275, -274, -273, -272, -271, -270, -269, -268, -267, -266, -265, -264, -263, -262, -261, -260, -259, -258, -257, -256, -255, -254, -253, -252, -251, -250, -249, -248, -247, -246, -245, -244, -243, -242, -241, -240, -239, -238, -237, -236, -235

Limitation: Interaction range is not specified in the evidence; the player may have been just outside the trigger distance. No subsequent interact attempt at closer range is recorded.

## 3. The key stays in inventory after walking; perhaps walking should consume it

Host signal: suspected_bug
Evidence steps: -266
AI: not_bug · none

The key remains in inventory after walking steps (-266, -265) following pickup at step -267. No evidence indicates walking should consume the key; it appears to be a persistent inventory item (likely for a door). No player-facing defect is demonstrated.

Evidence steps -267 (pickup with feedback 'Picked up the key.'), -266 (walk south, inventory retains key), and -265 (walk east, inventory retains key) show the key persists in inventory after movement. No evidence indicates walking should consume the key; this appears to be intended persistent inventory behavior for a key-door mechanic. No player-facing defect is demonstrated.

Human review required: key-use feedback and recorded inventory disagree. The AI report's initial-inventory and complete-playthrough assertions are not established by its cited transitions. Its verdict remains tentative.

Reproduction:

Fresh verification: None · Minimality verified: False
Actions: north → south → south → north → east → west → west → south → east → south → east → west → west → north → north → east → west → west → east → east → east → north → south → south → east → east → west → west → north → north → north → south → east → west → west → south → south → south → south → west → east → east → east → north → south → south → east → east → west → west → west → north → north → north → north → south → east → west → west → east → east → east → north → south → east → east → west → west → north → north → south
Original source steps: -336, -335, -334, -333, -332, -331, -330, -329, -328, -327, -326, -325, -324, -323, -322, -321, -320, -319, -318, -317, -316, -315, -314, -313, -312, -311, -310, -309, -308, -307, -306, -305, -304, -303, -302, -301, -300, -299, -298, -297, -296, -295, -294, -293, -292, -291, -290, -289, -288, -287, -286, -285, -284, -283, -282, -281, -280, -279, -278, -277, -276, -275, -274, -273, -272, -271, -270, -269, -268, -267, -266

Limitation: Evidence covers only steps -267 through -265; later door interaction or key consumption not observed.

Limitation: No game specification confirms whether walking should consume keys; judgment based solely on observed behavior in provided steps.
