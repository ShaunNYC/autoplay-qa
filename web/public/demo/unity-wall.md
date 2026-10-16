# Unity · Wall overlap

Source: recorded · Status: completed · Seed: 5
Controls: 348 · Resets: 1 · Requests: 25

Saved recording replay. No game controls or model requests are executed by playback.

Lightning exploration was recorded earlier; these Ultra assessments are from the final Step-1 evidence review on October 14, 2026.

Saved elapsed counters cover the recorded model session; the earlier collision sweep was written afterward and has no original per-control timing.

Host signals and AI judgments remain separate. Execution evidence does not prove model claims or intent.

Complete combined pricing data is unavailable for this saved session; no zero-cost claim is made.

## 1. Player passed through a solid wall

Host signal: passed_through_solid
Evidence steps: -151
AI: bug · medium

The player moves through a solid wall (wall-interior-5) without collision, bypassing a solid barrier. Observed in step -150 where the player's x position changes from -0.3 to 1.25, crossing the wall's volume at x=[-0.25,0.25] while the wall's state.solid remains true. This breaks level geometry and could allow unintended progression.

Step -150 shows the player at position [-0.3,1,6.5] before the move and [1.25,1,6.5] after an east action. The wall-interior-5 is at [0,1.25,6] with size [0.5,2.5,2] and state.solid=true, occupying x in [-0.25,0.25]. The player's trajectory passes through this volume with no feedback or collision response. Step -151 shows the player approaching but stopping just outside the wall (x=-0.3), confirming the wall's presence. The pass-through occurs on the subsequent east move.

Reproduction:

Fresh verification: True · Minimality verified: False
Actions: north → south → south → north → east → west → west → south → east → south → east → west → west → north → north → east → west → west → east → east → east → north → south → south → east → east → west → west → north → north → north → south → east → west → west → south → south → south → south → west → east → east → east → north → south → south → east → east → west → west → west → north → north → north → north → south → east → west → west → east → east → east → north → south → east → east → west → west → north → north → south → east → west → west → north → south → east → west → west → east → east → east → north → south → east → east → west → west → north → north → east → west → west → north → south → east → west → west → east → east → east → interact → north → south → east → west → west → north → north → south → south → north → east → west → west → north → south → east → west → west → south → east → east → east → east → interact → north → north → west → south → south → east → south → east → west → north → north → north → south → east → east → west → west → north → north → east → west → west → north → east → west → west → south → south → south → east → east → east → east → east → interact → north → south → south → north → east → west → west → west → north → north → north → north → east
Original source steps: -324, -323, -322, -321, -320, -319, -318, -317, -316, -315, -314, -313, -312, -311, -310, -309, -308, -307, -306, -305, -304, -303, -302, -301, -300, -299, -298, -297, -296, -295, -294, -293, -292, -291, -290, -289, -288, -287, -286, -285, -284, -283, -282, -281, -280, -279, -278, -277, -276, -275, -274, -273, -272, -271, -270, -269, -268, -267, -266, -265, -264, -263, -262, -261, -260, -259, -258, -257, -256, -255, -254, -253, -252, -251, -250, -249, -248, -247, -246, -245, -244, -243, -242, -241, -240, -239, -238, -237, -236, -235, -234, -233, -232, -231, -230, -229, -228, -227, -226, -225, -224, -223, -222, -221, -220, -219, -218, -217, -216, -215, -214, -213, -212, -211, -210, -209, -208, -207, -206, -205, -204, -203, -202, -201, -200, -199, -198, -197, -196, -195, -194, -193, -192, -191, -190, -189, -188, -187, -186, -185, -184, -183, -182, -181, -180, -179, -178, -177, -176, -175, -174, -173, -172, -171, -170, -169, -168, -167, -166, -165, -164, -163, -162, -161, -160, -159, -158, -157, -156, -155, -154, -153, -152, -151

Limitation: Only one occurrence observed in the supplied sweep steps; frequency and conditions not established.

Limitation: Reproduction controls are provided but not independently verified (software reproduction.verified is null).

Limitation: The candidate's actual summary only referenced step -151, but the critical pass-through is in step -150 which is included in the evidence.
