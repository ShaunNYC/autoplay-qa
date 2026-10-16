# Unity · Floor support

Source: recorded · Status: completed · Seed: 0
Controls: 277 · Resets: 1 · Requests: 25

Saved recording replay. No game controls or model requests are executed by playback.

Lightning exploration was recorded earlier; these Ultra assessments are from the final Step-1 evidence review on October 14, 2026.

Saved elapsed counters cover the recorded model session; the earlier collision sweep was written afterward and has no original per-control timing.

Host signals and AI judgments remain separate. Execution evidence does not prove model claims or intent.

Complete combined pricing data is unavailable for this saved session; no zero-cost claim is made.

## 1. Player sank into a visible floor tile

Host signal: sank_into_visible_floor
Evidence steps: -10
AI: bug · medium

Observed: player drops 0.6 units (exceeds 0.4 threshold) from Y=1.0 to Y=0.4 in one step, gains downward velocity -3.34, loses grounded state while positioned over solid floor-4-1 (top at Y=0.12). Player center at Y=0.4 implies player bottom ~Y=-0.1 (assuming height 1.0), intersecting floor volume. Possible: fall damage, death if over void, or progression block until reset.

At step -11 player stands on floor-4-2 (Z=-2.81) grounded. Step -10 moves south to Z=-4.4, which lies on adjacent floor-4-1 (Z range -5 to -3, solid=true in both before/after states). Despite floor-4-1 being present and solid at the new position, player becomes ungrounded with large downward velocity, sinking into the floor tile. Reset at step -9 recovers. This indicates a collision detection failure at the tile boundary.

Reproduction:

Fresh verification: True · Minimality verified: False
Actions: north → south → south → north → east → west → west → south → east → south → east → west → west → north → north → east → west → west → east → east → east → north → south → south → east → east → west → west → north → north → north → south → east → west → west → south → south → south → south → west → east → east → east → north → south → south → east → east → west → west → west → north → north → north → north → south → east → west → west → east → east → east → north → south → east → east → west → west → north → north → south → east → west → west → north → south → east → west → west → east → east → east → north → south → east → east → west → west → north → north → east → west → west → north → south → east → west → west → east → east → east → interact → north → south → east → west → west → north → north → south → south → north → east → west → west → north → south → east → west → west → south → east → east → east → east → interact → north → north → west → south → south → east → south → east → west → north → north → north → south → east → east → west → west → north → north → east → west → west → north → east → west → west → south → south → south → east → east → east → east → east → interact → north → south → south → north → east → west → west → west → north → north → north → north → east → east → east → west → south → south → south → east → east → east → south → interact → south → north → east → west → west → west → east → north → north → interact → north → east → west → west → west → south → south → east → south → south → south → east → west → west → west → east → east → north → north → interact → south → north → east → north → north → west → interact → north → south → east → west → west → north → north → north → south → north → east → west → west → west → east → south → south → south → south → south → south → south
Original source steps: -253, -252, -251, -250, -249, -248, -247, -246, -245, -244, -243, -242, -241, -240, -239, -238, -237, -236, -235, -234, -233, -232, -231, -230, -229, -228, -227, -226, -225, -224, -223, -222, -221, -220, -219, -218, -217, -216, -215, -214, -213, -212, -211, -210, -209, -208, -207, -206, -205, -204, -203, -202, -201, -200, -199, -198, -197, -196, -195, -194, -193, -192, -191, -190, -189, -188, -187, -186, -185, -184, -183, -182, -181, -180, -179, -178, -177, -176, -175, -174, -173, -172, -171, -170, -169, -168, -167, -166, -165, -164, -163, -162, -161, -160, -159, -158, -157, -156, -155, -154, -153, -152, -151, -150, -149, -148, -147, -146, -145, -144, -143, -142, -141, -140, -139, -138, -137, -136, -135, -134, -133, -132, -131, -130, -129, -128, -127, -126, -125, -124, -123, -122, -121, -120, -119, -118, -117, -116, -115, -114, -113, -112, -111, -110, -109, -108, -107, -106, -105, -104, -103, -102, -101, -100, -99, -98, -97, -96, -95, -94, -93, -92, -91, -90, -89, -88, -87, -86, -85, -84, -83, -82, -81, -80, -79, -78, -77, -76, -75, -74, -73, -72, -71, -70, -69, -68, -67, -66, -65, -64, -63, -62, -61, -60, -59, -58, -57, -56, -55, -54, -53, -52, -51, -50, -49, -48, -47, -46, -45, -44, -43, -42, -41, -40, -39, -38, -37, -36, -35, -34, -33, -32, -31, -30, -29, -28, -27, -26, -25, -24, -23, -22, -21, -20, -19, -18, -17, -16, -15, -14, -13, -12, -11, -10

Limitation: Single occurrence in automated sweep; no fresh replay verification (reproduction.verified=null).

Limitation: Unknown if consistent at all tile boundaries or only this seam.

Limitation: Player height and collider dimensions inferred from candidate, not directly observed in evidence.
