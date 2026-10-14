# Real-hardware checklist and test plan (Step 14)

Nothing in this document has been flown yet. Every item below is a gate: don't
start a phase until the previous phase's go/no-go criteria are met and logged.

## 1. Hardware

| Item | Choice | Why / notes |
|---|---|---|
| Airframe | 450–500 mm quad, ArduCopter 4.6+ | Same velocity-loop behaviour that sysid measured in SITL |
| Flight controller | Pixhawk-class (H7) | `TELEM2` to the companion: `SERIAL2_PROTOCOL=2`, `SERIAL2_BAUD=921` |
| Companion | Raspberry Pi 5 (8 GB) or Jetson Orin Nano | Runs `python -m lander deploy run`; must hold 10 Hz (budget below) |
| Camera | Global-shutter USB/CSI, ≥ 60° FOV, 640×480 | A rolling shutter skews the marker during manoeuvres; a fixed (rigid) mount like the sim |
| Position source | GPS (outdoor) | GUIDED velocity control needs a position estimate (pre-arm "Need Position Estimate") |
| Rangefinder (recommended) | Downward lidar (e.g. TF-Luna) | Independent height check for the safety floor; not used by the policy |
| Rover | RC car / UGV ≥ 0.8 × 0.8 m deck, or a hand-pushed cart for early phases | Deck must be flat, rigid, matte |
| Kill switch | RC transmitter with a mode switch (LOITER/LAND) and a motor-kill switch | **The pilot's switch always wins** (safety layer goes silent outside GUIDED) |

## 2. Parameters to apply on the flight controller
- `WP_YAW_BEHAVIOR 0`. **Required.** With the default, auto-yaw turns the
  nose toward the direction of travel, which rotates the level frame and makes
  body-frame velocity commands spiral (found by sysid, Step 8).
- `GUID_TIMEOUT 1`. Stop after 1 s without setpoints (default 3 s). The
  runtime streams at 10 Hz.
- `FENCE_ENABLE 1`, `FENCE_TYPE 3` (altitude + circle), `FENCE_RADIUS 50`,
  `FENCE_ALT_MAX 20`. This is the FC-side geofence, independent of ours.
- `FS_GCS_ENABLE` / `FS_OPTIONS` so that losing the companion link triggers
  RTL or LAND.
- `DISARM_DELAY` as default. The motor cut at touchdown is an explicit forced
  disarm from the runtime.
- Re-run `python -m lander sim sysid` *on the real drone* (tethered, then
  free): fit τ, delay, a_max and jerk. If they fall outside Stage A's DR
  ranges (τ ×0.8–1.25, delay ×0.8–1.25), retrain with ranges that cover them.

## 3. Camera calibration (checkerboard)
1. Print a 9×6 inner-corner checkerboard with 25 mm squares on rigid board.
   Measure the printed square size with calipers: printers scale.
2. Capture 25–40 images covering the whole image, including the corners and
   edges, with tilts up to ±45°, at the working resolution (640×480).
3. Run `cv2.calibrateCamera`. Accept if the RMS reprojection error is < 0.5 px.
   Store K and distortion in `configs/camera.yaml` (`CameraModel` takes `k`
   and `dist`).
4. Verify: detect the marker at 1, 2, 4 and 6 m, measured with a tape. Range
   error should be < 3%. A larger error usually means the marker size or the
   print scale is wrong.
5. Measure the mount: lens offset from the CoG (`mount_offset_body_frd_m`) and
   orientation. Check that "image top = nose, image right = right wing" by
   moving a marker by hand. This is the single most common sign error.

## 4. Marker
- DICT_4X4_50 id 0, printed matte (no glare) on rigid board, with a white quiet
  zone ≥ 1 cell wide.
- **Size from the measured detection range:** with 640 px and a 90° FOV,
  detection needs ≥ ~26 px of marker side (74% detection, measured in
  Gazebo). Side_px = f·s/h with f = 320 px, so s ≥ 26·h/320. For 6 m that's
  0.5 m; for 10 m, 0.8 m.
- Real cameras have motion blur and exposure changes, so re-measure the
  detection range outdoors (Phase 2 below).
- To land from higher up, use a nested marker: a large outer marker for
  acquisition plus a small one for the final metre, where the large one
  overfills the FOV (below ~0.5 m, measured).

## 5. Latency budget (10 Hz loop = 100 ms)
| Stage | Sim measurement | Budget on Pi 5 |
|---|---|---|
| Image age at use | 89 ms mean, 165 ms max (Gazebo, 15 Hz camera) | ≤ 150 ms (trained with up to 165 × 1.5) |
| ArUco + PnP (640×480, 1 thread) | 1.14 ms (laptop) | ≤ 15 ms (expect ~5–10× slower) |
| KF + obs | < 0.1 ms | < 1 ms |
| Policy (ONNX, 1 thread) | 0.046 ms mean (laptop) | < 2 ms |
| MAVLink send | async | — |

Pi 5 / Jetson numbers are **not measured yet**. Run `python -m lander deploy
timing` on the board before Phase 1. Overruns are logged by the runtime
(`overruns`, `loop_ms_p99`).

## 6. Test progression (go/no-go per phase)
| Phase | Setup | Go criteria to advance |
|---|---|---|
| 0. Bench | Props off, FC on USB, camera at a printed marker | Runtime at 10 Hz for 10 min, 0 overruns; safety layer goes SILENT on a mode switch and HOVERs when the marker is covered |
| 1. Tethered hover | 2 m tether, marker on the ground | Velocity commands tracked; sysid fit within DR ranges; RC override test passes 5/5 |
| 2. Static marker, 3–5 m | Policy in control, pilot on the switch, `cut_height_m` disabled (pilot lands) | 10/10 descents end aligned < 0.25 m at 1 m height; detection range measured outdoors |
| 3. Static marker, touchdown | Motor cut enabled | 10/10 on-deck landings, xy < 0.2 m |
| 4. Slow rover (0.5 m/s, straight) | Hand-pushed cart | 8/10 on-deck; never a landing outside the deck (the pilot aborts if the estimate diverges) |
| 5. Moving rover (≤ 1.5 m/s, straight, then curves) | RC rover | 8/10 per condition, compared against the PID baseline flown the same day |

**Abort criteria (any phase):**
- the estimate jumps more than 1 m between ticks;
- loop overruns above 1%;
- any safety-layer RTL;
- wind above 5 m/s (the edge of the Stage A DR range).

**Logging:** every tick is written to `results/flight_logs/*.csv` (image age,
detection/policy/loop ms, estimate, command, safety decision with reasons,
mode), plus the FC dataflash log. Fly with both logs, always.
