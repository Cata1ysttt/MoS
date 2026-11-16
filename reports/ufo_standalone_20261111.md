# UFO standalone command and release audit — 2026-11-11

## Decision

Original Isaac PD does not yet pass the standalone command gate. Do not start
soccer retraining/ablations on the assumption that UFO control is validated.
Some motion tracking survives, but tracking accuracy and forward locomotion
remain impaired. This does not isolate physics-engine error from actuator,
asset, or randomized-parameter differences.

## Official artifacts checked

- https://github.com/Roboparty/UFO — main describes mjlab training.
- GitHub branches API returned `main`, `deploy`, `page`; releases API returned no releases.
- https://huggingface.co/xuewang/ufo-g1-policy — API revision
  `da3727034a058eae588a16af142764ee34bdada4`; one G1 model package, ONNX exports,
  reference/goal/reward latents. Author/model search returned this UFO package.
- https://github.com/Roboparty/UFO/tree/deploy — deployment sim2sim uses MuJoCo;
  it is not an Isaac Sim validation or an Isaac-trained checkpoint.
- No Isaac Sim / Isaac Lab UFO checkpoint was found in these public channels.
  This is a scoped search result, not proof that none exists privately or historically.
- Roboparty's separate `roboparty_train` Isaac Lab repository is an RPO locomotion
  workspace; it must not be treated as a UFO G1 weight release.

## Experiment

`scripts/check_ufo_standalone.py`: frozen released actor, official exported
motion-7 latents and released `move-ego-0-0.7` latent. No soccer projector,
expert action, reward, task termination, periodic command update, or automatic
reset. Ball moved away. Original initialized Isaac robot gains and other robot
physics retained, including startup randomization; observations use the clean
UFO bridge. Control dt 0.02s. 16 lanes, seed 810311, 500 steps / 10s per command.

Tracking reset is official reference frame 0; latent 0 targets reference frame 1.
Walking reset is the release's default standing pose, height 0.8m, identity
orientation and zero velocity. Fall gate: root below 0.4m, tilt above 60 degrees,
or nonfinite position. This gate is for these upright tasks, not arbitrary
lying-down/get-up goal evaluation.

Authoritative Isaac result: `runs/ufo_standalone_original_pd_v2/summary.json`.
`contract.json` records all lane gains; per-step traces record positions,
velocities, joints, tilt and target clipping. v2 directly applies UFO native
target mapping to existing Isaac actuators, with no additional soccer action
limits. No model weights were modified.

`scripts/check_ufo_native_commands.py` uses the official mjlab environment and
checkpoint model with the same latent arrays and resets, DR/noise disabled,
one environment, 500 steps. This is a deterministic-style sanity baseline,
not a matched 16-lane randomized statistical comparison. Existing Python runtime
is reused; rendering imports require both MUJOCO_GL and PYOPENGL_PLATFORM=osmesa.
Results: `runs/ufo_standalone_native_v2/summary.json`.

| Metric | mjlab native baseline | Isaac original PD, native targets |
|---|---:|---:|
| Tracking upright for 10s | 1/1 | 16/16 |
| Tracking joint RMSE | 0.1125 rad | mean 0.2601 rad |
| Forward walking upright for 10s | 1/1 | 10/16 |
| Forward velocity, last 5s | 1.1137 m/s | 1.4381 m/s, surviving lanes only |

Isaac walking failures first cross the fall gate at 2.42–3.08s. The released
reward key says 0.7m/s, but the native measured speed is already 1.11m/s in this
protocol. Do not attribute the entire discrepancy from the label to sim2sim.
Survivor-only velocity is explicitly selected and must not be presented as
the performance of all 16 lanes. Tracking survival alone is not tracking success.
No goal-reaching command suite has been validated by this test.

The earlier v1 probe applied soccer target limits and started walking from the
motion-7 initial state. Tracking had maximum additional target clipping of
1.5193rad. Its walking results must not be compared causally with v2: both
the action path and walking initialization changed. v1 is diagnostic only.

## Migration work before soccer

1. Keep the above original-PD result as baseline. Use the same commands and
   resets for every next arm; first disable DR on both sides to isolate control,
   then restore and separately assess randomization robustness.
2. Preserve native UFO action scale/default pose, history timing, previous action
   convention, coordinate/joint ordering, 50Hz control and reference frame offset.
   The additional soccer action clipping is an identified interface difference;
   test it as a separate matched arm rather than silently retaining it.
3. Test native gains alone, then matched actuator behavior. Current Isaac uses
   implicit actuators; UFO training uses DC-motor actuator configuration and the
   deployment bridge explicitly calculates PD torque and caps it. Equal Kp/Kd
   does not establish equal control behavior. Match motor torque/velocity limits,
   armature, friction, delays, physics rate and torque saturation.
   Sources: `UFO/ufo/envs/actuators.py`, deploy `sim_env/utils/simulation_bridge.py`.
4. Compare robot mass/inertia, joint frames, collision geometry, foot contacts
   and ground friction; compare stepwise state/torque traces to identify where
   divergence starts. Change one factor at a time and use native rollout as reference.
5. Expand passing control checks to more tracking clips, the released goal suite,
   longer horizons and multiple seeds. Define success by command accuracy plus
   stability, not just lack of task termination.
6. If matching control/physics remains insufficient, consider target-simulator
   motor-policy adaptation while retaining broad native behaviors. Treat this as
   a separately trained model, not a claim of zero-shot transfer. Re-evaluate
   the teacher if its dynamics change; do not assume old expert actions remain valid.
7. Only after the control gate passes: fresh projector; matched budget, replay
   and optimizer update schedule; semantic weight 0 / small / 0.1; then paired
   autonomous soccer evaluations. Current old-projector failures cannot settle
   these ablations.
