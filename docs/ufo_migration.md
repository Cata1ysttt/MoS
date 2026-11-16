# UFO replacement experiment

Requested: replace AH-BFM with RoboParty UFO and test projected soccer passing.
Status (2026-11-11): bridge and pilot passing training implemented; original-PD standalone command gate remains failed. See `reports/ufo_standalone_20261111.md`. Do not report parity.

- Upstream clone: `/home/jiongye/UFO`, commit `4f1a6f6d31a8e62fb0baef6c7ad0b110d7c3fb82`.
- Source: https://github.com/Roboparty/UFO
- Paired release: https://huggingface.co/xuewang/ufo-g1-policy
- License: CC BY-NC 4.0, non-commercial research; not unrestricted commercial open source.
- Release has actor, backward encoder, full model weights; no optimizer state or replay buffer.
- `scripts/download_ufo.py` verifies downloaded files against official release SHA256. `--full` adds full checkpoint.
- `scripts/verify_ufo_exports.py` compares ONNX Runtime with converted frozen PyTorch exports, including input-gradient check. Optional trailing empty Clip input is removed in-memory for converter compatibility, original file retained.
- Audit: local `runs/ufo_export_audit.json`; synthetic only, real-state parity remains required.

## Contract differences

Actor takes one concatenated721D input: state64, previous action29, history372, latent256; confirm exact packing against official exporter before integration. History has4 frames, newest first, excludes current state. AH-BFM bridge uses9 frames and must not be reused unchanged.
Backward takes state64 + privileged full-body state463 and outputs256D latent norm16. Privileged state requires official heading-frame position/rotation/velocity construction, robot body ordering and virtual motion extension (head) handling.
Actions use native default pose, rescale5, per-joint scales and native PD. Source configs differ from AH-BFM. Need preserve expert teacher dynamics or explicitly define changed-actuator comparison; never silently change PD and call it same-environment parity.

## Next gates

1. Implement UFO observation/history/physical-target bridge and verify against upstream code on real states.
2. Verify virtual-body kinematics and463D encoder input; no invented zero-filled features.
3. Run stand/tracking sanity checks and measure physical-target clipping under existing expert environment.
4. Freeze actor and encoder; new projector (do not reuse AH-BFM projector weights) with action alignment + semantic loss, ablate semantic weight0 vs0.1.
5. Matched-seed autonomous evaluation with first-ball direction/speed, terminations, complete traces and videos. Maintain s1 baseline separately.

## Integration update

Full checkpoint download completed and SHA256 verified. Implemented UFOBridge, paired encoder loss and separate training entry `--ufo-train`; AH-BFM remains default. Full-body transform agrees with upstream `compute_humanoid_observations_max` on random31-body states (max error9.54e-7). Real simulation smoke v1 constructed environment and bridge, but stopped before training because container lacked onnxruntime; installing into existing runtime. No success claim.

Pilot driver: smoke ->64k teacher-executed warm-start ->201600 autonomous on-policy transitions ->64-scene autonomous evaluation seed810127. This is exploratory, not equal historical training-budget comparison. Semantic loss uses sum of squared256D latent errors with weight0.1 as paper equation, while action loss is mean29D physical target error. Record both to detect domination. Preserves expert PD/dynamics, explicitly different from UFO native PD. Simulator transfer may be limiting and must be audited before attributing failure to representation.

## Native runtime and measured actuator audit

Native official mjlab tracking motion7 completed500 steps (10s) without early termination; video `runs/ufo_native_tracking/tracking_inference/tracking_7.mp4`. Uses source runtime and checkpoint, no DR/observation noise. This is one motion, not a soccer result. Existing booster_mjlab Python3.11 environment reused (upstream lock requests3.10, so not exact locked reproduction). EGL rendering failed; OSMesa rendering succeeded, same GPU dynamics.

Measured Isaac runtime actuator audit: `runs/ufo_dynamics_audit_v1/dynamics_audit.json`. Environment0 waist yaw/roll/pitch stiffness41.41/28.73/29.45 versus UFO native300/300/300; hip pitch41.29/44.05 versus99.10/99.10. Damping also differs. These are actual initialized randomized gains, not merely source config guesses. All expert actuators are implicit. This is a substantial known contract mismatch; does not alone prove causality. Previous passing results cannot establish model inferiority.

Next necessary controlled experiment: identical UFO reference/state/latent and observation construction in Isaac, compare existing gains versus native gains, before any renewed soccer optimization. Changing native gains in expert environment also changes expert behavior; rerun teacher under each dynamics setting rather than using old expert scores. Keep old trained projectors separate; do not claim fixing PD rescues old checkpoint without testing.

## PD-only paired diagnosis launched

`scripts/run_ufo_pd_diagnosis.py`:16 environments seed810221,2000ticks each. Expert and already-trained UFO projector each evaluated with original vs native UFO PD. All other physics unchanged; this is not full native-dynamics reproduction. Override reapplied after each reset and implicit actuator gain tensors synchronized; runtime audit saved. Status `runs/ufo_pd_status.json`.

This evaluates sensitivity of existing policies, not fresh retraining or identical-reference cross-engine tracking. If old UFO policy stays poor, cannot conclude PD irrelevant because its weights were trained on mismatched dynamics. Teacher result under native PD determines whether that teacher can provide reliable supervision there. Original matching-reference diagnostic remains outstanding.

Reference preparation audit: exported official motion7 contains6574 frames and6573 latent commands; repeated latent extraction equals prior native rollout exactly (max absolute difference0). Official initial state is reference frame0, latent0 is inferred from frame1. `scripts/prepare_ufo_reference.py` makes this offset explicit and converts root quaternion xyzw to Isaac wxyz, preserving joint ordering and reference velocities. Artifact `runs/ufo_reference_export/isaac_reference.npz`; all source reference tensors finite. This only validates source preparation, not cross-engine rollout equivalence.

PD diagnosis interim: expert original PD16/16 complete; first-ball kick16/16, direction14/16, early termination1. Native-PD expert run started; pd_override.json confirms all3 waist stiffness300 for16 lanes. At200 ticks10 first episodes ended; full results/reasons pending, so no success claim. Summary metadata incorrectly labels an expert-only run backbone='ah_bfm'; actual policy is expert (latent_interval=null, no projector checkpoint). Interpret the arm by launch command, not that field; will correct metadata for future runs.

2026-11-11 completed update: all four PD diagnosis arms completed. Expert native-PD kick6/16, direction2/16, terminated14/16; UFO old projector kick0/16 under both PD settings. Standalone official commands now tested without projector: Isaac original PD + native UFO targets tracks motion7 for10s without falls16/16, joint RMSE mean0.2601rad; walking survives10/16. Native mjlab single-lane checks survive both commands, tracking RMSE0.1125rad. Released walking latent nominal0.7m/s measures1.1137m/s natively versus1.4381m/s in surviving Isaac lanes. These checks do not validate the goal suite or long horizons. Additional soccer target clipping up to1.5193rad was found in the earlier diagnostic. Full conditions, limitations and next controlled migration steps are in `reports/ufo_standalone_20261111.md`. Soccer retraining deferred until control validation passes.
