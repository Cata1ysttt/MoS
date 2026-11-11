# Implementation checkpoint

Independent repository created, official AH-BFM browser actor downloaded (SHA256
93b4e4becefb3a9f6096c1316ca287950a114c8e119cf74f311b90f04a20f9da).
No official backward encoder has been obtained. Only action-only projection is available.

Implemented paper-based projector (width 1024, four residual blocks, expansion 4),
sphere projection, SLERP, masked physical-target alignment, and full-LPP fail-closed
semantic-loss interface. Three unit tests pass.

Actor conversion gate FAILED: 3/1000 randomized inputs exceed atol=1e-5, rtol=1e-4;
maximum absolute error 4.11868e-5. Disabling ORT graph optimization did not resolve it.
Double-precision converted execution also failed on a sampled input. Latent directional
finite difference agrees (0.0020340 vs analytic 0.0020113), but this does not override
forward equivalence gate. No converted actor artifact is accepted; no LPP training started.

Expert baseline entrypoint written, attempted in existing batchmpc-isaaclab image
in dedicated runtime container with original repo bind mount and new repo mount.
The environment must still be validated before baseline results can be reported.
No claim of expert parity or successful soccer policy is made.

## Subsequent fixes and real-input audit

- Restored original asset path via container-only symlink /home/user/g1-rollingball.
- Reconstructed archived rewards.py from git blob 3c9729e plus checkpoint's saved diff;
  loaded inside the new runner, without modifying the original repository. Other current
  environment code remains in use, so this is not yet an exact historical replay.
- Expert smoke v5 completed: 4 environments, 100 control steps, strict checkpoint load,
  policy obs 1100, RMA obs100, control dt .02. No completed episodes in this short run;
  it is a load/interface check, NOT a performance baseline.
- 256 real-history inputs: strict tolerance passed in all samples, max error 3.69549e-6.
- Engineering gate now requires strict real-input pass AND randomized max error <5e-5
  (physical joint-target bound <0.01 degrees). Original strict randomized failures remain
  recorded. This is an explicit tolerance revision based on physical units, not a claim
  of exact equality. Closed-loop ONNX/PyTorch equivalence is still outstanding.
- Added five-step on-policy action-only training loop and website-compatible bridge.
  Short training smoke launched; no performance claims until it completes and is evaluated.

Short LPP run lpp_smoke_v3 completed 800 transitions / one collection round / 20
optimizer updates. Mean training joint-target MSE 0.09801835 (not task success).
Saved latest.pt and parameter_audit.json; actor gradients remain absent, projector
parameter change nonzero. SwanLab https://swanlab.cn/@cata1yst/g1-soccer-bfm/runs/zu493uaq
Runner initially reported exit 1 because SystemExit(0) was caught by BaseException;
changed error handler to Exception. Model saving and online logging completed before exit.

## Pilot and evaluation (continued)

16,000-transition pilot completed 20 rounds / 400 optimizer steps; first-round MSE
0.0980 and final-round MSE 0.1537. These use changing on-policy states, so they do not
measure fixed-set regression and do not establish improvement. Projector first-parameter
max change 0.04947, frozen BFM gradients absent.

Added first-episode evaluation with a pre-autoreset hook capturing original task quality
metrics and a physical state trace. Original episode horizon is 40 seconds / 2000 steps.
Expert and projected eval_v1 launched with the same seed and 16 environments. These are
preliminary diagnostics, not the planned 256/1024-episode statistical evaluation.

Fixed training reset handling: after termination within a five-step window, execute stand
latent for the remaining masked steps rather than stale pre-reset latent. Added nonfinite
loss fail-fast. Comprehensive evaluation/video delivery and long training remain pending.

Projected eval_v1 completed: 16/16 first episodes terminated before timeout;
1/16 recorded quality_kick_success, 0/16 direction within 10 degrees. Durations
3.8–15.64 seconds. Termination was not yet broken down into individual causes,
so do not describe all terminations as falls. Added termination-term recording for
future evaluations. Expert counterpart still running when this entry was written.

## Persistent passing objective

Expert first-episode evaluation completed: 16/16 kick events, 15/16 direction <=10deg,
1/16 early termination. LPP has not achieved the objective.
Started lpp_long_v1: resume pilot projector AND optimizer, 64 environments, additional
1M transitions, same 20 optimization minibatches per 50-step rollout. Microbatch64
bounds activation memory without changing minibatch loss weighting. This changes total
updates per sampled transition relative to the 16-env pilot; do not interpret it as a
controlled environment-count ablation. SwanLab run drq186uo. First two rounds finite.
Watcher evaluates immutable snapshots every 256k transitions in 16 diagnostic scenarios.
Full held-out evaluation and visual review are still required for completion.
Killed only stale owned expert_smoke_v4 process after verifying it remained live.

## Representation diagnostic during long training

Frozen actor per-state latent optimization (200 Adam steps, lr .05, 256 expert
states from first 1.28 s) reduced average physical joint-target MSE from 0.046787
to 0.001011. This tests local expressibility on early states only, not passing or
closed-loop control. Raw eight-batch results saved in oracle_data_v1/oracle.json.
Collecting a 20-second expert rollout sampled every .2 s to cover later phases.
Bridge tests now verify joint permutation/affine round-trip, physical clipping,
group-major history layout and reset locality; all five tests pass.

Added physical-trace renderer for first episode only, preserving actual joint order and
first-episode end time; no policy or dynamics rerun. Uses G1 visualization mesh from the
existing local asset (no I-BFM weights) and sphere geometry. Initial render camera could
crop an incoming ball; revised camera fits robot and ball, and radius matches Ball.usda
0.1155m. Video output remains a visualization, not a substitute for task metrics.
Long training confirmed live beyond 108,800 new transitions, first periodic evaluation
still pending at 256k. Full-motion oracle collection confirmed live, no restart.

Full-motion sample collection completed: 4 expert environments, 1000 control steps
(20 s), one sample every 10 steps. Includes autoresets, so this is a sampled state
set, not four uninterrupted successful demonstrations. Oracle fit completed and saved
to runs/oracle_full_result.json. First-episode expert/projected videos generated and
ffprobe-checked; expert env0 is 40s, projected env0 is 8.7s. Both depict recorded
physical states, with no success-based selection.

Detailed per-state oracle fit: mean squared physical-target error 0.001017 rad²;
largest errors are right ankle roll (0.003390), right ankle pitch (0.003004), left
ankle pitch (0.002896). These roughly 3-degree ankle RMS errors may matter for
balance/contact, despite small overall MSE. Per-state optimized latents and errors
saved in oracle_full_detailed.pt for diagnostics, not deployed or counted as skills.
Both video files verified: projected env0 8.7s, expert env0 40s. Training and checkpoint
watcher remain live; no goal completion claimed.

Long-v1 failed with Isaac Sim segmentation fault at 195,200 transitions; last complete
checkpoint is 192,000. No model nonfinite error observed. Terminated the old checkpoint
watcher to prevent concurrent simulator evaluation. Evaluating 192k checkpoint first;
serial continuation driver waits for active simulator to exit, then trains four 200k
segments with optimizer resume and evaluates each in a separate sequential process.
It stops on subprocess failure and records status, never overwrites a run. Removed
periodic faulthandler thread dumps; causal attribution for segfault remains unknown.

Evaluation audit: quality_* metrics can freeze first-ball results; they must not be
reported as episode-wide continuous pass rates. Added actual evaluated/accurate kick
counters from command state to future terminal records. Added seed CLI and provenance
to evaluation summaries for held-out validation. Future training checkpoint metadata
records source/local/cumulative steps and resume path (legacy pilot provenance still
requires adding the preceding 16k separately). Unit tests remain five passing.

192k checkpoint evaluation completed on the same 16 seeds/config: kick 5/16,
<=10deg direction 1/16, early termination 11/16, mean duration19.1775s.
Termination breakdown missed_ball6, anchor_pos4, anchor_ori1, time_out5.
Compared with 16k pilot (kick1/16, direction0/16, terminated16/16), this shows
preliminary improvement but remains far below expert (16/16,15/16,1/16).
Serial stage1 launched successfully, SwanLab qf6pcd6c. Goal remains unachieved.

Added projected-evaluation audit of realized physical targets vs expert targets and
fraction altered by environment clipping, restricted to each first episode. Added a
unit test explicitly exercising the [-10,10] environment command bound; six tests pass.
Longer independent-latent diagnostic: first64 states of full-motion sample set, 2000
Adam steps (vs200). Batch MSE drops to 2.3173e-5 and 1.41314e-4. Thus earlier residual
errors are not established frozen-model hard limits; optimization effort matters.
This remains per-state optimization without temporal consistency, not deployable skill.
Serial1 training live beyond41,600 additional transitions; fixed eval awaits segment end.

Added paired bootstrap comparison utility. Existing16-trial result remains clearly below
expert; old reports lack explicit seed/hash provenance, so this tool labels that limitation.
Future evaluations save initial physical states and SHA256 and reject paired comparisons
when hashes differ. This avoids assuming identical seeds necessarily imply identical scenes.
Serial1 verified alive at96k new transitions with finite losses; no new closed-loop result yet.

Serial1 completed 201,600 additional transitions in535.84s without crash; projector
parameter delta .11212, BFM gradients absent. Automatic serial transition to eval_serial_1
confirmed live (PID36557). Evaluator progressed through400 control steps; do not infer
full-round quality until it finishes. Driver592660 remains live. This turn verified
real process advancement and successful completion/switch, not merely stale log presence.

Serial1 evaluation: first-ball kick4/16, direction0/16, early termination12/16;
11 missed_ball and1 wrong_foot_kick,4 timeout. Across episode cumulative counters:
6 kicks,5 evaluated direction events,0 accurate. No continued improvement versus192k.
Serial2 already started as unchanged-budget continuation. Future evaluation now stores
10Hz expert/realized targets, z, projector input, BFM inputs and ball/root positions to
localize waiting vs swing errors and support supervised contact-phase diagnostics.

Important contract finding: mean across first episodes of target-clipped step fraction
is0.45965 (~46%), despite small threshold1e-5. Magnitude is not yet logged, so frequency
alone cannot establish causal severity. Added clipping_target_mse and target snapshots
for next evaluation. Current training compares nominal BFM targets, while physical
execution clips them; investigate before attributing plateau solely to data/time.

## Action execution diagnostic extension

Serial2 remains live (PID 636518 verified). Preserve unchanged training as the
training-volume comparison. Future evaluations now save effective clipped targets
alongside nominal BFM and teacher targets. `scripts/analyze_action_errors.py RUN`
reports per-joint clipping magnitude and imitation error for active first-episode
samples, split by horizontal ball-to-root distance <=0.6 m. This distance is not
claimed to identify contact phase. No loss/control changes made before measuring
the mismatch. Six existing tests pass; diagnostic and evaluator compile.

Serial2 follow-up: live PID 636518 verified at179200 transitions. Input audit of
saved env.yaml confirms 10-frame policy history including proprioception, actions,
ball position/velocity, target direction/speed, and event phase. Missing all history
is therefore not a supported explanation. Serial2 first10 loss mean0.02235 versus
last10 at150400 mean0.01760; this is on-policy distribution-dependent, not a skill
success measure. Evaluation now asserts predicted effective targets match actual
action-term processed targets for nonterminal environments after stepping. This
checks affine/clipping conversion only, not the randomized 0–1 step actuator queue.
Existing six tests pass; runtime assertion awaits serial2 evaluation.

## Serial2 completed: improved contact, direction still poor

16/16 first episodes complete; initial-state SHA256 matches serial1 exactly.
First-ball kick11/16, direction<=10deg2/16, early term12/16. Episode-wide
kick events22, evaluated direction events21, accurate6; these are not independent
first-ball trials. Termination: anchor_pos8, missed_ball3, wrong_foot1, timeout4.
Runtime processed-target assertion passed throughout. Offline 10Hz diagnostics:
3435 active samples, effective-target imitation MSE0.020768, clipping MSE0.000878.
Near ball (root distance<=0.6m)193 samples: imitation0.028184, clipping0.001920.
Largest clipping RMS: left/right ankle pitch0.1035/0.0989rad; imitation RMS
left shoulder pitch0.2343, right/left ankle pitch0.2240/0.2164rad.
Clipping is measurable but cannot alone explain all imitation error. Contact
improved with more unchanged training, so preserve serial3 as the next controlled
comparison (PID676727 verified live). No evidence yet of expert-level accuracy.

## Near-ball expressibility diagnostic on serial2 visited states

193 active first-episode states within0.6m horizontal ball-root distance were
selected without success filtering. Independent z fitting (2000 Adam steps/state,
frozen actor) reduces nominal physical-target MSE from0.030807 to0.00024214.
Recomputed with deployment actor clamp: identical MSE; maximum absolute actor
output0.65752<1 so omitted clamp in the fitting utility did not affect this result.
This is ~127x lower MSE, not a deployable trajectory or success demonstration.
Evidence favors improving projector optimization/temporal selection over concluding
that AH-BFM cannot express those instantaneous expert targets. Artifacts:
`runs/oracle_serial2_near/{source,fit,clamped_verification}.json` and fitted z.
Rendered fixed env0 for serial2, ffprobe duration40s: first-ball kick succeeds,
direction<=10deg fails, timeout. `runs/eval_serial_2/env_0.mp4` is a MuJoCo
visualization of recorded Isaac states, not an independent MuJoCo dynamics test.
Serial3 remains live, verified at54400 transitions during diagnostic.

## Temporal expressibility follow-up prepared

Upcoming evaluations save complete five-control-step windows with prior/endpoint
latents, exact actor inputs, teacher targets and first-episode masks. New
`scripts/fit_window_oracle.py RUN` optimizes one endpoint per window while holding
prior latent and visited states fixed, matching deployment SLERP timing. Uses
uniformly spaced valid windows, starts at actual projector endpoints, and applies
actor output clamp. This remains an offline diagnostic: no dynamics gradients or
closed-loop success claims. It distinguishes instantaneous expressibility from
five-step timing constraints. Runtime analysis pending serial3 evaluation; six
existing tests and compilation pass. Existing serial training unchanged.

## Next optimization ablation prepared

Runner now accepts --optimization-passes (positive integer, default5), recorded
in run config and SwanLab; trainer uses this instead of hardcoded5. No active
job changes and serial4 default remains5. This enables a controlled more-updates
experiment if temporal oracle indicates reachable low-error endpoints. This would
be an explicit deviation from paper optimization count, not claimed reproduction.
Serial2 grouping: anchor_pos8 episodes, mean17.35s, first-ball kicks5, accurate0;
missed_ball3, mean9.57s, kicks1, accurate0; timeout4, kicks4, accurate2; wrong-foot1,
15.44s, kicks1, accurate0. Direction and sustained stability both remain necessary.

## Serial3: no further task improvement

Same initial-state hash as serial2. First-ball kick9/16, direction<=10deg2/16,
early term12/16; all-episode events24 kicks,23 evaluated,4 accurate. Effective
imitation MSE0.02903 (near-ball0.04396), clipping MSE0.002151 (near0.004736).
Thus unchanged additional training is not monotonically improving the skill.
Five-step fixed-state endpoint oracle launched on idle GPU5,128 uniformly sampled
windows,500 Adam steps, actual previous z and actor history. First32 windows:
MSE0.02146 ->0.00777, substantially less reduction than independent per-state z;
full results pending. Serial4 live PID716480, unchanged final scheduled control.

Completed temporal oracle: weighted nominal target MSE0.027898 ->0.010031
(~2.8x improvement) over128 windows. Finite optimization is not proof of a hard
10Hz limit; comparisons with independent-state diagnostics also use different
samples. Next useful ablation should directly compare interpolation versus faster
latent updates on the SAME states/targets, before deciding to change deployment
frequency. Oracle report saved in eval_serial_3/window_oracle.json.

## Matched temporal oracle comparison

Same128 windows,500 Adam updates, actor fixed, starting from policy commands:
SLERP weighted MSE0.010031; 10Hz hold and independent50Hz results saved in
window_oracle_{hold,independent}_500.json. Both improve over SLERP, but independent
fit retains high residual in some batches. Longer2000-update independent oracle
launched on GPU5 to distinguish optimization failure from expressibility. These
are fixed-state fits, not deployable learned controllers; deployment unchanged.
Serial4 verified live at76800 transitions. Script now records mode/iteration in
separate result paths to preserve comparisons.

## Hold execution ablation queued

Shared execute_latent implements SLERP (default) and hold. Training rollout/loss
both use selected mode; checkpoints persist mode; evaluation loads it, defaulting
legacy checkpoints to SLERP. Seven tests and compileall pass. A serial waiting
driver run_hold_ablation.py is launched; it waits for stage4 successful evaluation
AND no simulator process, then trains hold for200k from serial3 (same parent as
SLERP serial4),64env,5 optimization passes, unchanged LR and frozen BFM. Then16env
same-seed evaluation. Outputs lpp_hold_v1/eval_hold_v1. This is an explicit temporal
execution ablation, not the original FIBER interpolation prescription.

## Temporal oracle convergence check

Independent-state endpoint optimization500 ->2000 updates leaves weighted error
essentially unchanged:0.003577 ->0.003572. Random-start500-update run does not
remove the high-error batches (report *_500_random.json). Analytic range check on
same128 windows: every expert target lies inside BFM nominal default +/-1.5*rescale
box; range-only lower bound MSE0.0 (target_range_bound.json). Thus neither a simple
nominal-range violation nor merely more iterations explains these residuals.
Not proof of actor impossibility: nonconvex optimization/history effects remain.
Hold closed-loop ablation remains queued, PID739166 verified; serial4 at192000.

## Serial4 complete; hold ablation running

Same initial-state hash,16 complete episodes: first-ball kick12/16, direction<=10deg
1/16, early term13/16. All-episode27 kicks,26 evaluated,5 accurate. More unchanged
training increases contact but does not reliably improve accurate passing.
Hold ablation automatically launched, verified live PID761285. Actual config:
64env,200k additional transitions, resume serial3, optimization_passes5,
latent_execution hold. The comparison is serial4 vs hold from the same parent,
not hold versus an earlier checkpoint with a different training budget.

## Passing quality beyond contact

Summary now reports speed/direction conditioned on first-ball kick, with denominator
explicit and missing metrics null. Existing environment snapshot metrics, NOT
instantaneous impact velocity. Expert16 kicked trials: mean direction error4.30deg,
forward speed1.113m/s, relative target-speed error0.548. Serial4 12 kicked trials:
101.30deg,0.219m/s,0.917 respectively. Relative speed error defined in current
commands.py as abs(forward_speed-target_speed)/target_speed. Even expert snapshot
speed differs from target; don't impose an arbitrary instantaneous-speed threshold
on this frozen metric. Need comparable recording times for stricter speed audit.
Hold training verified live at19200 transitions; no success claims yet.

## Projector input audit while hold training runs

3286 active serial3 diagnostic samples,1200 input dims, all finite. Across-feature
RMS quantiles0/50/90/99/100% =0/0.690/1.445/2.484/2.508. No obvious gross scale
outlier. Source robot_world_xy_yaw subtracts scene.env_origins, so raw parallel
environment world offsets are not present in that term. This excludes a simple
coordinate-offset explanation, not all observation-distribution issues. Saved
projector_input_audit.json. Hold training remains unchanged and live.
