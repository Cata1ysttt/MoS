# Expert replay and dynamic expressibility experiment

The target remains autonomous accurate passing through frozen AH-BFM. None of the
oracle or offline loss results count as that achievement.

Dense collection: expert_dense_v1, seed810124,16env x2000 control steps. Observed
pre-reset counters contain at least55 kicks and54 direction-accurate events in16
completed episodes. Terminal-step-only events may be absent from this count.

Warm start uses13 environments for training and3 for validation, contiguous five
step windows, reset masks and4x contact-window sampling. Event labels only control
sampling; they are not extra projector inputs. The previous endpoint is predicted
from the previous window observation (detached) or stand after resets. BFM action
history is inferred from expert physical commands, so this remains teacher-state
supervision with a deployment distribution mismatch.

30 epochs completed with lr1e-4, frozen actor exactly unchanged. Selected epoch10
by held-out-environment mean target MSE:0.013810 ->0.010935. Contact-window validation
MSE0.025135 ->0.018768. These are supervised errors, not passing success. SwanLab:
https://swanlab.cn/@cata1yst/g1-soccer-bfm/runs/9alr3rro

Dynamic diagnostic oracle_closed_loop_v1:4 environments, seed810125,40s horizon.
Each50Hz step queries the expert on the actual student state, then performs up to
64 Adam updates on z through the frozen actor; only BFM-produced actions execute.
This is expensive expert-assisted receding action inversion, not autonomous LPP,
not task-reward latent-trajectory search, and not proof of real-time feasibility.
A matched expert baseline is queued. Only a positive result establishes existence
of some executable latent sequence; failure does not prove no sequence exists.

Queued after oracle: matched expert baseline;16env warmstart evaluation on existing
development seed810123;200k on-policy transitions(seed810126) with expert replay;
then16env development evaluation. Replay uses only the13 training environments,
64 expert windows per optimizer update and coefficient1 alongside on-policy loss.
Learning rate resumes1e-4;5 optimization passes,10Hz SLERP; actor and expert frozen.
Final skill claims will require additional unseen seeds and larger evaluations.

## Completed results checked 2026-11-05

Frozen actor with expert-assisted 50Hz latent inversion:4/4 first balls kicked,
3/4 first-ball directions <=10deg,0/4 early terminations,all40s episodes.
Episode-wide18 kicks,13/18 direction-accurate events. Matched expert:3/4 first-ball
kicks and directions,1 early termination,13 kicks/11 accurate episode-wide.
Initial physical-state hashes match; recorded ball masses,material samples,
drag samples and command settings also match. Small sample does NOT establish
superiority. Conditional first-ball forward-speed snapshots: oracle1.205m/s,
expert1.218m/s; direction error7.11deg versus3.25deg. Oracle mean target MSE
0.0001139 with mean60.5 optimizer steps per50Hz control tick; not real-time or
independent of the expert. Supports existence of executable passing commands in
these cases; does not prove10Hz projector capacity or broad task coverage.

Autonomous results on16 development episodes,seed810123:
- Warmstart:13/16 kicks,2/16 accurate first-ball directions,11 early terminations.
- Warmstart +201600 on-policy transitions with replay:14/16 kicks,4/16 accurate,
  7 early terminations; mean duration31.67s. All-episode45 kicks/11 accurate.
  Conditional first-ball forward-speed snapshot0.615m/s,mean angle error49.41deg.

Thus autonomous passing remains substantially below expert accuracy. More replay
is justified by task improvement, but supervised losses alone are not success.
Started serial continuation: current policy and expert on64 fresh scenarios
(seed810127),then200k more replay training(seed810128),then original16 and new64
scenario evaluation. The new64 set becomes a development comparison once used
for decisions; final validation still requires untouched seeds.
