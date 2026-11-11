# Ball alignment audit

Expert saved env.pkl is used by both expert baseline and projected policy. No
candidate-specific ball overrides. Current runtime is not a complete historical
source checkout, so this is not proof of bitwise historical dynamics equivalence.

| Setting | Saved configuration | Observed runtime (16 environments) |
|---|---|---|
| Radius | USD sphere 0.1155 m | Same USD asset loaded |
| Mass | nominal0.45 kg; startup randomization0.40–0.50 | 0.40006–0.49962 kg |
| Static friction | 0.08–0.24 | 0.09129–0.23598 |
| Dynamic friction | 0.05–0.16 | 0.05356–0.14412 |
| Restitution | base0; randomized0.40–0.65 | 0.40270–0.64304 |
| Constant rolling resistance | 0.05–0.15 N | 0.05109–0.14498 N |

Launch x3.5–6m, y±1m; sampled receive point x±0.2m,y±0.8m;
configured incoming speed1.5–3.5m/s with point-trajectory sampling and minimum
arrival time1.1s. Therefore heading and speed are not necessarily independently
sampled from the fallback heading range. Launch delay0.75–5s, with prelaunch
training branch probability0.25 and delay5–8s. Target heading±20deg,
target speed2–3m/s, target distance3.2m. These are saved configuration fields;
full distributions across launches have not yet been empirically audited.

Runtime evidence: runs/ball_audit_v1/ball_runtime_audit.json. The runner now saves
this audit for subsequent runs. Prior runs lack material snapshots, so matching
initial-state hashes alone do not prove identical randomized physical parameters.

Hold ablation completed: first-ball kick8/16, direction<=10deg2/16,
early termination14/16. Does not demonstrate improvement over SLERP control.
Next paired-parent experiment retains SLERP but increases optimization passes
from5 to20,200k transitions from serial3, frozen actor/expert. Runtime budget
will increase; this explicitly deviates from paper optimization-pass count.
