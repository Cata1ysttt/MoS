# Paired fresh-seed planner validation

| Candidate | Seed | Task | Controller | Episodes | Kick | Direction ≤10° |
|---|---:|---|---|---:|---:|---:|
| planner_flow_dagger_r1_fresh | 812801 | pass | planner | 32 | 6.2% | 0.0% |
| planner_flow_dagger_r1_fresh | 812801 | pass | expert | 32 | 96.9% | 84.4% |
| planner_flow_dagger_r1_fresh | 812801 | shoot | planner | 32 | 43.8% | 6.2% |
| planner_flow_dagger_r1_fresh | 812801 | shoot | expert | 32 | 100.0% | 84.4% |
| planner_history_r2_fresh | 812901 | pass | planner | 32 | 53.1% | 3.1% |
| planner_history_r2_fresh | 812901 | pass | expert | 32 | 100.0% | 84.4% |
| planner_history_r2_fresh | 812901 | shoot | planner | 32 | 100.0% | 43.8% |
| planner_history_r2_fresh | 812901 | shoot | expert | 32 | 100.0% | 78.1% |

Candidate `planner_flow_dagger_r1_fresh` checkpoint SHA256: `5a7d25c5ec9ed38ec054ca63f5a79fcfcf13e4cf36dfeba4091d0d4ec9de064b`.
Candidate `planner_history_r2_fresh` checkpoint SHA256: `39e3ef5eb525950a1613921275845536c1445aaf5c28c772fcf375d0b6b12937`.

Only completed paired evaluations are included. The seed was absent from the candidate training metadata. Both skills use the same frozen checkpoint; initial physical states match each expert comparison. Rates count the first episode per environment, not the number of ball events. These evaluations do not establish continuous skill switching.
