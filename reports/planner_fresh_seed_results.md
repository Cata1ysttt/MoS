# Paired fresh-seed planner validation

| Candidate | Seed | Task | Controller | Episodes | Kick | Direction ≤10° |
|---|---:|---|---|---:|---:|---:|
| planner_flow_dagger_r1_fresh | 812801 | pass | planner | 32 | 6.2% | 0.0% |
| planner_flow_dagger_r1_fresh | 812801 | pass | expert | 32 | 96.9% | 84.4% |
| planner_flow_dagger_r1_fresh | 812801 | shoot | planner | 32 | 43.8% | 6.2% |
| planner_flow_dagger_r1_fresh | 812801 | shoot | expert | 32 | 100.0% | 84.4% |

Candidate checkpoint SHA256: `5a7d25c5ec9ed38ec054ca63f5a79fcfcf13e4cf36dfeba4091d0d4ec9de064b`.

Only completed paired evaluations are included. The seed was absent from the candidate training metadata. Both skills use the same frozen checkpoint; initial physical states match each expert comparison. Rates count the first episode per environment, not the number of ball events. These evaluations do not establish continuous skill switching.
