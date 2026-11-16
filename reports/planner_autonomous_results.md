# Unified planner autonomous results

| Model/task | Episodes | Kick | Direction ≤10° | Termination terms |
|---|---:|---:|---:|---|
| Original pass s1 | 16 | 100.0% | 50.0% | {'time_out': 16} |
| Unified v1 shoot | 16 | 0.0% | 0.0% | {'wrong_foot_kick': 3, 'time_out': 13} |
| Reset-balanced shoot | 16 | 6.2% | 0.0% | {'event_kick_elapsed': 1, 'wrong_foot_kick': 2, 'time_out': 13} |

All runs use development seed 811314; results are not unseen-seed validation. Assisted training rollouts and incomplete smoke runs are excluded. Normal post-kick termination is not a fall. A single checkpoint must perform both tasks; this table does not claim no-reset skill switching.
