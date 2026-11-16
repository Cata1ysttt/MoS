# Unified planner autonomous results

| Model/task | Episodes | Kick | Direction ≤10° | Termination terms |
|---|---:|---:|---:|---|
| Original pass s1 | 16 | 100.0% | 50.0% | {'time_out': 16} |
| Unified v1 pass | 16 | 18.8% | 0.0% | {'missed_ball': 14, 'wrong_foot_kick': 1, 'anchor_ori': 1} |
| Unified v1 shoot | 16 | 0.0% | 0.0% | {'wrong_foot_kick': 3, 'time_out': 13} |
| Reset-balanced shoot | 16 | 6.2% | 0.0% | {'event_kick_elapsed': 1, 'wrong_foot_kick': 2, 'time_out': 13} |
| DAgger r1 pass | 16 | 12.5% | 6.2% | {'anchor_pos': 9, 'missed_ball': 6, 'anchor_ori': 1} |
| DAgger r1 shoot | 16 | 0.0% | 0.0% | {'wrong_foot_kick': 5, 'anchor_pos': 7, 'anchor_ori': 4} |
| Unified v2 pass | 16 | 12.5% | 0.0% | {'anchor_pos': 4, 'missed_ball': 9, 'anchor_ori': 1, 'wrong_foot_kick': 1, 'time_out': 1} |
| Unified v2 shoot | 16 | 18.8% | 0.0% | {'wrong_foot_kick': 3, 'event_kick_elapsed': 3, 'anchor_pos': 4, 'anchor_ori': 3, 'time_out': 3} |
| Backward residual pass | 16 | 0.0% | 0.0% | {'anchor_pos': 5, 'wrong_foot_kick': 2, 'missed_ball': 6, 'anchor_ori': 3} |
| Backward residual shoot | 16 | 6.2% | 0.0% | {'wrong_foot_kick': 1, 'anchor_pos': 2, 'event_kick_elapsed': 1, 'time_out': 12} |
| Flow v1 pass | 16 | 0.0% | 0.0% | {'anchor_pos': 6, 'anchor_ori': 3, 'missed_ball': 5, 'wrong_foot_kick': 2} |
| Flow v1 shoot | 16 | 12.5% | 0.0% | {'anchor_ori': 1, 'event_kick_elapsed': 2, 'wrong_foot_kick': 2, 'time_out': 11} |
| Flow wide pass | 16 | 0.0% | 0.0% | {'anchor_pos': 4, 'anchor_ori': 1, 'missed_ball': 10, 'wrong_foot_kick': 1} |
| Flow wide shoot | 16 | 0.0% | 0.0% | {'wrong_foot_kick': 1, 'anchor_pos': 1, 'anchor_ori': 3, 'time_out': 11} |

All runs use development seed 811314; results are not unseen-seed validation. Assisted training rollouts and incomplete smoke runs are excluded. Normal post-kick termination is not a fall. A single checkpoint must perform both tasks; this table does not claim no-reset skill switching.
