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
| Flow action pass | 16 | 0.0% | 0.0% | {'anchor_ori': 4, 'missed_ball': 11, 'wrong_foot_kick': 1} |
| Flow action shoot | 16 | 6.2% | 0.0% | {'wrong_foot_kick': 4, 'anchor_ori': 3, 'event_kick_elapsed': 1, 'anchor_pos': 3, 'time_out': 5} |
| Flow DAgger r1 pass | 16 | 31.2% | 12.5% | {'anchor_pos': 2, 'missed_ball': 10, 'wrong_foot_kick': 1, 'anchor_ori': 1, 'time_out': 2} |
| Flow DAgger r1 shoot | 16 | 31.2% | 18.8% | {'wrong_foot_kick': 5, 'event_kick_elapsed': 5, 'anchor_pos': 5, 'time_out': 1} |
| Flow DAgger weighted pass | 16 | 18.8% | 0.0% | {'missed_ball': 12, 'anchor_pos': 1, 'wrong_foot_kick': 2, 'anchor_ori': 1} |
| Flow DAgger weighted shoot | 16 | 43.8% | 12.5% | {'wrong_foot_kick': 5, 'event_kick_elapsed': 7, 'anchor_pos': 3, 'time_out': 1} |
| Flow contact pass | 16 | 12.5% | 0.0% | {'missed_ball': 7, 'anchor_pos': 4, 'wrong_foot_kick': 2, 'time_out': 3} |
| Flow contact shoot | 16 | 43.8% | 12.5% | {'wrong_foot_kick': 4, 'event_kick_elapsed': 7, 'anchor_pos': 1, 'anchor_ori': 2, 'time_out': 2} |
| Action alignment pass | 16 | 31.2% | 6.2% | {'wrong_foot_kick': 5, 'missed_ball': 6, 'anchor_pos': 5} |
| Action alignment shoot | 16 | 37.5% | 12.5% | {'wrong_foot_kick': 5, 'event_kick_elapsed': 6, 'anchor_pos': 4, 'time_out': 1} |
| Flow continuation pass | 16 | 25.0% | 6.2% | {'missed_ball': 12, 'anchor_pos': 2, 'anchor_ori': 1, 'time_out': 1} |
| Flow continuation shoot | 16 | 37.5% | 6.2% | {'event_kick_elapsed': 6, 'wrong_foot_kick': 5, 'anchor_pos': 4, 'anchor_ori': 1} |
| Recovery r2 replay pass | 16 | 43.8% | 0.0% | {'wrong_foot_kick': 1, 'missed_ball': 9, 'anchor_ori': 1, 'anchor_pos': 4, 'time_out': 1} |
| Recovery r2 replay shoot | 16 | 75.0% | 25.0% | {'event_kick_elapsed': 12, 'anchor_pos': 3, 'wrong_foot_kick': 1} |
| Recovery r2 state only pass | 16 | 43.8% | 12.5% | {'missed_ball': 9, 'anchor_pos': 5, 'anchor_ori': 1, 'time_out': 1} |
| Recovery r2 state only shoot | 16 | 62.5% | 6.2% | {'wrong_foot_kick': 2, 'event_kick_elapsed': 10, 'anchor_pos': 4} |
| Recovery r2 history pass | 16 | 62.5% | 18.8% | {'missed_ball': 3, 'wrong_foot_kick': 3, 'anchor_pos': 3, 'time_out': 7} |
| Recovery r2 history shoot | 16 | 100.0% | 50.0% | {'event_kick_elapsed': 16} |

All runs use development seed 811314; results are not unseen-seed validation. Assisted training rollouts and incomplete smoke runs are excluded. Normal post-kick termination is not a fall. A single checkpoint must perform both tasks; this table does not claim no-reset skill switching.
