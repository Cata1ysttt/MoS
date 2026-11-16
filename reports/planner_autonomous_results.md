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
| Recovery r3 history pass | 16 | 93.8% | 18.8% | {'anchor_pos': 2, 'missed_ball': 8, 'time_out': 6} |
| Recovery r3 history shoot | 16 | 87.5% | 37.5% | {'wrong_foot_kick': 1, 'event_kick_elapsed': 14, 'anchor_pos': 1} |
| Recovery r3 contact pass | 16 | 87.5% | 37.5% | {'wrong_foot_kick': 2, 'anchor_pos': 2, 'missed_ball': 1, 'time_out': 11} |
| Recovery r3 contact shoot | 16 | 100.0% | 68.8% | {'event_kick_elapsed': 16} |
| Recovery r3 direct pass | 16 | 62.5% | 6.2% | {'missed_ball': 5, 'wrong_foot_kick': 1, 'anchor_pos': 4, 'anchor_ori': 2, 'time_out': 4} |
| Recovery r3 direct shoot | 16 | 87.5% | 25.0% | {'wrong_foot_kick': 2, 'event_kick_elapsed': 14} |
| Recovery r3 action alignment pass | 16 | 81.2% | 18.8% | {'wrong_foot_kick': 5, 'missed_ball': 1, 'anchor_pos': 2, 'time_out': 8} |
| Recovery r3 action alignment shoot | 16 | 93.8% | 68.8% | {'event_kick_elapsed': 15, 'wrong_foot_kick': 1} |
| Recovery r3 continuation pass | 16 | 93.8% | 12.5% | {'wrong_foot_kick': 1, 'anchor_pos': 1, 'time_out': 14} |
| Recovery r3 continuation shoot | 16 | 93.8% | 56.2% | {'wrong_foot_kick': 1, 'event_kick_elapsed': 15} |
| Recovery r3 approach pass | 16 | 93.8% | 12.5% | {'anchor_pos': 1, 'missed_ball': 2, 'time_out': 13} |
| Recovery r3 approach shoot | 16 | 93.8% | 75.0% | {'wrong_foot_kick': 1, 'event_kick_elapsed': 15} |
| Recovery r3 expert target pass | 16 | 100.0% | 37.5% | {'anchor_pos': 1, 'missed_ball': 1, 'time_out': 14} |
| Recovery r3 expert target shoot | 16 | 93.8% | 62.5% | {'event_kick_elapsed': 15, 'anchor_pos': 1} |
| Recovery r3 corrected labels pass | 16 | 81.2% | 18.8% | {'missed_ball': 3, 'anchor_ori': 1, 'wrong_foot_kick': 2, 'anchor_pos': 1, 'time_out': 9} |
| Recovery r3 corrected labels shoot | 16 | 93.8% | 75.0% | {'wrong_foot_kick': 1, 'event_kick_elapsed': 15} |
| Recovery r4 pass | 16 | 87.5% | 31.2% | {'anchor_pos': 4, 'wrong_foot_kick': 1, 'time_out': 11} |
| Recovery r4 shoot | 16 | 87.5% | 37.5% | {'wrong_foot_kick': 2, 'event_kick_elapsed': 14} |
| Recovery r4 pass replay pass | 16 | 93.8% | 31.2% | {'anchor_ori': 1, 'anchor_pos': 1, 'missed_ball': 2, 'time_out': 12} |
| Recovery r4 pass replay shoot | 16 | 87.5% | 37.5% | {'wrong_foot_kick': 1, 'event_kick_elapsed': 14, 'anchor_pos': 1} |
| Recovery r4 pass adapter pass | 16 | 75.0% | 18.8% | {'missed_ball': 4, 'wrong_foot_kick': 2, 'anchor_pos': 1, 'time_out': 9} |
| Recovery r4 pass adapter shoot | 16 | 100.0% | 68.8% | {'event_kick_elapsed': 16} |
| Recovery r4 adapter direct observations pass | 16 | 87.5% | 18.8% | {'missed_ball': 3, 'wrong_foot_kick': 1, 'anchor_pos': 1, 'time_out': 11} |
| Recovery r4 adapter direct observations shoot | 16 | 100.0% | 68.8% | {'event_kick_elapsed': 16} |
| Recovery r4 adapter direct observations expert target pass | 16 | 87.5% | 31.2% | {'missed_ball': 4, 'anchor_pos': 2, 'time_out': 10} |
| Recovery r4 adapter direct observations expert target shoot | 16 | 100.0% | 68.8% | {'event_kick_elapsed': 16} |
| Recovery r4 adapter ball boundaries pass | 16 | 81.2% | 37.5% | {'missed_ball': 3, 'wrong_foot_kick': 1, 'anchor_pos': 1, 'time_out': 11} |
| Recovery r4 adapter ball boundaries shoot | 16 | 100.0% | 68.8% | {'event_kick_elapsed': 16} |
| Recovery r4 adapter action only pass | 16 | 93.8% | 43.8% | {'wrong_foot_kick': 1, 'anchor_pos': 2, 'missed_ball': 1, 'time_out': 12} |
| Recovery r4 adapter action only shoot | 16 | 100.0% | 68.8% | {'event_kick_elapsed': 16} |
| Recovery r4 wider pass adapter pass | 16 | 87.5% | 18.8% | {'missed_ball': 3, 'anchor_pos': 2, 'wrong_foot_kick': 1, 'time_out': 10} |
| Recovery r4 wider pass adapter shoot | 16 | 100.0% | 68.8% | {'event_kick_elapsed': 16} |
| Recovery r4 final latent adapter pass | 16 | 93.8% | 31.2% | {'anchor_pos': 4, 'missed_ball': 3, 'time_out': 9} |
| Recovery r4 final latent adapter shoot | 16 | 100.0% | 68.8% | {'event_kick_elapsed': 16} |
| Recovery r5 continuation control pass | 16 | 87.5% | 50.0% | {'anchor_pos': 5, 'anchor_ori': 1, 'time_out': 10} |
| Recovery r5 continuation control shoot | 16 | 100.0% | 68.8% | {'event_kick_elapsed': 16} |
| Recovery r5 passing recovery replay pass | 16 | 93.8% | 31.2% | {'wrong_foot_kick': 2, 'anchor_pos': 1, 'time_out': 13} |
| Recovery r5 passing recovery replay shoot | 16 | 100.0% | 68.8% | {'event_kick_elapsed': 16} |
| Passing latent PPO pilot pass | 16 | 81.2% | 37.5% | {'wrong_foot_kick': 2, 'anchor_pos': 1, 'missed_ball': 1, 'time_out': 12} |
| Passing latent PPO pilot shoot | 16 | 100.0% | 68.8% | {'event_kick_elapsed': 16} |
| Passing latent PPO larger exploration pass | 16 | 93.8% | 18.8% | {'missed_ball': 4, 'anchor_pos': 2, 'wrong_foot_kick': 3, 'anchor_ori': 3, 'time_out': 4} |
| Passing latent PPO larger exploration shoot | 16 | 100.0% | 68.8% | {'event_kick_elapsed': 16} |
| Passing PPO reference-reward ablation pass | 16 | 93.8% | 31.2% | {'wrong_foot_kick': 1, 'missed_ball': 3, 'time_out': 12} |
| Passing PPO reference-reward ablation shoot | 16 | 100.0% | 68.8% | {'event_kick_elapsed': 16} |
| Passing PPO reference ablation continued pass | 16 | 87.5% | 12.5% | {'anchor_pos': 3, 'missed_ball': 3, 'wrong_foot_kick': 1, 'time_out': 9} |
| Passing PPO reference ablation continued shoot | 16 | 100.0% | 68.8% | {'event_kick_elapsed': 16} |
| Passing PPO shared shooting scene pass | 16 | 75.0% | 31.2% | {'missed_ball': 5, 'anchor_ori': 1, 'wrong_foot_kick': 2, 'time_out': 8} |
| Passing PPO shared shooting scene shoot | 16 | 100.0% | 68.8% | {'event_kick_elapsed': 16} |
| Post-shot recovery PPO pass | 16 | 93.8% | 18.8% | {'wrong_foot_kick': 2, 'missed_ball': 3, 'time_out': 11} |
| Post-shot recovery PPO shoot | 16 | 100.0% | 68.8% | {'event_kick_elapsed': 16} |
| Recovery r4 spatially gated adapter pass | 16 | 87.5% | 18.8% | {'missed_ball': 2, 'wrong_foot_kick': 2, 'anchor_pos': 3, 'time_out': 9} |

All runs use development seed 811314; results are not unseen-seed validation. Assisted training rollouts and incomplete smoke runs are excluded. Normal post-kick termination is not a fall. A single checkpoint must perform both tasks; this table does not claim no-reset skill switching.
