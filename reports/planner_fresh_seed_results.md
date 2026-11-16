# Paired planner validation with seed provenance

| Candidate | Seed | Seed protocol | Task | Controller | Episodes | Kick | Direction ≤10° |
|---|---:|---|---|---|---:|---:|---:|
| planner_flow_dagger_r1_fresh | 812801 | new | pass | planner | 32 | 6.2% | 0.0% |
| planner_flow_dagger_r1_fresh | 812801 | new | pass | expert | 32 | 96.9% | 84.4% |
| planner_flow_dagger_r1_fresh | 812801 | new | shoot | planner | 32 | 43.8% | 6.2% |
| planner_flow_dagger_r1_fresh | 812801 | new | shoot | expert | 32 | 100.0% | 84.4% |
| planner_history_r2_fresh | 812901 | new | pass | planner | 32 | 53.1% | 3.1% |
| planner_history_r2_fresh | 812901 | new | pass | expert | 32 | 100.0% | 84.4% |
| planner_history_r2_fresh | 812901 | new | shoot | planner | 32 | 100.0% | 43.8% |
| planner_history_r2_fresh | 812901 | new | shoot | expert | 32 | 100.0% | 78.1% |
| planner_history_r3_contact_fresh | 812902 | new | pass | planner | 32 | 87.5% | 15.6% |
| planner_history_r3_contact_fresh | 812902 | new | pass | expert | 32 | 100.0% | 96.9% |
| planner_history_r3_contact_fresh | 812902 | new | shoot | planner | 32 | 87.5% | 50.0% |
| planner_history_r3_contact_fresh | 812902 | new | shoot | expert | 32 | 93.8% | 68.8% |
| planner_history_r3_expert_target_fresh | 812903 | new | pass | planner | 32 | 100.0% | 21.9% |
| planner_history_r3_expert_target_fresh | 812903 | new | pass | expert | 32 | 100.0% | 93.8% |
| planner_history_r3_expert_target_fresh | 812903 | new | shoot | planner | 32 | 81.2% | 62.5% |
| planner_history_r3_expert_target_fresh | 812903 | new | shoot | expert | 32 | 100.0% | 78.1% |
| planner_history_r4_pass_adapter_obs_expert_fresh | 812904 | new | pass | planner | 32 | 90.6% | 31.2% |
| planner_history_r4_pass_adapter_obs_expert_fresh | 812904 | new | pass | baseline | 32 | 71.9% | 15.6% |
| planner_history_r4_pass_adapter_obs_expert_fresh | 812904 | new | pass | expert | 32 | 96.9% | 84.4% |
| planner_history_r4_pass_adapter_obs_expert_fresh | 812904 | new | shoot | planner | 32 | 96.9% | 43.8% |
| planner_history_r4_pass_adapter_obs_expert_fresh | 812904 | new | shoot | baseline | 32 | 96.9% | 43.8% |
| planner_history_r4_pass_adapter_obs_expert_fresh | 812904 | new | shoot | expert | 32 | 100.0% | 59.4% |
| planner_history_r4_pass_adapter_obs_expert_fresh2 | 812905 | new | pass | planner | 32 | 90.6% | 21.9% |
| planner_history_r4_pass_adapter_obs_expert_fresh2 | 812905 | new | pass | baseline | 32 | 87.5% | 31.2% |
| planner_history_r4_pass_adapter_obs_expert_fresh2 | 812905 | new | pass | expert | 32 | 100.0% | 84.4% |
| planner_history_r4_pass_adapter_obs_expert_fresh2 | 812905 | new | shoot | planner | 32 | 93.8% | 53.1% |
| planner_history_r4_pass_adapter_obs_expert_fresh2 | 812905 | new | shoot | baseline | 32 | 93.8% | 53.1% |
| planner_history_r4_pass_adapter_obs_expert_fresh2 | 812905 | new | shoot | expert | 32 | 93.8% | 56.2% |
| planner_history_r4_pass_adapter_action_only_reused | 812904 | reused | pass | planner | 32 | 90.6% | 21.9% |
| planner_history_r4_pass_adapter_action_only_reused | 812904 | reused | pass | baseline | 32 | 90.6% | 31.2% |
| planner_history_r4_pass_adapter_action_only_reused | 812904 | reused | pass | expert | 32 | 96.9% | 84.4% |
| planner_history_r4_pass_adapter_action_only_reused | 812904 | reused | shoot | planner | 32 | 96.9% | 43.8% |
| planner_history_r4_pass_adapter_action_only_reused | 812904 | reused | shoot | baseline | 32 | 96.9% | 43.8% |
| planner_history_r4_pass_adapter_action_only_reused | 812904 | reused | shoot | expert | 32 | 100.0% | 59.4% |

Candidate `planner_flow_dagger_r1_fresh` checkpoint SHA256: `5a7d25c5ec9ed38ec054ca63f5a79fcfcf13e4cf36dfeba4091d0d4ec9de064b`.
Candidate `planner_history_r2_fresh` checkpoint SHA256: `39e3ef5eb525950a1613921275845536c1445aaf5c28c772fcf375d0b6b12937`.
Candidate `planner_history_r3_contact_fresh` checkpoint SHA256: `5cdc03e0d5d5c44a20626d03c4a4088fab0d6f2ab5b229b66466c18bdade2133`.
Candidate `planner_history_r3_expert_target_fresh` checkpoint SHA256: `2d933fd7352262b468880c1b6c0ef45bd8af93a2e0864479bcb0247fadd40d10`.
Candidate `planner_history_r4_pass_adapter_obs_expert_fresh` checkpoint SHA256: `fc79ec46e35d0f57f6994b14c279ef7c1446a045a269cdf54b75c7f7235fe923`.
Baseline for `planner_history_r4_pass_adapter_obs_expert_fresh` checkpoint SHA256: `5cdc03e0d5d5c44a20626d03c4a4088fab0d6f2ab5b229b66466c18bdade2133`.
Candidate `planner_history_r4_pass_adapter_obs_expert_fresh2` checkpoint SHA256: `fc79ec46e35d0f57f6994b14c279ef7c1446a045a269cdf54b75c7f7235fe923`.
Baseline for `planner_history_r4_pass_adapter_obs_expert_fresh2` checkpoint SHA256: `5cdc03e0d5d5c44a20626d03c4a4088fab0d6f2ab5b229b66466c18bdade2133`.
Candidate `planner_history_r4_pass_adapter_action_only_reused` checkpoint SHA256: `783c0f355c6bbe5c0062f4cb6e0105be8914d6f699b7ef1a6f9c7a812d0d8a56`.
Controls for `planner_history_r4_pass_adapter_action_only_reused` reuse `planner_history_r4_pass_adapter_obs_expert_fresh`; this is a previously evaluated seed, not a new independent validation set.
Baseline for `planner_history_r4_pass_adapter_action_only_reused` checkpoint SHA256: `fc79ec46e35d0f57f6994b14c279ef7c1446a045a269cdf54b75c7f7235fe923`.

Only completed paired evaluations are included. The seed was absent from the candidate training metadata. Each planner uses one frozen checkpoint for both skills; initial physical states match each expert comparison. Rates count the first episode per environment, not the number of ball events. These evaluations do not establish continuous skill switching.
