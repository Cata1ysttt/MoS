# Recovery-data input comparison

Offline action regression only; these are not autonomous latent-policy success rates. Each run completed 20000 updates using recovery seeds 812402 (pass) and 812401 (shoot), held-out environment lanes, and the same fixed 512-state validation sample. The frozen flow reference scores 0.0168467 on this sample.

| Input | Overall action MSE | Pass | Shoot | Pass launched before contact (51 states) | Shoot before contact (234 states) |
|---|---:|---:|---:|---:|---:|
| state | 0.0074164 | 0.0071433 | 0.0076894 | 0.0066720 | 0.0077184 |
| state_copy | 0.0069822 | 0.0068690 | 0.0070955 | 0.0064343 | 0.0070878 |
| expert_input | 0.0038037 | 0.0037177 | 0.0038897 | 0.0052517 | 0.0039418 |

`state_copy` and `expert_input` each have 9,919,517 trainable parameters and the same zero-initialized added input path. The control copies the first 1200 existing state features into that path; the other supplies the recorded policy/RMA observation histories, padded to 1200 dimensions. Both use the same initialization seed, minibatch sampling settings, labels and validation states. Results are at the best overall validation checkpoint for each run, not separately selected per skill or phase.

The real observation histories reduce overall error about 46% relative to the capacity control. Pre-contact improvement is smaller for passing than shooting. This supports testing the histories in a latent planner; it does not establish closed-loop success or identify which history terms cause the improvement.

The v4 planner input is 2406 dimensions: the first 1200 v3 features, 1200 observation-history features, then the original six command features. The native actor inputs and backward prior retain their original positions. Expanding a trained v3 flow model zero-initializes only the new input weights, preserving its initial predictions. Online inference obtains histories from the observation manager and requires no expert policy action. No-reset cross-skill switching remains unvalidated.
