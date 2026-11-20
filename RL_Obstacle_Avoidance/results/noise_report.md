# Observation Noise Experiment

Frozen trained policies for seeds (7, 17, 27); 100 paired scenarios per policy and condition, starting at scenario seed 100000. No retraining or checkpoint selection.

## Noise settings

| Condition | Distance standard deviation | Bearing standard deviation |
|---|---:|---:|
| clean | 0.00 simulation units | 0 degrees |
| small | 0.02 simulation units | 2 degrees |
| moderate | 0.05 simulation units | 5 degrees |

Noise is independent Gaussian error on goal and obstacle distances and bearings at every observation. Distances are clamped to zero before rescaling; sine/cosine inputs are reconstructed from noisy bearings. Geometry, collision detection, reward and success use true simulator state. Noise RNG seed is 500000 + scenario_seed, separate from environment RNG. Noise levels share the same standard-normal draws, scaled to their respective standard deviations.

## Results by training seed

| Model seed | Condition | Success | Collisions | Timeouts |
|---|---|---:|---:|---:|
| 7 | clean | 93.0% | 7.0% | 0.0% |
| 7 | small | 93.0% | 7.0% | 0.0% |
| 7 | moderate | 93.0% | 7.0% | 0.0% |
| 17 | clean | 99.0% | 1.0% | 0.0% |
| 17 | small | 99.0% | 1.0% | 0.0% |
| 17 | moderate | 99.0% | 1.0% | 0.0% |
| 27 | clean | 92.0% | 8.0% | 0.0% |
| 27 | small | 91.0% | 9.0% | 0.0% |
| 27 | moderate | 91.0% | 9.0% | 0.0% |

## Combined comparison

| Condition | Success | Collisions | Timeouts | Change in success vs clean |
|---|---:|---:|---:|---:|
| clean | 94.67% | 5.33% | 0.00% | +0.00 percentage points |
| small | 94.33% | 5.67% | 0.00% | -0.33 percentage points |
| moderate | 94.33% | 5.67% | 0.00% | -0.33 percentage points |

Rates average the three policies on shared scenarios; these are not independent maps. This is one reproducible noise sequence per scenario, not multiple independent noise repeats. The clean case is rerun using the current saved weights rather than copied from an earlier report. The experiment measures sensitivity to this artificial noise model, not calibrated real sensors. A change is descriptive; no statistical significance or safety guarantee is claimed.

## Saved-model fingerprints

SHA-256 fingerprints were verified unchanged after the experiment:
- Seed 7: `cbd934e901fd4f63470c4d3c22429410f9714657de49219eb0d1e147c0b48818`
- Seed 17: `4a417eb2894e9b99ff089e7f9bb1abb1c891f9ba2c88cc4a539334801414f99f`
- Seed 27: `b7cd360c8dfe18a9a44ebcef2733345b06fda68ba3a906d4c9c39ed2499952ef`

Raw episode data: [noise_evaluation.csv](noise_evaluation.csv).
