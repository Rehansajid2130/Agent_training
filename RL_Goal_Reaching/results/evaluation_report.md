# Held-Out Evaluation Results

Each method was evaluated on the same 100 scenario seeds, starting at 100000.
PPO used deterministic actions from the final fixed-budget checkpoints. No test-based checkpoint selection was performed.

| Method | Success | Mean steps, all episodes | Mean steps, successes | Mean path, successes |
|---|---:|---:|---:|---:|
| random | 7.0% | 237.56 | 72.29 | 2.951 |
| rule | 100.0% | 45.07 | 45.07 | 2.249 |
| ppo_seed_7 | 100.0% | 31.57 | 31.57 | 2.320 |
| ppo_seed_17 | 100.0% | 31.66 | 31.66 | 2.333 |
| ppo_seed_27 | 100.0% | 31.70 | 31.70 | 2.357 |

PPO success across 3 training seeds: mean 100.00%, sample standard deviation 0.00 percentage points.

Steps are simulation steps; path lengths are arbitrary simulation units. Successful-only metrics exclude failures and may involve different subsets of scenarios.
All-episode means include timeouts. High success on this small, noise-free task does not establish real-robot reliability or general superiority over programmed control.

Raw measurements: [evaluation_episodes.csv](evaluation_episodes.csv). Summary: [evaluation_summary.json](evaluation_summary.json).
