# Held-Out Evaluation Results

Each method was evaluated on the same 100 scenario seeds, starting at 100000.
PPO used deterministic actions from the final fixed-budget checkpoints. No test-based checkpoint selection was performed.

| Method | Success | Collisions | Timeouts | Mean steps, successes | Mean path, successes |
|---|---:|---:|---:|---:|---:|
| random | 0.0% | 11.0% | 89.0% | N/A | N/A |
| goal_only_rule | 11.0% | 89.0% | 0.0% | 63.27 | 3.690 |
| untrained_seed_7 | 0.0% | 7.0% | 93.0% | N/A | N/A |
| untrained_seed_17 | 0.0% | 7.0% | 93.0% | N/A | N/A |
| untrained_seed_27 | 0.0% | 7.0% | 93.0% | N/A | N/A |
| ppo_seed_7 | 93.0% | 7.0% | 0.0% | 52.03 | 3.988 |
| ppo_seed_17 | 99.0% | 1.0% | 0.0% | 53.41 | 4.060 |
| ppo_seed_27 | 92.0% | 8.0% | 0.0% | 52.93 | 4.084 |

PPO success across 3 training seeds: mean 94.67%, sample standard deviation 3.79 percentage points.

Steps are simulation steps; path lengths are arbitrary simulation units. Successful-only metrics exclude failures and may involve different subsets of scenarios.
The goal-only programmed controller deliberately ignores obstacle observations; it is not an obstacle-avoidance baseline. Untrained policies use deterministic actions from their saved initial weights, not uniformly random actions. High success on this small, noise-free task does not establish real-robot reliability or unseen-map generalization.

Raw measurements: [evaluation_episodes.csv](evaluation_episodes.csv). Summary: [evaluation_summary.json](evaluation_summary.json).
