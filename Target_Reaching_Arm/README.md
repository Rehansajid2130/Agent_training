## Run

Double-click `Launch Simulator.bat`, or run `python arm_simulator.py` in this folder. Uses Python's standard library only; Tkinter must be available.

- Click in the drawing to choose a target.
- Switch between positive and negative elbow angles to see both solutions.
- Move either slider to explore forward kinematics manually.
- Click inside the inner circle or outside the outer circle to test unreachable targets. The arm stays at its previous pose.
- Run evaluation to save `evaluation_results.csv` beside the program. Rerunning replaces that generated file.

## The maths to explain

Link lengths are L1 = 1.0 and L2 = 0.8 in arbitrary units. Angles in the maths are radians. The elbow angle is relative to the first link, not the world x-axis.

Forward kinematics turns angles into a tip position:

    x = L1*cos(theta1) + L2*cos(theta1 + theta2)
    y = L1*sin(theta1) + L2*sin(theta1 + theta2)

Inverse kinematics turns a target position into angles. From the cosine rule:

    c = (x*x + y*y - L1*L1 - L2*L2) / (2*L1*L2)
    theta2 = +acos(c) or -acos(c)
    theta1 = atan2(y, x) - atan2(L2*sin(theta2), L1 + L2*cos(theta2))

`atan2` handles the target's quadrant. The two signs bend the elbow in opposite directions. At fully extended or folded boundaries, the two geometric solutions coincide.

A target is reachable when:

    abs(L1 - L2) <= sqrt(x*x + y*y) <= L1 + L2

Here the workspace is an annulus with radii 0.2 and 1.8. Unequal links mean the tip cannot reach the base. Floating-point round-off near the boundaries is handled with a small numerical tolerance.

Reaching error is Euclidean distance between the desired target and the tip computed by forward kinematics from the inverse-kinematics solution.

## Evaluation

Run `python arm_simulator.py --evaluate` for 22 checks: 11 target cases, each with both elbow branches. Cases include all four quadrants, inner and outer boundaries, nearby interior targets, and unreachable targets. The CSV records target coordinates, branch, reachability, angles, tip coordinates, and measured error. Unreachable cases have blank angle and error fields because no reaching solution exists.

This is a deterministic numerical consistency check, not 22 independent physical trials. Tiny numerical errors demonstrate agreement between the equations, not real-robot accuracy. No hardware experiments have been performed.

## Interview preparation

1. **What did you add?** Automatic target reaching, inverse kinematics, workspace rejection, two elbow configurations, and repeatable numerical evaluation.
2. **Why two solutions?** The elbow can bend on either side of the line from the base to the target.
3. **Why reject a close target?** With unequal links, the longer link cannot be cancelled completely by folding the shorter one back.
4. **Does it avoid obstacles?** No. It checks geometric reachability only.
5. **Does it simulate motors?** No. It draws the computed pose directly; it has no dynamics, motion trajectory, acceleration limits, feedback controller, or physical joint restrictions.
6. **What would you add next?** A simple interpolated joint motion, followed by joint limits. Interpolation alone would not guarantee collision-free motion.

Before adding this to your CV, run it yourself, explain the equations, try both solutions and unreachable points, and inspect the CSV. Describe it as a simulation and distinguish the assisted implementation from what you personally understand and verify.
