"""A planar two-link arm. Run with --evaluate for a repeatable numerical check."""

import argparse
import csv
import math
from pathlib import Path

L1, L2 = 1.0, 0.8  # Arbitrary length units; unequal links show the inner workspace hole.


def forward(theta1, theta2):
    """Angles are radians; return elbow and tip coordinates."""
    elbow = (L1 * math.cos(theta1), L1 * math.sin(theta1))
    tip = (elbow[0] + L2 * math.cos(theta1 + theta2),
           elbow[1] + L2 * math.sin(theta1 + theta2))
    return elbow, tip


def inverse(x, y, branch=1):
    """Return joint angles, or None outside the workspace. branch is +1 or -1."""
    cosine = (x*x + y*y - L1*L1 - L2*L2) / (2 * L1 * L2)
    if cosine < -1 - 1e-12 or cosine > 1 + 1e-12:
        return None
    cosine = max(-1.0, min(1.0, cosine))  # Round-off at the workspace boundary.
    theta2 = branch * math.acos(cosine)
    theta1 = math.atan2(y, x) - math.atan2(L2 * math.sin(theta2),
                                         L1 + L2 * math.cos(theta2))
    theta1 = (theta1 + math.pi) % (2 * math.pi) - math.pi
    return theta1, theta2


def evaluate(output):
    """Check both branches, boundaries and unreachable points; save actual errors."""
    cases = [("interior", 1, 0.7, True), ("negative_x", -1, 0.5, True),
             ("negative_y", 0.4, -1, True), ("vertical", 0, 1, True),
             ("outer_boundary", L1 + L2, 0, True),
             ("inner_boundary", L1 - L2, 0, True),
             ("near_outer", L1 + L2 - 1e-6, 0, True),
             ("near_inner", L1 - L2 + 1e-6, 0, True),
             ("outside", L1 + L2 + 0.01, 0, False),
             ("inside_hole", 0.1, 0, False), ("origin", 0, 0, False)]
    rows, errors = [], []
    for name, x, y, expected in cases:
        for branch in (1, -1):
            angles = inverse(x, y, branch)
            assert (angles is not None) == expected, (name, branch)
            row = dict(case=name, target_x=x, target_y=y, branch=branch,
                       reachable=angles is not None, theta1_deg="", theta2_deg="",
                       tip_x="", tip_y="", error="")
            if angles is not None:
                _, tip = forward(*angles)
                error = math.hypot(tip[0] - x, tip[1] - y)
                assert error < 1e-9, (name, branch, error)
                errors.append(error)
                row.update(theta1_deg=math.degrees(angles[0]),
                           theta2_deg=math.degrees(angles[1]),
                           tip_x=tip[0], tip_y=tip[1], error=error)
            rows.append(row)
    with Path(output).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    return f"{len(rows)} checks passed; maximum reachable-target error: {max(errors):.3e} units."


def launch():
    import tkinter as tk
    from tkinter import messagebox, ttk

    root = tk.Tk()
    root.title("Two-Joint Arm")
    root.resizable(False, False)
    canvas = tk.Canvas(root, width=640, height=600, bg="#f5f7fa", highlightthickness=0)
    canvas.grid(row=0, column=0)
    panel = ttk.Frame(root, padding=18, width=300)
    panel.grid(row=0, column=1, sticky="ns")
    target = [1.0, 0.7]
    angles = [0.0, 0.0]
    branch = tk.IntVar(value=1)
    status = tk.StringVar()
    details = tk.StringVar()
    mode = ["Manual joint control"]
    sliders = []
    updating = [False]
    origin_x, origin_y, scale = 320, 300, 140

    def screen(x, y):
        return origin_x + scale*x, origin_y - scale*y

    def draw():
        canvas.delete("all")
        for radius in (L1 + L2, abs(L1 - L2)):
            a, b = screen(-radius, radius)
            c, d = screen(radius, -radius)
            canvas.create_oval(a, b, c, d, outline="#aab6c4", dash=(5, 4))
        canvas.create_line(20, origin_y, 620, origin_y, fill="#ccd4dd", arrow=tk.LAST)
        canvas.create_line(origin_x, 580, origin_x, 20, fill="#ccd4dd", arrow=tk.LAST)
        canvas.create_text(610, origin_y + 16, text="x")
        canvas.create_text(origin_x + 16, 25, text="y")
        elbow, tip = forward(*angles)
        base_px, elbow_px, tip_px = screen(0, 0), screen(*elbow), screen(*tip)
        canvas.create_line(*base_px, *elbow_px, fill="#2768ac", width=10)
        canvas.create_line(*elbow_px, *tip_px, fill="#2b9b82", width=10)
        for px, py in (base_px, elbow_px, tip_px):
            canvas.create_oval(px-6, py-6, px+6, py+6, fill="#25364a", outline="white")
        tx, ty = screen(*target)
        reachable = inverse(*target) is not None
        colour = "#b74343" if not reachable else "#c06c12"
        canvas.create_line(tx-8, ty, tx+8, ty, fill=colour, width=2)
        canvas.create_line(tx, ty-8, tx, ty+8, fill=colour, width=2)
        canvas.create_line(*tip_px, tx, ty, fill=colour, dash=(3, 3))
       
        error = math.hypot(tip[0]-target[0], tip[1]-target[1])
        status.set(mode[0])
        details.set(f"Target: ({target[0]:.3f}, {target[1]:.3f})\n"
                    f"Tip: ({tip[0]:.3f}, {tip[1]:.3f})\n\n"
                    f"Shoulder: {math.degrees(angles[0]):.2f} deg\n"
                    f"Elbow: {math.degrees(angles[1]):.2f} deg\n\n"
                    f"Reaching error: {error:.3e} units")

    def solve():
        solution = inverse(*target, branch.get())
        if solution is None:
            mode[0] = "Unreachable target; arm held in place"
        else:
            angles[:] = solution
            updating[0] = True
            for slider, angle in zip(sliders, angles):
                slider.set(math.degrees(angle))
            updating[0] = False
            mode[0] = "Target reached using inverse kinematics"
        draw()

    def click(event):
        target[:] = [(event.x-origin_x)/scale, (origin_y-event.y)/scale]
        solve()

    def manual(index, value):
        if updating[0]:
            return
        angles[index] = math.radians(float(value))
        mode[0] = "Manual joint control"
        draw()

    def save_results():
        output = Path(__file__).with_name("evaluation_results.csv")
        try:
            summary = evaluate(output)
        except (OSError, AssertionError) as error:
            messagebox.showerror("Evaluation failed", str(error))
            return
        messagebox.showinfo("Evaluation complete", f"{summary}\n\nSaved to:\n{output}")

    ttk.Label(panel, text="Target Reaching", font=("Segoe UI", 17, "bold")).pack(anchor="w")
    ttk.Label(panel, text=f"Link lengths: {L1} and {L2} units").pack(anchor="w", pady=(8, 16))
    for index, label in enumerate(("Shoulder angle", "Elbow angle")):
        slider = tk.Scale(panel, label=label, from_=-180, to=180, resolution=0.1,
                          orient="horizontal", length=260,
                          command=lambda value, i=index: manual(i, value))
        slider.pack(anchor="w", pady=6)
        sliders.append(slider)
    ttk.Label(panel, text="Inverse-kinematics solution:").pack(anchor="w", pady=(16, 4))
    for value, label in ((1, "Positive elbow angle"), (-1, "Negative elbow angle")):
        ttk.Radiobutton(panel, text=label, variable=branch, value=value, command=solve).pack(anchor="w")
    ttk.Button(panel, text="Reach selected target", command=solve).pack(fill="x", pady=12)
    ttk.Label(panel, textvariable=status, wraplength=260).pack(anchor="w", pady=4)
    ttk.Label(panel, textvariable=details, justify="left").pack(anchor="w", pady=12)
    ttk.Button(panel, text="Run evaluation and save CSV", command=save_results).pack(fill="x", pady=8)
    canvas.bind("<Button-1>", click)
    solve()
    root.mainloop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluate", action="store_true", help="Run numerical checks without opening a window")
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("evaluation_results.csv"))
    args = parser.parse_args()
    if args.evaluate:
        print(evaluate(args.output))
    else:
        launch()
