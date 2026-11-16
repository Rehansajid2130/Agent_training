import math
import tkinter as tk
from collections import deque
from tkinter import ttk


WHEEL_SPACING = 0.30  # metres between the wheels
STEP_SECONDS = 0.02
PIXELS_PER_METRE = 90
ARRIVAL_DISTANCE = 0.05


def advance(x, y, heading, left, right, dt):
    forward_speed = (left + right) / 2
    turn_rate = (right - left) / WHEEL_SPACING
    heading_change = turn_rate * dt

    # Integrate motion along the arc, including the straight-line limit.
    half_turn = heading_change / 2
    distance = forward_speed * dt
    if half_turn:
        distance *= math.sin(half_turn) / half_turn
    direction = heading + half_turn
    return (
        x + distance * math.cos(direction),
        y + distance * math.sin(direction),
        (heading + heading_change) % math.tau,
    )


def target_wheel_speeds(x, y, heading, target, *, square_leg=False):
    dx, dy = target[0] - x, target[1] - y
    distance = math.hypot(dx, dy)
    arrival_distance = 1e-6 if square_leg else ARRIVAL_DISTANCE
    if distance <= arrival_distance:
        return 0.0, 0.0

    desired_heading = math.atan2(dy, dx)
    heading_error = (desired_heading - heading + math.pi) % math.tau - math.pi
    if square_leg:
        # Turn in place before driving straight to the next corner.
        if abs(heading_error) > 1e-6:
            turn_rate = max(-2.0, min(2.0, heading_error / STEP_SECONDS))
            wheel_speed = turn_rate * WHEEL_SPACING / 2
            return -wheel_speed, wheel_speed
        wheel_speed = min(0.6, distance / STEP_SECONDS)
        return wheel_speed, wheel_speed

    # Slow down when the target is near or the robot is facing away.
    forward_speed = min(0.6, distance) * max(0.0, math.cos(heading_error))
    turn_rate = 3.0 * heading_error
    wheel_offset = turn_rate * WHEEL_SPACING / 2
    left = forward_speed - wheel_offset
    right = forward_speed + wheel_offset
    scale = max(1.0, abs(left), abs(right))
    return left / scale, right / scale


class DriveSimulator:
    def __init__(self, root):
        root.title("Differential Drive Simulator")
        self.canvas = tk.Canvas(root, width=640, height=400, background="#f1f5f9")
        self.canvas.pack(fill="both", expand=True)
        self.canvas.bind("<Button-1>", self.set_target)
        self.controls = ttk.Frame(root, padding=12)
        self.controls.pack(fill="x")
        self.mode = ttk.Label(self.controls, text="Manual drive. Click the map to set a target.")
        self.mode.pack(anchor="w")

        self.left = self.add_slider("Left wheel (m/s)")
        self.right = self.add_slider("Right wheel (m/s)")
        ttk.Button(self.controls, text="Drive square", command=self.drive_square).pack(side="left")
        ttk.Button(self.controls, text="Stop", command=self.stop).pack(side="left")
        ttk.Button(self.controls, text="Reset", command=self.reset).pack(side="left")
        self.status = ttk.Label(self.controls)
        self.status.pack(side="right")

        self.trail = deque(maxlen=1500)
        self.waypoints = deque()
        self.square = ()
        self.reset()
        self.tick()

    def add_slider(self, label):
        slider = tk.Scale(
            self.controls, label=label, from_=-1.0, to=1.0,
            resolution=0.05, orient="horizontal",
        )
        slider.pack(fill="x")
        return slider

    def stop(self):
        self.target = None
        self.waypoints.clear()
        for slider in (self.left, self.right):
            slider.configure(state="normal")
            slider.set(0)
        self.mode.configure(text="Manual drive. Click the map to set a target.")

    def set_target(self, event):
        self.stop()
        self.square = ()
        center_x = self.canvas.winfo_width() / 2
        center_y = self.canvas.winfo_height() / 2
        target = (
            self.x + (event.x - center_x) / PIXELS_PER_METRE,
            self.y - (event.y - center_y) / PIXELS_PER_METRE,
        )
        self.drive_to(target, "Driving to target. Press Stop for manual control.")

    def drive_square(self):
        self.stop()
        self.trail.clear()
        self.trail.append((self.x, self.y))
        x, y = self.x, self.y
        self.square = ((x, y), (x + 1, y), (x + 1, y + 1), (x, y + 1), (x, y))
        self.waypoints.extend(self.square[1:])
        self.next_waypoint()

    def next_waypoint(self):
        number = 5 - len(self.waypoints)
        self.drive_to(self.waypoints.popleft(), f"Driving to waypoint {number} of 4.")

    def drive_to(self, target, message):
        self.target = target
        for slider in (self.left, self.right):
            slider.configure(state="disabled")
        self.mode.configure(text=message)

    def reset(self):
        self.stop()
        self.square = ()
        self.x = self.y = self.heading = 0.0
        self.trail.clear()
        self.trail.append((0.0, 0.0))

    def tick(self):
        left, right = self.left.get(), self.right.get()
        if self.target is not None:
            left, right = target_wheel_speeds(
                self.x, self.y, self.heading, self.target,
                square_leg=bool(self.square),
            )
            if left == right == 0:
                if self.waypoints:
                    self.next_waypoint()
                else:
                    message = "Square complete." if self.square else "Target reached."
                    self.stop()
                    self.mode.configure(text=message)

        self.x, self.y, self.heading = advance(
            self.x, self.y, self.heading, left, right, STEP_SECONDS,
        )
        if (self.x, self.y) != self.trail[-1]:
            self.trail.append((self.x, self.y))
        self.draw()
        self.canvas.after(round(STEP_SECONDS * 1000), self.tick)

    def draw(self):
        self.canvas.delete("all")
        center_x = self.canvas.winfo_width() / 2
        center_y = self.canvas.winfo_height() / 2

        def to_canvas(x, y):
            return (
                center_x + (x - self.x) * PIXELS_PER_METRE,
                center_y - (y - self.y) * PIXELS_PER_METRE,
            )

        if self.square:
            self.canvas.create_line(
                *(to_canvas(x, y) for x, y in self.square),
                fill="#64748b", dash=(5, 4), width=2,
            )
        if len(self.trail) >= 2:
            self.canvas.create_line(
                *(to_canvas(x, y) for x, y in self.trail), fill="#19766e", width=2,
            )

        if self.target is not None:
            target_x, target_y = to_canvas(*self.target)
            self.canvas.create_oval(
                target_x - 7, target_y - 7, target_x + 7, target_y + 7,
                outline="#19766e", width=3,
            )

        self.canvas.create_oval(
            center_x - 16, center_y - 16, center_x + 16, center_y + 16,
            fill="#20344a", outline="",
        )
        self.canvas.create_line(
            center_x, center_y,
            center_x + 28 * math.cos(self.heading),
            center_y - 28 * math.sin(self.heading),
            fill="#c04b17", width=3, arrow="last",
        )
        self.status.configure(
            text=f"x {self.x:.2f} m   y {self.y:.2f} m   Heading {math.degrees(self.heading) % 360:.0f}°",
        )


if __name__ == "__main__":
    window = tk.Tk()
    DriveSimulator(window)
    window.mainloop()
