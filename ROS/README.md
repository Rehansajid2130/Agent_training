# 🤖 ROS 2 + Gazebo Ball-Following Robot

An autonomous perception-action pipeline for mobile robotics combining **ROS 2**, **OpenCV computer vision**, and **Gazebo physics simulation**. The robot perceives its environment through a forward-facing camera, tracks a target ball using HSV color segmentation, and drives toward it using a closed-loop proportional controller.

---

## 📌 Architecture & Control Pipeline

```
  +------------------+         sensor_msgs/Image
  |  Gazebo Camera   | ------------------------------------+
  |  (or USB Webcam) |                                     |
  +------------------+                                     v
                                             +---------------------------+
                                             |  OpenCV Perception Engine |
                                             |  - Gaussian Blur          |
                                             |  - RGB -> HSV Conversion  |
                                             |  - Color Mask & Morphology|
                                             |  - Find MinEnclosingCircle|
                                             +---------------------------+
                                                           |
                                               Center (x, y) & Radius
                                                           v
  +------------------+          /cmd_vel     +---------------------------+
  |  Gazebo Diff     | <-------------------- | Proportional Controller   |
  |  Drive Robot     |      geometry_msgs/   | - Angular: Center steering|
  |  (Motors)        |          Twist        | - Linear: Proximity drive |
  +------------------+                       | - Search: 360° Scan       |
                                             +---------------------------+
```

---

## 📂 Folder Structure

```text
ROS/
├── ball_follower_node.py       # Full ROS 2 Python Node (OpenCV Perception + Control)
├── controller.py               # Basic open-loop velocity publisher demo
├── launch/
│   └── ball_follower.launch.py # Unified ROS 2 launch file (Gazebo + Robot + Perception)
├── urdf/
│   └── ball_follower_robot.urdf# Differential-drive robot model with camera sensor plugin
├── worlds/
│   └── ball_world.sdf          # Gazebo world containing ground, sunlight, and a yellow target ball
└── README.md                   # ROS documentation and execution guide
```

---

## 🧠 Perception & Control Strategy

### 1. OpenCV Perception (`ball_follower_node.py`)
- **Color Thresholding:** Converts incoming frames to **HSV** (Hue, Saturation, Value) color space to remain invariant to lighting changes.
  - Default range is set for **Yellow/Tennis Ball**: `Lower: [20, 100, 100]`, `Upper: [35, 255, 255]`.
- **Noise Reduction:** Applies a Gaussian blur `(11, 11)` followed by morphological erosion and dilation.
- **Centroid & Scale Estimation:** Finds the largest contour and fits a `cv2.minEnclosingCircle` to compute:
  - $(c_x, c_y)$: Centroid coordinates in the image frame.
  - $r$: Enclosing circle radius in pixels (proportional to proximity).

### 2. Proportional Control
- **Angular Steering:**
  $$\omega_z = K_{\text{angular}} \times \frac{\text{Center}_x - c_x}{\text{Center}_x}$$
  If the ball is to the left of the image center, the robot steers left; if to the right, it steers right.
- **Linear Approach:**
  $$v_x = K_{\text{linear}} \times \frac{r_{\text{target}} - r}{r_{\text{target}}}$$
  As the robot approaches the ball and the radius grows toward `target_radius_pixels`, it gently slows down and stops.
- **Autonomous Search State:**
  If no ball is visible, the robot rotates at a steady scan speed (`search_angular_speed = 0.3 rad/s`) until the ball re-enters the field of view.

---

## 🚀 How to Run

### Option A: Using ROS 2 + Gazebo Simulation

#### 1. Source ROS 2 (e.g. Humble)
```bash
source /opt/ros/humble/setup.bash
```

#### 2. Launch Gazebo with the Robot & World
```bash
# Terminal 1: Start Gazebo simulation with camera robot & ball
ros2 launch ball_follower.launch.py
```

#### 3. Run the Perception & Control Node
```bash
# Terminal 2: Run the ball-following node
python3 ball_follower_node.py
```

---

### Option B: Testing with TurtleBot3

If you already have `turtlebot3_gazebo` installed:

```bash
# Terminal 1: Launch TurtleBot3 in a world
export TURTLEBOT3_MODEL=waffle_pi
ros2 launch turtlebot3_gazebo turtlebot3_world.launch.py

# Terminal 2: Run Ball Follower Node listening to TurtleBot3 camera
python3 ball_follower_node.py --ros-args -p camera_topic:=/camera/image_raw
```

---

### Option C: Standalone Test with USB Webcam (No Gazebo Needed)

You can also run this with a standard USB or laptop camera publisher node (like `v4l2_camera` or `usb_cam`):

```bash
# Terminal 1: Start camera publisher
ros2 run v4l2_camera v4l2_camera_node

# Terminal 2: Run ball follower with debug UI enabled
python3 ball_follower_node.py --ros-args -p show_debug_window:=True
```
Hold a tennis ball or yellow object in front of the camera, and watch the green tracking ring follow it while `/cmd_vel` outputs steering velocity!

---

## ⚙️ Configurable Parameters

| Parameter | Default | Description |
| :--- | :--- | :--- |
| `camera_topic` | `/camera/image_raw` | Topic where robot camera frames are published |
| `cmd_vel_topic` | `/cmd_vel` | Output topic for mobile base velocity |
| `target_radius_pixels` | `65.0` | Target radius in pixels when robot reaches stopping distance |
| `linear_speed_max` | `0.4` | Max forward speed (m/s) |
| `angular_speed_max` | `1.0` | Max rotational speed (rad/s) |
| `search_angular_speed` | `0.3` | Scan speed when searching for the ball |
| `show_debug_window` | `False` | Displays OpenCV annotated visual window |

