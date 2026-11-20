# Target-Loss Handling and Recovery in a ROS 2 Vision-Based Tracking System

**Author:** Rehan Sajid  
**Project type:** Applied robotics software investigation  
**Status:** Implemented and evaluated with generated camera images; robot-motion validation remains future work.

## Abstract

Camera-based robot tracking requires a defined response when a target disappears or detection messages stop arriving. This project extends an existing ROS 2 red-target tracker with immediate zero-velocity publication on missing or invalid detections, explicit tracking-state logs, and a monotonic-clock watchdog. OpenCV colour segmentation provides target measurements to a proportional controller. An automated integration test supplies generated images to the actual detector and observes the controller's commands. All 13 recorded checks passed in the documented run. Across three target-loss events, mean observed stop-command latency was 31.73 ms; across three target returns, mean observed recovery-command latency was 76.89 ms. These measurements describe local software behaviour, rather than physical stopping performance or general tracking accuracy.

## 1. Objective and Research Questions

The objective is to implement and examine a simple, explainable response to target absence in a visual tracking pipeline.

1. Does the controller publish zero linear and angular velocity when the detector reports no valid target?
2. Does tracking resume when a valid target returns?
3. Does the controller stop issuing motion commands when detection data becomes stale, even without an explicit target-loss message?

The scope is software integration and command-response measurement. This project does not propose a novel detection or control algorithm.

## 2. Existing System and Contribution

The active Ubuntu controller already supported stopping on missing or stale detections and resuming on a valid detection. This extension adds:

- Zero-command publication directly in the invalid-detection callback, without waiting for the next periodic control tick.
- Explicit `WAITING`, `TRACKING`, `LOST`, and `STALE` transition logs.
- A monotonic wall clock for the detection watchdog.
- A repeatable integration test using generated camera images and invalid detection messages.

The red-target detector and proportional control law are retained. No before-and-after timing comparison was conducted, so these results do not establish a measured latency improvement over the previous implementation.

## 3. System Architecture

```text
Generated RGB images -> OpenCV detector -> Proportional controller -> Test recorder
                        |                  |                         |
                 [cx, cy, radius]     /cmd_vel commands           CSV results
                  or empty array      + timeout watchdog
```

The detector subscribes to `/camera/image` and publishes `/ball_detection/pixels`. The controller publishes `geometry_msgs/Twist` commands on `/cmd_vel`. The test executes the detector, controller, and measurement node in one Python process with a ROS 2 single-threaded executor. No Gazebo bridge or physical robot participates in this evaluation.

## 4. Methodology

### 4.1 Visual detection

The detector converts BGR images to HSV colour space and combines two red-colour masks. Morphological opening and closing with a 5 x 5 kernel reduce small regions and gaps. The largest external contour is accepted when its area exceeds 100 pixels squared and its centroid can be calculated. The detector publishes its centroid and enclosing-circle radius; otherwise, it publishes an empty array.

This method identifies a qualifying red region. It does not verify object identity or confirm that the region is spherical.

### 4.2 Proportional control

For a valid target, the controller calculates:

```text
horizontal_error = image_centre_x - detected_centre_x
radius_error     = desired_radius - detected_radius

angular_command = clip(0.0025 * horizontal_error, -0.5, 0.5)
linear_command  = clip(0.0030 * radius_error, -0.05, 0.12)
```

`clip` limits a value to the stated interval. Commands within their respective error deadbands are set to zero.

| Parameter | Value |
|---|---:|
| Image horizontal centre | 320 px |
| Desired apparent radius | 105 px |
| Horizontal-error deadband | 12 px |
| Radius-error deadband | 5 px |
| Maximum forward command | 0.12 m/s |
| Maximum reverse command magnitude | 0.05 m/s |
| Maximum angular command magnitude | 0.5 rad/s |
| Periodic control rate | 10 Hz |
| Detection timeout | 0.5 s |

Apparent radius is an image-space distance proxy, not a calibrated physical-distance measurement.

### 4.3 Target-loss and recovery logic

| State | Trigger | Behaviour |
|---|---|---|
| `WAITING` | Startup before a valid detection | Periodically publish zero commands |
| `TRACKING` | Valid target detection | Update measurements and apply proportional control |
| `LOST` | Empty/short detection array or non-positive radius | Publish zero commands immediately and remain stopped |
| `STALE` | Last valid detection is older than 0.5 s | Publish zero commands at the watchdog control tick |

A valid detection restores tracking; the next control tick calculates commands. There is no search rotation, motion prediction, or learned recovery policy. A target already centred at the desired radius can produce zero commands while tracking.

The watchdog checks age at 10 Hz. Its timeout does not guarantee subscriber receipt of a zero command exactly 0.5 s after the last valid detection. Timer scheduling and message delivery also affect the response.

## 5. Experimental Setup and Protocol

The existing environment uses Ubuntu 24.04 through WSL, ROS 2 Jazzy, Python, OpenCV, NumPy, and `cv_bridge`.

Generated images are 640 x 480 pixels with a dark-grey background. Visible-target frames contain a red circle centred at `(220, 240)` pixels with a drawn radius of 45 pixels. This creates nonzero steering and apparent-distance errors. Frames are scheduled at approximately 20 Hz; actual timing depends on processing and scheduling.

The protocol checks startup stopping, then records:

1. Three sequences of visible target, blank image, and target return, with phases of approximately 0.7 s.
2. A camera-stream interruption of approximately 1.0 s, followed by a visible-target return.
3. Separate injections of a short detection array and a zero-radius detection.

The startup assertion is additional to the 13 recorded checks. ROS domain 77 and localhost discovery separate the test from the normal default-domain simulation. Other nodes must not use domain 77 during testing.

## 6. Evaluation Metrics

All elapsed-time measurements use a monotonic clock.

- **Target-loss latency:** time from the beginning of a blank-image phase to receipt of the first zero linear and angular command at the test subscriber.
- **Recovery latency:** time from the beginning of a returning-target phase to receipt of the first nonzero command.
- **Stale-stream latency:** time from the controller's last valid detection receipt to the subscriber's first zero command after image delivery stops.
- **Invalid-input latency:** time from invalid-message injection to receipt of the first zero command.

The test also checks final command state and, after a stop, verifies that subsequent observed commands remain zero within that phase. Normal response checks require latency below 0.5 s; the stale-stream check allows 0.75 s, including the configured 0.5 s timeout. These are project-specific test criteria, not certified safety requirements.

## 7. Results

These values are a snapshot of the successful run in [`target_loss_results.csv`](target_loss_results.csv) used to prepare this report. Rerunning replaces that CSV; update this section before reporting a new run.

| Check | Command latency (ms) | Result |
|---|---:|---|
| Visible target, sequence 1 | 431.955 | Pass |
| Blank-image stop, sequence 1 | 26.314 | Pass |
| Target recovery, sequence 1 | 36.544 | Pass |
| Visible target, sequence 2 | 25.701 | Pass |
| Blank-image stop, sequence 2 | 31.817 | Pass |
| Target recovery, sequence 2 | 99.622 | Pass |
| Visible target, sequence 3 | 99.877 | Pass |
| Blank-image stop, sequence 3 | 37.053 | Pass |
| Target recovery, sequence 3 | 94.515 | Pass |
| Camera-stream timeout | 619.632 | Pass |
| Camera-stream return | 88.328 | Pass |
| Short detection array | 24.563 | Pass |
| Zero-radius detection | 427.138 | Pass |

Mean blank-image stopping latency was **31.73 ms** across three events, with a range of **26.31-37.05 ms**. Mean target-recovery latency was **76.89 ms** across three events, with a range of **36.54-99.62 ms**.

All recorded stop phases ended with zero linear and angular commands. Visible-target phases ended with commanded values of 0.12 m/s and 0.25 rad/s. These are published setpoints, not measured robot velocities. Visible phases immediately following earlier recovery phases check continuity; they are not independent first-acquisition trials.

## 8. Discussion and Limitations

The run supports the intended command-level behaviour: explicit target loss produces zero commands, target return restores tracking, and interrupted detection delivery triggers the watchdog. The 619.632 ms stale-stream response illustrates the difference between the configured timeout and observed command receipt.

Initial acquisition and the zero-radius check took more than 400 ms. The measurements do not isolate the causes of this variation. Publishing within a callback does not imply instantaneous delivery elsewhere in the pipeline.

The evidence has the following limits:

- One successful run with three loss/recovery sequences is not a statistical reliability study. Earlier development attempts included a test-window failure; the reported pass count is not an estimate across all executions.
- Blank frames model complete absence. Partial occlusion, lighting changes, clutter, and competing red objects were not tested.
- The generated target is fixed in image coordinates. Robot commands do not alter the camera view, so this is not a closed-loop moving-robot experiment.
- Timings include local processing, executor scheduling, and message delivery; no real-time response bound is established.
- The CSV contains selected response measurements and final commands, rather than complete timestamped image-and-command traces.
- Physical stopping, braking distance, actual robot speed, obstacle avoidance, and Gazebo motion were not measured.

## 9. Reproduction Instructions

From Windows, double-click **Run Demo.bat** for the generated-image display or **Run Tests.bat** for a test without a display. The image window requires working WSLg support.

Alternatively, in Ubuntu:

```bash
source /opt/ros/jazzy/setup.bash
cd /mnt/c/Users/lenovo/Desktop/AI_training/ROS_Ball_Detection/Target_Loss_Recovery
python3 test_target_loss.py
# Optional image display:
python3 test_target_loss.py --show
```

The script sets its own ROS domain and discovery range. It runs local detector/controller copies without starting Gazebo or the normal robot pipeline. Successful execution writes `target_loss_results.csv` beside the script. Use `--output another_run.csv` to retain separate runs instead of replacing the default file.

## 10. Project Files

| File | Purpose |
|---|---|
| [`ball_detector.py`](ball_detector.py) | Unchanged detector copied from the active Ubuntu project |
| [`ball_controller.py`](ball_controller.py) | Updated controller and target-loss logic |
| [`test_target_loss.py`](test_target_loss.py) | Integration test and CSV recorder |
| [`ball_controller_before_target_loss.py`](ball_controller_before_target_loss.py) | Backup of the previous active controller |
| [`target_loss_results.csv`](target_loss_results.csv) | Measurements from the latest successful test |
| `Run Demo.bat` / `Run Tests.bat` | Windows launchers for the Ubuntu test |

The updated controller was also installed into `~/ball_following_ws/src/ball_follower/ball_follower/ball_controller.py`, and the ROS package was rebuilt. The older parent-folder `ball_follower_node.py` is a separate implementation and is not used in this evaluation.

## 11. Conclusion and Future Work

This project implements an inspectable extension to an existing visual tracker and demonstrates its command-level target-loss and recovery behaviour under controlled software tests. It provides a reproducible basis for further evaluation rather than evidence of physical stopping performance.

Next experiments should repeat the protocol across separately saved runs, introduce partial occlusion and distracting red regions, and measure robot motion in Gazebo. A controlled comparison with the backed-up controller would be required to quantify any stop-command latency improvement.
