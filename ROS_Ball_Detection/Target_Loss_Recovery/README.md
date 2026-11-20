# Target Loss and Recovery - Simple Visual Tracking Extension

This extends Rehan Sajid's existing ROS 2 red-ball tracking project. It uses the actual OpenCV detector and proportional controller with generated camera images for a repeatable test.

## Try it

Double-click **Run Demo.bat** to see a generated red target disappear and return. The test runs automatically, prints results, and saves `target_loss_results.csv` here. Ubuntu 24.04, ROS 2 Jazzy and the existing OpenCV/cv_bridge installation are required. The image window uses WSLg. Use **Run Tests.bat** if the image window is unavailable; it runs the same checks without a window.

The test uses ROS domain 77 on localhost, separate from the normal project's default domain 0. Do not run other nodes in domain 77 during the test. It does not start Gazebo or control the Gazebo robot. Rerunning replaces the generated results CSV.

## What was already implemented?

The active Ubuntu controller already stopped on an empty detection or detection timeout and resumed on a new valid detection. The old Windows `ball_follower_node.py` instead searched by rotating and used a different colour threshold. That older file is not the active red-target pipeline and has not been changed.

## What this extension changes

- An empty or invalid detection now publishes zero linear AND angular velocity immediately in the detection callback, rather than waiting for the next 10 Hz control tick.
- Explicit WAITING, TRACKING, LOST and STALE states make the behaviour visible in transition logs.
- A monotonic wall clock measures the existing 0.5-second detection timeout, independent of ROS clock changes. The 10 Hz loop means a stale stop normally occurs at the first control tick after that timeout; scheduling and message delivery add latency.
- A repeatable test drives the real detector and controller with red-target images, blank images, interrupted image delivery, and invalid detection messages.

Fresh detection resumes normal proportional control on the next control tick. It does not necessarily command motion: a target already at the desired position and size produces a zero command. This test deliberately places the target off-centre and farther away so resumption produces nonzero commands.

## Simple explanation for an interview

1. OpenCV thresholds red pixels in HSV colour space and extracts the largest qualifying contour.
2. The detector publishes `[centre_x, centre_y, radius]`, or an empty array if nothing qualifies.
3. For a valid target, angular velocity is proportional to horizontal pixel error. Linear velocity is proportional to apparent-radius error. Deadbands reduce small corrections and speed limits bound the commands.
4. An empty detection immediately sends a zero command. The robot does not search while the target is absent.
5. If the whole image/detection stream stops, a watchdog sends a zero command after the timeout.
6. A new valid detection restores tracking. There is no prediction or learned recovery method.

The image radius is a distance proxy, not a calibrated distance in metres. The detector identifies a red region, not the identity of a particular ball; another large red object could be selected.

## Evaluation and honest interpretation

The script checks startup stop, then records 13 named checks: three visible-target phases, three loss stops, three reacquisitions, one stream timeout, one stream recovery, and two invalid-input stops. Startup is checked but is not a separate CSV row.

Command latency for a blank frame is measured from the start of the blank-image phase until the subscriber receives the first zero command. Recovery latency is measured from the start of the visible-image phase until the first nonzero command. These include local image processing and ROS message delivery. The timeout latency starts at the controller's last valid detection receipt. All timings use a monotonic clock.

These are software integration tests with generated images. They do NOT measure physical braking distance, actual robot speed, real-camera performance, Gazebo movement, natural occlusion robustness, or target-recognition accuracy. A blank frame represents total target absence; partial occlusion is not evaluated.

## Files and active Ubuntu package

- `ball_controller.py`: updated controller, also installed into the active Ubuntu source package.
- `ball_detector.py`: unchanged copy of the active detector.
- `test_target_loss.py`: repeatable ROS integration test.
- `ball_controller_before_target_loss.py`: backup of the previous active controller.
- `target_loss_results.csv`: actual measurements from the latest successful test.

The active source is `~/ball_following_ws/src/ball_follower/ball_follower/ball_controller.py`. Run the normal project's detector and controller in its usual domain for Gazebo use. The test's domain setting applies only to its own process.

## Before adding it to the CV

Run the demo, inspect the CSV, and explain the six steps above in your own words. A suitable description after your review is: "Extended a ROS 2 visual tracker with immediate stop commands on target loss, stale-detection timeout handling, transition logging, and repeatable loss/recovery tests using generated images."
