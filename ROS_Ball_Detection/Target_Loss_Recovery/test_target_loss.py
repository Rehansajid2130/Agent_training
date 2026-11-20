"""Exercise the real ROS detector/controller using generated images, without Gazebo.

Run inside Ubuntu after sourcing ROS 2. --show displays the generated camera feed.
An isolated ROS domain is used so test commands cannot reach the usual simulation.
"""

import argparse
import csv
import os
from pathlib import Path
import time

os.environ['ROS_DOMAIN_ID'] = '77'
os.environ.pop('ROS_LOCALHOST_ONLY', None)
os.environ['ROS_AUTOMATIC_DISCOVERY_RANGE'] = 'LOCALHOST'

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from geometry_msgs.msg import Twist
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Int32MultiArray

from ball_controller import BallController
from ball_detector import BallDetector


def run(show, output):
    rclpy.init()
    detector, controller = BallDetector(), BallController()
    probe = Node('target_loss_test')
    image_pub = probe.create_publisher(Image, '/camera/image', 10)
    detection_pub = probe.create_publisher(Int32MultiArray, '/ball_detection/pixels', 10)
    commands, detections, rows = [], [], []
    command_sub = probe.create_subscription(
        Twist, '/cmd_vel',
        lambda msg: commands.append((time.monotonic(), msg.linear.x, msg.angular.z)), 10)
    detection_sub = probe.create_subscription(
        Int32MultiArray, '/ball_detection/pixels',
        lambda msg: detections.append((time.monotonic(), list(msg.data))), 10)
    bridge = CvBridge()
    executor = SingleThreadedExecutor()
    for node in (detector, controller, probe):
        executor.add_node(node)

    def pump(seconds, visible=None):
        end, next_frame = time.monotonic() + seconds, 0.0
        while time.monotonic() < end:
            now = time.monotonic()
            if visible is not None and now >= next_frame:
                frame = np.full((480, 640, 3), 40, dtype=np.uint8)
                if visible:
                    cv2.circle(frame, (220, 240), 45, (0, 0, 255), -1)
                image_pub.publish(bridge.cv2_to_imgmsg(frame, encoding='bgr8'))
                if show:
                    display = frame.copy()
                    cv2.putText(display, controller.tracking_state,
                                (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                    cv2.imshow('Target loss test - generated camera', display)
                    cv2.waitKey(1)
                next_frame = now + 0.05
            executor.spin_once(timeout_sec=0.005)

    def record(name, started, expect_motion, reference=None, limit=0.5):
        received = [c for c in commands if c[0] >= started]
        matching = [c for c in received if (abs(c[1]) + abs(c[2]) > 1e-8) == expect_motion]
        assert matching, f'{name}: expected command never received'
        latency = matching[0][0] - (started if reference is None else reference)
        assert 0 <= latency < limit, f'{name}: excessive latency {latency:.3f}s'
        final_is_moving = abs(received[-1][1]) + abs(received[-1][2]) > 1e-8
        assert final_is_moving == expect_motion, name
        if not expect_motion:
            assert all(c[1:] == (0.0, 0.0) for c in received if c[0] >= matching[0][0]), name
        rows.append(dict(test=name, expected='motion' if expect_motion else 'stop',
                         command_latency_s=round(latency, 6),
                         final_linear_mps=received[-1][1],
                         final_angular_rps=received[-1][2], result='PASS'))
        print(f'{name}: PASS ({latency:.3f}s)', flush=True)

    try:
        # Allow DDS discovery before measuring the first response.
        deadline = time.monotonic() + 8
        while (image_pub.get_subscription_count() < 1 or command_sub.get_publisher_count() < 1
               or detection_sub.get_publisher_count() < 2):
            assert time.monotonic() < deadline, 'ROS topic discovery timed out'
            executor.spin_once(timeout_sec=0.05)
        pump(0.3)
        assert commands and commands[-1][1:] == (0.0, 0.0), 'Startup must be stopped'
        for trial in range(1, 4):
            started = time.monotonic()
            pump(0.7, True)
            record(f'visible_{trial}', started, True)
            assert any(data for stamp, data in detections if stamp >= started)
            started = time.monotonic()
            pump(0.7, False)
            record(f'blank_frame_stop_{trial}', started, False)
            assert any(not data for stamp, data in detections if stamp >= started)
            started = time.monotonic()
            pump(0.7, True)
            record(f'reacquire_{trial}', started, True)

        # A frozen feed produces no empty messages: the watchdog must still stop.
        started = time.monotonic()
        reference = controller.last_seen_time
        pump(1.0)
        record('camera_feed_timeout', started, False, reference,
               controller.detection_timeout + 0.25)
        assert controller.tracking_state == 'STALE'
        started = time.monotonic()
        pump(0.7, True)
        record('feed_return', started, True)

        # Inject invalid detection messages; there are no camera frames during these phases.
        for name, data in [('malformed_detection', [220]), ('zero_radius', [220, 240, 0])]:
            started = time.monotonic()
            msg = Int32MultiArray()
            msg.data = data
            detection_pub.publish(msg)
            pump(0.7)
            record(name, started, False)
            pump(0.4, True)

        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open('w', newline='', encoding='utf-8') as stream:
            writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        print(f'All {len(rows)} checks passed. Results: {output}', flush=True)
    finally:
        controller.publish_stop()
        executor.shutdown()
        for node in (probe, controller, detector):
            node.destroy_node()
        rclpy.shutdown()
        if show:
            cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--show', action='store_true')
    parser.add_argument('--output', type=Path,
                        default=Path(__file__).with_name('target_loss_results.csv'))
    args = parser.parse_args()
    run(args.show, args.output)
