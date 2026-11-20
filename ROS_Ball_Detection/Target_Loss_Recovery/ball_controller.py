
#!/usr/bin/env python3

import rclpy
from time import monotonic

from rclpy.node import Node

from std_msgs.msg import Int32MultiArray
from geometry_msgs.msg import Twist


class BallController(Node):

    def __init__(self):
        super().__init__('ball_controller')

        # Subscribe to OpenCV detection coordinates
        self.ball_sub = self.create_subscription(
            Int32MultiArray,
            '/ball_detection/pixels',
            self.ball_callback,
            10
        )

        # Publish robot movement commands
        self.cmd_pub = self.create_publisher(
            Twist,
            '/cmd_vel',
            10
        )

        # Camera configuration
        self.image_center_x = 320

        # Desired apparent ball radius
        self.target_radius = 105

        # Controller settings
        self.kp_angular = 0.0025
        self.kp_linear = 0.003

        # Safety limits
        self.max_linear_speed = 0.12
        self.max_reverse_speed = 0.05
        self.max_angular_speed = 0.5

        # Deadbands
        self.horizontal_deadband = 12
        self.radius_deadband = 5

        # Detection state
        self.ball_visible = False
        self.last_seen_time = None
        self.tracking_state = 'WAITING'

        self.ball_x = 0
        self.ball_y = 0
        self.ball_radius = 0

        # Safety timeout
        self.detection_timeout = 0.5

        # Controller runs at 10 Hz
        self.control_timer = self.create_timer(
            0.1,
            self.control_loop
        )

        self.get_logger().info(
            'Autonomous ball controller started!'
        )

        self.get_logger().info(
            'Waiting for ball detection...'
        )

    def ball_callback(self, msg):

        # Empty detection means the ball is not visible
        if len(msg.data) < 3:
            self.ball_visible = False
            self.set_tracking_state('LOST')
            self.publish_stop()
            return

        cx, cy, radius = msg.data[:3]

        if radius <= 0:
            self.ball_visible = False
            self.set_tracking_state('LOST')
            self.publish_stop()
            return

        self.ball_x = cx
        self.ball_y = cy
        self.ball_radius = radius

        self.ball_visible = True
        self.last_seen_time = monotonic()
        self.set_tracking_state('TRACKING')

    def set_tracking_state(self, state):
        # Log transitions once, rather than repeating the same message each frame.
        if state != self.tracking_state:
            self.tracking_state = state
            messages = {
                'TRACKING': 'Target detected: tracking enabled.',
                'LOST': 'Target lost: stopped; waiting for a new detection.',
                'STALE': 'Detection stream timed out: stopped.',
            }
            self.get_logger().info(messages[state])

    def publish_stop(self):

        command = Twist()

        self.cmd_pub.publish(command)

    def control_loop(self):

        # Stop if ball is not detected
        if not self.ball_visible:
            self.publish_stop()
            return

        # Check whether detection data is outdated
        if self.last_seen_time is None:
            self.publish_stop()
            return

        elapsed = monotonic() - self.last_seen_time

        if elapsed > self.detection_timeout:

            self.ball_visible = False
            self.set_tracking_state('STALE')
            self.publish_stop()

            return

        command = Twist()

        # ---------------------------------
        # STEERING CONTROL
        # ---------------------------------

        horizontal_error = (
            self.image_center_x - self.ball_x
        )

        if abs(horizontal_error) <= self.horizontal_deadband:

            command.angular.z = 0.0

        else:

            angular_speed = (
                self.kp_angular * horizontal_error
            )

            command.angular.z = max(
                -self.max_angular_speed,
                min(
                    self.max_angular_speed,
                    angular_speed
                )
            )

        # ---------------------------------
        # DISTANCE CONTROL
        # ---------------------------------

        radius_error = (
            self.target_radius - self.ball_radius
        )

        if abs(radius_error) <= self.radius_deadband:

            command.linear.x = 0.0

        else:

            linear_speed = (
                self.kp_linear * radius_error
            )

            command.linear.x = max(
                -self.max_reverse_speed,
                min(
                    self.max_linear_speed,
                    linear_speed
                )
            )

        # Publish robot movement
        self.cmd_pub.publish(command)


def main(args=None):

    rclpy.init(args=args)

    node = BallController()

    try:
        rclpy.spin(node)

    except KeyboardInterrupt:
        pass

    finally:

        # Stop robot before shutting down
        node.publish_stop()

        node.destroy_node()

        rclpy.shutdown()


if __name__ == '__main__':
    main()
