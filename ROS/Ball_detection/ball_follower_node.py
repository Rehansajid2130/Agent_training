import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
import numpy as np

# Optional OpenCV import (handled gracefully if environment is headless)
try:
    import cv2
except ImportError:
    cv2 = None

try:
    from cv_bridge import CvBridge
except ImportError:
    CvBridge = None


class BallFollowerNode(Node):
    """
    ROS 2 Node that detects a colored ball in a camera feed using OpenCV,
    and publishes velocity commands to make the robot turn and follow the ball.
    """

    def __init__(self):
        super().__init__('ball_follower_node')

        # -------------------------------------------------------------
        # Parameters (Can be dynamically tuned in ROS 2)
        # -------------------------------------------------------------
        self.declare_parameter('camera_topic', '/camera/image_raw')
        self.declare_parameter('cmd_vel_topic', '/cmd_vel')
        self.declare_parameter('target_radius_pixels', 65.0)  # Desired ball size (distance)
        self.declare_parameter('linear_speed_max', 0.4)       # m/s
        self.declare_parameter('angular_speed_max', 1.0)      # rad/s
        self.declare_parameter('search_angular_speed', 0.3)   # rad/s when scanning
        self.declare_parameter('show_debug_window', True)     # Pop up live visual window on screen

        self.camera_topic = self.get_parameter('camera_topic').value
        self.cmd_vel_topic = self.get_parameter('cmd_vel_topic').value
        self.target_radius = self.get_parameter('target_radius_pixels').value
        self.linear_speed_max = self.get_parameter('linear_speed_max').value
        self.angular_speed_max = self.get_parameter('angular_speed_max').value
        self.search_speed = self.get_parameter('search_angular_speed').value
        self.show_debug = self.get_parameter('show_debug_window').value

        # -------------------------------------------------------------
        # HSV Color Thresholds (Default: Bright Tennis Ball / Yellow-Orange)
        # -------------------------------------------------------------
        self.hsv_lower = np.array([20, 100, 100], dtype=np.uint8)
        self.hsv_upper = np.array([35, 255, 255], dtype=np.uint8)

        # -------------------------------------------------------------
        # CV Bridge & Communication
        # -------------------------------------------------------------
        self.bridge = CvBridge() if CvBridge is not None else None

        self.image_sub = self.create_subscription(
            Image,
            self.camera_topic,
            self.image_callback,
            10
        )

        self.cmd_vel_pub = self.create_publisher(
            Twist,
            self.cmd_vel_topic,
            10
        )

        self.debug_image_pub = self.create_publisher(
            Image,
            '/camera/ball_tracking_debug',
            10
        )

        self.last_seen_ball = False
        self.frame_count = 0
        self.get_logger().info(f"Ball Follower started! Listening to '{self.camera_topic}'...")

        # Timer to notify user if waiting for camera feed
        self.heartbeat_timer = self.create_timer(3.0, self.check_camera_heartbeat)

    def check_camera_heartbeat(self):
        if self.frame_count == 0:
            self.get_logger().info(f"⏳ Still waiting for camera frames on '{self.camera_topic}'... (Run a camera publisher or simulation)")

    def convert_ros_to_cv(self, msg: Image):
        """Converts sensor_msgs/Image to OpenCV BGR image with or without cv_bridge."""
        if self.bridge is not None:
            return self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        
        # Fallback raw buffer conversion if cv_bridge is not installed
        img_array = np.frombuffer(msg.data, dtype=np.uint8)
        if msg.encoding in ['rgb8', 'bgr8']:
            img = img_array.reshape((msg.height, msg.width, 3))
            return cv2.cvtColor(img, cv2.COLOR_RGB2BGR) if msg.encoding == 'rgb8' else img
        return None

    def convert_cv_to_ros(self, cv_img):
        """Converts an OpenCV image back to sensor_msgs/Image for ROS topic publishing."""
        if self.bridge is not None:
            return self.bridge.cv2_to_imgmsg(cv_img, encoding='bgr8')
        return None

    def image_callback(self, msg: Image):
        self.frame_count += 1
        if cv2 is None:
            self.get_logger().error("OpenCV (cv2) is not available. Please install python3-opencv.")
            return

        cv_img = self.convert_ros_to_cv(msg)
        if cv_img is None:
            return

        h, w, _ = cv_img.shape
        center_x = w // 2

        # 1. Perception: Convert to HSV and extract mask
        blurred = cv2.GaussianBlur(cv_img, (11, 11), 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.hsv_lower, self.hsv_upper)
        mask = cv2.erode(mask, None, iterations=2)
        mask = cv2.dilate(mask, None, iterations=2)

        # 2. Find ball contour
        contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        ball_found = False
        twist = Twist()

        if len(contours) > 0:
            # Pick the largest circular contour
            largest_contour = max(contours, key=cv2.contourArea)
            ((x, y), radius) = cv2.minEnclosingCircle(largest_contour)

            if radius > 8.0:  # Minimum filter threshold
                ball_found = True
                self.last_seen_ball = True

                # Calculate Proportional Error
                # Angular: Horizontal deviation from image center (-1.0 to 1.0)
                error_x = (center_x - x) / float(center_x)
                angular_z = error_x * self.angular_speed_max

                # Linear: Radius error (positive = ball is far away, move forward)
                error_radius = (self.target_radius - radius) / float(self.target_radius)
                linear_x = max(0.0, error_radius * self.linear_speed_max)

                # Stop moving forward if close enough
                if radius >= self.target_radius:
                    linear_x = 0.0

                twist.linear.x = float(np.clip(linear_x, 0.0, self.linear_speed_max))
                twist.angular.z = float(np.clip(angular_z, -self.angular_speed_max, self.angular_speed_max))

                if self.frame_count % 15 == 0:
                    self.get_logger().info(
                        f"🎯 Ball detected at (x={int(x)}, y={int(y)}, r={radius:.1f}) -> "
                        f"Linear: {twist.linear.x:.2f} m/s, Angular: {twist.angular.z:.2f} rad/s"
                    )

                # Draw tracking visuals
                cv2.circle(cv_img, (int(x), int(y)), int(radius), (0, 255, 0), 2)
                cv2.circle(cv_img, (int(x), int(y)), 4, (0, 0, 255), -1)
                cv2.putText(
                    cv_img,
                    f"Radius: {radius:.1f} | Speed: {twist.linear.x:.2f} m/s",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2
                )
        
        # 3. Search Mode: If ball is lost, slowly rotate in place to locate it
        if not ball_found:
            twist.linear.x = 0.0
            twist.angular.z = self.search_speed
            if self.frame_count % 30 == 0:
                self.get_logger().info("🔍 Ball not in frame. Scanning 360°...")
            cv2.putText(
                cv_img,
                "STATUS: Searching for Ball...",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2
            )

        # Publish velocity commands to the robot
        self.cmd_vel_pub.publish(twist)

        # Publish debug image to ROS topic
        debug_msg = self.convert_cv_to_ros(cv_img)
        if debug_msg is not None:
            self.debug_image_pub.publish(debug_msg)

        # Display desktop window if enabled
        if self.show_debug:
            cv2.imshow("Robot Vision - Ball Tracking", cv_img)
            cv2.waitKey(1)


def main(args=None):
    rclpy.init(args=args)
    node = BallFollowerNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        # Emergency stop on exit (fail-safe)
        try:
            if rclpy.ok():
                stop_twist = Twist()
                node.cmd_vel_pub.publish(stop_twist)
        except Exception:
            pass
        if cv2 is not None:
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()

