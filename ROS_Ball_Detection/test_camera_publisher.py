import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None

try:
    from cv_bridge import CvBridge
except ImportError:
    CvBridge = None


class TestCameraPublisher(Node):
    """
    Publishes synthetic camera frames with a moving yellow ball
    so you can test ball_follower_node without needing Gazebo or a real webcam!
    """

    def __init__(self):
        super().__init__('test_camera_publisher')

        self.publisher_ = self.create_publisher(Image, '/camera/image_raw', 10)
        self.timer = self.create_timer(1.0 / 30.0, self.timer_callback)  # 30 FPS
        self.bridge = CvBridge() if CvBridge is not None else None

        # Simulated Ball Physics
        self.width = 640
        self.height = 480
        self.ball_x = 320.0
        self.ball_y = 240.0
        self.ball_vx = 3.0
        self.ball_vy = 1.5
        self.radius = 45

        self.get_logger().info("🎥 Test Camera Publisher started! Publishing animated ball to '/camera/image_raw'...")

    def timer_callback(self):
        if cv2 is None:
            self.get_logger().error("OpenCV (cv2) is required. Run: sudo apt install python3-opencv")
            return

        # 1. Update ball position (bouncing inside screen)
        self.ball_x += self.ball_vx
        self.ball_y += self.ball_vy

        if self.ball_x - self.radius < 20 or self.ball_x + self.radius > self.width - 20:
            self.ball_vx *= -1.0
        if self.ball_y - self.radius < 20 or self.ball_y + self.radius > self.height - 20:
            self.ball_vy *= -1.0

        # 2. Draw synthetic frame
        frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        frame[:] = (50, 50, 50)  # Dark gray background

        # Draw bright yellow/tennis ball (BGR: 0, 220, 255)
        cv2.circle(frame, (int(self.ball_x), int(self.ball_y)), self.radius, (0, 220, 255), -1)
        cv2.putText(
            frame,
            "SIMULATED CAMERA FEED",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        # 3. Convert to ROS 2 Image message
        if self.bridge is not None:
            msg = self.bridge.cv2_to_imgmsg(frame, encoding='bgr8')
        else:
            msg = Image()
            msg.height = self.height
            msg.width = self.width
            msg.encoding = 'bgr8'
            msg.step = self.width * 3
            msg.data = frame.tobytes()

        self.publisher_.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = TestCameraPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()

