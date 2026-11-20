
#!/usr/bin/env python3

import rclpy
import cv2
import numpy as np

from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from sensor_msgs.msg import Image
from std_msgs.msg import Int32MultiArray

from cv_bridge import CvBridge, CvBridgeError


class BallDetector(Node):

    def __init__(self):
        super().__init__('ball_detector')

        self.bridge = CvBridge()

        # Subscribe to the robot's camera
        self.image_sub = self.create_subscription(
            Image,
            '/camera/image',
            self.image_callback,
            qos_profile_sensor_data
        )

        # Publish the annotated camera image
        self.image_pub = self.create_publisher(
            Image,
            '/ball_detection/annotated',
            10
        )

        # Publish ball center coordinates
        self.center_pub = self.create_publisher(
            Int32MultiArray,
            '/ball_detection/pixels',
            10
        )

        self.get_logger().info('Ball detector started!')
        self.get_logger().info('Waiting for camera images...')

    def image_callback(self, msg):

        try:
            # Convert ROS image into OpenCV image
            frame = self.bridge.imgmsg_to_cv2(
                msg,
                desired_encoding='bgr8'
            )

        except CvBridgeError as error:
            self.get_logger().error(f'Image conversion failed: {error}')
            return

        # Convert BGR image to HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Red has two ranges in HSV
        lower_red_1 = np.array([0, 100, 80])
        upper_red_1 = np.array([10, 255, 255])

        lower_red_2 = np.array([170, 100, 80])
        upper_red_2 = np.array([180, 255, 255])

        # Create masks for both red ranges
        mask_1 = cv2.inRange(
            hsv,
            lower_red_1,
            upper_red_1
        )

        mask_2 = cv2.inRange(
            hsv,
            lower_red_2,
            upper_red_2
        )

        mask = cv2.bitwise_or(mask_1, mask_2)

        # Remove small noise
        kernel = np.ones((5, 5), np.uint8)

        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_OPEN,
            kernel
        )

        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_CLOSE,
            kernel
        )

        # Find object contours
        contours, _ = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        detection = Int32MultiArray()

        if contours:

            # Select the largest detected red object
            largest = max(contours, key=cv2.contourArea)

            area = cv2.contourArea(largest)

            if area > 100:

                moments = cv2.moments(largest)

                if moments['m00'] > 0:

                    # Calculate center coordinates
                    cx = int(
                        moments['m10'] / moments['m00']
                    )

                    cy = int(
                        moments['m01'] / moments['m00']
                    )

                    # Calculate approximate radius
                    (x, y), radius = cv2.minEnclosingCircle(
                        largest
                    )

                    radius = int(radius)

                    # Draw circle around the ball
                    if radius > 2:

                        cv2.circle(
                            frame,
                            (int(x), int(y)),
                            radius,
                            (0, 255, 0),
                            2
                        )

                    # Draw center point
                    cv2.circle(
                        frame,
                        (cx, cy),
                        5,
                        (255, 0, 0),
                        -1
                    )

                    cv2.putText(
                        frame,
                        f'Ball: ({cx}, {cy})',
                        (cx + 10, cy - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 0),
                        2
                    )

                    # Publish pixel coordinates and radius
                    detection.data = [cx, cy, radius]

                else:
                    detection.data = []

            else:
                detection.data = []

        else:
            detection.data = []

        # Publish detection information
        self.center_pub.publish(detection)

        # Publish annotated image
        try:
            output_msg = self.bridge.cv2_to_imgmsg(
                frame,
                encoding='bgr8'
            )

            output_msg.header = msg.header

            self.image_pub.publish(output_msg)

        except CvBridgeError as error:
            self.get_logger().error(
                f'Output image conversion failed: {error}'
            )


def main(args=None):

    rclpy.init(args=args)

    node = BallDetector()

    try:
        rclpy.spin(node)

    except KeyboardInterrupt:
        pass

    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
