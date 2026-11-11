import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist


class TurtleController(Node):

    def __init__(self):
        super().__init__('turtle_controller')

        # Create a publisher for turtle movement
        self.publisher = self.create_publisher(
            Twist,
            '/turtle1/cmd_vel',
            10
        )

        # Send movement commands every 0.5 seconds
        self.timer = self.create_timer(0.5, self.move_turtle)

        self.get_logger().info('Turtle Controller Started!')

    def move_turtle(self):

        message = Twist()

        # Move forward
        message.linear.x = 1.0

        # Do not rotate
        message.angular.z = 0.0

        self.publisher.publish(message)

        self.get_logger().info('Moving forward...')


def main(args=None):

    rclpy.init(args=args)

    controller = TurtleController()

    try:
        rclpy.spin(controller)

    except KeyboardInterrupt:
        pass

    finally:
        # Send a stop command
        stop_message = Twist()
        controller.publisher.publish(stop_message)

        controller.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()