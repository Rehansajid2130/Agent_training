import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import ExecuteProcess
from launch_ros.actions import Node


def generate_launch_description():
    pkg_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    urdf_path = os.path.join(pkg_dir, 'urdf', 'ball_follower_robot.urdf')
    world_path = os.path.join(pkg_dir, 'worlds', 'ball_world.sdf')

    with open(urdf_path, 'r') as infp:
        robot_description = infp.read()

    return LaunchDescription([
        # 1. Start Gazebo with the custom ball world
        ExecuteProcess(
            cmd=['gazebo', '--verbose', world_path, '-s', 'libgazebo_ros_init.so', '-s', 'libgazebo_ros_factory.so'],
            output='screen'
        ),

        # 2. Publish Robot State / TF transforms
        Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            output='screen',
            parameters=[{'robot_description': robot_description}]
        ),

        # 3. Spawn Robot Model in Gazebo
        Node(
            package='gazebo_ros',
            executable='spawn_entity.py',
            arguments=['-topic', 'robot_description', '-entity', 'ball_follower_robot', '-z', '0.1'],
            output='screen'
        ),

        # 4. Launch OpenCV Ball-Following Perception & Control Node
        Node(
            package='ball_follower',
            executable='ball_follower_node',
            name='ball_follower_node',
            output='screen',
            parameters=[{
                'camera_topic': '/camera/image_raw',
                'cmd_vel_topic': '/cmd_vel',
                'target_radius_pixels': 65.0,
                'linear_speed_max': 0.4,
                'angular_speed_max': 1.0,
                'search_angular_speed': 0.3,
                'show_debug_window': True
            }]
        )
    ])

