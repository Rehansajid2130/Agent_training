@echo off
wsl -d Ubuntu-24.04 -e bash -lc "source /opt/ros/jazzy/setup.bash && python3 /mnt/c/Users/lenovo/Desktop/AI_training/ROS_Ball_Detection/Target_Loss_Recovery/test_target_loss.py --show"
pause
