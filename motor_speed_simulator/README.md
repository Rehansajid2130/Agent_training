# DC Motor Speed Simulator

A beginner Python project that simulates a 12 V DC motor, PWM speed control, and an encoder. The demo compares fixed PWM with feedback control when a simulated load is applied.

## Run the demo

Run `python3 motor_feedback.py` with `feedback_enabled = False`. Then change it to `True` and run the same command again. In both runs, the simulated load starts at step 21.

## Results

| Mode | PWM at step 60 | Encoder speed at step 60 |
| --- | ---: | ---: |
| Fixed PWM | 60.0% | 1,560 RPM |
| Feedback enabled | 76.6% | 2,160 RPM |

With feedback enabled, the controller increased PWM after the load slowed the motor. The encoder reading returned to the 2,160 RPM target by step 40.

## Electronics

The proposed hardware uses a motor driver to switch power from the 12 V supply. An encoder on the motor shaft sends pulses back to the controller, which estimates speed and adjusts PWM. See the [proposed hardware connections](electronics.txt). The results above come from the Python simulation.
