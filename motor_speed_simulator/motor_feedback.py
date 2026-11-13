supply_voltage = 12.0
target_rpm = 2160
pulses_per_revolution = 20
sample_seconds = 0.1
feedback_enabled = True

duty_cycle = 0.60
rpm = 0.0

print("Step  Load   PWM    Volts  Pulses  Encoder RPM")

for step in range(1, 61):
    load_rpm = 600 if step >= 21 else 0
    voltage = supply_voltage * duty_cycle
    steady_rpm = max(0, 300 * voltage - load_rpm)
    rpm += 0.25 * (steady_rpm - rpm)

    revolutions = rpm * sample_seconds / 60
    pulses = round(revolutions * pulses_per_revolution)
    measured_rpm = pulses * 60 / (pulses_per_revolution * sample_seconds)

    if step in (10, 20, 21, 25, 30, 40, 50, 60):
        print(f"{step:4}  {load_rpm:4}  {duty_cycle:6.1%}  "
              f"{voltage:5.2f}  {pulses:6}  {measured_rpm:11.0f}")

    if feedback_enabled:
        error = target_rpm - measured_rpm
        duty_cycle += 0.0002 * error
        duty_cycle = min(1.0, max(0.0, duty_cycle))
