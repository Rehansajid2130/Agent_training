supply_voltage = 12.0
duty_cycle = 0.60
load_rpm = 1600
rpm = 0.0

average_voltage = supply_voltage * duty_cycle
steady_speed = max(0, 300 * average_voltage - load_rpm)

print(f"PWM duty cycle: {duty_cycle:.0%}")
print(f"Average motor voltage: {average_voltage:.1f} V")

for step in range(1, 11):
    rpm += 0.25 * (steady_speed - rpm)
    print(f"Step {step:2}: {rpm:6.0f} RPM")
