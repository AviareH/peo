#pragma once

// Signed PWM requests: positive is forward, negative is reverse, zero is stop.
struct MotorCommand {
  int leftSpeed;
  int rightSpeed;
  bool cutZero;

  MotorCommand(int left = 0, int right = 0, bool cut = false)
      : leftSpeed(left), rightSpeed(right), cutZero(cut) {}
};

// Configure PWM and start the core-1 task that ramps toward queued requests.
bool initializeMotors();
bool startMotorControl();
// Replace any pending request with this latest left/right command.
void sendMotorCommand(MotorCommand command);
void stopMotorControl();
