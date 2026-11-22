#pragma once

struct MotorCommand {
  int leftSpeed;
  int rightSpeed;
};

void initializeMotors();
bool startMotorControl();
void sendMotorCommand(MotorCommand command);