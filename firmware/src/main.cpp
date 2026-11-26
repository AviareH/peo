#include <Arduino.h>
#include "as5600.h"
#include "camera_stream.h"
#include "motors.h"
#include "ps4_controller.h"


void setup() {
  // Initialize motor outputs first; the camera then starts Wi-Fi before the
  // UDP receiver is launched, so motors stay stopped until valid commands arrive.
  Serial.begin(115200);
  Serial.printf("PSRAM: %d bytes\n", ESP.getPsramSize());

  if (!startMotorControl()) {
    Serial.println("Motor control startup failed");
    return;
  }

  if (!startCameraStreaming()) {
    Serial.println("Camera streaming startup failed");
    return;
  }

  startWheelEncoders();
  if (!startPS4Controller()) {
    Serial.println("Motor command receiver startup failed");
  }
}

void loop() {
  // Camera streaming, UDP input, and PWM updates run in their own tasks.
  delay(1000);
}
