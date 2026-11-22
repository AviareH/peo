#include <Arduino.h>
#include "camera_stream.h"
#include "motors.h"


void setup() {
  // put your setup code here, to run once:
  Serial0.begin(115200);
  Serial0.printf("PSRAM: %d bytes\n", ESP.getPsramSize());

  if (!startMotorControl()) {
    Serial0.println("Motor control startup failed");
    return;
  }

  sendMotorCommand({100, 150});

  startCameraStreaming();
}

void loop() {
  // put your main code here, to run repeatedly:
  delay(1000);
}
