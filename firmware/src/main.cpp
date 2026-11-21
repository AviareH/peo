#include <Arduino.h>


void setup() {
  // put your setup code here, to run once:
  Serial0.begin(115200);
  Serial0.printf("PSRAM: %d bytes\n", ESP.getPsramSize());
}

void loop() {
  // put your main code here, to run repeatedly:
}
