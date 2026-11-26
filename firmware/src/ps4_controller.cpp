#include <Arduino.h>
#include <cstdio>
#include <WiFi.h>
#include <WiFiUdp.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "motors.h"
#include "ps4_controller.h"

static constexpr uint16_t MOTOR_COMMAND_PORT = 4210;
static constexpr uint32_t COMMAND_TIMEOUT_MS = 150;
static constexpr size_t COMMAND_BUFFER_SIZE = 32;

// Shared by the receiver task after the listener is started.
static WiFiUDP commandSocket;

static bool parseMotorCommand(const char* packet, int* left, int* right) {
  // The host sends signed PWM values as "left,right"; reject extra data and
  // constrain both sides to the motor driver's 8-bit duty range.
  char trailing = '\0';
  if (std::sscanf(packet, "%d,%d%c", left, right, &trailing) != 2) {
    return false;
  }
  *left = constrain(*left, -255, 255);
  *right = constrain(*right, -255, 255);
  return true;
}

static void motorCommandReceiverTask(void* parameters) {
  (void)parameters;
  char packet[COMMAND_BUFFER_SIZE];
  // Start the failsafe timer immediately so motors stop if no host connects.
  uint32_t lastCommandTime = millis();
  bool stoppedForTimeout = false;

  while (true) {
    const int packetLength = commandSocket.parsePacket();
    if (packetLength > 0) {
      if (packetLength < static_cast<int>(sizeof(packet))) {
        const int bytesRead = commandSocket.read(packet, sizeof(packet) - 1);
        if (bytesRead > 0) {
          packet[bytesRead] = '\0';
          int left = 0;
          int right = 0;
          if (parseMotorCommand(packet, &left, &right)) {
            // The motor task consumes the newest target and ramps toward it.
            sendMotorCommand({left, right});
            lastCommandTime = millis();
            stoppedForTimeout = false;
          }
        }
      } else {
        // Discard oversized datagrams instead of parsing a truncated command.
        while (commandSocket.available()) {
          commandSocket.read();
        }
      }
    }

    // Stop if the computer or Wi-Fi link stops sending fresh drive commands.
    if (!stoppedForTimeout &&
        millis() - lastCommandTime > COMMAND_TIMEOUT_MS) {
      sendMotorCommand({0, 0});
      stoppedForTimeout = true;
    }
    // Poll frequently enough for responsive control and a prompt link timeout.
    vTaskDelay(pdMS_TO_TICKS(5));
  }
}

bool startPS4Controller() {
  // The camera startup brings up the ESP32 access point before this listener.
  if (WiFi.getMode() == WIFI_OFF) {
    Serial0.println("Wi-Fi must be started before motor command receiver");
    return false;
  }

  if (commandSocket.begin(MOTOR_COMMAND_PORT) != 1) {
    Serial0.println("Motor command UDP listener startup failed");
    return false;
  }

  // Keep UDP polling asynchronous so startup and camera streaming stay unblocked.
  // Core 1 keeps command handling away from the camera server on core 0.
  const BaseType_t result = xTaskCreatePinnedToCore(
      motorCommandReceiverTask,
      "MotorCommandRx",
      3072,
      nullptr,
      4,
      nullptr,
      1
  );
  if (result != pdPASS) {
    commandSocket.stop();
    Serial0.println("Motor command receiver task creation failed");
    return false;
  }

  Serial0.printf("Motor command listener ready on UDP port %u\n",
                 MOTOR_COMMAND_PORT);
  return true;
}
