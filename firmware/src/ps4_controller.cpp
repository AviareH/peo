#include <Arduino.h>
#include <atomic>
#include <WiFi.h>
#include <WiFiUdp.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "motors.h"
#include "ps4_controller.h"
#include "drive_input.h"
#include "encoder_config.h"
#include "vision_drive.h"

static constexpr uint16_t MOTOR_COMMAND_PORT = 4210;
static constexpr uint32_t COMMAND_TIMEOUT_MS = 150;
static constexpr size_t COMMAND_BUFFER_SIZE = 64;

// Shared by the receiver task after the listener is started.
static WiFiUDP commandSocket;

static std::atomic<bool> autonomous{false};

const char* driveModeName() {
  return autonomous.load() ? "AUTONOMOUS" : "MANUAL";
}

static void applyMotorAction(InputAction action, MotorCommand pwm) {
  if (action == InputAction::Stop) stopMotorControl();
  if (action == InputAction::Drive) sendMotorCommand(pwm);
}

static void motorCommandReceiverTask(void* parameters) {
  (void)parameters;
  char packet[COMMAND_BUFFER_SIZE];
  DriveInput input;
  VisionDrive vision({encoderConfig::leftCountsPerMetre, encoderConfig::rightCountsPerMetre,
                     encoderConfig::trackWidthMm});
  // Start the failsafe timer immediately so motors stop if no host connects.
  uint32_t lastCommandTime = millis();
  bool stoppedForTimeout = true;

  while (true) {
    const EncoderSample encoders = readWheelEncoders();
    const uint32_t now = millis();
    if (!stoppedForTimeout && now - lastCommandTime > COMMAND_TIMEOUT_MS) {
      stopMotorControl();
      input.disconnect();
      vision.setSession(0);
      autonomous.store(false);
      stoppedForTimeout = true;
    }
    MotorCommand pwm{};
    const auto tick = vision.update(now, pwm, encoders);
    applyMotorAction(tick, pwm);

    const int packetLength = commandSocket.parsePacket();
    if (packetLength > 0) {
      if (packetLength < static_cast<int>(sizeof(packet))) {
        const int bytesRead = commandSocket.read(packet, sizeof(packet) - 1);
        if (bytesRead == packetLength && !std::memchr(packet, 0, bytesRead)) {
          packet[bytesRead] = '\0';
          const uint64_t peer = (uint64_t(uint32_t(commandSocket.remoteIP())) << 16) |
                                commandSocket.remotePort();
          if (std::strncmp(packet, "CV,", 3) == 0 || std::strncmp(packet, "ENC,", 4) == 0) {
            VisionRequest request{};
            if (input.isPadHost(peer) && VisionRequest::parse(packet, request)) {
              const IPAddress remote = commandSocket.remoteIP();
              const uint16_t port = commandSocket.remotePort();
              const auto action = vision.receive(request, now, pwm, encoders);
              applyMotorAction(action, pwm);
              char reply[COMMAND_BUFFER_SIZE];
              const int size = std::snprintf(reply, sizeof(reply), "CV,%llu,%llu,%llu,%s",
                                             request.sequence, (unsigned long long)vision.session(),
                                             (unsigned long long)vision.move(), vision.state());
              if (size > 0 && size < int(sizeof(reply)) && commandSocket.beginPacket(remote, port)) {
                commandSocket.write(reinterpret_cast<const uint8_t*>(reply), size);
                commandSocket.endPacket();
              }
            }
          } else {
            const auto action = input.receive(packet, peer, pwm);
            if (action != InputAction::Ignore) {
              if (action == InputAction::Stop) vision.setSession(input.autonomous ? input.padSequence() : 0);
              applyMotorAction(action, pwm);
              autonomous.store(input.autonomous);
              lastCommandTime = now;
              stoppedForTimeout = false;
            }
          }
        }
      } else {
        // Discard oversized datagrams instead of parsing a truncated command.
        while (commandSocket.available()) {
          commandSocket.read();
        }
      }
    }

    // Poll frequently enough for responsive control and a prompt link timeout.
    vTaskDelay(pdMS_TO_TICKS(5));
  }
}

bool startPS4Controller() {
  // The camera startup brings up the ESP32 access point before this listener.
  if (WiFi.getMode() == WIFI_OFF) {
    Serial.println("Wi-Fi must be started before motor command receiver");
    return false;
  }

  if (commandSocket.begin(MOTOR_COMMAND_PORT) != 1) {
    Serial.println("Motor command UDP listener startup failed");
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
    Serial.println("Motor command receiver task creation failed");
    return false;
  }

  Serial.printf("Motor command listener ready on UDP port %u\n",
                 MOTOR_COMMAND_PORT);
  return true;
}
