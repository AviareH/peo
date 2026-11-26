#include <Arduino.h>
#include <esp_arduino_version.h>
#include "motors.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/queue.h"

// used to access the motor command queue
// nullptr until the queue is created and static keeps variable local to motors.cpp
static QueueHandle_t commandQueue = nullptr;

// Each motor side uses one DRV8833 channel; the paired motors run together.
const int MOTOR_A_IN1 = 21;
const int MOTOR_A_IN2 = 47;
const int MOTOR_B_IN1 = 41;
const int MOTOR_B_IN2 = 42;

const int MOTOR_A_IN1_CHANNEL = 2;
const int MOTOR_A_IN2_CHANNEL = 3;
const int MOTOR_B_IN1_CHANNEL = 4;
const int MOTOR_B_IN2_CHANNEL = 5;

const int MOTOR_PWM_FREQUENCY = 20000;
const int MOTOR_PWM_RESOLUTION = 8;
const int MOTOR_PWM_MAX = (1 << MOTOR_PWM_RESOLUTION) - 1;
const int MOTOR_PWM_RAMP_STEP = 3;
const TickType_t MOTOR_UPDATE_PERIOD = pdMS_TO_TICKS(10);

static bool configureMotorPwm(int pin, int channel) {
#if ESP_ARDUINO_VERSION_MAJOR >= 3
  // Arduino-ESP32 3.x attaches PWM by GPIO; older versions use channel IDs.
  return ledcAttachChannel(
      pin,
      MOTOR_PWM_FREQUENCY,
      MOTOR_PWM_RESOLUTION,
      channel
  );
#else
  if (ledcSetup(channel, MOTOR_PWM_FREQUENCY, MOTOR_PWM_RESOLUTION) == 0) {
    return false;
  }
  ledcAttachPin(pin, channel);
  return true;
#endif
}

static void writeMotorPwm(int pin, int channel, int duty) {
#if ESP_ARDUINO_VERSION_MAJOR >= 3
  ledcWrite(pin, duty);
#else
  (void)pin;
  ledcWrite(channel, duty);
#endif
}

static void setMotorSpeed(
    int speed,
    int in1Pin,
    int in1Channel,
    int in2Pin,
    int in2Channel
) {
  // Sign chooses direction; magnitude sets PWM duty on the matching input.
  speed = constrain(speed, -MOTOR_PWM_MAX, MOTOR_PWM_MAX);

  if (speed > 0) {
    writeMotorPwm(in1Pin, in1Channel, speed);
    writeMotorPwm(in2Pin, in2Channel, 0);
  } else if (speed < 0) {
    writeMotorPwm(in1Pin, in1Channel, 0);
    writeMotorPwm(in2Pin, in2Channel, -speed);
  } else {
    // Both bridge inputs low leaves this motor channel stopped/coasting.
    writeMotorPwm(in1Pin, in1Channel, 0);
    writeMotorPwm(in2Pin, in2Channel, 0);
  }
}

static int rampMotorSpeed(int currentSpeed, int targetSpeed) {
  // Limit every change toward the requested PWM, including partial-stick values.
  if (currentSpeed < targetSpeed) {
    return min(currentSpeed + MOTOR_PWM_RAMP_STEP, targetSpeed);
  }
  if (currentSpeed > targetSpeed) {
    return max(currentSpeed - MOTOR_PWM_RAMP_STEP, targetSpeed);
  }
  return currentSpeed;
}

bool initializeMotors() {
  pinMode(MOTOR_A_IN1, OUTPUT);
  pinMode(MOTOR_A_IN2, OUTPUT);
  pinMode(MOTOR_B_IN1, OUTPUT);
  pinMode(MOTOR_B_IN2, OUTPUT);

  // Hold all H-bridge inputs low while configuring PWM to prevent startup drive.
  digitalWrite(MOTOR_A_IN1, LOW);
  digitalWrite(MOTOR_A_IN2, LOW);
  digitalWrite(MOTOR_B_IN1, LOW);
  digitalWrite(MOTOR_B_IN2, LOW);

  if (!configureMotorPwm(MOTOR_A_IN1, MOTOR_A_IN1_CHANNEL) ||
      !configureMotorPwm(MOTOR_A_IN2, MOTOR_A_IN2_CHANNEL) ||
      !configureMotorPwm(MOTOR_B_IN1, MOTOR_B_IN1_CHANNEL) ||
      !configureMotorPwm(MOTOR_B_IN2, MOTOR_B_IN2_CHANNEL)) {
    Serial0.println("Motor PWM setup failed");
    return false;
  }

  setMotorSpeed(
      0,
      MOTOR_A_IN1, MOTOR_A_IN1_CHANNEL,
      MOTOR_A_IN2, MOTOR_A_IN2_CHANNEL
  );
  setMotorSpeed(
      0,
      MOTOR_B_IN1, MOTOR_B_IN1_CHANNEL,
      MOTOR_B_IN2, MOTOR_B_IN2_CHANNEL
  );
  return true;
}

void sendMotorCommand(MotorCommand command) {
  if (commandQueue != nullptr) {
    // Keep only the newest joystick request if the motor task has not read yet.
    xQueueOverwrite(commandQueue, &command);
  }
}


void updateMotors(void* parameters) {
  (void)parameters;
  TickType_t lastWakeTime = xTaskGetTickCount();
  MotorCommand target = {};
  int leftSpeed = 0;
  int rightSpeed = 0;

  while (true) {
    // Keep the last target between packets; each side ramps independently.
    if (xQueueReceive(commandQueue, &target, 0) == pdTRUE) {
      target.leftSpeed = constrain(
          target.leftSpeed,
          -MOTOR_PWM_MAX,
          MOTOR_PWM_MAX
      );
      target.rightSpeed = constrain(
          target.rightSpeed,
          -MOTOR_PWM_MAX,
          MOTOR_PWM_MAX
      );
    }

    leftSpeed = rampMotorSpeed(leftSpeed, target.leftSpeed);
    rightSpeed = rampMotorSpeed(rightSpeed, target.rightSpeed);

    setMotorSpeed(
        leftSpeed,
        MOTOR_A_IN1, MOTOR_A_IN1_CHANNEL,
        MOTOR_A_IN2, MOTOR_A_IN2_CHANNEL
    );
    setMotorSpeed(
        rightSpeed,
        MOTOR_B_IN1, MOTOR_B_IN1_CHANNEL,
        MOTOR_B_IN2, MOTOR_B_IN2_CHANNEL
    );

    vTaskDelayUntil(&lastWakeTime, MOTOR_UPDATE_PERIOD);
  }
}


bool startMotorControl() {
  if (!initializeMotors()) {
    return false;
  }

  // A one-item overwrite queue ensures control always uses the newest input.
  commandQueue = xQueueCreate(1, sizeof(MotorCommand));

  if (commandQueue == nullptr) {
    return false;
  }

  // Run motor updates on core 1, away from the camera server on core 0.
  BaseType_t result = xTaskCreatePinnedToCore(
      updateMotors,
      "MotorControl",
      4096,
      nullptr,
      6,
      nullptr,
      1
  );

  if (result != pdPASS) {
    vQueueDelete(commandQueue);
    commandQueue = nullptr;
    return false;
  }

  return true;
}