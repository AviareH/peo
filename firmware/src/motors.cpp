#include <Arduino.h>
#include "motors.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/queue.h"

// used to access the motor command queue
// nullptr until the queue is created and static keeps variable local to motors.cpp
static QueueHandle_t commandQueue = nullptr;

// pins for DRV8833
const int MOTOR_A_IN1 = 21;
const int MOTOR_A_IN2 = 47;
const int MOTOR_B_IN1 = 41;
const int MOTOR_B_IN2 = 42;

// start motors coasting
void initializeMotors() {
  digitalWrite(MOTOR_A_IN1, LOW);
  digitalWrite(MOTOR_A_IN2, LOW);
  digitalWrite(MOTOR_B_IN1, LOW);
  digitalWrite(MOTOR_B_IN2, LOW);

  pinMode(MOTOR_A_IN1, OUTPUT);
  pinMode(MOTOR_A_IN2, OUTPUT);
  pinMode(MOTOR_B_IN1, OUTPUT);
  pinMode(MOTOR_B_IN2, OUTPUT);
}

//
void sendMotorCommand(MotorCommand command) {
  if (commandQueue != nullptr) {
    xQueueOverwrite(commandQueue, &command);
  }
}


// this is scheduled to run every 10 ms and has higher priority than camera stuff
void updateMotors(void* parameters) {
    TickType_t lastWakeTime = xTaskGetTickCount();
    MotorCommand command = {};

    // check for latest command every 10ms, xQueueReceive copies it to command and 
    // removes it from the queue
    // 0 means return immediately if the queue is empty
    // pdTRUE means a command was received, so update motors
    // vTaskDelayUntil waits until the next scheduled 10 ms update
    while (true) {
        if (xQueueReceive(commandQueue, &command, 0) == pdTRUE) {
            // current test for recieving a command from main
            Serial0.printf("Left: %d, Right: %d\n", command.leftSpeed, command.rightSpeed);

            // Motor control function calls/code would go here and run every 10ms
    }

    vTaskDelayUntil(&lastWakeTime, pdMS_TO_TICKS(10));
  }
}


bool startMotorControl() {
  initializeMotors();

  // create a queue to store motor command from future CV algo
  commandQueue = xQueueCreate(1, sizeof(MotorCommand));

  if (commandQueue == nullptr) {
    return false;
  }

  // schedule motor update task to run uninterupted, higher priority than cv stuff
  // uses core 1, cam stream uses core 0
  BaseType_t result = xTaskCreatePinnedToCore(
      updateMotors,
      "MotorControl",
      4096,
      nullptr,
      6,
      nullptr,
      1
  );

  // If task creation failed, free the queue and clear its handle
  // then return false to report failure. or return true to report successful startup
  if (result != pdPASS) {
    vQueueDelete(commandQueue);
    commandQueue = nullptr;
    return false;
  }

  return true;
}