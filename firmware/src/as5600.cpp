#include <Arduino.h>
#include <Wire.h>
#include "as5600.h"
#include "encoder_config.h"

static portMUX_TYPE sampleLock = portMUX_INITIALIZER_UNLOCKED;
static EncoderSample latest{};
static constexpr uint8_t as5600Address = 0x36;
static constexpr uint8_t statusRegister = 0x0B;

// The AS5600 does not stretch SCL. This second bus is dedicated to its reads.
static void rightSda(bool high) { digitalWrite(encoderConfig::rightSdaPin, high); }
static void rightScl(bool high) { digitalWrite(encoderConfig::rightSclPin, high); }
static bool readRightSda() { return digitalRead(encoderConfig::rightSdaPin); }

static void rightStart() {
  rightSda(true);
  delayMicroseconds(5);
  rightScl(true);
  delayMicroseconds(5);
  rightSda(false);
  delayMicroseconds(5);
  rightScl(false);
}

static void rightStop() {
  rightSda(false);
  delayMicroseconds(5);
  rightScl(true);
  delayMicroseconds(5);
  rightSda(true);
}

static bool rightClockBit(bool released) {
  rightSda(released);
  delayMicroseconds(5);
  rightScl(true);
  delayMicroseconds(5);
  const bool bit = readRightSda();
  rightScl(false);
  return bit;
}

static bool rightWrite(uint8_t byte) {
  for (uint8_t bit = 0x80; bit; bit >>= 1) rightClockBit(byte & bit);
  return !rightClockBit(true);
}

static bool readRightRegisters(uint8_t (&data)[3]) {
  if (!readRightSda() || !digitalRead(encoderConfig::rightSclPin)) return false;
  rightStart();
  bool ok = rightWrite(as5600Address << 1) && rightWrite(statusRegister);
  if (ok) {
    rightStart();
    ok = rightWrite((as5600Address << 1) | 1);
  }
  if (ok) {
    for (unsigned i = 0; i < 3; ++i) {
      uint8_t byte = 0;
      for (int bit = 0; bit < 8; ++bit) byte = (byte << 1) | rightClockBit(true);
      data[i] = byte;
      rightClockBit(i == 2);  // ACK all but the last byte.
    }
  }
  rightStop();
  return ok;
}

static bool readLeftRegisters(uint8_t (&data)[3]) {
  Wire.beginTransmission(as5600Address);
  Wire.write(statusRegister);
  if (Wire.endTransmission(false) != 0 || Wire.requestFrom(as5600Address, size_t(3), true) != 3) {
    return false;
  }
  for (auto& byte : data) byte = Wire.read();
  return true;
}

static bool readAngle(bool left, int& angle) {
  uint8_t data[3];
  if (!(left ? readLeftRegisters(data) : readRightRegisters(data))) return false;
  if ((data[0] & 0x38) != 0x20) return false;  // Require a detected magnet in range.
  angle = ((data[1] & 0x0F) << 8) | data[2];
  return true;
}

static void sampleEncoders(void*) {
  TickType_t wake = xTaskGetTickCount();
  EncoderSample sample{};
  int previousLeft = 0;
  int previousRight = 0;
  uint32_t lastPrint = millis();
  while (true) {
    int left = 0;
    int right = 0;
    const bool read = readAngle(true, left) && readAngle(false, right);
    const uint32_t now = millis();
    if (read) {
      if (sample.valid && now - sample.sampledAt < 40) {
        const int leftDelta = as5600Delta(left, previousLeft);
        const int rightDelta = as5600Delta(right, previousRight);
        if (abs(leftDelta) != 2048 && abs(rightDelta) != 2048) {
          sample.left += encoderConfig::leftDirection * leftDelta;
          sample.right += encoderConfig::rightDirection * rightDelta;
        } else {
          ++sample.continuity;
        }
      } else if (sample.valid) {
        ++sample.continuity;
      }
      previousLeft = left;
      previousRight = right;
    } else if (sample.valid) {
      ++sample.continuity;
    }
    sample.valid = read;
    sample.sampledAt = now;

    portENTER_CRITICAL(&sampleLock);
    latest = sample;
    portEXIT_CRITICAL(&sampleLock);

    if (encoderConfig::printCounts && now - lastPrint >= 1000) {
      if (sample.valid) Serial.printf("AS5600 left=%lld right=%lld\n", sample.left, sample.right);
      else Serial.println("AS5600 unavailable. Check I2C and both magnets.");
      lastPrint = now;
    }
    vTaskDelayUntil(&wake, pdMS_TO_TICKS(10));
  }
}

EncoderSample readWheelEncoders() {
  portENTER_CRITICAL(&sampleLock);
  const auto sample = latest;
  portEXIT_CRITICAL(&sampleLock);
  return sample;
}

void startWheelEncoders() {
  if (!encoderConfig::enabled) return;
  // The camera driver uses I2C1 in this build. Wire uses I2C0.
  if (!Wire.begin(encoderConfig::leftSdaPin, encoderConfig::leftSclPin, 100000)) {
    Serial.println("AS5600 I2C setup failed");
    return;
  }
  Wire.setTimeOut(4);
  rightSda(true);
  rightScl(true);
  pinMode(encoderConfig::rightSdaPin, OUTPUT_OPEN_DRAIN);
  pinMode(encoderConfig::rightSclPin, OUTPUT_OPEN_DRAIN);
  if (xTaskCreatePinnedToCore(sampleEncoders, "Encoders", 3072, nullptr, 3, nullptr, 1) != pdPASS) {
    Serial.println("AS5600 sample task startup failed");
    return;
  }
  Serial.println(encoderConfig::calibrated ? "AS5600 sampling enabled" : "AS5600 sampling for calibration");
}
