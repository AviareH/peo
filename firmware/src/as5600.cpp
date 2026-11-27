#include <Arduino.h>
#include <Wire.h>
#include "as5600.h"
#include "encoder_config.h"

static portMUX_TYPE sampleLock = portMUX_INITIALIZER_UNLOCKED;
static EncoderDiagnostics latest{};
static bool leftBusReady = false;
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

static EncoderState readRightRegisters(uint8_t (&data)[3]) {
  if (!readRightSda() || !digitalRead(encoderConfig::rightSclPin)) return EncoderState::BusHeldLow;
  rightStart();
  if (!rightWrite(as5600Address << 1)) {
    rightStop();
    return EncoderState::NoAck;
  }
  bool ok = rightWrite(statusRegister);
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
  return ok ? EncoderState::Ready : EncoderState::ReadError;
}

static EncoderState readLeftRegisters(uint8_t (&data)[3]) {
  if (!leftBusReady) return EncoderState::BusSetupFailed;
  Wire.beginTransmission(as5600Address);
  Wire.write(statusRegister);
  // Complete the register write so Wire reports address NACKs separately.
  const uint8_t result = Wire.endTransmission();
  if (result == 2 || result == 3) return EncoderState::NoAck;
  if (result != 0 || Wire.requestFrom(as5600Address, sizeof(data), true) != sizeof(data)) {
    return EncoderState::ReadError;
  }
  for (auto& byte : data) byte = Wire.read();
  return EncoderState::Ready;
}

static EncoderState readAngle(bool left, int& angle) {
  uint8_t data[3];
  const auto state = left ? readLeftRegisters(data) : readRightRegisters(data);
  if (state != EncoderState::Ready) return state;
  if (data[0] & 0x10) return EncoderState::MagnetWeak;
  if (data[0] & 0x08) return EncoderState::MagnetStrong;
  if (!(data[0] & 0x20)) return EncoderState::MagnetMissing;
  angle = ((data[1] & 0x0F) << 8) | data[2];
  return EncoderState::Ready;
}

struct WheelHistory {
  EncoderState state = EncoderState::Waiting;
  int angle = 0;
  int64_t count = 0;

  bool update(EncoderState nextState, int nextAngle, int direction, bool timely) {
    bool lostContinuity = false;
    if (state == EncoderState::Ready) {
      if (nextState == EncoderState::Ready && timely) {
        const int delta = as5600Delta(nextAngle, angle);
        lostContinuity = abs(delta) == 2048;
        if (!lostContinuity) count += direction * delta;
      } else {
        lostContinuity = true;
      }
    }
    state = nextState;
    angle = nextAngle;
    return lostContinuity;
  }
};

static void sampleEncoders(void*) {
  TickType_t wake = xTaskGetTickCount();
  EncoderSample sample{};
  WheelHistory left, right;
  uint32_t lastPrint = millis();
  while (true) {
    int leftAngle = 0;
    int rightAngle = 0;
    const auto leftState = readAngle(true, leftAngle);
    const auto rightState = readAngle(false, rightAngle);
    const uint32_t now = millis();
    const bool timely = now - sample.sampledAt < 40;
    const bool leftGap = left.update(leftState, leftAngle, encoderConfig::leftDirection, timely);
    const bool rightGap = right.update(rightState, rightAngle, encoderConfig::rightDirection, timely);
    if (leftGap || rightGap) ++sample.continuity;
    sample.left = left.count;
    sample.right = right.count;
    sample.valid = leftState == EncoderState::Ready && rightState == EncoderState::Ready;
    sample.sampledAt = now;

    portENTER_CRITICAL(&sampleLock);
    latest = {sample, leftState, rightState};
    portEXIT_CRITICAL(&sampleLock);

    if (encoderConfig::printCounts && now - lastPrint >= 1000) {
      Serial.printf("AS5600 left=%lld [%s] right=%lld [%s]\n",
                    sample.left, encoderStateName(leftState), sample.right, encoderStateName(rightState));
      lastPrint = now;
    }
    vTaskDelayUntil(&wake, pdMS_TO_TICKS(10));
  }
}

EncoderDiagnostics readEncoderDiagnostics() {
  portENTER_CRITICAL(&sampleLock);
  const auto sample = latest;
  portEXIT_CRITICAL(&sampleLock);
  return sample;
}

EncoderSample readWheelEncoders() {
  return readEncoderDiagnostics().sample;
}

const char* encoderStateName(EncoderState state) {
  switch (state) {
    case EncoderState::Waiting: return "waiting";
    case EncoderState::BusSetupFailed: return "bus_setup_failed";
    case EncoderState::BusHeldLow: return "bus_held_low";
    case EncoderState::NoAck: return "no_ack";
    case EncoderState::ReadError: return "read_error";
    case EncoderState::MagnetMissing: return "magnet_missing";
    case EncoderState::MagnetWeak: return "magnet_weak";
    case EncoderState::MagnetStrong: return "magnet_strong";
    case EncoderState::Ready: return "ok";
  }
  return "unknown";
}

void startWheelEncoders() {
  if (!encoderConfig::enabled) return;
  // The camera driver uses I2C1 in this build. Wire uses I2C0.
  leftBusReady = Wire.begin(encoderConfig::leftSdaPin, encoderConfig::leftSclPin, 100000);
  if (leftBusReady) Wire.setTimeOut(4);
  pinMode(encoderConfig::rightSdaPin, OUTPUT_OPEN_DRAIN);
  pinMode(encoderConfig::rightSclPin, OUTPUT_OPEN_DRAIN);
  rightSda(true);
  rightScl(true);
  if (xTaskCreatePinnedToCore(sampleEncoders, "Encoders", 3072, nullptr, 3, nullptr, 1) != pdPASS) {
    Serial.println("AS5600 sample task startup failed");
    return;
  }
  Serial.println(encoderConfig::calibrated ? "AS5600 sampling enabled" : "AS5600 sampling for calibration");
}
