#pragma once

#include <cstdint>

struct EncoderSample {
  int64_t left;
  int64_t right;
  uint32_t sampledAt;
  uint32_t continuity;
  bool valid;
};

enum class EncoderState : uint8_t {
  Waiting,
  BusSetupFailed,
  BusHeldLow,
  NoAck,
  ReadError,
  MagnetMissing,
  MagnetWeak,
  MagnetStrong,
  Ready
};

struct EncoderDiagnostics {
  EncoderSample sample;
  EncoderState left;
  EncoderState right;
};

constexpr int as5600CountsPerRevolution = 4096;

inline int as5600Delta(int next, int previous) {
  const int delta = next - previous;
  if (delta > as5600CountsPerRevolution / 2) return delta - as5600CountsPerRevolution;
  if (delta < -as5600CountsPerRevolution / 2) return delta + as5600CountsPerRevolution;
  return delta;
}

void startWheelEncoders();
EncoderSample readWheelEncoders();
EncoderDiagnostics readEncoderDiagnostics();
const char* encoderStateName(EncoderState state);
