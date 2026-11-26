#pragma once

#include <cstdint>

struct EncoderSample {
  int64_t left;
  int64_t right;
  uint32_t sampledAt;
  uint32_t continuity;
  bool valid;
};

inline int as5600Delta(int next, int previous) {
  const int delta = next - previous;
  if (delta > 2048) return delta - 4096;
  if (delta < -2048) return delta + 4096;
  return delta;
}

void startWheelEncoders();
EncoderSample readWheelEncoders();
