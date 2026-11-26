#pragma once

#include <cstdint>
#include <cstdio>
#include <cstring>
#include "motors.h"

enum class InputAction { Ignore, Heartbeat, Stop, Drive };

// Only the UDP task mutates this state. PWM and camera tasks consume its outputs.
class DriveInput {
 public:
  bool autonomous = false;

  InputAction receive(const char* text, uint64_t peer, MotorCommand& pwm) {
    unsigned long long seq;
    char mode, extra;
    int left, right;
    if (std::strncmp(text, "PAD,", 4) == 0) {
      if (text[4] < '0' || text[4] > '9' ||
          std::sscanf(text, "PAD,%16llu,%c,%4d,%4d%c", &seq, &mode, &left, &right, &extra) != 4 ||
          (mode != 'M' && mode != 'A') || left < -255 || left > 255 ||
          right < -255 || right > 255 || (mode == 'A' && (left || right))) {
        return InputAction::Ignore;
      }
      if ((padActive && peer != padPeer) || (peer == padPeer && seq <= lastSequence)) {
        return InputAction::Ignore;
      }
      if (!padActive && (mode != 'M' || left || right)) {
        return InputAction::Ignore;  // Start or rejoin with centred sticks in manual.
      }
      const bool switched = !padActive || autonomous != (mode == 'A');
      padPeer = peer;
      padActive = true;
      lastSequence = seq;
      autonomous = mode == 'A';
      if (switched) {
        needNeutral = left || right;
        return InputAction::Stop;
      }
      if (autonomous) return InputAction::Heartbeat;
    } else {
      if (padActive || std::sscanf(text, "%9d,%9d%c", &left, &right, &extra) != 2) {
        return InputAction::Ignore;
      }
    }
    if (needNeutral && (left || right)) return InputAction::Heartbeat;
    needNeutral = false;
    pwm = {clamp(left), clamp(right)};
    return InputAction::Drive;
  }

  void disconnect() {
    autonomous = false;
    padActive = false;
    needNeutral = true;
  }

  bool isPadHost(uint64_t peer) const {
    return padActive && (peer >> 16) == (padPeer >> 16);
  }

  uint64_t padSequence() const { return lastSequence; }

 private:
  static int clamp(int pwm) { return pwm < -255 ? -255 : (pwm > 255 ? 255 : pwm); }
  bool padActive = false;
  bool needNeutral = false;
  uint64_t padPeer = 0;
  uint64_t lastSequence = 0;
};
