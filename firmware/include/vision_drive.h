#pragma once

#include "as5600.h"
#include "drive_input.h"

struct VisionRequest {
  unsigned long long sequence, session, move;
  int turn, drive;
  bool encoder;

  VisionRequest(unsigned long long sequence = 0, unsigned long long session = 0,
                unsigned long long move = 0, int turn = 0, int drive = 0, bool encoder = false)
      : sequence(sequence), session(session), move(move), turn(turn), drive(drive), encoder(encoder) {}

  static bool parse(const char* text, VisionRequest& request) {
    char extra;
    constexpr uint64_t maxSequence = 9999999999999999ULL;
    request.encoder = std::strncmp(text, "ENC,", 4) == 0;
    if (!request.encoder && std::strncmp(text, "CV,", 3) != 0) return false;
    const int fields = std::sscanf(text + (request.encoder ? 4 : 3), "%16llu,%16llu,%16llu,%5d,%5d%c",
                                  &request.sequence, &request.session, &request.move,
                                  &request.turn, &request.drive, &extra);
    const int maxTurn = request.encoder ? 700 : 1500;
    const int maxDrive = request.encoder ? 300 : 1500;
    return fields == 5 && request.sequence > 0 && request.sequence <= maxSequence &&
           request.session <= maxSequence && request.move <= request.sequence &&
           request.turn >= -maxTurn && request.turn <= maxTurn &&
           request.drive >= 0 && request.drive <= maxDrive;
  }
};

struct WheelCalibration {
  uint32_t leftCountsPerMetre;
  uint32_t rightCountsPerMetre;
  uint16_t trackWidthMm;
};

// Used only by the UDP receiver task. The motor task still owns all PWM writes.
class VisionDrive {
 public:
  explicit VisionDrive(WheelCalibration calibration = {}) : calibration(calibration) {}

  void setSession(uint64_t session) {
    *this = VisionDrive(calibration);
    activeSession = session;
  }

  uint64_t session() const { return activeSession; }
  uint64_t move() const { return current.move; }

  const char* state() const {
    if (!activeSession) return "MANUAL";
    switch (phase) {
      case Phase::Idle: return "IDLE";
      case Phase::Done: return "DONE";
      case Phase::Stopped: return "STOPPED";
      case Phase::Fault: return "FAULT";
      default: return "RUNNING";
    }
  }

  InputAction receive(const VisionRequest& request, uint32_t now, MotorCommand& pwm,
                      const EncoderSample& sample = {}) {
    if (!activeSession || request.session != activeSession || !request.move ||
        request.sequence <= lastSequence || request.move < current.move) {
      return InputAction::Ignore;
    }
    lastSequence = request.sequence;
    if (phase == Phase::Fault) return InputAction::Ignore;
    if (request.move == current.move) {
      if (running() && request.turn == current.turn && request.drive == current.drive &&
          request.encoder == current.encoder) {
        lastVision = now;
      }
      return InputAction::Ignore;
    }
    if (request.turn == 0 && request.drive == 0) {
      current = request;
      phase = Phase::Stopped;
      return InputAction::Stop;
    }
    if (running()) return InputAction::Ignore;

    current = request;
    if (current.encoder) {
      if (!calibration.leftCountsPerMetre || !calibration.rightCountsPerMetre ||
          !calibration.trackWidthMm || !fresh(sample, now)) return fault();
      continuity = sample.continuity;
    }
    lastVision = now;
    startPhase(current.turn ? Phase::Turning : Phase::Driving, now, sample, pwm);
    return InputAction::Drive;
  }

  InputAction update(uint32_t now, MotorCommand& pwm, const EncoderSample& sample = {}) {
    if (!running()) return InputAction::Ignore;
    if (now - lastVision >= visionTimeoutMs) {
      phase = Phase::Stopped;
      return InputAction::Stop;
    }
    const uint32_t elapsed = now - phaseStarted;
    if (current.encoder && (!fresh(sample, now) || sample.continuity != continuity)) return fault();
    if (phase == Phase::BetweenPhases) {
      if (elapsed < turnPauseMs) return InputAction::Ignore;
      startPhase(Phase::Driving, now, sample, pwm);
      return InputAction::Drive;
    }

    const bool turning = phase == Phase::Turning;
    bool completed;
    if (current.encoder) {
      const bool leftWasDone = left.done;
      const bool rightWasDone = right.done;
      if (elapsed >= 3000 || !left.update(sample.left, now) || !right.update(sample.right, now)) {
        return fault();
      }
      completed = left.done && right.done;
      if (!completed && (left.done != leftWasDone || right.done != rightWasDone)) {
        pwm = encoderPwm(turning ? turnPwm : drivePwm);
        return InputAction::Drive;
      }
    } else {
      completed = elapsed >= uint32_t(turning ? turnMagnitude() : current.drive);
    }
    if (!completed) return InputAction::Ignore;
    if (turning && current.drive) {
      phase = Phase::BetweenPhases;
      phaseStarted = now;
    } else {
      phase = Phase::Done;
    }
    return InputAction::Stop;
  }

 private:
  enum class Phase { Idle, Turning, BetweenPhases, Driving, Done, Stopped, Fault };
  static constexpr uint32_t visionTimeoutMs = 250;
  static constexpr uint32_t turnPauseMs = 60;
  static constexpr int turnPwm = 80;
  static constexpr int drivePwm = 80;

  struct WheelProgress {
    int64_t origin = 0, target = 0, best = 0;
    uint32_t lastAdvance = 0;
    int direction = 1;
    bool done = false;

    void start(int64_t position, int64_t counts, int sign, uint32_t now) {
      origin = position;
      target = counts;
      best = 0;
      lastAdvance = now;
      direction = sign;
      done = counts == 0;
    }

    bool update(int64_t position, uint32_t now) {
      if (done) return true;
      const int64_t travelled = (position - origin) * direction;
      if (travelled >= target) {
        done = true;
      } else if (travelled >= best + 4) {
        best = travelled;
        lastAdvance = now;
      }
      return done || (travelled >= -32 && now - lastAdvance < 600);
    }
  };

  static bool fresh(const EncoderSample& sample, uint32_t now) {
    return sample.valid && now - sample.sampledAt < 40;
  }

  InputAction fault() {
    phase = Phase::Fault;
    return InputAction::Stop;
  }

  int turnMagnitude() const { return current.turn < 0 ? -current.turn : current.turn; }

  MotorCommand encoderPwm(int pwm) const {
    return {left.done ? 0 : left.direction * pwm, right.done ? 0 : right.direction * pwm, true};
  }

  void startPhase(Phase next, uint32_t now, const EncoderSample& sample, MotorCommand& pwm) {
    phase = next;
    phaseStarted = now;
    const bool turning = phase == Phase::Turning;
    const int rightSign = !turning || current.turn > 0 ? 1 : -1;
    const int leftSign = turning ? -rightSign : 1;
    const int speed = turning ? turnPwm : drivePwm;
    if (current.encoder) {
      // A point turn moves each side along an arc of radius half the track width.
      const int64_t distance = turning ? int64_t(turnMagnitude()) * calibration.trackWidthMm : current.drive;
      const int64_t divisor = turning ? 2000000 : 1000;
      left.start(sample.left, (distance * calibration.leftCountsPerMetre + divisor / 2) / divisor,
                 leftSign, now);
      right.start(sample.right, (distance * calibration.rightCountsPerMetre + divisor / 2) / divisor,
                  rightSign, now);
      pwm = encoderPwm(speed);
    } else {
      pwm = {leftSign * speed, rightSign * speed};
    }
  }

  bool running() const {
    return phase == Phase::Turning || phase == Phase::BetweenPhases || phase == Phase::Driving;
  }

  WheelCalibration calibration;
  WheelProgress left, right;
  uint32_t continuity = 0;
  uint64_t activeSession = 0;
  uint64_t lastSequence = 0;
  VisionRequest current{};
  Phase phase = Phase::Idle;
  uint32_t lastVision = 0;
  uint32_t phaseStarted = 0;
};
