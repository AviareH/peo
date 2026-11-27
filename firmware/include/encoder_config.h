#pragma once

#include "as5600.h"

namespace encoderConfig {
constexpr int leftSdaPin = 1;
constexpr int leftSclPin = 14;
constexpr int rightSdaPin = 38;
constexpr int rightSclPin = 39;
constexpr int leftDirection = 1;
constexpr int rightDirection = -1;

constexpr double wheelDiameterMm = 82.0;
constexpr uint16_t trackWidthMm = 210;
constexpr bool printCounts = true;

static_assert(wheelDiameterMm > 0, "Wheel diameter must be positive");
constexpr double wheelCircumferenceMm = 3.141592653589793 * wheelDiameterMm;
constexpr uint32_t countsPerMetre =
    static_cast<uint32_t>(1000.0 * as5600CountsPerRevolution / wheelCircumferenceMm + 0.5);
constexpr uint32_t leftCountsPerMetre = countsPerMetre;
constexpr uint32_t rightCountsPerMetre = countsPerMetre;

constexpr bool enabled = leftSdaPin >= 0 && leftSclPin >= 0 && rightSdaPin >= 0 && rightSclPin >= 0;
static_assert(enabled || (leftSdaPin == -1 && leftSclPin == -1 && rightSdaPin == -1 && rightSclPin == -1),
              "Configure both I2C pin pairs or leave all four pins at -1");
static_assert(!enabled || (leftSdaPin != leftSclPin && rightSdaPin != rightSclPin &&
                          leftSdaPin != rightSdaPin && leftSdaPin != rightSclPin &&
                          leftSclPin != rightSdaPin && leftSclPin != rightSclPin),
              "Each I2C bus needs its own pins");
static_assert((leftDirection == 1 || leftDirection == -1) &&
              (rightDirection == 1 || rightDirection == -1), "Encoder direction must be +1 or -1");

constexpr bool calibrated = leftCountsPerMetre && rightCountsPerMetre && trackWidthMm;
}
