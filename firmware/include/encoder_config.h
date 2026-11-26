#pragma once

#include <cstdint>

namespace encoderConfig {
constexpr int leftSdaPin = 1;
constexpr int leftSclPin = 14;
constexpr int rightSdaPin = 38;
constexpr int rightSclPin = 39;
constexpr int leftDirection = 1;
constexpr int rightDirection = 1;

constexpr uint32_t leftCountsPerMetre = 0;
constexpr uint32_t rightCountsPerMetre = 0;
constexpr uint16_t trackWidthMm = 0;
constexpr bool printCounts = true;

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
