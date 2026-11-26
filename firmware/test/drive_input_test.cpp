#include "../include/drive_input.h"
#include "../include/vision_drive.h"
#include <cassert>

static void testVisionDrive() {
  using Action = InputAction;
  VisionDrive vision;
  MotorCommand pwm{};
  VisionRequest request{};
  assert(VisionRequest::parse("CV,1,0,0,0,0", request));
  assert(vision.receive(request, 0, pwm) == Action::Ignore);
  assert(!VisionRequest::parse("CV,2,100,2,1501,250", request));
  assert(!VisionRequest::parse("CV,2,100,2,0,-1", request));
  assert(!VisionRequest::parse("CV,2,100,2,0,100x", request));
  assert(VisionRequest::parse("CV,2,100,2,200,300", request));
  assert(vision.receive(request, 0, pwm) == Action::Ignore);
  vision.setSession(101);
  assert(vision.receive(request, 1, pwm) == Action::Ignore);
  vision.setSession(100);
  assert(vision.receive(request, 10, pwm) == Action::Drive);
  assert(pwm.leftSpeed == -80 && pwm.rightSpeed == 80);
  assert(vision.receive({3, 100, 3, -200, 300}, 20, pwm) == Action::Ignore);
  assert(vision.move() == 2);
  assert(vision.update(209, pwm) == Action::Ignore);
  request.sequence = 4;
  assert(vision.receive(request, 209, pwm) == Action::Ignore);
  assert(vision.update(210, pwm) == Action::Stop);
  assert(vision.update(269, pwm) == Action::Ignore);
  assert(vision.update(270, pwm) == Action::Drive);
  assert(pwm.leftSpeed == 80 && pwm.rightSpeed == 80);
  request.sequence = 5;
  assert(vision.receive(request, 450, pwm) == Action::Ignore);
  assert(vision.update(570, pwm) == Action::Stop);
  assert(std::strcmp(vision.state(), "DONE") == 0);
  request.sequence = 6;
  assert(vision.receive(request, 571, pwm) == Action::Ignore);
  assert(vision.receive({7, 100, 7, -300, 100}, 580, pwm) == Action::Drive);
  assert(pwm.leftSpeed == 80 && pwm.rightSpeed == -80);
  assert(vision.receive({7, 100, 7, -300, 100}, 800, pwm) == Action::Ignore);
  assert(vision.update(830, pwm) == Action::Stop);  // A replay cannot renew the lease.
  assert(std::strcmp(vision.state(), "STOPPED") == 0);
  assert(vision.receive({8, 100, 7, -300, 100}, 831, pwm) == Action::Ignore);
  assert(vision.receive({9, 100, 9, 0, 100}, 840, pwm) == Action::Drive);
  assert(vision.receive({10, 100, 10, 0, 0}, 850, pwm) == Action::Stop);
  assert(vision.receive({11, 100, 9, 0, 100}, 851, pwm) == Action::Ignore);
  vision.setSession(0);
  assert(vision.update(1000, pwm) == Action::Ignore);
  assert(std::strcmp(vision.state(), "MANUAL") == 0);
}

static void testEncoderDrive() {
  using Action = InputAction;
  assert(as5600Delta(10, 4090) == 16);
  assert(as5600Delta(4090, 10) == -16);
  assert(as5600Delta(100, 105) == -5);
  VisionDrive vision({10000, 20000, 200});
  MotorCommand pwm{};
  VisionRequest move;
  assert(!VisionRequest::parse("ENC,1,100,1,701,100", move));
  assert(VisionRequest::parse("ENC,1,100,1,200,100", move));
  vision.setSession(100);
  assert(vision.receive(move, 10, pwm, {0, 0, 10, 0, true}) == Action::Drive);
  assert(pwm.leftSpeed == -80 && pwm.rightSpeed == 80 && pwm.cutZero);
  assert(vision.update(20, pwm, {-200, 300, 20, 0, true}) == Action::Drive);
  assert(pwm.leftSpeed == 0 && pwm.rightSpeed == 80 && pwm.cutZero);
  assert(vision.update(30, pwm, {-200, 400, 30, 0, true}) == Action::Stop);
  assert(vision.update(90, pwm, {-210, 410, 90, 0, true}) == Action::Drive);
  assert(pwm.leftSpeed == 80 && pwm.rightSpeed == 80);
  assert(vision.update(100, pwm, {790, 2410, 100, 0, true}) == Action::Stop);
  assert(std::strcmp(vision.state(), "DONE") == 0);
  assert(vision.receive({2, 100, 2, 0, 100, true}, 110, pwm, {790, 2410, 110, 0, true}) == Action::Drive);
  assert(vision.update(120, pwm, {790, 2410, 120, 1, true}) == Action::Stop);
  assert(std::strcmp(vision.state(), "FAULT") == 0);
  assert(vision.receive({3, 100, 3, 0, 100, true}, 130, pwm, {0, 0, 130, 1, true}) == Action::Ignore);
  vision.setSession(101);
  assert(vision.receive({4, 101, 4, 0, 100, true}, 140, pwm) == Action::Stop);
  assert(std::strcmp(vision.state(), "FAULT") == 0);

  vision.setSession(102);
  move = {5, 102, 5, 0, 100, true};
  assert(vision.receive(move, 150, pwm, {0, 0, 150, 0, true}) == Action::Drive);
  for (uint32_t now = 250; now < 750; now += 100) {
    assert(vision.receive({++move.sequence, 102, 5, 0, 100, true}, now, pwm) == Action::Ignore);
    assert(vision.update(now, pwm, {0, 0, now, 0, true}) == Action::Ignore);
  }
  assert(vision.update(750, pwm, {0, 0, 750, 0, true}) == Action::Stop);
  assert(std::strcmp(vision.state(), "FAULT") == 0);
}

static void testTurnOnly() {
  using Action = InputAction;
  VisionDrive vision({10000, 20000, 200});
  MotorCommand pwm{};
  VisionRequest turn;
  assert(VisionRequest::parse("ENC,1,100,1,200,0", turn));
  vision.setSession(100);
  assert(vision.receive(turn, 10, pwm, {0, 0, 10, 0, true}) == Action::Drive);
  assert(pwm.leftSpeed == -80 && pwm.rightSpeed == 80);
  assert(vision.update(20, pwm, {-200, 400, 20, 0, true}) == Action::Stop);
  assert(std::strcmp(vision.state(), "DONE") == 0);
  assert(vision.update(100, pwm, {-200, 400, 100, 0, true}) == Action::Ignore);

  assert(VisionRequest::parse("CV,2,100,2,-60,0", turn));
  assert(vision.receive(turn, 110, pwm) == Action::Drive);
  assert(pwm.leftSpeed == 80 && pwm.rightSpeed == -80);
  assert(vision.update(170, pwm) == Action::Stop);
  assert(std::strcmp(vision.state(), "DONE") == 0);
  assert(vision.update(230, pwm) == Action::Ignore);
}

int main() {
  testVisionDrive();
  testEncoderDrive();
  testTurnOnly();
  DriveInput input;
  MotorCommand pwm{};
  using Action = InputAction;
  assert(!input.autonomous);
  assert(input.receive("300,-400", 1, pwm) == Action::Drive);
  assert(pwm.leftSpeed == 255 && pwm.rightSpeed == -255);
  assert(input.receive("PAD,1,M,120,120", 2, pwm) == Action::Ignore);
  assert(input.receive("PAD,2,M,0,0", 2, pwm) == Action::Stop);
  assert(input.receive("PAD,3,M,-30,80", 2, pwm) == Action::Drive);
  assert(pwm.leftSpeed == -30 && pwm.rightSpeed == 80);
  assert(input.receive("PAD,4,A,0,0", 2, pwm) == Action::Stop);
  assert(input.autonomous);
  assert(input.receive("PAD,5,A,0,0", 2, pwm) == Action::Heartbeat);
  assert(input.receive("PAD,5,M,30,30", 2, pwm) == Action::Ignore);  // duplicate
  assert(input.receive("PAD,3,M,30,30", 2, pwm) == Action::Ignore);  // reordered
  assert(input.receive("PAD,6,M,0,0", 1, pwm) == Action::Ignore);    // other sender
  assert(input.receive("40,40", 2, pwm) == Action::Ignore);
  assert(input.receive("PAD,6,A,30,30", 2, pwm) == Action::Ignore);
  assert(input.receive("PAD,6,M,0,0more", 2, pwm) == Action::Ignore);
  assert(input.autonomous);
  assert(input.receive("PAD,7,M,80,80", 2, pwm) == Action::Stop);
  assert(!input.autonomous);
  assert(input.receive("PAD,8,M,80,80", 2, pwm) == Action::Heartbeat);
  assert(input.receive("PAD,9,M,0,0", 2, pwm) == Action::Drive);
  assert(input.receive("PAD,10,M,80,80", 2, pwm) == Action::Drive);
  assert(input.receive("PAD,11,A,0,0", 2, pwm) == Action::Stop);
  input.disconnect();
  assert(!input.autonomous);
  assert(input.receive("PAD,12,A,0,0", 2, pwm) == Action::Ignore);
  assert(input.receive("PAD,9,M,0,0", 2, pwm) == Action::Ignore);
  assert(input.receive("PAD,13,M,80,80", 2, pwm) == Action::Ignore);
  assert(input.receive("PAD,14,M,0,0", 2, pwm) == Action::Stop);

  DriveInput peers;
  assert(peers.receive("PAD,1,M,0,0", 0x00010001, pwm) == Action::Stop);
  assert(peers.isPadHost(0x00010002));
  assert(!peers.isPadHost(0x00020001));
  peers.disconnect();
  assert(!peers.isPadHost(0x00010002));
}
