#pragma once

// Initialize the camera, start its HTTP endpoint, and expose it over Wi-Fi AP.
bool initializeCamera();
bool startStreamingServer();
bool startCameraStreaming();