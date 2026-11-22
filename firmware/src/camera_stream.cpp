#include <Arduino.h>
#include <WiFi.h>
#include "esp_camera.h"
#include "esp_http_server.h"


const char* wifiName = "PEO-Camera";
const char* wifiPassword = "peocamera123";

bool initializeCamera() {
  camera_config_t config = {};

  // Camera connections for the ESP32
  config.pin_pwdn = -1;
  config.pin_reset = -1;
  config.pin_xclk = 15;
  config.pin_sccb_sda = 4;
  config.pin_sccb_scl = 5;

  config.pin_d0 = 11;
  config.pin_d1 = 9;
  config.pin_d2 = 8;
  config.pin_d3 = 10;
  config.pin_d4 = 12;
  config.pin_d5 = 18;
  config.pin_d6 = 17;
  config.pin_d7 = 16;

  config.pin_vsync = 6;
  config.pin_href = 7;
  config.pin_pclk = 13;

  // Generate the camera clock
  config.ledc_channel = LEDC_CHANNEL_0;
  config.ledc_timer = LEDC_TIMER_0;
  config.xclk_freq_hz = 20000000;

  // 640 x 480 with JPEG compression
  config.pixel_format = PIXFORMAT_JPEG;
  config.frame_size = FRAMESIZE_VGA;
  config.jpeg_quality = 12;

  // Keep two frame buffers in PSRAM and prefer recent frames
  config.fb_location = CAMERA_FB_IN_PSRAM;
  config.fb_count = 2;
  config.grab_mode = CAMERA_GRAB_LATEST;

  esp_err_t result = esp_camera_init(&config);

  if (result != ESP_OK) {
    Serial0.printf("Camera initialization failed: 0x%x\n", result);
    return false;
  }

  Serial0.println("Camera initialized");
  Serial0.printf("Password: %s\n", wifiPassword);
  return true;
}

// Runs when a client opens /stream
esp_err_t streamHandler(httpd_req_t* request) {
  esp_err_t result = httpd_resp_set_type(
      request,
      "multipart/x-mixed-replace; boundary=frame"
  );

  if (result != ESP_OK) {
    return result;
  }

  httpd_resp_set_hdr(request, "Cache-Control", "no-store");

  while (true) {
    // Borrow a captured image from the camera driver
    camera_fb_t* frame = esp_camera_fb_get();

    if (frame == nullptr) {
      Serial0.println("Camera capture failed");
      return ESP_FAIL;
    }

    // Describe the JPEG that follows
    char header[128];
    int headerLength = snprintf(
        header,
        sizeof(header),
        "--frame\r\n"
        "Content-Type: image/jpeg\r\n"
        "Content-Length: %u\r\n\r\n",
        static_cast<unsigned int>(frame->len)
    );

    result = httpd_resp_send_chunk(request, header, headerLength);

    if (result == ESP_OK) {
      result = httpd_resp_send_chunk(
          request,
          reinterpret_cast<const char*>(frame->buf),
          frame->len
      );
    }

    if (result == ESP_OK) {
      result = httpd_resp_send_chunk(request, "\r\n", 2);
    }

    // Always return the buffer, including after a disconnected client
    esp_camera_fb_return(frame);

    if (result != ESP_OK) {
      Serial0.println("Stream client disconnected");
      return result;
    }

    delay(1);
  }
}


bool startStreamingServer() {
    httpd_config_t config = HTTPD_DEFAULT_CONFIG();
    config.server_port = 80;
    config.core_id = 0;
    config.task_priority = 4;

  httpd_handle_t server = nullptr;
  esp_err_t result = httpd_start(&server, &config);

  if (result != ESP_OK) {
    Serial0.printf("Server startup failed: 0x%x\n", result);
    return false;
  }

  httpd_uri_t streamRoute = {};
  streamRoute.uri = "/stream";
  streamRoute.method = HTTP_GET;
  streamRoute.handler = streamHandler;

  result = httpd_register_uri_handler(server, &streamRoute);

  if (result != ESP_OK) {
    Serial0.printf("Stream route registration failed: 0x%x\n", result);
    httpd_stop(server);
    return false;
  }

  return true;
}


bool startCameraStreaming() {
  if (!psramFound() || !initializeCamera()) {
    Serial0.println("Camera or PSRAM initialization failed");
    return false;
  }

  WiFi.mode(WIFI_AP);

  if (!WiFi.softAP("PEO-Camera", "peocamera123")) {
    Serial0.println("Wi-Fi startup failed");
    return false;
  }

  if (!startStreamingServer()) {
    return false;
  }

  Serial0.print("Stream ready: http://");
  Serial0.print(WiFi.softAPIP());
  Serial0.println("/stream");
  return true;
}