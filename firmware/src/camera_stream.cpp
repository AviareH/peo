#include <Arduino.h>
#include <WiFi.h>
#include "esp_camera.h"
#include "esp_http_server.h"
#include "as5600.h"
#include "ps4_controller.h"


const char* wifiName = "PEO-Camera";
const char* wifiPassword = "peocamera123";

bool initializeCamera() {
  // Zero-initialize optional driver fields, then apply this board's pin map.
  camera_config_t config = {};

  // Camera data and sync pins connect directly to the ESP32-S3.
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

  // Configure the camera XCLK using LEDC channel 0.
  config.ledc_channel = LEDC_CHANNEL_0;
  config.ledc_timer = LEDC_TIMER_0;
  config.xclk_freq_hz = 20000000;

  // JPEG/VGA keeps frames compact enough for streaming over the access point.
  config.pixel_format = PIXFORMAT_JPEG;
  config.frame_size = FRAMESIZE_VGA;
  config.jpeg_quality = 12;

  // Keep two frame buffers in PSRAM and prefer recent frames
  config.fb_location = CAMERA_FB_IN_PSRAM;
  config.fb_count = 2;
  config.grab_mode = CAMERA_GRAB_LATEST;

  esp_err_t result = esp_camera_init(&config);

  if (result != ESP_OK) {
    Serial.printf("Camera initialization failed: 0x%x\n", result);
    return false;
  }

  // The camera is mounted upside down. Reverse both sensor axes for a 180° rotation.
  sensor_t* sensor = esp_camera_sensor_get();
  if (sensor == nullptr ||
      sensor->set_hmirror(sensor, !sensor->status.hmirror) != 0 ||
      sensor->set_vflip(sensor, !sensor->status.vflip) != 0) {
    Serial.println("Camera orientation setup failed");
    esp_camera_deinit();
    return false;
  }

  Serial.println("Camera initialized");
  Serial.printf("Password: %s\n", wifiPassword);
  return true;
}

// Runs when a client opens /stream
esp_err_t streamHandler(httpd_req_t* request) {
  // Browser navigation opens the labelled viewer. Image and OpenCV clients
  // keep the MJPEG endpoint and receive JPEGs directly from the camera.
  char accept[512] = {};
  if (httpd_req_get_hdr_value_str(request, "Accept", accept, sizeof(accept)) == ESP_OK &&
      strstr(accept, "text/html") && !strchr(request->uri, '?')) {
    String location = "http://" + WiFi.softAPIP().toString() + ":81/";
    httpd_resp_set_status(request, "302 Found");
    httpd_resp_set_hdr(request, "Location", location.c_str());
    httpd_resp_set_hdr(request, "Vary", "Accept");
    return httpd_resp_send(request, nullptr, 0);
  }
  esp_err_t result = httpd_resp_set_type(
      request,
      "multipart/x-mixed-replace; boundary=frame"
  );

  if (result != ESP_OK) {
    return result;
  }

  httpd_resp_set_hdr(request, "Cache-Control", "no-store");

  // Keep the HTTP response open and stream each captured JPEG as a frame.
  while (true) {
    // Borrow a captured image from the camera driver
    camera_fb_t* frame = esp_camera_fb_get();

    if (frame == nullptr) {
      Serial.println("Camera capture failed");
      return ESP_FAIL;
    }

    // Describe the JPEG that follows
    char header[128];
    int headerLength = snprintf(
        header,
        sizeof(header),
        "--frame\r\n"
        "Content-Type: image/jpeg\r\n"
        "X-Drive-Mode: %s\r\n"
        "Content-Length: %u\r\n\r\n",
        driveModeName(),
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
      Serial.println("Stream client disconnected");
      return result;
    }

    delay(1);
  }
}

static esp_err_t modeHandler(httpd_req_t* request) {
  httpd_resp_set_type(request, "text/plain");
  httpd_resp_set_hdr(request, "Cache-Control", "no-store");
  return httpd_resp_sendstr(request, driveModeName());
}

static esp_err_t encoderHandler(httpd_req_t* request) {
  const auto diagnostics = readEncoderDiagnostics();
  const auto& sample = diagnostics.sample;
  char response[256];
  const int length = snprintf(response, sizeof(response),
      "{\"left\":{\"status\":\"%s\",\"count\":%lld},"
      "\"right\":{\"status\":\"%s\",\"count\":%lld},\"sample_age_ms\":%lu}\n",
      encoderStateName(diagnostics.left), sample.left,
      encoderStateName(diagnostics.right), sample.right,
      static_cast<unsigned long>(millis() - sample.sampledAt));
  if (length < 0 || length >= static_cast<int>(sizeof(response))) {
    return httpd_resp_send_500(request);
  }
  httpd_resp_set_type(request, "application/json");
  httpd_resp_set_hdr(request, "Cache-Control", "no-store");
  return httpd_resp_send(request, response, length);
}

static esp_err_t viewerHandler(httpd_req_t* request) {
  static const char page[] = R"HTML(<!doctype html>
<html lang="en"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>PEO camera</title>
<style>body{margin:0;background:#151515;color:white;font:600 18px system-ui}main{position:relative;width:fit-content;margin:auto}img{display:block;max-width:100vw;max-height:100vh}output{position:absolute;top:8px;left:8px;background:#111e;padding:4px 10px}</style>
<main><img id="camera" alt="Robot camera"><output id="mode">Mode: UNKNOWN</output></main>
<script>
const stream = new URL('/stream?raw=1', location.href);
stream.port = '80';
document.getElementById('camera').src = stream;
async function updateMode() {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), 600);
  let mode = 'UNKNOWN';
  try {
    const result = await fetch('/mode', {cache:'no-store', signal:controller.signal});
    const value = await result.text();
    if (result.ok && ['MANUAL','AUTONOMOUS'].includes(value)) mode = value;
  } catch (_) {} finally {
    clearTimeout(timer);
    document.getElementById('mode').textContent = 'Mode: ' + mode;
    setTimeout(updateMode, 250);
  }
}
updateMode();
</script></html>)HTML";
  httpd_resp_set_type(request, "text/html");
  return httpd_resp_send(request, page, sizeof(page) - 1);
}

bool startStreamingServer() {
  // Pin the HTTP server to core 0, separate from motor control on core 1.
  httpd_config_t config = HTTPD_DEFAULT_CONFIG();
  config.server_port = 80;
  config.core_id = 0;
  config.task_priority = 4;
  config.max_open_sockets = 2;
  config.lru_purge_enable = true;

  httpd_handle_t server = nullptr;
  esp_err_t result = httpd_start(&server, &config);

  if (result != ESP_OK) {
    Serial.printf("Server startup failed: 0x%x\n", result);
    return false;
  }

  httpd_uri_t streamRoute = {};
  // Expose the multipart JPEG handler at /stream.
  streamRoute.uri = "/stream";
  streamRoute.method = HTTP_GET;
  streamRoute.handler = streamHandler;

  result = httpd_register_uri_handler(server, &streamRoute);

  if (result != ESP_OK) {
    Serial.printf("Stream route registration failed: 0x%x\n", result);
    // Do not leave a server running without its stream endpoint.
    httpd_stop(server);
    return false;
  }

  // A streaming handler occupies its HTTP worker for the lifetime of a client.
  // Give read-only status and the viewer their own worker and control socket.
  config.server_port = 81;
  ++config.ctrl_port;
  httpd_handle_t status = nullptr;
  httpd_uri_t viewer = {};
  viewer.uri = "/";
  viewer.method = HTTP_GET;
  viewer.handler = viewerHandler;
  httpd_uri_t mode = {};
  mode.uri = "/mode";
  mode.method = HTTP_GET;
  mode.handler = modeHandler;
  httpd_uri_t encoders = {};
  encoders.uri = "/encoders";
  encoders.method = HTTP_GET;
  encoders.handler = encoderHandler;
  if (httpd_start(&status, &config) != ESP_OK ||
      httpd_register_uri_handler(status, &viewer) != ESP_OK ||
      httpd_register_uri_handler(status, &mode) != ESP_OK ||
      httpd_register_uri_handler(status, &encoders) != ESP_OK) {
    if (status) httpd_stop(status);
    httpd_stop(server);
    return false;
  }
  return true;
}


bool startCameraStreaming() {
  // Camera frame buffers are allocated in PSRAM, so fail before Wi-Fi startup.
  if (!psramFound() || !initializeCamera()) {
    Serial.println("Camera or PSRAM initialization failed");
    return false;
  }

  // The computer joins this access point for both video and motor commands.
  WiFi.mode(WIFI_AP);

  if (!WiFi.softAP("PEO-Camera", "peocamera123")) {
    Serial.println("Wi-Fi startup failed");
    return false;
  }

  if (!startStreamingServer()) {
    return false;
  }

  Serial.print("Stream ready: http://");
  Serial.print(WiFi.softAPIP());
  Serial.println("/stream");
  return true;
}
