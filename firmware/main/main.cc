// Phase 1 replay-mode entry point.
//
// Protocol (over the same USB-serial port used for flashing/idf.py monitor):
//   host  -> device : 640 float32 values, little-endian (2560 bytes -- one
//                                          already-normalized log-mel vector,
//                                          see ml/config.py FEATURE_DIM)
//   device -> host  : 8 bytes             (4-byte float32 LE reconstruction-
//                                          error score, then a 4-byte
//                                          uint32 LE checksum of the request
//                                          bytes the device actually received)
// repeated once per vector. Sending float (not pre-quantized int8) lets the
// device do its own single quantization step and score against the true
// input, matching ml/evaluate_per_id_tflite.py's methodology -- see
// inference.h's RunOnFloatVector docstring for why that matters.
//
// The echoed checksum guards against corrupted requests: replay_client.py
// recomputes the same byte-sum over what it sent and retries the vector if
// they don't match. This caught THE key bug of Phase 1 -- the ESP-IDF console
// UART does CR<->LF line-ending translation on stdin/stdout by default, so
// every 0x0D byte in a binary float32 vector silently arrived as 0x0A,
// mangling ~9 bytes per vector and wrecking the on-device AUC (a 12-clip
// subset that scores 1.0 in Python came out as 0.44/0.64). The fix is the
// uart_vfs_dev_port_set_*_line_endings() calls below, which disable that
// translation; the checksum stays as a genuine integrity check.
//
// This lets firmware/tools/replay_client.py replay held-out MIMII test
// vectors and reconstruct an on-device AUC with no sensor attached -- see
// ../README.md.
//
// esp_log output must stay OFF once the loop starts: it shares the same UART
// as this binary protocol and would corrupt it. Setup logs are left on until
// the model is confirmed ready, then muted right before the READY marker.
#include <cstdint>
#include <cstdio>
#include <cstring>

#include "driver/uart.h"       // UART_NUM_0
#include "driver/uart_vfs.h"   // uart_vfs_dev_port_set_*_line_endings
#include "esp_log.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"     // vTaskDelay
#include "inference.h"

namespace {

uint32_t Checksum(const uint8_t* data, int len) {
  uint32_t sum = 0;
  for (int i = 0; i < len; i++) {
    sum += data[i];
  }
  return sum;
}

}  // namespace

extern "C" void app_main(void) {
  if (!inference::Init()) {
    ESP_LOGE("main", "inference::Init() failed -- halting");
    return;
  }

  const int vector_len = inference::FloatInputSize();

  // Human-readable marker printed BEFORE logs are muted, so `idf.py monitor`
  // (or any terminal) can confirm the device booted. The host replay client
  // waits for this exact line before it starts sending binary frames.
  printf("MACHINESENSE_READY vector_len=%d\n", vector_len);
  fflush(stdout);

  esp_log_level_set("*", ESP_LOG_NONE);
  setvbuf(stdin, nullptr, _IONBF, 0);
  setvbuf(stdout, nullptr, _IONBF, 0);

  // CRITICAL for a binary protocol: the ESP-IDF console UART converts line
  // endings on stdin/stdout by default (incoming CR -> LF, outgoing LF ->
  // CRLF). That silently rewrites any 0x0D/0x0A byte in the float32 vectors
  // and score frames -- it was the root cause of the on-device AUC being far
  // below the host figure. ESP_LINE_ENDINGS_LF means "no modification", so the
  // bytes pass through exactly as sent/received.
  uart_vfs_dev_port_set_rx_line_endings(UART_NUM_0, ESP_LINE_ENDINGS_LF);
  uart_vfs_dev_port_set_tx_line_endings(UART_NUM_0, ESP_LINE_ENDINGS_LF);

  float* buffer = new float[vector_len];
  const int buffer_bytes = vector_len * static_cast<int>(sizeof(float));
  auto* bytes = reinterpret_cast<uint8_t*>(buffer);

  while (true) {
    int received = 0;
    while (received < buffer_bytes) {
      int n = fread(bytes + received, 1, buffer_bytes - received, stdin);
      if (n <= 0) {
        // The UART VFS read is non-blocking (returns 0 when the RX FIFO is
        // momentarily empty), so without yielding here this loop busy-spins at
        // 100% CPU during the ~222ms it takes to receive one 2560-byte vector.
        // That starves the idle task and trips the Task Watchdog after ~5s
        // (~vector 17). Sleep one tick so idle runs and the WDT stays fed.
        vTaskDelay(1);
        continue;
      }
      received += n;
    }
    const uint32_t checksum = Checksum(bytes, buffer_bytes);

    float mse = 0.0f;
    if (!inference::RunOnFloatVector(buffer, vector_len, &mse)) {
      mse = -1.0f;  // sentinel: host treats a negative score as an error
    }

    fwrite(&mse, sizeof(mse), 1, stdout);
    fwrite(&checksum, sizeof(checksum), 1, stdout);
    fflush(stdout);
  }
}
