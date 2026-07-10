# firmware/ — Phase 1–2: on-device inference (ESP-IDF + FreeRTOS)

> Not started yet — this is the Phase 1 target. Phase 0 (`ml/`) produces the model this
> firmware runs.

## Plan

**Phase 1 — replay mode (no sensor):** flash the int8 autoencoder (`model_data.cc` from
`ml/export_tflite.py`) and run it with **TensorFlow Lite for Microcontrollers**. Feature
vectors from held-out MIMII test clips are streamed in over USB/UART; the device computes
reconstruction MSE and reports predictions back, so we can reproduce the AUC **on-device**
and measure latency + RAM (tensor arena).

**Phase 2 — FreeRTOS pipeline:** tasks `ingest → infer → score → report`, a ring buffer
between sampling and inference, anomaly threshold, and a serial/OLED readout.

**Stretch — live:** compute log-mel on-device with `esp-dsp` (FFT) from an I2S mic
(INMP441, ~$4) so it runs on real sound instead of replayed vectors.

## Layout (to be created)

```
firmware/
  CMakeLists.txt
  main/
    main.cc            # FreeRTOS tasks
    model_data.cc/.h   # copied from ml/artifacts/ (git-ignored)
    features.*         # feature ingest / (stretch) on-device log-mel
    inference.*        # TFLite-Micro wrapper
  test/                # native Unity unit tests (host x86) -- Phase 4
```

## Notes

- Target: **ESP32** (classic WROOM DevKit).
- TFLite-Micro model is int8 in / int8 out — feed quantized vectors, dequantize the score.
- Keep the tensor arena size measured and documented for the results table.
