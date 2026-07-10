# MachineSense architecture

## Pipeline

```
                          EDGE (ESP32 DevKit)
  log-mel vectors ──► TFLite-Micro autoencoder ──► reconstruction MSE ──► anomaly score
       ▲                                                                      │
       │ replay: USB/UART   |   live stretch: I2S mic + esp-dsp               │ MQTT/TLS
       ▼                                                                      ▼
                                CLOUD (self-hosted, Docker)
   ESP32 ──► EMQX (broker + rule engine + data bridge) ──► TimescaleDB ──► Grafana
              │                                            (SQL time-series)  (+ alerts)
              └─ retained topic = anomaly-threshold config push
   OTA: ESP-IDF signed OTA over HTTPS, triggered by an MQTT topic
```

## Design decisions

- **Unsupervised autoencoder**, not a classifier — trained on *normal* sound only, so no
  labeled failures are needed (realistic for predictive maintenance).
- **Sensor-free by default** via replay mode — the ESP32 runs the real quantized model on
  pre-processed MIMII vectors, so the whole project costs $0 and needs no hardware.
- **EMQX rule engine + data bridge** replaces a custom proxy — telemetry lands in
  TimescaleDB without glue code.
- **int8 full-integer quantization** — required for TFLite-Micro; the cost is measured in
  the evaluation study.

## What this project adds over library-desk-sense

| | library-desk-sense | MachineSense |
|---|---|---|
| Intelligence | server-side analytics | **on-device** (edge AI) |
| Signal processing | low-rate sensing | **log-mel / DSP features** |
| Broker | Mosquitto + Python proxy | **EMQX** (native rule engine) |
| Storage | InfluxDB | **TimescaleDB (SQL)** |
| CI/CD | none | **GitHub Actions + tests** |
| OTA | none | **signed OTA** |

## Results (fill in after each phase)

| Metric | Value |
|---|---|
| Overall AUC (float32) | _tbd_ |
| Overall AUC (int8, on-device) | _tbd_ |
| Model size (int8) | _tbd_ KiB |
| Tensor-arena RAM | _tbd_ KiB |
| Inference latency (ESP32) | _tbd_ ms |
| Bytes/clip: edge score vs raw audio | _tbd_ |
