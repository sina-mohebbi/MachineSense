# MachineSense — On-Device Industrial Anomaly Detection

An **ESP32** that listens to a machine, runs a quantized **autoencoder** on-device, and
flags *"this machine sounds wrong"* in real time — trained entirely on a public dataset,
shipped with the CI/CD, tests, and OTA that a production embedded product needs.

> Successor to [library-desk-sense](https://github.com/mohebbixsina-debug/library-desk-sense):
> that project proved the full IoT pipeline (sensing → protocols → cloud → dashboards).
> **MachineSense pushes the intelligence to the edge** and ships it like a professional.

## What it does

Train an autoencoder on the sound of a **healthy** machine (public **MIMII** dataset).
Anything it cannot reconstruct well = an anomaly. No failure data required — which is how
real predictive-maintenance works. The model is quantized to **int8** and runs on a plain
**ESP32 DevKit** via **TensorFlow Lite for Microcontrollers**.

## Architecture

```
                          EDGE (ESP32 DevKit)
  log-mel vectors ──► TFLite-Micro autoencoder ──► reconstruction MSE ──► anomaly score
       ▲                                                                      │
       │ (replay: fed over USB/UART   |   live stretch: I2S mic + esp-dsp)    │ MQTT/TLS
       │                                                                      ▼
                                CLOUD (self-hosted, Docker)
   ESP32 ──► EMQX (broker + rule engine + data bridge) ──► TimescaleDB ──► Grafana
              │                                            (SQL time-series)  (+ alerts)
              └─ retained topic = anomaly-threshold config push
   OTA: ESP-IDF signed OTA over HTTPS, triggered by an MQTT topic
```

## Zero-hardware by design

The primary demo path is **replay mode**: held-out MIMII test clips (normal **and**
anomalous) are pre-processed to log-mel vectors on a PC and streamed to the ESP32 over the
USB cable. The device runs the **real quantized model** and computes **AUC on-device**.
Total hardware cost: **$0**. (Optional live stretch: a ~$4 I2S mic.)

## Repo layout

| Folder | Contents |
|---|---|
| `ml/` | **Phase 0** — train the autoencoder, evaluate AUC, export an int8 C header |
| `firmware/` | Phase 1–2 — ESP-IDF + FreeRTOS on-device inference |
| `cloud/` | Phase 3 — EMQX + TimescaleDB + Grafana (`docker-compose`) |
| `evaluation/` | Phase 5 — the benchmark study (int8-vs-float, edge-vs-cloud, …) |
| `docs/` | architecture notes, results table, wiring |

## Roadmap

- [x] **Phase 0** — Python: MIMII → log-mel → autoencoder → AUC → int8 export *(this scaffold)*
- [ ] **Phase 1** — TFLite-Micro on ESP32 in replay mode; reproduce AUC on-device
- [ ] **Phase 2** — FreeRTOS pipeline, threshold, serial/OLED readout
- [ ] **Phase 3** — EMQX → TimescaleDB → Grafana + alerts
- [ ] **Phase 4** — GitHub Actions CI, native unit tests, Docker, signed OTA
- [ ] **Phase 5** — evaluation study, README results, demo GIF

## Quick start (Phase 0)

```bash
cd ml
pip install -r requirements.txt
# download a MIMII machine type (e.g. fan) and unzip into ml/data/fan/
#   ml/data/fan/id_00/normal/*.wav
#   ml/data/fan/id_00/abnormal/*.wav
python train.py            # trains, prints AUC, saves artifacts/model.keras
python export_tflite.py    # int8 quantize -> artifacts/model_int8.tflite + model_data.cc
```

See [`ml/README.md`](ml/README.md) for details and where to get the dataset.
