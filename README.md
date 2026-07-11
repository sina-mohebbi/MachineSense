# MachineSense - ESP32 On-Device Anomaly Detection

MachineSense is an embedded ML project for industrial sound anomaly detection. It
trains an autoencoder on healthy machine audio, exports the model to int8
TensorFlow Lite, and runs the anomaly detector on a real ESP32 with TensorFlow
Lite for Microcontrollers.

Current status: **Phase 1 and Phase 2 are working.** The ML pipeline trains on a
laptop, exports deployable models, and the ESP32 firmware has been built,
flashed, and validated in replay mode on real hardware.

## What Works Now

- Laptop ML pipeline for the MIMII fan dataset.
- Leakage-safe train/test split by machine ID and clip.
- Pooled autoencoder baseline and per-machine-ID autoencoders.
- Full int8 TFLite export for ESP32 deployment.
- Generated C/C++ model data and anomaly thresholds for firmware.
- ESP32 TensorFlow Lite Micro inference.
- FreeRTOS replay pipeline: UART input, inference, scoring, anomaly flag, LED.
- Host replay client that streams held-out feature vectors to the board.
- Dataset-free tests for the ML smoke path and firmware replay tooling.
- Basic GitHub Actions CI for Python lint and tests.

## Latest Results

| Path | Result |
|---|---:|
| Pooled float model AUC | 0.7130 |
| Pooled int8 model AUC | 0.6947 |
| Per-ID float macro AUC | 0.7677 approx |
| Per-ID int8 macro AUC | 0.7677 |
| Best per-ID int8 AUC (`id_06`) | 0.9256 |
| Deployed ESP32 target (`id_02`) int8 AUC | 0.8578 host-side |
| ESP32 replay validation | Built, flashed, verified |
| Firmware binary size | 501,616 bytes |

The firmware README documents real ESP32 WROOM validation. On-device replay for
`id_02` matched the host path closely, with no checksum or anomaly-flag
mismatches in the documented runs. A short 20-clip hardware spot check is also
saved in `ml/artifacts/per_id/id_02/metrics_on_device.json`.

## Architecture

```text
Laptop training
  MIMII WAV files
    -> log-mel features
    -> autoencoder training
    -> int8 TFLite export
    -> C/C++ model data + threshold

ESP32 replay mode
  PC replay_client.py
    -> UART feature vectors
    -> ESP32 FreeRTOS rx task
    -> TFLite Micro inference task
    -> reconstruction error + anomaly threshold
    -> UART result + LED alert

Next cloud loop
  ESP32 anomaly event
    -> MQTT / EMQX
    -> TimescaleDB
    -> Grafana dashboard + alerts
```

## Repo Layout

| Folder | Purpose |
|---|---|
| `ml/` | Training, evaluation, quantization, model export, thresholds |
| `firmware/` | ESP-IDF firmware, TFLite Micro inference, UART replay |
| `cloud/` | Phase 3 scaffold for EMQX, TimescaleDB, and Grafana |
| `evaluation/` | Planned benchmark/report package |
| `docs/` | Architecture notes and supporting documentation |

## Roadmap

- [x] **Phase 0 - ML baseline:** MIMII preprocessing, autoencoder training, AUC evaluation.
- [x] **Phase 1 - ESP32 replay inference:** int8 TFLite Micro model running on the board.
- [x] **Phase 2 - Firmware pipeline:** FreeRTOS tasks, thresholding, LED/serial result path.
- [ ] **Phase 3 - Connected telemetry:** publish anomaly events over MQTT to EMQX.
- [ ] **Phase 4 - Production hardening:** firmware CI, OTA, TLS/auth, device configuration.
- [ ] **Phase 5 - Final evaluation:** repeatable benchmark package, dashboard screenshots, demo media.

## Quick Start

### Train and export the model

```powershell
cd ml
pip install -r requirements.txt
python train.py
python export_tflite.py
```

For the per-machine-ID deployment path:

```powershell
cd ml
python train_per_id.py
python export_per_id.py
python compute_threshold.py --machine-id id_02
```

### Build and test the ESP32 firmware

```powershell
cd firmware
idf.py set-target esp32
idf.py -p COM7 build flash

cd tools
pip install -r requirements.txt
python replay_client.py --port COM7 --machine-id id_02
```

No board attached? Use the mock path to test the replay aggregation logic:

```powershell
cd firmware/tools
python replay_client.py --mock --machine-id id_02
```

### Run tests

```powershell
.\ml\.venv\Scripts\python.exe -m pytest -q ml firmware/tools/tests
```

Current local result: **15 passed**.

## Known Gaps

- Live microphone capture is not implemented yet; current hardware validation uses
  replayed feature vectors over UART.
- The cloud stack is scaffolded, but EMQX rules, TimescaleDB schema, Grafana
  provisioning, TLS, and MQTT auth still need to be completed.
- Firmware build CI is planned but not active yet.
- OTA and secure device-management flows are not implemented yet.
- Final evaluation artifacts, screenshots, and demo media are still pending.

## Next Step

The next small milestone is to send one anomaly event into MQTT:

```text
ESP32 or replay bridge -> EMQX topic -> visible message in broker dashboard
```

After that, the project can connect the event to TimescaleDB and Grafana.
