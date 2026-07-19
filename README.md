# MachineSense - ESP32 On-Device Anomaly Detection

MachineSense is an embedded ML project for industrial sound anomaly detection. It
trains an autoencoder on healthy machine audio, exports the model to int8
TensorFlow Lite, and runs the anomaly detector on a real ESP32 with TensorFlow
Lite for Microcontrollers.

Current status: **final ESP32 model-evaluation scope is complete.** The ML
pipeline trains on a laptop, exports deployable int8 models, and the ESP32
firmware has been built, flashed, and validated in replay mode on real hardware.

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
- GitHub Actions CI running lint plus both dataset-free test suites (ML and
  firmware replay tooling).
- Follow-up experiments for the difficult `id_00` case, including alternate
  clip scoring, longer context windows, and a small Conv2D autoencoder.

## Results

The deployed configuration is the per-machine-ID model for `id_02`, running on an
ESP32 WROOM in replay mode.

| Deployed model (`id_02`) | Result |
|---|---:|
| Anomaly detection AUC (int8) | 0.8578 |
| Anomaly detection F1 at threshold | 0.81 (precision 0.85, recall 0.76) |
| On-device inference latency | 49.2 ms per feature vector |
| Tensor-arena RAM used | 15,756 of 24,576 bytes |
| Firmware binary size | 495,088 bytes |
| Hardware validation | Built, flashed, replay-verified on real ESP32 |

On-device replay matched the host path closely: per-vector scores agreed to within
about 0.001, with no checksum or anomaly-flag mismatches across multi-thousand-vector
runs. A 20-clip hardware spot check is saved in
`ml/artifacts/per_id/id_02/metrics_on_device.json`.

### Model comparison

| Path | AUC |
|---|---:|
| Pooled float | 0.7130 |
| Pooled int8 | 0.6947 |
| Per-ID float (macro) | 0.7677 |
| Per-ID int8 (macro) | 0.7677 |
| Best per-ID int8 (`id_06`) | 0.9256 |
| Weakest per-ID int8 (`id_00`) | 0.5626 |

Per-machine-ID models beat the pooled baseline, and int8 quantization did not
meaningfully reduce the macro AUC.

### Inference latency

Measured on the board at boot: 100 timed `Invoke()` calls, printed as
`MACHINESENSE_LATENCY`. It comes out at 49.2 ms per feature vector and varies by only
about 52 us, as expected for a fixed-size dense int8 graph. Host replay throughput is
not a proxy for this - at 115200 baud the 2560-byte request alone takes about 222 ms,
so UART transfer hides the real compute cost.

A 10 s clip is about 309 feature vectors, so roughly 15 s of inference: about 1.5x
slower than real time at the full frame rate. That is fine for replay-mode evaluation,
but a live-microphone build would need frame subsampling (a hop of ~768 instead of 512)
or an ESP32-S3. Measured: the cost is not compiler-related, since `-Og` to `-O2` moved
it only from 49.3 ms to 49.2 ms. Inferred but not profiled: a dense-only model on a
classic ESP32 has no SIMD to exploit, and `esp-nn` mainly accelerates convolution
rather than fully-connected layers.

### `id_00`: a documented limitation

`id_00` stays close to random ranking, and four approaches failed to fix it:

| Experiment | `id_00` AUC |
|---|---:|
| Dense autoencoder, `FRAMES=5` (baseline) | 0.5626 |
| Dense autoencoder, `FRAMES=10` | 0.5931 |
| Conv2D autoencoder | 0.5392 |
| Z-score detector | 0.5453 |

Score diagnostics show that `id_00` normal and abnormal clips have heavily overlapping
reconstruction-error distributions. This is a separability limit of log-mel
reconstruction scoring for this machine, not a deployment bug: the identical pipeline
reaches 0.8578 on `id_02` and 0.9256 on `id_06`. It is reported as a limitation rather
than dropped from the results.

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
```

## Repo Layout

| Folder | Purpose |
|---|---|
| `ml/` | Training, evaluation, quantization, model export, thresholds |
| `firmware/` | ESP-IDF firmware, TFLite Micro inference, UART replay |
| `docs/` | Architecture notes and supporting documentation |

## Roadmap

- [x] **Phase 0 - ML baseline:** MIMII preprocessing, autoencoder training, AUC evaluation.
- [x] **Phase 1 - ESP32 replay inference:** int8 TFLite Micro model running on the board.
- [x] **Phase 2 - Firmware pipeline:** FreeRTOS tasks, thresholding, LED/serial result path.
- [x] **Phase 3 - Final evaluation:** compare laptop, int8, and ESP32 replay behavior.
- [ ] **Optional future work - Production hardening:** firmware CI, OTA, device configuration.

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
- Firmware build CI, OTA, and secure device-management flows are not
  implemented because the project focus is on model-on-chip evaluation.
- `id_00` is not reliably detectable with this method (see the limitation section
  above). The deployed `id_02` path is unaffected.

## Next Step

The project is ready for final reporting/demo at the current scope:

```text
laptop training -> int8 export -> ESP32 replay inference -> laptop/board comparison
```
