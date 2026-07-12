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
- Basic GitHub Actions CI for Python lint and tests.
- Follow-up experiments for the difficult `id_00` case, including alternate
  clip scoring, longer context windows, and a small Conv2D autoencoder.

## Latest Results

| Path | Result |
|---|---:|
| Pooled float model AUC | 0.7130 |
| Pooled int8 model AUC | 0.6947 |
| Per-ID float macro AUC | 0.7677 |
| Per-ID int8 macro AUC | 0.7677 |
| Best per-ID int8 AUC (`id_06`) | 0.9256 |
| Deployed ESP32 target (`id_02`) int8 AUC | 0.8578 host-side |
| ESP32 replay validation (`id_02`) | Built, flashed, verified |
| Firmware binary size | 501,616 bytes |

The firmware README documents real ESP32 WROOM validation. On-device replay for
`id_02` matched the host path closely, with no checksum or anomaly-flag
mismatches in the documented runs. A short 20-clip hardware spot check is also
saved in `ml/artifacts/per_id/id_02/metrics_on_device.json`.

Final evaluation takeaway: the laptop and ESP32 paths agree closely for the
deployed `id_02` model, and quantization did not meaningfully reduce the per-ID
macro AUC. The strongest results are on `id_02` and `id_06`; `id_00` is kept as a
documented hard case rather than hidden.

### `id_00` follow-up experiments

| Experiment | `id_00` AUC | Outcome |
|---|---:|---|
| Dense per-ID autoencoder, `FRAMES=5` | 0.5626 | Current baseline |
| Dense per-ID autoencoder, `FRAMES=10` | 0.5931 int8 | Small improvement, larger input/model |
| Conv2D autoencoder, `FRAMES=5` | 0.5392 | Worse than baseline |
| Simple Z-score detector | 0.5453 | Worse than baseline |

The `id_00` diagnostics showed heavy overlap between normal and abnormal
reconstruction-error scores, so the weak result appears to be a data/model
separability limitation for this machine ID rather than an ESP32 deployment bug.

### How `id_00` should be reported

`id_00` should be presented as a negative/limitation case in the final report,
not as a broken deployment. The same training, export, quantization, and replay
pipeline works well for stronger IDs such as `id_02` and `id_06`, while `id_00`
remains close to random ranking under reconstruction-error scoring.

The best interpretation is:

- The ESP32/TFLite-Micro implementation is not the source of the weak `id_00`
  result, because the laptop and board paths agree closely on the deployed
  `id_02` model.
- The dense autoencoder baseline for `id_00` reaches only AUC 0.5626, and
  alternative scoring strategies did not solve the issue.
- Increasing the context window to `FRAMES=10` improved `id_00` only modestly to
  0.5931 int8, while increasing model/input size.
- A small Conv2D autoencoder and simple Z-score detector both performed worse
  than the dense baseline.
- Score diagnostics show that `id_00` normal and abnormal clips have very similar
  reconstruction-error distributions, meaning this machine ID is weakly
  separable with the current log-mel autoencoder approach.

For the final project, the correct conclusion is therefore: MachineSense
successfully demonstrates laptop-to-ESP32 model deployment and on-device anomaly
scoring, while also identifying `id_00` as a documented limitation of the chosen
unsupervised reconstruction method.

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
| `evaluation/` | Planned benchmark/report package |
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
- `id_00` remains a difficult machine ID: reconstruction-error scores for normal
  and abnormal clips overlap strongly. Follow-up experiments improved it only
  slightly or made it worse, so the limitation is documented rather than hidden.

## Next Step

The project is ready for final reporting/demo at the current scope:

```text
laptop training -> int8 export -> ESP32 replay inference -> laptop/board comparison
```
