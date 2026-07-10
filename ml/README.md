# ml/ — Phase 0: train the anomaly detector

Trains a dense autoencoder on **normal** machine sound (MIMII) and evaluates anomaly-detection
**AUC**, then quantizes to **int8** and exports a C header for TensorFlow Lite for Microcontrollers.

## 1. Install

```bash
pip install -r requirements.txt
```

## 2. Get the dataset

MIMII (DCASE 2020 Task 2) is public on Zenodo: https://zenodo.org/record/3384388

Download **one** machine type to start (fan is a good first pick) at the cleanest SNR
(the `+6 dB` archive), and unzip so the layout is:

```
ml/data/fan/id_00/normal/*.wav
ml/data/fan/id_00/abnormal/*.wav
ml/data/fan/id_02/...
```

Set `MACHINE` / `MACHINE_IDS` in `config.py` if you use a different type or ids.

## 3. Train + evaluate

```bash
python train.py
```

Prints overall + per-id AUC and writes `artifacts/model.keras` and `artifacts/metrics.json`.
Extracted log-mel vectors are cached under `artifacts/feature_cache/`, so later runs do
not process the same WAV files again.

Feature-wise mean and standard deviation are learned from normal training data only and
saved to `artifacts/normalization.npz`. The same transformation must be applied before
inference on the ESP32.

Validation is split by complete WAV files, preventing overlapping feature windows from
the same recording appearing in both training and validation. Early stopping restores
the weights with the best validation loss, and `artifacts/history.json` stores the loss
curves.

Start with a small end-to-end experiment before the full run:

```bash
python train.py --epochs 5 --max-train-files 200 --max-test-per-class 20
```

These limits are for pipeline validation only; do not report their AUC as the final result.

## 4. Quantize + export for the ESP32

```bash
python export_tflite.py
python evaluate_tflite.py
```

Writes `artifacts/model_int8.tflite` and `artifacts/model_data.cc` (+ `.h`). In Phase 1
these get copied into `firmware/main/` and compiled into the ESP32 image. The evaluation
command writes `artifacts/metrics_int8.json` so quantization can be compared with float32.

## Files

| File | Role |
|---|---|
| `config.py` | all audio/feature/model/export parameters (DCASE baseline) |
| `data.py` | MIMII loading + log-mel context-vector extraction |
| `model.py` | the autoencoder topology |
| `train.py` | train + AUC evaluation + save artifacts |
| `export_tflite.py` | int8 quantization + C-header emit |
| `evaluate_tflite.py` | host int8 AUC evaluation before device deployment |
| `tests/test_smoke.py` | dataset-free tests (also run in CI) |

## How the anomaly score works

Each 10 s clip becomes many 640-dim log-mel context vectors. The autoencoder is trained
only on normal clips, so it reconstructs normal sound with low error. A clip's anomaly
score is the **mean reconstruction MSE** over its vectors; higher = more anomalous. AUC
measures how well that score separates normal from abnormal test clips.

## Reproduce the smoke tests locally

```bash
cd ml && pytest -q          # no dataset required
```
