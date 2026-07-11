# firmware/ — Phase 1: on-device inference (replay mode)

Runs the real, quantized `id_02` autoencoder (from [`ml/`](../ml/)) on an ESP32 via
**TensorFlow Lite for Microcontrollers**, fed by held-out MIMII test vectors streamed
over the same USB-serial cable used for flashing. No sensor required — this
reproduces the host `evaluate_per_id_tflite.py` AUC (**0.858 for `id_02`**) on
real hardware.

> **Status:** this is a from-scratch scaffold. The Python side (`tools/replay_client.py`)
> is tested and passing (`firmware/tools/tests/`, no hardware/dataset needed). The
> **C++/ESP-IDF side has not been compiled or flashed** — there's no ESP-IDF toolchain
> or physical board in the environment this was written in. It follows the standard
> `esp-tflite-micro` API, but treat the first `idf.py build` as the real test; see
> **Troubleshooting** below for the likely first issues.

## How it works

```
firmware/tools/replay_client.py (PC)              firmware/main/*.cc (ESP32)
──────────────────────────────────────            ───────────────────────────
load id_02 held-out test WAVs (ml/data.py)
normalize + quantize each log-mel vector   -- 640 int8 bytes -->  TFLite-Micro
                                                                   .Invoke()
average per-vector scores into per-file    <-- 4 bytes float32 -- reconstruction
AUC (== evaluate_per_id_tflite.py's method)                       MSE score
```

The device never sees WAV files or normalization stats — only already-quantized
640-byte vectors. `main.cc` prints a `MACHINESENSE_READY` line once the model is
loaded, then goes silent (esp_log is muted) and speaks pure binary so the protocol
isn't corrupted by console text.

## 1. Regenerate the model files (they're git-ignored, generated from `ml/`)

```powershell
cd ..\ml
python export_per_id.py          # if not already run
```

Then copy `id_02`'s exported model in and rename its symbol to the generic name
`main.cc`/`inference.cc` expect (`g_model_data`, not `g_model_data_id_02`):

```powershell
cd ..\firmware
Copy-Item ..\ml\artifacts\per_id\id_02\model_data.cc main\model_data.cc
Copy-Item ..\ml\artifacts\per_id\id_02\model_data.h  main\model_data.h
(Get-Content main\model_data.cc) -replace 'g_model_data_id_02', 'g_model_data' | Set-Content main\model_data.cc
(Get-Content main\model_data.h)  -replace 'g_model_data_id_02', 'g_model_data' | Set-Content main\model_data.h
```

To target a different machine (e.g. `id_06`, the strongest at AUC 0.925), swap
`id_02` for that ID above and pass `--machine-id id_06` to `replay_client.py` later.

## 2. Install ESP-IDF (one-time)

Follow Espressif's [Windows install guide](https://docs.espressif.com/projects/esp-idf/en/stable/esp32/get-started/windows-setup.html)
(the **ESP-IDF Tools Installer** is the easiest path — it sets up the toolchain,
Python env, and an "ESP-IDF PowerShell" shortcut). Target: **esp32** (classic WROOM).

## 3. Build + flash

From an **ESP-IDF PowerShell** prompt:

```powershell
cd firmware
idf.py set-target esp32
idf.py build              # first build also pulls in esp-tflite-micro (idf_component.yml)
idf.py -p COM5 flash monitor   # replace COM5 with your device's port
```

You should see `MACHINESENSE_READY vector_len=640` in the monitor, then silence
(expected — logs are muted once the binary protocol starts). Exit the monitor
(`Ctrl+]`) before running the replay client, since only one program can hold the
serial port at a time.

## 4. Run the replay client

```powershell
cd ..\ml
pip install -r ..\firmware\tools\requirements.txt   # adds pyserial
cd ..\firmware\tools
python replay_client.py --port COM5 --machine-id id_02
```

Expect an on-device AUC close to the host int8 figure (**0.858**) — small
differences are normal, see `inference.h`'s docstring on the dequantize-input-vs-
original-float nuance. Try `--limit-files 20` first for a fast smoke test.

**No hardware yet?** Validate the client logic against the host TFLite interpreter
instead of real silicon:

```powershell
python replay_client.py --mock --machine-id id_02
```

## Troubleshooting (first build)

- **`Didn't find op 'X'`**: the resolver in `main/inference.cc` only registers
  `FullyConnected` + `Relu` (the ops a folded, int8-in/out dense autoencoder should
  need). If the real export uses another op, add `resolver.AddX()` there and bump
  `kNumOps`.
- **`AllocateTensors() failed`**: raise `kTensorArenaSize` in `inference.cc` (starts
  at 60 KB, generous but unverified against the real model's actual footprint).
  Once it works, the boot log prints the true `arena_used_bytes()` — shrink to that.
- **Garbled/no replay output**: something printed to stdout after logs were muted
  (e.g. a library that logs internally). Check `main.cc`'s mute point is truly last.

## Files

| File | Role |
|---|---|
| `main/main.cc` | boot, print `READY`, mute logs, binary UART loop |
| `main/inference.h/.cc` | TFLite-Micro interpreter setup + scoring |
| `main/model_data.h/.cc` | int8 model bytes (generated, see step 1) |
| `tools/replay_client.py` | PC-side driver: quantize, send, aggregate, AUC |
| `tools/tests/` | dataset-free, hardware-free tests for the client |

## Next: Phase 2

FreeRTOS task pipeline (`ingest → infer → score → report`), a real threshold +
serial/OLED readout, and (stretch) on-device log-mel via `esp-dsp` + an I2S mic
for a fully live demo. See [`../docs/architecture.md`](../docs/architecture.md).
