# firmware/ — on-device inference + anomaly detection (Phases 1–2)

Runs the quantized `id_02` autoencoder (from [`ml/`](../ml/)) on a real **ESP32** via
**TensorFlow Lite for Microcontrollers**, organized as a **FreeRTOS task pipeline**,
and decides anomalies against a threshold — driving the onboard LED. Held-out MIMII
test vectors are streamed over the USB-serial cable (no sensor needed), so the
on-device results can be checked against the host.

> **Status: built, flashed, and verified on real hardware** (ESP-IDF v6.0.1, ESP32
> WROOM). On-device AUC matches the host (`0.858` full set; `1.000` on an easy
> 12-clip subset, per-vector scores within ~0.001 of the host). Anomaly detection at
> the exported threshold: **F1 ≈ 0.81** (precision 0.85, recall 0.76). Sustained
> multi-thousand-vector runs are crash-free with zero checksum/flag mismatches.

## Architecture (Phase 2)

```
 replay_client.py (PC)                 ESP32 — 3 FreeRTOS tasks + 2 queues
 ─────────────────────                 ────────────────────────────────────
 load id_02 test WAVs                  rx_task   read 2560B float vector,
 normalize each log-mel vector  ─2560B→           checksum it → [vector queue]
                                       infer_task quantize once, TFLite Invoke,
                                                  score, score>threshold?→anomaly,
                                                  set LED → [result queue]
 aggregate clip scores → AUC    ←─12B─ tx_task    write {score, anomaly, checksum}
 + precision/recall/F1 @ threshold
```

- **Protocol** — host → device: 640 `float32` LE (2560 B), one normalized log-mel
  vector. Device → host: 12 B = `float32` score + `uint32` anomaly flag + `uint32`
  byte-sum checksum of the received request.
- **Why float32, not int8, over the wire:** the device quantizes once internally, so
  it scores the reconstruction against the *true* input — matching `ml/`'s
  methodology exactly (see `inference.h`).
- **Why a checksum:** it caught the console UART silently translating CR↔LF in the
  binary stream (see notes below). It also guards against transient bit errors.

## Setup

### 1. Generate the model + threshold (git-ignored, produced from `ml/`)

```powershell
cd ..\ml
python export_per_id.py        # -> artifacts/per_id/id_02/model_data.cc/.h
python compute_threshold.py --machine-id id_02   # -> firmware/main/threshold.h
```

Copy the model in and rename its symbol to the generic name the firmware expects:

```powershell
cd ..\firmware
Copy-Item ..\ml\artifacts\per_id\id_02\model_data.cc main\model_data.cc
Copy-Item ..\ml\artifacts\per_id\id_02\model_data.h  main\model_data.h
(Get-Content main\model_data.cc) -replace 'g_model_data_id_02','g_model_data' | Set-Content main\model_data.cc
(Get-Content main\model_data.h)  -replace 'g_model_data_id_02','g_model_data' | Set-Content main\model_data.h
```

To target a different unit (e.g. `id_06`, strongest at AUC 0.925), swap `id_02`
above and pass `--machine-id id_06` to both `compute_threshold.py` and the client.

### 2. Install ESP-IDF (one-time)

Espressif's [Windows installer](https://docs.espressif.com/projects/esp-idf/en/stable/esp32/get-started/windows-setup.html)
(target **esp32**). Then, in an ESP-IDF-enabled shell:

### 3. Build + flash + run

```powershell
cd firmware
idf.py set-target esp32          # first time only
idf.py -p COM7 build flash       # first build also fetches esp-tflite-micro

cd tools
pip install -r requirements.txt  # pyserial
python replay_client.py --port COM7 --machine-id id_02 --limit-files 20
```

Boot prints `MACHINESENSE_READY vector_len=640 threshold=0.594029`, then goes silent
(logs muted so they don't corrupt the binary protocol). The client reports on-device
AUC, precision/recall/F1 at the threshold, and a per-vector flag-integrity check.
Full runs are slow (~3.6 vec/s); use `--limit-files` for spot checks.

**No hardware?** `python replay_client.py --mock --machine-id id_02` runs the exact
device math in Python (validates the full pipeline, reproduces AUC 0.858 / F1 0.81).

## Notes / gotchas found during bring-up (all fixed)

- **UART CR↔LF translation** — the ESP-IDF console converts line endings on
  stdin/stdout by default, silently corrupting any `0x0D`/`0x0A` byte in binary data.
  Disabled via `uart_vfs_dev_port_set_*_line_endings(..., ESP_LINE_ENDINGS_LF)`.
- **Task Watchdog** — the UART VFS read is non-blocking; a naive `if(n<=0) continue;`
  busy-spins and starves the idle task, tripping the WDT. Fixed with `vTaskDelay(1)`
  on an empty read.
- **`Didn't find op`** at build → add the missing kernel in `inference.cc`'s resolver.
- **`AllocateTensors() failed`** → raise `kTensorArenaSize` (currently 24 KB; the boot
  log prints the true `arena_used_bytes()` ≈ 15.8 KB).

## Files

| File | Role |
|---|---|
| `main/main.cc` | FreeRTOS pipeline (rx→infer→tx), threshold, LED, UART setup |
| `main/inference.h/.cc` | TFLite-Micro interpreter, quantize-once + score |
| `main/model_data.h/.cc` | int8 model bytes (generated) |
| `main/threshold.h` | anomaly threshold (generated by `compute_threshold.py`) |
| `tools/replay_client.py` | PC driver: stream vectors, AUC + anomaly metrics |
| `tools/diagnose_mismatch.py` | compare mock vs real per-vector (bring-up aid) |
| `tools/tests/` | dataset-free, hardware-free tests |

## Next: Phase 3

Stream anomaly scores over MQTT into the EMQX + TimescaleDB + Grafana stack already
scaffolded in [`../cloud/`](../cloud/). Stretch: on-device log-mel via `esp-dsp` + an
I2S mic for a fully live (sensor-driven) demo. See
[`../docs/architecture.md`](../docs/architecture.md).
