# MachineSense architecture

## Pipeline

```
                          EDGE (ESP32 DevKit)
  log-mel vectors ──► TFLite-Micro autoencoder ──► reconstruction MSE ──► score
       ▲                                                                     │
       │ replay: host streams held-out MIMII vectors over USB/UART           │
       │ (live stretch: I2S mic + esp-dsp)                score > threshold ──┤
       │                                                  ─► anomaly flag + LED
       └──────────── 12-byte reply (score, anomaly flag, checksum) ◄─────────┘
```

## Design decisions

- **Unsupervised autoencoder**, not a classifier — trained on *normal* sound only, so no
  labeled failures are needed (realistic for predictive maintenance).
- **Sensor-free by default** via replay mode — the ESP32 runs the real quantized model on
  pre-processed MIMII vectors, so validation needs no extra hardware.
- **Per-machine-ID models** — one autoencoder per unit; a device ships only its own model.
- **int8 full-integer quantization** — required for TFLite-Micro; measured lossless vs the
  float model (per-ID macro AUC 0.768 either way).
- **FreeRTOS task pipeline** (rx → infer → tx) with a byte-sum checksum guarding the binary
  UART protocol (it caught the console's CR↔LF translation during bring-up).

## What this project adds over library-desk-sense

| | library-desk-sense | MachineSense |
|---|---|---|
| Intelligence | server-side analytics | **on-device** (edge AI) |
| Signal processing | low-rate sensing | **log-mel / DSP features** |
| Concurrency | basic tasks | **FreeRTOS rx→infer→tx pipeline** |
| Quantization | none | **int8 TFLite-Micro (lossless vs float)** |
| Testing / CI | none | **dataset-free tests + GitHub Actions** |

## Results

| Metric | Value |
|---|---|
| Per-ID macro AUC (float32) | 0.768 |
| Per-ID macro AUC (int8) | 0.768 |
| Deployed `id_02` AUC (int8, host) | 0.858 |
| `id_02` anomaly F1 @ threshold | 0.81 (precision 0.85 / recall 0.76) |
| int8 model size (`id_02`) | ~311 KiB |
| Tensor-arena RAM used | ~15.8 KiB (24 KiB allocated) |
| Inference latency (ESP32 @ 240 MHz) | 49.2 ms/vector (100-iter mean; ~52 µs spread) |
| Real-time margin | ~309 vectors per 10 s clip → ~15 s compute (~1.5× slower than real time) |
| On-device vs host agreement | per-vector scores within ~0.001; 0 flag mismatches |

Latency is measured on-device at boot rather than inferred from replay throughput —
at 115200 baud the 2560-byte request takes ~222 ms, so UART transfer would otherwise
mask the compute entirely. It is also compiler-independent: `-Og` → `-O2` moved it only
49.3 → 49.2 ms. The cost is structural — a dense-only model on a classic ESP32 (no SIMD),
where `esp-nn`'s optimised kernels mainly accelerate convolution, not fully-connected.
