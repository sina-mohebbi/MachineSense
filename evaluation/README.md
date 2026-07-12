# evaluation/ — Phase 5: the benchmark study

The rigorous comparison that turns a demo into a portfolio piece (in the spirit of the
HTTP-vs-CoAP benchmark in library-desk-sense). Planned experiments:

| Study | Question | Metrics |
|---|---|---|
| **int8 vs float32** | What does quantization cost? | AUC, model size, tensor-arena RAM, latency |
| **on-device vs host AUC** | Does the ESP32 match the PC? | AUC delta |
| **score vs raw-audio uplink** | Why compute at the edge? | bytes-on-wire (send a score vs stream raw audio) |
| **autoencoder vs classifier** | Unsupervised vs supervised trade-off | AUC, footprint |

Outputs: a results table for the top-level README and plots (`anomaly-score`, ROC).
