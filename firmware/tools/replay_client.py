"""Phase 1 replay client: stream held-out MIMII test vectors to the ESP32 over
UART and reconstruct an on-device AUC, with no sensor attached.

Protocol (must match firmware/main/main.cc):
    host  -> device : 640 float32 values, little-endian (2560 bytes -- one
                       already-normalized log-mel vector)
    device -> host  : 4 bytes             (float32 LE reconstruction-error score)

The device does its own quantization from the float32 vector (matching
ml/evaluate_per_id_tflite.py's methodology exactly -- see inference.h). An
earlier version had the host pre-quantize to int8 before sending, which
quantized the vector twice (once for the wire, once implicitly by comparing
two independently-dequantized values) and measurably degraded on-device AUC
(0.858 host int8 AUC vs ~0.58 on-device on real hardware) -- this version
fixed that.

Usage:
    python replay_client.py --port COM5 --machine-id id_02
    python replay_client.py --mock --machine-id id_02       # no hardware needed,
                                                              # runs the "device"
                                                              # step in Python
"""
from __future__ import annotations

import argparse
import json
import struct
import sys
from pathlib import Path
from typing import Sequence

import numpy as np
from sklearn.metrics import roc_auc_score

# ml/ is a sibling of firmware/; reuse its data loading + normalization exactly
# so the vectors sent over the wire match what evaluate_per_id_tflite.py used.
ML_DIR = Path(__file__).resolve().parents[2] / "ml"
sys.path.insert(0, str(ML_DIR))

import config       # noqa: E402
import data         # noqa: E402
from evaluate_tflite import Int8Autoencoder  # noqa: E402
from train import normalize                  # noqa: E402
from train_per_id import group_by_id  # noqa: E402

READY_MARKER = "MACHINESENSE_READY"


class Device:
    """Something that answers one normalized float32 vector with one float32 score."""

    def query(self, vector: np.ndarray) -> float:
        raise NotImplementedError


class SerialDevice(Device):
    """Talks to the real ESP32 over UART."""

    def __init__(self, port: str, baud: int = 115200, timeout: float = 5.0,
                 ready_timeout: float = 15.0):
        import serial  # local import: only required for real-hardware runs
        import time

        # readline()'s per-call timeout governs each read attempt; boot has quiet
        # gaps between ROM bootloader / 2nd-stage / app_main output that can each
        # exceed it without meaning the device is stuck -- ready_timeout below
        # bounds the whole wait, not any single read.
        self._serial = serial.Serial(port, baud, timeout=timeout)

        # pyserial's default DTR/RTS state on open() is not a reliable reset on
        # every USB-serial chipset, especially on rapid reconnects (observed:
        # relying on it produced an intermittent boot-time panic). Do the same
        # explicit, controlled EN-pin reset esptool/idf_monitor use instead.
        self._serial.dtr = False
        self._serial.rts = True
        time.sleep(0.1)
        self._serial.rts = False
        time.sleep(0.1)

        self._wait_for_ready(ready_timeout)

        # Let the device finish booting into its binary loop, then drop any
        # boot-log bytes still buffered from the readline()-based wait above so
        # the first raw read(8) starts on a clean frame boundary.
        time.sleep(0.2)
        self._serial.reset_input_buffer()

    def _wait_for_ready(self, ready_timeout: float):
        import time

        deadline = time.time() + ready_timeout
        while time.time() < deadline:
            line = self._serial.readline().decode(errors="replace").strip()
            if not line:
                continue  # a quiet gap between boot-log lines, not a failure
            print(f"[device] {line}")
            if line.startswith(READY_MARKER):
                return
        raise TimeoutError(
            f"Timed out after {ready_timeout}s waiting for {READY_MARKER} "
            "-- is the board freshly flashed/reset and on the right port?"
        )

    def query(self, vector: np.ndarray, max_attempts: int = 3) -> float:
        # The protocol is strictly synchronous (device sends exactly one 8-byte
        # reply -- 4-byte score + 4-byte byte-sum checksum -- per 2560-byte
        # request, in order). The device's checksum lets us verify the request
        # arrived byte-exact; on a mismatch (or a wrong-length / non-finite
        # reply) we flush the host read buffer and resend the SAME vector,
        # which is safe because the device is already idle awaiting its next
        # request and the score is deterministic.
        #
        # This check earned its keep: it caught the ESP-IDF console UART
        # silently CR<->LF-translating the binary stream (fixed device-side in
        # main.cc via uart_vfs_dev_port_set_*_line_endings). With that fixed
        # the checksum should now match on every request; it stays as a guard
        # against genuine transient bit errors over long runs.
        payload = vector.astype("<f4").tobytes()
        expected_checksum = sum(payload) & 0xFFFFFFFF
        last_error = None
        for attempt in range(max_attempts):
            if attempt > 0:
                self._serial.reset_input_buffer()
            self._serial.write(payload)
            response = self._serial.read(8)
            if len(response) != 8:
                last_error = f"expected 8 response bytes, got {len(response)}"
                continue
            score, checksum = struct.unpack("<fI", response)
            if checksum != expected_checksum:
                last_error = (f"checksum mismatch (device received corrupted "
                              f"request: got {checksum:#010x}, expected "
                              f"{expected_checksum:#010x})")
                continue
            if not np.isfinite(score):
                last_error = f"non-finite score decoded ({score!r}) -- likely a bit error"
                continue
            return score
        raise IOError(f"query failed after {max_attempts} attempts: {last_error}")


class LoopbackDevice(Device):
    """Runs the exact same math as firmware/main/inference.cc, but in Python.

    Lets you validate this script (and the on-device scoring methodology) with
    zero hardware. Delegates to Int8Autoencoder.reconstruct(), which quantizes
    once, invokes, and dequantizes the output -- identical to inference.cc's
    RunOnFloatVector -- so the MSE computed here (true float input vs
    dequantized output) matches evaluate_per_id_tflite.py's file_score exactly.
    """

    def __init__(self, tflite_path):
        self._model = Int8Autoencoder(tflite_path)

    def query(self, vector: np.ndarray) -> float:
        reconstruction = self._model.reconstruct(vector[None, :])[0]
        return float(np.mean((vector - reconstruction) ** 2))


def score_files(device: Device, files: Sequence[tuple], progress: bool = False) -> tuple[float, list, list]:
    """Score (label, vectors) tuples via `device`; return (auc, scores, labels).

    Pure aggregation logic, independent of how `files` was built -- lets tests
    exercise this with synthetic data and a fake Device, no dataset required.
    One UART round-trip per vector is slow (~60-100ms incl. USB-serial/Python
    overhead), so a real board run can take minutes to hours; `progress=True`
    prints per-file timing and an ETA so a long run doesn't look hung.
    """
    import time

    scores, labels = [], []
    start = time.time()
    total_vectors = sum(len(v) for _, v in files)
    vectors_done = 0

    for i, (label, vectors) in enumerate(files):
        if len(vectors) == 0:
            continue
        per_vector = [device.query(v) for v in vectors]
        scores.append(float(np.mean(per_vector)))  # == file-level MSE, see README
        labels.append(label)

        if progress:
            vectors_done += len(vectors)
            elapsed = time.time() - start
            rate = vectors_done / elapsed if elapsed > 0 else 0
            remaining = (total_vectors - vectors_done) / rate if rate > 0 else float("inf")
            print(f"[progress] file {i + 1}/{len(files)}  "
                  f"vectors {vectors_done}/{total_vectors}  "
                  f"({rate:.1f} vec/s, ~{remaining / 60:.1f} min left)")

    auc = float(roc_auc_score(labels, scores)) if len(set(labels)) == 2 else float("nan")
    return auc, scores, labels


def build_normalized_test_files(machine_id: str, limit_files=None, seed: int = 0):
    """Load + normalize (but do NOT quantize) this machine's held-out test clips.

    The device quantizes internally, so vectors are sent as float32.

    limit_files samples a BALANCED, shuffled mix of normal/anomalous clips (not a
    raw truncation) so small values still produce a meaningful AUC -- test_files
    are ordered normal-then-abnormal per ID, so limit_files[:N] alone would grab
    only normal clips and AUC would be undefined.
    """
    stats = np.load(config.PER_ID_ARTIFACTS / machine_id / "normalization.npz")
    _, test_files = data.build_dataset()
    id_test_files = group_by_id(test_files).get(machine_id, [])

    if limit_files is not None:
        rng = np.random.default_rng(seed)
        normal = [f for f in id_test_files if f[2] == 0]
        anomalous = [f for f in id_test_files if f[2] == 1]
        rng.shuffle(normal)
        rng.shuffle(anomalous)
        per_class = max(1, limit_files // 2)
        id_test_files = normal[:per_class] + anomalous[:per_class]

    out = []
    for _, path, label in id_test_files:
        vectors = data.cached_file_to_vectors(path)
        vectors = normalize(vectors, stats["mean"], stats["std"])
        out.append((label, list(vectors)))
    return out


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--machine-id", default="id_02")
    parser.add_argument("--port", help="Serial port, e.g. COM5 (required unless --mock)")
    parser.add_argument("--baud", type=int, default=115200)
    parser.add_argument("--mock", action="store_true",
                         help="Score in Python instead of over serial (no hardware)")
    parser.add_argument("--limit-files", type=int, default=None,
                         help="Replay a balanced sample of ~this many clips "
                              "(half normal, half anomalous) instead of the full set")
    return parser.parse_args()


def main():
    args = parse_args()
    model_dir = config.PER_ID_ARTIFACTS / args.machine_id
    tflite_path = model_dir / "model_int8.tflite"
    if not tflite_path.exists():
        raise FileNotFoundError(f"{tflite_path} missing -- run export_per_id.py first")

    files = build_normalized_test_files(args.machine_id, args.limit_files)
    print(f"[data] {args.machine_id}: {len(files)} held-out test clips")

    if args.mock:
        device: Device = LoopbackDevice(tflite_path)
        print("[device] mock mode (Python loopback, no hardware)")
    else:
        if not args.port:
            raise SystemExit("--port is required unless --mock is passed")
        device = SerialDevice(args.port, args.baud)

    auc, scores, labels = score_files(device, files, progress=not args.mock)
    print(f"\n[result] {args.machine_id}: on-device AUC = {auc:.4f}  "
          f"({len(scores)} clips)")

    result = {"machine_id": args.machine_id, "on_device_auc": round(auc, 4),
              "clips": len(scores), "mock": args.mock}
    out_path = model_dir / "metrics_on_device.json"
    out_path.write_text(json.dumps(result, indent=2))
    print(f"[saved] {out_path}")


if __name__ == "__main__":
    main()
