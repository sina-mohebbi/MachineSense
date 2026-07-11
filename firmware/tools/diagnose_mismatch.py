"""Diagnostic: compare per-vector scores between LoopbackDevice (Python) and the
real ESP32, for the exact same vectors -- isolates whether the C++ inference
math diverges from the verified-correct Python math, independent of AUC/aggregation.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from replay_client import LoopbackDevice, SerialDevice, build_normalized_test_files
import config


def main():
    machine_id = "id_02"
    files = build_normalized_test_files(machine_id, limit_files=2)
    label, vectors = files[0]
    print(f"[data] file label={label}, {len(vectors)} vectors; comparing first 5")

    tflite_path = config.PER_ID_ARTIFACTS / machine_id / "model_int8.tflite"
    mock = LoopbackDevice(tflite_path)
    real = SerialDevice("COM7")

    import struct
    v = vectors[0]
    payload = v.astype("<f4").tobytes()
    print(f"payload length = {len(payload)} bytes")
    n = real._serial.write(payload)
    print(f"serial.write() reported {n} bytes written (expected {len(payload)})")
    response = real._serial.read(8)
    print(f"response length = {len(response)} bytes")
    if len(response) == 8:
        score, checksum = struct.unpack("<fI", response)
        expected = sum(payload) & 0xFFFFFFFF
        print(f"score={score}  device_checksum={checksum:#010x}  "
              f"host_checksum={expected:#010x}")


if __name__ == "__main__":
    main()
