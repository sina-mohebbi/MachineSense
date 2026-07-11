"""Send vectors until the device stops replying cleanly, then dump the raw serial
text so we can read the actual panic/backtrace message."""
import struct
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from replay_client import SerialDevice, build_normalized_test_files  # noqa: E402


def main():
    files = build_normalized_test_files("id_02", limit_files=12)
    _, vectors = files[0]
    dev = SerialDevice("COM7")
    ser = dev._serial

    for vi, v in enumerate(vectors):
        payload = v.astype("<f4").tobytes()
        host = sum(payload) & 0xFFFFFFFF
        ser.reset_input_buffer()
        ser.write(payload)
        resp = ser.read(8)
        if len(resp) == 8:
            _, cksum = struct.unpack("<fI", resp)
            if cksum == host:
                continue
        print(f"[vector {vi}] clean reply stopped. Dumping raw serial:")
        ser.timeout = 2
        raw = resp + ser.read(4000)
        text = "".join(chr(b) if 32 <= b < 127 or b in (10, 13) else "." for b in raw)
        print(text)
        return


if __name__ == "__main__":
    main()
