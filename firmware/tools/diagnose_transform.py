"""Find the FIRST vector whose device checksum mismatches, then brute-force what
byte transformation the device applied (to identify remaining UART corruption)."""
import struct
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from replay_client import SerialDevice, build_normalized_test_files  # noqa: E402


def main():
    files = build_normalized_test_files("id_02", limit_files=12)
    dev = SerialDevice("COM7")
    ser = dev._serial

    for fi, (label, vectors) in enumerate(files):
        for vi, v in enumerate(vectors):
            payload = v.astype("<f4").tobytes()
            host = sum(payload) & 0xFFFFFFFF
            ser.reset_input_buffer()
            ser.write(payload)
            resp = ser.read(8)
            if len(resp) != 8:
                print(f"file {fi} vec {vi}: short response {len(resp)}")
                return
            score, dev_cksum = struct.unpack("<fI", resp)
            if dev_cksum == host:
                continue

            target = (host - dev_cksum) & 0xFFFFFFFF
            signed = host - dev_cksum
            print(f"MISMATCH file {fi} vec {vi}: host={host:#010x} "
                  f"dev={dev_cksum:#010x} host-dev={signed}")
            print(f"  byte counts: 0x0D={payload.count(0x0D)} 0x0A={payload.count(0x0A)} "
                  f"0x11={payload.count(0x11)} 0x13={payload.count(0x13)} 0x00={payload.count(0)}")

            def trans(fn):
                return sum(fn(b) for b in payload) & 0xFFFFFFFF

            # brute-force single-byte substitution X->Y explaining the diff
            counts = Counter(payload)
            found = False
            for x, cnt in counts.items():
                if cnt == 0 or signed % cnt != 0:
                    continue
                delta = signed // cnt
                y = x - delta
                if 0 <= y <= 255 and y != x:
                    if trans(lambda b, x=x, y=y: y if b == x else b) == dev_cksum:
                        print(f"  CANDIDATE substitution: 0x{x:02x} -> 0x{y:02x} "
                              f"(occurs {cnt}x, delta {delta})")
                        found = True
            if not found:
                print("  no single-byte substitution explains it "
                      "(may be a drop/insert or multi-byte effect)")
            return

    print("no mismatch found across all 12 files' vectors!")


if __name__ == "__main__":
    main()
