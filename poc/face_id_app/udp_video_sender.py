import argparse
import socket
import struct
import time

import cv2

HEADER_FORMAT = "!IHH"
HEADER_SIZE = struct.calcsize(HEADER_FORMAT)


def parse_targets(value):
    targets = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        host, port_str = item.rsplit(":", 1)
        targets.append((host, int(port_str)))
    return targets


def main():
    parser = argparse.ArgumentParser(description="UDP webcam sender (JPEG frames)")
    parser.add_argument("--targets", required=True, help="Comma-separated host:port list")
    parser.add_argument("--source", default="0", help="Camera index or video path")
    parser.add_argument("--jpeg-quality", type=int, default=80)
    parser.add_argument("--max-datagram", type=int, default=1400)
    parser.add_argument("--fps", type=float, default=0.0, help="0 = no limit")
    args = parser.parse_args()

    targets = parse_targets(args.targets)
    if not targets:
        raise SystemExit("No valid targets provided.")

    try:
        source = int(args.source)
    except ValueError:
        source = args.source

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise SystemExit(f"Failed to open source: {args.source}")

    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 1 << 20)

    max_payload = max(1, args.max_datagram - HEADER_SIZE)
    frame_id = 0
    frame_interval = 1.0 / args.fps if args.fps > 0 else 0.0
    next_frame_time = time.monotonic()

    try:
        while True:
            if frame_interval:
                now = time.monotonic()
                if now < next_frame_time:
                    time.sleep(next_frame_time - now)
                next_frame_time = time.monotonic() + frame_interval

            ok, frame = cap.read()
            if not ok:
                break

            encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), args.jpeg_quality]
            ok, encoded = cv2.imencode(".jpg", frame, encode_params)
            if not ok:
                continue

            data = encoded.tobytes()
            total_chunks = (len(data) + max_payload - 1) // max_payload
            for chunk_id in range(total_chunks):
                start = chunk_id * max_payload
                end = start + max_payload
                header = struct.pack(HEADER_FORMAT, frame_id, chunk_id, total_chunks)
                packet = header + data[start:end]
                for target in targets:
                    sock.sendto(packet, target)

            frame_id = (frame_id + 1) & 0xFFFFFFFF
    finally:
        cap.release()
        sock.close()


if __name__ == "__main__":
    main()
