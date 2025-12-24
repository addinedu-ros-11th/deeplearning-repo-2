import argparse
import json
import socket
import struct
import time

import cv2
import numpy as np

HEADER_FORMAT = "!IHH"
HEADER_SIZE = struct.calcsize(HEADER_FORMAT)


def load_model(task, model_path):
    try:
        from ultralytics import YOLO
    except Exception:
        return None

    if model_path:
        return YOLO(model_path)

    if task == "pose":
        return YOLO("yolov8n-pose.pt")
    return YOLO("yolov8n.pt")


def run_inference(model, frame):
    if model is None:
        return None
    results = model(frame, verbose=False)
    if not results:
        return None
    return results[0]


def parse_target(value):
    host, port_str = value.rsplit(":", 1)
    return host, int(port_str)


def parse_class_filter(value):
    if not value:
        return None
    return {name.strip() for name in value.split(",") if name.strip()}


def resolve_class_name(names, class_id):
    if isinstance(names, dict):
        return names.get(class_id, str(class_id))
    if isinstance(names, (list, tuple)) and 0 <= class_id < len(names):
        return names[class_id]
    return str(class_id)


def build_events(result, task, threshold, allowed_classes, include_keypoints):
    if result is None:
        return []

    names = getattr(result, "names", {})
    events = []

    boxes = getattr(result, "boxes", None)
    if boxes is not None and len(boxes) > 0:
        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        clses = boxes.cls.cpu().numpy().astype(int)
        for box, score, class_id in zip(xyxy, confs, clses):
            if score < threshold:
                continue
            label = resolve_class_name(names, int(class_id))
            if allowed_classes and label not in allowed_classes:
                continue
            events.append({
                "label": label,
                "score": float(score),
                "bbox": [float(v) for v in box],
            })

    if task == "pose" and include_keypoints:
        keypoints = getattr(result, "keypoints", None)
        if keypoints is not None and len(keypoints) > 0:
            data = keypoints.data.cpu().numpy()
            for idx, points in enumerate(data):
                if idx < len(events):
                    events[idx]["keypoints"] = points.tolist()
                else:
                    events.append({
                        "label": "person",
                        "score": None,
                        "bbox": None,
                        "keypoints": points.tolist(),
                    })

    return events


def main():
    parser = argparse.ArgumentParser(description="UDP webcam receiver (JPEG frames)")
    parser.add_argument("--bind", default="0.0.0.0")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--task", choices=["detect", "pose"], default="detect")
    parser.add_argument("--model", default="", help="YOLO model path (optional)")
    parser.add_argument("--frame-timeout", type=float, default=0.5)
    parser.add_argument("--report-target", default="", help="host:port for event reporting")
    parser.add_argument("--report-threshold", type=float, default=0.5)
    parser.add_argument("--report-classes", default="", help="Comma-separated class names")
    parser.add_argument("--report-include-keypoints", action="store_true")
    parser.add_argument("--source-id", default="", help="Source identifier")
    parser.add_argument("--show", action="store_true", help="Display frames")
    args = parser.parse_args()

    model = load_model(args.task, args.model)
    report_target = parse_target(args.report_target) if args.report_target else None
    report_classes = parse_class_filter(args.report_classes)
    report_sock = None
    if report_target:
        report_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1 << 20)
    sock.bind((args.bind, args.port))

    frames = {}
    last_cleanup = time.monotonic()

    try:
        while True:
            packet, _addr = sock.recvfrom(65535)
            if len(packet) <= HEADER_SIZE:
                continue

            header = packet[:HEADER_SIZE]
            payload = packet[HEADER_SIZE:]
            frame_id, chunk_id, total_chunks = struct.unpack(HEADER_FORMAT, header)

            entry = frames.get(frame_id)
            if entry is None:
                entry = {"total": total_chunks, "chunks": {}, "time": time.monotonic()}
                frames[frame_id] = entry

            entry["chunks"][chunk_id] = payload
            if len(entry["chunks"]) == entry["total"]:
                data = b"".join(entry["chunks"][i] for i in range(entry["total"]))
                del frames[frame_id]

                np_buf = np.frombuffer(data, dtype=np.uint8)
                frame = cv2.imdecode(np_buf, cv2.IMREAD_COLOR)
                if frame is None:
                    continue

                result = run_inference(model, frame)
                if report_sock:
                    events = build_events(
                        result,
                        args.task,
                        args.report_threshold,
                        report_classes,
                        args.report_include_keypoints,
                    )
                    if events:
                        payload = {
                            "task": args.task,
                            "source_id": args.source_id,
                            "timestamp": time.time(),
                            "events": events,
                        }
                        report_sock.sendto(
                            json.dumps(payload, ensure_ascii=True).encode("utf-8"),
                            report_target,
                        )

                if args.show:
                    cv2.imshow(f"udp-{args.task}", frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break

            now = time.monotonic()
            if now - last_cleanup > 0.2:
                stale = [
                    key
                    for key, value in frames.items()
                    if now - value["time"] > args.frame_timeout
                ]
                for key in stale:
                    del frames[key]
                last_cleanup = now
    finally:
        sock.close()
        if report_sock:
            report_sock.close()
        if args.show:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
