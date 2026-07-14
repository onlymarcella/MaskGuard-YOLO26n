from __future__ import annotations

import argparse
from pathlib import Path
import sys

import cv2
from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Deteksi masker real-time dari webcam lokal."
    )
    parser.add_argument(
        "--weights",
        type=Path,
        default=Path("models/best.pt"),
        help="Lokasi bobot YOLO hasil training.",
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="Indeks kamera OpenCV.",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Confidence threshold.",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Ukuran input inference.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if not args.weights.exists():
        print(f"ERROR: Bobot tidak ditemukan: {args.weights}", file=sys.stderr)
        return 1

    camera = cv2.VideoCapture(args.camera)
    if not camera.isOpened():
        print(
            f"ERROR: Kamera dengan indeks {args.camera} tidak dapat dibuka.",
            file=sys.stderr,
        )
        return 1

    model = YOLO(str(args.weights))

    print("Webcam aktif. Tekan q untuk keluar.")

    try:
        while True:
            success, frame = camera.read()
            if not success:
                print("Frame tidak dapat dibaca.", file=sys.stderr)
                break

            results = model.predict(
                source=frame,
                conf=args.conf,
                imgsz=args.imgsz,
                verbose=False,
            )
            annotated_frame = results[0].plot()

            cv2.imshow("MaskGuard YOLO26n", annotated_frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        camera.release()
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
