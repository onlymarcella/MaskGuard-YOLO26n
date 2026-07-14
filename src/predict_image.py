from __future__ import annotations

import argparse
from pathlib import Path
import sys

from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prediksi status penggunaan masker pada gambar atau video."
    )
    parser.add_argument(
        "--weights",
        type=Path,
        default=Path("models/best.pt"),
        help="Lokasi bobot YOLO hasil training.",
    )
    parser.add_argument(
        "--source",
        type=str,
        required=True,
        help="Lokasi gambar/video atau URL sumber.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs"),
        help="Folder penyimpanan hasil.",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Confidence threshold antara 0 dan 1.",
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

    if not 0.0 <= args.conf <= 1.0:
        print("ERROR: --conf harus berada di antara 0 dan 1.", file=sys.stderr)
        return 1

    args.output.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(args.weights))
    results = model.predict(
        source=args.source,
        conf=args.conf,
        imgsz=args.imgsz,
        save=True,
        project=str(args.output),
        name="predict",
        exist_ok=True,
    )

    detection_count = sum(
        len(result.boxes) if result.boxes is not None else 0
        for result in results
    )

    print(f"Prediksi selesai. Jumlah deteksi: {detection_count}")
    print(f"Hasil tersimpan di: {args.output / 'predict'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
