#!/usr/bin/env python3
from pathlib import Path
import argparse


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--area-threshold", type=float, default=0.02)
    parser.add_argument("--extra-copies", type=int, default=2)
    args = parser.parse_args()

    dataset = args.dataset.resolve()
    images_dir = dataset / "train" / "images"
    labels_dir = dataset / "train" / "labels"
    images = sorted(p for p in images_dir.iterdir() if p.is_file())
    selected = []
    for image in images:
        label = labels_dir / f"{image.stem}.txt"
        if not label.exists():
            continue
        small_phone = False
        for line in label.read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if len(parts) >= 5 and parts[0] == "0":
                if float(parts[3]) * float(parts[4]) <= args.area_threshold:
                    small_phone = True
                    break
        if small_phone:
            selected.append(image)

    train_list = dataset / "train_small_phone_oversampled.txt"
    entries = images + selected * args.extra_copies
    train_list.write_text("\n".join(str(p) for p in entries) + "\n", encoding="utf-8")
    yaml_path = dataset / "data_small_phone.yaml"
    yaml_path.write_text(
        f"path: {dataset}\n"
        f"train: {train_list}\n"
        "val: val/images\n"
        "test: test/images\n"
        "names:\n  0: phone\n  1: face\n",
        encoding="utf-8",
    )
    print({
        "base_images": len(images),
        "small_phone_images": len(selected),
        "extra_copies": args.extra_copies,
        "effective_train_entries": len(entries),
        "data_yaml": str(yaml_path),
    })


if __name__ == "__main__":
    main()
