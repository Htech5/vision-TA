"""
Build human+vehicle dataset from COCO (segmentation labels) into one YOLO dataset + data.yaml.

COCO already contains both classes we want (person + all vehicle types) with
segmentation polygons, so we filter/remap it instead of downloading and
merging two separate datasets by hand.

Sources:
    coco128  - 128 images, ~7MB. Smoke-test only, too small to be accurate.
    coco-val - COCO2017 val2017 split: 5000 real images, ~1GB. Default: real
               COCO quality without the 19GB train2017.zip. Split 90/10 here
               into our own train/val.
    coco     - full COCO2017, ~20GB (train2017.zip + val2017.zip). Most data,
               most accurate, biggest download.

Usage:
    python prepare_dataset.py                 # coco-val, ~1GB, default
    python prepare_dataset.py --source coco128
    python prepare_dataset.py --source coco

Output:
    datasets/human_vehicle/{images,labels}/{train,val}/...
    data.yaml
"""
import argparse
import random
import shutil
from pathlib import Path

from ultralytics.data.utils import check_det_dataset
from ultralytics.utils import ASSETS_URL
from ultralytics.utils.downloads import download

# COCO class id -> our class id, for the classes we keep.
TARGET_CLASSES = {
    0: 0,  # person
    1: 1,  # bicycle
    2: 2,  # car
    3: 3,  # motorcycle
    5: 4,  # bus
    7: 5,  # truck
}
NAMES = ["person", "bicycle", "car", "motorcycle", "bus", "truck"]

OUT_ROOT = Path("datasets/human_vehicle")
VAL_SPLIT_FRACTION = 0.1  # for coco-val, which only has one raw split to work with


def filter_pairs(images_dir: Path, labels_dir: Path) -> list[tuple[Path, Path]]:
    """Return (image, label) pairs that contain at least one target class."""
    kept = []
    for label_file in labels_dir.glob("*.txt"):
        lines_out = []
        for line in label_file.read_text().splitlines():
            parts = line.split()
            if not parts:
                continue
            cls = int(parts[0])
            if cls not in TARGET_CLASSES:
                continue
            parts[0] = str(TARGET_CLASSES[cls])
            lines_out.append(" ".join(parts))
        if not lines_out:
            continue  # skip background-only images, keeps dataset lean
        img_file = images_dir / (label_file.stem + ".jpg")
        if img_file.exists():
            kept.append((img_file, "\n".join(lines_out) + "\n"))
    return kept


def write_split(pairs, out_images: Path, out_labels: Path):
    out_images.mkdir(parents=True, exist_ok=True)
    out_labels.mkdir(parents=True, exist_ok=True)
    for img_file, label_text in pairs:
        (out_labels / (img_file.stem + ".txt")).write_text(label_text)
        try:
            (out_images / img_file.name).symlink_to(img_file.resolve())
        except OSError:
            shutil.copy(img_file, out_images / img_file.name)  # no symlink perms (Windows)


def prep_from_ultralytics_yaml(yaml_name: str) -> int:
    """coco128 / full coco: already has train+val splits, filter each as-is."""
    data = check_det_dataset(yaml_name)
    root = Path(data["path"])
    total = 0
    for split, rel in {"train": data.get("train"), "val": data.get("val")}.items():
        if rel is None:
            continue
        images_dir = (root / rel) if not str(rel).startswith(str(root)) else Path(rel)
        labels_dir = Path(str(images_dir).replace("images", "labels"))
        pairs = filter_pairs(images_dir, labels_dir)
        write_split(pairs, OUT_ROOT / "images" / split, OUT_ROOT / "labels" / split)
        print(f"{split}: kept {len(pairs)} images")
        total += len(pairs)
    return total


def prep_from_coco_val() -> int:
    """val2017 only (~1GB): real COCO images/labels, no 19GB train2017.zip. We split it ourselves."""
    raw = Path("datasets/_coco_val_raw")
    labels_dir = raw / "coco" / "labels" / "val2017"  # zip extracts under a 'coco/' subfolder
    if not labels_dir.exists():
        download([ASSETS_URL + "/coco2017labels-segments.zip"], dir=raw)
    if not (raw / "images" / "val2017").exists():
        download(["http://images.cocodataset.org/zips/val2017.zip"], dir=raw / "images", threads=1)

    pairs = filter_pairs(raw / "images" / "val2017", labels_dir)
    random.Random(0).shuffle(pairs)
    n_val = int(len(pairs) * VAL_SPLIT_FRACTION)
    val_pairs, train_pairs = pairs[:n_val], pairs[n_val:]

    write_split(train_pairs, OUT_ROOT / "images" / "train", OUT_ROOT / "labels" / "train")
    write_split(val_pairs, OUT_ROOT / "images" / "val", OUT_ROOT / "labels" / "val")
    print(f"train: kept {len(train_pairs)} images")
    print(f"val: kept {len(val_pairs)} images")
    return len(pairs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["coco-val", "coco128", "coco"], default="coco-val",
                     help="coco-val = ~1GB real COCO subset (default), coco128 = tiny smoke test, coco = full ~20GB")
    args = ap.parse_args()

    if args.source == "coco-val":
        total = prep_from_coco_val()
    else:
        yaml_name = "coco128-seg.yaml" if args.source == "coco128" else "coco.yaml"
        total = prep_from_ultralytics_yaml(yaml_name)

    yaml_text = (
        f"path: {OUT_ROOT.resolve().as_posix()}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"names:\n" + "\n".join(f"  {i}: {n}" for i, n in enumerate(NAMES)) + "\n"
    )
    Path("data.yaml").write_text(yaml_text)
    print(f"wrote data.yaml, {total} images total")


if __name__ == "__main__":
    main()
