import argparse
import shutil
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np


def binarize_mask(mask: np.ndarray) -> np.ndarray:
    """
    Convert a grayscale or RGB mask to a binary mask (uint8 {0,255}).
    Inputs:
        mask: HxW or HxWxC numpy array, mask where pothole pixels are non-zero.
    Returns:
        bin_mask: HxW uint8 array with values {0,255}.
    """
    if mask.ndim == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    # Normalize possible {0,1} to {0,255}
    if mask.max() <= 1:
        mask = (mask * 255).astype(np.uint8)
    # Otsu threshold to robustly binarize varied masks
    _, bin_mask = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    return bin_mask


def extract_bounding_boxes_from_mask(mask: np.ndarray, min_area: int = 50) -> List[Tuple[int, int, int, int]]:
    """
    Extract axis-aligned bounding boxes from a binary mask using connected components.
    Inputs:
        mask: HxW uint8 binary mask (non-zero is foreground).
        min_area: Minimum connected-component area to keep (in pixels).
    Returns:
        boxes: list of (x, y, w, h) integer tuples.
    """
    bin_mask = binarize_mask(mask)
    # Ensure uint8 mask for OpenCV (bool not supported by connectedComponentsWithStats)
    cc_input = (bin_mask > 0).astype(np.uint8)
    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(cc_input, connectivity=8)
    boxes: List[Tuple[int, int, int, int]] = []
    for label_idx in range(1, num_labels):
        x, y, w, h, area = stats[label_idx]
        if area >= min_area and w > 0 and h > 0:
            boxes.append((int(x), int(y), int(w), int(h)))
    return boxes


def rects_intersect(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> bool:
    """
    Check if two rectangles (x,y,w,h) intersect with positive area.
    """
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    a_right, a_bottom = ax + aw, ay + ah
    b_right, b_bottom = bx + bw, by + bh
    inter_w = min(a_right, b_right) - max(ax, bx)
    inter_h = min(a_bottom, b_bottom) - max(ay, by)
    return inter_w > 0 and inter_h > 0


def union_rect(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> Tuple[int, int, int, int]:
    """
    Return the union rectangle that tightly bounds both input rectangles.
    """
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x1 = min(ax, bx)
    y1 = min(ay, by)
    x2 = max(ax + aw, bx + bw)
    y2 = max(ay + ah, by + bh)
    return (x1, y1, x2 - x1, y2 - y1)


def merge_overlapping_boxes(boxes: List[Tuple[int, int, int, int]]) -> List[Tuple[int, int, int, int]]:
    """
    Iteratively merge boxes that intersect (any positive-area overlap) until stable.
    Inputs:
        boxes: list of (x, y, w, h)
    Returns:
        merged: list of merged (x, y, w, h)
    """
    if not boxes:
        return []
    boxes = boxes.copy()
    merged = True
    while merged:
        merged = False
        new_boxes: List[Tuple[int, int, int, int]] = []
        used = [False] * len(boxes)
        for i in range(len(boxes)):
            if used[i]:
                continue
            current = boxes[i]
            for j in range(i + 1, len(boxes)):
                if used[j]:
                    continue
                if rects_intersect(current, boxes[j]):
                    current = union_rect(current, boxes[j])
                    used[j] = True
                    merged = True
            used[i] = True
            new_boxes.append(current)
        boxes = new_boxes
    return boxes


def write_yolo_label_file(label_path: Path, boxes: List[Tuple[int, int, int, int]], img_w: int, img_h: int) -> None:
    """
    Write YOLO-format labels for one image.
    Inputs:
        label_path: path to save .txt
        boxes: list of (x, y, w, h) in pixel units
        img_w, img_h: image dimensions in pixels
    Output:
        Writes a text file with normalized YOLO boxes for class 0.
    """
    label_path.parent.mkdir(parents=True, exist_ok=True)
    with open(label_path, "w") as f:
        for (x, y, w, h) in boxes:
            xc = (x + w / 2.0) / float(img_w)
            yc = (y + h / 2.0) / float(img_h)
            ww = w / float(img_w)
            hh = h / float(img_h)
            f.write(f"0 {xc:.6f} {yc:.6f} {ww:.6f} {hh:.6f}\n")


def convert_split(
    split_src_images: Path,
    split_src_masks: Path,
    split_dst_images: Path,
    split_dst_labels: Path,
    min_area: int,
) -> Tuple[int, int]:
    """
    Convert one dataset split from mask to YOLO labels and copy images.
    Inputs:
        split_src_images: source images directory
        split_src_masks: source masks directory
        split_dst_images: destination images directory
        split_dst_labels: destination labels directory
        min_area: minimum connected component area to keep
    Returns:
        (num_images_processed, num_with_boxes)
    """
    split_dst_images.mkdir(parents=True, exist_ok=True)
    split_dst_labels.mkdir(parents=True, exist_ok=True)

    processed = 0
    with_boxes = 0
    for img_path in sorted(split_src_images.glob("*.png")):
        mask_path = split_src_masks / img_path.name
        if not mask_path.exists():
            print(f"[WARN] Missing mask for {img_path.name}")
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            print(f"[WARN] Failed to read image {img_path}")
            continue
        mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
        if mask is None:
            print(f"[WARN] Failed to read mask {mask_path}")
            continue

        h, w = img.shape[:2]
        boxes = extract_bounding_boxes_from_mask(mask, min_area=min_area)
        boxes = merge_overlapping_boxes(boxes)

        # Copy image
        shutil.copy2(str(img_path), str(split_dst_images / img_path.name))
        # Write label (possibly empty file if no boxes)
        label_file = split_dst_labels / f"{img_path.stem}.txt"
        write_yolo_label_file(label_file, boxes, w, h)

        processed += 1
        if len(boxes) > 0:
            with_boxes += 1

    return processed, with_boxes


def write_dataset_yaml(dst_root: Path) -> None:
    """
    Write dataset YAML compatible with YOLO training.
    Inputs:
        dst_root: dataset root (contains images/ and labels/ subdirs)
    """
    yaml_text = (
        f"path: {dst_root.resolve()}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"test: images/test\n"
        f"\n"
        f"nc: 1\n"
        f"names:\n"
        f"  0: pothole\n"
    )
    (dst_root / "pothole.yaml").write_text(yaml_text)


def main() -> None:
    """
    CLI entry to convert pothole600 masks into YOLO detection dataset.
    """
    parser = argparse.ArgumentParser(description="Convert pothole600 masks to YOLOv3 detection labels.")
    parser.add_argument(
        "--src",
        type=str,
        default="/Users/pjiang/Documents/Projects/pothole600",
        help="Path to pothole600 dataset root (contains training/ validation/ testing/).",
    )
    parser.add_argument(
        "--dst",
        type=str,
        default="/Users/pjiang/Documents/Projects/Pothole-Detection/dataset",
        help="Output YOLO dataset directory.",
    )
    parser.add_argument(
        "--min-area",
        type=int,
        default=50,
        help="Minimum connected component area to keep as a box.",
    )
    args = parser.parse_args()

    src_root = Path(args.src)
    dst_root = Path(args.dst)

    # Map pothole600 splits to YOLO splits
    split_map = {
        "training": "train",
        "validation": "val",
        "testing": "test",
    }

    total_processed = 0
    total_with_boxes = 0
    for src_split, dst_split in split_map.items():
        src_images = src_root / src_split / "images"
        src_masks = src_root / src_split / "masks"
        dst_images = dst_root / "images" / dst_split
        dst_labels = dst_root / "labels" / dst_split

        if not src_images.exists() or not src_masks.exists():
            print(f"[WARN] Skipping split '{src_split}' due to missing directories.")
            continue

        processed, with_boxes = convert_split(
            src_images, src_masks, dst_images, dst_labels, min_area=args.min_area
        )
        print(f"[INFO] {src_split} → {dst_split}: processed={processed}, with_boxes={with_boxes}")
        total_processed += processed
        total_with_boxes += with_boxes

    write_dataset_yaml(dst_root)
    print(f"[DONE] Wrote dataset yaml to {dst_root / 'pothole.yaml'}")
    print(f"[SUMMARY] images processed={total_processed}, images with boxes={total_with_boxes}")


if __name__ == "__main__":
    main()


