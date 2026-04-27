"""Update COCO annotations after image resolution changes and visualize checks.

Example:
    python scripts/update_coco_resolution.py \
        --input-json data/eval/instances_default.json \
        --output-json data/eval/instances_default_2464x3248.json \
        --frame-dir data/eval/frame \
        --target-height 2464 \
        --target-width 3248 \
        --num-vis 16
"""

import argparse
import copy
import json
import math
import random
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Scale COCO annotations to a new resolution.")
    parser.add_argument(
        "--input-json",
        type=Path,
        default=Path("data/eval/instances_default.json"),
        help="Path to the original COCO annotation JSON.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("data/eval/instances_default_2464x3248.json"),
        help="Path for the updated COCO annotation JSON.",
    )
    parser.add_argument(
        "--frame-dir",
        type=Path,
        default=Path("data/eval/frame"),
        help="Directory of resized frames for visualization.",
    )
    parser.add_argument("--target-height", type=int, default=2464, help="Target image height.")
    parser.add_argument("--target-width", type=int, default=3248, help="Target image width.")
    parser.add_argument(
        "--num-vis",
        type=int,
        default=16,
        help="Number of annotation samples to visualize.",
    )
    parser.add_argument(
        "--vis-dir",
        type=Path,
        default=Path("data/eval/plots/coco_rescale_check"),
        help="Directory for saved visualization images.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used for visualization sampling.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output JSON if it already exists.",
    )
    return parser.parse_args()


def scale_bbox(bbox: List[float], sx: float, sy: float, target_w: int, target_h: int) -> List[float]:
    x, y, w, h = bbox
    x2 = x * sx
    y2 = y * sy
    w2 = w * sx
    h2 = h * sy

    x2 = max(0.0, min(x2, target_w - 1.0))
    y2 = max(0.0, min(y2, target_h - 1.0))
    w2 = max(0.0, min(w2, target_w - x2))
    h2 = max(0.0, min(h2, target_h - y2))
    return [x2, y2, w2, h2]


def scale_segmentation(segmentation, sx: float, sy: float):
    if isinstance(segmentation, list):
        scaled = []
        for poly in segmentation:
            if not isinstance(poly, list):
                scaled.append(poly)
                continue
            scaled_poly = []
            for i, value in enumerate(poly):
                scaled_poly.append(value * sx if i % 2 == 0 else value * sy)
            scaled.append(scaled_poly)
        return scaled
    return segmentation


def scale_keypoints(keypoints: List[float], sx: float, sy: float) -> List[float]:
    if len(keypoints) % 3 != 0:
        return keypoints
    scaled = []
    for i, value in enumerate(keypoints):
        mod = i % 3
        if mod == 0:
            scaled.append(value * sx)
        elif mod == 1:
            scaled.append(value * sy)
        else:
            scaled.append(value)
    return scaled


def load_json(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, content: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(content, f, ensure_ascii=False)


def update_coco_resolution(
    coco: Dict,
    target_h: int,
    target_w: int,
) -> Tuple[Dict, Dict[int, Tuple[float, float, int, int]]]:
    updated = copy.deepcopy(coco)

    image_scale: Dict[int, Tuple[float, float, int, int]] = {}
    for image in updated.get("images", []):
        image_id = image["id"]
        old_w = int(image["width"])
        old_h = int(image["height"])
        sx = target_w / old_w
        sy = target_h / old_h
        image_scale[image_id] = (sx, sy, old_h, old_w)
        image["width"] = target_w
        image["height"] = target_h

    for ann in updated.get("annotations", []):
        image_id = ann["image_id"]
        if image_id not in image_scale:
            continue

        sx, sy, _, _ = image_scale[image_id]

        if "bbox" in ann and isinstance(ann["bbox"], list) and len(ann["bbox"]) == 4:
            ann["bbox"] = scale_bbox(ann["bbox"], sx, sy, target_w, target_h)

        if "area" in ann and isinstance(ann["area"], (int, float)):
            ann["area"] = float(ann["area"]) * sx * sy

        if "segmentation" in ann:
            ann["segmentation"] = scale_segmentation(ann["segmentation"], sx, sy)

        if "keypoints" in ann and isinstance(ann["keypoints"], list):
            ann["keypoints"] = scale_keypoints(ann["keypoints"], sx, sy)

    return updated, image_scale


def draw_bbox(img, bbox, color, label=None, thickness=2):
    x, y, w, h = bbox
    p1 = (int(round(x)), int(round(y)))
    p2 = (int(round(x + w)), int(round(y + h)))
    cv2.rectangle(img, p1, p2, color, thickness)
    if label:
        cv2.putText(
            img,
            label,
            (p1[0], max(16, p1[1] - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
            lineType=cv2.LINE_AA,
        )


def save_contact_sheet(vis_images: List, vis_titles: List[str], out_path: Path, cols: int = 4) -> None:
    if not vis_images:
        return
    rows = math.ceil(len(vis_images) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 4))

    if rows == 1 and cols == 1:
        axes = [[axes]]
    elif rows == 1:
        axes = [axes]
    elif cols == 1:
        axes = [[a] for a in axes]

    idx = 0
    for r in range(rows):
        for c in range(cols):
            ax = axes[r][c]
            if idx < len(vis_images):
                ax.imshow(cv2.cvtColor(vis_images[idx], cv2.COLOR_BGR2RGB))
                ax.set_title(vis_titles[idx], fontsize=9)
                ax.axis("off")
            else:
                ax.axis("off")
            idx += 1

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def visualize_samples(
    coco_before: Dict,
    coco_after: Dict,
    image_scale: Dict[int, Tuple[float, float, int, int]],
    frame_dir: Path,
    vis_dir: Path,
    num_vis: int,
    seed: int,
) -> None:
    before_anns = {ann["id"]: ann for ann in coco_before.get("annotations", [])}
    after_anns = {ann["id"]: ann for ann in coco_after.get("annotations", [])}
    image_by_id = {img["id"]: img for img in coco_after.get("images", [])}

    common_ann_ids = [ann_id for ann_id in after_anns if ann_id in before_anns]
    if not common_ann_ids:
        print("[VIS] No common annotation IDs found. Skip visualization.")
        return

    rng = random.Random(seed)
    chosen_ann_ids = rng.sample(common_ann_ids, k=min(num_vis, len(common_ann_ids)))

    vis_updated_only = []
    vis_compare = []
    titles_updated = []
    titles_compare = []

    for ann_id in chosen_ann_ids:
        ann_before = before_anns[ann_id]
        ann_after = after_anns[ann_id]
        image_id = ann_after["image_id"]

        image_meta = image_by_id.get(image_id)
        if image_meta is None:
            continue
        img_path = frame_dir / image_meta["file_name"]
        if not img_path.exists():
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            continue

        img_h, img_w = img.shape[:2]
        if (img_h, img_w) != (image_meta["height"], image_meta["width"]):
            print(
                f"[VIS][WARN] {img_path.name}: actual=({img_h},{img_w}), "
                f"json=({image_meta['height']},{image_meta['width']})"
            )

        if image_id not in image_scale:
            continue

        sx, sy, _, _ = image_scale[image_id]
        old_bbox = ann_before.get("bbox", None)
        new_bbox = ann_after.get("bbox", None)
        if old_bbox is None or new_bbox is None:
            continue

        projected_old_bbox = scale_bbox(old_bbox, sx, sy, image_meta["width"], image_meta["height"])
        max_err = max(abs(a - b) for a, b in zip(projected_old_bbox, new_bbox))

        panel_updated = img.copy()
        draw_bbox(panel_updated, new_bbox, (0, 255, 0), label="updated")
        vis_updated_only.append(panel_updated)
        titles_updated.append(f"ann={ann_id}, img={image_id}")

        panel_compare = img.copy()
        draw_bbox(panel_compare, projected_old_bbox, (0, 215, 255), label="old->scaled")
        draw_bbox(panel_compare, new_bbox, (0, 255, 0), label="updated")
        vis_compare.append(panel_compare)
        titles_compare.append(f"ann={ann_id}, max_err={max_err:.6f}")

    save_contact_sheet(
        vis_updated_only,
        titles_updated,
        vis_dir / "bbox_on_resized_images.png",
        cols=4,
    )
    save_contact_sheet(
        vis_compare,
        titles_compare,
        vis_dir / "bbox_consistency_check.png",
        cols=4,
    )


def main() -> None:
    args = parse_args()

    if args.output_json.exists() and not args.overwrite:
        raise FileExistsError(
            f"Output exists: {args.output_json}. Use --overwrite to replace it."
        )

    coco_before = load_json(args.input_json)
    coco_after, image_scale = update_coco_resolution(
        coco_before,
        target_h=args.target_height,
        target_w=args.target_width,
    )
    save_json(args.output_json, coco_after)

    visualize_samples(
        coco_before=coco_before,
        coco_after=coco_after,
        image_scale=image_scale,
        frame_dir=args.frame_dir,
        vis_dir=args.vis_dir,
        num_vis=args.num_vis,
        seed=args.seed,
    )

    unique_old_shapes = sorted({(v[2], v[3]) for v in image_scale.values()})
    sx_example, sy_example, _, _ = next(iter(image_scale.values()))

    print("=== COCO resolution update done ===")
    print(f"Input JSON:  {args.input_json}")
    print(f"Output JSON: {args.output_json}")
    print(f"Old shapes in JSON (H,W): {unique_old_shapes}")
    print(f"Target shape (H,W): ({args.target_height}, {args.target_width})")
    print(f"Scale factors (x,y): ({sx_example:.8f}, {sy_example:.8f})")
    print(f"Visualizations saved in: {args.vis_dir}")


if __name__ == "__main__":
    main()