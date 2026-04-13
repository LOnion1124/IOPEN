"""
Visualize training samples after preprocessing.

For each sampled item, this script saves a diagnostic canvas including:
1) Network input image (de-normalized for display)
2) Corner coordinates and cuboid edges overlaid on the image
3) Max-pooled heatmap across 8 channels
4) Per-channel heatmaps (8 channels)
"""

import argparse
import os
import random
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import cfg
from src.datasets import make_dataset


_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 0),
    (4, 5), (5, 6), (6, 7), (7, 4),
    (0, 4), (1, 5), (2, 6), (3, 7),
]


def _to_display_image(img_chw: torch.Tensor) -> np.ndarray:
    """Convert model input tensor (C,H,W) into displayable RGB image in [0, 1]."""
    img = img_chw.detach().cpu().float()

    norm_cfg = cfg.get("dataset", {}).get("normalize", {})
    if norm_cfg.get("enabled", True):
        mean_vals = norm_cfg.get("mean", [0.485, 0.456, 0.406])
        std_vals = norm_cfg.get("std", [0.229, 0.224, 0.225])
        mean = torch.tensor(mean_vals, dtype=img.dtype).view(3, 1, 1)
        std = torch.tensor(std_vals, dtype=img.dtype).view(3, 1, 1)
        img = img * std + mean

    img = img.clamp(0.0, 1.0)
    return img.permute(1, 2, 0).numpy()


def _valid_corner_mask(coords: np.ndarray, width: int, height: int) -> np.ndarray:
    return (
        (coords[:, 0] >= 0)
        & (coords[:, 0] < width)
        & (coords[:, 1] >= 0)
        & (coords[:, 1] < height)
    )


def _draw_corner_overlay(ax, img_hwc: np.ndarray, coords_xy: np.ndarray):
    h, w = img_hwc.shape[:2]
    valid = _valid_corner_mask(coords_xy, w, h)

    ax.imshow(img_hwc)
    ax.set_title("Corners + Edges")
    ax.axis("off")

    for edge in _EDGES:
        i, j = edge
        if valid[i] and valid[j]:
            x = [coords_xy[i, 0], coords_xy[j, 0]]
            y = [coords_xy[i, 1], coords_xy[j, 1]]
            ax.plot(x, y, color="lime", linewidth=1.5)

    for i in range(8):
        if valid[i]:
            ax.scatter(coords_xy[i, 0], coords_xy[i, 1], s=25, c="red")
            ax.text(coords_xy[i, 0] + 2, coords_xy[i, 1] + 2, str(i), color="yellow", fontsize=8)


def _plot_sample(sample: dict, sample_idx: int, output_path: str):
    img = sample["img"]
    heatmap = sample["heatmap"].detach().cpu().float().numpy()
    coords = sample["coords"].detach().cpu().float().numpy()

    img_hwc = _to_display_image(img)
    h, w = img_hwc.shape[:2]

    fig, axes = plt.subplots(3, 4, figsize=(14, 10))
    axes_flat = axes.flatten()

    axes_flat[0].imshow(img_hwc)
    axes_flat[0].set_title("Input (de-normalized)")
    axes_flat[0].axis("off")

    _draw_corner_overlay(axes_flat[1], img_hwc, coords)

    max_heat = np.max(heatmap, axis=0)
    axes_flat[2].imshow(img_hwc)
    axes_flat[2].imshow(max_heat, cmap="magma", alpha=0.55)
    axes_flat[2].set_title("Max Heatmap Overlay")
    axes_flat[2].axis("off")

    valid_mask = _valid_corner_mask(coords, w, h)
    axes_flat[3].axis("off")
    axes_flat[3].text(
        0.0,
        0.9,
        f"Sample index: {sample_idx}\nImage size: {w}x{h}\nValid corners: {int(valid_mask.sum())}/8",
        fontsize=11,
        va="top",
    )

    for ch in range(8):
        ax = axes_flat[4 + ch]
        ax.imshow(heatmap[ch], cmap="viridis")
        if valid_mask[ch]:
            ax.scatter(coords[ch, 0], coords[ch, 1], s=20, c="red")
        ax.set_title(f"Heatmap ch{ch}")
        ax.axis("off")

    fig.tight_layout()
    fig.savefig(output_path, dpi=140)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Visualize train/validate samples after preprocessing")
    parser.add_argument("--split", type=str, default="train", choices=["train", "validate"])
    parser.add_argument("--num-samples", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default="data/eval/result/train_sample_viz")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    dataset = make_dataset(split=args.split)
    total = len(dataset)
    if total == 0:
        print("Dataset is empty. Nothing to visualize.")
        return

    random.seed(args.seed)
    count = min(max(1, args.num_samples), total)
    indices = random.sample(range(total), count)

    print(f"Split: {args.split}")
    print(f"Dataset size: {total}")
    print(f"Visualizing {count} samples to: {args.output_dir}")

    for n, idx in enumerate(indices):
        sample = dataset[idx]
        out_path = os.path.join(args.output_dir, f"{args.split}_idx_{idx:06d}.png")
        _plot_sample(sample, idx, out_path)
        if (n + 1) % 5 == 0 or (n + 1) == count:
            print(f"Saved {n + 1}/{count}")

    print("Done.")


if __name__ == "__main__":
    main()
