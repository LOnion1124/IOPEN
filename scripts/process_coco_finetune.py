#!/usr/bin/env python3
"""Process a COCO file, uniformly sample 500 keyframes, preprocess and save denormalized inputs.

Saves images into `data/finetune/rgb/camera_left`, `data/finetune/rgb/camera_right`,
and `data/finetune/rgb/camera_unknown` when the filename doesn't indicate side.

Usage: python scripts/process_coco_finetune.py --coco data/eval/instances_default_2464x3248.json
"""
import os
import sys
import argparse
import json
import numpy as np
import imageio.v3 as iio
import cv2
from tqdm import tqdm
import torch

# Ensure repository root is on sys.path so `src` imports work when script
# is executed from `scripts/` or elsewhere.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from src.eval.utils import (
    preprocess_coco_image,
    denormalize_for_visualization,
    get_sorted_image_ids_for_temporal_coco,
)


def mkdir_p(path):
    os.makedirs(path, exist_ok=True)


# containers for tensors and names collected during processing
# each entry corresponds to one saved crop image (normalized tensor shape [3,H,W])
left_tensors = []
left_names = []
right_tensors = []
right_names = []


def detect_camera_side(file_name: str):
    name = file_name.lower()
    if 'left' in name:
        return 'camera_left'
    if 'right' in name:
        return 'camera_right'
    return 'camera_unknown'


def main(args):
    with open(args.coco, 'r') as f:
        coco = json.load(f)

    images = {img['id']: img for img in coco.get('images', [])}
    anns_by_image = {}
    for ann in coco.get('annotations', []):
        img_id = ann['image_id']
        anns_by_image.setdefault(img_id, []).append(ann)

    # build category id -> name map
    cat_map = {c['id']: c.get('name', '') for c in coco.get('categories', [])}

    sorted_ids = get_sorted_image_ids_for_temporal_coco(coco)
    total = len(sorted_ids)
    if total == 0:
        print('No images found in COCO file')
        return

    num_samples = min(args.num, total)
    if num_samples >= total:
        chosen_ids = sorted_ids
    else:
        idxs = np.linspace(0, total - 1, num_samples, dtype=int)
        chosen_ids = [sorted_ids[i] for i in idxs]

    out_base = args.out_base
    left_dir = os.path.join(out_base, 'camera_left')
    right_dir = os.path.join(out_base, 'camera_right')
    unknown_dir = os.path.join(out_base, 'camera_unknown')
    for d in (left_dir, right_dir, unknown_dir):
        mkdir_p(d)

    for img_id in tqdm(chosen_ids, desc='Processing'):
        img_rec = images.get(img_id)
        if img_rec is None:
            continue
        file_name = img_rec.get('file_name', str(img_id))

        # Resolve path relative to COCO file: assume files are under data/eval/
        # Try a few sensible locations
        candidate_paths = [
            os.path.join(os.path.dirname(args.coco), file_name),
            os.path.join(os.path.dirname(args.coco), 'frame', file_name),
            os.path.join(os.path.dirname(args.coco), file_name.replace('frame/', '')),
            file_name,
        ]

        img_path = None
        for p in candidate_paths:
            if os.path.exists(p):
                img_path = p
                break

        if img_path is None:
            # try joining with data/eval
            p = os.path.join(os.getcwd(), 'data', 'eval', file_name)
            if os.path.exists(p):
                img_path = p

        if img_path is None:
            # if file_name is simple like frame_000001.png, try data/eval/frame/
            p = os.path.join(os.getcwd(), 'data', 'eval', 'frame', file_name)
            if os.path.exists(p):
                img_path = p

        if img_path is None:
            # skip if cannot find image
            print(f'Warning: image file not found for {file_name} (id={img_id})')
            continue

        try:
            rgb = iio.imread(img_path)
        except Exception:
            rgb = cv2.imread(img_path)[:, :, ::-1]

        anns = anns_by_image.get(img_id, [])
        # process each annotation labelled as camera_left or camera_right
        saved_any = False
        for ann in anns:
            cat_name = cat_map.get(ann.get('category_id'), '')
            if cat_name not in ('camera_left', 'camera_right'):
                continue

            bbox = ann.get('bbox')
            img_scaled, meta = preprocess_coco_image(rgb, bbox)
            if img_scaled is None:
                continue

            # save the normalized tensor (model input) for later use
            try:
                tensor_to_save = img_scaled.detach().cpu()
            except Exception:
                tensor_to_save = img_scaled.cpu()

            img_denorm = denormalize_for_visualization(img_scaled)
            if hasattr(img_denorm, 'cpu'):
                img_denorm = img_denorm.detach().cpu().numpy()

            # img_denorm is (3, H, W) in float range ~0-255
            if img_denorm.ndim == 3 and img_denorm.shape[0] in (1, 3):
                img_hwc = img_denorm.transpose(1, 2, 0)
            else:
                img_hwc = img_denorm

            img_hwc = np.clip(img_hwc, 0, 255).astype(np.uint8)

            # create output filename: original stem + _left/_right.png
            stem, _ext = os.path.splitext(os.path.basename(file_name))
            out_name = f"{stem}_{cat_name}.png"
            if cat_name == 'camera_left':
                out_path = os.path.join(left_dir, out_name)
                left_tensors.append(tensor_to_save)
                left_names.append(out_name)
            else:
                out_path = os.path.join(right_dir, out_name)
                right_tensors.append(tensor_to_save)
                right_names.append(out_name)

            cv2.imwrite(out_path, cv2.cvtColor(img_hwc, cv2.COLOR_RGB2BGR))
            saved_any = True

        # If no left/right annotations found, optionally save whole-image crops to unknown
        if not saved_any:
            h, w = rgb.shape[:2]
            bbox = (0, 0, w, h)
            img_scaled, meta = preprocess_coco_image(rgb, bbox)
            if img_scaled is None:
                continue
            img_denorm = denormalize_for_visualization(img_scaled)
            if hasattr(img_denorm, 'cpu'):
                img_denorm = img_denorm.detach().cpu().numpy()
            if img_denorm.ndim == 3 and img_denorm.shape[0] in (1, 3):
                img_hwc = img_denorm.transpose(1, 2, 0)
            else:
                img_hwc = img_denorm
            img_hwc = np.clip(img_hwc, 0, 255).astype(np.uint8)
            out_path = os.path.join(unknown_dir, os.path.basename(file_name))
            cv2.imwrite(out_path, cv2.cvtColor(img_hwc, cv2.COLOR_RGB2BGR))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--coco', type=str, default='data/eval/instances_default_2464x3248.json')
    parser.add_argument('--num', type=int, default=500)
    parser.add_argument('--out-base', type=str, default='data/finetune/rgb')
    args = parser.parse_args()
    main(args)

    # After processing, save collected tensors and write simple indices for reading
    import json

    out_base = args.out_base
    mkdir_p(out_base)

    left_out_file = os.path.join(out_base, 'camera_left_tensors.pt')
    right_out_file = os.path.join(out_base, 'camera_right_tensors.pt')
    left_index_file = os.path.join(out_base, 'camera_left_index.json')
    right_index_file = os.path.join(out_base, 'camera_right_index.json')

    try:
        torch.save({'tensors': left_tensors, 'names': left_names}, left_out_file)
        with open(left_index_file, 'w') as f:
            json.dump({'names': left_names}, f)
        print(f'Saved {len(left_names)} left tensors -> {left_out_file}')
    except Exception as e:
        print(f'Warning: failed to save left tensors: {e}')

    try:
        torch.save({'tensors': right_tensors, 'names': right_names}, right_out_file)
        with open(right_index_file, 'w') as f:
            json.dump({'names': right_names}, f)
        print(f'Saved {len(right_names)} right tensors -> {right_out_file}')
    except Exception as e:
        print(f'Warning: failed to save right tensors: {e}')
