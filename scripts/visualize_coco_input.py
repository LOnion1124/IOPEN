"""
Visualize COCO mode network input images.
Display 16 random images in 4x4 grid showing preprocessed images as they are fed to the network.
"""

import os
import json
import random
import numpy as np
import matplotlib.pyplot as plt
import torch
import imageio.v3 as iio
from collections import defaultdict
from pathlib import Path

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import cfg
from src.eval.utils import preprocess_coco_image, bbox_to_crop_xyhw


def visualize_coco_input(num_images=16, grid_size=4, save_path=None):
    """
    Visualize COCO mode network input images.
    
    Args:
        num_images: Number of images to display (default 16)
        grid_size: Grid size (default 4x4)
        save_path: Optional path to save the figure
    """
    
    # Load COCO annotations
    coco_path = cfg['eval']['coco_path']
    frame_dir = cfg['eval']['coco_frame_dir']
    
    print(f"Loading COCO data from: {coco_path}")
    print(f"Frame directory: {frame_dir}")
    
    with open(coco_path, 'r') as f:
        coco = json.load(f)
    
    # Get all valid image-annotation pairs
    image_by_id = {img['id']: img for img in coco.get('images', [])}
    ann_by_image = defaultdict(list)
    
    for ann in coco.get('annotations', []):
        if int(ann.get('iscrowd', 0)) != 0:
            continue
        ann_by_image[ann['image_id']].append(ann)
    
    # Filter images that have at least one annotation
    valid_images = [img_id for img_id in image_by_id.keys() if img_id in ann_by_image]
    
    if len(valid_images) == 0:
        print("No valid images found in COCO dataset!")
        return
    
    print(f"Found {len(valid_images)} valid images with annotations")
    
    # Randomly select images
    num_images = min(num_images, len(valid_images))
    selected_image_ids = random.sample(valid_images, num_images)
    
    # Prepare figure
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(12, 12))
    axes = axes.flatten()
    
    processed_count = 0
    
    for idx, image_id in enumerate(selected_image_ids):
        if processed_count >= num_images:
            break
        
        anns = ann_by_image[image_id]
        if not anns:
            continue
        
        image_meta = image_by_id[image_id]
        img_path = os.path.join(frame_dir, image_meta['file_name'])
        
        if not os.path.exists(img_path):
            print(f"Warning: Image not found: {img_path}")
            continue
        
        # Load image
        rgb_original = iio.imread(img_path)
        if rgb_original.ndim == 2:
            rgb_original = np.stack([rgb_original] * 3, axis=-1)
        if rgb_original.shape[-1] == 4:
            rgb_original = rgb_original[..., :3]
        
        # Randomly select one annotation from this image
        ann = random.choice(anns)
        bbox = ann['bbox']
        
        try:
            # Preprocess image as in COCO eval mode
            img_scaled, crop = preprocess_coco_image(rgb_original, bbox)
            
            if img_scaled is None:
                print(f"Warning: Failed to preprocess image {image_id}")
                continue
            
            # Convert to numpy for display
            # img_scaled is (3, H, W) tensor in CHW format
            img_display = img_scaled.permute(1, 2, 0).numpy().astype(np.uint8)
            
            # Normalize to [0, 1] for display
            img_display = img_display / 255.0
            
            # Display
            ax = axes[processed_count]
            ax.imshow(img_display)
            ax.set_title(f"ID: {image_id}\nBBox: {bbox[:2]}", fontsize=8)
            ax.axis('off')
            
            processed_count += 1
            
        except Exception as e:
            print(f"Error processing image {image_id}: {e}")
            continue
    
    # Hide unused subplots
    for idx in range(processed_count, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to: {save_path}")
    
    plt.show()


def visualize_with_crop_info(num_images=16, grid_size=4, save_path=None):
    """
    Visualize COCO mode input with crop region visualization on original image.
    """
    
    # Load COCO annotations
    coco_path = cfg['eval']['coco_path']
    frame_dir = cfg['eval']['coco_frame_dir']
    
    print(f"Loading COCO data from: {coco_path}")
    
    with open(coco_path, 'r') as f:
        coco = json.load(f)
    
    # Get all valid image-annotation pairs
    image_by_id = {img['id']: img for img in coco.get('images', [])}
    ann_by_image = defaultdict(list)
    
    for ann in coco.get('annotations', []):
        if int(ann.get('iscrowd', 0)) != 0:
            continue
        ann_by_image[ann['image_id']].append(ann)
    
    # Filter images that have at least one annotation
    valid_images = [img_id for img_id in image_by_id.keys() if img_id in ann_by_image]
    
    if len(valid_images) == 0:
        print("No valid images found in COCO dataset!")
        return
    
    print(f"Found {len(valid_images)} valid images with annotations")
    
    # Randomly select images
    num_images = min(num_images, len(valid_images))
    selected_image_ids = random.sample(valid_images, num_images)
    
    # Prepare figure - two rows per image (original with crop + processed)
    fig, axes = plt.subplots(grid_size, grid_size * 2, figsize=(16, 12))
    
    processed_count = 0
    
    for idx, image_id in enumerate(selected_image_ids):
        if processed_count >= num_images:
            break
        
        anns = ann_by_image[image_id]
        if not anns:
            continue
        
        image_meta = image_by_id[image_id]
        img_path = os.path.join(frame_dir, image_meta['file_name'])
        
        if not os.path.exists(img_path):
            continue
        
        # Load image
        rgb_original = iio.imread(img_path)
        if rgb_original.ndim == 2:
            rgb_original = np.stack([rgb_original] * 3, axis=-1)
        if rgb_original.shape[-1] == 4:
            rgb_original = rgb_original[..., :3]
        
        # Randomly select one annotation
        ann = random.choice(anns)
        bbox = ann['bbox']
        
        try:
            img_scaled, crop = preprocess_coco_image(rgb_original, bbox)
            
            if img_scaled is None:
                continue
            
            # Show original with crop box
            ax_orig = axes[processed_count, 0]
            rgb_with_box = rgb_original.copy()
            x, y, h, w = crop
            # Draw rectangle
            cv2.rectangle(rgb_with_box, (x, y), (x + w, y + h), (0, 255, 0), 3)
            ax_orig.imshow(rgb_with_box)
            ax_orig.set_title(f"Original (ID: {image_id})", fontsize=8)
            ax_orig.axis('off')
            
            # Show processed
            ax_proc = axes[processed_count, 1]
            img_display = img_scaled.permute(1, 2, 0).numpy().astype(np.uint8)
            img_display = img_display / 255.0
            ax_proc.imshow(img_display)
            ax_proc.set_title(f"Preprocessed\nCrop: ({x}, {y}, {w}x{h})", fontsize=8)
            ax_proc.axis('off')
            
            processed_count += 1
            
        except Exception as e:
            print(f"Error processing image {image_id}: {e}")
            continue
    
    # Hide unused subplots
    for idx in range(processed_count, grid_size):
        axes[idx, 0].axis('off')
        axes[idx, 1].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to: {save_path}")
    
    plt.show()


if __name__ == '__main__':
    import cv2
    
    # Create output directory if needed
    output_dir = os.path.dirname(cfg['eval']['output_dir'])
    os.makedirs(output_dir, exist_ok=True)
    
    # Visualize preprocessed images
    save_path = os.path.join(output_dir, 'coco_input_visualization.png')
    print("\n=== Visualizing COCO preprocessed inputs (4x4 grid) ===\n")
    visualize_coco_input(num_images=16, grid_size=4, save_path=save_path)
    
    # Optional: visualize with crop information (2x8 subplot layout)
    # save_path_crop = os.path.join(output_dir, 'coco_input_with_crop.png')
    # print("\n=== Visualizing COCO input with crop info ===\n")
    # visualize_with_crop_info(num_images=8, grid_size=4, save_path=save_path_crop)
