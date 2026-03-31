import numpy as np
import json
import cv2
import torch
import torch.nn.functional as F
import random
from src.config import cfg, args

def gen_masked_img(rgb, mask):
    """
    Generate a masked image by applying a binary mask to an RGB image.
    
    :param rgb: RGB image array of shape (height, width, 3)
    :param mask: Binary mask array of shape (height, width)
    :return: Masked image with same shape as rgb
    """
    masked_img = rgb.copy()
    masked_img[mask == 0] = [0, 255, 0]
    return masked_img

def gen_gt(camera, model, cam_R_m2c, cam_t_m2c):
    """
    Projects the 3D bounding box corners of a model onto the image plane and generates heatmaps for each keypoint.
    Args:
        camera (dict): Camera intrinsics with keys 'cx', 'cy', 'fx', 'fy', 'height', 'width'.
        model (dict): Model dimensions with keys 'size_x', 'size_y', 'size_z'.
        cam_R_m2c (np.ndarray): 3x3 rotation matrix from model to camera coordinates.
        cam_t_m2c (np.ndarray): 3-element translation vector from model to camera coordinates.
    Returns:
        tuple:
            heatmap (np.ndarray): Array of shape (8, H, W) containing Gaussian heatmaps for each projected keypoint.
            bbox_2d (np.ndarray): Array of shape (8, 2) with 2D image coordinates of projected 3D bounding box corners.
            bbox_2d_xyhw (tuple): Tuple (x_min, y_min, h, w) representing the padded 2D bounding box in image coordinates.
    Notes:
        - The heatmap for each keypoint is generated using a Gaussian centered at the projected location.
        - Padding is applied to the 2D bounding box to account for object size.
    """

    cx, cy, fx, fy = camera['cx'], camera['cy'], camera['fx'], camera['fy']
    H, W = camera['height'], camera['width']
    dx, dy, dz = model['size_x'], model['size_y'], model['size_z']
    
    bbox_3d = np.array([
        [-dx/2, -dy/2, -dz/2],
        [ dx/2, -dy/2, -dz/2],
        [ dx/2,  dy/2, -dz/2],
        [-dx/2,  dy/2, -dz/2],
        [-dx/2, -dy/2,  dz/2],
        [ dx/2, -dy/2,  dz/2],
        [ dx/2,  dy/2,  dz/2],
        [-dx/2,  dy/2,  dz/2],
    ])

    bbox_cam = (cam_R_m2c @ bbox_3d.T + cam_t_m2c).T # shape (8, 3)
    bbox_2d = []
    for x, y, z in bbox_cam:
        u = fx * x / z + cx
        v = fy * y / z + cy
        bbox_2d.append([u, v])
    bbox_2d = np.array(bbox_2d) # shape (8, 2)

    # Compute obj_size as the average of length and width from projected 2D bbox corners
    x_coords = bbox_2d[:, 0]
    y_coords = bbox_2d[:, 1]
    length = np.max(x_coords) - np.min(x_coords)
    width = np.max(y_coords) - np.min(y_coords)
    obj_size = 0.5 * (length + width)

    x_min = np.min(x_coords)
    x_max = np.max(x_coords)
    y_min = np.min(y_coords)
    y_max = np.max(y_coords)
    pad = int(round(0.1 * obj_size))
    x_min_padded = max(0, int(round(x_min)) - pad)
    y_min_padded = max(0, int(round(y_min)) - pad)
    h_padded = min(H - y_min_padded, int(round(y_max - y_min)) + 2 * pad)
    w_padded = min(W - x_min_padded, int(round(x_max - x_min)) + 2 * pad)
    bbox_2d_xyhw = (x_min_padded, y_min_padded, h_padded, w_padded)

    heatmap = np.zeros((8, H, W), dtype=np.float32)
    denominator = (obj_size / 10) ** 2  # Standard deviation for Gaussian

    for i, (u, v) in enumerate(bbox_2d):
        u, v = int(round(u)), int(round(v))
        if 0 <= u < W and 0 <= v < H:
            y, x = np.ogrid[:H, :W]
            gaussian = np.exp(-((x - u)**2 + (y - v)**2) ** 0.5 / denominator)
            heatmap[i] = gaussian

    return heatmap, bbox_2d, bbox_2d_xyhw

def gen_cropped_data(img, heatmap, coords, bbox):
    """
    Crops the input image, heatmap, and coordinates based on the given bounding box.
    Args:
        img (np.ndarray): Input image of shape (H, W, 3).
        heatmap (np.ndarray): Heatmap of shape (8, H, W).
        coords (list): List of 8 (x, y) coordinates.
        bbox (tuple): Bounding box specified as (x, y, h, w).
    Returns:
        tuple:
            cropped_img (np.ndarray): Cropped image of shape (h, w, 3).
            cropped_heatmap (np.ndarray): Cropped heatmap of shape (8, h, w).
            cropped_coords (np.ndarray): Array of shape (8, 2) with updated coordinates.
                Coordinates outside the crop are marked as [-1, -1].
    """

    H, W = img.shape[:2]
    x, y, h, w = bbox

    # Randomly scale and shift the crop box while keeping all valid GT corners inside.
    crop_aug_cfg = cfg.get('dataset', {}).get('crop_aug', {})
    aug_enabled = crop_aug_cfg.get('enabled', True)
    scale_min = float(crop_aug_cfg.get('scale_min', 0.9))
    scale_max = float(crop_aug_cfg.get('scale_max', 1.2))
    shift_ratio = float(crop_aug_cfg.get('shift_ratio', 0.1))

    if aug_enabled:
        scale_min = max(1e-3, scale_min)
        scale_max = max(scale_min, scale_max)
        shift_ratio = max(0.0, shift_ratio)

        cx = x + w * 0.5
        cy = y + h * 0.5

        scale = random.uniform(scale_min, scale_max)
        h_aug = max(1, int(round(h * scale)))
        w_aug = max(1, int(round(w * scale)))

        dx = random.uniform(-shift_ratio, shift_ratio) * w
        dy = random.uniform(-shift_ratio, shift_ratio) * h
        cx_aug = cx + dx
        cy_aug = cy + dy

        # Keep the sampled center within image so the crop is always valid.
        half_w = 0.5 * w_aug
        half_h = 0.5 * h_aug
        cx_aug = float(np.clip(cx_aug, half_w, max(half_w, W - half_w)))
        cy_aug = float(np.clip(cy_aug, half_h, max(half_h, H - half_h)))

        x_aug = int(round(cx_aug - half_w))
        y_aug = int(round(cy_aug - half_h))
        x_aug = min(max(0, x_aug), max(0, W - w_aug))
        y_aug = min(max(0, y_aug), max(0, H - h_aug))

        valid_coords = coords[
            (coords[:, 0] >= 0) & (coords[:, 0] < W) &
            (coords[:, 1] >= 0) & (coords[:, 1] < H)
        ]

        if valid_coords.shape[0] > 0:
            gt_x_min = int(np.floor(np.min(valid_coords[:, 0])))
            gt_x_max = int(np.ceil(np.max(valid_coords[:, 0])))
            gt_y_min = int(np.floor(np.min(valid_coords[:, 1])))
            gt_y_max = int(np.ceil(np.max(valid_coords[:, 1])))

            x_aug = min(x_aug, gt_x_min)
            y_aug = min(y_aug, gt_y_min)

            end_x = max(x_aug + w_aug, gt_x_max + 1)
            end_y = max(y_aug + h_aug, gt_y_max + 1)

            end_x = min(W, end_x)
            end_y = min(H, end_y)

            x_aug = max(0, min(x_aug, end_x - 1))
            y_aug = max(0, min(y_aug, end_y - 1))
            w_aug = max(1, end_x - x_aug)
            h_aug = max(1, end_y - y_aug)

        x, y, h, w = x_aug, y_aug, h_aug, w_aug

    x = max(0, min(x, W - 1))
    y = max(0, min(y, H - 1))
    h = max(1, min(h, H - y))
    w = max(1, min(w, W - x))

    cropped_img = img[y:y+h, x:x+w]
    cropped_heatmap = heatmap[:, y:y+h, x:x+w]
    cropped_coords = []
    for coord in coords:
        u, v = coord
        cropped_u = u - x
        cropped_v = v - y
        if 0 <= cropped_u < w and 0 <= cropped_v < h:
            cropped_coords.append([cropped_u, cropped_v])
        else:
            cropped_coords.append([-1, -1])  # Mark out-of-crop points
    cropped_coords = np.array(cropped_coords)

    return cropped_img, cropped_heatmap, cropped_coords

def gen_scaled_data(img, heatmap, coords):
    """
    Resizes the input image and heatmap to the target dimensions specified in `cfg`,
    and scales the coordinates accordingly.
    Args:
        img (torch.Tensor): Input image tensor of shape (3, H, W).
        heatmap (torch.Tensor): Input heatmap tensor of shape (8, H, W).
        coords (torch.Tensor): Tensor of shape (8, 2) containing 8 (x, y) coordinates.
    Returns:
        tuple:
            scaled_img (torch.Tensor): Resized image tensor of shape (3, new_H, new_W).
            scaled_heatmap (torch.Tensor): Resized heatmap tensor of shape (8, new_H, new_W).
            scaled_coords (torch.Tensor): Scaled coordinates tensor of shape (8, 2).
    Notes:
        - Uses bilinear interpolation for resizing.
        - Coordinates with negative values are not scaled.
        - Target dimensions are taken from `cfg['height']` and `cfg['width']`.
    """

    new_H, new_W = cfg['height'], cfg['width']
    _, H, W = img.shape

    if H == new_H and W == new_W:
        return img, heatmap, coords
    
    scaled_img = F.interpolate(
        img.unsqueeze(0),
        size=(new_H, new_W),
        mode='bilinear',
        align_corners=False
    ).squeeze(0)

    scaled_heatmap = F.interpolate(
        heatmap.unsqueeze(0),
        size=(new_H, new_W),
        mode='bilinear',
        align_corners=False
    ).squeeze(0)

    scale_x = new_W / W
    scale_y = new_H / H
    scaled_coords = coords.clone()
    valid = (scaled_coords[:, 0] >= 0) & (scaled_coords[:, 1] >= 0)
    scaled_coords[valid, 0] = scaled_coords[valid, 0] * scale_x
    scaled_coords[valid, 1] = scaled_coords[valid, 1] * scale_y

    return scaled_img, scaled_heatmap, scaled_coords

def load_data(root, num_scene=1, img_per_scene=1000):
    """
    Load dataset from PBR training data with scene and image filtering.
    
    :param root: Root directory path containing models, camera.json, and train_pbr/
    :param num_scene: Number of scenes to load (default: 6)
    :param img_per_scene: Number of images per scene to process (default: 1000)
    :return: Dictionary containing model info, camera parameters, pbr_root path, and samples with rgb/mask paths and poses
    """
    data_dict = {}

    with open(root + "models/models_info.json") as f:
        models = json.load(f)
        data_dict['model'] = models["1"]
    
    with open(root + "camera.json") as f:
        data_dict['camera'] = json.load(f)
    
    pbr_root = root + "train_pbr/"
    data_dict['pbr_root'] = pbr_root

    data_dict['samples'] = {
        'rgb_path': [],
        'mask_path': [],
        'cam_R_m2c': [],
        'cam_t_m2c': [],
        # 'obj_bbox': []
    }
    for scene_id in range(num_scene):
        scene_path = "00000" + str(scene_id) + "/"
        with open(pbr_root + scene_path + "scene_gt.json") as f:
            scene_gt = json.load(f)
        with open(pbr_root + scene_path + "scene_gt_info.json") as f:
            scene_gt_info = json.load(f)
        
        for i in range(img_per_scene):
            rgb_path = scene_path + "rgb/" + str(i).zfill(6) + ".jpg"
            num_instance = len(scene_gt[str(i)])

            if num_instance > 1:
                continue

            for j in range(num_instance):
                mask_path = scene_path + "mask_visib/" + str(i).zfill(6) + "_" + str(j).zfill(6) + ".png"

                cam_R_m2c = scene_gt[str(i)][j]["cam_R_m2c"]
                cam_t_m2c = scene_gt[str(i)][j]["cam_t_m2c"]

                visib_fract = scene_gt_info[str(i)][j]["visib_fract"]
                # x, y, h, w = scene_gt_info[str(i)][j]["bbox_obj"]

                if visib_fract > 0.2:
                    data_dict['samples']['rgb_path'].append(rgb_path)
                    data_dict['samples']['mask_path'].append(mask_path)
                    data_dict['samples']['cam_R_m2c'].append(cam_R_m2c)
                    data_dict['samples']['cam_t_m2c'].append(cam_t_m2c)
                    # data_dict['samples']['obj_bbox'].append([x, y, h, w])

    # Apply random sampling based on sample_rate from config
    sample_rate = cfg.get('dataset', {}).get('sample_rate', 1.0)
    if sample_rate < 1.0:
        num_samples = len(data_dict['samples']['rgb_path'])
        num_to_keep = max(1, int(num_samples * sample_rate))
        indices = random.sample(range(num_samples), num_to_keep)
        
        data_dict['samples']['rgb_path'] = [data_dict['samples']['rgb_path'][i] for i in indices]
        data_dict['samples']['mask_path'] = [data_dict['samples']['mask_path'][i] for i in indices]
        data_dict['samples']['cam_R_m2c'] = [data_dict['samples']['cam_R_m2c'][i] for i in indices]
        data_dict['samples']['cam_t_m2c'] = [data_dict['samples']['cam_t_m2c'][i] for i in indices]

    return data_dict