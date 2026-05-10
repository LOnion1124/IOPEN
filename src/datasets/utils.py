import numpy as np
import json
import torch
import torch.nn.functional as F
import random
from src.config import cfg, args


def _parse_vec3(value, default):
    """Parse a scalar or length-3 sequence into a float vec3 numpy array."""
    if value is None:
        return np.array(default, dtype=np.float32)
    if isinstance(value, (int, float)):
        return np.array([float(value), float(value), float(value)], dtype=np.float32)
    if isinstance(value, (list, tuple)) and len(value) == 3:
        try:
            return np.array([float(value[0]), float(value[1]), float(value[2])], dtype=np.float32)
        except (TypeError, ValueError):
            return np.array(default, dtype=np.float32)
    return np.array(default, dtype=np.float32)


def _build_resize_crop_meta(img_h, img_w, bbox_xywh, target_h, target_w):
    """Build a uniform-resize + crop transform centered on the object bbox."""

    x, y, bbox_w, bbox_h = bbox_xywh
    x = float(x)
    y = float(y)
    bbox_w = float(bbox_w)
    bbox_h = float(bbox_h)

    if not np.isfinite([x, y, bbox_w, bbox_h]).all() or bbox_w <= 1e-6 or bbox_h <= 1e-6:
        scale = min(float(target_w) / max(float(img_w), 1.0), float(target_h) / max(float(img_h), 1.0))
        crop_center_x = 0.5 * float(img_w) * scale
        crop_center_y = 0.5 * float(img_h) * scale
    else:
        scale = min(float(target_w) / bbox_w, float(target_h) / bbox_h)
        crop_center_x = (x + 0.5 * bbox_w) * scale
        crop_center_y = (y + 0.5 * bbox_h) * scale

    scale = float(max(scale, 1e-6))
    resized_h = max(1, int(round(float(img_h) * scale)))
    resized_w = max(1, int(round(float(img_w) * scale)))
    crop_x = int(round(crop_center_x - 0.5 * float(target_w)))
    crop_y = int(round(crop_center_y - 0.5 * float(target_h)))

    return {
        'scale': scale,
        'resized_h': resized_h,
        'resized_w': resized_w,
        'crop_x': crop_x,
        'crop_y': crop_y,
        'target_h': int(target_h),
        'target_w': int(target_w),
    }


def _resize_tensor(tensor, size, mode='bilinear'):
    if tensor.shape[-2:] == size:
        return tensor

    resized = F.interpolate(
        tensor.unsqueeze(0),
        size=size,
        mode=mode,
        align_corners=False,
    )
    return resized.squeeze(0)


def _crop_tensor_with_padding(tensor, crop_x, crop_y, target_h, target_w):
    channels, height, width = tensor.shape
    output = tensor.new_zeros((channels, target_h, target_w))

    src_x0 = max(0, crop_x)
    src_y0 = max(0, crop_y)
    src_x1 = min(width, crop_x + target_w)
    src_y1 = min(height, crop_y + target_h)

    if src_x1 <= src_x0 or src_y1 <= src_y0:
        return output

    dst_x0 = max(0, -crop_x)
    dst_y0 = max(0, -crop_y)
    dst_x1 = dst_x0 + (src_x1 - src_x0)
    dst_y1 = dst_y0 + (src_y1 - src_y0)

    output[:, dst_y0:dst_y1, dst_x0:dst_x1] = tensor[:, src_y0:src_y1, src_x0:src_x1]
    return output


def _transform_coords_with_meta(coords, meta):
    transformed = coords.clone()
    valid = (
        torch.isfinite(transformed).all(dim=-1) &
        (transformed[:, 0] >= 0) &
        (transformed[:, 1] >= 0)
    )

    transformed[valid, 0] = transformed[valid, 0] * meta['scale'] - meta['crop_x']
    transformed[valid, 1] = transformed[valid, 1] * meta['scale'] - meta['crop_y']
    return transformed


def _direct_resize_tensor(tensor, target_h, target_w, mode='bilinear'):
    if tensor.shape[-2:] == (target_h, target_w):
        return tensor
    return F.interpolate(
        tensor.unsqueeze(0),
        size=(target_h, target_w),
        mode=mode,
        align_corners=False,
    ).squeeze(0)

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
            bbox_2d_xywh (tuple): Tuple (x_min, y_min, w, h) representing the padded 2D bounding box in image coordinates.
    Notes:
        - The heatmap for each keypoint is generated using a Gaussian centered at the projected location.
        - Padding is applied to the 2D bounding box to account for object size.
    """

    cx, cy, fx, fy = camera['cx'], camera['cy'], camera['fx'], camera['fy']
    H, W = camera['height'], camera['width']
    dx, dy, dz = float(model['size_x']), float(model['size_y']), float(model['size_z'])

    gt_bbox_cfg = cfg.get('dataset', {}).get('gt_bbox_3d', {})
    gt_bbox_enabled = bool(gt_bbox_cfg.get('enabled', False))
    if gt_bbox_enabled:
        scale_xyz = _parse_vec3(gt_bbox_cfg.get('scale_xyz', [1.0, 1.0, 1.0]), [1.0, 1.0, 1.0])
        scale_xyz = np.maximum(scale_xyz, 1e-6)
        center_offset_xyz = _parse_vec3(
            gt_bbox_cfg.get('center_offset_xyz', [0.0, 0.0, 0.0]),
            [0.0, 0.0, 0.0],
        )
    else:
        scale_xyz = np.array([1.0, 1.0, 1.0], dtype=np.float32)
        center_offset_xyz = np.array([0.0, 0.0, 0.0], dtype=np.float32)

    dx *= float(scale_xyz[0])
    dy *= float(scale_xyz[1])
    dz *= float(scale_xyz[2])
    
    bbox_3d = np.array([
        [-dx/2, -dy/2, -dz/2],
        [ dx/2, -dy/2, -dz/2],
        [ dx/2,  dy/2, -dz/2],
        [-dx/2,  dy/2, -dz/2],
        [-dx/2, -dy/2,  dz/2],
        [ dx/2, -dy/2,  dz/2],
        [ dx/2,  dy/2,  dz/2],
        [-dx/2,  dy/2,  dz/2],
    ], dtype=np.float32)
    bbox_3d = bbox_3d + center_offset_xyz.reshape(1, 3)

    bbox_cam = (cam_R_m2c @ bbox_3d.T + cam_t_m2c).T # shape (8, 3)
    bbox_2d = np.full((8, 2), -1.0, dtype=np.float32)
    valid_proj = np.zeros(8, dtype=bool)
    z_eps = 1e-6
    for i, (x, y, z) in enumerate(bbox_cam):
        if (not np.isfinite(z)) or z <= z_eps:
            continue
        u = fx * x / z + cx
        v = fy * y / z + cy
        if not (np.isfinite(u) and np.isfinite(v)):
            continue
        bbox_2d[i] = [u, v]
        valid_proj[i] = True

    if not np.any(valid_proj):
        heatmap = np.zeros((8, H, W), dtype=np.float32)
        return heatmap, bbox_2d, (0, 0, W, H)

    # Compute obj_size as the average of length and width from projected 2D bbox corners
    valid_2d = bbox_2d[valid_proj]
    x_coords = valid_2d[:, 0]
    y_coords = valid_2d[:, 1]
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
    bbox_2d_xywh = (x_min_padded, y_min_padded, w_padded, h_padded)

    heatmap = np.zeros((8, H, W), dtype=np.float32)
    sigma = max(obj_size / 10, 1e-3)

    for i, (u, v) in enumerate(bbox_2d):
        if not valid_proj[i]:
            continue
        u, v = int(round(u)), int(round(v))
        if 0 <= u < W and 0 <= v < H:
            y, x = np.ogrid[:H, :W]
            dist2 = (x - u) ** 2 + (y - v) ** 2
            gaussian = np.exp(-dist2 / (2.0 * sigma * sigma))
            heatmap[i] = gaussian

    return heatmap, bbox_2d, bbox_2d_xywh

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

def gen_scaled_data(img, heatmap, coords, bbox=None, return_meta=False):
    """
    Uniformly resizes the input image, then crops or pads to the target dimensions.
    Args:
        img (torch.Tensor): Input image tensor of shape (3, H, W).
        heatmap (torch.Tensor): Input heatmap tensor of shape (8, H, W).
        coords (torch.Tensor): Tensor of shape (8, 2) containing 8 (x, y) coordinates.
        bbox (tuple, optional): Object bbox in (x, y, w, h) format on the input image.
        return_meta (bool, optional): If True, also return the resize/crop transform.
    Returns:
        tuple:
            scaled_img (torch.Tensor): Resized image tensor of shape (3, new_H, new_W).
            scaled_heatmap (torch.Tensor): Resized heatmap tensor of shape (8, new_H, new_W).
            scaled_coords (torch.Tensor): Scaled coordinates tensor of shape (8, 2).
    Notes:
        - Uses bilinear interpolation for resizing.
        - Coordinates with negative values are not transformed.
        - Target dimensions are taken from `cfg['height']` and `cfg['width']`.
    """

    new_H, new_W = cfg['height'], cfg['width']
    _, H, W = img.shape

    if bbox is None:
        scaled_img = _direct_resize_tensor(img, new_H, new_W)
        scaled_heatmap = _direct_resize_tensor(heatmap, new_H, new_W)

        scale_x = new_W / W
        scale_y = new_H / H
        scaled_coords = coords.clone()
        valid = (scaled_coords[:, 0] >= 0) & (scaled_coords[:, 1] >= 0)
        scaled_coords[valid, 0] = scaled_coords[valid, 0] * scale_x
        scaled_coords[valid, 1] = scaled_coords[valid, 1] * scale_y

        if return_meta:
            return scaled_img, scaled_heatmap, scaled_coords, None
        return scaled_img, scaled_heatmap, scaled_coords

    meta = _build_resize_crop_meta(H, W, bbox, new_H, new_W)
    resized_size = (meta['resized_h'], meta['resized_w'])

    scaled_img = _resize_tensor(img, resized_size, mode='bilinear')
    scaled_heatmap = _resize_tensor(heatmap, resized_size, mode='bilinear')

    scaled_img = _crop_tensor_with_padding(
        scaled_img,
        meta['crop_x'],
        meta['crop_y'],
        meta['target_h'],
        meta['target_w'],
    )
    scaled_heatmap = _crop_tensor_with_padding(
        scaled_heatmap,
        meta['crop_x'],
        meta['crop_y'],
        meta['target_h'],
        meta['target_w'],
    )
    scaled_coords = _transform_coords_with_meta(coords, meta)

    if return_meta:
        return scaled_img, scaled_heatmap, scaled_coords, meta
    return scaled_img, scaled_heatmap, scaled_coords

def _normalize_scene_ids(scene_ids):
    normalized = []
    for scene_id in scene_ids:
        if isinstance(scene_id, str):
            scene_id = int(scene_id)
        normalized.append(int(scene_id))
    return sorted(set(normalized))


def load_data(root, scene_ids=None, num_scene=1, img_per_scene=1000):
    """
    Load dataset from PBR training data with scene and image filtering.
    
    :param root: Root directory path containing models, camera.json, and train_pbr/
    :param scene_ids: Explicit scene ids to load, e.g. [0, 1, 4]
    :param num_scene: Fallback number of scenes to load if scene_ids is None
    :param img_per_scene: Number of images per scene to process (default: 1000)
    :return: Dictionary containing model info, camera parameters, pbr_root path, and samples with rgb/mask paths and poses
    """
    data_dict = {}

    target_obj_id = int(cfg.get('dataset', {}).get('target_obj_id', 1))
    target_obj_key = str(target_obj_id)

    with open(root + "models/models_info.json") as f:
        models = json.load(f)
        if target_obj_key in models:
            data_dict['model'] = models[target_obj_key]
        else:
            # Fallback to keep backward compatibility when object id config is missing.
            first_key = sorted(models.keys(), key=lambda x: int(x))[0]
            data_dict['model'] = models[first_key]
            target_obj_id = int(first_key)
    
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
    if scene_ids is None:
        scene_ids = list(range(num_scene))
    scene_ids = _normalize_scene_ids(scene_ids)

    for scene_id in scene_ids:
        scene_path = str(scene_id).zfill(6) + "/"
        with open(pbr_root + scene_path + "scene_gt.json") as f:
            scene_gt = json.load(f)
        with open(pbr_root + scene_path + "scene_gt_info.json") as f:
            scene_gt_info = json.load(f)

        frame_keys = sorted(scene_gt.keys(), key=lambda k: int(k))
        if img_per_scene is not None and img_per_scene > 0:
            frame_keys = frame_keys[:img_per_scene]

        for frame_key in frame_keys:
            anns = scene_gt.get(frame_key, [])
            anns_info = scene_gt_info.get(frame_key, [])
            frame_id = int(frame_key)
            rgb_path = scene_path + "rgb/" + str(frame_id).zfill(6) + ".jpg"

            num_instance = min(len(anns), len(anns_info))
            for j in range(num_instance):
                ann = anns[j]
                ann_info = anns_info[j]

                if int(ann.get("obj_id", -1)) != target_obj_id:
                    continue

                visib_fract = float(ann_info.get("visib_fract", 0.0))
                if visib_fract <= 0.8:
                    continue

                mask_path = scene_path + "mask_visib/" + str(frame_id).zfill(6) + "_" + str(j).zfill(6) + ".png"
                cam_R_m2c = ann["cam_R_m2c"]
                cam_t_m2c = ann["cam_t_m2c"]

                data_dict['samples']['rgb_path'].append(rgb_path)
                data_dict['samples']['mask_path'].append(mask_path)
                data_dict['samples']['cam_R_m2c'].append(cam_R_m2c)
                data_dict['samples']['cam_t_m2c'].append(cam_t_m2c)

    # Apply random sampling based on sample_rate from config
    sample_rate = cfg.get('dataset', {}).get('sample_rate', 1.0)
    if sample_rate < 1.0:
        num_samples = len(data_dict['samples']['rgb_path'])
        if num_samples > 0:
            num_to_keep = max(1, int(num_samples * sample_rate))
            indices = random.sample(range(num_samples), num_to_keep)
        else:
            indices = []
        
        data_dict['samples']['rgb_path'] = [data_dict['samples']['rgb_path'][i] for i in indices]
        data_dict['samples']['mask_path'] = [data_dict['samples']['mask_path'][i] for i in indices]
        data_dict['samples']['cam_R_m2c'] = [data_dict['samples']['cam_R_m2c'][i] for i in indices]
        data_dict['samples']['cam_t_m2c'] = [data_dict['samples']['cam_t_m2c'][i] for i in indices]

    return data_dict