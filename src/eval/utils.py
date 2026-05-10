import torch
import cv2
import numpy as np
import re
from src.config import cfg, args
from src.datasets.utils import gen_scaled_data


def _normalize_image_tensor(img_tensor):
    norm_cfg = cfg.get('dataset', {}).get('normalize', {})
    if not norm_cfg.get('enabled', True):
        return img_tensor

    mean_vals = norm_cfg.get('mean', [0.485, 0.456, 0.406])
    std_vals = norm_cfg.get('std', [0.229, 0.224, 0.225])
    mean = torch.tensor(mean_vals, dtype=img_tensor.dtype, device=img_tensor.device).view(3, 1, 1)
    std = torch.tensor(std_vals, dtype=img_tensor.dtype, device=img_tensor.device).view(3, 1, 1).clamp_min(1e-6)
    return (img_tensor / 255.0 - mean) / std


def _denormalize_image_tensor(img_tensor):
    norm_cfg = cfg.get('dataset', {}).get('normalize', {})
    if not norm_cfg.get('enabled', True):
        return img_tensor

    mean_vals = norm_cfg.get('mean', [0.485, 0.456, 0.406])
    std_vals = norm_cfg.get('std', [0.229, 0.224, 0.225])
    mean = torch.tensor(mean_vals, dtype=img_tensor.dtype, device=img_tensor.device).view(3, 1, 1)
    std = torch.tensor(std_vals, dtype=img_tensor.dtype, device=img_tensor.device).view(3, 1, 1).clamp_min(1e-6)
    return (img_tensor * std + mean) * 255.0

def gen_coords(heatmap):
    """
    Extract 2D coordinates of maximum values from a batch of heatmaps.
    This function finds the location of the maximum value in each heatmap
    channel and converts the flattened index to 2D (x, y) coordinates.
    Args:
        heatmap (torch.Tensor): A 4D tensor of shape (B, 8, H, W) where:
            - B: batch size
            - 8: number of heatmap channels
            - H: height of each heatmap
            - W: width of each heatmap
    Returns:
        list: A list of length B, where each element is a list of 8 tuples.
              Each tuple contains (x, y) coordinates representing the position
              of the maximum value in the corresponding heatmap channel.
              Coordinates are 0-indexed integers.
    """

    B, H, W = heatmap.shape[0], heatmap.shape[-2], heatmap.shape[-1]
    coords = []
    for b in range(B):
        batch_coords = []
        for i in range(8):
            heatmap_2d = heatmap[b, i]
            max_idx = torch.argmax(heatmap_2d)
            y = max_idx // W
            x = max_idx % W
            batch_coords.append((x.item(), y.item()))
        coords.append(batch_coords)
    return coords

def draw_border(obj_corners, img, color_list=None):
    """
    Draws 3D bounding box borders on an image by connecting corner points.
        obj_corners (list): List of shape (N, 8, 2), where each element contains 8 (x, y) corner coordinates for a 3D bounding box.
                            - Indices 0-3: Top face corners
                            - Indices 4-7: Bottom face corners
        img (torch.Tensor or numpy.ndarray): Input image of shape (H, W, 3) or (3, H, W), on which the bounding boxes will be drawn.
        color_list (list, optional): List of BGR color tuples for each bounding box. Defaults to green [(0, 255, 0)].
        numpy.ndarray: Image with 3D bounding box borders drawn.
    Raises:
        ValueError: If the input image does not have 3 dimensions.
    Notes:
        - The function connects corners to form the top and bottom faces, as well as vertical edges of the bounding box.
        - Handles both PyTorch tensors and NumPy arrays as input images.
    """
    
    if isinstance(img, torch.Tensor):
        img = img.detach().cpu().numpy()

    if img.ndim != 3:
        raise ValueError(f"Expected image with 3 dimensions, got shape {img.shape}")

    if img.shape[0] in (1, 3) and img.shape[-1] not in (1, 3):
        img = np.transpose(img, (1, 2, 0))

    img = np.ascontiguousarray(img)

    if not color_list:
        color_list = [(0, 255, 0)]

    for idx, corners in enumerate(obj_corners):
        # corners is a (8, 2) list containing 8 3D bounding box
        # corner points' cordinates for one object
        # Draw lines connecting the 8 corners to form a 3D bounding box
        # Order: 0-1-2-3-0 (top face), 4-5-6-7-4 (bottom face), 0-4, 1-5, 2-6, 3-7 (vertical edges)
        edges = [(0, 1), (1, 2), (2, 3), (3, 0),  # top face
                 (4, 5), (5, 6), (6, 7), (7, 4),  # bottom face
                 (0, 4), (1, 5), (2, 6), (3, 7)]  # vertical edges

        color = tuple(map(int, color_list[idx % len(color_list)]))

        for edge in edges:
            pt1 = tuple(map(int, corners[edge[0]]))
            pt2 = tuple(map(int, corners[edge[1]]))
            cv2.line(img, pt1, pt2, color, 2)
    
    return img

def bbox_to_crop_xyhw(bbox, img_h, img_w):
    x, y, w, h = bbox

    x0 = max(0, int(np.floor(x)))
    y0 = max(0, int(np.floor(y)))
    x1 = min(img_w, int(np.ceil(x + w)))
    y1 = min(img_h, int(np.ceil(y + h)))

    if x1 <= x0 or y1 <= y0:
        return None

    return x0, y0, y1 - y0, x1 - x0

def preprocess_coco_image(rgb, bbox):
    if rgb.ndim != 3 or rgb.shape[-1] < 3:
        return None, None

    img_tensor = torch.from_numpy(rgb[..., :3]).permute(2, 0, 1).float()
    heatmap_dummy = torch.zeros((8, rgb.shape[0], rgb.shape[1]), dtype=torch.float32)
    coords_dummy = torch.zeros((8, 2), dtype=torch.float32)

    img_scaled, _, _, meta = gen_scaled_data(
        img_tensor,
        heatmap_dummy,
        coords_dummy,
        bbox=bbox,
        return_meta=True,
    )

    img_scaled = _normalize_image_tensor(img_scaled)

    return img_scaled, meta


def crop_coords_to_original(coords, crop_meta, img_shape):
    if crop_meta is None:
        return None

    scale = float(crop_meta['scale'])
    crop_x = float(crop_meta['crop_x'])
    crop_y = float(crop_meta['crop_y'])
    img_h, img_w = img_shape[:2]

    corners_on_original = []
    for u, v in coords:
        u_original = int(round((float(u) + crop_x) / scale))
        v_original = int(round((float(v) + crop_y) / scale))
        u_original = min(max(u_original, 0), img_w - 1)
        v_original = min(max(v_original, 0), img_h - 1)
        corners_on_original.append((u_original, v_original))

    return corners_on_original


def denormalize_for_visualization(img_tensor):
    if isinstance(img_tensor, torch.Tensor):
        return _denormalize_image_tensor(img_tensor)
    return img_tensor


def coco_bbox_to_xyxy(bbox):
    x, y, w, h = bbox
    return [float(x), float(y), float(x + w), float(y + h)]


def bbox_iou_xyxy(box_a, box_b):
    ax0, ay0, ax1, ay1 = box_a
    bx0, by0, bx1, by1 = box_b

    ix0 = max(ax0, bx0)
    iy0 = max(ay0, by0)
    ix1 = min(ax1, bx1)
    iy1 = min(ay1, by1)

    iw = max(0.0, ix1 - ix0)
    ih = max(0.0, iy1 - iy0)
    inter = iw * ih

    area_a = max(0.0, ax1 - ax0) * max(0.0, ay1 - ay0)
    area_b = max(0.0, bx1 - bx0) * max(0.0, by1 - by0)
    union = area_a + area_b - inter
    if union <= 0:
        return 0.0
    return inter / union


def bbox_center_xyxy(box):
    x0, y0, x1, y1 = box
    return (0.5 * (x0 + x1), 0.5 * (y0 + y1))


def bbox_diag_xyxy(box):
    x0, y0, x1, y1 = box
    w = max(0.0, x1 - x0)
    h = max(0.0, y1 - y0)
    return float(np.hypot(w, h))


def match_bboxes_to_prev_tracks(current_bboxes_xyxy, prev_tracks, iou_thr=0.3, center_thr=1.2):
    """
    Frame-level one-to-one matching for stable temporal track IDs.
    Stage 1: greedy IoU matching.
    Stage 2: center-distance fallback for still-unmatched items.
    """

    n_cur = len(current_bboxes_xyxy)
    n_prev = len(prev_tracks)
    matched_prev_ids = [None] * n_cur
    if n_cur == 0 or n_prev == 0:
        return matched_prev_ids

    iou_pairs = []
    for cur_idx, cur_box in enumerate(current_bboxes_xyxy):
        for prev_idx, track in enumerate(prev_tracks):
            prev_box = track.get('bbox_xyxy')
            if prev_box is None:
                continue
            iou = bbox_iou_xyxy(cur_box, prev_box)
            if iou >= iou_thr:
                iou_pairs.append((iou, cur_idx, prev_idx))

    iou_pairs.sort(key=lambda x: x[0], reverse=True)
    used_cur = set()
    used_prev = set()
    for _, cur_idx, prev_idx in iou_pairs:
        if cur_idx in used_cur or prev_idx in used_prev:
            continue
        matched_prev_ids[cur_idx] = prev_idx
        used_cur.add(cur_idx)
        used_prev.add(prev_idx)

    remain_cur = [i for i in range(n_cur) if i not in used_cur]
    remain_prev = [j for j in range(n_prev) if j not in used_prev]
    if not remain_cur or not remain_prev:
        return matched_prev_ids

    center_pairs = []
    for cur_idx in remain_cur:
        cur_box = current_bboxes_xyxy[cur_idx]
        cur_cx, cur_cy = bbox_center_xyxy(cur_box)
        cur_diag = max(1e-6, bbox_diag_xyxy(cur_box))

        for prev_idx in remain_prev:
            prev_box = prev_tracks[prev_idx].get('bbox_xyxy')
            if prev_box is None:
                continue
            prev_cx, prev_cy = bbox_center_xyxy(prev_box)
            prev_diag = max(1e-6, bbox_diag_xyxy(prev_box))

            dist = float(np.hypot(cur_cx - prev_cx, cur_cy - prev_cy))
            norm = max(cur_diag, prev_diag)
            norm_dist = dist / norm
            if norm_dist <= center_thr:
                center_pairs.append((norm_dist, cur_idx, prev_idx))

    center_pairs.sort(key=lambda x: x[0])
    for _, cur_idx, prev_idx in center_pairs:
        if matched_prev_ids[cur_idx] is not None:
            continue
        if prev_idx in used_prev:
            continue
        matched_prev_ids[cur_idx] = prev_idx
        used_prev.add(prev_idx)

    return matched_prev_ids


def parse_video_frame_from_filename(file_name, image_id=None):
    """
    Infer video key and frame index from COCO image file name.
    """

    norm_name = str(file_name).replace('\\', '/')
    slash_idx = norm_name.rfind('/')
    video_key = norm_name[:slash_idx] if slash_idx >= 0 else '__root__'

    stem = norm_name[slash_idx + 1:]
    dot_idx = stem.rfind('.')
    stem = stem if dot_idx < 0 else stem[:dot_idx]

    m = re.search(r'(\d+)$', stem)
    frame_idx = int(m.group(1)) if m else (int(image_id) if image_id is not None else 0)
    return video_key, frame_idx


def get_sorted_image_ids_for_temporal_coco(coco):
    """
    Return image ids sorted by (video_key, frame_index, file_name).
    """

    records = []
    for img in coco.get('images', []):
        image_id = img['id']
        file_name = img.get('file_name', str(image_id))
        video_key, frame_idx = parse_video_frame_from_filename(file_name, image_id=image_id)
        records.append((video_key, frame_idx, file_name, image_id))

    records.sort(key=lambda r: (r[0], r[1], r[2]))
    return [r[3] for r in records]

