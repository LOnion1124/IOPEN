import torch
import cv2
import numpy as np
import imageio.v3 as iio
import re
from src.config import cfg, args
from src.datasets.utils import gen_scaled_data

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
    H, W = rgb.shape[:2]
    crop = bbox_to_crop_xyhw(bbox, H, W)
    if crop is None:
        return None, None

    x, y, h, w = crop
    img_cropped = rgb[y:y+h, x:x+w]
    if img_cropped.size == 0:
        return None, None

    img_cropped = torch.from_numpy(img_cropped).permute(2, 0, 1).float()
    heatmap_dummy = torch.zeros((8, h, w), dtype=torch.float32)
    coords_dummy = torch.zeros((8, 2), dtype=torch.float32)

    img_scaled, _, _ = gen_scaled_data(
        img_cropped,
        heatmap_dummy,
        coords_dummy
    )

    return img_scaled, crop


def smooth_heatmap_ema(current_heatmap, prev_heatmap, alpha=0.7):
    """
    Apply EMA smoothing on heatmaps.
    H_smooth = alpha * H_t + (1 - alpha) * H_prev
    """

    if prev_heatmap is None:
        return current_heatmap

    if not isinstance(prev_heatmap, torch.Tensor):
        prev_heatmap = torch.as_tensor(prev_heatmap)

    prev_heatmap = prev_heatmap.to(
        device=current_heatmap.device,
        dtype=current_heatmap.dtype
    )
    return alpha * current_heatmap + (1.0 - alpha) * prev_heatmap


def smooth_corners_ema(current_corners, prev_corners, alpha=0.7):
    """
    Apply EMA smoothing on corner coordinates to reduce jitter.
    corners_smooth = alpha * corners_t + (1 - alpha) * corners_prev
    
    Args:
        current_corners: list of (u, v) tuples for current frame
        prev_corners: list of (u, v) tuples from previous frame, or None
        alpha: EMA weight for current frame (0.0-1.0), higher = more weight on current
    
    Returns:
        list of smoothed (u, v) tuples
    """
    if prev_corners is None or len(prev_corners) == 0:
        return current_corners
    
    if len(current_corners) != len(prev_corners):
        return current_corners
    
    smoothed = []
    for (cu, cv), (pu, pv) in zip(current_corners, prev_corners):
        u_smooth = alpha * cu + (1.0 - alpha) * pu
        v_smooth = alpha * cv + (1.0 - alpha) * pv
        smoothed.append((u_smooth, v_smooth))
    
    return smoothed


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


def match_prev_track_by_iou(current_bbox_xyxy, prev_tracks, used_track_ids, iou_thr=0.3):
    """
    Greedy matching of current bbox to previous-frame tracks by IoU.
    """

    best_id = None
    best_iou = iou_thr
    for track_id, track in enumerate(prev_tracks):
        if track_id in used_track_ids:
            continue
        prev_bbox = track.get('bbox_xyxy')
        if prev_bbox is None:
            continue
        iou = bbox_iou_xyxy(current_bbox_xyxy, prev_bbox)
        if iou > best_iou:
            best_iou = iou
            best_id = track_id
    return best_id


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


def _scaled_camera_matrix_for_image(img_h, img_w, camera_cfg=None):
    if camera_cfg is None:
        camera_cfg = cfg.get('cam')
    if camera_cfg is None:
        return None

    fx = float(camera_cfg.get('fx', 0.0))
    fy = float(camera_cfg.get('fy', 0.0))
    cx = float(camera_cfg.get('cx', 0.0))
    cy = float(camera_cfg.get('cy', 0.0))
    base_w = float(camera_cfg.get('width', img_w))
    base_h = float(camera_cfg.get('height', img_h))

    if fx <= 0 or fy <= 0 or base_w <= 0 or base_h <= 0:
        return None

    sx = float(img_w) / base_w
    sy = float(img_h) / base_h

    k = np.array([
        [fx * sx, 0.0, cx * sx],
        [0.0, fy * sy, cy * sy],
        [0.0, 0.0, 1.0],
    ], dtype=np.float32)
    return k


def _canonical_box_points_3d(dims_xyz=None):
    """
    Canonical 8 corners (matching 0-1-2-3 top, 4-5-6-7 bottom).
    """

    if dims_xyz is None:
        dims_xyz = (1.0, 1.0, 1.0)
    dx, dy, dz = [float(v) for v in dims_xyz]
    hx, hy, hz = 0.5 * dx, 0.5 * dy, 0.5 * dz

    return np.array([
        [-hx, -hy, hz],
        [hx, -hy, hz],
        [hx, hy, hz],
        [-hx, hy, hz],
        [-hx, -hy, -hz],
        [hx, -hy, -hz],
        [hx, hy, -hz],
        [-hx, hy, -hz],
    ], dtype=np.float32)


def _select_outer_corner_mask(img_pts, inner_ignore_quantile=0.25, min_keep=4):
    """
    Keep corners farther from center and drop overly inner points.
    """

    n = img_pts.shape[0]
    if n == 0 or inner_ignore_quantile <= 0:
        return np.ones((n,), dtype=bool)

    center = np.mean(img_pts, axis=0)
    radii = np.linalg.norm(img_pts - center, axis=1)
    q = float(np.quantile(radii, inner_ignore_quantile))
    keep = radii >= q

    if int(np.count_nonzero(keep)) < int(min_keep):
        keep = np.zeros((n,), dtype=bool)
        keep_idx = np.argsort(radii)[-int(min_keep):]
        keep[keep_idx] = True
    return keep


def refine_corners_with_rigid_box(
    corners_2d,
    img_shape,
    camera_cfg=None,
    clip_xyxy=None,
    dims_xyz=(1.0, 1.0, 1.0),
    inner_ignore_quantile=0.25,
    enclose_scale=1.05,
):
    """
    Fit a rigid 3D box via PnP (least-squares) and reproject 8 corners to 2D.
    """

    if corners_2d is None or len(corners_2d) != 8:
        return corners_2d

    img_h, img_w = img_shape[:2]
    k = _scaled_camera_matrix_for_image(img_h=img_h, img_w=img_w, camera_cfg=camera_cfg)
    if k is None:
        return corners_2d

    obj_pts = _canonical_box_points_3d(dims_xyz=dims_xyz)
    img_pts = np.asarray(corners_2d, dtype=np.float32).reshape(-1, 2)
    dist = np.zeros((4, 1), dtype=np.float32)

    keep_mask = _select_outer_corner_mask(
        img_pts=img_pts,
        inner_ignore_quantile=inner_ignore_quantile,
        min_keep=4
    )
    obj_fit = obj_pts[keep_mask]
    img_fit = img_pts[keep_mask]

    ok, rvec, tvec = cv2.solvePnP(
        objectPoints=obj_fit,
        imagePoints=img_fit,
        cameraMatrix=k,
        distCoeffs=dist,
        flags=cv2.SOLVEPNP_EPNP
    )
    if not ok:
        return corners_2d

    if obj_fit.shape[0] >= 6:
        ok_refine, rvec, tvec = cv2.solvePnP(
            objectPoints=obj_fit,
            imagePoints=img_fit,
            cameraMatrix=k,
            distCoeffs=dist,
            rvec=rvec,
            tvec=tvec,
            useExtrinsicGuess=True,
            flags=cv2.SOLVEPNP_ITERATIVE
        )
        if not ok_refine:
            return corners_2d

    reproj_pts, _ = cv2.projectPoints(obj_pts, rvec, tvec, k, dist)
    reproj_pts = reproj_pts.reshape(-1, 2)

    if float(enclose_scale) > 1.0:
        center = np.mean(reproj_pts, axis=0)
        reproj_pts = center + float(enclose_scale) * (reproj_pts - center)

    clip_x0 = 0
    clip_y0 = 0
    clip_x1 = img_w - 1
    clip_y1 = img_h - 1
    if clip_xyxy is not None:
        x0, y0, x1, y1 = clip_xyxy
        clip_x0 = max(clip_x0, int(np.floor(x0)))
        clip_y0 = max(clip_y0, int(np.floor(y0)))
        clip_x1 = min(clip_x1, int(np.ceil(x1)))
        clip_y1 = min(clip_y1, int(np.ceil(y1)))

    if clip_x1 < clip_x0:
        clip_x1 = clip_x0
    if clip_y1 < clip_y0:
        clip_y1 = clip_y0

    refined = []
    for u, v in reproj_pts:
        u = int(round(float(u)))
        v = int(round(float(v)))
        u = min(max(u, clip_x0), clip_x1)
        v = min(max(v, clip_y0), clip_y1)
        refined.append((u, v))
    return refined