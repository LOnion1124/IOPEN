from src.config import cfg, args, get_device
from .utils import *
from src.models import make_network
import torch
import numpy as np
import imageio.v3 as iio
import tqdm
import json
import os
from collections import defaultdict

class IOPENEvaluator:
    def __init__(self, device=None):
        self.eval_cfg = cfg['eval']
        self.device = device or get_device()
        self.model = make_network().to(self.device)
        model_state = torch.load(self.eval_cfg['model_path'], map_location=self.device)
        if isinstance(model_state, dict) and 'model_state' in model_state:
            model_state = model_state['model_state']
        self.model.load_state_dict(model_state)
        self.model.eval()

    def inference_batch(self, batch):
        x = batch['img'].to(self.device)
        with torch.no_grad():
            pred = self.model(x) # (B, 8, H, W)
        corners = gen_coords(heatmap=pred) # (B, 8, 2) list
        return corners

    def inference_coco(self):
        coco_path = self.eval_cfg['coco_path']
        frame_dir = self.eval_cfg['coco_frame_dir']
        output_dir = self.eval_cfg['output_dir']
        os.makedirs(output_dir, exist_ok=True)

        post_cfg = self.eval_cfg.get('postprocess', {})
        ema_cfg = post_cfg.get('ema_heatmap', {})
        box_cfg = post_cfg.get('box_fit', {})
        ema_enabled = bool(ema_cfg.get('enabled', False))
        ema_alpha = float(ema_cfg.get('alpha', 0.7))
        track_iou_thr = float(ema_cfg.get('track_iou_thr', 0.3))
        box_fit_enabled = bool(box_cfg.get('enabled', False))
        raw_box_dims = box_cfg.get('dims_xyz', 'auto')
        box_dims = [1.0, 1.0, 1.0]
        if isinstance(raw_box_dims, (list, tuple)) and len(raw_box_dims) == 3:
            box_dims = [float(raw_box_dims[0]), float(raw_box_dims[1]), float(raw_box_dims[2])]
        elif isinstance(raw_box_dims, str) and raw_box_dims.lower() == 'auto':
            obj_cfg = cfg.get('obj', {}) if isinstance(cfg.get('obj', {}), dict) else {}
            if all(k in obj_cfg for k in ('size_x', 'size_y', 'size_z')):
                box_dims = [float(obj_cfg['size_x']), float(obj_cfg['size_y']), float(obj_cfg['size_z'])]
            elif all(k in obj_cfg for k in ('min_x', 'max_x', 'min_y', 'max_y', 'min_z', 'max_z')):
                box_dims = [
                    float(obj_cfg['max_x']) - float(obj_cfg['min_x']),
                    float(obj_cfg['max_y']) - float(obj_cfg['min_y']),
                    float(obj_cfg['max_z']) - float(obj_cfg['min_z']),
                ]
        box_inner_ignore_q = float(box_cfg.get('inner_ignore_quantile', 0.25))
        box_enclose_scale = float(box_cfg.get('enclose_scale', 1.05))
        smooth_cfg = box_cfg.get('result_smooth', {})
        box_smooth_enabled = bool(smooth_cfg.get('enabled', False))
        box_smooth_alpha = float(smooth_cfg.get('alpha', 0.7))
        key_cfg = self.eval_cfg.get('keyframe_propagation', {})
        keyframe_enabled = bool(key_cfg.get('enabled', False))
        keyframe_interval = max(1, int(key_cfg.get('interval', 5)))
        keyframe_match_iou = float(key_cfg.get('match_iou_thr', 0.1))
        keyframe_match_center = float(key_cfg.get('match_center_thr', 2.0))

        with open(coco_path, 'r') as f:
            coco = json.load(f)

        image_by_id = {img['id']: img for img in coco.get('images', [])}
        ann_by_image = defaultdict(list)
        for ann in coco.get('annotations', []):
            if int(ann.get('iscrowd', 0)) != 0:
                continue
            ann_by_image[ann['image_id']].append(ann)

        sorted_image_ids = get_sorted_image_ids_for_temporal_coco(coco)
        video_states = defaultdict(lambda: {
            'tracks': [],
            'next_track_uid': 0,
            'frame_local_idx': 0,
            'last_key_tracks': [],
        })
        palette = [
            (0, 255, 255),
            (0, 0, 255),
            (255, 0, 0),
            (0, 255, 0)
        ]

        for image_id in tqdm.tqdm(sorted_image_ids, desc='coco-eval'):
            anns = ann_by_image.get(image_id, [])
            if not anns:
                continue

            image_meta = image_by_id.get(image_id)
            if image_meta is None:
                continue

            video_key, _ = parse_video_frame_from_filename(
                image_meta.get('file_name', str(image_id)),
                image_id=image_id
            )
            state = video_states[video_key]
            prev_tracks = state['tracks']
            next_track_uid = state['next_track_uid']
            frame_local_idx = state['frame_local_idx']

            is_keyframe = True
            if keyframe_enabled:
                is_keyframe = (frame_local_idx % keyframe_interval == 0) or (len(state['last_key_tracks']) == 0)
                if not is_keyframe:
                    prev_tracks = state['last_key_tracks']

            anns = sorted(
                anns,
                key=lambda a: (
                    float(a.get('bbox', [0, 0, 0, 0])[1]),
                    float(a.get('bbox', [0, 0, 0, 0])[0])
                )
            )
            ann_bboxes_xyxy = [coco_bbox_to_xyxy(ann['bbox']) for ann in anns]
            ann_prev_matches = match_bboxes_to_prev_tracks(
                current_bboxes_xyxy=ann_bboxes_xyxy,
                prev_tracks=prev_tracks,
                iou_thr=track_iou_thr
            )

            img_path = os.path.join(frame_dir, image_meta['file_name'])
            if not os.path.exists(img_path):
                continue

            rgb_original = iio.imread(img_path)
            if rgb_original.ndim == 2:
                rgb_original = np.stack([rgb_original] * 3, axis=-1)
            if rgb_original.shape[-1] == 4:
                rgb_original = rgb_original[..., :3]

            pred_corners_list = []
            draw_color_list = []
            current_tracks = []
            if is_keyframe:
                for ann, bbox_xyxy, prev_track_id in zip(anns, ann_bboxes_xyxy, ann_prev_matches):
                    img_scaled, crop = preprocess_coco_image(rgb_original, ann['bbox'])
                    if img_scaled is None:
                        continue

                    x, y, h, w = crop
                    x_model = img_scaled.unsqueeze(0).to(self.device)

                    prev_heatmap = None
                    if prev_track_id is not None:
                        prev_heatmap = prev_tracks[prev_track_id].get('heatmap')
                        track_uid = prev_tracks[prev_track_id].get('track_uid', 0)
                    else:
                        track_uid = next_track_uid
                        next_track_uid += 1

                    with torch.no_grad():
                        pred_heatmap = self.model(x_model)

                    if ema_enabled:
                        pred_heatmap = smooth_heatmap_ema(
                            current_heatmap=pred_heatmap,
                            prev_heatmap=prev_heatmap,
                            alpha=ema_alpha
                        )

                    corners_pred = gen_coords(heatmap=pred_heatmap)[0]
                    corners_norm = [
                        (u / float(cfg['width']), v / float(cfg['height']))
                        for u, v in corners_pred
                    ]

                    corners_on_original = []
                    for nu, nv in corners_norm:
                        u_original = int(round(x + nu * w))
                        v_original = int(round(y + nv * h))
                        u_original = min(max(u_original, x), x + w - 1)
                        v_original = min(max(v_original, y), y + h - 1)
                        corners_on_original.append((u_original, v_original))

                    if box_fit_enabled:
                        corners_on_original = refine_corners_with_rigid_box(
                            corners_2d=corners_on_original,
                            img_shape=rgb_original.shape,
                            camera_cfg=cfg.get('cam'),
                            clip_xyxy=(x, y, x + w - 1, y + h - 1),
                            dims_xyz=box_dims,
                            inner_ignore_quantile=box_inner_ignore_q,
                            enclose_scale=box_enclose_scale,
                        )

                    if box_smooth_enabled and box_fit_enabled:
                        prev_corners = None
                        if prev_track_id is not None:
                            prev_corners = prev_tracks[prev_track_id].get('corners_on_original')
                        corners_on_original = smooth_corners_ema(
                            current_corners=corners_on_original,
                            prev_corners=prev_corners,
                            alpha=box_smooth_alpha
                        )

                    pred_corners_list.append(corners_on_original)
                    draw_color_list.append(palette[track_uid % len(palette)])
                    current_tracks.append({
                        'bbox_xyxy': bbox_xyxy,
                        'track_uid': track_uid,
                        'heatmap': pred_heatmap.detach().cpu() if ema_enabled else None,
                        'corners_norm': corners_norm,
                        'corners_on_original': corners_on_original,
                    })
            else:
                for ann, bbox_xyxy, prev_track_id in zip(anns, ann_bboxes_xyxy, ann_prev_matches):
                    if prev_track_id is None:
                        continue

                    track = prev_tracks[prev_track_id]
                    corners_norm = track.get('corners_norm')
                    if corners_norm is None:
                        continue

                    crop = bbox_to_crop_xyhw(ann['bbox'], rgb_original.shape[0], rgb_original.shape[1])
                    if crop is None:
                        continue

                    x, y, h, w = crop
                    track_uid = track.get('track_uid', 0)

                    corners_on_original = []
                    for nu, nv in corners_norm:
                        u_original = int(round(x + nu * w))
                        v_original = int(round(y + nv * h))
                        u_original = min(max(u_original, x), x + w - 1)
                        v_original = min(max(v_original, y), y + h - 1)
                        corners_on_original.append((u_original, v_original))

                    if box_fit_enabled:
                        corners_on_original = refine_corners_with_rigid_box(
                            corners_2d=corners_on_original,
                            img_shape=rgb_original.shape,
                            camera_cfg=cfg.get('cam'),
                            clip_xyxy=(x, y, x + w - 1, y + h - 1),
                            dims_xyz=box_dims,
                            inner_ignore_quantile=box_inner_ignore_q,
                            enclose_scale=box_enclose_scale,
                        )

                    if box_smooth_enabled and box_fit_enabled:
                        prev_corners = track.get('corners_on_original')
                        corners_on_original = smooth_corners_ema(
                            current_corners=corners_on_original,
                            prev_corners=prev_corners,
                            alpha=box_smooth_alpha
                        )

                    pred_corners_list.append(corners_on_original)
                    draw_color_list.append(palette[track_uid % len(palette)])
                    current_tracks.append({
                        'bbox_xyxy': bbox_xyxy,
                        'track_uid': track_uid,
                        'heatmap': track.get('heatmap') if ema_enabled else None,
                        'corners_norm': corners_norm,
                        'corners_on_original': corners_on_original,
                    })

            if not pred_corners_list:
                state['tracks'] = current_tracks
                state['next_track_uid'] = next_track_uid
                state['frame_local_idx'] = frame_local_idx + 1
                continue

            state['tracks'] = current_tracks
            state['next_track_uid'] = next_track_uid
            if is_keyframe:
                state['last_key_tracks'] = current_tracks
            state['frame_local_idx'] = frame_local_idx + 1

            result = draw_border(
                obj_corners=pred_corners_list,
                img=rgb_original.copy(),
                color_list=draw_color_list
            )
            result = np.clip(result, 0, 255).astype(np.uint8, copy=False)

            file_stem = os.path.splitext(os.path.basename(image_meta['file_name']))[0]
            save_path = os.path.join(output_dir, f'{file_stem}.jpg')
            iio.imwrite(save_path, result)
    
    def evaluate(self, mode='batch', batch=None):
        output_dir = self.eval_cfg['output_dir']

        if mode == 'batch':
            corners_gt = batch['coords'].int().tolist()
            corners_pred = self.inference_batch(batch)

            imgs = batch['img'].detach().cpu()
            for idx, img in enumerate(imgs):
                img = img.permute(1, 2, 0).numpy()
                img = np.clip(img, 0, 255).astype(np.uint8, copy=False)
                gt = corners_gt[idx]
                pred = corners_pred[idx]
                result = draw_border(
                    obj_corners=[gt, pred],
                    img=img,
                    color_list=[
                        (0, 255, 0),
                        (0, 0, 255)
                    ]
                )
                result = np.clip(result, 0, 255).astype(np.uint8, copy=False)
                iio.imwrite(f'{output_dir}/{idx:06d}.jpg', result)
        
        elif mode == 'coco':
            self.inference_coco()

def make_evaluator():
    evaluator = IOPENEvaluator()
    return evaluator