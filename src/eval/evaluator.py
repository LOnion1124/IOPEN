from src.config import cfg, args, get_device
from .utils import *
from .utils import _build_resize_crop_meta
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
        # palette = [
        #     (0, 255, 255),
        #     (0, 0, 255),
        #     (255, 0, 0),
        #     (0, 255, 0)
        # ]
        palette = [
            (0, 255, 255)
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
                iou_thr=keyframe_match_iou
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
                    img_scaled, crop_meta = preprocess_coco_image(rgb_original, ann['bbox'])
                    if img_scaled is None:
                        continue
                    x_model = img_scaled.unsqueeze(0).to(self.device)

                    with torch.no_grad():
                        pred_heatmap = self.model(x_model)

                    if prev_track_id is not None:
                        track_uid = prev_tracks[prev_track_id].get('track_uid', 0)
                    else:
                        track_uid = next_track_uid
                        next_track_uid += 1

                    corners_input = gen_coords(heatmap=pred_heatmap)[0]
                    corners_on_original = crop_coords_to_original(
                        corners_input,
                        crop_meta,
                        rgb_original.shape,
                    )

                    pred_corners_list.append(corners_on_original)
                    draw_color_list.append(palette[track_uid % len(palette)])
                    current_tracks.append({
                        'bbox_xyxy': bbox_xyxy,
                        'track_uid': track_uid,
                        'corners_input': corners_input,
                        'crop_meta': crop_meta,
                        'corners_on_original': corners_on_original,
                    })
            else:
                for ann, bbox_xyxy, prev_track_id in zip(anns, ann_bboxes_xyxy, ann_prev_matches):
                    if prev_track_id is None:
                        continue

                    track = prev_tracks[prev_track_id]
                    corners_input = track.get('corners_input')
                    if corners_input is None:
                        continue

                    crop_meta = _build_resize_crop_meta(
                        rgb_original.shape[0],
                        rgb_original.shape[1],
                        ann['bbox'],
                        cfg['height'],
                        cfg['width'],
                    )
                    if crop_meta is None:
                        continue
                    track_uid = track.get('track_uid', 0)

                    corners_on_original = crop_coords_to_original(
                        corners_input,
                        crop_meta,
                        rgb_original.shape,
                    )

                    pred_corners_list.append(corners_on_original)
                    draw_color_list.append(palette[track_uid % len(palette)])
                    current_tracks.append({
                        'bbox_xyxy': bbox_xyxy,
                        'track_uid': track_uid,
                        'corners_input': corners_input,
                        'crop_meta': crop_meta,
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
    
    def evaluate(self):
        self.inference_coco()

def make_evaluator():
    evaluator = IOPENEvaluator()
    return evaluator