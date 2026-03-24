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

        with open(coco_path, 'r') as f:
            coco = json.load(f)

        image_by_id = {img['id']: img for img in coco.get('images', [])}
        ann_by_image = defaultdict(list)
        for ann in coco.get('annotations', []):
            if int(ann.get('iscrowd', 0)) != 0:
                continue
            ann_by_image[ann['image_id']].append(ann)

        for image_id, anns in tqdm.tqdm(ann_by_image.items(), desc='coco-eval'):
            image_meta = image_by_id.get(image_id)
            if image_meta is None:
                continue

            img_path = os.path.join(frame_dir, image_meta['file_name'])
            if not os.path.exists(img_path):
                continue

            rgb_original = iio.imread(img_path)
            if rgb_original.ndim == 2:
                rgb_original = np.stack([rgb_original] * 3, axis=-1)
            if rgb_original.shape[-1] == 4:
                rgb_original = rgb_original[..., :3]

            pred_corners_list = []
            for ann in anns:
                img_scaled, crop = preprocess_coco_image(rgb_original, ann['bbox'])
                if img_scaled is None:
                    continue

                x, y, h, w = crop
                x_model = img_scaled.unsqueeze(0).to(self.device)

                with torch.no_grad():
                    pred_heatmap = self.model(x_model)

                corners_pred = gen_coords(heatmap=pred_heatmap)[0]

                scale_x = w / float(cfg['width'])
                scale_y = h / float(cfg['height'])

                corners_on_original = []
                for u, v in corners_pred:
                    u_original = int(round(x + u * scale_x))
                    v_original = int(round(y + v * scale_y))
                    corners_on_original.append((u_original, v_original))

                pred_corners_list.append(corners_on_original)

            if not pred_corners_list:
                continue

            result = draw_border(
                obj_corners=pred_corners_list,
                img=rgb_original.copy(),
                color_list=[
                    (0, 0, 255),
                    (0, 255, 255),
                    (255, 0, 0),
                    (0, 255, 0)
                ]
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