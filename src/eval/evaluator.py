from src.config import cfg, args, get_device
from .utils import *
from src.models import make_network
import torch
import numpy as np
import imageio.v3 as iio
import tqdm

class IOPENEvaluator:
    def __init__(self, device=None):
        self.eval_cfg = cfg['eval']
        self.device = device or get_device()
        self.model = make_network().to(self.device)
        model_state = torch.load(self.eval_cfg['model_path'], map_location=self.device)
        if isinstance(model_state, dict) and 'model_state' in model_state:
            model_state = model_state['model_state']
        self.model.load_state_dict(model_state)

    def inference(self, batch):
        x = batch['img'].to(self.device)
        pred = self.model(x) # (B, 8, H, W)
        corners = gen_coords(heatmap=pred) # (B, 8, 2) list
        return corners
    
    def evaluate(self, batch):
        output_dir = self.eval_cfg['output_dir']

        corners_gt = batch['coords'].int().tolist()
        corners_pred = self.inference(batch)

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

def make_evaluator():
    evaluator = IOPENEvaluator()
    return evaluator