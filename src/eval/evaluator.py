from src.config import cfg, args
from .utils import *
from src.models import make_network
import torch
import imageio.v3 as iio
import tqdm

class IOPENEvaluator:
    def __init__(self, device='cuda'):
        self.eval_cfg = cfg['eval']
        self.device = device
        self.model = make_network().to(device)
        model_state = torch.load(self.eval_cfg['model_path'])
        self.model.load_state_dict(model_state)

    def inference(self, imgs):
        x = torch.from_numpy(np.stack(imgs)).permute(0, 3, 1, 2).float().to(self.device)
        pred = self.model(x) # (n, 8, H, W)
        corners = gen_coords(heatmap=pred) # (n, 8, 2) list
        return corners

    def evaluate(self):
        original_dir = self.eval_cfg['original_img_dir']
        masked_dirs = self.eval_cfg['masked_img_dirs']
        output_dir = self.eval_cfg['output_dir']
        frame_cnt = self.eval_cfg['frame_cnt']

        for frame_id in tqdm.tqdm(range(frame_cnt), desc='Evaluating'):
            original_img, masked_imgs = load_frame_src(frame_id, original_dir, masked_dirs)
            corners = self.inference(masked_imgs)
            frame = draw_border(corners, original_img)
            iio.imwrite(f'{output_dir}/{frame_id:06d}.png', frame)

def make_evaluator():
    evaluator = IOPENEvaluator(device='cuda')
    return evaluator