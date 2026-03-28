import torch
from torch.utils.data import Dataset, DataLoader
import imageio.v3 as iio
import numpy as np
from .utils import *
from src.config import cfg, args

class IOPENDataset(Dataset):
    def __init__(self, data_root):
        self.data_dict = load_data(data_root, num_scene=cfg['dataset']['num_scene'], img_per_scene=1000)
        self.use_mask = cfg['dataset']['use_mask']
    
    def __len__(self):
        return len(self.data_dict['samples']['rgb_path'])
    
    def __getitem__(self, index):
        pbr_root = self.data_dict['pbr_root']
        model = self.data_dict['model']
        camera = self.data_dict['camera']

        rgb_path = self.data_dict['samples']['rgb_path'][index]
        rgb = iio.imread(pbr_root + rgb_path)
        mask_path = self.data_dict['samples']['mask_path'][index]
        mask = iio.imread(pbr_root + mask_path)

        cam_R_m2c = np.array(self.data_dict['samples']['cam_R_m2c'][index], dtype=np.float32).reshape(3, 3)
        cam_t_m2c = np.array(self.data_dict['samples']['cam_t_m2c'][index], dtype=np.float32).reshape(3, 1)

        if self.use_mask:
            img_original = gen_masked_img(rgb, mask) # (H, W, 3) np array
        else:
            img_original = rgb
        
        heatmap_original, coords_original, bbox = gen_gt(
            camera, model, cam_R_m2c, cam_t_m2c
        ) # (8, H, W) & (8, 2) np array
        
        img_cropped, heatmap_cropped, coords_cropped = gen_cropped_data(
            img_original, heatmap_original, coords_original, bbox
        ) # (H', W', 3), (8, H', W') * (8, 2) np array

        img_cropped = torch.from_numpy(img_cropped).permute(2, 0, 1).float()
        heatmap_cropped = torch.from_numpy(heatmap_cropped).float()
        coords_cropped = torch.from_numpy(coords_cropped).float()

        img_scaled, heatmap_scaled, coords_scaled = gen_scaled_data(
            img_cropped, heatmap_cropped, coords_cropped
        )

        return {
            'img': img_scaled,
            'heatmap': heatmap_scaled,
            'coords': coords_scaled
        }

def make_dataset():
    """
    Create and return the IOPEN dataset.
    """
    data_root = cfg['train']['dataset_path']
    dataset = IOPENDataset(data_root)
    return dataset

def make_dataloader():
    """
    Create and return the IOPEN dataloader.
    """
    dataset = make_dataset()
    B = cfg['train']['batch_size']
    dataloader = DataLoader(dataset, batch_size=B, shuffle=True)
    return dataloader