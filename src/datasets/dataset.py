import torch
from torch.utils.data import Dataset, DataLoader
import imageio.v3 as iio
import numpy as np
from .utils import *
from src.config import cfg, args

class IOPENDataset(Dataset):
    def __init__(self, data_root):
        self.data_dict = load_data(data_root, num_scene=6, img_per_scene=1000)
    
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

        # Generate masked image
        img_original = gen_masked_img(rgb, mask)
        
        # Preprocess image for DINOv2 (resize to patch_size multiples)
        original_h, original_w = img_original.shape[:2]
        img_dinov2 = preprocess_image_for_dinov2(img_original, patch_size=14)
        new_h, new_w = img_dinov2.shape[:2]
        
        # Adjust camera parameters to match resized image
        camera_adjusted = adjust_camera_params(camera, original_h, original_w, new_h, new_w)
        
        # Generate heatmap with adjusted camera parameters
        heatmap = gen_heatmap(camera_adjusted, model, cam_R_m2c, cam_t_m2c)

        img_dinov2 = torch.from_numpy(img_dinov2).permute(2, 0, 1).float()
        heatmap = torch.from_numpy(heatmap).float()

        return {'img': img_dinov2, 'heatmap': heatmap}

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