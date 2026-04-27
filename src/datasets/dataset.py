import torch
from torch.utils.data import Dataset, DataLoader
import imageio.v3 as iio
import numpy as np
from .utils import *
from src.config import cfg, args

class IOPENDataset(Dataset):
    def __init__(self, data_root, split='train'):
        dataset_cfg = cfg.get('dataset', {})
        
        # Support both old (scene_split) and new (num_train/num_val) formats
        scene_split_cfg = dataset_cfg.get('scene_split', {})
        if scene_split_cfg:
            scene_ids = scene_split_cfg.get(split)
        else:
            # Use num_train and num_val to generate scene_ids
            num_train = dataset_cfg.get('num_train', 8)
            num_val = dataset_cfg.get('num_val', 2)
            if split == 'train':
                scene_ids = list(range(num_train))
            else:  # validate
                scene_ids = list(range(num_train, num_train + num_val))
        
        if scene_ids is None:
            # Backward-compatible fallback: use the first num_scene scenes.
            scene_ids = list(range(dataset_cfg['num_scene']))

        self.data_dict = load_data(data_root, scene_ids=scene_ids, img_per_scene=1000)
        self.use_mask = cfg['dataset']['use_mask']
    
    def __len__(self):
        return len(self.data_dict['samples']['rgb_path'])
    
    def __getitem__(self, index):
        pbr_root = self.data_dict['pbr_root']
        model = self.data_dict['model']
        camera = self.data_dict['camera']

        rgb_path = self.data_dict['samples']['rgb_path'][index]
        rgb = iio.imread(pbr_root + rgb_path)

        cam_R_m2c = np.array(self.data_dict['samples']['cam_R_m2c'][index], dtype=np.float32).reshape(3, 3)
        cam_t_m2c = np.array(self.data_dict['samples']['cam_t_m2c'][index], dtype=np.float32).reshape(3, 1)

        if self.use_mask:
            mask_path = self.data_dict['samples']['mask_path'][index]
            mask = iio.imread(pbr_root + mask_path)
            img_original = gen_masked_img(rgb, mask) # (H, W, 3) np array
        else:
            img_original = rgb
        
        heatmap_original, coords_original, bbox = gen_gt(
            camera, model, cam_R_m2c, cam_t_m2c
        ) # (8, H, W) & (8, 2) np array

        img_original = torch.from_numpy(img_original).permute(2, 0, 1).float()
        heatmap_original = torch.from_numpy(heatmap_original).float()
        coords_original = torch.from_numpy(coords_original).float()

        img_scaled, heatmap_scaled, coords_scaled = gen_scaled_data(
            img_original, heatmap_original, coords_original, bbox=bbox
        )

        # DINOv2 backbone expects normalized RGB input in [0,1] with ImageNet stats.
        img_scaled = img_scaled / 255.0
        norm_cfg = cfg.get('dataset', {}).get('normalize', {})
        if norm_cfg.get('enabled', True):
            mean_vals = norm_cfg.get('mean', [0.485, 0.456, 0.406])
            std_vals = norm_cfg.get('std', [0.229, 0.224, 0.225])
            mean = torch.tensor(mean_vals, dtype=img_scaled.dtype).view(3, 1, 1)
            std = torch.tensor(std_vals, dtype=img_scaled.dtype).view(3, 1, 1).clamp_min(1e-6)
            img_scaled = (img_scaled - mean) / std

        return {
            'img': img_scaled,
            'heatmap': heatmap_scaled,
            'coords': coords_scaled
        }

def make_dataset(split='train'):
    """
    Create and return the IOPEN dataset.
    """
    data_root = cfg['train']['dataset_path']
    dataset = IOPENDataset(data_root, split=split)
    return dataset

def make_dataloader(split='train', shuffle=None, batch_size=None):
    """
    Create and return the IOPEN dataloader.
    """
    dataset = make_dataset(split=split)
    if batch_size is None:
        batch_size = cfg['train']['batch_size']
    if shuffle is None:
        shuffle = (split == 'train')
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader