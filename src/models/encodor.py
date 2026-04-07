from src.config import cfg, args
import torch

encoder_path = cfg["encoder_path"]

def make_encoder():
    """
    Return a DINOv2 ViT encoder
    """
    encoder = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14_reg')
    checkpoint = torch.load(encoder_path, map_location='cpu', weights_only=False)
    encoder.load_state_dict(checkpoint)
    return encoder