from src.config import cfg, args
import torch

encoder_path = cfg["encoder_path"]

def make_encoder():
    """
    Return a DINOv2 ViT encoder
    """
    encoder = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14_reg').cuda()
    checkpoint = torch.load(encoder_path)
    encoder.load_state_dict(checkpoint)
    return encoder