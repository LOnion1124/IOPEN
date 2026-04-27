from src.config import cfg, args
import torch

encoder_path = cfg["encoder_path"]

def make_encoder():
    """
    Return a DINOv2 ViT encoder
    """
    scale = cfg.get("model_scale", "small")
    if scale == "small":
        model_name = "dinov2_vits14_reg"
    elif scale == "base":
        model_name = "dinov2_vitb14_reg"
    else:
        raise ValueError(f"Unsupported model_scale: {scale}. Expected 'small' or 'base'.")

    encoder = torch.hub.load('facebookresearch/dinov2', model_name)
    checkpoint = torch.load(encoder_path, map_location='cpu')
    encoder.load_state_dict(checkpoint)
    return encoder