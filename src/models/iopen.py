import torch
import torch.nn as nn
import torch.nn.functional as F
from src.config import cfg, args
from .encodor import make_encoder

class IOPEN(nn.Module):
    def __init__(self):
        super().__init__()

        self.p = cfg['patch']
        self.H, self.W = cfg['height'], cfg['width']
        self.N = self.H * self.W // (self.p ** 2) # N = H * W / p^2

        self.vit = make_encoder()
        self.query_embed = nn.Embedding(self.N, 768)
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
            d_model=768,
            nhead=8,
            dim_feedforward=2048
            ),
            num_layers=12
        )
        self.fc = nn.Linear(768, 8 * self.p * self.p)
    
    def forward(self, x):
        """
        Forward pass of the IOPEN model.
        Args:
            x (torch.Tensor): Input image tensor of shape (B, 3, H, W) where:
                - B: Batch size
                - 3: RGB channels
                - H: Image height
                - W: Image width
        Returns:
            torch.Tensor: Output tensor of shape (B, 8, H, W) containing:
                - B: Batch size
                - 8: Number of output channels
                - H: Height (same as input)
                - W: Width (same as input)
        Process:
            1. ViT Encoding: Extracts patch tokens from input image using Vision Transformer
            2. Decoder: Applies transformer decoder with learned query embeddings to memory
            3. Linear Layer: Projects decoder output to patch space (8p^2 values per patch)
            4. Unpatchify: Reshapes and rearranges output back to image format (B, 8, H, W)
        """

        B = x.shape[0]
        H, W = self.H, self.W
        # 1. ViT
        memory = self.vit.forward_features(x)["x_norm_patchtokens"]  # (B, N, 384)
        # 2. decoder
        query = self.query_embed.weight.unsqueeze(1).expand(-1, B, -1)  # (N, B, 384)
        decoder_out = self.decoder(query, memory.permute(1, 0, 2)).permute(1, 0, 2)  # (B, N, 384)
        # 3. Linear
        out = self.fc(decoder_out) # (B, N, 8p^2)
        # 4. Unpatchify
        out = out.reshape(B, self.H // self.p, self.W // self.p, self.p, self.p, 8)
        out = out.permute(0, 5, 1, 3, 2, 4).contiguous()
        out = out.reshape(B, 8, H, W)
        
        return out

def make_network():
    """
    Create and return the IOPEN network.
    """
    network = IOPEN()
    return network