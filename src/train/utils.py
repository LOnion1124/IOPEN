from src.config import cfg, args
import torch
import torch.nn.functional as F

def soft_argmax_2d(heatmap, temperature=1.0):
    """
    Differentiable soft-argmax for 2D heatmaps
    Args:
        heatmap: (B, C, H, W) tensor
        temperature: softmax temperature for sharpness control
    Returns:
        coords: (B, C, 2) tensor of (x, y) coordinates normalized to [0, 1]
    """
    B, C, H, W = heatmap.shape
    
    # Flatten spatial dimensions
    heatmap_flat = heatmap.view(B, C, -1)  # (B, C, H*W)
    
    # Apply softmax to get probability distribution
    weights = F.softmax(heatmap_flat / temperature, dim=2)  # (B, C, H*W)
    
    # Create coordinate grids
    y_coords = torch.arange(H, device=heatmap.device, dtype=heatmap.dtype)
    x_coords = torch.arange(W, device=heatmap.device, dtype=heatmap.dtype)
    
    # Normalize coordinates to [0, 1]
    y_coords = y_coords / (H - 1) if H > 1 else y_coords
    x_coords = x_coords / (W - 1) if W > 1 else x_coords
    
    yy, xx = torch.meshgrid(y_coords, x_coords, indexing='ij')
    coords_grid = torch.stack([xx, yy], dim=0).view(2, -1)  # (2, H*W)
    
    # Compute weighted average of coordinates
    coords = torch.matmul(weights, coords_grid.T)  # (B, C, 2)
    
    return coords

def get_loss(pred, gt, lambda_weight=2.0, temperature=1.0):
    """
    Combined loss for heatmap-based corner detection
    Args:
        pred: (B, 8, H, W) predicted heatmaps
        gt: (B, 8, H, W) ground truth heatmaps
        lambda_weight: weight for fine coordinate loss
        temperature: softmax temperature for soft-argmax
    Returns:
        loss: scalar total loss value
        loss_coarse: scalar coarse heatmap loss
        loss_fine: scalar fine coordinate loss
    """
    # Coarse loss: SmoothL1 on heatmaps
    loss_coarse = F.smooth_l1_loss(pred, gt)
    
    # Fine loss: differentiable coordinate loss using soft-argmax
    pred_coords = soft_argmax_2d(pred, temperature)  # (B, 8, 2)
    gt_coords = soft_argmax_2d(gt, temperature)      # (B, 8, 2)
    
    # Calculate fine loss on normalized coordinates
    loss_fine = F.smooth_l1_loss(pred_coords, gt_coords)
    
    # Total loss with lambda weighting
    loss = loss_coarse + lambda_weight * loss_fine
    
    return loss, loss_coarse, loss_fine

# def train_step(model, optimizer, batch):
#     """
#     Executes a single training step for the model.
#     Performs forward pass, computes loss, and updates model parameters through
#     backpropagation.
#     Args:
#         model: The neural network model to train.
#         optimizer: The optimizer used for model parameter updates.
#         batch (dict): A dictionary containing:
#             - 'img': Input images tensor.
#             - 'heatmap': Ground truth heatmap tensor.
#     Returns:
#         dict: A dictionary containing loss values:
#             - 'total': Total loss value (float).
#             - 'coarse': Coarse-level loss value (float).
#             - 'fine': Fine-level loss value (float).
#     """
#     model.train()
#     img = batch['img']
#     heatmap = batch['heatmap']
#     optimizer.zero_grad()
#     pred_heatmap = model(img)
#     loss, loss_coarse, loss_fine = get_loss(pred=pred_heatmap, gt=heatmap)
#     loss.backward()
#     optimizer.step()
#     return {'total': loss.item, 'coarse': loss_coarse.item(), 'fine':loss_fine.item()}