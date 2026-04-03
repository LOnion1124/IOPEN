import torch
import torch.nn.functional as F


def _normalize_coords(coords, height, width):
    """
    Normalize pixel coordinates to [0, 1] range.
    Args:
        coords: (B, C, 2) tensor in pixel coordinates
        height: heatmap height
        width: heatmap width
    Returns:
        normalized coords tensor with the same shape
    """
    coords_norm = coords.clone()
    denom_x = max(width - 1, 1)
    denom_y = max(height - 1, 1)
    coords_norm[..., 0] = coords_norm[..., 0] / denom_x
    coords_norm[..., 1] = coords_norm[..., 1] / denom_y
    return coords_norm

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

def get_loss(pred, gt, lambda_weight=1.0, temperature=0.05, alpha=50.0, use_adaptive_weight=True):
    """
    Combined loss for heatmap-based corner detection with foreground weighting
    Args:
        pred: (B, 8, H, W) predicted heatmap logits (before sigmoid)
        gt: dict with 'heatmap' (B, 8, H, W) [0,1] and 'coords' (B, 8, 2) normalized [0,1]
        lambda_weight: weight for fine coordinate loss (used as target ratio if adaptive)
        temperature: softmax temperature for soft-argmax (smaller = sharper)
        alpha: foreground weight multiplier (higher = more focus on peaks)
        use_adaptive_weight: if True, auto-balance loss scales; lambda_weight becomes target ratio
    Returns:
        loss: scalar total loss
        loss_coarse: scalar weighted BCE heatmap loss
        loss_fine: scalar coordinate loss
    """
    gt_heatmap, gt_coords = gt['heatmap'], gt['coords']
    _, _, H, W = pred.shape
    
    # Coarse loss: Foreground-weighted BCE on heatmaps
    # Weight: w = 1 + alpha * gt (higher weight on Gaussian peaks)
    pos_weight = 1.0 + alpha * gt_heatmap  # (B, 8, H, W)
    
    # Binary cross-entropy with logits (numerically stable)
    bce_loss = F.binary_cross_entropy_with_logits(pred, gt_heatmap, reduction='none')
    
    # Apply foreground weighting
    weighted_bce = bce_loss * pos_weight
    loss_coarse = weighted_bce.mean()
    
    # Fine loss: differentiable coordinate loss using soft-argmax on logits.
    # Using logits preserves contrast; sigmoid can flatten weak early predictions and
    # bias soft-argmax toward image center.
    pred_coords = soft_argmax_2d(pred, temperature)  # (B, 8, 2)

    # Mask invalid GT points and normalize valid coordinates to [0, 1].
    valid = (
        (gt_coords[..., 0] >= 0.0) & (gt_coords[..., 0] < W) &
        (gt_coords[..., 1] >= 0.0) & (gt_coords[..., 1] < H)
    )
    gt_coords_norm = _normalize_coords(gt_coords, H, W)

    if valid.any():
        loss_fine = F.smooth_l1_loss(pred_coords[valid], gt_coords_norm[valid])
    else:
        loss_fine = pred.new_zeros(())
    
    # Adaptive weight balancing: auto-adjust to keep losses at similar scale
    if use_adaptive_weight and valid.any():
        # Compute adaptive lambda to maintain target ratio
        # Target: loss_fine_weighted / loss_coarse = lambda_weight
        # So: adaptive_lambda = lambda_weight * loss_coarse / (loss_fine + 1e-8)
        with torch.no_grad():
            adaptive_lambda = lambda_weight * loss_coarse / (loss_fine + 1e-8)
            # Clip to reasonable range to avoid instability
            adaptive_lambda = torch.clamp(adaptive_lambda, 0.001, 100.0)
        loss = loss_coarse + adaptive_lambda * loss_fine
    else:
        # Fixed weight (original behavior)
        loss = loss_coarse + lambda_weight * loss_fine
    
    return loss, loss_coarse, loss_fine