import torch
import torch.nn.functional as F


# Edge definitions for cuboid corners generated in src/datasets/utils.py::gen_gt.
# Corner order:
# 0(-,-,-), 1(+,-,-), 2(+,+,-), 3(-,+,-), 4(-,-,+), 5(+,-,+), 6(+,+,+), 7(-,+,+)
_EDGE_GROUPS = {
    "x": [(0, 1), (3, 2), (4, 5), (7, 6)],
    "y": [(0, 3), (1, 2), (4, 7), (5, 6)],
    "z": [(0, 4), (1, 5), (2, 6), (3, 7)],
}

# At each corner, pick one edge vector per axis (x, y, z) for orthogonality checks.
_CORNER_AXIS_EDGES = [
    ((0, 1), (0, 3), (0, 4)),
    ((1, 0), (1, 2), (1, 5)),
    ((2, 3), (2, 1), (2, 6)),
    ((3, 2), (3, 0), (3, 7)),
    ((4, 5), (4, 7), (4, 0)),
    ((5, 4), (5, 6), (5, 1)),
    ((6, 7), (6, 5), (6, 2)),
    ((7, 6), (7, 4), (7, 3)),
]


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


def _build_edge_vectors(coords):
    """
    Build per-direction edge vectors from predicted 2D corner coordinates.
    Args:
        coords: (B, 8, 2) normalized coordinates
    Returns:
        dict[str, Tensor]: each value has shape (B, 4, 2)
    """
    edge_vecs = {}
    for axis, edges in _EDGE_GROUPS.items():
        vecs = [coords[:, j] - coords[:, i] for i, j in edges]
        edge_vecs[axis] = torch.stack(vecs, dim=1)
    return edge_vecs


def cuboid_geometry_regularization(
    pred_coords,
    equal_length_weight=1.0,
    orthogonality_weight=1.0,
    parallel_weight=1.0,
    eps=1e-6,
):
    """
    Geometric plausibility regularization for 8 projected cuboid corners.

    Components:
    1) Same-direction edge length consistency.
    2) Orthogonality of three neighboring edges at each corner.
    3) Parallelism of same-direction edges.

    Args:
        pred_coords: (B, 8, 2) normalized coordinates in [0, 1]
        equal_length_weight: weight for same-direction edge length consistency
        orthogonality_weight: weight for 3-neighbor orthogonality
        parallel_weight: weight for same-direction edge parallelism
        eps: numeric stability epsilon
    Returns:
        scalar regularization loss
    """
    edge_vecs = _build_edge_vectors(pred_coords)

    # 1) Same-direction edge length consistency.
    equal_length_loss = pred_coords.new_zeros(())
    for axis in ("x", "y", "z"):
        lengths = torch.linalg.norm(edge_vecs[axis], dim=-1).clamp_min(eps)  # (B, 4)
        mean_len = lengths.mean(dim=1, keepdim=True).clamp_min(eps)  # (B, 1)
        rel = lengths / mean_len
        equal_length_loss = equal_length_loss + ((rel - 1.0) ** 2).mean()
    equal_length_loss = equal_length_loss / 3.0

    # 2) Orthogonality of three neighboring edges at each corner.
    ortho_terms = []
    for ex_edge, ey_edge, ez_edge in _CORNER_AXIS_EDGES:
        ex = pred_coords[:, ex_edge[1]] - pred_coords[:, ex_edge[0]]
        ey = pred_coords[:, ey_edge[1]] - pred_coords[:, ey_edge[0]]
        ez = pred_coords[:, ez_edge[1]] - pred_coords[:, ez_edge[0]]

        ex_u = F.normalize(ex, dim=-1, eps=eps)
        ey_u = F.normalize(ey, dim=-1, eps=eps)
        ez_u = F.normalize(ez, dim=-1, eps=eps)

        ortho_terms.append((ex_u * ey_u).sum(dim=-1).abs())
        ortho_terms.append((ex_u * ez_u).sum(dim=-1).abs())
        ortho_terms.append((ey_u * ez_u).sum(dim=-1).abs())
    orthogonality_loss = torch.cat(ortho_terms, dim=0).mean()

    # 3) Parallelism of same-direction edges.
    parallel_terms = []
    for axis in ("x", "y", "z"):
        vec = F.normalize(edge_vecs[axis], dim=-1, eps=eps)  # (B, 4, 2)
        for i in range(4):
            for j in range(i + 1, 4):
                cos_ij = (vec[:, i] * vec[:, j]).sum(dim=-1).abs()
                parallel_terms.append(1.0 - cos_ij)
    parallel_loss = torch.cat(parallel_terms, dim=0).mean()

    return (
        equal_length_weight * equal_length_loss
        + orthogonality_weight * orthogonality_loss
        + parallel_weight * parallel_loss
    )


def get_loss(
    pred,
    gt,
    lambda_weight=1.0,
    temperature=0.05,
    alpha=50.0,
    use_adaptive_weight=True,
    geom_reg_enabled=False,
    geom_reg_weight=0.0,
    geom_equal_length_weight=1.0,
    geom_orthogonality_weight=1.0,
    geom_parallel_weight=1.0,
    geom_eps=1e-6,
):
    """
    Combined loss for heatmap-based corner detection with foreground weighting
    and optional cuboid geometry regularization.
    Args:
        pred: (B, 8, H, W) predicted heatmap logits (before sigmoid)
        gt: dict with 'heatmap' (B, 8, H, W) [0,1] and 'coords' (B, 8, 2) pixel coords
        lambda_weight: weight for fine coordinate loss (used as target ratio if adaptive)
        temperature: softmax temperature for soft-argmax (smaller = sharper)
        alpha: foreground weight multiplier (higher = more focus on peaks)
        use_adaptive_weight: if True, auto-balance loss scales; lambda_weight becomes target ratio
        geom_reg_enabled: whether to enable cuboid geometry regularization
        geom_reg_weight: global weight for geometric regularization
        geom_equal_length_weight: sub-weight for same-direction edge length consistency
        geom_orthogonality_weight: sub-weight for adjacent-edge orthogonality
        geom_parallel_weight: sub-weight for same-direction edge parallelism
        geom_eps: epsilon for numerical stability in geometric loss
    Returns:
        loss: scalar total loss
        loss_coarse: scalar weighted BCE heatmap loss
        loss_fine: scalar coordinate loss
        loss_geom: scalar geometric regularization loss
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
    pred_coords = soft_argmax_2d(pred, temperature)  # (B, 8, 2), normalized [0,1]

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

    if geom_reg_enabled and geom_reg_weight > 0.0:
        loss_geom_raw = cuboid_geometry_regularization(
            pred_coords,
            equal_length_weight=geom_equal_length_weight,
            orthogonality_weight=geom_orthogonality_weight,
            parallel_weight=geom_parallel_weight,
            eps=geom_eps,
        )
        loss_geom = geom_reg_weight * loss_geom_raw
    else:
        loss_geom = pred.new_zeros(())

    # Adaptive weight balancing: auto-adjust to keep losses at similar scale
    if use_adaptive_weight and valid.any():
        # Compute adaptive lambda to maintain target ratio
        # Target: loss_fine_weighted / loss_coarse = lambda_weight
        # So: adaptive_lambda = lambda_weight * loss_coarse / (loss_fine + 1e-8)
        with torch.no_grad():
            adaptive_lambda = lambda_weight * loss_coarse / (loss_fine + 1e-8)
            # Clip to reasonable range to avoid instability
            adaptive_lambda = torch.clamp(adaptive_lambda, 0.001, 100.0)
        loss = loss_coarse + adaptive_lambda * loss_fine + loss_geom
    else:
        # Fixed weight (original behavior)
        loss = loss_coarse + lambda_weight * loss_fine + loss_geom

    return loss, loss_coarse, loss_fine, loss_geom
