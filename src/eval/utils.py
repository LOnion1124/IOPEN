import torch
import cv2
import numpy as np
import imageio.v3 as iio
from src.config import cfg, args

def gen_coords(heatmap):
    """
    Extract 2D coordinates of maximum values from a batch of heatmaps.
    This function finds the location of the maximum value in each heatmap
    channel and converts the flattened index to 2D (x, y) coordinates.
    Args:
        heatmap (torch.Tensor): A 4D tensor of shape (B, 8, H, W) where:
            - B: batch size
            - 8: number of heatmap channels
            - H: height of each heatmap
            - W: width of each heatmap
    Returns:
        list: A list of length B, where each element is a list of 8 tuples.
              Each tuple contains (x, y) coordinates representing the position
              of the maximum value in the corresponding heatmap channel.
              Coordinates are 0-indexed integers.
    """

    B, H, W = heatmap.shape[0], heatmap.shape[-2], heatmap.shape[-1]
    coords = []
    for b in range(B):
        batch_coords = []
        for i in range(8):
            heatmap_2d = heatmap[b, i]
            max_idx = torch.argmax(heatmap_2d)
            y = max_idx // W
            x = max_idx % W
            batch_coords.append((x.item(), y.item()))
        coords.append(batch_coords)
    return coords

def draw_border(obj_corners, img, color_list=None):
    """
    Draws 3D bounding box borders on an image by connecting corner points.
        obj_corners (list): List of shape (N, 8, 2), where each element contains 8 (x, y) corner coordinates for a 3D bounding box.
                            - Indices 0-3: Top face corners
                            - Indices 4-7: Bottom face corners
        img (torch.Tensor or numpy.ndarray): Input image of shape (H, W, 3) or (3, H, W), on which the bounding boxes will be drawn.
        color_list (list, optional): List of BGR color tuples for each bounding box. Defaults to green [(0, 255, 0)].
        numpy.ndarray: Image with 3D bounding box borders drawn.
    Raises:
        ValueError: If the input image does not have 3 dimensions.
    Notes:
        - The function connects corners to form the top and bottom faces, as well as vertical edges of the bounding box.
        - Handles both PyTorch tensors and NumPy arrays as input images.
    """
    
    if isinstance(img, torch.Tensor):
        img = img.detach().cpu().numpy()

    if img.ndim != 3:
        raise ValueError(f"Expected image with 3 dimensions, got shape {img.shape}")

    if img.shape[0] in (1, 3) and img.shape[-1] not in (1, 3):
        img = np.transpose(img, (1, 2, 0))

    img = np.ascontiguousarray(img)

    if not color_list:
        color_list = [(0, 255, 0)]

    for idx, corners in enumerate(obj_corners):
        # corners is a (8, 2) list containing 8 3D bounding box
        # corner points' cordinates for one object
        # Draw lines connecting the 8 corners to form a 3D bounding box
        # Order: 0-1-2-3-0 (top face), 4-5-6-7-4 (bottom face), 0-4, 1-5, 2-6, 3-7 (vertical edges)
        edges = [(0, 1), (1, 2), (2, 3), (3, 0),  # top face
                 (4, 5), (5, 6), (6, 7), (7, 4),  # bottom face
                 (0, 4), (1, 5), (2, 6), (3, 7)]  # vertical edges

        color = tuple(map(int, color_list[idx % len(color_list)]))

        for edge in edges:
            pt1 = tuple(map(int, corners[edge[0]]))
            pt2 = tuple(map(int, corners[edge[1]]))
            cv2.line(img, pt1, pt2, color, 2)
    
    return img