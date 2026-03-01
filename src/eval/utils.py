import torch
import cv2
import numpy as np
import imageio.v3 as iio
from src.datasets.utils import preprocess_image_for_dinov2

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

def load_frame_src(frame_id, original_dir, masked_dirs):
    original_path = original_dir + 'frame_' + str(frame_id).zfill(6) + '.png'
    original_img = iio.imread(original_path)
    original_img = preprocess_image_for_dinov2(original_img)
    masked_imgs = []
    for masked_dir in masked_dirs:
        masked_img_path = masked_dir + 'frame_' + str(frame_id).zfill(6) + '.png'
        masked_img = iio.imread(masked_img_path)
        masked_img = preprocess_image_for_dinov2(masked_img)
        masked_imgs.append(masked_img)
    return original_img, masked_imgs

def draw_border(obj_corners, img):
    """
    Draw 3D bounding boxes on an image by connecting corner points.
    This function takes a list of 3D bounding box corners and draws them on the provided image
    by connecting the corners with green lines to visualize the bounding boxes.
    Args:
        obj_corners (list): A list of shape (N, 8, 2) containing N sets of 3D bounding box corner points.
                           Each set contains 8 corner points with (x, y) coordinates representing:
                           - 0-3: Top face corners
                           - 4-7: Bottom face corners
        img (torch.Tensor or numpy.ndarray): Input image of shape (H, W, 3) where the bounding boxes will be drawn.
    Returns:
        numpy.ndarray: The input image with drawn 3D bounding boxes overlaid as green lines.
    """
    if isinstance(img, torch.Tensor):
        img = img.detach().cpu().numpy()

    if img.ndim != 3:
        raise ValueError(f"Expected image with 3 dimensions, got shape {img.shape}")

    if img.shape[0] in (1, 3) and img.shape[-1] not in (1, 3):
        img = np.transpose(img, (1, 2, 0))

    img = np.ascontiguousarray(img)

    for corners in obj_corners:
        # corners is a (8, 2) list containing 8 3D bounding box
        # corner points' cordinates for one object
        # Draw lines connecting the 8 corners to form a 3D bounding box
        # Order: 0-1-2-3-0 (top face), 4-5-6-7-4 (bottom face), 0-4, 1-5, 2-6, 3-7 (vertical edges)
        edges = [(0, 1), (1, 2), (2, 3), (3, 0),  # top face
                 (4, 5), (5, 6), (6, 7), (7, 4),  # bottom face
                 (0, 4), (1, 5), (2, 6), (3, 7)]  # vertical edges

        for edge in edges:
            pt1 = tuple(map(int, corners[edge[0]]))
            pt2 = tuple(map(int, corners[edge[1]]))
            cv2.line(img, pt1, pt2, (0, 255, 0), 2)
    
    return img