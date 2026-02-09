import numpy as np
import json

def gen_masked_img(rgb, mask):
    """
    Generate a masked image by applying a binary mask to an RGB image.
    
    :param rgb: RGB image array of shape (height, width, 3)
    :param mask: Binary mask array of shape (height, width)
    :return: Masked image with same shape as rgb
    """
    mask = (mask > 0).astype(np.bool)
    masked_img = (rgb * mask[:, :, None]).astype(np.float32)
    return masked_img

def gen_heatmap(camera, model, cam_R_m2c, cam_t_m2c):
    """
    Generate a heatmap for 3D bounding box keypoints projected onto image.
    
    :param camera: Camera intrinsics dict with keys 'cx', 'cy', 'fx', 'fy', 'height', 'width'
    :param model: Model dict with keys 'size_x', 'size_y', 'size_z'
    :param cam_R_m2c: Camera rotation matrix (3x3)
    :param cam_t_m2c: Camera translation vector (3,)
    :return: Heatmap array of shape (8, H, W) with Gaussian peaks at projected keypoints
    """
    cx, cy, fx, fy = camera['cx'], camera['cy'], camera['fx'], camera['fy']
    H, W = camera['height'], camera['width']
    dx, dy, dz = model['size_x'], model['size_y'], model['size_z']
    
    bbox_3d = np.array([
        [-dx/2, -dy/2, -dz/2],
        [ dx/2, -dy/2, -dz/2],
        [ dx/2,  dy/2, -dz/2],
        [-dx/2,  dy/2, -dz/2],
        [-dx/2, -dy/2,  dz/2],
        [ dx/2, -dy/2,  dz/2],
        [ dx/2,  dy/2,  dz/2],
        [-dx/2,  dy/2,  dz/2],
    ])

    bbox_cam = (cam_R_m2c @ bbox_3d.T + cam_t_m2c).T # shape (8, 3)
    bbox_2d = []
    for x, y, z in bbox_cam:
        u = fx * x / z + cx
        v = fy * y / z + cy
        bbox_2d.append([u, v])
    bbox_2d = np.array(bbox_2d) # shape (8, 2)

    heatmap = np.zeros((8, H, W), dtype=np.float32)
    sigma = 5  # Standard deviation for Gaussian

    for i, (u, v) in enumerate(bbox_2d):
        u, v = int(round(u)), int(round(v))
        if 0 <= u < W and 0 <= v < H:
            y, x = np.ogrid[:H, :W]
            gaussian = np.exp(-((x - u)**2 + (y - v)**2) / (2 * sigma**2))
            heatmap[i] = gaussian

    return heatmap

def load_data(root, num_scene=6, img_per_scene=1000):
    """
    Load dataset from PBR training data with scene and image filtering.
    
    :param root: Root directory path containing models, camera.json, and train_pbr/
    :param num_scene: Number of scenes to load (default: 6)
    :param img_per_scene: Number of images per scene to process (default: 1000)
    :return: Dictionary containing model info, camera parameters, pbr_root path, and samples with rgb/mask paths and poses
    """
    data_dict = {}

    with open(root + "models/models_info.json") as f:
        models = json.load(f)
        data_dict['model'] = models["1"]
    
    with open(root + "camera.json") as f:
        data_dict['camera'] = json.load(f)
    
    pbr_root = root + "train_pbr/"
    data_dict['pbr_root'] = pbr_root

    data_dict['samples'] = {
        'rgb_path': [],
        'mask_path': [],
        'cam_R_m2c': [],
        'cam_t_m2c': []
    }
    for scene_id in range(num_scene):
        scene_path = "00000" + str(scene_id) + "/"
        with open(pbr_root + scene_path + "scene_gt.json") as f:
            scene_gt = json.load(f)
        with open(pbr_root + scene_path + "scene_gt_info.json") as f:
            scene_gt_info = json.load(f)
        
        for i in range(img_per_scene):
            rgb_path = scene_path + "rgb/" + str(i).zfill(6) + ".jpg"
            num_instance = len(scene_gt[str(i)])
            for j in range(num_instance):
                mask_path = scene_path + "mask_visib/" + str(i).zfill(6) + "_" + str(j).zfill(6) + ".png"

                cam_R_m2c = scene_gt[str(i)][j]["cam_R_m2c"]
                cam_t_m2c = scene_gt[str(i)][j]["cam_t_m2c"]

                visib_fract = scene_gt_info[str(i)][j]["visib_fract"]

                if visib_fract > 0.2:
                    data_dict['samples']['rgb_path'].append(rgb_path)
                    data_dict['samples']['mask_path'].append(mask_path)
                    data_dict['samples']['cam_R_m2c'].append(cam_R_m2c)
                    data_dict['samples']['cam_t_m2c'].append(cam_t_m2c)

    return data_dict