import json
import os
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch

from src.config import cfg
from src.eval.utils import draw_border
from src.finetune import make_dataset
from src.eval import make_evaluator


def _resolve_path(path: str) -> str:
	return os.path.abspath(os.path.expanduser(path))


def _as_list(value: Any) -> List[Any]:
	if value is None:
		return []
	if isinstance(value, (list, tuple)):
		return list(value)
	return [value]


def _load_coco(path: str) -> Dict[str, Any]:
	with open(_resolve_path(path), "r") as handle:
		return json.load(handle)


def _annotation_to_coords(annotation: Dict[str, Any]) -> np.ndarray:
	if "keypoints" in annotation and isinstance(annotation["keypoints"], list):
		keypoints = annotation["keypoints"]
		if len(keypoints) % 3 != 0:
			raise ValueError("COCO keypoints must be stored as x, y, v triplets")

		coords = np.full((len(keypoints) // 3, 2), -1.0, dtype=np.float32)
		for idx in range(0, len(keypoints), 3):
			point_idx = idx // 3
			x = float(keypoints[idx])
			y = float(keypoints[idx + 1])
			visibility = float(keypoints[idx + 2])
			if visibility > 0:
				coords[point_idx] = [x, y]
		return coords

	for key in ("coords", "corners", "points", "gt_coords"):
		if key in annotation and isinstance(annotation[key], (list, tuple)):
			values = list(annotation[key])
			if len(values) == 8 and isinstance(values[0], (list, tuple)):
				coords = np.asarray(values, dtype=np.float32)
			else:
				if len(values) % 2 != 0:
					raise ValueError(f"Annotation field '{key}' must contain pairs of coordinates")
				coords = np.asarray(values, dtype=np.float32).reshape(-1, 2)
			if coords.shape[0] != 8:
				raise ValueError(f"Annotation field '{key}' must contain 8 corner points")
			return coords

	raise KeyError("Annotation does not contain coordinates")


def _select_annotation(annotations: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
	if not annotations:
		raise ValueError("Image does not have any annotations")

	for annotation in annotations:
		if "keypoints" in annotation or any(key in annotation for key in ("coords", "corners", "points", "gt_coords")):
			return annotation
	return annotations[0]


def _build_image_index(image_root: str) -> Dict[str, str]:
	index: Dict[str, str] = {}
	for root, _, files in os.walk(image_root):
		for file_name in files:
			path = os.path.join(root, file_name)
			index.setdefault(file_name, path)
			index.setdefault(os.path.splitext(file_name)[0], path)
	return index


def _load_rgb_image(image_path: str) -> np.ndarray:
	img = plt.imread(image_path)
	if img.ndim == 2:
		img = np.repeat(img[..., None], 3, axis=-1)
	elif img.shape[-1] == 4:
		img = img[..., :3]

	if np.issubdtype(img.dtype, np.floating) and img.max() <= 1.0:
		img = (img * 255.0).round()

	return np.clip(img, 0, 255).astype(np.uint8, copy=False)


def _collect_split_samples(split: str, image_root: str) -> List[Dict[str, Any]]:
	dataset_cfg = cfg.get("finetune", {}).get("dataset", {})
	tensor_paths = _as_list(dataset_cfg.get("tensor_path"))
	coco_paths = _as_list(dataset_cfg.get("coco_path"))
	if len(tensor_paths) != len(coco_paths):
		raise ValueError("finetune.dataset.tensor_path and coco_path must have the same length")

	train_ratio = float(dataset_cfg.get("train_ratio", 0.9))
	train_ratio = min(max(train_ratio, 0.0), 1.0)
	index = _build_image_index(image_root)
	selected_samples: List[Dict[str, Any]] = []

	for coco_path in coco_paths:
		coco = _load_coco(coco_path)
		images = coco.get("images", [])
		annotations = coco.get("annotations", [])
		annotations_by_image_id: Dict[Any, List[Dict[str, Any]]] = {}
		for annotation in annotations:
			annotations_by_image_id.setdefault(annotation.get("image_id"), []).append(annotation)

		total_count = len(images)
		train_count = int(round(total_count * train_ratio))
		train_count = min(max(train_count, 0), total_count)

		all_indices = np.arange(total_count)
		if train_count == 0:
			train_indices = np.array([], dtype=int)
		elif train_count == total_count:
			train_indices = all_indices
		else:
			train_indices = np.floor(np.arange(train_count) * total_count / train_count).astype(int)

		if split in ("train", "training"):
			current_indices = train_indices
		elif split in ("validate", "val", "valid", "validation"):
			current_indices = np.setdiff1d(all_indices, train_indices, assume_unique=False)
		else:
			raise ValueError(f"Unsupported split '{split}'")

		current_images = [images[index] for index in current_indices]

		for image_record in current_images:
			file_name = str(image_record.get("file_name", ""))
			image_path = index.get(file_name) or index.get(os.path.basename(file_name)) or index.get(
				os.path.splitext(os.path.basename(file_name))[0]
			)
			if image_path is None:
				raise KeyError(f"No RGB image found for '{file_name}' under '{image_root}'")

			image_id = image_record.get("id")
			annotation = _select_annotation(annotations_by_image_id.get(image_id, []))
			coords = _annotation_to_coords(annotation)
			selected_samples.append(
				{
					"file_name": file_name,
					"image_path": image_path,
					"coords": coords,
				}
			)

	return selected_samples


def build_result_grid(split: str, batch_size: int = 16, save_path: Optional[str] = None):
	image_root = _resolve_path("data/finetune/rgb")
	samples = _collect_split_samples(split, image_root=image_root)

	# Randomly select indices for visualization
	num_samples = len(samples)
	grid_size = min(num_samples, batch_size)
	rng = np.random.default_rng()
	selected_indices = rng.choice(num_samples, size=grid_size, replace=False).tolist()

	# Manually construct a batch from the dataset using the selected random indices
	dataset = make_dataset(split=split)
	batch_items = [dataset[int(i)] for i in selected_indices]
	batch = {
		"img": torch.stack([item["img"] for item in batch_items]),
		"heatmap": torch.stack([item["heatmap"] for item in batch_items]),
		"coords": torch.stack([item["coords"] for item in batch_items]),
	}

	# Run model inference
	evaluator = make_evaluator()
	evaluator.model.eval()

	with torch.no_grad():
		corners_pred = evaluator.inference_batch(batch)

	# Prepare GT using the same random indices
	corners_gt = [samples[i]["coords"].astype(int).tolist() for i in selected_indices]

	fig, axes = plt.subplots(4, 4, figsize=(16, 16))
	fig.suptitle(f"Finetune {split.capitalize()} split (random samples)", fontsize=16)

	for idx, ax in enumerate(axes.flat):
		if idx >= grid_size:
			ax.axis("off")
			continue

		sample_idx = selected_indices[idx]
		img = _load_rgb_image(samples[sample_idx]["image_path"])
		result = draw_border(
			obj_corners=[corners_gt[idx], corners_pred[idx]],
			img=img,
			color_list=[(0, 255, 0), (0, 0, 255)],
		)
		result = np.clip(result, 0, 255).astype(np.uint8, copy=False)

		ax.imshow(result)
		title = samples[sample_idx]["file_name"]
		ax.set_title(title, fontsize=8)
		ax.axis("off")

	plt.tight_layout()
	if save_path is not None:
		os.makedirs(os.path.dirname(save_path), exist_ok=True)
		plt.savefig(save_path, bbox_inches="tight", dpi=200)

	return fig


def main():
	output_dir = "data/finetune/plots"
	validate_fig = build_result_grid(
		split="validate",
		batch_size=16,
		save_path=os.path.join(output_dir, "validate_grid_4x4.png"),
	)
	train_fig = build_result_grid(
		split="train",
		batch_size=16,
		save_path=os.path.join(output_dir, "train_grid_4x4.png"),
	)

	plt.show()
	plt.close(validate_fig)
	plt.close(train_fig)


if __name__ == "__main__":
	main()