import json
import os
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from src.config import cfg


def _as_list(value: Any) -> List[Any]:
	if value is None:
		return []
	if isinstance(value, (list, tuple)):
		return list(value)
	return [value]


def _resolve_path(path: str) -> str:
	return os.path.abspath(os.path.expanduser(path))


def _load_tensor_bundle(path: str) -> Tuple[List[torch.Tensor], Optional[List[str]]]:
	try:
		bundle = torch.load(_resolve_path(path), map_location="cpu", weights_only=True)
	except TypeError:
		bundle = torch.load(_resolve_path(path), map_location="cpu")

	if isinstance(bundle, dict):
		tensors = bundle.get("tensors")
		if tensors is None:
			tensors = bundle.get("tensor")
		if tensors is None:
			tensors = bundle.get("data")

		names = bundle.get("names")
		if names is None:
			names = bundle.get("file_names")
		if names is None:
			names = bundle.get("filenames")
	else:
		tensors = bundle
		names = None

	if tensors is None:
		raise ValueError(f"Unsupported tensor bundle format: {path}")

	tensor_list = []
	for tensor in tensors:
		if isinstance(tensor, torch.Tensor):
			tensor_list.append(tensor.detach().cpu())
		else:
			tensor_list.append(torch.as_tensor(tensor))

	if names is not None:
		name_list = [str(name) for name in names]
		if len(name_list) != len(tensor_list):
			name_list = None
	else:
		name_list = None

	return tensor_list, name_list


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


def _generate_heatmap(coords: np.ndarray, height: int, width: int, bbox: Optional[Sequence[float]] = None) -> np.ndarray:
	heatmap = np.zeros((8, height, width), dtype=np.float32)
	valid_mask = (
		(coords[:, 0] >= 0.0)
		& (coords[:, 0] < width)
		& (coords[:, 1] >= 0.0)
		& (coords[:, 1] < height)
	)

	if not np.any(valid_mask):
		return heatmap

	if bbox is not None and len(bbox) >= 4:
		obj_size = 0.5 * (float(bbox[2]) + float(bbox[3]))
	else:
		valid_coords = coords[valid_mask]
		x_coords = valid_coords[:, 0]
		y_coords = valid_coords[:, 1]
		obj_size = 0.5 * ((np.max(x_coords) - np.min(x_coords)) + (np.max(y_coords) - np.min(y_coords)))

	sigma = max(obj_size / 10.0, 1e-3)
	yy, xx = np.ogrid[:height, :width]

	for channel_index, (x_coord, y_coord) in enumerate(coords):
		if not valid_mask[channel_index]:
			continue

		center_x = int(round(float(x_coord)))
		center_y = int(round(float(y_coord)))
		if 0 <= center_x < width and 0 <= center_y < height:
			distance_sq = (xx - center_x) ** 2 + (yy - center_y) ** 2
			heatmap[channel_index] = np.exp(-distance_sq / (2.0 * sigma * sigma))

	return heatmap


def _normalize_image_tensor(img_tensor: torch.Tensor) -> torch.Tensor:
	if img_tensor.ndim != 3:
		raise ValueError(f"Expected an image tensor with 3 dimensions, got shape {tuple(img_tensor.shape)}")

	if img_tensor.shape[0] not in (1, 3) and img_tensor.shape[-1] in (1, 3):
		img_tensor = img_tensor.permute(2, 0, 1)

	img_tensor = img_tensor.float()

	target_height = int(cfg.get("height", img_tensor.shape[-2]))
	target_width = int(cfg.get("width", img_tensor.shape[-1]))
	if img_tensor.shape[-2:] != (target_height, target_width):
		img_tensor = F.interpolate(
			img_tensor.unsqueeze(0),
			size=(target_height, target_width),
			mode="bilinear",
			align_corners=False,
		).squeeze(0)

	return img_tensor


class IOPENDataset(Dataset):
	def __init__(self, data_root=None, split: str = "train"):
		del data_root

		dataset_cfg = cfg.get("finetune", {}).get("dataset", {})
		tensor_paths = _as_list(dataset_cfg.get("tensor_path"))
		coco_paths = _as_list(dataset_cfg.get("coco_path"))

		if not tensor_paths:
			raise ValueError("finetune.dataset.tensor_path is not configured")
		if not coco_paths:
			raise ValueError("finetune.dataset.coco_path is not configured")
		if len(tensor_paths) != len(coco_paths):
			raise ValueError("finetune.dataset.tensor_path and coco_path must have the same length")

		self.split = split
		train_ratio = float(dataset_cfg.get("train_ratio", 0.9))
		train_ratio = min(max(train_ratio, 0.0), 1.0)
		self._is_train_split = split in ("train", "training")
		self._is_val_split = split in ("validate", "val", "valid", "validation")
		self._train_ratio = train_ratio

		self.samples: List[Dict[str, Any]] = []
		for tensor_path, coco_path in zip(tensor_paths, coco_paths):
			tensors, names = _load_tensor_bundle(tensor_path)
			coco = _load_coco(coco_path)

			images = coco.get("images", [])
			annotations = coco.get("annotations", [])
			anns_by_image_id: Dict[Any, List[Dict[str, Any]]] = {}
			for annotation in annotations:
				anns_by_image_id.setdefault(annotation.get("image_id"), []).append(annotation)

			image_by_name: Dict[str, Dict[str, Any]] = {}
			for image_record in images:
				file_name = str(image_record.get("file_name", ""))
				image_by_name[file_name] = image_record
				image_by_name[os.path.basename(file_name)] = image_record
				image_by_name[os.path.splitext(file_name)[0]] = image_record

			if names is not None:
				paired_records: Iterable[Tuple[torch.Tensor, str]] = zip(tensors, names)
				for tensor, tensor_name in paired_records:
					image_record = image_by_name.get(tensor_name)
					if image_record is None:
						image_record = image_by_name.get(os.path.basename(tensor_name))
					if image_record is None:
						image_record = image_by_name.get(os.path.splitext(os.path.basename(tensor_name))[0])
					if image_record is None:
						raise KeyError(f"No COCO image record found for tensor name '{tensor_name}'")

					image_id = image_record.get("id")
					image_annotations = anns_by_image_id.get(image_id, [])
					annotation = _select_annotation(image_annotations)
					coords = _annotation_to_coords(annotation)
					height = int(image_record.get("height", cfg.get("height", tensor.shape[-2])))
					width = int(image_record.get("width", cfg.get("width", tensor.shape[-1])))
					heatmap = _generate_heatmap(coords, height, width, bbox=annotation.get("bbox"))

					self.samples.append(
						{
							"img": tensor,
							"heatmap": torch.from_numpy(heatmap),
							"coords": torch.from_numpy(coords),
						}
					)
			else:
				if len(images) != len(tensors):
					raise ValueError(
						f"Tensor count ({len(tensors)}) does not match image count ({len(images)}) for {coco_path}"
					)

				for tensor, image_record in zip(tensors, images):
					image_id = image_record.get("id")
					image_annotations = anns_by_image_id.get(image_id, [])
					annotation = _select_annotation(image_annotations)
					coords = _annotation_to_coords(annotation)
					height = int(image_record.get("height", cfg.get("height", tensor.shape[-2])))
					width = int(image_record.get("width", cfg.get("width", tensor.shape[-1])))
					heatmap = _generate_heatmap(coords, height, width, bbox=annotation.get("bbox"))

					self.samples.append(
						{
							"img": tensor,
							"heatmap": torch.from_numpy(heatmap),
							"coords": torch.from_numpy(coords),
						}
					)

		if self._is_train_split or self._is_val_split:
			total_count = len(self.samples)
			train_count = int(round(total_count * self._train_ratio))
			train_count = min(max(train_count, 0), total_count)

			all_indices = np.arange(total_count)
			if train_count == 0:
				train_indices = np.array([], dtype=int)
			elif train_count == total_count:
				train_indices = all_indices
			else:
				train_indices = np.floor(np.arange(train_count) * total_count / train_count).astype(int)

			if self._is_train_split:
				selected_indices = train_indices
			else:
				selected_indices = np.setdiff1d(all_indices, train_indices, assume_unique=False)

			self.samples = [self.samples[index] for index in selected_indices]

	def __len__(self):
		return len(self.samples)

	def __getitem__(self, index):
		sample = self.samples[index]

		img = _normalize_image_tensor(sample["img"])
		heatmap = sample["heatmap"].float()
		coords = sample["coords"].float()

		target_height = int(cfg.get("height", heatmap.shape[-2]))
		target_width = int(cfg.get("width", heatmap.shape[-1]))
		if heatmap.shape[-2:] != (target_height, target_width):
			source_height, source_width = heatmap.shape[-2:]
			heatmap = F.interpolate(
				heatmap.unsqueeze(0),
				size=(target_height, target_width),
				mode="bilinear",
				align_corners=False,
			).squeeze(0)

			scale_x = target_width / max(source_width, 1)
			scale_y = target_height / max(source_height, 1)
			valid = (coords[:, 0] >= 0) & (coords[:, 1] >= 0)
			coords = coords.clone()
			coords[valid, 0] = coords[valid, 0] * scale_x
			coords[valid, 1] = coords[valid, 1] * scale_y

		return {
			"img": img,
			"heatmap": heatmap,
			"coords": coords,
		}


def make_dataset(split="train"):
	"""
	Create and return the finetune dataset.
	"""
	return IOPENDataset(split=split)


def make_dataloader(split="train", shuffle=None, batch_size=None):
	"""
	Create and return the finetune dataloader.
	"""
	dataset = make_dataset(split=split)
	if batch_size is None:
		batch_size = cfg.get("train", {}).get("batch_size", 1)
	if shuffle is None:
		shuffle = (split == "train")
	return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
