import os

import matplotlib.pyplot as plt
import numpy as np
import torch

from src.config import cfg
from src.datasets import make_dataloader
from src.eval import make_evaluator
from src.eval.utils import draw_border


def build_result_grid(split, batch_size=16, save_path=None):
	dataloader = make_dataloader(split=split, shuffle=False, batch_size=batch_size)
	batch = next(iter(dataloader))

	evaluator = make_evaluator()
	evaluator.model.eval()

	with torch.no_grad():
		corners_pred = evaluator.inference_batch(batch)

	corners_gt = batch['coords'].int().tolist()
	imgs = batch['img'].detach().cpu()

	fig, axes = plt.subplots(4, 4, figsize=(16, 16))
	fig.suptitle(f'{split.capitalize()} split results', fontsize=16)

	for idx, ax in enumerate(axes.flat):
		if idx >= len(imgs):
			ax.axis('off')
			continue

		img = imgs[idx].permute(1, 2, 0).numpy()
		img = np.clip(img, 0, 255).astype(np.uint8, copy=False)
		result = draw_border(
			obj_corners=[corners_gt[idx], corners_pred[idx]],
			img=img,
			color_list=[(0, 255, 0), (0, 0, 255)]
		)
		result = np.clip(result, 0, 255).astype(np.uint8, copy=False)

		ax.imshow(result)
		ax.set_title(f'{split} #{idx + 1}')
		ax.axis('off')

	plt.tight_layout()
	if save_path is not None:
		os.makedirs(os.path.dirname(save_path), exist_ok=True)
		plt.savefig(save_path, bbox_inches='tight', dpi=200)

	return fig


def main():
	output_dir = cfg['eval']['output_dir']
	validate_fig = build_result_grid(
		split='validate',
		batch_size=16,
		save_path=os.path.join(output_dir, 'validate_grid_4x4.png')
	)
	train_fig = build_result_grid(
		split='train',
		batch_size=16,
		save_path=os.path.join(output_dir, 'train_grid_4x4.png')
	)

	plt.show()
	plt.close(validate_fig)
	plt.close(train_fig)


if __name__ == '__main__':
	main()
