import os
import time
from typing import Optional

import torch

from src.config import cfg, args, get_device
from src.datasets import make_dataloader
from src.models import make_network
from .utils import get_loss


class IOPENTrainer:
    def __init__(
        self,
        model: Optional[torch.nn.Module] = None,
        dataloader: Optional[torch.utils.data.DataLoader] = None,
        device: Optional[torch.device] = None,
        lr: float = 1e-4,
        weight_decay: float = 0.0,
        epochs: int = 1,
        log_interval: int = 20,
        ckpt_dir: str = "result/train",
        save_interval: int = 1,
        temperature: float = 1.0,
    ):
        self.device = device or get_device()
        self.model = model or make_network()
        self.model = self.model.to(self.device)
        self.dataloader = dataloader or make_dataloader()

        self.lr = lr
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.log_interval = log_interval
        self.ckpt_dir = ckpt_dir
        self.save_interval = save_interval
        self.temperature = temperature
        self.lambda_weight = cfg['train'].get("loss_lambda", 2.0)

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )

        self.global_step = 0
        self.start_epoch = 0

        os.makedirs(self.ckpt_dir, exist_ok=True)

    def _move_batch_to_device(self, batch):
        img = batch["img"].to(self.device)
        heatmap = batch["heatmap"].to(self.device)
        return img, heatmap

    def save_checkpoint(self, epoch: int):
        ckpt_path = os.path.join(self.ckpt_dir, f"checkpoint_epoch_{epoch:04d}.pth")
        payload = {
            "epoch": epoch,
            "global_step": self.global_step,
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "cfg": cfg['train'],
        }
        torch.save(payload, ckpt_path)

    def load_checkpoint(self, ckpt_path: str):
        payload = torch.load(ckpt_path, map_location=self.device)
        self.model.load_state_dict(payload["model_state"])
        self.optimizer.load_state_dict(payload["optimizer_state"])
        self.start_epoch = int(payload.get("epoch", 0)) + 1
        self.global_step = int(payload.get("global_step", 0))

    def train_one_epoch(self, epoch: int):
        self.model.train()
        epoch_total = 0.0
        epoch_coarse = 0.0
        epoch_fine = 0.0
        num_batches = 0
        start_time = time.time()

        for batch_idx, batch in enumerate(self.dataloader):
            img, heatmap = self._move_batch_to_device(batch)

            self.optimizer.zero_grad(set_to_none=True)
            pred_heatmap = self.model(img)
            loss, loss_coarse, loss_fine = get_loss(
                pred=pred_heatmap,
                gt=heatmap,
                lambda_weight=self.lambda_weight,
                temperature=self.temperature,
            )
            loss.backward()
            self.optimizer.step()

            self.global_step += 1
            num_batches += 1
            epoch_total += float(loss.item())
            epoch_coarse += float(loss_coarse.item())
            epoch_fine += float(loss_fine.item())

            if self.log_interval > 0 and (batch_idx + 1) % self.log_interval == 0:
                elapsed = time.time() - start_time
                avg_total = epoch_total / num_batches
                avg_coarse = epoch_coarse / num_batches
                avg_fine = epoch_fine / num_batches
                print(
                    f"Epoch {epoch} | Step {batch_idx + 1}/{len(self.dataloader)} | "
                    f"Loss {avg_total:.6f} (coarse {avg_coarse:.6f}, fine {avg_fine:.6f}) | "
                    f"{elapsed:.1f}s"
                )

        if num_batches == 0:
            return {"total": 0.0, "coarse": 0.0, "fine": 0.0}

        return {
            "total": epoch_total / num_batches,
            "coarse": epoch_coarse / num_batches,
            "fine": epoch_fine / num_batches,
        }

    def train(self):
        for epoch in range(self.start_epoch, self.epochs):
            metrics = self.train_one_epoch(epoch)
            print(
                f"Epoch {epoch} done | "
                f"Loss {metrics['total']:.6f} (coarse {metrics['coarse']:.6f}, "
                f"fine {metrics['fine']:.6f})"
            )

            if self.save_interval > 0 and (epoch + 1) % self.save_interval == 0:
                self.save_checkpoint(epoch)


def make_trainer(**kwargs):
    """
    Create and return the IOPEN trainer.
    """
    trainer = IOPENTrainer(**kwargs)
    return trainer