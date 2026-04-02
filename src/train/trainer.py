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
        temperature: float = 0.05,
        alpha: float = 50.0,
        use_adaptive_weight: bool = True,
        coarse_only_epochs: int = 0,
        early_stopping_enabled: bool = True,
        early_stopping_patience: int = 10,
        early_stopping_min_delta: float = 0.0,
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
        self.alpha = alpha
        self.use_adaptive_weight = use_adaptive_weight
        self.coarse_only_epochs = max(0, int(coarse_only_epochs))
        self.lambda_weight = cfg['train'].get("loss_lambda", 1.0)
        self.early_stopping_enabled = bool(early_stopping_enabled)
        self.early_stopping_patience = max(1, int(early_stopping_patience))
        self.early_stopping_min_delta = float(early_stopping_min_delta)
        self.no_improve_count = 0

        train_cfg = cfg.get('train', {})
        self.validate_enabled = bool(train_cfg.get('validate_enabled', True))
        self.validate_interval = int(train_cfg.get('validate_interval', 1))
        self.val_batch_size = int(train_cfg.get('val_batch_size', train_cfg.get('batch_size', 1)))
        self.save_best_checkpoint_enabled = bool(train_cfg.get('save_best_checkpoint', True))
        self.best_checkpoint_path = train_cfg.get(
            'best_checkpoint_path', os.path.join(self.ckpt_dir, 'best.pth')
        )
        self.best_val_total = float('inf')
        self.val_dataloader = None
        if self.validate_enabled:
            self.val_dataloader = make_dataloader(
                split='validate',
                shuffle=False,
                batch_size=self.val_batch_size,
            )

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )

        self.global_step = 0
        self.start_epoch = 0

        os.makedirs(self.ckpt_dir, exist_ok=True)
        best_ckpt_dir = os.path.dirname(self.best_checkpoint_path)
        if best_ckpt_dir:
            os.makedirs(best_ckpt_dir, exist_ok=True)

    def _move_batch_to_device(self, batch):
        img = batch["img"].to(self.device)
        heatmap = batch["heatmap"].to(self.device)
        coords = batch["coords"].to(self.device)
        return img, heatmap, coords

    def save_checkpoint(self, epoch: int):
        ckpt_path = os.path.join(self.ckpt_dir, f"checkpoint_epoch_{epoch:04d}.pth")
        payload = {
            "epoch": epoch,
            "global_step": self.global_step,
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "cfg": cfg['train'],
            "best_val_total": self.best_val_total,
            "no_improve_count": self.no_improve_count,
        }
        torch.save(payload, ckpt_path)

    def save_best_checkpoint(self, epoch: int, val_metrics: dict):
        payload = {
            "epoch": epoch,
            "global_step": self.global_step,
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "cfg": cfg['train'],
            "val_metrics": val_metrics,
            "best_val_total": val_metrics["total"],
        }
        torch.save(payload, self.best_checkpoint_path)

    def load_checkpoint(self, ckpt_path: str):
        payload = torch.load(ckpt_path, map_location=self.device)
        self.model.load_state_dict(payload["model_state"])
        self.optimizer.load_state_dict(payload["optimizer_state"])
        self.start_epoch = int(payload.get("epoch", 0)) + 1
        self.global_step = int(payload.get("global_step", 0))
        self.best_val_total = float(payload.get("best_val_total", self.best_val_total))
        self.no_improve_count = int(payload.get("no_improve_count", self.no_improve_count))

    def train_one_epoch(self, epoch: int):
        self.model.train()
        epoch_total = 0.0
        epoch_coarse = 0.0
        epoch_fine = 0.0
        num_batches = 0
        start_time = time.time()
        use_coarse_only = epoch < self.coarse_only_epochs

        for batch_idx, batch in enumerate(self.dataloader):
            img, heatmap, coords = self._move_batch_to_device(batch)

            self.optimizer.zero_grad(set_to_none=True)
            pred_heatmap = self.model(img)
            loss, loss_coarse, loss_fine = get_loss(
                pred=pred_heatmap,
                gt={'heatmap': heatmap, 'coords': coords},
                lambda_weight=self.lambda_weight,
                temperature=self.temperature,
                alpha=self.alpha,
                use_adaptive_weight=self.use_adaptive_weight,
            )
            optimize_loss = loss_coarse if use_coarse_only else loss
            optimize_loss.backward()
            self.optimizer.step()

            self.global_step += 1
            num_batches += 1
            epoch_total += float(optimize_loss.item())
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
                    f"mode {'coarse-only' if use_coarse_only else 'coarse+fine'} | "
                    f"{elapsed:.1f}s"
                )

        if num_batches == 0:
            return {"total": 0.0, "coarse": 0.0, "fine": 0.0}

        return {
            "total": epoch_total / num_batches,
            "coarse": epoch_coarse / num_batches,
            "fine": epoch_fine / num_batches,
        }

    @torch.no_grad()
    def validate_one_epoch(self, epoch: int):
        if self.val_dataloader is None:
            return None

        self.model.eval()
        epoch_total = 0.0
        epoch_coarse = 0.0
        epoch_fine = 0.0
        num_batches = 0

        for batch in self.val_dataloader:
            img, heatmap, coords = self._move_batch_to_device(batch)
            pred_heatmap = self.model(img)
            loss, loss_coarse, loss_fine = get_loss(
                pred=pred_heatmap,
                gt={'heatmap': heatmap, 'coords': coords},
                lambda_weight=self.lambda_weight,
                temperature=self.temperature,
                alpha=self.alpha,
                use_adaptive_weight=self.use_adaptive_weight,
            )

            num_batches += 1
            epoch_total += float(loss.item())
            epoch_coarse += float(loss_coarse.item())
            epoch_fine += float(loss_fine.item())

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
            train_mode = 'coarse-only' if epoch < self.coarse_only_epochs else 'coarse+fine'
            print(
                f"Epoch {epoch} done | "
                f"Loss {metrics['total']:.6f} (coarse {metrics['coarse']:.6f}, "
                f"fine {metrics['fine']:.6f}) | mode {train_mode}"
            )

            should_validate = (
                self.validate_enabled and
                self.val_dataloader is not None and
                self.validate_interval > 0 and
                (epoch + 1) % self.validate_interval == 0
            )
            if should_validate:
                val_metrics = self.validate_one_epoch(epoch)
                print(
                    f"Epoch {epoch} validate | "
                    f"Loss {val_metrics['total']:.6f} "
                    f"(coarse {val_metrics['coarse']:.6f}, fine {val_metrics['fine']:.6f})"
                )

                improved = val_metrics['total'] < (self.best_val_total - self.early_stopping_min_delta)

                if improved:
                    self.best_val_total = val_metrics['total']
                    self.no_improve_count = 0
                else:
                    self.no_improve_count += 1

                if self.save_best_checkpoint_enabled and improved:
                    self.save_best_checkpoint(epoch, val_metrics)
                    print(
                        f"Epoch {epoch} best updated | "
                        f"val_total {self.best_val_total:.6f} | "
                        f"saved {self.best_checkpoint_path}"
                    )

                if self.early_stopping_enabled:
                    print(
                        f"Epoch {epoch} early-stop monitor | "
                        f"best {self.best_val_total:.6f} | "
                        f"no_improve {self.no_improve_count}/{self.early_stopping_patience}"
                    )
                    if self.no_improve_count >= self.early_stopping_patience:
                        print(
                            f"Early stopping at epoch {epoch} | "
                            f"best_val_total {self.best_val_total:.6f}"
                        )
                        if self.save_interval > 0:
                            self.save_checkpoint(epoch)
                        break

            if self.save_interval > 0 and (epoch + 1) % self.save_interval == 0:
                self.save_checkpoint(epoch)


def make_trainer(**kwargs):
    """
    Create and return the IOPEN trainer.
    """
    trainer = IOPENTrainer(**kwargs)
    return trainer