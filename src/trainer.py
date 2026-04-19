"""Training loop for CNN experiments on CIFAR-100.

Optimizer : SGD with momentum + weight decay
Scheduler : CosineAnnealingLR (T_max = total epochs)
Augment   : Optional CutMix (enabled via config strategy.cutmix)
Metrics   : Top-1 accuracy logged every epoch; history saved to CSV
"""

import logging
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.augmentation import cutmix_data, cutmix_criterion
from src.utils import Timer, save_training_log

logger = logging.getLogger(__name__)


class CNNTrainer:
    """Encapsulates training and evaluation for a single CNN experiment.

    Args:
        model:        nn.Module (already moved to device externally or by train()).
        config:       Merged experiment config dict.
        device:       torch.device to run on.
    """

    def __init__(self, model: nn.Module, config: dict, device: torch.device):
        self.model  = model.to(device)
        self.config = config
        self.device = device

        strategy = config.get("strategy", {})
        lr           = strategy.get("learning_rate", 0.01)
        weight_decay = config.get("weight_decay", 5e-4)
        momentum     = config.get("momentum", 0.9)
        self.epochs  = config.get("epochs", 50)
        self.cutmix  = strategy.get("cutmix", False)
        self.alpha   = strategy.get("cutmix_alpha", 1.0)

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            nesterov=True,
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=self.epochs
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def train(
        self,
        train_loader: DataLoader,
        test_loader:  DataLoader,
        log_path: str = None,
    ) -> list:
        """Run the full training loop.

        Args:
            train_loader: DataLoader for the training split.
            test_loader:  DataLoader for the test / val split.
            log_path:     If provided, save per-epoch history CSV to this path.

        Returns:
            history: List of dicts, one per epoch, with keys:
                     epoch, train_loss, train_acc, test_acc, lr, epoch_time_s.
        """
        history = []
        for epoch in range(1, self.epochs + 1):
            with Timer() as t:
                train_loss, train_acc = self._train_one_epoch(train_loader)
                test_acc              = self._evaluate(test_loader)

            self.scheduler.step()
            current_lr = self.scheduler.get_last_lr()[0]

            record = {
                "epoch":         epoch,
                "train_loss":    round(train_loss, 4),
                "train_acc":     round(train_acc, 4),
                "test_acc":      round(test_acc, 4),
                "lr":            round(current_lr, 6),
                "epoch_time_s":  t.elapsed,
            }
            history.append(record)

            logger.info(
                "Epoch %3d/%d | loss %.4f | train_acc %.2f%% | test_acc %.2f%% | "
                "lr %.6f | %.1fs",
                epoch, self.epochs,
                train_loss, train_acc * 100, test_acc * 100,
                current_lr, t.elapsed,
            )

        if log_path:
            save_training_log(history, log_path)

        return history

    def evaluate(self, test_loader: DataLoader) -> float:
        """Return Top-1 accuracy on the given DataLoader."""
        return self._evaluate(test_loader)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _train_one_epoch(self, loader: DataLoader) -> tuple:
        """Run one epoch of (optionally CutMix-augmented) SGD.

        Returns:
            (avg_loss, top1_accuracy) over the epoch.
        """
        self.model.train()
        total_loss = 0.0
        correct    = 0
        total      = 0

        for images, labels in loader:
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            if self.cutmix and torch.rand(1).item() < 0.5:
                images, labels_a, labels_b, lam = cutmix_data(
                    images, labels, alpha=self.alpha
                )
                outputs = self.model(images)
                loss    = cutmix_criterion(
                    self.criterion, outputs, labels_a, labels_b, lam
                )
                # Top-1 accuracy uses the dominant label (lam >= 0.5 → labels_a)
                preds   = outputs.argmax(dim=1)
                correct += (
                    lam * preds.eq(labels_a).sum().item()
                    + (1.0 - lam) * preds.eq(labels_b).sum().item()
                )
            else:
                outputs = self.model(images)
                loss    = self.criterion(outputs, labels)
                preds   = outputs.argmax(dim=1)
                correct += preds.eq(labels).sum().item()

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item() * images.size(0)
            total      += images.size(0)

        avg_loss = total_loss / total
        accuracy = correct / total
        return avg_loss, accuracy

    def _evaluate(self, loader: DataLoader) -> float:
        """Return Top-1 accuracy on loader (no gradients)."""
        self.model.eval()
        correct = 0
        total   = 0

        with torch.no_grad():
            for images, labels in loader:
                images = images.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)
                preds  = self.model(images).argmax(dim=1)
                correct += preds.eq(labels).sum().item()
                total   += labels.size(0)

        return correct / total
