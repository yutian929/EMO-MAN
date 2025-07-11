import torch

from efficientvit.apps.utils.dist import sync_tensor

__all__ = ["AverageMeter"]


class AverageMeter:
    """Computes and stores the average and current value."""

    def __init__(self, is_distributed=True):
        self.is_distributed = is_distributed
        self.sum = 0
        self.count = 0

    def _sync(self, val: torch.Tensor | int | float) -> torch.Tensor | int | float:
        return sync_tensor(val, reduce="sum") if self.is_distributed else val

    def update(self, val: torch.Tensor | int | float, delta_n=1):
        self.count += self._sync(delta_n)
        self.sum += self._sync(val * delta_n)

    def get_count(self) -> torch.Tensor | int | float:
        return (
            self.count.item()
            if isinstance(self.count, torch.Tensor) and self.count.numel() == 1
            else self.count
        )

    @property
    def avg(self):
        avg = -1 if self.count == 0 else self.sum / self.count
        return avg.item() if isinstance(avg, torch.Tensor) and avg.numel() == 1 else avg
