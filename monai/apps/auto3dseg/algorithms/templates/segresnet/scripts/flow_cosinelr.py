from torch.optim.lr_scheduler import _LRScheduler
import math, warnings

class CosineAnnealingLRWithWarmup(_LRScheduler):

    def __init__(self, optimizer, T_max, eta_min=0, last_epoch=-1, verbose=False, warmup_epochs=2, warmup_multiplier=0.1):
        self.T_max = T_max
        self.eta_min = eta_min

        warmup_epochs = max(0, min(warmup_epochs, T_max-1))

        # assert warmup_epochs >= 0
        # assert warmup_epochs < T_max
        assert warmup_multiplier > 0
        assert warmup_multiplier < 1

        self.warmup_epochs=warmup_epochs
        self.warmup_multiplier=warmup_multiplier

        super().__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        return self._get_closed_form_lr()

    def _get_closed_form_lr(self):

        lr = 0
        if self.warmup_epochs > 0 and self.last_epoch < self.warmup_epochs:
            lr = [base_lr * (self.warmup_multiplier + (1-self.warmup_multiplier) * (max(0, self.last_epoch) /float(self.warmup_epochs))) for base_lr in self.base_lrs]
        else:
            fraction = (self.last_epoch - self.warmup_epochs) / (self.T_max  - self.warmup_epochs)
            lr = [self.eta_min + (base_lr - self.eta_min) * (1 + math.cos(math.pi * fraction)) / 2 for base_lr in self.base_lrs]

        # print('CosineAnnealingLRWithWarmup', self.last_epoch, lr)
        return lr