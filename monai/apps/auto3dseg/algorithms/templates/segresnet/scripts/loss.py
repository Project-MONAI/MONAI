import torch
from monai.losses import DiceCELoss

class DiceCELoss2(DiceCELoss):
    def ce(self, input: torch.Tensor, target: torch.Tensor):
        n_pred_ch, n_target_ch = input.shape[1], target.shape[1]
        if n_pred_ch != n_target_ch and n_target_ch==1:
            target = torch.squeeze(target, dim=1)
            target = target.long()

        return self.cross_entropy(input, target)
