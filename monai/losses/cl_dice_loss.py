import numpy as np
import torch
from monai.losses import DiceLoss
from monai.losses.spatial_mask import MaskedLoss
from skimage.morphology import skeletonize_3d, binary_dilation, ball
from typing import Callable, List, Optional


class CLDiceLoss(DiceLoss):
    """
    a cl-mask-based loss function to get cl-dice loss from 3d ndarray input,only 3d input is valid
    """

    def __init__(self,
                 device,
                 lambda_dice: float = 1.0,
                 lambda_cl: float = 1.0,
                 ball_width: int = 2,
                 *args, **kwargs) -> None:
        """
        Args follow :DiceLoss
        """
        super().__init__(*args, **kwargs)
        self.device = device
        if lambda_dice < 0.0:
            raise ValueError("lambda_dice should be no less than 0.0.")
        if lambda_cl < 0.0:
            raise ValueError("lambda_cl should be no less than 0.0.")
        self.lambda_dice = lambda_dice
        self.lambda_cl = lambda_cl
        self.ball_width = ball_width
        self.dice = super().forward
        self.spatial_weighted = MaskedLoss(loss=self.dice)

    def get_clmask(self, target: np.ndarray):
        if np.ndim(target) != 3:
            raise ValueError("the target requires to be a 3d array")
        if np.max(target) < 0.5:
            return np.zeros_like(target)
        target = np.where(target < 0.5, 0., 1.)
        res = skeletonize_3d(target)
        res = np.where(res > 0, 1., 0.)
        res = binary_dilation(res, ball(self.ball_width)).astype(np.float32)
        return res

    def get_clmasks(self, target: torch.Tensor):
        res = []
        target_ = target.detach().cpu().numpy()
        if np.max(target_) < 0.5:
            return torch.from_numpy(np.zeros_like(target_)).to(self.device)  # not patch-wise check
        for i in range(target_.shape[0]):
            curr = target_[i, 0]
            curr_cl = self.get_clmask(curr)
            res.append(curr_cl)
        return torch.from_numpy(np.stack(res, axis=0).astype(np.float32)[:, np.newaxis, :, :, :]).to(self.device)

    def forward(self, input: torch.Tensor, target: torch.Tensor, mask: Optional[torch.Tensor] | None = None):
        if mask is None:
            mask = self.get_clmasks(target)
        dice_loss = self.dice(input, target)
        cldice_loss = self.spatial_weighted(input=input, target=target, mask=mask)
        total_loss: torch.Tensor = self.lambda_dice * dice_loss + self.lambda_cl * cldice_loss
        return total_loss
