import torch
from monai.utils.to_onehot import to_onehot


def compute_meandice(y_pred, y, remove_bg, is_onehot_targets, logit_thresh=0.5):
    """Computes dice score metric from full size Tensor and collects average.

    Args:
        remove_bg (Bool): skip dice computation on the first channel of the predicted output or not.
        is_onehot_targets (Bool): whether the label data(y) is already in One-Hot format, will convert if not.
        logit_thresh (Float): the threshold value to round value to 0.0 and 1.0, default is 0.5.

    Note:
        (1) if this is multi-labels task(One-Hot label), use logit_thresh to convert y_pred to 0 or 1.
        (2) if this is multi-classes task(non-Ono-Hot label), use Argmax to select index and convert to One-Hot.

    """
    n_channels_y_pred = y_pred.shape[1]
    n_len = len(y_pred.shape)
    assert n_len == 4 or n_len == 5, 'unsupported input shape.'

    if is_onehot_targets is True:
        y_pred = (y_pred >= logit_thresh).float()
        y = (y >= logit_thresh).float()
    else:
        y_pred = (torch.argmax(y_pred, dim=1)).float()
        y_pred = to_onehot(y_pred, n_channels_y_pred)
        y = to_onehot(y, n_channels_y_pred)

    if remove_bg:
        y = y[:, 1:]
        y_pred = y_pred[:, 1:]

    # reducing only spatial dimensions (not batch nor channels)
    reduce_axis = list(range(2, n_len))
    intersection = torch.sum(y * y_pred, reduce_axis)

    y_o = torch.sum(y, reduce_axis)
    y_pred_o = torch.sum(y_pred, reduce_axis)
    denominator = y_o + y_pred_o

    f = (2.0 * intersection) / denominator
    # final reduce_mean across batches and channels
    return torch.mean(f)
