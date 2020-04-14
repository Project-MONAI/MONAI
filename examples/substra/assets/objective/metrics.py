import substratools as tools
import torch

from monai.metrics import DiceMetric


class MonaiMetrics(tools.Metrics):
    dice_metric = DiceMetric(include_background=True, to_onehot_y=False, sigmoid=True, reduction="mean")

    def score(self, y_true, y_pred):
        metric_sum = 0.0
        metric_count = 0
        with torch.no_grad():
            for (val_true, val_pred) in zip(y_true, y_pred):
                val_true, _ = val_true
                val_pred, _ = val_pred
                value = self.dice_metric(
                    y_pred=val_pred,
                    y=val_true,
                )
                metric_count += len(value)
                metric_sum += value.item() * len(value)
        metric = metric_sum / metric_count
        return metric


if __name__ == "__main__":
    tools.metrics.execute(MonaiMetrics())
