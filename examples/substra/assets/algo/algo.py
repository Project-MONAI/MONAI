import substratools as tools
import torch

import monai
from monai.inferers import sliding_window_inference


class MonaiAlgo(tools.algo.Algo):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loss_function = monai.losses.DiceLoss(sigmoid=True)

    def _get_model(self):
        model = monai.networks.nets.UNet(
            dimensions=3,
            in_channels=1,
            out_channels=1,
            channels=(16, 32, 64, 128, 256),
            strides=(2, 2, 2, 2),
            num_res_units=2,
        ).to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), 1e-3)
        return (model, optimizer)

    def train(self, X, y, models, rank):  # noqa: N803
        # create UNet, DiceLoss and Adam optimizer
        if not models:
            model, optimizer = self._get_model()
        else:
            model, optimizer = models[0]

        # start a typical PyTorch training
        epoch_loss = 0
        step = 0
        model.train()
        for data in X:
            inputs = data["img"].to(self.device)
            labels = data["seg"].to(self.device)
            step += 1
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = self.loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            print(f"Step {step}, train_loss:{loss.item()}")

        print(f"Average loss: {epoch_loss / step}")

        return (model, optimizer)

    def predict(self, X, model):  # noqa: N803
        y_pred = []
        model, _ = model  # drop the optimizer
        model.eval()
        with torch.no_grad():
            for inputs, metadata in X:
                inputs = inputs.to(self.device)
                roi_size = (96, 96, 96)
                sw_batch_size = 4
                outputs = sliding_window_inference(inputs, roi_size, sw_batch_size, model)
                y_pred.append((outputs, metadata))
        return y_pred

    def load_model(self, path):
        model, optimizer = self._get_model()
        data = torch.load(path)
        model.load_state_dict(data["model_state_dict"])
        optimizer.load_state_dict(data["optimizer_state_dict"])
        return model, optimizer

    def save_model(self, model, path):
        model, optimizer = model
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            },
            path,
        )


if __name__ == "__main__":
    tools.algo.execute(MonaiAlgo())
