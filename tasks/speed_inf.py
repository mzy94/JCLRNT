import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error


class Regressor(nn.Module):
    def __init__(self, input_size):
        super(Regressor, self).__init__()
        self.fc = nn.Linear(input_size, 1)

    def forward(self, x):
        return self.fc(x).squeeze(1)


def evaluation(model, feature_df, fold=5):
    print("\n--- Speed Inference ---")
    model.eval()
    x = model.encode_graph()
    if isinstance(x, tuple):
        x = x[0]
    x = x.detach()
    y = torch.tensor(feature_df['road_speed'].tolist()).cuda()
    split = x.shape[0] // fold

    y_preds = []
    y_trues = []
    for _ in range(fold):
        x = torch.cat((x[split:], x[:split]), 0)
        y = torch.cat((y[split:], y[:split]), 0)
        x_train, x_eval = x[split:], x[:split]
        y_train, y_eval = y[split:], y[:split]

        model = Regressor(x.shape[1]).cuda()
        opt = torch.optim.Adam(model.parameters())

        best_mae = 1e9
        for e in range(1, 101):
            model.train()
            opt.zero_grad()
            loss = nn.MSELoss()(model(x_train), y_train)
            loss.backward()
            opt.step()

            model.eval()
            y_pred = model(x_eval).detach().cpu()
            mse = mean_squared_error(y_eval.cpu(), y_pred)
            if mse < best_mae:
                best_mae = mse
                best_pred = y_pred
        y_preds.append(best_pred)
        y_trues.append(y_eval.cpu())

    y_preds = torch.cat(y_preds, dim=0)
    y_trues = torch.cat(y_trues, dim=0)
    mae = mean_absolute_error(y_trues, y_preds)
    rmse = mean_squared_error(y_trues, y_preds) ** 0.5
    print(f'MAE: {mae}, RMSE: {rmse}')
