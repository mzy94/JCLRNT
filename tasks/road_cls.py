import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score


class Classifier(nn.Module):
    def __init__(self, input_size, num_classes):
        super(Classifier, self).__init__()
        self.fc = nn.Linear(input_size, num_classes)

    def forward(self, x):
        return self.fc(x)


def evaluation(model, feature_df, fold=100):
    print("\n--- Road Classification ---")
    model.eval()
    x = model.encode_graph()
    if isinstance(x, tuple):
        x = x[0]
    x = x.detach()
    split = x.shape[0] // fold

    valid_labels = ['primary', 'secondary', 'tertiary', 'residential']
    id_dict = {idx: i for i, idx in enumerate(valid_labels)}
    y_df = feature_df.loc[feature_df['highway'].isin(valid_labels)]
    x = x[y_df.index]
    y = torch.tensor(y_df['highway'].map(id_dict).tolist()).cuda()

    y_preds = []
    y_trues = []
    for _ in range(fold):
        x_train, x_eval = x[split:], x[:split]
        y_train, y_eval = y[split:], y[:split]
        x = torch.cat((x[split:], x[:split]), 0)
        y = torch.cat((y[split:], y[:split]), 0)

        model = Classifier(x.shape[1], y.max().item() + 1).cuda()
        opt = torch.optim.Adam(model.parameters())

        best_acc = 0.
        for e in range(1, 101):
            model.train()
            opt.zero_grad()
            ce_loss = nn.CrossEntropyLoss()(model(x_train), y_train)
            ce_loss.backward()
            opt.step()

            model.eval()
            y_pred = torch.argmax(model(x_eval), -1).detach().cpu()
            acc = accuracy_score(y_eval.cpu(), y_pred, normalize=False)
            if acc > best_acc:
                best_acc = acc
                best_pred = y_pred
        y_preds.append(best_pred)
        y_trues.append(y_eval.cpu())

    y_preds = torch.cat(y_preds, dim=0)
    y_trues = torch.cat(y_trues, dim=0)
    macro_f1 = f1_score(y_trues, y_preds, average='macro')
    micro_f1 = f1_score(y_trues, y_preds, average='micro')
    print(f'micro F1: {micro_f1}, macro F1: {macro_f1}')
