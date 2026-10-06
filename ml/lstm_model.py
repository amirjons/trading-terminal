import torch.nn as nn


class LSTMNet(nn.Module):
    def __init__(self, n_feat, hidden=64, layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(n_feat, hidden, layers, batch_first=True, dropout=dropout)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden, 5))

    def forward(self, x):
        out, _ = self.lstm(x)
        o = self.head(out[:, -1])
        return o[:, :4], o[:, 4]
