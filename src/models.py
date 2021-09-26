from typing import List
import torch.nn as nn
import torch.nn.functional as F


class BaselineModel(nn.Module):
    def __init__(self, n_features: int, n_targets: int) -> None:
        super().__init__()
        self.n_features = n_features
        self.n_targets = n_targets
        self.hidden_sizes: List[int] = [2048, 1024, 1024, 512]
        self.dropouts: List[float] = [0.3, 0.3, 0.3, 0.3]

        self.batch_norm_1 = nn.BatchNorm1d(self.n_features)
        self.dense_1 = nn.Linear(self.n_features, self.hidden_sizes[0])

        self.batch_norm_2 = nn.BatchNorm1d(self.hidden_sizes[0])
        self.dropout_2 = nn.Dropout(self.dropouts[0])
        self.dense_2 = nn.Linear(self.hidden_sizes[0], self.hidden_sizes[1])

        self.batch_norm_3 = nn.BatchNorm1d(self.hidden_sizes[1])
        self.dropout_3 = nn.Dropout(self.dropouts[1])
        self.dense_3 = nn.Linear(self.hidden_sizes[1], self.hidden_sizes[2])

        self.batch_norm_4 = nn.BatchNorm1d(self.hidden_sizes[2])
        self.dropout_4 = nn.Dropout(self.dropouts[2])
        self.dense_4 = nn.Linear(self.hidden_sizes[2], self.hidden_sizes[3])

        self.batch_norm_5 = nn.BatchNorm1d(self.hidden_sizes[3])
        self.dropout_5 = nn.Dropout(self.dropouts[3])
        self.dense_5 = nn.Linear(self.hidden_sizes[3], self.n_targets)

    def forward(self, x):
        x = self.batch_norm1(x)
        x = F.leaky_relu(self.dense1(x))

        x = self.batch_norm2(x)
        x = self.dropout2(x)
        x = F.leaky_relu(self.dense2(x))

        x = self.batch_norm3(x)
        x = self.dropout3(x)
        x = F.leaky_relu(self.dense3(x))

        x = self.batch_norm4(x)
        x = self.dropout4(x)
        x = F.leaky_relu(self.dense4(x))

        x = self.batch_norm5(x)
        x = self.dropout5(x)
        x = self.dense5(x)
        return x
