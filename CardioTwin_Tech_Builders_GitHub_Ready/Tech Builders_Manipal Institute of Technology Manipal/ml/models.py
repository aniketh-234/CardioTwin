"""Small multimodal PyTorch models: static EHR + seven daily wearable vectors."""
import torch
from torch import nn


class CardioModel(nn.Module):
    def __init__(self, kind, static_dim, dynamic_dim, days=7, hidden=16):
        super().__init__()
        self.kind = kind
        self.static = nn.Sequential(nn.Linear(static_dim, hidden), nn.ReLU())
        if kind == 'mlp':
            self.temporal = nn.Sequential(nn.Flatten(), nn.Linear(days * dynamic_dim, hidden), nn.ReLU())
        elif kind == 'cnn':
            self.temporal = nn.Sequential(nn.Conv1d(dynamic_dim, hidden, 3, padding=1), nn.ReLU(),
                                          nn.Conv1d(hidden, hidden, 3, padding=1), nn.ReLU(),
                                          nn.AdaptiveAvgPool1d(1), nn.Flatten())
        elif kind in ('lstm', 'gru'):
            cls = nn.LSTM if kind == 'lstm' else nn.GRU
            self.temporal = cls(dynamic_dim, hidden, batch_first=True)
        else:
            raise ValueError(f'Unknown model: {kind}')
        self.head = nn.Sequential(nn.Linear(2 * hidden, hidden), nn.ReLU(),
                                  nn.Dropout(0.2), nn.Linear(hidden, 1))

    def forward(self, static, sequence):
        if self.kind == 'cnn':
            dynamic = self.temporal(sequence.transpose(1, 2))
        elif self.kind in ('lstm', 'gru'):
            output, _ = self.temporal(sequence)
            dynamic = output[:, -1, :]
        else:
            dynamic = self.temporal(sequence)
        return self.head(torch.cat([self.static(static), dynamic], dim=1)).squeeze(1)
