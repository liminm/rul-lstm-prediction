import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence


class RulLstm(nn.Module):
    def __init__(
        self,
        n_features: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, padded: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        lengths_sorted, sort_idx = lengths.sort(descending=True)
        padded_sorted = padded.index_select(0, sort_idx)

        packed = pack_padded_sequence(
            padded_sorted,
            lengths_sorted.cpu(),
            batch_first=True,
            enforce_sorted=True,
        )

        _, (h_n, _) = self.lstm(packed)

        last_layer_h = h_n[-1]
        preds_sorted = self.head(last_layer_h).squeeze(1)

        _, unsort_idx = sort_idx.sort()
        preds = preds_sorted.index_select(0, unsort_idx)

        return preds
