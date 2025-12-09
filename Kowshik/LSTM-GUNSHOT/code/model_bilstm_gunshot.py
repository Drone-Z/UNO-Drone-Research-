import torch
import torch.nn as nn

class BiLSTMGunshot(nn.Module):
    def __init__(self,
                 input_dim,
                 hidden_dim=256,
                 num_layers=2,
                 num_classes=18,
                 dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,  # input [B, T, F]
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=True,
        )
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x, lengths=None, return_features: bool = False):
        """
        x: [B, T, F]
        lengths: [B] (optional, for packing variable length)
        return_features: if True, also return the last hidden features [B, 2*H]
        """
        if lengths is not None:
            # pack for efficiency
            packed = nn.utils.rnn.pack_padded_sequence(
                x, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            out_packed, (h_n, c_n) = self.lstm(packed)
            out, _ = nn.utils.rnn.pad_packed_sequence(out_packed, batch_first=True)
        else:
            out, (h_n, c_n) = self.lstm(x)

        # Use last time-step
        if lengths is not None:
            # gather last valid output for each sequence
            last_outputs = []
            for i, L in enumerate(lengths):
                last_outputs.append(out[i, L - 1, :])
            last_outputs = torch.stack(last_outputs, dim=0)
        else:
            last_outputs = out[:, -1, :]  # [B, 2*H]

        logits = self.fc(last_outputs)
        if return_features:
                return last_outputs, logits
        else:
                return logits